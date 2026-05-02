"""
asre/compliance/hash_ledger.py — ASRE Tamper-Proof Hash Chain Ledger (v1.0)

Purpose
-------
Provides an append-only SHA-256 hash chain that links every ASRE run to its
PDF outputs and score hashes. Each entry cryptographically references the
previous entry, making post-hoc tampering detectable.

Satisfies
---------
- SEBI Circular Dec 2024: AI-assisted research output requires an auditable trail.
- SEBI IA Regulations 2013 Reg 19: 5-year record retention for investment advice.
- SEBI RA Regulations 2014 Reg 24: 5-year retention for research reports.

Chain structure
---------------
GENESIS
  └─ entry_1  { ..., prev_hash: "GENESIS",        entry_hash: SHA256(entry_1_json) }
       └─ entry_2  { ..., prev_hash: entry_1.hash, entry_hash: SHA256(entry_2_json) }
            └─ entry_3  { ..., prev_hash: entry_2.hash, entry_hash: SHA256(entry_3_json) }

Tamper detection
----------------
HashLedger.verify() re-computes every entry_hash from its canonical JSON and
checks that prev_hash chains are unbroken. Any modification to any field of
any past entry will be detected.

Files written
-------------
~/.asre/ledger.jsonl           Append-only chain (one JSON object per line)
~/.asre/ledger.jsonl.lock      File lock (held only during append, auto-released)

Usage (from cli.py finally block)
----------------------------------
from asre.compliance.hash_ledger import HashLedger

entry_hash = HashLedger.append(
    run_id     = decision_log.run_id,
    pdf_paths  = decision_log.pdf_exports,
    score_hash = decision_log.score_hash,
    mode       = decision_log.mode,
    tickers    = decision_log.tickers,
    version    = ASRE_VERSION,
)
logger.info("Ledger entry: %s", entry_hash[:16])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_VERSION  = "1.0"
GENESIS_HASH    = "GENESIS"
MAX_LOCK_WAIT   = 5.0   # seconds to wait for file lock before giving up
LOCK_RETRY_MS   = 50    # milliseconds between lock retries


# ---------------------------------------------------------------------------
# Platform-safe file locking
# ---------------------------------------------------------------------------

class _FileLock:
    """
    Cross-platform advisory file lock.
    Uses fcntl on POSIX, msvcrt on Windows.
    Falls back to no-op if neither is available (e.g. some CI environments).
    """

    def __init__(self, path: Path) -> None:
        self._path   = path
        self._handle = None
        self._locked = False

    def acquire(self, timeout: float = MAX_LOCK_WAIT) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a")
        deadline = time.monotonic() + timeout

        if platform.system() == "Windows":
            import msvcrt
            while time.monotonic() < deadline:
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                    self._locked = True
                    return True
                except OSError:
                    time.sleep(LOCK_RETRY_MS / 1000)
        else:
            import fcntl                              # ← deferred POSIX-only import
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._locked = True
                    return True
                except OSError:
                    time.sleep(LOCK_RETRY_MS / 1000)

        return False

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if self._locked:
                if platform.system() == "Windows":
                    import msvcrt
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl                      # ← deferred POSIX-only import
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None
            self._locked = False


    def __enter__(self) -> "_FileLock":
        acquired = self.acquire()
        if not acquired:
            logger.warning(
                "hash_ledger: could not acquire file lock within %.1fs — "
                "proceeding without lock (risk of concurrent write collision).",
                MAX_LOCK_WAIT,
            )
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Ledger entry schema
# ---------------------------------------------------------------------------

def _build_entry(
    run_id:     str,
    pdf_paths:  List[str],
    score_hash: str,
    prev_hash:  str,
    mode:       str        = "",
    tickers:    List[str]  = None,
    version:    str        = LEDGER_VERSION,
    extra:      Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build a canonical ledger entry dict.

    The entry is serialised with sort_keys=True, separators=(',', ':')
    for deterministic JSON — the hash depends on this exact serialisation.
    Do NOT change the serialisation parameters without a version bump.
    """
    entry: Dict[str, Any] = {
        "ledger_version" : version,
        "ts"             : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id"         : run_id,
        "mode"           : mode.upper() if mode else "",
        "tickers"        : sorted(tickers) if tickers else [],
        "pdfs"           : [str(p) for p in (pdf_paths or [])],
        "score_hash"     : score_hash or "",
        "prev_hash"      : prev_hash,
    }
    if extra:
        entry["extra"] = extra

    return entry


def _hash_entry(entry: Dict[str, Any]) -> str:
    """
    Compute SHA-256 of the entry's canonical JSON.
    entry_hash field must NOT be present when hashing.
    """
    payload = {k: v for k, v in entry.items() if k != "entry_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HashLedger
# ---------------------------------------------------------------------------

class HashLedger:
    """
    Append-only SHA-256 hash chain for ASRE run records.

    All methods are class methods — no instance state.
    The ledger file is never read in full except during verify() and export.

    Thread / process safety
    -----------------------
    append() acquires a file lock before writing.
    Concurrent reads (verify, export) do not lock.

    Retention
    ---------
    The JSONL file must be retained for 5 years per SEBI regulations.
    Use export_verification_report() to produce auditor-ready summaries.
    """

    LEDGER_PATH: Path = Path.home() / ".asre" / "ledger.jsonl"
    LOCK_PATH:   Path = Path.home() / ".asre" / "ledger.jsonl.lock"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @classmethod
    def append(
        cls,
        run_id:     str,
        pdf_paths:  Optional[List[str]] = None,
        score_hash: str = "",
        mode:       str = "",
        tickers:    Optional[List[str]] = None,
        version:    str = LEDGER_VERSION,
        extra:      Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Append a new entry to the ledger and return its entry_hash.

        Parameters
        ----------
        run_id      : UUID string from DecisionLog.run_id
        pdf_paths   : list of PDF paths generated in this run
        score_hash  : hex digest of the score DataFrame (from report_generator)
        mode        : 'ia' or 'ra'
        tickers     : list of tickers analysed
        version     : ledger schema version (default: current)
        extra       : optional dict of additional metadata

        Returns
        -------
        str   entry_hash (SHA-256 hex digest of this entry)

        Never raises — ledger failure must not block the primary run.
        """
        try:
            return cls._append_locked(
                run_id=run_id,
                pdf_paths=pdf_paths or [],
                score_hash=score_hash,
                mode=mode,
                tickers=tickers or [],
                version=version,
                extra=extra or {},
            )
        except Exception as exc:
            logger.error(
                "hash_ledger: append failed for run_id=%s — %s. "
                "Audit record NOT written.",
                run_id[:8] if run_id else "?", exc,
            )
            return ""

    @classmethod
    def _append_locked(cls, **kwargs) -> str:
        cls.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        prev_hash = cls._last_hash()

        entry = _build_entry(prev_hash=prev_hash, **kwargs)
        entry_hash = _hash_entry(entry)
        entry["entry_hash"] = entry_hash

        line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"

        with _FileLock(cls.LOCK_PATH):
            # Re-read last hash inside lock to handle concurrent writers
            prev_hash_locked = cls._last_hash()
            if prev_hash_locked != prev_hash:
                # Another process wrote between our _last_hash() and lock acquire
                entry["prev_hash"] = prev_hash_locked
                entry_hash = _hash_entry(entry)
                entry["entry_hash"] = entry_hash
                line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"

            with open(cls.LEDGER_PATH, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

        logger.info(
            "hash_ledger: entry written — run=%s hash=%s prev=%s pdfs=%d",
            kwargs.get("run_id", "?")[:8],
            entry_hash[:16],
            entry["prev_hash"][:16],
            len(kwargs.get("pdf_paths", [])),
        )
        return entry_hash

    # ------------------------------------------------------------------
    # Chain navigation
    # ------------------------------------------------------------------

    @classmethod
    def _last_hash(cls) -> str:
        """
        Return entry_hash of the last line in the ledger.
        Returns GENESIS_HASH if ledger is empty or absent.
        """
        if not cls.LEDGER_PATH.exists():
            return GENESIS_HASH

        last_line: Optional[str] = None
        try:
            with open(cls.LEDGER_PATH, "rb") as fh:
                # Efficient tail-read for large ledgers
                fh.seek(0, 2)
                size = fh.tell()
                if size == 0:
                    return GENESIS_HASH
                chunk = min(size, 4096)
                fh.seek(-chunk, 2)
                tail = fh.read().decode("utf-8", errors="replace")
                lines = [l for l in tail.splitlines() if l.strip()]
                if lines:
                    last_line = lines[-1]
        except Exception as exc:
            logger.warning("hash_ledger: could not read last entry — %s", exc)
            return GENESIS_HASH

        if not last_line:
            return GENESIS_HASH
        try:
            return json.loads(last_line).get("entry_hash", GENESIS_HASH)
        except (json.JSONDecodeError, KeyError):
            return GENESIS_HASH

    @classmethod
    def last_entry(cls) -> Optional[Dict[str, Any]]:
        """Return the last ledger entry as a dict, or None if ledger is empty."""
        if not cls.LEDGER_PATH.exists():
            return None
        last = None
        try:
            with open(cls.LEDGER_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = line
            return json.loads(last) if last else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Chain integrity
    # ------------------------------------------------------------------

    @classmethod
    def verify(cls) -> bool:
        """
        Verify the full chain integrity.

        Checks (in order for each entry):
        1. prev_hash matches previous entry's entry_hash (or GENESIS for first).
        2. entry_hash matches SHA-256 of the entry's canonical JSON.

        Returns
        -------
        True   if all entries are intact.
        False  if any entry is missing, malformed, or tampered.

        Logs a warning for each violation found.
        """
        if not cls.LEDGER_PATH.exists():
            logger.info("hash_ledger: ledger not found — nothing to verify.")
            return True

        prev   = GENESIS_HASH
        count  = 0
        valid  = True

        try:
            with open(cls.LEDGER_PATH, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "hash_ledger: line %d — JSON parse error: %s", lineno, exc
                        )
                        valid = False
                        continue

                    # Check 1: chain linkage
                    if entry.get("prev_hash") != prev:
                        logger.warning(
                            "hash_ledger: CHAIN BREAK at line %d — "
                            "expected prev_hash=%s, got %s",
                            lineno, prev[:16], str(entry.get("prev_hash", ""))[:16],
                        )
                        valid = False

                    # Check 2: entry hash integrity
                    stored_hash   = entry.get("entry_hash", "")
                    computed_hash = _hash_entry(entry)
                    if stored_hash != computed_hash:
                        logger.warning(
                            "hash_ledger: HASH MISMATCH at line %d — "
                            "run=%s stored=%s computed=%s",
                            lineno,
                            entry.get("run_id", "?")[:8],
                            stored_hash[:16],
                            computed_hash[:16],
                        )
                        valid = False

                    prev  = entry.get("entry_hash", prev)
                    count += 1

        except Exception as exc:
            logger.error("hash_ledger: verify() failed — %s", exc)
            return False

        if valid:
            logger.info(
                "hash_ledger: chain verified — %d entries, integrity VALID.", count
            )
        else:
            logger.error(
                "hash_ledger: chain verification FAILED — %d entries checked.", count
            )
        return valid

    # ------------------------------------------------------------------
    # Auditor export
    # ------------------------------------------------------------------

    @classmethod
    def export_verification_report(cls, output_path: str) -> None:
        """
        Write a human-readable chain verification report to output_path.

        Format: plain text, auditor-readable, no binary content.
        Suitable for SEBI auditor submission to prove 5-year retention integrity.

        Parameters
        ----------
        output_path : destination file path (e.g. 'asre_audit_chain_2026.txt')

        Raises
        ------
        FileNotFoundError  if ledger does not exist.
        IOError            if output_path cannot be written.
        """
        if not cls.LEDGER_PATH.exists():
            raise FileNotFoundError(
                f"Ledger not found at {cls.LEDGER_PATH}. "
                "No runs have been recorded yet."
            )

        entries: List[Dict[str, Any]] = []
        parse_errors = 0

        with open(cls.LEDGER_PATH, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_errors += 1
                    logger.warning(
                        "hash_ledger: export — skipping malformed line %d", lineno
                    )

        chain_valid = cls.verify()

        # Build report
        sep     = "─" * 80
        lines   = []

        lines.append("ASRE Hash Chain Verification Report")
        lines.append("=" * 80)
        lines.append(f"Generated       : {datetime.now(timezone.utc).isoformat()}Z")
        lines.append(f"Ledger path     : {cls.LEDGER_PATH}")
        lines.append(f"Total entries   : {len(entries)}")
        lines.append(f"Parse errors    : {parse_errors}")
        lines.append(f"Chain integrity : {'VALID' if chain_valid else 'BROKEN — TAMPERING DETECTED'}")
        lines.append(f"Ledger version  : {LEDGER_VERSION}")
        lines.append("")
        lines.append("SEBI Retention Requirement: 5 years from date of generation.")
        lines.append("Algorithm: SHA-256. Encoding: UTF-8. Format: JSONL, sort_keys=True.")
        lines.append("")
        lines.append(sep)
        lines.append(
            f"{'#':<5} {'Timestamp':<22} {'Run ID':<10} {'Mode':<5} "
            f"{'Tickers':<30} {'PDFs':<5} {'Hash (first 20)':<22} {'Prev Hash (first 20)':<22} {'OK'}"
        )
        lines.append(sep)

        prev = GENESIS_HASH
        for idx, entry in enumerate(entries, start=1):
            stored_hash   = entry.get("entry_hash", "")
            computed_hash = _hash_entry(entry)
            prev_ok       = entry.get("prev_hash", "") == prev
            hash_ok       = stored_hash == computed_hash
            status        = "OK" if (prev_ok and hash_ok) else "FAIL"

            tickers_str = ", ".join(entry.get("tickers", []))
            if len(tickers_str) > 28:
                tickers_str = tickers_str[:25] + "..."

            lines.append(
                f"{idx:<5} "
                f"{entry.get('ts', 'N/A'):<22} "
                f"{entry.get('run_id', '')[:8]:<10} "
                f"{entry.get('mode', ''):<5} "
                f"{tickers_str:<30} "
                f"{len(entry.get('pdfs', [])):<5} "
                f"{stored_hash[:20]:<22} "
                f"{entry.get('prev_hash', '')[:20]:<22} "
                f"{status}"
            )

            if not prev_ok:
                lines.append(
                    f"  [!] CHAIN BREAK: expected prev={prev[:20]}, "
                    f"got {entry.get('prev_hash', '')[:20]}"
                )
            if not hash_ok:
                lines.append(
                    f"  [!] HASH MISMATCH: stored={stored_hash[:20]}, "
                    f"computed={computed_hash[:20]}"
                )

            prev = stored_hash

        lines.append(sep)
        lines.append("")

        # PDF inventory
        if any(entry.get("pdfs") for entry in entries):
            lines.append("PDF Inventory")
            lines.append(sep)
            for entry in entries:
                for pdf in entry.get("pdfs", []):
                    lines.append(
                        f"  {entry.get('ts', 'N/A')[:19]}  "
                        f"run={entry.get('run_id', '')[:8]}  {pdf}"
                    )
            lines.append("")

        lines.append(
            "END OF REPORT — "
            f"{'Chain valid. All entries intact.' if chain_valid else 'CHAIN INVALID. Escalate immediately.'}"
        )

        output = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(output)

        logger.info(
            "hash_ledger: verification report written to %s (%d entries, chain=%s)",
            output_path, len(entries), "VALID" if chain_valid else "BROKEN",
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """
        Return basic ledger statistics without full verification.
        Fast — reads only first and last entries.
        """
        if not cls.LEDGER_PATH.exists():
            return {"exists": False, "entries": 0}

        first_entry: Optional[Dict] = None
        last_entry:  Optional[Dict] = None
        count = 0

        with open(cls.LEDGER_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if first_entry is None:
                        first_entry = parsed
                    last_entry = parsed
                    count += 1
                except json.JSONDecodeError:
                    pass

        return {
            "exists"       : True,
            "entries"      : count,
            "ledger_path"  : str(cls.LEDGER_PATH),
            "first_ts"     : (first_entry or {}).get("ts"),
            "last_ts"      : (last_entry  or {}).get("ts"),
            "last_run_id"  : (last_entry  or {}).get("run_id", "")[:8],
            "last_hash"    : (last_entry  or {}).get("entry_hash", "")[:16],
        }


# ---------------------------------------------------------------------------
# CLI entry point — python -m asre.compliance.hash_ledger verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASRE Hash Ledger CLI")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("verify", help="Verify chain integrity")

    exp = sub.add_parser("export", help="Export auditor verification report")
    exp.add_argument("output", help="Output path (e.g. audit_report_2026.txt)")

    sub.add_parser("stats", help="Show ledger statistics")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "verify":
        ok = HashLedger.verify()
        sys.exit(0 if ok else 1)

    elif args.cmd == "export":
        try:
            HashLedger.export_verification_report(args.output)
            print(f"Report written: {args.output}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "stats":
        s = HashLedger.stats()
        for k, v in s.items():
            print(f"{k:<20}: {v}")

    else:
        parser.print_help()
