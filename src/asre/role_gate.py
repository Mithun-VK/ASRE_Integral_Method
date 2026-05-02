"""
asre/role_gate.py — SEBI Role Gate (v1.0)

Validates that the current environment is authorised to run ASRE in the
requested regulatory mode before any computation begins.

Modes
-----
ia  Investment Adviser    SEBI IA Regulations 2013 (amended 2020)
ra  Research Analyst      SEBI RA Regulations 2014

Validation layers (applied in order, all must pass)
----------------------------------------------------
1. Mode value is a recognised string ('ia' or 'ra').
2. ASRE_MODE env-var, if set, must match the requested mode.
3. Registration credential present:
     ASRE_IA_REG_NO  for mode=ia
     ASRE_RA_REG_NO  for mode=ra
4. Credential format check:
     IA numbers  INA000000000  (INA + 9 digits)
     RA numbers  INH000000000  (INH + 9 digits)
5. Optional: config-file credential cross-check
     ~/.asre/credentials.json  ->  {"ia_reg_no": "...", "ra_reg_no": "..."}
6. Optional: lock-file guard  (~/.asre/.role_lock)
     If the lock file exists and contains a different mode, hard-abort.
     Prevents accidental mode switching mid-session in shared environments.

Environment variables
---------------------
ASRE_MODE          Override/lock the allowed mode  (optional)
ASRE_IA_REG_NO     SEBI IA registration number     (required for mode=ia)
ASRE_RA_REG_NO     SEBI RA registration number     (required for mode=ra)
ASRE_SKIP_REG      Set to '1' to bypass reg check  (DEV / CI only)
ASRE_CREDENTIALS   Path to credentials JSON file   (optional override)

Exit behaviour
--------------
RoleGateError is raised on any failure.
cli.py catches it, logs to DecisionLog, flushes audit, and calls sys.exit(1).
No computation ever starts after a RoleGateError.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RoleGateError(Exception):
    """
    Raised by RoleGate.validate() on any authorisation failure.

    Attributes
    ----------
    reason   : short machine-readable code  (e.g. 'MISSING_REG_NO')
    message  : human-readable explanation
    mode     : the mode that was being validated
    """
    def __init__(self, reason: str, message: str, mode: str = ""):
        super().__init__(message)
        self.reason  = reason
        self.message = message
        self.mode    = mode

    def __str__(self) -> str:
        mode_tag = f"[{self.mode.upper()}] " if self.mode else ""
        return f"{mode_tag}{self.reason}: {self.message}"


# ---------------------------------------------------------------------------
# Registration number validators
# ---------------------------------------------------------------------------

# SEBI IA numbers:  INA followed by exactly 9 digits
_IA_PATTERN = re.compile(r'^INA\d{9}$')

# SEBI RA numbers:  INH followed by exactly 9 digits
_RA_PATTERN = re.compile(r'^INH\d{9}$')

# Dev/CI placeholder values that are explicitly NOT accepted in strict mode
_PLACEHOLDER_VALUES = {
    "XXXXXXXXX", "000000000", "TEST", "DEMO", "PLACEHOLDER",
    "INA000000000", "INH000000000",
}


def _validate_ia_number(reg_no: str) -> bool:
    return bool(_IA_PATTERN.match(reg_no)) and reg_no not in _PLACEHOLDER_VALUES


def _validate_ra_number(reg_no: str) -> bool:
    return bool(_RA_PATTERN.match(reg_no)) and reg_no not in _PLACEHOLDER_VALUES


# ---------------------------------------------------------------------------
# Credentials file loader
# ---------------------------------------------------------------------------

def _load_credentials_file(path: Optional[str] = None) -> dict:
    """
    Load ~/.asre/credentials.json (or ASRE_CREDENTIALS override).
    Returns empty dict if file is absent — absence is not an error.
    Logs a warning if the file exists but cannot be parsed.
    """
    if path is None:
        path = os.environ.get(
            "ASRE_CREDENTIALS",
            str(Path.home() / ".asre" / "credentials.json"),
        )
    cred_path = Path(path)
    if not cred_path.exists():
        return {}
    try:
        with open(cred_path, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("role_gate: credentials file is not a JSON object — ignored.")
            return {}
        return data
    except Exception as exc:
        logger.warning("role_gate: could not parse credentials file %s — %s", cred_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Role lock file
# ---------------------------------------------------------------------------

_LOCK_FILE = Path.home() / ".asre" / ".role_lock"


def _check_role_lock(mode: str) -> None:
    """
    If a role lock file exists and contains a different mode, raise.
    Lock files are written by write_role_lock() and removed on clean exit
    by the caller if desired.
    """
    if not _LOCK_FILE.exists():
        return
    try:
        locked_mode = _LOCK_FILE.read_text().strip().lower()
    except Exception:
        return   # Unreadable lock → ignore (don't block)

    if locked_mode and locked_mode != mode:
        raise RoleGateError(
            reason="ROLE_LOCK_CONFLICT",
            message=(
                f"A role lock file exists for mode '{locked_mode}'. "
                f"You requested mode '{mode}'. "
                f"Remove {_LOCK_FILE} to switch modes."
            ),
            mode=mode,
        )


def write_role_lock(mode: str) -> None:
    """Write a role lock file.  Called by RoleGate after successful validation."""
    try:
        _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOCK_FILE.write_text(mode.lower())
    except Exception as exc:
        logger.warning("role_gate: could not write role lock file — %s", exc)


def clear_role_lock() -> None:
    """Remove the role lock file (call on clean session exit if desired)."""
    try:
        if _LOCK_FILE.exists():
            _LOCK_FILE.unlink()
    except Exception as exc:
        logger.warning("role_gate: could not remove role lock file — %s", exc)


# ---------------------------------------------------------------------------
# RoleGate
# ---------------------------------------------------------------------------

class RoleGate:
    """
    Validates the current environment against the requested SEBI role mode.

    Usage (from cli.py)
    -------------------
    gate   = RoleGate(mode=args.mode)
    detail = gate.validate()          # raises RoleGateError on failure
    decision_log.approve_role(detail)

    Parameters
    ----------
    mode        : 'ia' or 'ra'
    strict      : if True, placeholder reg numbers are rejected even in dev
                  environments.  Default True.
    write_lock  : if True, write a role lock file after successful validation
                  to prevent accidental mode switching.  Default False.
    """

    VALID_MODES = ("ia", "ra")

    def __init__(
        self,
        mode:       str,
        strict:     bool = True,
        write_lock: bool = False,
    ) -> None:
        self.mode       = mode.lower().strip()
        self.strict     = strict
        self.write_lock = write_lock

        # Populated during validate()
        self._reg_no:     str = ""
        self._source:     str = ""  # where the reg number came from

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> str:
        """
        Run all validation layers in order.

        Returns
        -------
        str   Human-readable approval detail string (for DecisionLog).

        Raises
        ------
        RoleGateError on any failure.
        """
        self._check_mode_value()
        self._check_env_mode_override()
        self._check_role_lock_file()

        # Dev/CI bypass — explicit opt-out of reg check
        if os.environ.get("ASRE_SKIP_REG", "").strip() == "1":
            logger.warning(
                "role_gate: ASRE_SKIP_REG=1 — registration check bypassed. "
                "This is NOT permitted in production."
            )
            detail = (
                f"mode={self.mode.upper()} | reg_check=BYPASSED "
                f"(ASRE_SKIP_REG) | ts={_utc_now()}"
            )
            if self.write_lock:
                write_role_lock(self.mode)
            return detail

        self._resolve_reg_no()
        self._check_reg_format()

        if self.write_lock:
            write_role_lock(self.mode)

        detail = (
            f"mode={self.mode.upper()} | "
            f"reg_no={self._redact(self._reg_no)} | "
            f"source={self._source} | "
            f"ts={_utc_now()}"
        )
        logger.info("role_gate: validated — %s", detail)
        return detail

    # ------------------------------------------------------------------
    # Validation layers
    # ------------------------------------------------------------------

    def _check_mode_value(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise RoleGateError(
                reason="INVALID_MODE",
                message=(
                    f"'{self.mode}' is not a recognised mode. "
                    f"Valid modes: {', '.join(self.VALID_MODES)}."
                ),
                mode=self.mode,
            )

    def _check_env_mode_override(self) -> None:
        env_mode = os.environ.get("ASRE_MODE", "").strip().lower()
        if env_mode and env_mode != self.mode:
            raise RoleGateError(
                reason="MODE_ENV_MISMATCH",
                message=(
                    f"ASRE_MODE environment variable is set to '{env_mode}' "
                    f"but '--mode {self.mode}' was requested. "
                    f"Unset ASRE_MODE or align it with the requested mode."
                ),
                mode=self.mode,
            )

    def _check_role_lock_file(self) -> None:
        try:
            _check_role_lock(self.mode)
        except RoleGateError:
            raise
        except Exception as exc:
            logger.debug("role_gate: lock file check skipped — %s", exc)

    def _resolve_reg_no(self) -> None:
        """
        Resolve registration number from (in priority order):
          1. Environment variable  (ASRE_IA_REG_NO / ASRE_RA_REG_NO)
          2. Credentials file      (~/.asre/credentials.json)
        """
        env_key = f"ASRE_{self.mode.upper()}_REG_NO"
        env_val = os.environ.get(env_key, "").strip()

        if env_val:
            self._reg_no = env_val
            self._source = f"env:{env_key}"
            return

        # Fall back to credentials file
        creds    = _load_credentials_file()
        file_key = f"{self.mode}_reg_no"
        file_val = creds.get(file_key, "").strip()

        if file_val:
            self._reg_no = file_val
            self._source = "credentials_file"
            return

        # Nothing found
        raise RoleGateError(
            reason="MISSING_REG_NO",
            message=(
                f"No SEBI {self.mode.upper()} registration number found. "
                f"Set the {env_key} environment variable "
                f"or add '{file_key}' to ~/.asre/credentials.json.\n"
                f"  IA numbers: INA followed by 9 digits  (e.g. INA000012345)\n"
                f"  RA numbers: INH followed by 9 digits  (e.g. INH000012345)"
            ),
            mode=self.mode,
        )

    def _check_reg_format(self) -> None:
        reg = self._reg_no

        if self.mode == "ia":
            valid = _validate_ia_number(reg)
            expected = "INA + 9 digits  (e.g. INA000012345)"
        else:
            valid = _validate_ra_number(reg)
            expected = "INH + 9 digits  (e.g. INH000012345)"

        if not valid:
            # Distinguish placeholder from bad format for clearer errors
            if reg.upper() in _PLACEHOLDER_VALUES or reg in _PLACEHOLDER_VALUES:
                raise RoleGateError(
                    reason="PLACEHOLDER_REG_NO",
                    message=(
                        f"Registration number '{self._redact(reg)}' appears to be a "
                        f"placeholder value and is not accepted in strict mode. "
                        f"Use a real SEBI {self.mode.upper()} registration number."
                    ),
                    mode=self.mode,
                )
            raise RoleGateError(
                reason="INVALID_REG_FORMAT",
                message=(
                    f"Registration number '{self._redact(reg)}' does not match "
                    f"the expected SEBI {self.mode.upper()} format: {expected}."
                ),
                mode=self.mode,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _redact(reg_no: str) -> str:
        """Show first 4 + last 2 chars, mask the rest.  INA000012345 -> INA0*****45"""
        if len(reg_no) <= 6:
            return reg_no
        return reg_no[:4] + "*" * (len(reg_no) - 6) + reg_no[-2:]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def create_credentials_file(
    ia_reg_no: Optional[str] = None,
    ra_reg_no: Optional[str] = None,
    path:      Optional[str] = None,
) -> Path:
    """
    Helper to create / update ~/.asre/credentials.json programmatically.

    Example
    -------
    from asre.role_gate import create_credentials_file
    create_credentials_file(ia_reg_no="INA000012345")
    """
    if path is None:
        path = str(Path.home() / ".asre" / "credentials.json")

    cred_path = Path(path)
    cred_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing so we don't clobber the other key
    existing: dict = {}
    if cred_path.exists():
        try:
            with open(cred_path) as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}

    if ia_reg_no is not None:
        existing["ia_reg_no"] = ia_reg_no
    if ra_reg_no is not None:
        existing["ra_reg_no"] = ra_reg_no

    with open(cred_path, "w") as fh:
        json.dump(existing, fh, indent=2)

    logger.info("role_gate: credentials written to %s", cred_path)
    return cred_path


__all__ = [
    "RoleGate",
    "RoleGateError",
    "write_role_lock",
    "clear_role_lock",
    "create_credentials_file",
]