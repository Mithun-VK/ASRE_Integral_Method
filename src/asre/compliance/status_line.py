"""
asre/compliance/status_line.py — ASRE SEBI-Compliant Status Line Renderer (v1.0)

Purpose
-------
Produces exactly one plain-English sentence per ticker for every ASRE output
surface — terminal, PDF footer, audit log, and scan table.

SEBI Requirement
----------------
SEBI IA Regulations 2013 and SEBI Circular Dec 2024 require that AI-assisted
research output be expressed in non-technical, non-directional language that
a non-specialist can read without implying an investment action.

This module satisfies that requirement by:
  1. Using a rule table (not runtime inference) — output is deterministic.
  2. Producing no emojis, symbols, or action verbs (BUY/SELL/HOLD/APPROVE).
  3. Expressing all scores as analytical descriptors, not recommendations.
  4. Versioning the output format so historical records are reproducible.

Design constraints (MUST NOT be violated)
------------------------------------------
- render() is a pure function: same inputs → same output, always.
- No datetime.now(), no random, no locale-dependent float formatting.
- No imports from cli.py, composite.py, or any other ASRE module.
- No Rich markup, no ANSI escape codes in the returned string.
- The version tag [ASRE StatusLine vX.Y] is mandatory — it links the
  narrative to the rule table version for audit trail purposes.

Versioning policy
-----------------
Increment ASRE_STATUS_LINE_VERSION when:
  - Any label text changes (even punctuation).
  - Any threshold boundary changes.
  - Any new dimension is added.
Do NOT increment for docstring or comment changes.

Usage
-----
    from asre.compliance.status_line import StatusLineRenderer

    line = StatusLineRenderer.render(
        f=74.0, t=13.0, m=18.0,
        r_final=70.4, tier="A",
    )
    # -> "The fundamental profile is strong. The price is technically at the
    #     low end of its recent range. Recent price momentum is weak. The
    #     composite score of 70 out of 100 represents high analytical
    #     conviction. This instrument is classified as a high-quality
    #     instrument. [ASRE StatusLine v1.0]"

    # Scan-table variant (first sentence only)
    short = StatusLineRenderer.render_short(f=74.0, t=13.0, m=18.0,
                                            r_final=70.4, tier="A")
    # -> "The fundamental profile is strong."

    # Audit log variant (all scores embedded)
    audit = StatusLineRenderer.render_audit(f=74.0, t=13.0, m=18.0,
                                            r_final=70.4, r_asre=90.0,
                                            tier="A", ticker="RELIANCE.NS",
                                            run_id="a3213147")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

ASRE_STATUS_LINE_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Rule tables (module-level constants — referenced by version)
# ---------------------------------------------------------------------------
# Each table is a list of (threshold, label) tuples sorted descending.
# _lookup() returns the label for the first threshold the value exceeds.
# The final entry (threshold=0) is the unconditional fallback.
#
# All labels:
#   - Begin with "The" or "Recent" — neutral, third-person.
#   - Contain no emojis, symbols, or punctuation other than full stops.
#   - Contain no verbs implying action (avoid: buy, sell, enter, exit,
#     approve, flag, watch, review, hold, add, reduce, avoid).
#   - Are grammatically complete sentences ending with a full stop.
# ---------------------------------------------------------------------------

_F_LABELS_V1: List[Tuple[float, str]] = [
    (65, "The fundamental profile is strong."),
    (50, "The fundamental profile is moderate."),
    (35, "The fundamental profile is below average."),
    ( 0, "The fundamental profile is weak."),
]

_T_LABELS_V1: List[Tuple[float, str]] = [
    (80, "The price is technically elevated relative to recent history."),
    (60, "The price is technically neutral."),
    (40, "The price is technically below its recent average."),
    (20, "The price is technically near its lower range."),
    ( 0, "The price is technically at the low end of its recent range."),
]

_M_LABELS_V1: List[Tuple[float, str]] = [
    (60, "Recent price momentum is positive."),
    (40, "Recent price momentum is neutral."),
    ( 0, "Recent price momentum is weak."),
]

_SCORE_BANDS_V1: List[Tuple[float, str]] = [
    (70, "high"),
    (50, "moderate"),
    ( 0, "low"),
]

_TIER_LABELS_V1 = {
    "A": "classified as a high-quality instrument",
    "B": "classified as a median-quality instrument",
    "C": "classified as a below-median instrument",
    "D": "classified as a speculative instrument",
}

_DIVERGENCE_LABELS_V1: List[Tuple[float, str]] = [
    # Threshold is abs(r_final - r_asre)
    (30, "A material divergence exists between the composite score and the"
         " timing-adjusted score."),
    (15, "A moderate divergence exists between the composite score and the"
         " timing-adjusted score."),
    ( 0, ""),   # No divergence note below 15 pts gap — render() guards with `if div_note:`
]

# ---------------------------------------------------------------------------
# Versioned rule table registry
# Each version maps to its full set of tables so historical records can be
# re-rendered identically even after table updates.
# ---------------------------------------------------------------------------

_RULE_TABLES = {
    "1.0": {
        "f"          : _F_LABELS_V1,
        "t"          : _T_LABELS_V1,
        "m"          : _M_LABELS_V1,
        "score_bands": _SCORE_BANDS_V1,
        "tier"       : _TIER_LABELS_V1,
        "divergence" : _DIVERGENCE_LABELS_V1,
    }
}

_CURRENT_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# StatusLineRenderer
# ---------------------------------------------------------------------------

class StatusLineRenderer:
    """
    Converts ASRE numeric scores into a single SEBI-compliant plain-English
    sentence.

    All public methods are pure functions — no side effects, no I/O,
    no mutable state. The class has no __init__ and no instance variables.

    Methods
    -------
    render()         Full status line (terminal, PDF, audit log).
    render_short()   First sentence only (scan table column).
    render_audit()   Full line + scores embedded (machine-readable audit log).
    render_pdf()     Formatted for PDF footer — tighter layout.
    validate_tables() Sanity-check all rule tables. Call at startup.
    """

    # ------------------------------------------------------------------
    # Primary render
    # ------------------------------------------------------------------

    @classmethod
    def render(
        cls,
        f:       float,
        t:       float,
        m:       float,
        r_final: float,
        tier:    str,
        r_asre:  Optional[float] = None,
        version: str = _CURRENT_VERSION,
    ) -> str:
        """
        Produce the full SEBI-compliant status line.

        Parameters
        ----------
        f        : F-Score (0–100)
        t        : T-Score (0–100)
        m        : M-Score (0–100)
        r_final  : Composite Kalman-filtered score (0–100)
        tier     : Quality tier ('A', 'B', 'C', or 'D')
        r_asre   : ASRE Medallion score (0–100) — optional, used for
                   divergence note only
        version  : Rule table version to use (default: current)

        Returns
        -------
        str   One plain-English paragraph. No emojis. No action language.
              Always ends with [ASRE StatusLine vX.Y].

        Raises
        ------
        ValueError  if version is not in _RULE_TABLES.
        """
        tables = cls._get_tables(version)

        f_label    = cls._lookup(f,       tables["f"])
        t_label    = cls._lookup(t,       tables["t"])
        m_label    = cls._lookup(m,       tables["m"])
        score_band = cls._lookup(r_final, tables["score_bands"])
        tier_label = tables["tier"].get(
            str(tier).strip().upper(), "of unclassified quality"
        )

        parts = [
            f_label,
            t_label,
            m_label,
            (
                f"The composite score of {r_final:.0f} out of 100 represents "
                f"{score_band} analytical conviction."
            ),
            f"This instrument is {tier_label}.",
        ]

        # Optional divergence note (only if r_asre is provided)
        if r_asre is not None:
            div_note = cls._lookup(
                abs(r_final - r_asre), tables["divergence"]
            )
            if div_note:
                parts.append(div_note)

        parts.append(f"[ASRE StatusLine v{version}]")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Scan-table variant — first sentence only
    # ------------------------------------------------------------------

    @classmethod
    def render_short(
        cls,
        f:       float,
        t:       float,
        m:       float,
        r_final: float,
        tier:    str,
        version: str = _CURRENT_VERSION,
    ) -> str:
        """
        Return the first sentence of the full status line — the F-label.
        Used in morning scan tables where column width is limited.

        Example
        -------
        "The fundamental profile is strong."
        """
        tables = cls._get_tables(version)
        return cls._lookup(f, tables["f"])

    # ------------------------------------------------------------------
    # Audit log variant — scores embedded for machine parsing
    # ------------------------------------------------------------------

    @classmethod
    def render_audit(
        cls,
        f:        float,
        t:        float,
        m:        float,
        r_final:  float,
        tier:     str,
        r_asre:   Optional[float] = None,
        ticker:   str = "",
        run_id:   str = "",
        version:  str = _CURRENT_VERSION,
    ) -> str:
        """
        Full status line with scores embedded as a structured suffix.
        Written to ~/.asre/audit/YYYY-MM-DD.jsonl via DecisionLog.

        Format
        ------
        <narrative> [scores: F=74 T=13 M=18 R=70 ASRE=90 Tier=A
                     ticker=RELIANCE.NS run=a3213147 v=1.0]

        The scores suffix is machine-parseable with a simple regex:
          r'\\[scores: (.+?)\\]'
        """
        narrative = cls.render(
            f=f, t=t, m=m, r_final=r_final, tier=tier,
            r_asre=r_asre, version=version,
        )

        asre_str  = f" ASRE={r_asre:.0f}" if r_asre is not None else ""
        tick_str  = f" ticker={ticker}"    if ticker              else ""
        run_str   = f" run={run_id[:8]}"   if run_id              else ""

        scores_tag = (
            f"[scores: F={f:.0f} T={t:.0f} M={m:.0f} R={r_final:.0f}"
            f"{asre_str} Tier={str(tier).upper()}{tick_str}{run_str} v={version}]"
        )

        # Replace the existing version tag with scores tag
        # (avoids duplicate tagging)
        return narrative.replace(f"[ASRE StatusLine v{version}]", scores_tag)

    # ------------------------------------------------------------------
    # PDF footer variant — compact single line
    # ------------------------------------------------------------------

    @classmethod
    def render_pdf(
        cls,
        f:       float,
        t:       float,
        m:       float,
        r_final: float,
        tier:    str,
        version: str = _CURRENT_VERSION,
    ) -> str:
        """
        Compact form for PDF page footer. Max ~120 characters.

        Format
        ------
        "Fundamentals: strong | Price: low-range | Momentum: weak |
         Score: 70/100 (high conviction) | Tier A | ASRE StatusLine v1.0"
        """
        tables = cls._get_tables(version)

        # Extract key adjectives only (strip leading "The ... is/are")
        def _adjective(label: str) -> str:
            for strip in [
                "The fundamental profile is ",
                "The price is technically ",
                "Recent price momentum is ",
            ]:
                if label.startswith(strip):
                    return label[len(strip):].rstrip(".")
            return label.rstrip(".")

        f_adj    = _adjective(cls._lookup(f,       tables["f"]))
        t_adj    = _adjective(cls._lookup(t,       tables["t"]))
        m_adj    = _adjective(cls._lookup(m,       tables["m"]))
        band     = cls._lookup(r_final, tables["score_bands"])
        tier_key = str(tier).strip().upper()

        return (
            f"Fundamentals: {f_adj} | "
            f"Price: {t_adj} | "
            f"Momentum: {m_adj} | "
            f"Score: {r_final:.0f}/100 ({band} conviction) | "
            f"Tier {tier_key} | "
            f"ASRE StatusLine v{version}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup(value: float, table: List[Tuple[float, str]]) -> str:
        """
        Return label for the first threshold the value strictly exceeds.
        Falls through to the last entry (threshold=0) as the unconditional
        fallback.

        NaN / None safety: treats non-numeric values as 0.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0

        for threshold, label in table:
            if v > threshold:
                return label
        return table[-1][1]

    @classmethod
    def _get_tables(cls, version: str) -> dict:
        if version not in _RULE_TABLES:
            raise ValueError(
                f"StatusLineRenderer: unknown version '{version}'. "
                f"Available: {sorted(_RULE_TABLES.keys())}"
            )
        return _RULE_TABLES[version]

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_tables(cls, version: str = _CURRENT_VERSION) -> None:
        """
        Sanity-check rule tables for the given version.
        Call once at ASRE startup (in cli.py or __init__.py).
        Raises AssertionError with a clear message if any table is malformed.
        """
        tables = cls._get_tables(version)

        # All list tables must end with threshold=0 (unconditional fallback)
        for name in ("f", "t", "m", "score_bands", "divergence"):
            tbl = tables[name]
            assert isinstance(tbl, list) and len(tbl) >= 1, \
                f"Table '{name}' must be a non-empty list."
            assert tbl[-1][0] == 0, \
                f"Table '{name}' final threshold must be 0 (unconditional fallback). Got {tbl[-1][0]}."
            # Must be sorted descending by threshold
            thresholds = [row[0] for row in tbl]
            assert thresholds == sorted(thresholds, reverse=True), \
                f"Table '{name}' thresholds must be in descending order. Got {thresholds}."
            # All labels must be strings.
            # Exception: the 'divergence' table's fallback row (threshold=0) is
            # intentionally "" — render() guards with `if div_note:` before
            # appending, so the empty string is a valid sentinel meaning
            # "no note required at this gap level".  All other rows in all
            # tables must be non-empty.
            for i, (thresh, label) in enumerate(tbl):
                allow_empty = (name == "divergence" and thresh == 0)
                assert isinstance(label, str), \
                    f"Table '{name}' row {i}: label must be a str, got {type(label).__name__}."
                if not allow_empty:
                    assert label.strip(), \
                        f"Table '{name}' row {i}: label must be a non-empty string."

        # Tier table must cover A, B, C, D
        tier_tbl = tables["tier"]
        for key in ("A", "B", "C", "D"):
            assert key in tier_tbl, \
                f"Tier table missing key '{key}'."

        # Smoke-test render with known values
        line = cls.render(f=74.0, t=13.0, m=18.0, r_final=70.4,
                          tier="A", version=version)
        assert f"[ASRE StatusLine v{version}]" in line, \
            "render() must embed the version tag."
        assert "\n" not in line, \
            "render() must not contain newlines."

        # Purity check — render twice, must be identical
        line2 = cls.render(f=74.0, t=13.0, m=18.0, r_final=70.4,
                           tier="A", version=version)
        assert line == line2, \
            "render() is not pure — identical inputs produced different outputs."


# ---------------------------------------------------------------------------
# Module-level startup validation
# ---------------------------------------------------------------------------
# Runs once on import. If tables are malformed, fail loudly at startup
# rather than silently producing wrong output at runtime.

StatusLineRenderer.validate_tables(_CURRENT_VERSION)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "StatusLineRenderer",
    "ASRE_STATUS_LINE_VERSION",
]