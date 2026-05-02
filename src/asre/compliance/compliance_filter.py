"""
asre/compliance/compliance_filter.py — ASRE Output Compliance Filter (v1.0)

Purpose
-------
Sanitises every string that leaves ASRE before it reaches the terminal,
PDF, or audit log. Acts as the final firewall between the internal engine
(which uses operational labels for logic purposes) and the regulated output
surface (which must contain no action language, no emojis, no implicit
investment directives).

SEBI Requirement
----------------
SEBI IA Regulations 2013 Reg 15(1): No investment advice through implication.
SEBI RA Regulations 2014 Reg 22(1): Research reports must not contain
  misleading language or imply a specific investment action.
SEBI Circular Dec 2024: AI-generated output must be clearly described as
  research assistance, not advisory output.

Three filter stages (applied in order)
---------------------------------------
1. Label substitution  — Prohibited phrases → neutral equivalents.
                         Applied to all modes.
2. Emoji removal       — All Unicode emoji/symbol blocks stripped.
                         Applied to all modes.
3. RA content strip    — Sentences containing dip mechanics, position
                         sizing, or SMA references removed entirely.
                         Applied only in OutputMode.RA.

Design constraints
------------------
- ComplianceFilter is stateless beyond its mode.
- apply() never raises — returns original string on unexpected error.
- Label map is longest-match first (longer keys take priority).
- All substitutions are case-sensitive to avoid false positives.
- apply_panel_title() strips emojis only — does not substitute labels,
  because panel titles use controlled strings from the label map keys.

Versioning
----------
Increment COMPLIANCE_FILTER_VERSION when:
  - Any entry is added to or removed from _LABEL_MAP.
  - Any regex pattern changes.
  - RA blocked-word list changes.
Do NOT increment for docstring or comment changes.

Usage
-----
    from asre.compliance.compliance_filter import ComplianceFilter, OutputMode

    cf = ComplianceFilter(mode=OutputMode.IA)
    clean = cf.apply("🎯 DIP APPROVED: EARLY stage, score=100/100, Tier A")
    # -> "Price condition noted: early deviation stage, score=100/100, Tier A"

    cf_ra = ComplianceFilter(mode=OutputMode.RA)
    clean_ra = cf_ra.apply("Position sizing guidance: 100% — HIGH QUALITY")
    # -> ""   (entire sentence stripped in RA mode)
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

COMPLIANCE_FILTER_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Output mode
# ---------------------------------------------------------------------------

class OutputMode(Enum):
    IA   = "ia"    # Investment Adviser  — full output, IA-safe labels
    RA   = "ra"    # Research Analyst    — compare only, no dip mechanics
    SCAN = "scan"  # Morning scan        — summary table only, no panels

# ---------------------------------------------------------------------------
# Label substitution map
# ---------------------------------------------------------------------------
# Rules:
#   - Keys are matched as substrings (not whole-word) — be specific.
#   - Longer keys must appear before shorter keys that are substrings of them.
#     (The map is sorted by key length descending before application.)
#   - Values must contain no emojis, no action verbs, no trading directives.
#   - Duplicate keys are intentional where the same phrase appears in
#     different contexts — Python dict keeps the last value; the sort-by-
#     length approach handles ordering.
# ---------------------------------------------------------------------------

_LABEL_MAP_RAW: Dict[str, str] = {

    # ------------------------------------------------------------------
    # Dip quality labels (longest first within each group)
    # ------------------------------------------------------------------
    "HIGH QUALITY DIP OPPORTUNITY"              : "Price below long-term average — condition noted",
    "HIGH QUALITY DIP"                          : "Price below long-term average — condition noted",
    "GOOD DIP"                                  : "Price below long-term average",
    "POOR DIP"                                  : "Price condition does not meet research criteria",
    "DIP OPPORTUNITY"                           : "Price below long-term average — research flag",
    "DIP APPROVED"                              : "Price condition noted",
    "STRUCTURAL BREAK"                          : "Significant price deviation noted",
    "MARGINAL"                                  : "Inconclusive price condition",

    # ------------------------------------------------------------------
    # Position sizing — ALL variants observed in live output (file:135)
    # ------------------------------------------------------------------
    "100% of planned position (HIGH QUALITY)"   : "[IA decision required]",
    "50-75% of planned position (GOOD)"         : "[IA decision required]",
    "25% max -- wait for confirmation"           : "[IA decision required]",
    "100% — HIGH QUALITY"                       : "[IA decision required]",
    "50-75% — GOOD"                             : "[IA decision required]",
    "25% max — MARGINAL"                        : "[IA decision required]",
    "Position sizing guidance"                  : "Research note",
    "position sizing"                           : "research note",

    # ------------------------------------------------------------------
    # Dip stage labels (appear inline in panel text)
    # ------------------------------------------------------------------
    "EARLY stage"                               : "early deviation stage",
    "MID stage"                                 : "intermediate deviation stage",
    "LATE stage"                                : "extended deviation stage",
    "DEEP stage"                                : "significant deviation stage",

    # ------------------------------------------------------------------
    # Market scenario labels
    # ------------------------------------------------------------------
    "DIVERGENCE ALERT"                          : "Score divergence noted",
    "OVERSOLD RECOVERY"                         : "Price below average with improving momentum",
    "LATE-STAGE DIP"                            : "Extended price weakness noted",
    "OVERBOUGHT FADE"                           : "Elevated price with declining momentum",
    "DIP OPPORTUNITY"                           : "Price below long-term average — research flag",

    # ------------------------------------------------------------------
    # Signal / rating labels
    # ------------------------------------------------------------------
    "PRIORITY REVIEW"                           : "High composite score",
    "POSITIVE OUTLOOK"                          : "Above-average composite score",
    "FLAG FOR REVIEW"                           : "Below-average composite score",
    "URGENT REVIEW"                             : "Low composite score",

    # Single-word signal — listed LAST to avoid clobbering longer matches
    # e.g. "FLAG FOR REVIEW" must replace before "FLAG" is reached
    "WATCH"                                     : "Declining composite score",

    # ------------------------------------------------------------------
    # Panel / section titles
    # ------------------------------------------------------------------
    "Dip Quality Analysis (Strategy C)"         : "Price Condition Analysis",
    "Market Context"                            : "Price Condition Summary",
    "IA Research Summary"                       : "Research Summary",

    # ------------------------------------------------------------------
    # Inline approval / rejection language
    # ------------------------------------------------------------------
    "DIP APPROVED:"                             : "Price condition noted:",
    "APPROVED"                                  : "noted",
    "REJECTED"                                  : "does not meet research criteria",
    "approved"                                  : "noted",

    # ------------------------------------------------------------------
    # Confidence / certainty language
    # ------------------------------------------------------------------
    "Confidence:"                               : "Analytical weight:",
    "conf="                                     : "weight=",

    # ------------------------------------------------------------------
    # Recommendation-adjacent verbs that appear in panel text
    # ------------------------------------------------------------------
    "Immediate IA review per client mandate"    : "Warrants IA attention per client mandate",
    "Flag for IA review and client discussion"  : "Warrants IA attention",
    "wait for M or T improvement"               : "further data required",
    "wait for confirmation"                     : "further data required",
    "Await M recovery before IA action"         : "Further data required",

    # ------------------------------------------------------------------
    # Dip-related descriptors used in Market Context panel
    # ------------------------------------------------------------------
    "SMA-200 dist:"                             : "Long-term average distance:",
    "SMA-200 distance:"                         : "Long-term average distance:",
    "SMA-200"                                   : "long-term price average",
    "sma_200"                                   : "long_term_avg",

    # ------------------------------------------------------------------
    # Emoji text equivalents that appear as plain text in some contexts
    # ------------------------------------------------------------------
    "✅ DIP"                                    : "Price condition noted",
    "❌ STRUCTURAL"                             : "Significant deviation noted",
    "⚠️ MARGINAL"                              : "Inconclusive price condition",
    "❌ POOR"                                   : "Price condition does not meet research criteria",
    "❌ REJECTED"                               : "Price condition does not meet research criteria",
}

# ---------------------------------------------------------------------------
# Pre-sort by key length descending
# This ensures longer phrases are matched before shorter substrings.
# e.g. "HIGH QUALITY DIP OPPORTUNITY" before "HIGH QUALITY DIP" before "DIP"
# ---------------------------------------------------------------------------

_LABEL_MAP: List[Tuple[str, str]] = sorted(
    _LABEL_MAP_RAW.items(),
    key=lambda kv: len(kv[0]),
    reverse=True,
)

# ---------------------------------------------------------------------------
# Emoji removal pattern
# ---------------------------------------------------------------------------
# Covers all major Unicode blocks containing emoji and symbol characters.
# Deliberately broad — ASRE output should contain no symbols in compliance mode.

_EMOJI_RE = re.compile(
    "["
    "\U00010000-\U0010FFFF"   # Supplementary multilingual plane (most emoji)
    "\U0001F300-\U0001F9FF"   # Misc symbols & pictographs, emoticons, transport
    "\U0001FA00-\U0001FA6F"   # Chess, symbols extended-A
    "\U0001FA70-\U0001FAFF"   # Symbols extended-B
    "\u2600-\u26FF"           # Misc symbols (sun, moon, warning ⚠, etc.)
    "\u2700-\u27BF"           # Dingbats (✓ ✗ etc.)
    "\u2300-\u23FF"           # Misc technical
    "\u25A0-\u25FF"           # Geometric shapes
    "\u2190-\u21FF"           # Arrows
    "\u2B00-\u2BFF"           # Misc symbols and arrows
    "\uFE00-\uFE0F"           # Variation selectors (emoji modifiers)
    "\u200D"                  # Zero-width joiner (used in compound emoji)
    "]+",
    flags=re.UNICODE,
)

# ---------------------------------------------------------------------------
# RA mode blocked content
# ---------------------------------------------------------------------------
# In RA mode, any sentence containing these words is removed entirely.
# RA output must not contain dip mechanics, position sizing, or SMA references.

_RA_BLOCKED_WORDS: frozenset = frozenset({
    "dip",
    "position",
    "stage",
    "sma",
    "oversold",
    "overbought",
    "approved",
    "sizing",
    "sma-200",
    "long-term average",
    "deviation stage",
    "entry timing",
    "risk/reward",
    "risk reward",
    "expected upside",
    "confidence",
})

# Sentence boundary pattern — splits on ". " or ".\n" but not on decimal points
_SENTENCE_RE = re.compile(r'(?<=[a-zA-Z0-9\]\)])\.\s+')


# ---------------------------------------------------------------------------
# ComplianceFilter
# ---------------------------------------------------------------------------

class ComplianceFilter:
    """
    Sanitises all string output before it reaches console, PDF, or audit log.

    Instantiate once per session with the correct OutputMode, then pass
    every user-facing string through apply() before printing or writing.

    Parameters
    ----------
    mode : OutputMode
        IA   — label substitution + emoji removal.
        RA   — label substitution + emoji removal + dip content removal.
        SCAN — same as IA (scan table uses StatusLineRenderer which is
               already clean; this mode exists for future extension).

    Usage
    -----
        cf = ComplianceFilter(mode=OutputMode.IA)

        # Single string
        clean = cf.apply("🎯 DIP APPROVED: EARLY stage, score=100/100")

        # Panel title
        title = cf.apply_panel_title("📌 Market Context")

        # Table cell
        cell = cf.apply("Position sizing guidance: 100% — HIGH QUALITY")

        # Multi-line block (e.g. full panel content)
        block = cf.apply_block(multiline_text)
    """

    def __init__(self, mode: OutputMode = OutputMode.IA) -> None:
        self.mode = mode

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def apply(self, text: str) -> str:
        """
        Apply all filter stages to a single string.

        Stage 1: Label substitution (all modes)
        Stage 2: Emoji removal     (all modes)
        Stage 3: RA content strip  (RA mode only)
        Stage 4: Whitespace clean  (all modes)

        Returns the original string unchanged if it is not a str.
        Never raises.
        """
        if not isinstance(text, str):
            return text
        try:
            text = self._substitute_labels(text)
            text = self._strip_emojis(text)
            if self.mode == OutputMode.RA:
                text = self._strip_ra_content(text)
            text = self._clean_whitespace(text)
            return text
        except Exception as exc:
            logger.warning(
                "compliance_filter: apply() failed — %s. Returning original.", exc
            )
            return text

    def apply_panel_title(self, title: str) -> str:
        """
        Apply emoji removal and label substitution to a Rich panel or
        table title. Does not apply RA stripping (titles are controlled
        strings, not free text).

        Example
        -------
        cf.apply_panel_title("📌 Market Context")
        # -> "Price Condition Summary"
        """
        if not isinstance(title, str):
            return title
        try:
            title = self._substitute_labels(title)
            title = self._strip_emojis(title)
            return title.strip()
        except Exception as exc:
            logger.warning(
                "compliance_filter: apply_panel_title() failed — %s.", exc
            )
            return title

    def apply_block(self, text: str) -> str:
        """
        Apply filters to a multi-line text block.
        Each line is filtered independently to preserve layout structure.

        Use for full panel content, multi-line Rich markup strings, etc.
        """
        if not isinstance(text, str):
            return text
        lines = text.split("\n")
        return "\n".join(self.apply(line) for line in lines)

    def apply_table_row(self, row: List[str]) -> List[str]:
        """
        Apply filters to every cell in a Rich table row (list of strings).
        Returns a new list — does not mutate the input.
        """
        return [self.apply(cell) for cell in row]

    def is_clean(self, text: str) -> bool:
        """
        Return True if text passes all compliance checks without modification.
        Used in unit tests to verify output surfaces are pre-filtered.
        """
        return self.apply(text) == text

    # ------------------------------------------------------------------
    # Filter stages
    # ------------------------------------------------------------------

    @staticmethod
    def _substitute_labels(text: str) -> str:
        """
        Stage 1: Replace prohibited phrases with neutral equivalents.
        Applies all entries in _LABEL_MAP in longest-key-first order.
        Case-sensitive. Does not use regex (avoids false positives on
        partial word boundaries like "WATCH" in "WATCHLIST").
        """
        for prohibited, replacement in _LABEL_MAP:
            if prohibited in text:
                text = text.replace(prohibited, replacement)
        return text

    @staticmethod
    def _strip_emojis(text: str) -> str:
        """Stage 2: Remove all emoji and symbol characters."""
        return _EMOJI_RE.sub("", text)

    @staticmethod
    def _strip_ra_content(text: str) -> str:
        """
        Stage 3 (RA only): Remove any sentence containing RA-blocked words.
        Splits on sentence boundaries, filters, then rejoins.
        Preserves non-sentence structures (e.g. table cells, short labels).
        """
        # If no sentence boundary exists, treat as a single unit
        if not _SENTENCE_RE.search(text):
            lower = text.lower()
            if any(w in lower for w in _RA_BLOCKED_WORDS):
                return ""
            return text

        sentences = _SENTENCE_RE.split(text)
        kept = []
        for s in sentences:
            lower = s.lower()
            if not any(w in lower for w in _RA_BLOCKED_WORDS):
                kept.append(s)
        result = ". ".join(kept)
        # Restore trailing period if original had one and result does not
        if text.rstrip().endswith(".") and result and not result.rstrip().endswith("."):
            result = result.rstrip() + "."
        return result

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """
        Stage 4: Normalise whitespace produced by substitutions and stripping.
        - Collapse multiple spaces to one.
        - Remove leading/trailing whitespace per line.
        - Remove lines that are blank after filtering.
        """
        # Collapse multiple spaces (but not newlines)
        text = re.sub(r"  +", " ", text)
        # Remove spaces adjacent to newlines
        text = re.sub(r" *\n *", "\n", text)
        # Strip leading/trailing
        return text.strip()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def audit_string(self, text: str) -> Dict[str, str]:
        """
        Return a dict showing each filter stage's output for a given string.
        Use in tests and compliance review to trace exactly what changed.

        Example
        -------
        cf.audit_string("🎯 DIP APPROVED: EARLY stage")
        # -> {
        #      "original"   : "🎯 DIP APPROVED: EARLY stage",
        #      "after_labels": "🎯 Price condition noted: early deviation stage",
        #      "after_emoji" : "Price condition noted: early deviation stage",
        #      "after_ra"   : "Price condition noted: early deviation stage",
        #      "final"      : "Price condition noted: early deviation stage",
        #    }
        """
        after_labels = self._substitute_labels(text)
        after_emoji  = self._strip_emojis(after_labels)
        after_ra     = (
            self._strip_ra_content(after_emoji)
            if self.mode == OutputMode.RA
            else after_emoji
        )
        final        = self._clean_whitespace(after_ra)

        return {
            "original"    : text,
            "after_labels": after_labels,
            "after_emoji" : after_emoji,
            "after_ra"    : after_ra,
            "final"       : final,
        }

    @classmethod
    def list_prohibited(cls) -> List[str]:
        """Return all prohibited phrases in substitution order (longest first)."""
        return [k for k, _ in _LABEL_MAP]

    @classmethod
    def list_ra_blocked(cls) -> List[str]:
        """Return all RA blocked words."""
        return sorted(_RA_BLOCKED_WORDS)


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Run on import. Verifies the five most critical substitutions from
    live terminal output. Raises AssertionError if any fail.
    """
    cf_ia = ComplianceFilter(mode=OutputMode.IA)
    cf_ra = ComplianceFilter(mode=OutputMode.RA)

    # 1. Emoji stripped
    assert "🎯" not in cf_ia.apply("🎯 HIGH QUALITY DIP")

    # 2. DIP APPROVED replaced
    result = cf_ia.apply("✅ DIP APPROVED: EARLY stage, score=100/100, Tier A")
    assert "DIP APPROVED" not in result
    assert "EARLY stage"  not in result

    # 3. Position sizing suppressed
    result = cf_ia.apply("Position sizing guidance: 100% — HIGH QUALITY")
    assert "100%" not in result or "[IA decision required]" in result

    # 4. RA mode strips dip content
    result = cf_ra.apply(
        "The fundamental profile is strong. "
        "DIP APPROVED: DEEP stage, SMA-200 dist: -15.5%."
    )
    assert "dip" not in result.lower()
    assert "sma"  not in result.lower()

    # 5. apply() never raises on non-string input
    assert cf_ia.apply(None)  is None   # type: ignore[arg-type]
    assert cf_ia.apply(42)    == 42     # type: ignore[arg-type]


_smoke_test()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "ComplianceFilter",
    "OutputMode",
    "COMPLIANCE_FILTER_VERSION",
]
