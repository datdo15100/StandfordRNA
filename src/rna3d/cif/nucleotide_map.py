"""Map (possibly modified) nucleotide residue names to a canonical RNA base.

Strategy:
  1. Trust gemmi's tabulated one-letter code where it yields A/C/G/U (gemmi already
     covers most modified bases, e.g. PSU->u, 1MA->a, OMG->g, H2U->u).
  2. Fall back to an explicit table for common modifications gemmi leaves blank.
  3. Otherwise return 'N' (unknown) and let the caller record it.

T is folded to U because we only ever call this on residues belonging to a chain
already typed as RNA by gemmi's polymer-type check.
"""
from __future__ import annotations

import gemmi

_CANON = set("ACGU")

# Explicit fallbacks for modified residues gemmi does not resolve to A/C/G/U.
MODIFIED_FALLBACK = {
    # cytosine analogs
    "5MC": "C", "OMC": "C", "4OC": "C", "5IC": "C", "CBV": "C", "CCC": "C",
    "A5M": "C", "M4C": "C", "1SC": "C", "C2L": "C", "5HC": "C", "NMT": "C",
    # uracil analogs
    "5MU": "U", "OMU": "U", "4SU": "U", "5BU": "U", "70U": "U", "UR3": "U",
    "U2L": "U", "UD5": "U", "SUR": "U", "DHU": "U", "2MU": "U", "3MU": "U",
    "5IU": "U", "FHU": "U", "RUS": "U", "PYO": "U", "T6A": "U",
    # adenine analogs
    "2MA": "A", "6MA": "A", "MA6": "A", "A2M": "A", "MIA": "A", "T6A_": "A",
    "1MA": "A", "6IA": "A", "RIA": "A", "A23": "A", "12A": "A", "AET": "A",
    # guanine analogs
    "2MG": "G", "7MG": "G", "1MG": "G", "M2G": "G", "OMG": "G", "YG": "G",
    "YYG": "G", "G7M": "G", "GDP": "G", "GTP": "G", "QUO": "G", "1MeG": "G",
    "G46": "G", "G48": "G", "GOM": "G", "2EG": "G", "G2L": "G",
    # inosine / iso-bases (inosine reads as G)
    "I": "G", "IG": "G", "IC": "C", "IU": "U",
    # locked nucleic acids (LNA)
    "LCG": "G", "LCA": "A", "LCC": "C", "LCU": "U", "LKC": "C",
    # 6-position modified series
    "6HG": "G", "6HC": "C", "6HA": "A", "6HT": "U",
    # misc resolved-by-eye
    "SRA": "A", "U34": "U", "U8U": "U", "PPU": "A", "PU": "A",
    # generic
    "N": "N", "UNK": "N",
}


def canonical_base(resname: str) -> str:
    name = resname.strip().upper()
    info = gemmi.find_tabulated_residue(name)
    if info is not None:
        olc = info.one_letter_code.upper().strip()
        if olc in _CANON:
            return olc
        if olc == "T":  # DNA T appearing in an RNA chain -> U
            return "U"
    if name in MODIFIED_FALLBACK:
        return MODIFIED_FALLBACK[name]
    return "N"
