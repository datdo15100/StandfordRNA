"""Local TM-score evaluation via US-align, mirroring the official scorer.

The official Kaggle scorer (``ribonaza2_tm_score.py``) writes each structure as a
C1'-only PDB, runs ``USalign pred native -atom " C1'"``, and reads the TM-score
*normalised by the reference (second) structure*. The final score is, per target,
the best TM over (5 predictions x all reference conformations), averaged over
targets. We reproduce that exactly here for offline validation.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..paths import usalign_bin

_TM_RE = re.compile(r"TM-score=\s+([\d.]+)")


def write_c1_pdb(coords: np.ndarray, resnames: list[str], path: str | Path) -> int:
    """Write a C1'-only PDB. Rows with NaN coordinates are skipped (unresolved).

    Returns the number of resolved residues actually written.
    """
    coords = np.asarray(coords, dtype=float)
    atom_name = "C1'"
    written = 0
    with open(path, "w") as fh:
        for i, (xyz, rn) in enumerate(zip(coords, resnames)):
            if not np.all(np.isfinite(xyz)):
                continue
            x, y, z = xyz
            resid = i + 1
            fh.write(
                f"ATOM  {written + 1:>5d}  {atom_name:<5s} "
                f"{rn:<3s} {resid:>3d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}           C\n"
            )
            written += 1
    return written


def tm_score(pred_pdb: str | Path, ref_pdb: str | Path) -> float:
    """Run US-align and return the TM-score normalised by the reference structure."""
    cmd = [str(usalign_bin()), str(pred_pdb), str(ref_pdb), "-atom", " C1'"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    matches = _TM_RE.findall(out)
    if len(matches) < 2:
        raise ValueError(f"could not parse two TM-scores from US-align output:\n{out}")
    return float(matches[1])  # normalised by the second (reference) structure


def score_target(
    pred_structs: list[np.ndarray],
    ref_structs: list[np.ndarray],
    resnames: list[str],
) -> float:
    """Best TM over (predictions x references) for one target."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref_pdbs = []
        for j, ref in enumerate(ref_structs):
            p = td / f"ref_{j}.pdb"
            if write_c1_pdb(ref, resnames, p) > 0:
                ref_pdbs.append(p)
        if not ref_pdbs:
            raise ValueError("no resolved reference residues for target")

        best = 0.0
        for i, pred in enumerate(pred_structs):
            pp = td / f"pred_{i}.pdb"
            write_c1_pdb(pred, resnames, pp)
            for rp in ref_pdbs:
                best = max(best, tm_score(pp, rp))
    return best
