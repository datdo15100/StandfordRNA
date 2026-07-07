"""Parse a PDB_RNA CIF file into per-RNA-chain C1' coordinate records.

For each chain typed as RNA (or RNA/DNA hybrid) by gemmi's polymer-type check we
extract, in residue order, the canonical base and the C1' atom position. Residues
without a modeled C1' (disordered/missing) keep a NaN row so downstream alignment
sees the true gaps. Altlocs are resolved by highest occupancy.
"""
from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from .nucleotide_map import canonical_base

_RNA_TYPES = {gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}


@dataclass
class ChainRecord:
    pdb_id: str
    chain_id: str
    seq: str               # canonical bases, 'N' for unmapped, one char per residue
    resids: np.ndarray     # int32 (L,) author residue numbers
    coords: np.ndarray     # float32 (L, 3), NaN where C1' missing
    n_resolved: int
    n_unknown: int         # residues mapped to 'N'
    polymer_type: str

    @property
    def key(self) -> str:
        return f"{self.pdb_id}_{self.chain_id}"


def _best_c1(residue: gemmi.Residue):
    """Return the C1' atom with the highest occupancy, or None."""
    best = None
    for atom in residue:
        if atom.name == "C1'":
            if best is None or atom.occ > best.occ:
                best = atom
    return best


def parse_cif(path: str, pdb_id: str | None = None,
              unmapped_out: dict | None = None) -> list[ChainRecord]:
    if pdb_id is None:
        import os

        pdb_id = os.path.basename(path).split(".")[0].upper()

    st = gemmi.read_structure(path)
    if len(st) == 0:
        return []
    model = st[0]  # first model only

    records: list[ChainRecord] = []
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        ptype = polymer.check_polymer_type()
        if ptype not in _RNA_TYPES:
            continue

        seq_chars, resids, coords = [], [], []
        n_unknown = 0
        for res in polymer:
            base = canonical_base(res.name)
            if base == "N":
                n_unknown += 1
                if unmapped_out is not None and res.name.strip().upper() not in ("N", "UNK"):
                    nm = res.name.strip().upper()
                    unmapped_out[nm] = unmapped_out.get(nm, 0) + 1
            seq_chars.append(base)
            resids.append(res.seqid.num)
            atom = _best_c1(res)
            if atom is None:
                coords.append((np.nan, np.nan, np.nan))
            else:
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))

        coords = np.asarray(coords, dtype=np.float32)
        n_resolved = int(np.isfinite(coords).all(axis=1).sum())
        if n_resolved == 0:
            continue  # nothing usable as a template
        records.append(
            ChainRecord(
                pdb_id=pdb_id,
                chain_id=chain.name,
                seq="".join(seq_chars),
                resids=np.asarray(resids, dtype=np.int32),
                coords=coords,
                n_resolved=n_resolved,
                n_unknown=n_unknown,
                polymer_type=str(ptype).split(".")[-1],
            )
        )
    return records
