"""Model-agnostic candidate contract and disk cache for GeoFuse-RNA.

The cache deliberately uses ``npz`` with ``allow_pickle=False``.  Raw model
outputs can be large and model-specific; the normalized cache contains only the
arrays and provenance needed by downstream clustering, fusion, and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_KINDS = {"template", "pretrained", "denovo"}


def sequence_digest(sequence: str) -> str:
    """Stable digest used to reject a cache created for a different sequence."""
    return hashlib.sha256(sequence.upper().encode("ascii")).hexdigest()


def safe_name(value: str) -> str:
    """Convert an identifier to a portable filename component."""
    cleaned = _SAFE_NAME.sub("_", value).strip("._")
    if not cleaned:
        raise ValueError(f"identifier has no filename-safe characters: {value!r}")
    return cleaned


@dataclass
class StructureCandidate:
    """One RNA structure hypothesis in the common GeoFuse representation.

    ``support_mask`` is distinct from coordinate validity.  A gap-filled TBM
    candidate has finite coordinates everywhere, but only transferred residues
    have direct source support.  Pretrained candidates normally support every
    residue they emit.
    """

    target_id: str
    sequence: str
    candidate_id: str
    kind: str
    source: str
    model: str
    coords: np.ndarray
    confidence: np.ndarray
    support_mask: np.ndarray
    global_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    priors: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper().replace("T", "U")
        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.support_mask = np.asarray(self.support_mask, dtype=bool)
        self.global_confidence = float(self.global_confidence)

        length = len(self.sequence)
        if self.kind not in _KINDS:
            raise ValueError(f"unknown candidate kind {self.kind!r}; expected one of {sorted(_KINDS)}")
        if self.coords.shape != (length, 3):
            raise ValueError(
                f"{self.candidate_id}: expected coords {(length, 3)}, got {self.coords.shape}"
            )
        if self.confidence.shape != (length,):
            raise ValueError(
                f"{self.candidate_id}: expected confidence {(length,)}, got {self.confidence.shape}"
            )
        if self.support_mask.shape != (length,):
            raise ValueError(
                f"{self.candidate_id}: expected support_mask {(length,)}, got {self.support_mask.shape}"
            )
        finite_rows = np.isfinite(self.coords).all(axis=1)
        partial_rows = np.isfinite(self.coords).any(axis=1) & ~finite_rows
        if partial_rows.any():
            raise ValueError(f"{self.candidate_id}: coordinate rows must be fully finite or fully NaN")
        if np.any(self.support_mask & ~finite_rows):
            raise ValueError(f"{self.candidate_id}: supported residues must have finite coordinates")
        if not np.isfinite(self.confidence).all():
            raise ValueError(f"{self.candidate_id}: confidence contains NaN/inf")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError(f"{self.candidate_id}: confidence must lie in [0, 1]")
        if not np.isfinite(self.global_confidence) or not 0.0 <= self.global_confidence <= 1.0:
            raise ValueError(f"{self.candidate_id}: global_confidence must lie in [0, 1]")
        for name, value in self.priors.items():
            if safe_name(name) != name:
                raise ValueError(f"prior name must already be filename-safe: {name!r}")
            array = np.asarray(value)
            if array.dtype == object:
                raise ValueError(f"{self.candidate_id}: prior {name!r} has object dtype")
            self.priors[name] = array

    @property
    def valid_mask(self) -> np.ndarray:
        """Residues with a complete finite XYZ coordinate."""
        return np.isfinite(self.coords).all(axis=1)


def from_tbm_candidate(candidate: Any, sequence: str) -> StructureCandidate:
    """Adapt the existing TBM ``Candidate`` without coupling the two dataclasses."""
    search_source = str(getattr(candidate, "source", "tbm"))
    chain_key = str(candidate.chain_key)
    metadata = dict(getattr(candidate, "meta", {}))
    metadata.update(
        {
            "chain_key": chain_key,
            "identity": float(candidate.identity),
            "coverage": float(candidate.coverage),
            "tbm_search_source": search_source,
        }
    )
    return StructureCandidate(
        target_id=str(candidate.target_id),
        sequence=sequence,
        candidate_id=f"tbm__{safe_name(chain_key)}",
        kind="template",
        source="tbm",
        model="composite_tbm_v1",
        coords=candidate.coords,
        confidence=candidate.conf_residue,
        support_mask=candidate.mask,
        global_confidence=float(candidate.confidence),
        metadata=metadata,
    )


class CandidateCache:
    """Filesystem-backed cache organized as ``root/split/target/candidate.npz``."""

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = safe_name(split)

    @property
    def split_dir(self) -> Path:
        return self.root / self.split

    def candidate_path(self, candidate: StructureCandidate) -> Path:
        return self.split_dir / safe_name(candidate.target_id) / f"{safe_name(candidate.candidate_id)}.npz"

    def save(self, candidate: StructureCandidate, *, overwrite: bool = False) -> Path:
        """Atomically save one candidate, rejecting accidental overwrites by default."""
        path = self.candidate_path(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            existing = self.load_file(path)
            if existing.sequence != candidate.sequence:
                raise ValueError(f"cache collision at {path}: sequence differs")
            return path

        document = {
            "schema_version": SCHEMA_VERSION,
            "target_id": candidate.target_id,
            "sequence": candidate.sequence,
            "sequence_sha256": sequence_digest(candidate.sequence),
            "candidate_id": candidate.candidate_id,
            "kind": candidate.kind,
            "source": candidate.source,
            "model": candidate.model,
            "global_confidence": candidate.global_confidence,
            "metadata": candidate.metadata,
            "prior_keys": sorted(candidate.priors),
        }
        arrays: dict[str, np.ndarray] = {
            "coords": candidate.coords,
            "confidence": candidate.confidence,
            "support_mask": candidate.support_mask,
            "document": np.asarray(json.dumps(document, sort_keys=True)),
        }
        arrays.update({f"prior__{key}": value for key, value in candidate.priors.items()})

        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
            ) as tmp:
                tmp_name = tmp.name
                np.savez_compressed(tmp, **arrays)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return path

    def load_file(self, path: str | Path) -> StructureCandidate:
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            document = json.loads(str(payload["document"].item()))
            if document.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported candidate cache schema at {path}: "
                    f"{document.get('schema_version')} != {SCHEMA_VERSION}"
                )
            sequence = document["sequence"]
            if document.get("sequence_sha256") != sequence_digest(sequence):
                raise ValueError(f"candidate cache sequence digest mismatch: {path}")
            priors = {
                key: payload[f"prior__{key}"].copy()
                for key in document.get("prior_keys", [])
            }
            return StructureCandidate(
                target_id=document["target_id"],
                sequence=sequence,
                candidate_id=document["candidate_id"],
                kind=document["kind"],
                source=document["source"],
                model=document["model"],
                coords=payload["coords"].copy(),
                confidence=payload["confidence"].copy(),
                support_mask=payload["support_mask"].copy(),
                global_confidence=document["global_confidence"],
                metadata=document.get("metadata", {}),
                priors=priors,
            )

    def load_target(self, target_id: str, sequence: str | None = None) -> list[StructureCandidate]:
        target_dir = self.split_dir / safe_name(target_id)
        if not target_dir.exists():
            return []
        candidates = [self.load_file(path) for path in sorted(target_dir.glob("*.npz"))]
        if sequence is not None:
            normalized = sequence.upper().replace("T", "U")
            wrong = [candidate.candidate_id for candidate in candidates if candidate.sequence != normalized]
            if wrong:
                raise ValueError(f"{target_id}: stale candidates for another sequence: {wrong}")
        return candidates

    def inventory(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.split_dir.exists():
            return rows
        for path in sorted(self.split_dir.glob("*/*.npz")):
            candidate = self.load_file(path)
            rows.append(
                {
                    "target_id": candidate.target_id,
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "source": candidate.source,
                    "model": candidate.model,
                    "length": len(candidate.sequence),
                    "resolved_fraction": float(candidate.valid_mask.mean()),
                    "supported_fraction": float(candidate.support_mask.mean()),
                    "global_confidence": candidate.global_confidence,
                    "path": str(path),
                }
            )
        return rows
