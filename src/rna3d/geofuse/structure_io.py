"""Import pretrained structure files into the GeoFuse candidate contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .candidate import StructureCandidate, safe_name


def _chain_c1_coords(path: Path) -> list[tuple[str, np.ndarray]]:
    """Return C1' coordinates separately for every chain in PDB/mmCIF input."""
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - environment.yml supplies gemmi
        raise RuntimeError("gemmi is required to import PDB/mmCIF candidates") from exc

    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"no models in structure: {path}")
    chains: list[tuple[str, np.ndarray]] = []
    for chain in structure[0]:
        coords = []
        for residue in chain:
            atom = next((a for a in residue if a.name.strip() in {"C1'", "C1*"}), None)
            if atom is not None:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        if coords:
            chains.append((chain.name, np.asarray(coords, dtype=np.float32)))
    return chains


def read_c1_coords(path: str | Path, expected_length: int) -> tuple[np.ndarray, str]:
    """Read the unique chain matching ``expected_length`` (or an exact concatenation)."""
    path = Path(path)
    chains = _chain_c1_coords(path)
    exact = [(name, coords) for name, coords in chains if len(coords) == expected_length]
    if len(exact) == 1:
        name, coords = exact[0]
        return coords, name
    if len(exact) > 1:
        names = [name for name, _ in exact]
        raise ValueError(f"{path}: multiple chains match length {expected_length}: {names}")
    total = sum(len(coords) for _, coords in chains)
    if chains and total == expected_length:
        return np.concatenate([coords for _, coords in chains], axis=0), "+".join(
            name for name, _ in chains
        )
    observed = {name: len(coords) for name, coords in chains}
    raise ValueError(
        f"{path}: no unambiguous C1' chain of length {expected_length}; observed {observed}"
    )


def _normalize_confidence(values: np.ndarray, length: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=float).squeeze()
    if values.shape != (length,) or not np.isfinite(values).all():
        return None
    if float(values.max(initial=0.0)) > 1.0:
        values = values / 100.0
    if np.any((values < 0.0) | (values > 1.0)):
        return None
    return values.astype(np.float32)


def infer_sidecar_confidence(
    structure_path: str | Path, length: int, default: float = 0.5
) -> tuple[np.ndarray, float, dict]:
    """Read Boltz-style pLDDT/JSON sidecars, with an explicit neutral fallback."""
    path = Path(structure_path)
    stem = path.stem
    directory = path.parent
    local: np.ndarray | None = None
    provenance: dict = {"confidence_origin": "default"}

    for npz_path in (directory / f"plddt_{stem}.npz", directory / f"{stem}_plddt.npz"):
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as payload:
            for key in ("plddt", "confidence", *payload.files):
                if key in payload.files:
                    local = _normalize_confidence(payload[key], length)
                    if local is not None:
                        provenance = {"confidence_origin": "sidecar_plddt", "confidence_path": str(npz_path)}
                        break
        if local is not None:
            break

    global_confidence: float | None = None
    for json_path in (directory / f"confidence_{stem}.json", directory / f"{stem}_confidence.json"):
        if not json_path.exists():
            continue
        document = json.loads(json_path.read_text())
        for key in ("confidence_score", "ptm", "complex_plddt"):
            value = document.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                global_confidence = float(value)
                if global_confidence > 1.0:
                    global_confidence /= 100.0
                provenance["global_confidence_origin"] = key
                provenance["global_confidence_path"] = str(json_path)
                break
        if global_confidence is not None:
            break

    fallback = float(default)
    if not 0.0 <= fallback <= 1.0:
        raise ValueError("default confidence must lie in [0, 1]")
    if local is None:
        local = np.full(length, fallback, dtype=np.float32)
    if global_confidence is None:
        global_confidence = float(local.mean())
    return local, global_confidence, provenance


def infer_sidecar_priors(structure_path: str | Path, length: int) -> dict[str, np.ndarray]:
    """Load an optional safe ``priors_<structure-stem>.npz`` sidecar.

    DRfold's native ``.ret`` files are Python pickles.  The model runner converts
    trusted locally-generated ret files once; all downstream code then consumes
    this pickle-free sidecar instead.
    """
    path = Path(structure_path)
    prior_path = path.parent / f"priors_{path.stem}.npz"
    if not prior_path.exists():
        return {}
    priors: dict[str, np.ndarray] = {}
    with np.load(prior_path, allow_pickle=False) as payload:
        for key in payload.files:
            value = np.asarray(payload[key])
            if value.ndim < 2 or value.shape[:2] != (length, length):
                raise ValueError(
                    f"{prior_path}: prior {key!r} starts with {value.shape[:2]}, "
                    f"expected {(length, length)}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{prior_path}: prior {key!r} contains NaN/inf")
            priors[safe_name(key)] = value
    return priors


def import_structure(
    path: str | Path,
    *,
    target_id: str,
    sequence: str,
    candidate_id: str,
    source: str,
    model: str,
    default_confidence: float = 0.5,
    priors: dict[str, np.ndarray] | None = None,
) -> StructureCandidate:
    """Normalize one trusted model output structure and its available confidence."""
    path = Path(path)
    coords, chain_id = read_c1_coords(path, len(sequence))
    confidence, global_confidence, confidence_meta = infer_sidecar_confidence(
        path, len(sequence), default=default_confidence
    )
    metadata = {
        "raw_structure_path": str(path.resolve()),
        "raw_structure_format": path.suffix.lower().lstrip("."),
        "chain_id": chain_id,
        **confidence_meta,
    }
    return StructureCandidate(
        target_id=target_id,
        sequence=sequence,
        candidate_id=safe_name(candidate_id),
        kind="pretrained",
        source=source,
        model=model,
        coords=coords,
        confidence=confidence,
        support_mask=np.ones(len(sequence), dtype=bool),
        global_confidence=global_confidence,
        metadata=metadata,
        priors=priors if priors is not None else infer_sidecar_priors(path, len(sequence)),
    )


def discover_structure_files(root: str | Path, patterns: Iterable[str], target_id: str) -> list[Path]:
    """Expand target-aware glob patterns and return deterministic unique paths."""
    root = Path(root)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.glob(pattern.format(target_id=target_id)) if path.is_file())
    return sorted(found)
