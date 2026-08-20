#!/usr/bin/env python
"""Close the remaining V4 Phase-1 audits without reading native performance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.baselines.john_hybrid import run_public_hybrid_route


OUT = REPO / "reports" / "thesis_v4" / "phase1_completion"
MASTER = REPO / "reports" / "thesis_v4" / "preregistration" / "v4_master_rna_ledger.csv"
KNOWN = REPO / "reports" / "thesis_v4" / "preregistration" / "development_exposure_ledger.csv"
PRETRAINED = REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "pretrained_provenance_audit.json"
HYBRID_CAPTURE = REPO / "utilities" / "top1_4_4_hybrid_final_take.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def external_exposure_audit() -> tuple[pd.DataFrame, dict]:
    master = pd.read_csv(MASTER, dtype=str).fillna("")
    known = set(pd.read_csv(KNOWN, dtype=str)["target_id"])
    candidate_ids = set(master.loc[master["provisional_after_repository_audit"] == "True", "target_id"])

    roots = [
        REPO / "docs",
        REPO / "reports" / "thesis_v1",
        REPO / "reports" / "thesis_v2",
        REPO / "reports" / "thesis_v3",
        REPO / "reports" / "thesis_notes",
        Path("/home/datdo/.codex/attachments"),
    ]
    single_files = [Path("/home/datdo/.bash_history")]
    suffixes = {".md", ".txt", ".tex", ".csv", ".json", ".yaml", ".yml", ".log"}
    evidence: dict[str, set[str]] = {target_id: set() for target_id in candidate_ids}
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.stat().st_size > 25 * 1024 * 1024:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            scanned += 1
            tokens = set(re.findall(r"(?<![A-Za-z0-9])(?:[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+)(?![A-Za-z0-9])", text))
            for target_id in tokens & candidate_ids:
                evidence[target_id].add(relative(path))
    for path in single_files:
        if not path.exists() or path.stat().st_size > 25 * 1024 * 1024:
            continue
        text = path.read_text(errors="ignore")
        scanned += 1
        for target_id in candidate_ids:
            if target_id in text:
                evidence[target_id].add(relative(path))

    rows = []
    for target_id in sorted(candidate_ids):
        paths = sorted(evidence[target_id])
        rows.append(
            {
                "target_id": target_id,
                "repository_known_exposed": target_id in known,
                "additional_external_text_evidence": bool(paths),
                "evidence_paths": " | ".join(paths),
                "manual_researcher_memory_status": "UNAVAILABLE_NOT_INDEPENDENTLY_VERIFIABLE",
                "audit_decision": "EXCLUDE_ADDITIONAL_EXPOSURE" if paths else "NO_ADDITIONAL_EVIDENCE_FOUND",
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "scope": "conservative text/artifact scan outside the already frozen repository result-table audit",
        "files_scanned": scanned,
        "provisional_targets_checked": len(candidate_ids),
        "additional_targets_with_evidence": int(frame["additional_external_text_evidence"].sum()),
        "manual_researcher_memory_status": "UNAVAILABLE_NOT_INDEPENDENTLY_VERIFIABLE",
        "limitation": (
            "Absence from accessible files is not proof that a target was never viewed. "
            "The audit can exclude positive evidence but cannot reconstruct unaudited human memory."
        ),
        "performance_accessed": False,
    }
    return frame, summary


def hybrid_smoke() -> dict:
    base = np.arange(12, dtype=float).reshape(4, 3)
    template = lambda: [base + index for index in range(5)]
    hybrid = lambda: [base]
    direct = run_public_hybrid_route(
        retained_dataframe_index=0,
        target_count=12,
        elapsed_seconds=0,
        drfold_time_limit_seconds=7 * 60 * 60,
        boltz_conditioned_drfold_runner=hybrid,
        template_runner=template,
    )
    unavailable = run_public_hybrid_route(
        retained_dataframe_index=0,
        target_count=12,
        elapsed_seconds=0,
        drfold_time_limit_seconds=7 * 60 * 60,
        boltz_conditioned_drfold_runner=None,
        template_runner=template,
    )
    outside = run_public_hybrid_route(
        retained_dataframe_index=1,
        target_count=12,
        elapsed_seconds=0,
        drfold_time_limit_seconds=7 * 60 * 60,
        boltz_conditioned_drfold_runner=hybrid,
        template_runner=template,
    )
    return {
        "captured_small_cohort_retained_index_range": [0, 0],
        "injected_hybrid_route": direct.executed_route,
        "injected_hybrid_candidate_count": len(direct.structures),
        "missing_artifact_route": unavailable.executed_route,
        "missing_artifact_reason": unavailable.fallback_reason,
        "outside_index_route": outside.executed_route,
        "outside_index_reason": outside.fallback_reason,
        "status": "EXECUTABLE_CONTROL_FLOW_MODEL_BRANCH_NOT_REPRODUCIBLE",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exposure, exposure_summary = external_exposure_audit()
    exposure_path = OUT / "external_exposure_audit.csv"
    exposure.to_csv(exposure_path, index=False)
    pretrained = json.loads(PRETRAINED.read_text())
    audit = {
        "phase": "V4 Phase 1 completion",
        "performance_accessed": False,
        "john_public_hybrid": {
            "required_name": "reproduced publicly released John pipeline",
            "capture_path": relative(HYBRID_CAPTURE),
            "capture_sha256": sha256(HYBRID_CAPTURE),
            "smoke": hybrid_smoke(),
            "boltz_status": "UNAVAILABLE_NOT_REPRODUCIBLE_WITH_AVAILABLE_ARTIFACTS",
            "structural_conclusion": (
                "The captured routing, padding and template fallback are executable. "
                "The Boltz-conditioned DRfold2 branch is not structurally reproducible "
                "because the original Boltz checkpoint/source configuration is absent."
            ),
        },
        "j_controlled_full": {
            "status": "EXECUTABLE_TEMPLATE_FALLBACK_PARTIAL_HYBRID",
            "definition": (
                "Controlled John routing on DB-controlled; unavailable Boltz-conditioned "
                "DRfold2 requests enter the documented five-candidate template fallback."
            ),
            "use_for_component_attribution": False,
        },
        "external_exposure": exposure_summary,
        "drfold2": {
            "structural_provenance_status": "AUDITED_FROM_PAPER_AND_LOCAL_TRAINING_FASTA_IDENTITY_UNVERIFIED",
            "structural_overlap_audit_status": (
                f"{pretrained['drfold2']['v4_overlap']['structural_overlap_pass']} of "
                f"{pretrained['drfold2']['v4_overlap']['candidate_targets']} provisional targets pass 80/80 overlap filtering"
            ),
            "rclm_status": "UNAVAILABLE_CORPUS_MEMBERSHIP_MANIFEST",
            "rclm_public_scope": "RNAcentral release 22, approximately 30 million sequences, reported by the DRfold2 paper/code audit",
            "claim_constraint": "Do not claim complete pretrained time-safety.",
        },
        "boltz": {
            "status": "UNAVAILABLE_NOT_REPRODUCIBLE_WITH_AVAILABLE_ARTIFACTS",
            "primary_v4": False,
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    audit_path = OUT / "phase1_completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    readme = f"""# V4 Phase 1 completion

Status: **COMPLETE WITH DECLARED PARTIAL REPRODUCTIONS**

- Native performance opened: **No**.
- Public John hybrid: executable control-flow reproduction; Boltz-conditioned structural branch unavailable.
- J-controlled full: executable DB-controlled template fallback, retained only as a descriptive partial comparator.
- DRfold2 structural provenance/overlap: audited; RCLM corpus membership manifest: **UNAVAILABLE**.
- Boltz: **UNAVAILABLE** and excluded from primary V4 evidence.
- Additional external text evidence among the repository-provisional pool: **{exposure_summary['additional_targets_with_evidence']} target(s)**.
- Manual memory cannot be reconstructed independently; this limitation remains explicit.

No exact-winning-pipeline or complete pretrained time-safety claim is permitted.
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
