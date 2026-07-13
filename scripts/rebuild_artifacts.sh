#!/usr/bin/env bash
# Rebuild all derived artifacts from the raw competition data, in dependency order.
# Safe to run on a fresh machine after `conda env create -f environment.yml`.
#
# Prereqs:
#   - conda env `rna-fold` active (or set PY to its python)
#   - competition data present; point RNA3D_DATA at it if not under data/
#   - USalign compiled at external/binaries/USalign (see SETUP.md)
#
# Usage:  bash scripts/rebuild_artifacts.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
NPROC="$(nproc)"
# CIF parsing is both CPU- and memory-heavy.  Keep enough headroom for Windows
# and the WSL filesystem cache on a 24 GB workstation; advanced users can still
# opt in to more parallelism with RNA3D_WORKERS.
if (( NPROC < 6 )); then
  DEFAULT_WORKERS="$NPROC"
else
  DEFAULT_WORKERS=6
fi
WORKERS="${RNA3D_WORKERS:-$DEFAULT_WORKERS}"
echo "== using python: $($PY -c 'import sys;print(sys.executable)')"
echo "== data dir: $($PY -c 'import sys;sys.path.insert(0,"src");from rna3d.paths import comp_dir;print(comp_dir())')"
echo "== CIF parser workers: $WORKERS (override with RNA3D_WORKERS)"

echo "== [1/3] geometry priors (temporal-safe) =="
$PY scripts/run_phase2_priors.py

echo "== [2/3] template DB (parse ~8.6k CIFs; minutes on NVMe, ~35min on a slow mount) =="
$PY scripts/build_template_db.py --workers "$WORKERS"

echo "== [3/4] warm the MMseqs target DB (createdb over template FASTA) =="
$PY - <<'PYEOF'
import sys; sys.path.insert(0, "src")
from rna3d.template.mmseqs_search import ensure_target_db, ensure_template_fasta
ensure_template_fasta(); ensure_target_db()
print("MMseqs target DB ready")
PYEOF

echo "== [4/4] composite-search library (deduped template set for the search fallback) =="
$PY scripts/build_top1_from_existing.py

echo "== done. Derived artifacts:"
ls -lh data/processed/geometry_priors.json data/processed/template_meta.parquet \
       data/cache/template_coords.pkl 2>/dev/null || true
echo "Now run:  $PY scripts/run_eval.py   (sanity: scripts/run_phase1_scoring.py)"
