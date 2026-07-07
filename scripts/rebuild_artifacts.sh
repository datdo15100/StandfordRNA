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
echo "== using python: $($PY -c 'import sys;print(sys.executable)')"
echo "== data dir: $($PY -c 'import sys;sys.path.insert(0,"src");from rna3d.paths import comp_dir;print(comp_dir())')"

echo "== [1/3] geometry priors (temporal-safe) =="
$PY scripts/run_phase2_priors.py

echo "== [2/3] template DB (parse ~8.6k CIFs; minutes on NVMe, ~35min on a slow mount) =="
$PY scripts/build_template_db.py --workers "$(nproc)"

echo "== [3/3] warm the MMseqs target DB (createdb over template FASTA) =="
$PY - <<'PYEOF'
import sys; sys.path.insert(0, "src")
from rna3d.template.mmseqs_search import ensure_target_db, ensure_template_fasta
ensure_template_fasta(); ensure_target_db()
print("MMseqs target DB ready")
PYEOF

echo "== done. Derived artifacts:"
ls -lh data/processed/geometry_priors.json data/processed/template_meta.parquet \
       data/cache/template_coords.pkl 2>/dev/null || true
echo "Now run:  $PY scripts/run_eval.py   (sanity: scripts/run_phase1_scoring.py)"
