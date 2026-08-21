#!/usr/bin/env python
"""Generate the small Kaggle notebook that invokes frozen V4 inference."""
from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).with_name("v4_frozen_submission.ipynb")


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


document = {
    "cells": [
        {
            "id": "v4-description",
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "# RNA3D thesis V4 frozen pipeline\n\n"
                "Native-blind external deployment check for the method frozen before the "
                "97-target confirmatory opening.\n"
            ),
        },
        {
            "id": "v4-runtime-setup",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                "from pathlib import Path\n"
                "import json\n"
                "import os\n"
                "import shutil\n"
                "import subprocess\n"
                "import sys\n"
                "import tarfile\n\n"
                "import pandas as pd\n"
                "import torch\n\n"
                "INPUT = Path('/kaggle/input')\n"
                "WORKING = Path('/kaggle/working')\n"
                "runtime_priors = list(INPUT.rglob('geofuse_geometry_v2_priors.json'))\n"
                "runtime_archives = list(INPUT.rglob('rna3d_v4_runtime.tar.gz'))\n"
                "if len(runtime_priors) == 1:\n"
                "    RUNTIME = runtime_priors[0].parent\n"
                "elif len(runtime_archives) == 1:\n"
                "    RUNTIME = WORKING / 'rna3d_v4_runtime'\n"
                "    if RUNTIME.exists():\n"
                "        shutil.rmtree(RUNTIME)\n"
                "    RUNTIME.mkdir(parents=True)\n"
                "    with tarfile.open(runtime_archives[0], 'r:gz') as archive:\n"
                "        archive.extractall(RUNTIME)\n"
                "else:\n"
                "    raise FileNotFoundError(f'V4 runtime is missing: priors={runtime_priors}, archives={runtime_archives}')\n\n"
                "bundle_manifests = list(INPUT.rglob('bundle_manifest.json'))\n"
                "bundle_archives = list(INPUT.rglob('rna3d_bundle.tar.gz'))\n"
                "if bundle_manifests:\n"
                "    BUNDLE = bundle_manifests[0].parent\n"
                "elif len(bundle_archives) == 1:\n"
                "    BUNDLE = WORKING / 'rna3d_bundle'\n"
                "    if BUNDLE.exists():\n"
                "        shutil.rmtree(BUNDLE)\n"
                "    BUNDLE.mkdir(parents=True)\n"
                "    with tarfile.open(bundle_archives[0], 'r:gz') as archive:\n"
                "        archive.extractall(BUNDLE)\n"
                "else:\n"
                "    raise FileNotFoundError(f'expected one TBM artifact bundle, found {bundle_archives}')\n\n"
                "competition_roots = [\n"
                "    path.parent for path in INPUT.rglob('test_sequences.csv')\n"
                "    if (path.parent / 'sample_submission.csv').is_file()\n"
                "]\n"
                "if len(competition_roots) != 1:\n"
                "    raise FileNotFoundError(f'expected one competition input, found {competition_roots}')\n"
                "COMPETITION = competition_roots[0]\n"
                "SITE_PACKAGES = Path('/kaggle/temp/rna3d_site_packages')\n"
                "subprocess.run([\n"
                "    sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',\n"
                "    '--find-links', str(RUNTIME / 'wheels'), '--target', str(SITE_PACKAGES), 'biopython'\n"
                "], check=True)\n"
                "sys.path.insert(0, str(SITE_PACKAGES))\n"
                "sys.path.insert(0, str(RUNTIME))\n"
                "sys.path.insert(0, str(RUNTIME / 'src'))\n"
                "os.environ['RNA3D_PROCESSED'] = str(BUNDLE)\n"
                "os.environ['RNA3D_CACHE'] = str(BUNDLE)\n"
                "old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')\n"
                "os.environ['LD_LIBRARY_PATH'] = f'{BUNDLE / \"lib\"}:{old_ld_library_path}' if old_ld_library_path else str(BUNDLE / 'lib')\n"
                "print(f'cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')\n"
            ),
        },
        {
            "id": "v4-inference",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                "from kaggle.v4_frozen_inference import run_v4_frozen_inference\n\n"
                "test = pd.read_csv(COMPETITION / 'test_sequences.csv', dtype=str)\n"
                "sample = pd.read_csv(COMPETITION / 'sample_submission.csv')\n"
                "submission, manifest = run_v4_frozen_inference(\n"
                "    test, BUNDLE, RUNTIME, INPUT,\n"
                "    work_dir=Path('/kaggle/temp/rna3d_v4_work'),\n"
                "    sample_submission=sample,\n"
                "    drfold_max_len=600,\n"
                "    drfold_deadline_seconds=6.5 * 60 * 60,\n"
                ")\n"
                "output = WORKING / 'submission.csv'\n"
                "submission.to_csv(output, index=False)\n"
                "manifest['submission_sha256'] = __import__('hashlib').sha256(output.read_bytes()).hexdigest()\n"
                "(WORKING / 'v4_inference_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\\n')\n"
                "assert submission.shape == sample.shape\n"
                "assert submission['ID'].tolist() == sample['ID'].tolist()\n"
                "assert not submission.isna().any().any()\n"
                "print(f'wrote {output}; targets={len(test)}; rows={len(submission)}; runtime={manifest[\"elapsed_seconds\"]}s')\n"
            ),
        },
        {
            "id": "v4-preview",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["submission.head(3)"],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
print(OUTPUT)
