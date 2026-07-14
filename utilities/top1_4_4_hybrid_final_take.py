# Credits:

# https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission
Boltz-1
!pip install --no-index /kaggle/input/boltz-dependencies/*whl --no-deps
!pip install --no-index /kaggle/input/fairscale-0413/*whl --no-deps
!pip install --no-index /kaggle/input/biopython/*whl --no-deps
Processing /kaggle/input/boltz-dependencies/click-8.1.7-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/dm_tree-0.1.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Processing /kaggle/input/boltz-dependencies/einops-0.8.0-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/einx-0.3.0-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/hydra_core-1.3.2-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/ihm-2.2-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/mashumaro-3.14-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/modelcif-1.3-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Processing /kaggle/input/boltz-dependencies/pandas-2.2.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Processing /kaggle/input/boltz-dependencies/pytorch_lightning-2.4.0-py3-none-any.whl
Processing /kaggle/input/boltz-dependencies/PyYAML-6.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Processing /kaggle/input/boltz-dependencies/rdkit-2024.9.5-cp310-cp310-manylinux_2_28_x86_64.whl
Processing /kaggle/input/boltz-dependencies/scipy-1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
click is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
dm-tree is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
einops is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
pandas is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
PyYAML is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
scipy is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.
Installing collected packages: rdkit, modelcif, ihm, hydra-core, pytorch-lightning, numpy, mashumaro, einx
  Attempting uninstall: pytorch-lightning
    Found existing installation: pytorch-lightning 2.5.0.post0
    Uninstalling pytorch-lightning-2.5.0.post0:
      Successfully uninstalled pytorch-lightning-2.5.0.post0
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.4
    Uninstalling numpy-1.26.4:
      Successfully uninstalled numpy-1.26.4
Successfully installed einx-0.3.0 hydra-core-1.3.2 ihm-2.2 mashumaro-3.14 modelcif-1.3 numpy-1.26.3 pytorch-lightning-2.4.0 rdkit-2024.9.5
Processing /kaggle/input/fairscale-0413/fairscale-0.4.13-py3-none-any.whl
Installing collected packages: fairscale
Successfully installed fairscale-0.4.13
Processing /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Installing collected packages: biopython
Successfully installed biopython-1.85
pwd
'/kaggle/working'
%mkdir inputs_prediction
%mkdir outputs_prediction
%cp -rf /kaggle/input/rna-prediction-boltz/boltz/src/boltz .
%ls boltz
data/  __init__.py  main.py  model/
%%writefile inference.py

import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        click.echo(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )

        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            click.echo("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        click.echo(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        click.echo(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []
    for path in tqdm(data):
        try:
            # Parse data
            if path.suffix in (".fa", ".fas", ".fasta"):
                target = parse_fasta(path, ccd)
            elif path.suffix in (".yml", ".yaml"):
                target = parse_yaml(path, ccd)
            elif path.is_dir():
                msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

            # Get target id
            target_id = target.record.id

            # Get all MSA ids and decide whether to generate MSA
            to_generate = {}
            prot_id = const.chain_type_ids["PROTEIN"]
            for chain in target.record.chains:
                # Add to generate list, assigning entity id
                if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                    entity_id = chain.entity_id
                    msa_id = f"{target_id}_{entity_id}"
                    to_generate[msa_id] = target.sequences[entity_id]
                    chain.msa_id = msa_dir / f"{msa_id}.csv"

                # We do not support msa generation for non-protein chains
                elif chain.msa_id == 0:
                    chain.msa_id = -1

            # Generate MSA
            if to_generate and not use_msa_server:
                msg = "Missing MSA's in input and --use_msa_server flag not set."
                raise RuntimeError(msg)

            if to_generate:
                msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target_id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                )

            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(int(seed))

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


if __name__ == "__main__":
    predict(data="./inputs_prediction",
            out_dir="./outputs_prediction",
            cache="/kaggle/input/rna-prediction-boltz/",
            diffusion_samples=1,
            recycling_steps=10,
            accelerator="gpu",
            sampling_steps=500,
            seed=42,
            override=True)
Writing inference.py
import os
import pandas as pd
import numpy as np

sub_file = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

names = sub_file['target_id'].tolist()
sequences = sub_file['sequence'].tolist()

# Inference
idx = 0
for tmp_id, tmp_sequence in zip(names, sequences):
    with open(f'/kaggle/working/inputs_prediction/{tmp_id}.yaml', 'w') as f:
        f.write("constraints: []\n")
        f.write("sequences:\n")
        f.write("- rna:\n")
        f.write("    id:\n")
        f.write("    - A1\n")
        f.write(f"    sequence: {tmp_sequence}")
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

import subprocess
import logging
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

result = subprocess.run(['python', 'inference.py'], capture_output=True, text=True)
logger.info(f"Command output: {result.stdout}")
logger.error(f"Command error: {result.stderr}")

def get_coords(tmp_id, idx):
    cif_file = f"outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif"
    mmcif_dict = MMCIF2Dict(cif_file)
    entity_poly_seq = mmcif_dict.get("_entity_poly_seq.mon_id", [])
    sequence = "".join(entity_poly_seq)
    print("RNA sequence:", sequence)
    x_coords = mmcif_dict["_atom_site.Cartn_x"]
    y_coords = mmcif_dict["_atom_site.Cartn_y"]
    z_coords = mmcif_dict["_atom_site.Cartn_z"]
    atom_names = mmcif_dict["_atom_site.label_atom_id"]
    c1_coords = []
    for i, atom in enumerate(atom_names):
        if atom == "C1'":
            c1_coords.append((float(x_coords[i]), float(y_coords[i]), float(z_coords[i])))
    return c1_coords

all_preds = os.listdir('outputs_prediction/boltz_results_inputs_prediction/predictions')
submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

idx = 0
for tmp_id in all_preds:
    print('#'*20, f'inferences for {tmp_id}')
    for idx in range(1):
        c1_coords = get_coords(tmp_id, idx)
        submission.loc[submission['ID'].apply(lambda x: tmp_id in x), [f'x_{idx+1}', f'y_{idx+1}', f'z_{idx+1}']] = c1_coords
    print()

# Save the submission
submission.to_csv('submission_boltz.csv', index=False)
logger.info("Created submission from predictions")
#################### inferences for R1136
RNA sequence: GGAUACGUCUACGCUCAGUGACGGACUCUCUUCGGAGAGUCUGACAUCCGAACCAUACACGGAUGUGCCUCGCCGAACAGUCUACGGCGAGCUUAAGCGCUGGGGACGCCCAACGCAUCACAAAGACUGAGUGAUGAACCAGAAGUAUGGACUGGUUGCGUUGGUGGAGACGGUCGGGUCCAGUUCGCUGUCGAGUAGAGUGUGGGCUCCAUCGACGCCGCUUUAAGGUCCCCAAUCGUGGCGUGUCGGCCUGCUUCGGCAGGCACUGGCGCCGGGACCUUGAAGAGAUGAGAUUUCGAUCUCAUCUUUGGGUGUCUCUGGUGCUUGAGGGCCCUGUGUUCGCACAGGGCCGCUCACUGGGUGUGGACGUAUCC

#################### inferences for R1190
RNA sequence: GCGUACAGGGAACACGCAACCCCGAAGGAUCGGGGAAGGGACGUCGCCAGGGAGGCGAUUCCAUCAGGAUGAUGACGAGGGACUGAAGAGUGGGCGGGGUAAUACCCCGCCCCUUUUU

#################### inferences for R1108
RNA sequence: GGGGGCCACAGCAGAAGCGUUCACGUCGCGGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU

#################### inferences for R1116
RNA sequence: CGCCCGGAUAGCUCAGUCGGUAGAGCAGCGGCUAAAACAGCUCUGGGGUUGUACCCACCCCAGAGGCCCACGUGGCGGCUAGUACUCCGGUAUUGCGGUACCCUUGUACGCCUGUUUUAGCCGCGGGUCCAGGGUUCAAGUCCCUGUUCGGGCGCCA

#################### inferences for R1149
RNA sequence: GGACACGAGUAACUCGUCUAUCUUCUGCAGGCUGCUUACGGUUUCGUCCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAGCCUUGUCCC

#################### inferences for R1107
RNA sequence: GGGGGCCACAGCAGAAGCGUUCACGUCGCAGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU

#################### inferences for R1117v2
RNA sequence: UUGGGUUCCCUCACCCCAAUCAUAAAAAGG

#################### inferences for R1128
RNA sequence: GGAAUAUCGUCAUGGUGAUUCGUCACCAUGAGGCUAGAUCUCAUAUCUAGCGCUUUCGAGCGCUAGAGUCCUUAUCUAGCCGGUUUAUACUUUCGAGUGUGAACCCGAUAUUCCGCGGAUCACUAUGAGUCGUUCGCGGCUCAUAGUCCGGCUCAAAGGACAUCAUGGCCUGUUCGCAGGUUGUGAUUAUGAGUGAGCCGGGUAAGGCAUACCGUUCGCGGUAUGUCUUACGAUCCGC

#################### inferences for R1156
RNA sequence: GGAGCAUCGUGUCUCAAGUGCUUCACGGUCACAAUAUACCGUUUCGUCGGGUGCGUGGCAAUUCGGUGCACAUCAUGUCUUUCGUGGCUGGUGUGGCUCCUCAAGGUGCGAGGGGCAAGUAUAGAGCAGAGCUCC

#################### inferences for R1126
RNA sequence: GGAAUCUCGCCCGAUGUUCGCAUCGGGAUUUGCAGGUCCAUGGAUUACACCAUGCAACGCAGACCUGUAGAUGCCACGCUAGCCGUGGUGAGGGUCGGGUCCAGAUGUCAUUCGACUUUAACGCGCCUAAGCGUUGAAGGCGUGUUAGAGCAGAUAGUUCGCUAUCUGGGGAGCCUGUUCGCAGGCUCAGGAGCCUUCGGGCUCCUAGCGCUAUUACCCCGGACACCACCGGGCAGACAAGUAAUGGUGCUCCUCGAAUGACUUCUGUUGAGUAGAGUGUGGGCUCCGCGGCUAGUGUGCACCUUAGCGGUGAAUGUCUGACACCGUUAAGGUGGUUACUCUUCGGAGUAACGCCGAGAUUCC

#################### inferences for R1138
RNA sequence: GGGAGAGUACUAUUCAGAUGCAGACCGCAAGUUCAGAGCGGUUUGCAUCUAGGGUACGUUUUCGAACGUAUCCUCCGACUAAGUGUAUUCGUAUACUUAGUGCCUUGUGCCUGCUUCGGCAGGCAUGACCCAAAUGUGCCUUUCGGGGCACAUUUCCGGUCAUCCAAGUUCGCUUGGGUGAUGCGGGCGUAUAGGUUCGUCUAUACGUCCGCGUUUUCCGAGAAGAGGUAACUCGGGAAACCGGUCCACGUGACAAAGGUAGAGUUACGUGGAGGGAGCAGCUGCAAAGGGAUAAUGCAGUUGCUGGCUGGAUGCCAGAACUCACGACUGGCAUCUACGGGGAUGGUGCUCUCCCAAUUCUCCAUUUACCGCCGAAUCGACCCCAACGUGAGAGGGGUCGGUUCCCCGAGCAUAGACCAAUAUCCCAGGUUUAUGCUCCCCAACGCUGGACGAACUACCUACGUCUAGCGUUCCGGCAAAUGAGUCAAUACCUCAGACUUAUUUGCGGUGCCUGAGCCUAAACUGAACAUGGGUUCAGGCAUCUUGGCUCCAGUUCGCUGGAGCCGACGGUAGCGCUGCGUUCGCGCAGUGCUAGGGAGCAUCCGUUUUCGAGCGGAUGCUGGGCGGUUGCCUGUUCGCAGGCAAUCGGGCCUACUCAUGAUUCGUCAUGAGUGGUGACAGCGUGAUGUUCGCAUUACGCUGUCGGGUAGAUGGAGAAUU

#################### inferences for R1189
RNA sequence: GCGUACAGGGAACACGCAACCCCGAAGGAUCGGGGAAGGGACGUCGCCAGGGAGGCGAUUCCAUCAGGAUGAUGACGAGGGACUGAAGAGUGGGCGGGGUAAUACCCCGCCCCUUUUU

pwd
'/kaggle/working'
!ls
boltz	      inputs_prediction   outputs_prediction
inference.py  __notebook__.ipynb  submission_boltz.csv
submission
ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
0	R1107_1	G	1	6.50434	-12.16070	-11.78114	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	R1107_2	G	2	6.28621	-7.02678	-13.83277	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2	R1107_3	G	3	4.02922	-1.92618	-14.61284	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
3	R1107_4	G	4	0.78219	2.16863	-12.73632	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
4	R1107_5	G	5	-1.83363	4.35521	-8.35424	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2510	R1189_114	U	114	-19.90772	-9.26861	10.50842	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2511	R1189_115	U	115	-21.97411	-9.46059	5.59679	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2512	R1189_116	U	116	-23.58990	-7.85262	0.52375	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2513	R1189_117	U	117	-23.07915	-4.46932	-3.61029	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2514	R1189_118	U	118	-20.21993	-0.85038	-6.12332	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2515 rows × 18 columns

# %rm -rf boltz
# %rm -rf inputs_prediction
# %rm -rf outputs_prediction
# %rm -rf inference.py
# Free up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
DRfold2 + Template
import os
import time
import torch
import random
import shutil
import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio import pairwise2

from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings('ignore')


test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

is_submission_mode = len(test_sequences) != 12
/usr/local/lib/python3.10/dist-packages/Bio/pairwise2.py:278: BiopythonDeprecationWarning: Bio.pairwise2 has been deprecated, and we intend to remove it in a future release of Biopython. As an alternative, please consider using Bio.Align.PairwiseAligner as a replacement, and contact the Biopython developers if you still need the Bio.pairwise2 module.
  warnings.warn(
train_seqs = pd.read_csv('/kaggle/input/rna-all-data/merged_sequences_final.csv')
train_labels = pd.read_csv('/kaggle/input/rna-all-data/merged_labels_final.csv')
train_seqs.head()
target_id	sequence
0	1SCL_A	GGGUGCUCAGUACGAGAGGAACCGCACCC
1	1RNK_A	GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU
2	1RHT_A	GGGACUGACGAUCACGCAGUCUAU
3	1HLX_A	GGGAUAACUUCGGUUGUCCC
4	1HMH_E	GGCGACCCUGAUGAGGCCGAAAGGCCGAAACCGU
train_labels.head()
ID	resname	resid	x_1	y_1	z_1
0	1SCL_A_1	G	1	13.760	-25.974001	0.102
1	1SCL_A_2	G	2	9.310	-29.638000	2.669
2	1SCL_A_3	G	3	5.529	-27.813000	5.878
3	1SCL_A_4	U	4	2.678	-24.900999	9.793
4	1SCL_A_5	G	5	1.827	-20.136000	11.793
# !ls /kaggle/input/stanford-rna-3d-folding/PDB_RNA
# Check for new CIF files that need processing
import os
import pandas as pd
from pathlib import Path

# Get existing target_ids (without chain suffix)
existing_pdb_ids = set()
for target_id in train_seqs['target_id']:
    pdb_id = target_id.rsplit('_', 1)[0]  # Remove chain suffix
    existing_pdb_ids.add(pdb_id.lower())

print(f"Existing PDB IDs in train_seqs: {len(existing_pdb_ids)}")

# Get all CIF files in directory
cif_dir = '/kaggle/input/stanford-rna-3d-folding/PDB_RNA'
all_cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
all_pdb_ids = set(Path(f).stem.lower() for f in all_cif_files)

print(f"Total CIF files found: {len(all_cif_files)}")

# Find new files to process
new_pdb_ids = all_pdb_ids - existing_pdb_ids
new_cif_files = [f"{pdb_id}.cif" for pdb_id in new_pdb_ids]

print(f"New files to process: {len(new_cif_files)}")
print(f"New PDB IDs: {sorted(list(new_pdb_ids))}")

if new_cif_files:
    print(f"\nFirst 10 new CIF files: {new_cif_files[:10]}")
else:
    print("No new files to process!")
Existing PDB IDs in train_seqs: 8779
Total CIF files found: 8670
New files to process: 38
New PDB IDs: ['1eqq', '1i7j', '1pbl', '1qd7', '1r3o', '1s9l', '1xv6', '2g32', '2gpm', '2gq4', '2gq5', '2gq6', '2gq7', '2jja', '2kwg', '2m39', '2n4j', '2wna', '2x2q', '2xc6', '310d', '3af6', '3iem', '3p4a', '3t3n', '468d', '469d', '470d', '471d', '4r8i', '4rcm', '5js2', '5swm', '8f24', '8wak', '9hbv', '9hbw', '9hbz']

First 10 new CIF files: ['2gq6.cif', '9hbw.cif', '2x2q.cif', '4rcm.cif', '8wak.cif', '2gq7.cif', '1pbl.cif', '2kwg.cif', '471d.cif', '4r8i.cif']
# Comprehensive modified nucleotide mapping
nucleotide_mapping = {
    # Standard nucleotides
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',

    # === ADENOSINE MODIFICATIONS ===
    'I': 'A',      # Inosine (hypoxanthine)
    '1MA': 'A',    # 1-methyladenosine
    '2MA': 'A',    # 2-methyladenosine
    '6MA': 'A',    # N6-methyladenosine (m6A)
    'M2A': 'A',    # N2-methyladenosine
    'MS2': 'A',    # 2-methylthio-N6-isopentenyladenosine
    'AET': 'A',    # 2-aminoethylthio-adenosine
    'A2L': 'A',    # 2'-O-methyladenosine
    'A44': 'A',    # Modified adenosine
    '6OP': 'A',    # Modified adenosine
    '8XA': 'A',    # Modified adenosine
    'ZAD': 'A',    # Modified adenosine

    # === URIDINE MODIFICATIONS ===
    'PSU': 'U',    # Pseudouridine (most common)
    'H2U': 'U',    # Dihydrouridine
    '5MU': 'U',    # 5-methyluridine (ribothymidine)
    '4SU': 'U',    # 4-thiouridine
    '2MU': 'U',    # 2'-O-methyluridine
    'OMU': 'U',    # O-methyluridine
    'T': 'U',      # Thymine (in RNA)
    'RT': 'U',     # Ribothymidine
    'DHU': 'U',    # Dihydrouridine
    'UMS': 'U',    # 5-methoxycarbonylmethyluridine
    'U2L': 'U',    # Modified uridine
    'U36': 'U',    # Modified uridine
    'Y5P': 'U',    # Modified uridine
    'P5P': 'U',    # Modified uridine
    'UFT': 'U',    # Modified uridine
    'F2T': 'U',    # Modified uridine
    '0U': 'U',     # Modified uridine
    '8XU': 'U',    # Modified uridine
    'ZBU': 'U',    # Modified uridine
    'ZTH': 'U',    # Modified uridine
    'ZHP': 'U',    # Modified uridine
    'SSU': 'U',    # Modified uridine

    # === GUANOSINE MODIFICATIONS ===
    'M2G': 'G',    # N2-methylguanosine
    'M7G': 'G',    # 7-methylguanosine (cap structure)
    'OMG': 'G',    # O-methylguanosine
    '1MG': 'G',    # 1-methylguanosine
    '2MG': 'G',    # 2'-O-methylguanosine
    'YYG': 'G',    # Modified guanosine
    'QUO': 'G',    # Queuosine
    'G7M': 'G',    # 7-methylguanosine
    'GTP': 'G',    # Guanosine triphosphate
    'GDP': 'G',    # Guanosine diphosphate
    'GMP': 'G',    # Guanosine monophosphate
    'G2L': 'G',    # Modified guanosine
    'G48': 'G',    # Modified guanosine
    '6OO': 'G',    # Modified guanosine
    '0G': 'G',     # Modified guanosine
    '8XG': 'G',    # Modified guanosine
    'ZGU': 'G',    # Modified guanosine
    'LCG': 'G',    # Modified guanosine

    # === CYTIDINE MODIFICATIONS ===
    '5MC': 'C',    # 5-methylcytidine
    'OMC': 'C',    # O-methylcytidine
    '2MC': 'C',    # 2'-O-methylcytidine
    'M5C': 'C',    # 5-methylcytidine
    'CBV': 'C',    # Carbovir cytidine
    'C2L': 'C',    # Modified cytidine
    'C43': 'C',    # Modified cytidine
    '6NW': 'C',    # Modified cytidine
    '0C': 'C',     # Modified cytidine
    '8XC': 'C',    # Modified cytidine
    'ZCY': 'C',    # Modified cytidine
    'ZBC': 'C',    # Modified cytidine

    # === RARE/SYNTHETIC MODIFICATIONS ===
    'ADP': 'A',    # Adenosine diphosphate
    'ATP': 'A',    # Adenosine triphosphate
    'AMP': 'A',    # Adenosine monophosphate
    'UDP': 'U',    # Uridine diphosphate
    'UTP': 'U',    # Uridine triphosphate
    'UMP': 'U',    # Uridine monophosphate
    'CDP': 'C',    # Cytidine diphosphate
    'CTP': 'C',    # Cytidine triphosphate
    'CMP': 'C',    # Cytidine monophosphate

    # === WYOSINE DERIVATIVES ===
    'YW1': 'G',    # Wybutosine
    'YW2': 'G',    # Wybutosine derivative
    'YW3': 'G',    # Wybutosine derivative

    # === HYPERMODIFIED BASES ===
    'Q': 'G',      # Queuosine
    'X': 'G',      # Xanthosine
    'D': 'U',      # Dihydrouridine
    'P': 'U',      # Pseudouridine

    # === METHYLATION VARIANTS ===
    'M1A': 'A',    # 1-methyladenosine
    'M1G': 'G',    # 1-methylguanosine
    'M3C': 'C',    # 3-methylcytidine
    'M5U': 'U',    # 5-methyluridine
    'M6A': 'A',    # N6-methyladenosine

    # === THIO MODIFICATIONS ===
    'S2C': 'C',    # 2-thiocytidine
    'S2U': 'U',    # 2-thiouridine
    'S4U': 'U',    # 4-thiouridine

    # === CAP STRUCTURES ===
    '7MG': 'G',    # 7-methylguanosine (5' cap)
    'M7G': 'G',    # 7-methylguanosine
    'G7M': 'G',    # 7-methylguanosine
}
# Complete final code for processing new CIF files with comprehensive support
from Bio.PDB import MMCIFParser
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

# Original function (for standard nucleotides)
def extract_rna_data_from_cif(cif_file_path):
    """Extract unique RNA sequences and C1' coordinates from a CIF file"""
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()

        sequences_data = []
        coordinates_data = []
        seen_sequences = set()  # Track unique sequences

        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"

                # Check if chain contains RNA residues
                rna_residues = []
                for residue in chain:
                    if residue.get_resname() in ['A', 'U', 'G', 'C']:  # RNA nucleotides
                        rna_residues.append(residue)

                if rna_residues:  # Only process if RNA residues found
                    # Build sequence
                    sequence = ''.join([res.get_resname() for res in rna_residues])

                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })

                        # Extract C1' coordinates for this unique sequence
                        for i, residue in enumerate(rna_residues, 1):
                            if "C1'" in residue:
                                atom = residue["C1'"]
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': residue.get_resname(),
                                    'resid': i,
                                    'x_1': atom.coord[0],
                                    'y_1': atom.coord[1],
                                    'z_1': atom.coord[2]
                                })

        return sequences_data, coordinates_data

    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []

# Disorder-aware glycosidic carbon detection
def get_glycosidic_carbon_disorder_aware(residue):
    """
    Find C1' or C1{suffix} atoms, handling DisorderedAtom objects
    """
    # Get all available atom names
    available_atoms = [atom.get_name() for atom in residue]

    # Look for C1' first (most common)
    if "C1'" in available_atoms:
        atom = residue["C1'"]
        # Handle DisorderedAtom by getting the first conformation
        if hasattr(atom, 'selected_child'):
            return atom.selected_child
        return atom

    # Look for any C1{suffix} pattern
    c1_variants = [atom_name for atom_name in available_atoms if atom_name.startswith('C1') and len(atom_name) > 2]

    if c1_variants:
        # If multiple C1 variants, prefer the shortest one
        best_variant = min(c1_variants, key=len)
        atom = residue[best_variant]
        # Handle DisorderedAtom by getting the first conformation
        if hasattr(atom, 'selected_child'):
            return atom.selected_child
        return atom

    return None

# Comprehensive extraction function with disorder handling
def extract_rna_data_from_cif_comprehensive_final(cif_file_path):
    """Extract RNA with comprehensive modified nucleotide recognition and disorder handling"""
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()

        sequences_data = []
        coordinates_data = []
        seen_sequences = set()

        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"

                # Check if chain contains RNA residues (including all modified ones)
                rna_residues = []
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in nucleotide_mapping:
                        rna_residues.append(residue)

                if rna_residues:  # Only process if RNA residues found
                    # Build sequence using standard nucleotides
                    sequence = ''.join([nucleotide_mapping[res.get_resname()] for res in rna_residues])

                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })

                        # Extract coordinates using disorder-aware detection
                        for i, residue in enumerate(rna_residues, 1):
                            carbon_atom = get_glycosidic_carbon_disorder_aware(residue)

                            if carbon_atom is not None:  # Use 'is not None' to avoid DisorderedAtom issues
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': nucleotide_mapping[residue.get_resname()],  # Use standard name
                                    'resid': i,
                                    'x_1': carbon_atom.coord[0],
                                    'y_1': carbon_atom.coord[1],
                                    'z_1': carbon_atom.coord[2]
                                })

        return sequences_data, coordinates_data

    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []

# Smart extraction function
def extract_rna_data_smart_final(cif_file_path):
    """
    Final smart extraction: standard nucleotides first, then comprehensive with disorder handling
    """
    # First try the original function (standard nucleotides only)
    sequences_std, coordinates_std = extract_rna_data_from_cif(cif_file_path)

    # If we found RNA data with standard function, use it
    if sequences_std:
        return sequences_std, coordinates_std, "standard"

    # If no standard RNA found, try the comprehensive function with disorder handling
    sequences_mod, coordinates_mod = extract_rna_data_from_cif_comprehensive_final(cif_file_path)

    if sequences_mod:
        return sequences_mod, coordinates_mod, "modified"
    else:
        return [], [], "none"

# Main processing function
def process_new_cif_files_final(train_seqs, train_labels, cif_dir, new_cif_files):
    """Final processing function with comprehensive nucleotide support and disorder handling"""

    if not new_cif_files:
        print("No new files to process - all CIF files have already been processed!")
        return train_seqs, train_labels

    print(f"Processing {len(new_cif_files)} new CIF files with final comprehensive extraction...")
    print(f"Nucleotide mapping includes {len(nucleotide_mapping)} variants")
    print(f"Includes disorder handling for DisorderedAtom objects")

    new_sequences = []
    new_coordinates = []
    processing_stats = {"standard": 0, "modified": 0, "none": 0}

    for cif_file in tqdm(new_cif_files):
        cif_path = os.path.join(cif_dir, cif_file)
        sequences, coordinates, extraction_type = extract_rna_data_smart_final(cif_path)

        processing_stats[extraction_type] += 1

        if sequences:  # Only add if we found RNA data
            new_sequences.extend(sequences)
            new_coordinates.extend(coordinates)
            print(f"✅ {cif_file} ({extraction_type}): {len(sequences)} sequences, {len(coordinates)} coordinates")
        else:
            print(f"❌ {cif_file}: No RNA data found")

    print(f"\nFINAL PROCESSING SUMMARY:")
    print(f"Files with standard nucleotides: {processing_stats['standard']}")
    print(f"Files with modified nucleotides: {processing_stats['modified']}")
    print(f"Files with no RNA data: {processing_stats['none']}")
    print(f"Success rate: {processing_stats['standard'] + processing_stats['modified']} / {len(new_cif_files)} = {((processing_stats['standard'] + processing_stats['modified']) / len(new_cif_files) * 100):.1f}%")

    if new_sequences:
        # Create DataFrames for new data
        new_sequences_df = pd.DataFrame(new_sequences)
        new_coordinates_df = pd.DataFrame(new_coordinates)

        print(f"\nFINAL NEW DATA SUMMARY:")
        print(f"Total new sequences: {len(new_sequences)}")
        print(f"Total new coordinates: {len(new_coordinates)}")

        # Add to existing dataframes
        train_seqs_updated = pd.concat([train_seqs, new_sequences_df], ignore_index=True)
        train_labels_updated = pd.concat([train_labels, new_coordinates_df], ignore_index=True)

        print(f"\nFINAL UPDATED DATAFRAMES:")
        print(f"train_seqs: {train_seqs.shape} -> {train_seqs_updated.shape}")
        print(f"train_labels: {train_labels.shape} -> {train_labels_updated.shape}")


        print(f"\nFinal improvement summary:")
        print(f"Sequences added: {len(train_seqs_updated) - len(train_seqs)}")
        print(f"Coordinates added: {len(train_labels_updated) - len(train_labels)}")

        return train_seqs_updated, train_labels_updated

    else:
        print("No new RNA sequences found in any of the new files")
        return train_seqs, train_labels

# EXECUTE THE FINAL COMPREHENSIVE PROCESSING
print("="*90)
print("FINAL COMPREHENSIVE PROCESSING WITH DISORDER HANDLING AND 93 NUCLEOTIDE VARIANTS")
print("="*90)

# Process the new files
train_seqs_final, train_labels_final = process_new_cif_files_final(
    train_seqs=train_seqs,
    train_labels=train_labels,
    cif_dir=cif_dir,
    new_cif_files=new_cif_files
)

print("\n" + "="*90)
print("FINAL COMPREHENSIVE PROCESSING COMPLETE")
print("="*90)
print(f"Final train_seqs shape: {train_seqs_final.shape}")
print(f"Final train_labels shape: {train_labels_final.shape}")

# Show which files were successfully processed
successful_files = []
failed_files = []

for cif_file in new_cif_files:
    cif_path = os.path.join(cif_dir, cif_file)
    sequences, coordinates, extraction_type = extract_rna_data_smart_final(cif_path)
    if sequences:
        successful_files.append(cif_file)
    else:
        failed_files.append(cif_file)

print(f"\nSuccessfully processed files ({len(successful_files)}):")
for f in successful_files:
    print(f"  ✅ {f}")

if failed_files:
    print(f"\nFiles with no RNA data ({len(failed_files)}):")
    for f in failed_files:
        print(f"  ❌ {f}")

print(f"\nFinal statistics:")
print(f"Original dataset: {len(train_seqs)} sequences, {len(train_labels)} coordinates")
print(f"Final dataset: {len(train_seqs_final)} sequences, {len(train_labels_final)} coordinates")
==========================================================================================
FINAL COMPREHENSIVE PROCESSING WITH DISORDER HANDLING AND 93 NUCLEOTIDE VARIANTS
==========================================================================================
Processing 38 new CIF files with final comprehensive extraction...
Nucleotide mapping includes 93 variants
Includes disorder handling for DisorderedAtom objects
  3%|▎         | 1/38 [00:00<00:07,  5.27it/s]
✅ 2gq6.cif (modified): 2 sequences, 16 coordinates
  5%|▌         | 2/38 [00:01<00:22,  1.57it/s]
✅ 9hbw.cif (modified): 3 sequences, 46 coordinates
✅ 2x2q.cif (modified): 1 sequences, 5 coordinates
 11%|█         | 4/38 [00:01<00:11,  2.89it/s]
❌ 4rcm.cif: No RNA data found
 16%|█▌        | 6/38 [00:12<01:18,  2.44s/it]
✅ 8wak.cif (modified): 1 sequences, 2 coordinates
✅ 2gq7.cif (modified): 2 sequences, 16 coordinates
✅ 1pbl.cif (modified): 1 sequences, 6 coordinates
 21%|██        | 8/38 [00:12<00:45,  1.53s/it]
✅ 2kwg.cif (modified): 2 sequences, 23 coordinates
✅ 471d.cif (modified): 1 sequences, 12 coordinates
 29%|██▉       | 11/38 [00:13<00:22,  1.19it/s]
✅ 4r8i.cif (modified): 1 sequences, 33 coordinates
✅ 8f24.cif (modified): 3 sequences, 11 coordinates
 32%|███▏      | 12/38 [00:14<00:22,  1.16it/s]
✅ 1s9l.cif (modified): 1 sequences, 3 coordinates
 34%|███▍      | 13/38 [00:15<00:18,  1.38it/s]
✅ 2gq5.cif (modified): 2 sequences, 16 coordinates
 37%|███▋      | 14/38 [00:15<00:15,  1.60it/s]
✅ 3t3n.cif (modified): 1 sequences, 6 coordinates
 39%|███▉      | 15/38 [00:16<00:18,  1.23it/s]
✅ 9hbz.cif (modified): 2 sequences, 22 coordinates
 42%|████▏     | 16/38 [00:17<00:15,  1.43it/s]
✅ 3af6.cif (modified): 2 sequences, 5 coordinates
✅ 2xc6.cif (modified): 1 sequences, 8 coordinates
✅ 2jja.cif (modified): 1 sequences, 8 coordinates
 50%|█████     | 19/38 [00:17<00:07,  2.63it/s]
✅ 1i7j.cif (modified): 1 sequences, 6 coordinates
✅ 469d.cif (modified): 1 sequences, 12 coordinates
 55%|█████▌    | 21/38 [00:18<00:08,  2.11it/s]
✅ 3iem.cif (modified): 4 sequences, 11 coordinates
 58%|█████▊    | 22/38 [00:19<00:07,  2.15it/s]
✅ 9hbv.cif (modified): 2 sequences, 30 coordinates
 61%|██████    | 23/38 [00:19<00:07,  2.09it/s]
✅ 5swm.cif (modified): 1 sequences, 3 coordinates
 66%|██████▌   | 25/38 [00:20<00:05,  2.25it/s]
✅ 5js2.cif (modified): 1 sequences, 6 coordinates
✅ 2g32.cif (modified): 2 sequences, 16 coordinates
 68%|██████▊   | 26/38 [00:20<00:04,  2.62it/s]
✅ 3p4a.cif (modified): 1 sequences, 2 coordinates
 71%|███████   | 27/38 [00:21<00:05,  2.01it/s]
✅ 2m39.cif (modified): 1 sequences, 5 coordinates
✅ 1r3o.cif (modified): 2 sequences, 16 coordinates
 76%|███████▋  | 29/38 [00:22<00:04,  2.24it/s]
✅ 2n4j.cif (modified): 1 sequences, 8 coordinates
✅ 468d.cif (modified): 1 sequences, 12 coordinates
 82%|████████▏ | 31/38 [00:22<00:02,  2.80it/s]
✅ 2gpm.cif (modified): 2 sequences, 16 coordinates
✅ 310d.cif (modified): 1 sequences, 6 coordinates
 87%|████████▋ | 33/38 [00:24<00:02,  2.18it/s]
✅ 1xv6.cif (modified): 1 sequences, 12 coordinates
✅ 2wna.cif (modified): 1 sequences, 6 coordinates
 95%|█████████▍| 36/38 [00:24<00:00,  3.18it/s]
❌ 1qd7.cif: No RNA data found
✅ 2gq4.cif (modified): 2 sequences, 16 coordinates
100%|██████████| 38/38 [00:25<00:00,  1.52it/s]
✅ 1eqq.cif (modified): 2 sequences, 4 coordinates
✅ 470d.cif (modified): 1 sequences, 12 coordinates

FINAL PROCESSING SUMMARY:
Files with standard nucleotides: 0
Files with modified nucleotides: 36
Files with no RNA data: 2
Success rate: 36 / 38 = 94.7%

FINAL NEW DATA SUMMARY:
Total new sequences: 55
Total new coordinates: 437
FINAL UPDATED DATAFRAMES:
train_seqs: (19393, 2) -> (19448, 2)
train_labels: (10283286, 6) -> (10283723, 6)

Final improvement summary:
Sequences added: 55
Coordinates added: 437

==========================================================================================
FINAL COMPREHENSIVE PROCESSING COMPLETE
==========================================================================================
Final train_seqs shape: (19448, 2)
Final train_labels shape: (10283723, 6)

Successfully processed files (36):
  ✅ 2gq6.cif
  ✅ 9hbw.cif
  ✅ 2x2q.cif
  ✅ 8wak.cif
  ✅ 2gq7.cif
  ✅ 1pbl.cif
  ✅ 2kwg.cif
  ✅ 471d.cif
  ✅ 4r8i.cif
  ✅ 8f24.cif
  ✅ 1s9l.cif
  ✅ 2gq5.cif
  ✅ 3t3n.cif
  ✅ 9hbz.cif
  ✅ 3af6.cif
  ✅ 2xc6.cif
  ✅ 2jja.cif
  ✅ 1i7j.cif
  ✅ 469d.cif
  ✅ 3iem.cif
  ✅ 9hbv.cif
  ✅ 5swm.cif
  ✅ 5js2.cif
  ✅ 2g32.cif
  ✅ 3p4a.cif
  ✅ 2m39.cif
  ✅ 1r3o.cif
  ✅ 2n4j.cif
  ✅ 468d.cif
  ✅ 2gpm.cif
  ✅ 310d.cif
  ✅ 1xv6.cif
  ✅ 2wna.cif
  ✅ 2gq4.cif
  ✅ 1eqq.cif
  ✅ 470d.cif

Files with no RNA data (2):
  ❌ 4rcm.cif
  ❌ 1qd7.cif

Final statistics:
Original dataset: 19393 sequences, 10283286 coordinates
Final dataset: 19448 sequences, 10283723 coordinates
# Success rate: 29 / 38 files contained RNA
# Set up directories
predictions_dir = "/kaggle/working/predictions"
os.makedirs(predictions_dir, exist_ok=True)
fasta_dir = "/kaggle/working/fasta_files"
os.makedirs(fasta_dir, exist_ok=True)

# Set time limit for DRfold2 (in seconds)
DRFOLD_TIME_LIMIT = 7 * 60 * 60  # 7 hours
start_time_global = time.time()
!cp -r /kaggle/input/drfold2-repo/DRfold2 /kaggle/working/
%cd DRfold2
%cd Arena
!make Arena
%cd ..
/kaggle/working/DRfold2
/kaggle/working/DRfold2/Arena
clang++ -O3 Arena.cpp -o Arena
/kaggle/working/DRfold2
!cp -r /kaggle/input/drfold2/model_hub /kaggle/working/DRfold2/
%%writefile /kaggle/working/DRfold2/DRfold_infer.py
import os, sys
import torch
import numpy as np
from subprocess import Popen, PIPE, STDOUT

# Get the directory where the script is located
exp_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
# dlexps = ['cfg_95','cfg_96','cfg_97','cfg_99']
dlexps = ['cfg_97']

print(f"[DRfold2] Starting prediction pipeline on {device} device")

# Get input FASTA file and output directory from command line arguments
fastafile = os.path.realpath(sys.argv[1])
outdir = os.path.realpath(sys.argv[2])

print(f"[DRfold2] Input: {fastafile}")
print(f"[DRfold2] Output: {outdir}")

# Initialize clustering flag and AF3 file path
pclu = False
af3file = None

# Parse command line arguments
# Acceptable formats:
# python DRfold_infer.py input.fasta output_dir
# python DRfold_infer.py input.fasta output_dir 1
# python DRfold_infer.py input.fasta output_dir --af3 af3_model.pdb
# python DRfold_infer.py input.fasta output_dir 1 --af3 af3_model.pdb

for i in range(3, len(sys.argv)):
    if sys.argv[i] == "1" and i == 3:
        pclu = True
        print('[DRfold2] Clustering enabled - will generate multiple models')
    elif sys.argv[i] == "--af3" and i+1 < len(sys.argv):
        af3file = os.path.realpath(sys.argv[i+1])
        print(f'[DRfold2] Using AlphaFold3 structure: {af3file}')

if not pclu:
    print('[DRfold2] Clustering disabled - will generate single model')

# Create output directory if it doesn't exist
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    print(f"[DRfold2] Created output directory: {outdir}")

# Create subdirectories for different outputs
ret_dir = os.path.join(outdir,'rets_dir')  # For return files
if not os.path.isdir(ret_dir):
    os.makedirs(ret_dir)
    print(f"[DRfold2] Created returns directory: {ret_dir}")

folddir = os.path.join(outdir,'folds')     # For folded structures
if not os.path.isdir(folddir):
    os.makedirs(folddir)
    print(f"[DRfold2] Created folds directory: {folddir}")

refdir = os.path.join(outdir,'relax')      # For relaxed structures
if not os.path.isdir(refdir):
    os.makedirs(refdir)
    print(f"[DRfold2] Created relaxation directory: {refdir}")


# Helper function to run commands and capture output
def run_cmd(cmd, description):
    print(f"[DRfold2] {description}")
    print(f"[DRfold2] Command: {cmd}")

    # Execute the command and capture output in real-time
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)

    # Print output line by line as it becomes available
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(f"[DRfold2 subprocess] {line}")

    # Get return code
    return_code = process.wait()
    if return_code == 0:
        print(f"[DRfold2] {description} completed successfully")
    else:
        print(f"[DRfold2] {description} failed with return code {return_code}")
    return return_code


# Create paths for model directories and test scripts
dlmains = [os.path.join(exp_dir, one_exp, 'test_modeldir.py') for one_exp in dlexps]
dirs = [os.path.join(exp_dir, 'model_hub', one_exp) for one_exp in dlexps]


# Check if processing has been done before
if not os.path.isfile(ret_dir + '/done'):
    print("[DRfold2] Step 1/4: GENERATING INITIAL PREDICTIONS")
    print(f"[DRfold2] No previous predictions found, will generate e2e and geo files")

    # Run each model configuration
    for idx, (dlmain, one_exp, mdir) in enumerate(zip(dlmains, dlexps, dirs)):
        # Construct command to run the model
        cmd = f'python {dlmain} {device} {fastafile} {ret_dir}/{one_exp}_ {mdir}'
        description = f"Running model {idx+1}/{len(dlexps)}: {one_exp}"
        run_cmd(cmd, description)

    # Mark processing as complete
    wfile = open(ret_dir+'/done','w')
    wfile.write('1')
    wfile.close()
    print("[DRfold2] Initial predictions generation completed")
else:
    print("[DRfold2] Step 1/4: USING EXISTING PREDICTIONS")
    print(f"[DRfold2] Found previous predictions in {ret_dir}, using existing e2e and geo files")

# Helper function to get model PDB file
def get_model_pdb(tdir,opt):
    files = os.listdir(tdir)
    files = [afile for afile in files if afile.startswith(opt)][0]
    return files


# Set up directory paths and configuration files
cso_dir = folddir                                                    # Directory for coarse-grained structures
clufile = os.path.join(folddir,'clu.txt')                            # Clustering results file
config_sel = os.path.join(exp_dir,'cfg_for_selection.json')          # Selection configuration
foldconfig = os.path.join(exp_dir,'cfg_for_folding.json')            # Folding configuration
selpython = os.path.join(exp_dir,'PotentialFold','Selection.py')     # Selection script
optpython = os.path.join(exp_dir,'PotentialFold','Optimization.py')  # Optimization script
clupy = os.path.join(exp_dir,'PotentialFold','Clust.py')             # Clustering script
arena = os.path.join(exp_dir,'Arena','Arena')                        # Arena executable for structure refinement


# Set up initial save prefixes for optimization and selection
optsaveprefix = os.path.join(cso_dir, f'opt_0')
save_prefix = os.path.join(cso_dir, f'sel_0')

# Get all .ret files from the return directory
rets = os.listdir(ret_dir)
rets = [afile for afile in rets if afile.endswith('.ret')]
rets = [os.path.join(ret_dir,aret) for aret in rets ]
ret_str = ' '.join(rets)

print("[DRfold2] Step 2/4: SELECTION PROCESS")
print(f"[DRfold2] Found {len(rets)} return files for selection")
print(f"[DRfold2] Using selection config: {config_sel}")
print(f"[DRfold2] Output prefix: {save_prefix}")


# Run selection process
cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
run_cmd(cmd, "Running selection process")

print("[DRfold2] Step 3/4: OPTIMIZATION PROCESS")
print(f"[DRfold2] Using fold config: {foldconfig}")
print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

# Run optimization process with optional AF3 file
cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
if af3file and os.path.exists(af3file):
    cmd += f' {af3file}'
run_cmd(cmd, "Running optimization process")

# Get the coarse-grained PDB and save refined structure
cgpdb = os.path.join(folddir,get_model_pdb(folddir,'opt_0'))
savepdb = os.path.join(refdir,'model_1.pdb')

print("[DRfold2] Step 4/4: STRUCTURE REFINEMENT")
print(f"[DRfold2] Found optimized structure: {cgpdb}")
print(f"[DRfold2] Final output will be saved to: {savepdb}")

cmd = f'{arena} {cgpdb} {savepdb} 7'
run_cmd(cmd, "Running structure refinement")

# If clustering is enabled (pclu=True)
if pclu:
    print("[DRfold2] ADDITIONAL STEP: CLUSTERING")
    print(f"[DRfold2] Running clustering process, output: {clufile}")

    # Run clustering process
    cmd = f'python {clupy} {ret_dir} {clufile}'
    run_cmd(cmd, "Running clustering")

    # Read clustering results
    lines = open(clufile).readlines()
    lines = [aline.strip() for aline in lines]
    lines = [aline for aline in lines if aline]

    cluster_count = len(lines) - 1
    print(f"[DRfold2] Found {cluster_count} additional clusters to process")

    # Process each cluster
    for i in range(1,len(lines)):
        print(f"[DRfold2] PROCESSING CLUSTER {i}/{cluster_count}")

        # Get return files for this cluster
        rets = lines[i].split()
        rets = [os.path.join(ret_dir,aret.replace('.pdb','.ret')) for aret in rets ]
        ret_str = ' '.join(rets)

        # Set up save prefixes for this cluster
        optsaveprefix =  os.path.join(cso_dir,f'opt_{str(i+1)}')
        save_prefix = os.path.join(cso_dir,f'sel_{str(i+1)}')

        print(f"[DRfold2] Cluster {i} Selection Process")
        print(f"[DRfold2] Found {len(rets)} return files for selection")
        print(f"[DRfold2] Selection output prefix: {save_prefix}")

        # Run selection process for this cluster
        cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
        run_cmd(cmd, f"Running selection for cluster {i}")

        print(f"[DRfold2] Cluster {i} Optimization Process")
        print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

        # Run optimization process for this cluster with optional AF3 file
        cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
        if af3file and os.path.exists(af3file):
            cmd += f' {af3file}'
        run_cmd(cmd, f"Running optimization for cluster {i}")

        # Get the coarse-grained PDB and save refined structure for this cluster
        cgpdb = os.path.join(folddir,get_model_pdb(folddir,f'opt_{str(i+1)}'))
        savepdb = os.path.join(refdir,f'model_{str(i+1)}.pdb')

        print(f"[DRfold2] Cluster {i} Refinement Process")
        print(f"[DRfold2] Found optimized structure: {cgpdb}")
        print(f"[DRfold2] Final output will be saved to: {savepdb}")

        cmd = f'{arena} {cgpdb} {savepdb} 7'
        run_cmd(cmd, f"Running refinement for cluster {i}")

print("[DRfold2] PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
Overwriting /kaggle/working/DRfold2/DRfold_infer.py
%%writefile /kaggle/working/DRfold2/PotentialFold/operations.py
"""
operations.py: Core Mathematical Operations for RNA Structure Analysis

This module provides essential mathematical operations for manipulating and analyzing
RNA 3D structures, organized into four main categories:

1. Basic Vector Operations:
   Functions for selecting coordinates and calculating distances between points,
   which form the foundation for all structural calculations.

2. Angle Calculations:
   Functions for computing bond angles and dihedral (torsion) angles between atoms,
   with differentiable implementations suitable for gradient-based optimization.

3. Rigid Body Transformations:
   Functions for determining optimal rotations and translations between sets of
   coordinates, enabling structure alignment and manipulation.

4. Sequence Utilities:
   Functions for converting RNA sequence data into standard 3D coordinate templates,
   allowing sequence-structure mapping.

These operations support the core functionality of RNA structure prediction, analysis,
and optimization throughout the codebase.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import math, sys, math
from io import BytesIO
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from subprocess import Popen, PIPE, STDOUT

# Use consistent epsilon value across all functions
EPS = 1e-8


# === Basic Vector Operations ===
def coor_selection(coor,mask):
    #[L,n,3],[L,n],byte
    return torch.masked_select(coor,mask.bool()).view(-1,3)

def pair_distance(x1, x2, eps=1e-6, p=2):
    # Use torch.cdist for p=2 (Euclidean) which is highly optimized
    if p == 2:
        return torch.cdist(x1, x2, p=2)

    # For other p-norms, avoid memory expansion with broadcasting
    x1_ = x1.unsqueeze(1)  # [n1, 1, dim]
    x2_ = x2.unsqueeze(0)  # [1, n2, dim]
    diff = torch.abs(x1_ - x2_)
    out = torch.pow(diff + eps, p).sum(dim=2)
    return torch.pow(out, 1. / p)


# === Angle Calculations ===
def angle(p0, p1, p2):
    # [b 3]
    b0 = p0-p1
    b1 = p2-p1

    b0 = b0 / (torch.norm(b0, dim =-1, keepdim=True) + EPS)
    b1 = b1 / (torch.norm(b1, dim =-1, keepdim=True) + EPS)

    recos = torch.sum(b0*b1, -1)
    recos = torch.clamp(recos, -0.9999, 0.9999)
    return torch.acos(recos)

class torsion(Function):
    #PyTorch class to calculate differentiable torsion angle
    #https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    #https://salilab.org/modeller/manual/node492.html
    @staticmethod
    def forward(ctx, p0, p1, p2, p3):
        # Save input points for backward pass
        ctx.save_for_backward(p0, p1, p2, p3)

        # Calculate bond vectors
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize the middle bond vector
        b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
        b1_unit = b1 / b1_norm

        # Project the other bonds onto the plane perpendicular to middle bond
        v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
        w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

        # Calculate torsion using the arctan2 formula (more stable than arccos)
        x = torch.sum(v * w, dim=-1)                                # cosine component
        y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)  # sine component

        return torch.atan2(y, x)


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from forward pass
        p0, p1, p2, p3 = ctx.saved_tensors

        # Calculate bond vectors
        r01 = p0 - p1
        r12 = p2 - p1
        r23 = p3 - p2

        # Calculate bond lengths with numerical stability
        d01 = torch.norm(r01, dim=-1, keepdim=True) + 1e-8
        d12 = torch.norm(r12, dim=-1, keepdim=True) + 1e-8
        d23 = torch.norm(r23, dim=-1, keepdim=True) + 1e-8

        # Normalize bond vectors
        e01 = r01 / d01
        e12 = r12 / d12
        e23 = r23 / d23

        # Calculate normal vectors to the two planes
        n1 = torch.cross(e01, e12, dim=-1)
        n2 = torch.cross(e12, e23, dim=-1)

        # Normalize normal vectors
        n1_norm = torch.norm(n1, dim=-1, keepdim=True) + 1e-8
        n2_norm = torch.norm(n2, dim=-1, keepdim=True) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate gradients for each atom
        # These are based on the analytical derivatives of dihedral angles
        g0 = torch.cross(e01, n1, dim=-1) / d01
        g1 = -g0 - torch.cross(e12, n1, dim=-1) / d12
        g2 = torch.cross(e12, n2, dim=-1) / d12 - torch.cross(e23, n2, dim=-1) / d23
        g3 = torch.cross(e23, n2, dim=-1) / d23

        # Apply chain rule with incoming gradient
        g0 = g0 * grad_output.unsqueeze(-1)
        g1 = g1 * grad_output.unsqueeze(-1)
        g2 = g2 * grad_output.unsqueeze(-1)
        g3 = g3 * grad_output.unsqueeze(-1)

        return g0, g1, g2, g3


def dihedral(input1, input2, input3, input4):
    return torsion.apply(input1, input2, input3, input4)



# === Rigid Body Transformations ===
def rigidFrom3Points(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    v1 = x3 - x2
    v2 = x1 - x2

    # Normalize v1 to get e1
    e1 = F.normalize(v1, p=2, dim=-1)

    # Project v2 onto e1 and subtract to get the component orthogonal to e1
    u2 = v2 - e1 * (torch.einsum('bn,bn->b', e1, v2)[:, None])

    # Normalize u2 to get e2
    e2 = F.normalize(u2, p=2, dim=-1)

    # Cross product to get e3
    e3 = torch.cross(e1, e2, dim=-1)

    return torch.stack([e1, e2, e3], dim=1)


# return the direction from to_q to from_p
def Kabsch_rigid(bases,x1,x2,x3):
    # Early return for empty input
    if x1.shape[0] == 0:
        return torch.empty(0, 3, 3), torch.empty(0, 3)

    the_dim=1
    to_q = torch.stack([x1,x2,x3],dim=the_dim)
    biasq=torch.mean(to_q,dim=the_dim,keepdim=True)
    q=to_q-biasq
    m = torch.einsum('bnz,bny->bzy',bases,q)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r,biasq.squeeze()



# === Sequence Utilities ===
def Get_base(seq,basenpy_standard):
    base_num = basenpy_standard.shape[1]
    basenpy = np.zeros([len(seq),base_num,3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy=='A']=basenpy_standard[0]
    basenpy[seqnpy=='a']=basenpy_standard[0]

    basenpy[seqnpy=='G']=basenpy_standard[1]
    basenpy[seqnpy=='g']=basenpy_standard[1]

    basenpy[seqnpy=='C']=basenpy_standard[2]
    basenpy[seqnpy=='c']=basenpy_standard[2]

    basenpy[seqnpy=='U']=basenpy_standard[3]
    basenpy[seqnpy=='u']=basenpy_standard[3]

    basenpy[seqnpy=='T']=basenpy_standard[3]
    basenpy[seqnpy=='t']=basenpy_standard[3]

    return torch.from_numpy(basenpy).double()
Overwriting /kaggle/working/DRfold2/PotentialFold/operations.py
%%writefile /kaggle/working/DRfold2/PotentialFold/Optimization.py
#! /nfs/amino-home/liyangum/miniconda3/bin/python
import torch
import random
import numpy as np
import os, json, sys

import Cubic, Potential
import operations
import a2b, rigid
import torch.optim as opt
from scipy.optimize import minimize
import pickle

torch.manual_seed(6)
np.random.seed(9)
random.seed(9)


Scale_factor = 1.0
USEGEO = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config


class Structure:
    def __init__(self,fastafile,geofiles,saveprefix,initial_ret,foldconfig,af3file=None):
        self.config=readconfig(foldconfig)
        self.seqfile=fastafile
        self.init_ret = initial_ret
        self.foldconfig = foldconfig
        self.geofiles = geofiles
        self.rets = [pickle.load(open(refile,'rb')) for refile  in geofiles]
        self.txs=[]
        for ret in self.rets:
            self.txs.append( torch.from_numpy(ret['coor']).double().to(device))
        self.handle_geo()
        self.pair = []
        for ret in self.rets:
            self.pair.append(torch.from_numpy(ret['plddt']).double().to(device))
        self.saveprefix=saveprefix
        self.seq=open(fastafile).readlines()[1].strip()
        self.L=len(self.seq)
        basenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','base.npy'))
        self.basex = operations.Get_base(self.seq,basenpy).to(device)
        othernpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','other2.npy'))
        self.otherx = operations.Get_base(self.seq,othernpy).to(device)
        sidenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','side.npy'))
        self.sidex = operations.Get_base(self.seq,sidenpy).to(device)

        self.init_mask()
        self.init_paras()
        self._init_fape()
        self.tx2ds = [td.to(device) for td in self.tx2ds]
        self.local_weight = torch.ones(self.L,self.L).to(device)

        for i in range(self.L):
            for j in range(i+1,min(self.L,i+2)):
                self.local_weight[i,j] = self.local_weight[j,i] = 4
            for j in range(i+2,min(self.L,i+3)):
                self.local_weight[i,j] = self.local_weight[j,i] = 3
            for j in range(i+3,min(self.L,i+4)):
                self.local_weight[i,j] = self.local_weight[j,i] = 2

        # Load AlphaFold3 prediction if provided
        if af3file and os.path.exists(af3file):
            self.af3_coords = self.load_af3_structure(af3file)
            print(f"[DRfold2] Loaded AlphaFold3 structure from {af3file}")
            self._init_af3()
        elif self.config.get('weight_af3', 0) > 0:
            print(f"[DRfold2] Warning: AlphaFold3 weight is set but no AF3 file provided")

    def load_af3_structure(self, pdbfile):
        """Load AlphaFold3 predicted structure from PDB file"""
        print("[DRfold2] Loading AlphaFold3 prediction from {}".format(pdbfile))
        af3_coords = torch.zeros((self.L, 3, 3), device=device, dtype=torch.double)  # L residues, 3 atoms (P, C4', N1/N9), 3 coordinates

        # Read PDB file and extract P, C4', N1/N9 coordinates
        atom_types = [' P  ', " C4'", ' N1 ']
        purine_bases = ['A', 'G', 'a', 'g']

        with open(pdbfile, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                atom_type = line[12:16]
                res_id = int(line[22:26]) - 1  # PDB is 1-indexed
                if res_id < 0 or res_id >= self.L:
                    continue

                # Handle purines (A, G) which have N9 instead of N1
                if atom_type == ' N1 ' and self.seq[res_id].upper() in ['A', 'G']:
                    continue
                if atom_type == ' N9 ' and self.seq[res_id].upper() in ['A', 'G']:
                    atom_idx = 2  # Index for the third atom (N1/N9)
                elif atom_type in atom_types:
                    atom_idx = atom_types.index(atom_type)
                else:
                    continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                af3_coords[res_id, atom_idx] = torch.tensor([x, y, z], device=device, dtype=torch.double)

        # Check if structure is complete
        if torch.any(torch.sum(af3_coords, dim=-1) == 0):
            print("[DRfold2] Warning: AlphaFold3 structure is incomplete")

        return af3_coords

    def _init_af3(self):
        """Initialize AlphaFold3 aligned coordinates for FAPE calculation"""
        if not hasattr(self, 'af3_coords'):
            return

        self.af3_rot, self.af3_trans = operations.Kabsch_rigid(
            self.basex,
            self.af3_coords[:, 0],  # P atoms
            self.af3_coords[:, 1],  # C4' atoms
            self.af3_coords[:, 2]   # N1/N9 atoms
        )

        # Create aligned coordinates for fast energy calculation
        # Shape: [L, L, 3, 3]
        self.af3_aligned = self.af3_coords.unsqueeze(0).repeat(self.L, 1, 1, 1)

        # Translate by af3_trans
        self.af3_aligned = self.af3_aligned - self.af3_trans.unsqueeze(1).unsqueeze(1)

        # Rotate by af3_rot
        self.af3_aligned = torch.einsum('ijkl,ild->ijkd', self.af3_aligned, self.af3_rot.transpose(-1, -2))

        print(f"[DRfold2] AlphaFold3 alignment initialized (shape: {self.af3_aligned.shape})")

    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot,true_trans = operations.Kabsch_rigid(self.basex,tx[:,0],tx[:,1],tx[:,2])
            true_x2 = tx[:,None,:,:] - true_trans[None,:,None,:]
            true_x2 = torch.einsum('ijnd,jde->ijne',true_x2,true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)

    def handle_geo(self):
        oldkeys=['dist_p','dist_c','dist_n']
        newkeys=['pp','cc','nn']
        self.geos=[]
        for ret in self.rets:
            geo = {}
            for nk,ok in zip(newkeys,oldkeys):
                geo[nk] = torch.from_numpy(ret[ok].astype(np.float64)).to(device) + 0
            self.geos.append(geo)


    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=(torch.DoubleTensor(halfmask) > 0.5).to(device)
        self.fullmask=(torch.DoubleTensor(fullmask) > 0.5).to(device)
        self.clash_mask = torch.zeros([self.L,self.L,22,22], device=device)
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq).to(device)
        self.side_mask = (self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]).to(device)
        self.clash_mask = ((self.clash_mask > 0.5) * (self.side_mask > 0.5)).to(device)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = geo['cc'][:,:,-1] < 0.5
            confimask_pp = geo['pp'][:,:,-1] < 0.5
            confimask_nn = geo['nn'][:,:,-1] < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {'cc': [], 'pp': [], 'nn': []}
        self.cs_knots = {'cc': [], 'pp': [], 'nn': []}
        for geo in self.geos:
            cc_cs,cc_decs=Cubic.dis_cubic(geo['cc'],2,40,36)
            pp_cs,pp_decs=Cubic.dis_cubic(geo['pp'],2,40,36)
            nn_cs,nn_decs=Cubic.dis_cubic(geo['nn'],2,40,36)
            self.geo_cc.append([cc_cs,cc_decs])
            self.geo_pp.append([pp_cs,pp_decs])
            self.geo_nn.append([nn_cs,nn_decs])

            L = self.L
            cc_coefs_np  = np.stack([[cc_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            cc_knots_np  = np.stack([[cc_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['cc'].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots['cc'].append(torch.from_numpy(cc_knots_np).to(device))

            pp_coefs_np  = np.stack([[pp_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            pp_knots_np  = np.stack([[pp_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['pp'].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots['pp'].append(torch.from_numpy(pp_knots_np).to(device))

            nn_coefs_np  = np.stack([[nn_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            nn_knots_np  = np.stack([[nn_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['nn'].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots['nn'].append(torch.from_numpy(nn_knots_np).to(device))


    def compute_bb_clash(self,coor,other_coor):
        com_coor = torch.cat([coor,other_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 3.15) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],3.15)
        return vdw_dynamic.sum()*self.config['weight_vdw']

    def compute_full_clash(self,coor,other_coor,side_coor):
        com_coor = torch.cat([coor[:,:2],other_coor,side_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 2.5) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],2.5)
        return vdw_dynamic.sum()*self.config['weight_vdw']


    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key   = weight_key.split('_')[1]  # 'cc', 'pp', or 'nn'
        coeffs_list  = self.cs_coefs[spline_key]
        knots_list   = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (atom_map <= upper_th) & mask_block & self.fullmask & (atom_map >= lower_th)
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef  = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = Potential.cubic_distance(atom_map[mask], coef, knots, min_dis, max_dis, bin_num).sum() * self.config[weight_key] * 0.5
            else:
                part1 = torch.zeros((), device=dev)
            part2 = ((atom_map <= lower_th) & mask_block & self.fullmask).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,1], coor[:,1])
        return self._cubic_pair_energy(atom_map, self.geo_cc, self.geo_confimask_cc, 'weight_cc')

    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,0], coor[:,0])
        return self._cubic_pair_energy(atom_map, self.geo_pp, self.geo_confimask_pp, 'weight_pp')

    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,-1], coor[:,-1])
        return self._cubic_pair_energy(atom_map, self.geo_nn, self.geo_confimask_nn, 'weight_nn')

    def compute_pccp_energy(self,coor):
        p_atoms=coor[:,0]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( p_atoms[self.pccpi], c_atoms[self.pccpi], c_atoms[self.pccpj] ,p_atoms[self.pccpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pccp_coe,self.pccp_x,36)
        return neg_log.sum()*self.config['weight_pccp']

    def compute_cnnc_energy(self,coor):
        n_atoms=coor[:,-1]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( c_atoms[self.cnnci], n_atoms[self.cnnci], n_atoms[self.cnncj] ,c_atoms[self.cnncj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.cnnc_coe,self.cnnc_x,36)
        return neg_log.sum()*self.config['weight_cnnc']

    def compute_pnnp_energy(self,coor):
        n_atoms=coor[:,-1]
        p_atoms=coor[:,0]
        pccpmap=operations.dihedral( p_atoms[self.pnnpi], n_atoms[self.pnnpi], n_atoms[self.pnnpj] ,p_atoms[self.pnnpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pnnp_coe,self.pnnp_x,36)
        return neg_log.sum()*self.config['weight_pnnp']

    def compute_pcc_energy(self,coor):
        p_atoms=coor[:,1]
        c_atoms=coor[:,2]
        pccmap=operations.angle( p_atoms[self.pcci], c_atoms[self.pcci], c_atoms[self.pccj]                   )
        neg_log = Potential.cubic_angle(pccmap,self.pcc_coe,self.pcc_x,12)
        return neg_log.sum()*self.config['weight_pcc']

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)

    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting

    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        for tx,weightplddt in zip(self.tx2ds,self.pair):

            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) * self.local_weight[...,None] )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']

    def compute_af3_energy(self, coor, ep=1e-6):
        """Compute energy based on deviation from AlphaFold3 structure"""
        # Skip if no AF3 structure is available
        if not hasattr(self, 'af3_coords'):
            return 0

        # Print status message to indicate AF3 integration is active
        print(f"[DRfold2] Computing AlphaFold3 energy contribution (weight: {self.config['weight_af3']})")

        # Calculate rigid transformation
        p_rot = operations.rigidFrom3Points(coor)
        p_trans = coor[:, 1].clone()  # Use C4' as center of transform

        # Apply transformation to coordinates
        pred_coords = coor.clone().unsqueeze(1)  # Shape: [L, 1, 3, 3]
        pred_coords = pred_coords - p_trans.unsqueeze(1).unsqueeze(1)  # Translate
        pred_coords = torch.einsum('ijkl,ild->ijkd', pred_coords, p_rot.transpose(-1, -2))  # Rotate

        # Calculate error map between aligned structures
        af3_aligned = self.af3_aligned.clone()  # Shape: [L, L, 3, 3]
        errmap = torch.sqrt(((pred_coords - af3_aligned)**2).sum(dim=-1) + ep)  # Distance error

        # Apply error power and clamping
        max_dist = self.config.get('AF3_max', 20.0)
        error_power = self.config.get('af3_error_power', 2.0)

        # Calculate per-atom distances to filter out distant pairs
        atom_dists = torch.min(af3_aligned.norm(dim=-1), dim=2)[0]  # Min distance between atoms

        # Get pair weighting based on distance threshold
        pair_min_dist = self.config.get('pair_rest_min_dist', 2.0)
        mask = atom_dists > pair_min_dist

        # Apply error function
        energy = torch.sum((torch.clamp(errmap, max=max_dist)**error_power)[mask])

        return energy * self.config['weight_af3']

    def energy(self,rama):
        coor=a2b.quat2b(self.basex,rama[:,9:])
        other_coor = a2b.quat2b(self.otherx,rama[:,9:])
        side_coor = a2b.quat2b(self.sidex,torch.cat([rama[:,:9],coor[:,-1]],dim=-1))

        if self.config['weight_cc']>0:
            E_cc= self.compute_cc_energy(coor) / len(self.rets)
        else:
            E_cc=0
        if self.config['weight_pp']>0:
            E_pp= self.compute_pp_energy(coor) / len(self.rets)
        else:
            E_pp=0
        if self.config['weight_nn']>0:
            E_nn= self.compute_nn_energy(coor) / len(self.rets)
        else:
            E_nn=0

        if self.config['weight_pccp']>0:
            E_pccp= self.compute_pccp_energy(coor) / len(self.rets)
        else:
            E_pccp=0

        if self.config['weight_cnnc']>0:
            E_cnnc= self.compute_cnnc_energy(coor)  / len(self.rets)
        else:
            E_cnnc=0

        if self.config['weight_pnnp']>0:
            E_pnnp= self.compute_pnnp_energy(coor) / len(self.rets)
        else:
            E_pnnp=0

        if self.config['weight_vdw']>0:
            E_vdw= self.compute_full_clash(coor,other_coor,side_coor)
        else:
            E_vdw=0

        if self.config['weight_fape']>0:
            E_fape= self.compute_fape_energy_fromquat(rama[:,9:],coor) / len(self.rets)
        else:
            E_fape=0

        if self.config.get('weight_af3', 0) > 0 and hasattr(self, 'af3_coords'):
            E_af3 = self.compute_af3_energy(coor)
        else:
            E_af3 = 0

        if self.config['weight_bond']>0:
            E_bond= self.compute_bond_energy(coor,other_coor)
        else:
            E_bond=0

        return E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp + E_af3


    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)

    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f=self.energy(rama)*Scale_factor
            return f.item()


    def foldning(self):
        ilter = self.init_ret
        # 1) get initial quaternions (double precision)
        try:
            init_q = self.init_quat(ilter).double()
        except:
            init_q = self.init_quat_safe(ilter).double()

        # 2) move to target device (GPU if available), enable grad
        param = init_q.to(device).clone().detach().requires_grad_(True)

        # 3) set up PyTorch LBFGS optimizer over `param`
        optimizer = opt.LBFGS(
            [param],
            max_iter=self.config.get('max_iter', 300),
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        # 4) define the "closure" that LBFGS will call to reevaluate loss + gradients
        def closure():
            optimizer.zero_grad()                                 # clear old grads
            E = self.energy(param.view(self.L,21)) * Scale_factor # compute ∂E/∂param
            E.backward()
            return E

        # 5) run LBFGS until convergence (it calls closure repeatedly)
        optimizer.step(closure)

        # 6) write out final PDB
        final_energy = self.energy(param.view(self.L,21)).item()
        self.outpdb(param, self.saveprefix + '.pdb', energystr=str(final_energy))


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        # bring baseframes and quaternion data onto CPU to prevent device mismatch
        basex_cpu = self.basex.detach().cpu()
        otherx_cpu = self.otherx.detach().cpu()
        sidex_cpu = self.sidex.detach().cpu()
        shaped_rama = rama.view(self.L,21).detach().cpu()
        # compute backbone and other coords
        coor_np = a2b.quat2b(basex_cpu, shaped_rama[:,9:]).detach().cpu().numpy()
        other_np = a2b.quat2b(otherx_cpu, shaped_rama[:,9:]).detach().cpu().numpy()
        coor = torch.FloatTensor(coor_np)
        # compute side atom coords
        side_coor_NP = a2b.quat2b(sidex_cpu, torch.cat([shaped_rama[:,:9], coor[:,-1]], dim=-1)).detach().cpu().numpy()

        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']
                #atoms = ['P','C4']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()

    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1

        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


    def init_quat(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor,dim=1,keepdim=True)
        q = init_coor - biasq
        m = torch.einsum('bnz,bny->bzy',self.basex,q).reshape([self.L,-1])
        x[:,:9] = x[:,9:18] = m
        x.requires_grad_()
        return x

    def init_quat_safe(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor,dim=1,keepdim=True)
        q = init_coor - biasq + torch.rand([self.L,3,3])
        m = (torch.einsum('bnz,bny->bzy',self.basex,q) + torch.eye(3)[None,:,:]).reshape([self.L,-1])
        x[:,:9] = x[:,9:18] = m
        x.requires_grad_()
        return x


if __name__ == '__main__':

    fastafile=sys.argv[1]
    saveprefix=sys.argv[2]
    retdirs  =sys.argv[3]
    ret_score = sys.argv[4]
    foldconfig = sys.argv[5]

    # Check for optional AF3 file
    af3file = None
    if len(sys.argv) > 6:
        af3file = sys.argv[6]
        if os.path.exists(af3file):
            print(f"[DRfold2] Using AlphaFold3 structure: {af3file}")
        else:
            print(f"[DRfold2] Warning: AlphaFold3 file not found: {af3file}")
            af3file = None

    savepare = os.path.dirname(saveprefix)
    if not os.path.isdir(savepare):
        os.makedirs(savepare)

    num_of_models = readconfig(foldconfig)['num_of_models']

    score_dict = readconfig(ret_score)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1])
    lowest_n_keys = [item[0] for item in sorted_items][:num_of_models]
    bestkey = lowest_n_keys[0] + ''
    print("Before sort:", lowest_n_keys)
    lowest_n_keys.sort()
    print("After sort:", lowest_n_keys)
    bestindex = lowest_n_keys.index(bestkey)

    current_ret = bestkey
    retfiles = [os.path.join(retdirs, afile) for afile in lowest_n_keys]
    stru = Structure(fastafile, retfiles, saveprefix + '_from_' + current_ret, bestindex, foldconfig, af3file)
    stru.foldning()
Overwriting /kaggle/working/DRfold2/PotentialFold/Optimization.py
%%writefile /kaggle/working/DRfold2/PotentialFold/Selection.py
#! /nfs/amino-home/liyangum/miniconda3/bin/python
import numpy
import torch
import torch.autograd as autograd
import numpy as np

import random
import Cubic, Potential
import operations
import os, json, sys

import a2b, rigid
import torch.optim as opt
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from scipy.optimize import fmin_l_bfgs_b,fmin_cg,fmin_bfgs
from scipy.optimize import minimize
import lbfgs_rosetta
import pickle
import shutil

torch.manual_seed(6)
torch.set_num_threads(4)
np.random.seed(9)
random.seed(9)

Scale_factor = 1.0
USEGEO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config


class Structure:
    def __init__(self, fastafile, geofiles, foldconfig, saveprefix):
        # Load Configuration and Inputs
        self.config = readconfig(foldconfig)
        self.seqfile = fastafile
        self.foldconfig = foldconfig
        self.geofiles = geofiles

        # Load Model Results
        self.rets = [pickle.load(open(refile, 'rb')) for refile  in geofiles]

        # Extract Coordinates
        self.txs = []
        for ret in self.rets:
            self.txs.append(torch.from_numpy(ret['coor']).double().to(device))

        # Handle Geometrical Data
        self.handle_geo()

        # Extract pLDDT Scores
        self.pair = []
        for ret in self.rets:
            self.pair.append( torch.from_numpy(ret['plddt']).double().to(device))

        # Store Output and Sequence Info
        self.saveprefix = saveprefix
        self.seq = open(fastafile).readlines()[1].strip()
        self.L = len(self.seq)

        # Load Reference Arrays for Structure Construction
        basenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'base.npy'))
        self.basex = operations.Get_base(self.seq, basenpy).double().to(device)

        othernpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'other2.npy'))
        self.otherx = operations.Get_base(self.seq, othernpy).double().to(device)

        sidenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'side.npy'))
        self.sidex = operations.Get_base(self.seq, sidenpy).double().to(device)

        # Initialize Masks, Parameters, and FAPE
        self.init_mask()
        self.init_paras()
        self._init_fape()


    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot, true_trans = operations.Kabsch_rigid(self.basex, tx[:, 0], tx[:, 1], tx[:, 2])
            true_x2 = tx[:, None, :, :] - true_trans[None, :, None, :]
            true_x2 = torch.einsum('ijnd,jde->ijne', true_x2, true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)


    def handle_geo(self):
        oldkeys = ['dist_p', 'dist_c', 'dist_n']
        newkeys = ['pp', 'cc', 'nn']
        self.geos = []
        geo = {'pp':0, 'cc':0, 'nn':0}

        for ret in self.rets:
            for nk, ok in zip(newkeys, oldkeys):
                geo[nk] = geo[nk] + (ret[ok].astype(np.float64) /(len(self.rets)))
        self.geos.append(geo)


    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=torch.DoubleTensor(halfmask) > 0.5
        self.fullmask=torch.DoubleTensor(fullmask) > 0.5
        self.clash_mask = torch.zeros([self.L,self.L,22,22])
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq)
        self.side_mask = self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]
        self.clash_mask = (self.clash_mask > 0.5) * (self.side_mask > 0.5)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = torch.DoubleTensor(geo['cc'][:,:,-1]) < 0.5
            confimask_pp = torch.DoubleTensor(geo['pp'][:,:,-1]) < 0.5
            confimask_nn = torch.DoubleTensor(geo['nn'][:,:,-1]) < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)

        # Move masks and confimasks to the GPU/CPU device
        self.halfmask = self.halfmask.to(device)
        self.fullmask = self.fullmask.to(device)
        self.clash_mask = self.clash_mask.to(device)
        self.side_mask = self.side_mask.to(device)
        # geo_confimasks are lists
        self.geo_confimask_cc = [m.to(device) for m in self.geo_confimask_cc]
        self.geo_confimask_pp = [m.to(device) for m in self.geo_confimask_pp]
        self.geo_confimask_nn = [m.to(device) for m in self.geo_confimask_nn]


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {'cc': [], 'pp': [], 'nn': []}
        self.cs_knots = {'cc': [], 'pp': [], 'nn': []}
        for geo in self.geos:
            cc_cs, cc_decs = Cubic.dis_cubic(geo['cc'], 2, 40, 36)
            pp_cs, pp_decs = Cubic.dis_cubic(geo['pp'], 2, 40, 36)
            nn_cs, nn_decs = Cubic.dis_cubic(geo['nn'], 2, 40, 36)
            self.geo_cc.append([cc_cs, cc_decs])
            self.geo_pp.append([pp_cs, pp_decs])
            self.geo_nn.append([nn_cs, nn_decs])
            L = self.L
            cc_coefs_np = np.stack([[cc_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            cc_knots_np = np.stack([[cc_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['cc'].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots['cc'].append(torch.from_numpy(cc_knots_np).to(device))
            pp_coefs_np = np.stack([[pp_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            pp_knots_np = np.stack([[pp_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['pp'].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots['pp'].append(torch.from_numpy(pp_knots_np).to(device))
            nn_coefs_np = np.stack([[nn_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            nn_knots_np = np.stack([[nn_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['nn'].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots['nn'].append(torch.from_numpy(nn_knots_np).to(device))


    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key = weight_key.split('_')[1]
        coeffs_list = self.cs_coefs[spline_key]
        knots_list = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (atom_map <= upper_th) & mask_block & self.fullmask & (atom_map >= lower_th)
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = Potential.cubic_distance(atom_map[mask], coef, knots, min_dis, max_dis, bin_num).sum() * self.config[weight_key] * 0.5
            else:
                part1 = torch.zeros((), device=dev, dtype=torch.double)
            part2 = ((atom_map <= lower_th) & mask_block & self.fullmask).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    # GPU-friendly torsion and angle energy helpers
    def _cubic_torsion_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_torsion(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def _cubic_angle_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_angle(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,1], coor[:,1])
        return self._cubic_pair_energy(atom_map, self.geo_cc, self.geo_confimask_cc, 'weight_cc')

    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,0], coor[:,0])
        return self._cubic_pair_energy(atom_map, self.geo_pp, self.geo_confimask_pp, 'weight_pp')

    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,-1], coor[:,-1])
        return self._cubic_pair_energy(atom_map, self.geo_nn, self.geo_confimask_nn, 'weight_nn')

    def compute_pccp_energy(self, coor):
        # P-C-C-P dihedral energy on GPU
        p = coor[:, 0]
        c = coor[:, 1]
        dia = operations.dihedral(
            p[self.pccpi], c[self.pccpi], c[self.pccpj], p[self.pccpj]
        )
        return self._cubic_torsion_energy(dia, self.pccp_coe, self.pccp_x, 'weight_pccp', 36)

    def compute_cnnc_energy(self, coor):
        # C-N-N-C dihedral energy on GPU
        n = coor[:, -1]
        c = coor[:, 1]
        dia = operations.dihedral(
            c[self.cnnci], n[self.cnnci], n[self.cnncj], c[self.cnncj]
        )
        return self._cubic_torsion_energy(dia, self.cnnc_coe, self.cnnc_x, 'weight_cnnc', 36)

    def compute_pnnp_energy(self, coor):
        # P-N-N-P dihedral energy on GPU
        n = coor[:, -1]
        p = coor[:, 0]
        dia = operations.dihedral(
            p[self.pnnpi], n[self.pnnpi], n[self.pnnpj], p[self.pnnpj]
        )
        return self._cubic_torsion_energy(dia, self.pnnp_coe, self.pnnp_x, 'weight_pnnp', 36)

    def compute_pcc_energy(self, coor):
        # P-C-C angle energy on GPU
        p = coor[:, 1]
        c = coor[:, 2]
        ang = operations.angle(
            p[self.pcci], c[self.pcci], c[self.pccj]
        )
        return self._cubic_angle_energy(ang, self.pcc_coe, self.pcc_x, 'weight_pcc', 12)

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)

    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting

    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse

        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']

    def compute_fape_energy_fromcoor(self,coor,ep=1e-6,epmax=100):
        energy= 0

        p_rot,px_mean = operations.Kabsch_rigid(self.basex,coor[:,0],coor[:,1],coor[:,2])
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse

        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']


    def energy(self, rama):
        coor = a2b.quat2b(self.basex, rama[:, 9:])
        other_coor = a2b.quat2b(self.otherx, rama[:, 9:])
        side_coor = a2b.quat2b(self.sidex, torch.cat([rama[:, :9], coor[:, -1]], dim=-1))

        E_cc = self.compute_cc_energy(coor) / len(self.geofiles) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) / len(self.geofiles) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) / len(self.geofiles) if self.config['weight_nn'] > 0 else 0
        E_pccp = self.compute_pccp_energy(coor) / len(self.geofiles) if self.config['weight_pccp'] > 0 else 0
        E_cnnc = self.compute_cnnc_energy(coor) / len(self.geofiles) if self.config['weight_cnnc'] > 0 else 0
        E_pnnp = self.compute_pnnp_energy(coor) / len(self.geofiles) if self.config['weight_pnnp'] > 0 else 0
        E_vdw = self.compute_full_clash(coor, other_coor, side_coor) if self.config['weight_vdw'] > 0 else 0
        E_fape = self.compute_fape_energy_fromquat(rama[:, 9:], coor) / len(self.geofiles) if self.config['weight_fape'] > 0 else 0
        E_bond = self.compute_bond_energy(coor, other_coor) if self.config['weight_bond'] > 0 else 0

        return E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp


    def energy_from_coor(self, coor):
        E_cc = self.compute_cc_energy(coor) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) if self.config['weight_nn'] > 0 else 0
        E_fape = (self.compute_fape_energy_fromcoor(coor) / len(self.geofiles)) if self.config['weight_fape'] > 0 else 0
        print(E_fape, E_pp, E_cc, E_nn)
        return E_fape + E_pp + E_cc + E_nn

    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)

    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f = self.energy(rama)*Scale_factor
            return f.item()

    def saveconfig(self,dict,confile):
        json_object = json.dumps(dict, indent = 4)
        wfile = open(confile,'w')
        wfile.write(json_object)
        wfile.close()

    def scoring(self):
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] = geoscale * self.config['weight_pp']
        self.config['weight_cc'] = geoscale * self.config['weight_cc']
        self.config['weight_nn'] = geoscale * self.config['weight_nn']
        self.config['weight_pccp'] = geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] = geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] = geoscale * self.config['weight_pnnp']

        energy_dict = {}
        saveenergy_dict  = {}

        with torch.no_grad():
            for retfile, tx in zip(self.geofiles, self.txs):
                one = self.energy_from_coor(tx)
                aaretfile = os.path.basename(retfile)
                energy_dict[aaretfile] = one.item()
                saveenergy_dict[retfile] = one.item()
            self.saveconfig(energy_dict, self.saveprefix)


    def foldning(self):
        minenergy=1e16
        count=0
        for tx in self.txs:
            count+=1

        minirama=None

        ilter = self.init_ret
        selected_ret = self.geofiles[ilter]
        try:
            rama=self.init_quat(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()
        except:
            rama=self.init_quat_safe(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()

        self.config=readconfig(self.foldconfig)
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] =geoscale * self.config['weight_pp']
        self.config['weight_cc'] =geoscale * self.config['weight_cc']
        self.config['weight_nn'] =geoscale * self.config['weight_nn']
        self.config['weight_pccp'] =geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] =geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] =geoscale * self.config['weight_pnnp']
        for i in range(3):
            line_min = lbfgs_rosetta.ArmijoLineMinimization(self.obj_func_np,self.obj_func_grad_np,True,len(rama),120)
            lbfgs_opt = lbfgs_rosetta.lbfgs(self.obj_func_np,self.obj_func_grad_np)
            rama=lbfgs_opt.run(rama,256,lbfgs_rosetta.absolute_converge_test,line_min,8000,self.obj_func_np,self.obj_func_grad_np,1e-9)
        newrama=rama+0.0
        newrama=torch.DoubleTensor(newrama)
        current_energy =self.obj_func_np(rama)

        if current_energy < minenergy:
            print(current_energy,minenergy)
            minenergy=current_energy
            self.outpdb(newrama,self.saveprefix+'.pdb',energystr=str(current_energy))


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        coor_np=a2b.quat2b(self.basex,rama.view(self.L,21)[:,9:]).data.numpy()
        other_np=a2b.quat2b(self.otherx,rama.view(self.L,21)[:,9:]).data.numpy()
        shaped_rama=rama.view(self.L,21)
        coor = torch.FloatTensor(coor_np)
        side_coor_NP = a2b.quat2b(self.sidex,torch.cat([shaped_rama[:,:9],coor[:,-1]],dim=-1)).data.numpy()

        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()

        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']

            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1

        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


if __name__ == '__main__':

    fastafile = sys.argv[1]
    foldconfig = sys.argv[2]
    save_prefix = sys.argv[3]
    retfiles = sys.argv[4:]

    save_parent_dir = os.path.dirname(save_prefix)
    if not os.path.isdir(save_parent_dir):
        os.makedirs(save_parent_dir)

    retfiles.sort()
    print(retfiles)

    stru = Structure(fastafile, retfiles, foldconfig, save_prefix)
    stru.scoring()
Overwriting /kaggle/working/DRfold2/PotentialFold/Selection.py
%%writefile /kaggle/working/DRfold2/PotentialFold/Cubic.py
import numpy as np
from scipy.interpolate import CubicSpline,UnivariateSpline
import os
from torch.autograd import Function
import torch
import math

def fit_dis_cubic(dis_matrix,min_dis,max_dis,num_bin):
    # convert torch Tensor on GPU to numpy array for SciPy
    if isinstance(dis_matrix, torch.Tensor):
        dis_matrix = dis_matrix.detach().cpu().numpy()
    dis_region=np.zeros(num_bin)
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      (dis_matrix[i,j,1:-1]+1e-8) / (dis_matrix[i,j,[-2]]+1e-8)              )
            x=dis_region
            x[0]=-0.0001
            y[0]= max(10,y[1]+4)
            cs= CubicSpline(x,y)
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)

def dis_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic distance')
    cs,decs=fit_dis_cubic(out,min_dis,max_dis,num_bin)
    return cs,decs



def cubic_matrix_torsion(dis_matrix,min_dis,max_dis,num_bin):
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region
            x=np.append(x,x[-1]+bin_size)
            y=np.append(y,y[0])
            cs= CubicSpline(x,y,bc_type='periodic')
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)
def torsion_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic')
    cs,decs=cubic_matrix_torsion(out,min_dis,max_dis,num_bin)
    return cs,decs

def cubic_matrix_angle(dis_matrix,min_dis,max_dis,num_bin): # 0 - np.pi 12
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region

            x=np.concatenate([[x[0]-bin_size*3,x[0]-bin_size*2,x[0]-bin_size], x,[x[-1]+bin_size,x[-1]+bin_size*2,x[-1]+bin_size*3]               ])
            y=np.concatenate([ [y[2],y[1],y[0]],y,[y[-1],y[-2],y[-3]]                                                                                                                    ])

            cs= CubicSpline(x,y)
            decs=cs.derivative()

            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)

    return np.array(csnp),np.array(decsnp)
def angle_cubic(out,min_dis,max_dis,num_bin):

    print('fitting angle cubic')
    cs,decs=cubic_matrix_angle(out,min_dis,max_dis,num_bin)

    return cs,decs
Overwriting /kaggle/working/DRfold2/PotentialFold/Cubic.py
%%writefile /kaggle/working/DRfold2/cfg_for_folding.json
{
    "weight_pp": 1,
    "weight_cc": 1,
    "weight_nn": 1,
    "weight_pccp": 0,
    "weight_cnnc": 0,
    "weight_pnnp": 0,
    "weight_pcc": 0,
    "weight_cnn": 0,
    "weight_pnn": 0,
    "weight_vdw": 1,
    "weight_nn_contact": 0,
    "weight_cc_contact": 0,
    "weight_beta": 0,
    "weight_fape": 2,
    "weight_af3": 2.5,
    "weight_bond": 5000,
    "pair_weight_power": 0.25,
    "pair_weight_min": 0.2,
    "pair_error_power": 3,
    "af3_error_power": 2.0,
    "pair_rest_min_dist": 2,
    "FAPE_max": 30,
    "AF3_max": 20,
    "geo_scale": 450,
    "num_of_models": 5
}
Overwriting /kaggle/working/DRfold2/cfg_for_folding.json
%%writefile /kaggle/working/DRfold2/cfg_for_selection.json
{
    "weight_pp": 1,
    "weight_cc": 1,
    "weight_nn": 1,
    "weight_pccp": 0,
    "weight_cnnc": 0,
    "weight_pnnp": 0,
    "weight_pcc": 0,
    "weight_cnn": 0,
    "weight_pnn": 0,
    "weight_vdw": 1,
    "weight_nn_contact": 0,
    "weight_cc_contact": 0,
    "weight_beta": 0,
    "weight_fape": 1,
    "weight_af3": 2.0,
    "weight_bond": 1000,
    "pair_weight_power": 0.5,
    "pair_weight_min": 0.3,
    "pair_error_power": 3.5,
    "af3_error_power": 2.0,
    "pair_rest_min_dist": 2,
    "FAPE_max": 30,
    "AF3_max": 20,
    "geo_scale": 450,
    "num_of_models": 5
}
Overwriting /kaggle/working/DRfold2/cfg_for_selection.json
Hybrid Method Follows
from Bio.PDB import MMCIFParser, PDBIO, Select

class RNAAtomSelect(Select):
    """Select only RNA backbone and base atoms needed by DRfold2"""
    def accept_atom(self, atom):
        # Check if this is an atom DRfold2 needs
        atom_name = atom.name
        residue = atom.get_parent()
        resname = residue.get_resname()

        # Main backbone atoms needed by DRfold2
        if atom_name in ["P", "C4'"]:
            return True

        # For purines (A, G) we need N9, for pyrimidines (C, U) we need N1
        if atom_name == "N9" and resname in ["A", "G"]:
            return True
        if atom_name == "N1" and resname in ["C", "U"]:
            return True

        return False

def convert_cif_to_pdb(cif_file, pdb_file):
    """Convert mmCIF file to PDB format, fixing chain IDs and keeping only needed atoms.

    Args:
        cif_file: Path to input mmCIF file
        pdb_file: Path to output PDB file

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Parse the mmCIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('', cif_file)

        # Fix chain IDs (map multi-character IDs like 'A1' to single characters)
        for model in structure:
            for chain in model:
                if len(chain.id) > 1:
                    # Just take the first character of the chain ID
                    print(f"Fixing chain ID: {chain.id} -> {chain.id[0]}")
                    chain.id = chain.id[0]

        # Write to PDB format, selecting only atoms needed by DRfold2
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file, RNAAtomSelect())
        print(f"Successfully converted {cif_file} to {pdb_file}")
        return True
    except Exception as e:
        print(f"Error converting {cif_file} to PDB: {str(e)}")

        # Attempt alternative approach if primary method fails
        try:
            print("Trying alternative method...")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('', cif_file)

            # Create a new PDB file manually
            with open(pdb_file, 'w') as f:
                atom_num = 1

                for model in structure:
                    for chain in model:
                        chain_id = 'A'  # Use 'A' regardless of original ID

                        for residue in chain:
                            resname = residue.get_resname()
                            resnum = residue.id[1]

                            # Select atoms based on residue type
                            needed_atoms = ["P", "C4'"]
                            if resname in ["A", "G"]:
                                needed_atoms.append("N9")
                            else:  # C, U
                                needed_atoms.append("N1")

                            for atom_name in needed_atoms:
                                if atom_name in residue:
                                    atom = residue[atom_name]
                                    x, y, z = atom.coord

                                    # Format as PDB ATOM line
                                    line = f"ATOM  {atom_num:5d} {atom_name:<4s} {resname:3s} {chain_id:1s}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom.element:>2s}  \n"
                                    f.write(line)
                                    atom_num += 1

                f.write("END\n")

            print(f"Successfully created PDB file using alternative method")
            return True

        except Exception as e2:
            print(f"Alternative method also failed: {str(e2)}")
            return False
# Define a function to run DRfold2 that captures output
def predict_rna_structures_drfold2(sequence, target_id, af3_pdb=None):
    """
    Use DRfold2 to predict RNA structures with proper output capture

    Parameters:
    -----------
    sequence : str
        RNA sequence to predict
    target_id : str
        Identifier for the target
    af3_pdb : str, optional
        Path to AlphaFold3 PDB file for hybrid prediction mode

    Returns:
    --------
    list of list of tuples
        Five predictions, each containing C1' coordinates for each residue
    """
    import subprocess
    from subprocess import PIPE, STDOUT

    # Create FASTA file for this sequence
    fasta_path = os.path.join(fasta_dir, f"{target_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")

    # Run DRfold2 with proper output capture
    output_dir = os.path.join(predictions_dir, target_id)

    # Build command with optional AF3 integration
    cmd = f"python /kaggle/working/DRfold2/DRfold_infer.py {fasta_path} {output_dir} 1"
    if af3_pdb and os.path.exists(af3_pdb):
        cmd += f" --af3 {af3_pdb}"
        print(f"Using AlphaFold3 structure from: {af3_pdb}")
    elif af3_pdb:
        print(f"Warning: AlphaFold3 file not found: {af3_pdb}")

    print(f"Running command: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(line)

    # Get return code and check success
    return_code = process.wait()
    if return_code != 0:
        print(f"DRfold2 failed with return code {return_code}")
        return None

    # Clean up FASTA file to save space
    os.remove(fasta_path)

    # Extract coordinates
    relax_dir = os.path.join(output_dir, "relax")
    if not os.path.isdir(relax_dir):
        print(f"Warning: No relax directory found for {target_id}")
        relax_dir = output_dir

    # Get up to 5 PDB files
    pdb_files = sorted([f for f in os.listdir(relax_dir) if f.endswith(".pdb")])[:5]

    if not pdb_files:
        print(f"Warning: No PDB files found for {target_id}")
        # Return None to indicate failure
        return None

    # Parse PDB files to extract C1' coordinates
    predictions = []
    for pdb_file in pdb_files:
        file_path = os.path.join(relax_dir, pdb_file)

        # Read PDB file
        coords = []
        with open(file_path, "r") as f:
            residue_map = {}
            for line in f:
                if line.startswith("ATOM") and " C1' " in line:
                    parts = line.split()
                    resid = int(parts[5])  # Residue ID as integer
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                    residue_map[resid] = (x, y, z)

            # Ensure we have coordinates for all residues
            for j in range(1, len(sequence) + 1):
                if j in residue_map:
                    coords.append(residue_map[j])
                else:
                    # If residue not found, use zeros
                    print(f"Warning: Residue {j} not found in {pdb_file} for {target_id}")
                    coords.append((0.0, 0.0, 0.0))

        predictions.append(coords)

    # Clean up PDB files to save space
    if is_submission_mode:
        shutil.rmtree(output_dir)

    # If we have fewer than 5 predictions, duplicate the last one
    while len(predictions) < 5:
        predictions.append(predictions[-1] if predictions else [(0.0, 0.0, 0.0) for _ in range(len(sequence))])

    return predictions[:5]  # Return exactly 5 predictions
# Vectorized version of process_labels function
def process_labels_vectorized(labels_df):
    # Extract target_id from ID column (remove last part after underscore)
    labels_df = labels_df.copy()
    labels_df['target_id'] = labels_df['ID'].str.rsplit('_', n=1).str[0]

    # Sort by target_id and resid for proper ordering
    labels_df = labels_df.sort_values(['target_id', 'resid'])

    # Group by target_id and convert coordinates to arrays
    coords_dict = {}
    for target_id, group in labels_df.groupby('target_id'):
        # Extract coordinates as numpy array in one operation
        coords_dict[target_id] = group[['x_1', 'y_1', 'z_1']].values

    return coords_dict

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    similar_seqs = []
    query_seq_obj = Seq(query_seq)

    for _, row in train_seqs_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue

        # Skip if sequence is too different in length (more than 40% difference)
        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.4:
            continue

        # Perform sequence alignment
        alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)

        if alignments:
            alignment = alignments[0]
            similarity_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
            similar_seqs.append((target_id, train_seq, similarity_score, train_coords_dict[target_id]))

    # Sort by similarity score (higher is better) and return top N
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]



# ======= adaptive_rna_constraints =================
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    """Apply realistic RNA structural constraints"""
    # Make a copy of coordinates to refine
    refined_coords = coordinates.copy()
    n_residues = len(sequence)

    # Calculate constraint strength (inverse of confidence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))

    # 1. Sequential distance constraints (consecutive nucleotides)
    seq_min_dist = 5.5  # Minimum sequential distance
    seq_max_dist = 6.5  # Maximum sequential distance

    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i+1]

        # Calculate current distance
        current_dist = np.linalg.norm(next_pos - current_pos)

        # Only adjust if significantly outside expected range
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            # Calculate target distance (midpoint of range)
            target_dist = (seq_min_dist + seq_max_dist) / 2

            # Get direction vector
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Apply partial adjustment based on constraint strength
            adjustment = (target_dist - current_dist) * constraint_strength

            # Only adjust the next position to preserve the overall fold
            refined_coords[i+1] = current_pos + direction * (current_dist + adjustment)

    # 2. Steric clash prevention
    min_allowed_distance = 3.8  # Minimum distance between non-consecutive C1' atoms

    # Calculate all pairwise distances
    dist_matrix = distance_matrix(refined_coords, refined_coords)

    # Find severe clashes (atoms too close)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))

    # Fix severe clashes
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]

        # Skip consecutive nucleotides and previously processed pairs
        if abs(i - j) <= 1 or i >= j:
            continue

        # Get current positions and distance
        pos_i = refined_coords[i]
        pos_j = refined_coords[j]
        current_dist = dist_matrix[i, j]

        # Calculate necessary adjustment but scale by constraint strength
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Calculate partial adjustment
        adjustment = (min_allowed_distance - current_dist) * constraint_strength

        # Move points apart
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)

    return refined_coords

def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    if alignment is None:
        from Bio.Seq import Seq
        from Bio import pairwise2

        query_seq_obj = Seq(query_seq)
        template_seq_obj = Seq(template_seq)
        alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj, 2.9, -1, -10, -0.5, one_alignment_only=True)

        if not alignments:
            return generate_improved_rna_structure(query_seq)

        alignment = alignments[0]

    aligned_query = alignment.seqA
    aligned_template = alignment.seqB

    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)

    # Map template coordinates to query
    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]

        if query_char != '-' and template_char != '-':
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != '-' and template_char == '-':
            query_idx += 1
        elif query_char == '-' and template_char != '-':
            template_idx += 1

    # IMPROVED GAP FILLING - maintains RNA backbone geometry
    backbone_distance = 5.9  # Typical C1'-C1' distance

    # Fill gaps by maintaining realistic backbone connectivity
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            # Find nearest valid neighbors
            prev_valid = next_valid = None

            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j
                    break

            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j
                    break

            if prev_valid is not None and next_valid is not None:
                # Interpolate along realistic RNA backbone path
                gap_size = next_valid - prev_valid
                total_distance = np.linalg.norm(query_coords[next_valid] - query_coords[prev_valid])
                expected_distance = gap_size * backbone_distance

                # If gap is compressed, extend it realistically
                if total_distance < expected_distance * 0.7:
                    direction = query_coords[next_valid] - query_coords[prev_valid]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    # Place intermediate points along extended path
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        progress = (k + 1) / gap_size
                        base_pos = query_coords[prev_valid] + direction * expected_distance * progress

                        # Add slight curvature for realism
                        perpendicular = np.cross(direction, [0, 0, 1])
                        if np.linalg.norm(perpendicular) < 1e-6:
                            perpendicular = np.cross(direction, [1, 0, 0])
                        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)

                        curve_amplitude = 2.0 * np.sin(progress * np.pi)
                        query_coords[idx] = base_pos + perpendicular * curve_amplitude
                else:
                    # Linear interpolation for normal gaps
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        weight = (k + 1) / gap_size
                        query_coords[idx] = (1 - weight) * query_coords[prev_valid] + weight * query_coords[next_valid]

            elif prev_valid is not None:
                # Extend from previous position
                if prev_valid > 0 and not np.isnan(query_coords[prev_valid-1, 0]):
                    direction = query_coords[prev_valid] - query_coords[prev_valid-1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1.0, 0.0, 0.0])

                steps_needed = i - prev_valid
                for step in range(1, steps_needed + 1):
                    pos_idx = prev_valid + step
                    if pos_idx < len(query_coords):
                        query_coords[pos_idx] = query_coords[prev_valid] + direction * backbone_distance * step

            elif next_valid is not None:
                # Work backwards from next position
                direction = np.array([-1.0, 0.0, 0.0])  # Default backward direction
                steps_needed = next_valid - i
                for step in range(steps_needed, 0, -1):
                    pos_idx = next_valid - step
                    if pos_idx >= 0:
                        query_coords[pos_idx] = query_coords[next_valid] - direction * backbone_distance * step

    # Final cleanup
    query_coords = np.nan_to_num(query_coords)
    return query_coords


# ========== generate_improved_rna_structure ========================
def generate_improved_rna_structure(sequence):
    """
    Generate a more realistic RNA structure fallback based on sequence patterns
    and basic RNA structure principles.

    Args:
        sequence: RNA sequence string

    Returns:
        Array of 3D coordinates
    """
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))

    # Analyze sequence to predict structural elements
    # Look for complementary regions that could form base pairs
    potential_stems = identify_potential_stems(sequence)

    # Default parameters
    radius_helix = 10.0
    radius_loop = 15.0
    rise_per_residue_helix = 2.5
    rise_per_residue_loop = 1.5
    angle_per_residue_helix = 0.6
    angle_per_residue_loop = 0.3

    # Assign structural classifications
    structure_types = assign_structure_types(sequence, potential_stems)

    # Generate coordinates based on predicted structure
    current_pos = np.array([0.0, 0.0, 0.0])
    current_direction = np.array([0.0, 0.0, 1.0])
    current_angle = 0.0

    for i in range(n_residues):
        if structure_types[i] == 'stem':
            # Part of a helical stem
            current_angle += angle_per_residue_helix
            coordinates[i] = [
                radius_helix * np.cos(current_angle),
                radius_helix * np.sin(current_angle),
                current_pos[2] + rise_per_residue_helix
            ]
            current_pos = coordinates[i]
        elif structure_types[i] == 'loop':
            # Part of a loop
            current_angle += angle_per_residue_loop
            z_shift = rise_per_residue_loop * np.sin(current_angle * 0.5)
            coordinates[i] = [
                radius_loop * np.cos(current_angle),
                radius_loop * np.sin(current_angle),
                current_pos[2] + z_shift
            ]
            current_pos = coordinates[i]
        else:
            # Single-stranded region
            # Add some randomness to make it look more realistic
            jitter = np.random.normal(0, 1, 3) * 2.0
            coordinates[i] = current_pos + jitter
            current_pos = coordinates[i]

    return coordinates

def identify_potential_stems(sequence):
    """
    Identify potential stem regions by looking for self-complementary segments.

    Args:
        sequence: RNA sequence string

    Returns:
        List of tuples (start1, end1, start2, end2) representing potentially paired regions
    """
    complementary_bases = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    min_stem_length = 3
    potential_stems = []

    # Simple stem identification
    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length + 1):
            # Check if regions could form a stem
            potential_stem_len = min(min_stem_length, len(sequence) - j)
            is_stem = True

            for k in range(potential_stem_len):
                if sequence[i+k] not in complementary_bases or \
                   complementary_bases[sequence[i+k]] != sequence[j+potential_stem_len-k-1]:
                    is_stem = False
                    break

            if is_stem:
                potential_stems.append((i, i+potential_stem_len-1, j, j+potential_stem_len-1))

    return potential_stems

def assign_structure_types(sequence, potential_stems):
    """
    Assign each nucleotide to a structural element type.

    Args:
        sequence: RNA sequence string
        potential_stems: List of tuples representing stem regions

    Returns:
        List of structure types ('stem', 'loop', 'single')
    """
    structure_types = ['single'] * len(sequence)

    # Mark stem regions
    for stem in potential_stems:
        start1, end1, start2, end2 = stem
        for i in range(end1 - start1 + 1):
            structure_types[start1 + i] = 'stem'
            structure_types[end2 - i] = 'stem'

    # Mark loop regions (regions between paired regions)
    for i in range(len(potential_stems) - 1):
        _, end1, start2, _ = potential_stems[i]
        next_start1, _, _, _ = potential_stems[i+1]

        if next_start1 > end1 + 1 and start2 > next_start1:
            for j in range(end1 + 1, next_start1):
                structure_types[j] = 'loop'

    return structure_types


# =========== generate_rna_structure ======================
def generate_rna_structure(sequence, seed=None):
    """Generate a more realistic RNA structure when no good templates are found"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))

    # Initialize the first few residues in a helix
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]

    # Add more complex folding patterns
    current_direction = np.array([0.0, 0.0, 1.0])  # Start moving along z-axis

    # Define base-pairing tendencies (G-C and A-U pairs)
    for i in range(3, n_residues):
        # Check for potential base-pairing in the sequence
        has_pair = False
        pair_idx = -1

        # Simple detection of complementary bases (G-C, A-U)
        complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
        current_base = sequence[i]

        # Look for potential base-pairing within a window before the current position
        window_size = min(i, 15)  # Look back up to 15 bases
        for j in range(i-window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, 'X'):
                # Found a potential pair
                has_pair = True
                pair_idx = j
                break

        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Try to create a base-pair by positioning this nucleotide near its pair
            pair_pos = coordinates[pair_idx]

            # Create a position that's roughly opposite to the pair
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)

            # Calculate a vector from base-pair toward center of structure
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Position new nucleotide in the general direction of the "center"
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset

            # Update direction for next nucleotide
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)

        else:
            # No base-pairing detected, continue with the current fold direction
            # Randomly rotate current direction to simulate RNA flexibility
            if random.random() < 0.3:
                # More significant direction change
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                # Small random changes in direction
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)

            # Distance between consecutive nucleotides (3.5-4.5Å is typical)
            step_size = random.uniform(3.5, 4.5)

            # Update position
            coordinates[i] = coordinates[i-1] + step_size * current_direction

    return coordinates


# ========== predict_rna_structures ==================
def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []

    # Find similar sequences in the training data
    similar_seqs = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, top_n=n_predictions)

    # If we found any similar sequences, use them as templates
    if similar_seqs:
        for i, (template_id, template_seq, similarity_score, template_coords) in enumerate(similar_seqs):
            # Adapt template coordinates to the query sequence
            adapted_coords = adapt_template_to_query(sequence, template_seq, template_coords)

            if adapted_coords is not None:
                # Apply adaptive constraints based on template similarity
                # For high similarity templates, apply very gentle constraints
                refined_coords = adaptive_rna_constraints(adapted_coords, sequence, confidence=similarity_score)

                # Add some randomness (less for better templates)
                random_scale = max(0.05, 0.8 - similarity_score)  # Reduced randomness
                randomized_coords = refined_coords.copy()
                randomized_coords += np.random.normal(0, random_scale, randomized_coords.shape)

                predictions.append(randomized_coords)

                if len(predictions) >= n_predictions:
                    break

    # If we don't have enough predictions from templates, generate de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)

        # Apply stronger constraints to de novo structures (lower confidence)
        refined_de_novo = adaptive_rna_constraints(de_novo_coords, sequence, confidence=0.2)

        predictions.append(refined_de_novo)

    return predictions[:n_predictions]
# Initialize counters and range settings
if is_submission_mode:
    DRFOLD_START_IDX = 14
    DRFOLD_END_IDX = len(test_sequences) - 1
else:
    DRFOLD_START_IDX = 0
    DRFOLD_END_IDX = 0

drfold_processed = 0
template_processed = 0

train_coords_dict = process_labels_vectorized(train_labels_final)
# from IPython.display import clear_output
# Sort test sequences by length to process shorter ones with DRfold2
test_sequences = test_sequences.sort_values(by=['sequence'], key=lambda x: x.str.len())

# List to store all prediction records
all_predictions = []

# Set up time tracking
start_time = time.time()
total_targets = len(test_sequences)

# For each sequence in the test set
for idx, row in test_sequences.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']

    # Progress tracking
    elapsed = time.time() - start_time
    targets_processed = idx
    if targets_processed > 0:
        avg_time_per_target = elapsed / targets_processed
        est_time_remaining = avg_time_per_target * (total_targets - targets_processed)
        time_left = DRFOLD_TIME_LIMIT - (time.time() - start_time_global)
        print(f"Processing target {targets_processed+1}/{total_targets}: {target_id} ({len(sequence)} nt), "
              f"elapsed: {elapsed:.1f}s, est. remaining: {est_time_remaining:.1f}s, time left: {time_left:.1f}s")

    # Check if we should use DRfold2 or template-based approach
    use_drfold = (DRFOLD_START_IDX <= idx <= DRFOLD_END_IDX and
                 (time.time() - start_time_global) < DRFOLD_TIME_LIMIT)

    # Generate 5 different structure predictions
    if use_drfold:
        print(f"Using DRfold2 for target {target_id} (index {idx})")

        cif_file = f"/kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/{target_id}/{target_id}_model_0.cif"
        pdb_file = cif_file.replace('.cif', '.pdb')

        if convert_cif_to_pdb(cif_file, pdb_file):
            # Use the converted PDB file with DRfold2
            print("af3file generated")
            af3file = pdb_file
        else:
            print("Failed to convert mmCIF file, skipping AF3 integration")
            af3file = None


        print(f"Using af3file: {af3file}")

        # Without AlphaFold3 (original mode)
        # predictions = predict_rna_structures_drfold2(sequence, target_id)

        # With AlphaFold3 integration
        predictions = predict_rna_structures_drfold2(sequence, target_id, af3_pdb=af3file)

        # If DRfold2 fails, fall back to template approach
        if predictions is None:
            print(f"DRfold2 failed for {target_id}, falling back to template approach")
            predictions = predict_rna_structures(sequence, target_id, train_seqs_final, train_coords_dict)
            template_processed += 1
        else:
            drfold_processed += 1
    else:
        if idx > DRFOLD_END_IDX:
            reason = "index out of DRfold range"
        elif idx < DRFOLD_START_IDX:
            reason = "index before DRfold start range"
        else:
            reason = "time limit reached"
        print(f"Using template approach for {target_id} ({reason})")
        predictions = predict_rna_structures(sequence, target_id, train_seqs_final, train_coords_dict)
        template_processed += 1

    # For each residue in the sequence
    for j in range(len(sequence)):
        pred_row = {
            'ID': f"{target_id}_{j+1}",
            'resname': sequence[j],
            'resid': j + 1
        }

        # Add coordinates from all 5 predictions
        for i in range(5):
            pred_row[f'x_{i+1}'] = predictions[i][j][0]
            pred_row[f'y_{i+1}'] = predictions[i][j][1]
            pred_row[f'z_{i+1}'] = predictions[i][j][2]

        all_predictions.append(pred_row)

    # clear_output(wait=False)

    # Free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Create DataFrame with predictions
submission_df = pd.DataFrame(all_predictions)

# Ensure the submission file has the correct format
column_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for coord in ['x', 'y', 'z']:
        column_order.append(f'{coord}_{i}')

submission_df = submission_df[column_order]

# Save the submission
submission_df.to_csv('/kaggle/working/submission_dr.csv', index=False)
print(f"Generated predictions for {len(test_sequences)} RNA sequences")
print(f"Used DRfold2 for {drfold_processed} targets and template approach for {template_processed} targets")
print(f"Total runtime: {time.time() - start_time_global:.1f} seconds")
Processing target 4/12: R1117v2 (30 nt), elapsed: 0.0s, est. remaining: 0.0s, time left: 25147.1s
Using template approach for R1117v2 (index out of DRfold range)
Using DRfold2 for target R1107 (index 0)
Fixing chain ID: A1 -> A
Successfully converted /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.cif to /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
af3file generated
Using af3file: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
Using AlphaFold3 structure from: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
Running command: python /kaggle/working/DRfold2/DRfold_infer.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107 1 --af3 /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2] Starting prediction pipeline on cuda device
[DRfold2] Input: /kaggle/working/fasta_files/R1107.fasta
[DRfold2] Output: /kaggle/working/predictions/R1107
[DRfold2] Clustering enabled - will generate multiple models
[DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2] Created output directory: /kaggle/working/predictions/R1107
[DRfold2] Created returns directory: /kaggle/working/predictions/R1107/rets_dir
[DRfold2] Created folds directory: /kaggle/working/predictions/R1107/folds
[DRfold2] Created relaxation directory: /kaggle/working/predictions/R1107/relax
[DRfold2] Step 1/4: GENERATING INITIAL PREDICTIONS
[DRfold2] No previous predictions found, will generate e2e and geo files
[DRfold2] Running model 1/1: cfg_97
[DRfold2] Command: python /kaggle/working/DRfold2/cfg_97/test_modeldir.py cuda /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/rets_dir/cfg_97_ /kaggle/working/DRfold2/model_hub/cfg_97
[DRfold2 subprocess] /kaggle/working/DRfold2/cfg_97/EvoMSA2XYZ.py:29: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
[DRfold2 subprocess] RNAlm.load_state_dict(torch.load(saved_model,map_location=torch.device('cpu')),strict=False)
[DRfold2 subprocess] /kaggle/working/DRfold2/cfg_97/test_modeldir.py:82: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
[DRfold2 subprocess] model.load_state_dict(torch.load(saved_model,map_location='cpu'),strict=True)
[DRfold2 subprocess] /usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
[DRfold2 subprocess] return fn(*args, **kwargs)
[DRfold2 subprocess] /usr/local/lib/python3.10/dist-packages/torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
[DRfold2 subprocess] warnings.warn(
[DRfold2 subprocess] /kaggle/working/DRfold2/cfg_97/test_modeldir.py:82: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
[DRfold2 subprocess] model.load_state_dict(torch.load(saved_model,map_location='cpu'),strict=True)
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2 subprocess] will do checkpoint
[DRfold2] Running model 1/1: cfg_97 completed successfully
[DRfold2] Initial predictions generation completed
[DRfold2] Step 2/4: SELECTION PROCESS
[DRfold2] Found 20 return files for selection
[DRfold2] Using selection config: /kaggle/working/DRfold2/cfg_for_selection.json
[DRfold2] Output prefix: /kaggle/working/predictions/R1107/folds/sel_0
[DRfold2] Running selection process
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Selection.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/DRfold2/cfg_for_selection.json /kaggle/working/predictions/R1107/folds/sel_0 /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_8.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_7.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_6.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_17.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_15.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_11.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_5.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_12.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_9.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_4.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_10.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_0.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_3.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_16.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_18.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_19.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_13.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_2.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_14.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_1.ret
[DRfold2 subprocess] ['/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_0.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_1.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_10.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_11.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_12.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_13.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_14.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_15.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_16.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_17.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_18.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_19.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_2.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_3.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_4.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_5.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_6.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_7.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_8.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_9.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] tensor(5849694.2115, device='cuda:0', dtype=torch.float64) tensor(-2803293.3324, device='cuda:0', dtype=torch.float64) tensor(-2858034.8373, device='cuda:0', dtype=torch.float64) tensor(-2921733.6686, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(20341768.3910, device='cuda:0', dtype=torch.float64) tensor(-2371068.8110, device='cuda:0', dtype=torch.float64) tensor(-2428500.7218, device='cuda:0', dtype=torch.float64) tensor(-2465429.8469, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5479323.5857, device='cuda:0', dtype=torch.float64) tensor(-2782459.9783, device='cuda:0', dtype=torch.float64) tensor(-2868038.3519, device='cuda:0', dtype=torch.float64) tensor(-2898925.4775, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(6276773.0182, device='cuda:0', dtype=torch.float64) tensor(-2726689.0302, device='cuda:0', dtype=torch.float64) tensor(-2794574.8153, device='cuda:0', dtype=torch.float64) tensor(-2825637.7119, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(4827938.5040, device='cuda:0', dtype=torch.float64) tensor(-2777073.9430, device='cuda:0', dtype=torch.float64) tensor(-2842167.0904, device='cuda:0', dtype=torch.float64) tensor(-2849716.1989, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(6489278.2558, device='cuda:0', dtype=torch.float64) tensor(-2793275.2725, device='cuda:0', dtype=torch.float64) tensor(-2839365.5508, device='cuda:0', dtype=torch.float64) tensor(-2886340.7823, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(3573858.6280, device='cuda:0', dtype=torch.float64) tensor(-2412042.3027, device='cuda:0', dtype=torch.float64) tensor(-2453522.7335, device='cuda:0', dtype=torch.float64) tensor(-2369783.8385, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(6263521.2651, device='cuda:0', dtype=torch.float64) tensor(-2352533.5391, device='cuda:0', dtype=torch.float64) tensor(-2359883.4984, device='cuda:0', dtype=torch.float64) tensor(-2270981.0446, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(4864761.6621, device='cuda:0', dtype=torch.float64) tensor(-2785915.1236, device='cuda:0', dtype=torch.float64) tensor(-2859027.8392, device='cuda:0', dtype=torch.float64) tensor(-2868146.7559, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5266207.3347, device='cuda:0', dtype=torch.float64) tensor(-2772383.9560, device='cuda:0', dtype=torch.float64) tensor(-2853933.7449, device='cuda:0', dtype=torch.float64) tensor(-2878039.2460, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(6962225.5211, device='cuda:0', dtype=torch.float64) tensor(-2803397.4924, device='cuda:0', dtype=torch.float64) tensor(-2866454.7886, device='cuda:0', dtype=torch.float64) tensor(-2887349.5492, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5210858.0549, device='cuda:0', dtype=torch.float64) tensor(-2687906.3337, device='cuda:0', dtype=torch.float64) tensor(-2724266.4154, device='cuda:0', dtype=torch.float64) tensor(-2721671.8300, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(14250387.5722, device='cuda:0', dtype=torch.float64) tensor(-2363109.4649, device='cuda:0', dtype=torch.float64) tensor(-2430869.8963, device='cuda:0', dtype=torch.float64) tensor(-2480844.0534, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(10289233.6623, device='cuda:0', dtype=torch.float64) tensor(-2330554.2520, device='cuda:0', dtype=torch.float64) tensor(-2405244.7660, device='cuda:0', dtype=torch.float64) tensor(-2399747.3550, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(4112191.3527, device='cuda:0', dtype=torch.float64) tensor(-2493970.2779, device='cuda:0', dtype=torch.float64) tensor(-2550606.2047, device='cuda:0', dtype=torch.float64) tensor(-2484517.0774, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(12735487.3853, device='cuda:0', dtype=torch.float64) tensor(-2416415.5182, device='cuda:0', dtype=torch.float64) tensor(-2490090.3246, device='cuda:0', dtype=torch.float64) tensor(-2543002.5823, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5600400.5311, device='cuda:0', dtype=torch.float64) tensor(-2502184.2687, device='cuda:0', dtype=torch.float64) tensor(-2533210.9054, device='cuda:0', dtype=torch.float64) tensor(-2508011.4489, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(7018702.8274, device='cuda:0', dtype=torch.float64) tensor(-2550181.3734, device='cuda:0', dtype=torch.float64) tensor(-2572791.2878, device='cuda:0', dtype=torch.float64) tensor(-2526808.7734, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5906748.2398, device='cuda:0', dtype=torch.float64) tensor(-2661486.1493, device='cuda:0', dtype=torch.float64) tensor(-2693536.3099, device='cuda:0', dtype=torch.float64) tensor(-2706351.0604, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(5257158.4274, device='cuda:0', dtype=torch.float64) tensor(-2341969.2709, device='cuda:0', dtype=torch.float64) tensor(-2406362.6079, device='cuda:0', dtype=torch.float64) tensor(-2403095.4293, device='cuda:0', dtype=torch.float64)
[DRfold2] Running selection process completed successfully
[DRfold2] Step 3/4: OPTIMIZATION PROCESS
[DRfold2] Using fold config: /kaggle/working/DRfold2/cfg_for_folding.json
[DRfold2] Optimization output prefix: /kaggle/working/predictions/R1107/folds/opt_0
[DRfold2] Running optimization process
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Optimization.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/folds/opt_0 /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/sel_0 /kaggle/working/DRfold2/cfg_for_folding.json /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] Before sort: ['cfg_97_model_14.ret', 'cfg_97_model_16.ret', 'cfg_97_model_12.ret', 'cfg_97_model_4.ret', 'cfg_97_model_17.ret']
[DRfold2 subprocess] After sort: ['cfg_97_model_12.ret', 'cfg_97_model_14.ret', 'cfg_97_model_16.ret', 'cfg_97_model_17.ret', 'cfg_97_model_4.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] [DRfold2] Loading AlphaFold3 prediction from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Loaded AlphaFold3 structure from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] AlphaFold3 alignment initialized (shape: torch.Size([69, 69, 3, 3]))
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2] Running optimization process completed successfully
[DRfold2] Step 4/4: STRUCTURE REFINEMENT
[DRfold2] Found optimized structure: /kaggle/working/predictions/R1107/folds/opt_0_from_cfg_97_model_14.ret.pdb
[DRfold2] Final output will be saved to: /kaggle/working/predictions/R1107/relax/model_1.pdb
[DRfold2] Running structure refinement
[DRfold2] Command: /kaggle/working/DRfold2/Arena/Arena /kaggle/working/predictions/R1107/folds/opt_0_from_cfg_97_model_14.ret.pdb /kaggle/working/predictions/R1107/relax/model_1.pdb 7
[DRfold2 subprocess] G1 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G1  with 15 missing atoms
[DRfold2 subprocess] G2 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G2  with 15 missing atoms
[DRfold2 subprocess] G3 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G3  with 15 missing atoms
[DRfold2 subprocess] G4 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G4  with 15 missing atoms
[DRfold2 subprocess] G5 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G5  with 15 missing atoms
[DRfold2 subprocess] C6 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C6  with 12 missing atoms
[DRfold2 subprocess] C7 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C7  with 12 missing atoms
[DRfold2 subprocess] A8 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A8  with 14 missing atoms
[DRfold2 subprocess] C9 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C9  with 12 missing atoms
[DRfold2 subprocess] A10 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A10  with 14 missing atoms
[DRfold2 subprocess] G11 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G11  with 15 missing atoms
[DRfold2 subprocess] C12 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C12  with 12 missing atoms
[DRfold2 subprocess] A13 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A13  with 14 missing atoms
[DRfold2 subprocess] G14 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G14  with 15 missing atoms
[DRfold2 subprocess] A15 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A15  with 14 missing atoms
[DRfold2 subprocess] A16 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A16  with 14 missing atoms
[DRfold2 subprocess] G17 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G17  with 15 missing atoms
[DRfold2 subprocess] C18 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C18  with 12 missing atoms
[DRfold2 subprocess] G19 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G19  with 15 missing atoms
[DRfold2 subprocess] U20 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U20  with 12 missing atoms
[DRfold2 subprocess] U21 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U21  with 12 missing atoms
[DRfold2 subprocess] C22 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C22  with 12 missing atoms
[DRfold2 subprocess] A23 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A23  with 14 missing atoms
[DRfold2 subprocess] C24 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C24  with 12 missing atoms
[DRfold2 subprocess] G25 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G25  with 15 missing atoms
[DRfold2 subprocess] U26 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U26  with 12 missing atoms
[DRfold2 subprocess] C27 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C27  with 12 missing atoms
[DRfold2 subprocess] G28 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G28  with 15 missing atoms
[DRfold2 subprocess] C29 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C29  with 12 missing atoms
[DRfold2 subprocess] A30 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A30  with 14 missing atoms
[DRfold2 subprocess] G31 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G31  with 15 missing atoms
[DRfold2 subprocess] C32 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C32  with 12 missing atoms
[DRfold2 subprocess] C33 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C33  with 12 missing atoms
[DRfold2 subprocess] C34 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C34  with 12 missing atoms
[DRfold2 subprocess] C35 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C35  with 12 missing atoms
[DRfold2 subprocess] U36 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U36  with 12 missing atoms
[DRfold2 subprocess] G37 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G37  with 15 missing atoms
[DRfold2 subprocess] U38 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U38  with 12 missing atoms
[DRfold2 subprocess] C39 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C39  with 12 missing atoms
[DRfold2 subprocess] A40 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A40  with 14 missing atoms
[DRfold2 subprocess] G41 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G41  with 15 missing atoms
[DRfold2 subprocess] C42 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C42  with 12 missing atoms
[DRfold2 subprocess] C43 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C43  with 12 missing atoms
[DRfold2 subprocess] A44 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A44  with 14 missing atoms
[DRfold2 subprocess] U45 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U45  with 12 missing atoms
[DRfold2 subprocess] U46 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U46  with 12 missing atoms
[DRfold2 subprocess] G47 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G47  with 15 missing atoms
[DRfold2 subprocess] C48 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C48  with 12 missing atoms
[DRfold2 subprocess] A49 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A49  with 14 missing atoms
[DRfold2 subprocess] C50 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C50  with 12 missing atoms
[DRfold2 subprocess] U51 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U51  with 12 missing atoms
[DRfold2 subprocess] C52 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C52  with 12 missing atoms
[DRfold2 subprocess] C53 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C53  with 12 missing atoms
[DRfold2 subprocess] G54 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G54  with 15 missing atoms
[DRfold2 subprocess] G55 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G55  with 15 missing atoms
[DRfold2 subprocess] C56 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C56  with 12 missing atoms
[DRfold2 subprocess] U57 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U57  with 12 missing atoms
[DRfold2 subprocess] G58 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G58  with 15 missing atoms
[DRfold2 subprocess] C59 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C59  with 12 missing atoms
[DRfold2 subprocess] G60 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G60  with 15 missing atoms
[DRfold2 subprocess] A61 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A61  with 14 missing atoms
[DRfold2 subprocess] A62 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A62  with 14 missing atoms
[DRfold2 subprocess] U63 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U63  with 12 missing atoms
[DRfold2 subprocess] U64 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U64  with 12 missing atoms
[DRfold2 subprocess] C65 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C65  with 12 missing atoms
[DRfold2 subprocess] U66 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U66  with 12 missing atoms
[DRfold2 subprocess] G67 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G67  with 15 missing atoms
[DRfold2 subprocess] C68 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C68  with 12 missing atoms
[DRfold2 subprocess] U69 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U69  with 12 missing atoms
[DRfold2 subprocess] t=0 moved=1418
[DRfold2 subprocess] t=1 moved=1879
[DRfold2 subprocess] t=2 moved=1733
[DRfold2 subprocess] t=3 moved=1678
[DRfold2 subprocess] t=4 moved=1630
[DRfold2 subprocess] t=5 moved=1602
[DRfold2 subprocess] t=6 moved=1580
[DRfold2 subprocess] t=7 moved=1544
[DRfold2 subprocess] t=8 moved=1529
[DRfold2 subprocess] t=9 moved=1532
[DRfold2 subprocess] t=10 moved=1511
[DRfold2 subprocess] t=11 moved=1497
[DRfold2 subprocess] t=12 moved=1518
[DRfold2 subprocess] t=13 moved=1496
[DRfold2 subprocess] t=14 moved=1489
[DRfold2 subprocess] t=15 moved=1484
[DRfold2 subprocess] t=16 moved=1472
[DRfold2 subprocess] t=17 moved=1474
[DRfold2 subprocess] t=18 moved=1491
[DRfold2 subprocess] t=19 moved=1474
[DRfold2 subprocess] t=20 moved=1483
[DRfold2 subprocess] t=21 moved=1463
[DRfold2 subprocess] t=22 moved=1479
[DRfold2 subprocess] t=23 moved=1457
[DRfold2 subprocess] t=24 moved=1463
[DRfold2 subprocess] t=25 moved=1449
[DRfold2 subprocess] t=26 moved=1476
[DRfold2 subprocess] t=27 moved=1464
[DRfold2 subprocess] t=28 moved=1458
[DRfold2 subprocess] t=29 moved=1456
[DRfold2 subprocess] t=30 moved=1452
[DRfold2 subprocess] t=31 moved=1435
[DRfold2 subprocess] t=32 moved=1446
[DRfold2 subprocess] t=33 moved=1440
[DRfold2 subprocess] t=34 moved=1447
[DRfold2 subprocess] t=35 moved=1458
[DRfold2 subprocess] t=36 moved=1445
[DRfold2 subprocess] t=37 moved=1462
[DRfold2 subprocess] t=38 moved=1443
[DRfold2 subprocess] t=39 moved=1442
[DRfold2 subprocess] t=40 moved=1447
[DRfold2 subprocess] t=41 moved=1441
[DRfold2 subprocess] t=42 moved=1442
[DRfold2 subprocess] t=43 moved=1448
[DRfold2 subprocess] t=44 moved=1445
[DRfold2 subprocess] t=45 moved=1432
[DRfold2 subprocess] t=46 moved=1431
[DRfold2 subprocess] t=47 moved=1438
[DRfold2 subprocess] t=48 moved=1430
[DRfold2 subprocess] t=49 moved=1434
[DRfold2 subprocess] t=50 moved=1438
[DRfold2 subprocess] t=51 moved=1439
[DRfold2 subprocess] t=52 moved=1414
[DRfold2 subprocess] t=53 moved=1437
[DRfold2 subprocess] t=54 moved=1441
[DRfold2 subprocess] t=55 moved=1439
[DRfold2 subprocess] t=56 moved=1432
[DRfold2 subprocess] t=57 moved=1433
[DRfold2 subprocess] t=58 moved=1444
[DRfold2 subprocess] t=59 moved=1439
[DRfold2 subprocess] t=60 moved=1439
[DRfold2 subprocess] t=61 moved=1431
[DRfold2 subprocess] t=62 moved=1438
[DRfold2 subprocess] t=63 moved=1419
[DRfold2 subprocess] t=64 moved=1409
[DRfold2 subprocess] t=65 moved=1442
[DRfold2 subprocess] t=66 moved=1413
[DRfold2 subprocess] t=67 moved=1432
[DRfold2 subprocess] t=68 moved=1421
[DRfold2 subprocess] t=69 moved=1424
[DRfold2 subprocess] t=70 moved=1423
[DRfold2 subprocess] t=71 moved=1439
[DRfold2 subprocess] t=72 moved=1447
[DRfold2 subprocess] t=73 moved=1444
[DRfold2 subprocess] t=74 moved=1442
[DRfold2 subprocess] t=75 moved=1440
[DRfold2 subprocess] t=76 moved=1426
[DRfold2 subprocess] t=77 moved=1426
[DRfold2 subprocess] t=78 moved=1418
[DRfold2 subprocess] t=79 moved=1404
[DRfold2 subprocess] t=80 moved=1426
[DRfold2 subprocess] t=81 moved=1420
[DRfold2 subprocess] t=82 moved=1409
[DRfold2 subprocess] t=83 moved=1432
[DRfold2 subprocess] t=84 moved=1422
[DRfold2 subprocess] t=85 moved=1413
[DRfold2 subprocess] t=86 moved=1413
[DRfold2 subprocess] t=87 moved=1419
[DRfold2 subprocess] t=88 moved=1409
[DRfold2 subprocess] t=89 moved=1439
[DRfold2 subprocess] t=90 moved=1410
[DRfold2 subprocess] t=91 moved=1421
[DRfold2 subprocess] t=92 moved=1399
[DRfold2 subprocess] t=93 moved=1420
[DRfold2 subprocess] t=94 moved=1412
[DRfold2 subprocess] t=95 moved=1410
[DRfold2 subprocess] t=96 moved=1407
[DRfold2 subprocess] t=97 moved=1396
[DRfold2 subprocess] t=98 moved=1395
[DRfold2 subprocess] t=99 moved=1406
[DRfold2 subprocess] t=100 moved=1408
[DRfold2 subprocess] t=101 moved=1401
[DRfold2 subprocess] t=102 moved=1391
[DRfold2 subprocess] stage1 t=0 moved=1485
[DRfold2 subprocess] stage1 t=1 moved=1419
[DRfold2 subprocess] stage1 t=2 moved=1192
[DRfold2 subprocess] stage1 t=3 moved=1008
[DRfold2 subprocess] stage1 t=4 moved=917
[DRfold2 subprocess] stage1 t=5 moved=739
[DRfold2 subprocess] stage1 t=6 moved=674
[DRfold2 subprocess] stage1 t=7 moved=557
[DRfold2 subprocess] stage1 t=8 moved=481
[DRfold2 subprocess] stage1 t=9 moved=468
[DRfold2 subprocess] stage1 t=10 moved=426
[DRfold2 subprocess] stage1 t=11 moved=349
[DRfold2 subprocess] stage1 t=12 moved=375
[DRfold2 subprocess] stage1 t=13 moved=346
[DRfold2 subprocess] stage1 t=14 moved=281
[DRfold2 subprocess] stage1 t=15 moved=299
[DRfold2 subprocess] stage1 t=16 moved=239
[DRfold2 subprocess] stage1 t=17 moved=229
[DRfold2 subprocess] stage1 t=18 moved=231
[DRfold2 subprocess] stage1 t=19 moved=263
[DRfold2 subprocess] stage1 t=20 moved=278
[DRfold2 subprocess] stage1 t=21 moved=255
[DRfold2 subprocess] stage1 t=22 moved=186
[DRfold2 subprocess] stage1 t=23 moved=162
[DRfold2 subprocess] stage1 t=24 moved=158
[DRfold2 subprocess] stage1 t=25 moved=136
[DRfold2 subprocess] stage1 t=26 moved=111
[DRfold2 subprocess] stage1 t=27 moved=74
[DRfold2 subprocess] stage1 t=28 moved=78
[DRfold2 subprocess] stage1 t=29 moved=62
[DRfold2 subprocess] stage1 t=30 moved=66
[DRfold2 subprocess] stage1 t=31 moved=44
[DRfold2 subprocess] stage1 t=32 moved=69
[DRfold2 subprocess] stage1 t=33 moved=98
[DRfold2 subprocess] stage1 t=34 moved=134
[DRfold2 subprocess] stage1 t=35 moved=84
[DRfold2 subprocess] stage1 t=36 moved=74
[DRfold2 subprocess] stage1 t=37 moved=94
[DRfold2 subprocess] stage1 t=38 moved=86
[DRfold2 subprocess] stage1 t=39 moved=37
[DRfold2 subprocess] stage1 t=40 moved=58
[DRfold2 subprocess] stage1 t=41 moved=77
[DRfold2 subprocess] stage1 t=42 moved=63
[DRfold2 subprocess] stage1 t=43 moved=54
[DRfold2 subprocess] stage1 t=44 moved=42
[DRfold2 subprocess] stage1 t=45 moved=26
[DRfold2 subprocess] stage1 t=46 moved=43
[DRfold2 subprocess] stage1 t=47 moved=68
[DRfold2 subprocess] stage1 t=48 moved=78
[DRfold2 subprocess] stage1 t=49 moved=38
[DRfold2 subprocess] stage1 t=50 moved=55
[DRfold2 subprocess] stage1 t=51 moved=64
[DRfold2 subprocess] stage1 t=52 moved=86
[DRfold2 subprocess] stage1 t=53 moved=70
[DRfold2 subprocess] stage1 t=54 moved=80
[DRfold2 subprocess] stage1 t=55 moved=111
[DRfold2 subprocess] stage1 t=56 moved=68
[DRfold2 subprocess] stage1 t=57 moved=58
[DRfold2 subprocess] stage1 t=58 moved=48
[DRfold2 subprocess] stage1 t=59 moved=62
[DRfold2 subprocess] stage1 t=60 moved=61
[DRfold2 subprocess] stage1 t=61 moved=64
[DRfold2 subprocess] stage1 t=62 moved=64
[DRfold2 subprocess] stage1 t=63 moved=62
[DRfold2 subprocess] stage1 t=64 moved=51
[DRfold2 subprocess] stage1 t=65 moved=37
[DRfold2 subprocess] stage1 t=66 moved=25
[DRfold2 subprocess] stage1 t=67 moved=49
[DRfold2 subprocess] stage1 t=68 moved=63
[DRfold2 subprocess] stage1 t=69 moved=76
[DRfold2 subprocess] stage1 t=70 moved=67
[DRfold2 subprocess] stage1 t=71 moved=38
[DRfold2 subprocess] stage1 t=72 moved=51
[DRfold2 subprocess] stage1 t=73 moved=74
[DRfold2 subprocess] stage1 t=74 moved=64
[DRfold2 subprocess] stage1 t=75 moved=49
[DRfold2 subprocess] stage1 t=76 moved=44
[DRfold2 subprocess] stage1 t=77 moved=29
[DRfold2 subprocess] stage1 t=78 moved=13
[DRfold2 subprocess] stage1 t=79 moved=10
[DRfold2 subprocess] stage1 t=80 moved=10
[DRfold2 subprocess] stage1 t=81 moved=1
[DRfold2 subprocess] stage1 t=82 moved=1
[DRfold2 subprocess] stage1 t=83 moved=0
[DRfold2 subprocess] stage2 t=0 moved=416
[DRfold2 subprocess] stage2 t=1 moved=860
[DRfold2 subprocess] stage2 t=2 moved=759
[DRfold2 subprocess] stage2 t=3 moved=679
[DRfold2 subprocess] stage2 t=4 moved=597
[DRfold2 subprocess] stage2 t=5 moved=418
[DRfold2 subprocess] stage2 t=6 moved=386
[DRfold2 subprocess] stage2 t=7 moved=248
[DRfold2 subprocess] stage2 t=8 moved=255
[DRfold2 subprocess] stage2 t=9 moved=135
[DRfold2 subprocess] stage2 t=10 moved=112
[DRfold2 subprocess] stage2 t=11 moved=137
[DRfold2 subprocess] stage2 t=12 moved=125
[DRfold2 subprocess] stage2 t=13 moved=108
[DRfold2 subprocess] stage2 t=14 moved=101
[DRfold2 subprocess] stage2 t=15 moved=90
[DRfold2 subprocess] stage2 t=16 moved=88
[DRfold2 subprocess] stage2 t=17 moved=43
[DRfold2 subprocess] stage2 t=18 moved=41
[DRfold2 subprocess] stage2 t=19 moved=38
[DRfold2 subprocess] stage2 t=20 moved=42
[DRfold2 subprocess] stage2 t=21 moved=45
[DRfold2 subprocess] stage2 t=22 moved=37
[DRfold2 subprocess] stage2 t=23 moved=45
[DRfold2 subprocess] stage2 t=24 moved=38
[DRfold2 subprocess] stage2 t=25 moved=30
[DRfold2 subprocess] stage2 t=26 moved=33
[DRfold2 subprocess] stage2 t=27 moved=2
[DRfold2 subprocess] stage2 t=28 moved=1
[DRfold2 subprocess] stage2 t=29 moved=1
[DRfold2 subprocess] stage2 t=30 moved=3
[DRfold2 subprocess] stage2 t=31 moved=37
[DRfold2 subprocess] stage2 t=32 moved=3
[DRfold2 subprocess] stage2 t=33 moved=2
[DRfold2 subprocess] stage2 t=34 moved=5
[DRfold2 subprocess] stage2 t=35 moved=2
[DRfold2 subprocess] stage2 t=36 moved=0
[DRfold2 subprocess] stage3 t=0 moved=0
[DRfold2] Running structure refinement completed successfully
[DRfold2] ADDITIONAL STEP: CLUSTERING
[DRfold2] Running clustering process, output: /kaggle/working/predictions/R1107/folds/clu.txt
[DRfold2] Running clustering
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Clust.py /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/clu.txt
[DRfold2 subprocess] 1.7854969501495361 10.003416049166729 19.37909698486328
[DRfold2 subprocess] 10
[DRfold2 subprocess] [ 0  2  3  4  6  8  9 11 14 18] 4.872664589756414 ['cfg_97_model_0.pdb', 'cfg_97_model_10.pdb', 'cfg_97_model_11.pdb', 'cfg_97_model_12.pdb', 'cfg_97_model_14.pdb', 'cfg_97_model_16.pdb', 'cfg_97_model_17.pdb', 'cfg_97_model_19.pdb', 'cfg_97_model_4.pdb', 'cfg_97_model_8.pdb'] 4 cfg_97_model_12.pdb
[DRfold2 subprocess] 7
[DRfold2 subprocess] [ 5 10 17] 4.872664589756414 ['cfg_97_model_13.pdb', 'cfg_97_model_18.pdb', 'cfg_97_model_7.pdb'] 5 cfg_97_model_13.pdb
[DRfold2 subprocess] 4
[DRfold2 subprocess] [ 7 16 19] 7.072664589756416 ['cfg_97_model_15.pdb', 'cfg_97_model_6.pdb', 'cfg_97_model_9.pdb'] 16 cfg_97_model_6.pdb
[DRfold2 subprocess] 2
[DRfold2 subprocess] [ 1 12] 11.472664589756404 ['cfg_97_model_1.pdb', 'cfg_97_model_2.pdb'] 1 cfg_97_model_1.pdb
[DRfold2 subprocess] 1
[DRfold2 subprocess] [13] 11.472664589756404 ['cfg_97_model_3.pdb'] 13 cfg_97_model_3.pdb
[DRfold2] Running clustering completed successfully
[DRfold2] Found 4 additional clusters to process
[DRfold2] PROCESSING CLUSTER 1/4
[DRfold2] Cluster 1 Selection Process
[DRfold2] Found 3 return files for selection
[DRfold2] Selection output prefix: /kaggle/working/predictions/R1107/folds/sel_2
[DRfold2] Running selection for cluster 1
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Selection.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/DRfold2/cfg_for_selection.json /kaggle/working/predictions/R1107/folds/sel_2 /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_13.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_18.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_7.ret
[DRfold2 subprocess] ['/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_13.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_18.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_7.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] tensor(34353.1532, device='cuda:0', dtype=torch.float64) tensor(-3422272.2777, device='cuda:0', dtype=torch.float64) tensor(-3548962.8729, device='cuda:0', dtype=torch.float64) tensor(-3593604.9495, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(53993.8701, device='cuda:0', dtype=torch.float64) tensor(-3420810.7508, device='cuda:0', dtype=torch.float64) tensor(-3582082.1108, device='cuda:0', dtype=torch.float64) tensor(-3586600.1847, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(1168483.9673, device='cuda:0', dtype=torch.float64) tensor(-3086369.4216, device='cuda:0', dtype=torch.float64) tensor(-3203755.9998, device='cuda:0', dtype=torch.float64) tensor(-3117729.6608, device='cuda:0', dtype=torch.float64)
[DRfold2] Running selection for cluster 1 completed successfully
[DRfold2] Cluster 1 Optimization Process
[DRfold2] Optimization output prefix: /kaggle/working/predictions/R1107/folds/opt_2
[DRfold2] Running optimization for cluster 1
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Optimization.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/folds/opt_2 /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/sel_2 /kaggle/working/DRfold2/cfg_for_folding.json /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] Before sort: ['cfg_97_model_18.ret', 'cfg_97_model_13.ret', 'cfg_97_model_7.ret']
[DRfold2 subprocess] After sort: ['cfg_97_model_13.ret', 'cfg_97_model_18.ret', 'cfg_97_model_7.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] [DRfold2] Loading AlphaFold3 prediction from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Loaded AlphaFold3 structure from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] AlphaFold3 alignment initialized (shape: torch.Size([69, 69, 3, 3]))
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2] Running optimization for cluster 1 completed successfully
[DRfold2] Cluster 1 Refinement Process
[DRfold2] Found optimized structure: /kaggle/working/predictions/R1107/folds/opt_2_from_cfg_97_model_18.ret.pdb
[DRfold2] Final output will be saved to: /kaggle/working/predictions/R1107/relax/model_2.pdb
[DRfold2] Running refinement for cluster 1
[DRfold2] Command: /kaggle/working/DRfold2/Arena/Arena /kaggle/working/predictions/R1107/folds/opt_2_from_cfg_97_model_18.ret.pdb /kaggle/working/predictions/R1107/relax/model_2.pdb 7
[DRfold2 subprocess] G1 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G1  with 15 missing atoms
[DRfold2 subprocess] G2 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G2  with 15 missing atoms
[DRfold2 subprocess] G3 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G3  with 15 missing atoms
[DRfold2 subprocess] G4 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G4  with 15 missing atoms
[DRfold2 subprocess] G5 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G5  with 15 missing atoms
[DRfold2 subprocess] C6 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C6  with 12 missing atoms
[DRfold2 subprocess] C7 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C7  with 12 missing atoms
[DRfold2 subprocess] A8 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A8  with 14 missing atoms
[DRfold2 subprocess] C9 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C9  with 12 missing atoms
[DRfold2 subprocess] A10 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A10  with 14 missing atoms
[DRfold2 subprocess] G11 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G11  with 15 missing atoms
[DRfold2 subprocess] C12 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C12  with 12 missing atoms
[DRfold2 subprocess] A13 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A13  with 14 missing atoms
[DRfold2 subprocess] G14 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G14  with 15 missing atoms
[DRfold2 subprocess] A15 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A15  with 14 missing atoms
[DRfold2 subprocess] A16 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A16  with 14 missing atoms
[DRfold2 subprocess] G17 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G17  with 15 missing atoms
[DRfold2 subprocess] C18 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C18  with 12 missing atoms
[DRfold2 subprocess] G19 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G19  with 15 missing atoms
[DRfold2 subprocess] U20 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U20  with 12 missing atoms
[DRfold2 subprocess] U21 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U21  with 12 missing atoms
[DRfold2 subprocess] C22 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C22  with 12 missing atoms
[DRfold2 subprocess] A23 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A23  with 14 missing atoms
[DRfold2 subprocess] C24 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C24  with 12 missing atoms
[DRfold2 subprocess] G25 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G25  with 15 missing atoms
[DRfold2 subprocess] U26 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U26  with 12 missing atoms
[DRfold2 subprocess] C27 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C27  with 12 missing atoms
[DRfold2 subprocess] G28 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G28  with 15 missing atoms
[DRfold2 subprocess] C29 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C29  with 12 missing atoms
[DRfold2 subprocess] A30 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A30  with 14 missing atoms
[DRfold2 subprocess] G31 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G31  with 15 missing atoms
[DRfold2 subprocess] C32 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C32  with 12 missing atoms
[DRfold2 subprocess] C33 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C33  with 12 missing atoms
[DRfold2 subprocess] C34 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C34  with 12 missing atoms
[DRfold2 subprocess] C35 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C35  with 12 missing atoms
[DRfold2 subprocess] U36 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U36  with 12 missing atoms
[DRfold2 subprocess] G37 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G37  with 15 missing atoms
[DRfold2 subprocess] U38 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U38  with 12 missing atoms
[DRfold2 subprocess] C39 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C39  with 12 missing atoms
[DRfold2 subprocess] A40 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A40  with 14 missing atoms
[DRfold2 subprocess] G41 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G41  with 15 missing atoms
[DRfold2 subprocess] C42 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C42  with 12 missing atoms
[DRfold2 subprocess] C43 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C43  with 12 missing atoms
[DRfold2 subprocess] A44 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A44  with 14 missing atoms
[DRfold2 subprocess] U45 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U45  with 12 missing atoms
[DRfold2 subprocess] U46 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U46  with 12 missing atoms
[DRfold2 subprocess] G47 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G47  with 15 missing atoms
[DRfold2 subprocess] C48 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C48  with 12 missing atoms
[DRfold2 subprocess] A49 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A49  with 14 missing atoms
[DRfold2 subprocess] C50 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C50  with 12 missing atoms
[DRfold2 subprocess] U51 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U51  with 12 missing atoms
[DRfold2 subprocess] C52 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C52  with 12 missing atoms
[DRfold2 subprocess] C53 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C53  with 12 missing atoms
[DRfold2 subprocess] G54 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G54  with 15 missing atoms
[DRfold2 subprocess] G55 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G55  with 15 missing atoms
[DRfold2 subprocess] C56 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C56  with 12 missing atoms
[DRfold2 subprocess] U57 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U57  with 12 missing atoms
[DRfold2 subprocess] G58 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G58  with 15 missing atoms
[DRfold2 subprocess] C59 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C59  with 12 missing atoms
[DRfold2 subprocess] G60 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G60  with 15 missing atoms
[DRfold2 subprocess] A61 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A61  with 14 missing atoms
[DRfold2 subprocess] A62 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A62  with 14 missing atoms
[DRfold2 subprocess] U63 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U63  with 12 missing atoms
[DRfold2 subprocess] U64 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U64  with 12 missing atoms
[DRfold2 subprocess] C65 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C65  with 12 missing atoms
[DRfold2 subprocess] U66 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U66  with 12 missing atoms
[DRfold2 subprocess] G67 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G67  with 15 missing atoms
[DRfold2 subprocess] C68 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C68  with 12 missing atoms
[DRfold2 subprocess] U69 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U69  with 12 missing atoms
[DRfold2 subprocess] t=0 moved=1442
[DRfold2 subprocess] t=1 moved=1821
[DRfold2 subprocess] t=2 moved=1690
[DRfold2 subprocess] t=3 moved=1652
[DRfold2 subprocess] t=4 moved=1615
[DRfold2 subprocess] t=5 moved=1584
[DRfold2 subprocess] t=6 moved=1563
[DRfold2 subprocess] t=7 moved=1536
[DRfold2 subprocess] t=8 moved=1530
[DRfold2 subprocess] t=9 moved=1493
[DRfold2 subprocess] t=10 moved=1491
[DRfold2 subprocess] t=11 moved=1472
[DRfold2 subprocess] t=12 moved=1477
[DRfold2 subprocess] t=13 moved=1448
[DRfold2 subprocess] t=14 moved=1456
[DRfold2 subprocess] t=15 moved=1454
[DRfold2 subprocess] t=16 moved=1425
[DRfold2 subprocess] t=17 moved=1422
[DRfold2 subprocess] t=18 moved=1418
[DRfold2 subprocess] t=19 moved=1421
[DRfold2 subprocess] t=20 moved=1408
[DRfold2 subprocess] t=21 moved=1412
[DRfold2 subprocess] t=22 moved=1422
[DRfold2 subprocess] t=23 moved=1421
[DRfold2 subprocess] t=24 moved=1427
[DRfold2 subprocess] t=25 moved=1414
[DRfold2 subprocess] t=26 moved=1428
[DRfold2 subprocess] t=27 moved=1437
[DRfold2 subprocess] t=28 moved=1413
[DRfold2 subprocess] t=29 moved=1417
[DRfold2 subprocess] t=30 moved=1407
[DRfold2 subprocess] t=31 moved=1408
[DRfold2 subprocess] t=32 moved=1388
[DRfold2 subprocess] t=33 moved=1407
[DRfold2 subprocess] t=34 moved=1385
[DRfold2 subprocess] t=35 moved=1385
[DRfold2 subprocess] t=36 moved=1376
[DRfold2 subprocess] t=37 moved=1399
[DRfold2 subprocess] t=38 moved=1386
[DRfold2 subprocess] t=39 moved=1390
[DRfold2 subprocess] t=40 moved=1396
[DRfold2 subprocess] t=41 moved=1374
[DRfold2 subprocess] t=42 moved=1377
[DRfold2 subprocess] t=43 moved=1391
[DRfold2 subprocess] t=44 moved=1390
[DRfold2 subprocess] t=45 moved=1381
[DRfold2 subprocess] t=46 moved=1378
[DRfold2 subprocess] t=47 moved=1385
[DRfold2 subprocess] t=48 moved=1382
[DRfold2 subprocess] t=49 moved=1374
[DRfold2 subprocess] t=50 moved=1370
[DRfold2 subprocess] t=51 moved=1373
[DRfold2 subprocess] t=52 moved=1381
[DRfold2 subprocess] t=53 moved=1389
[DRfold2 subprocess] t=54 moved=1381
[DRfold2 subprocess] t=55 moved=1375
[DRfold2 subprocess] t=56 moved=1384
[DRfold2 subprocess] t=57 moved=1373
[DRfold2 subprocess] t=58 moved=1384
[DRfold2 subprocess] t=59 moved=1371
[DRfold2 subprocess] t=60 moved=1365
[DRfold2 subprocess] t=61 moved=1365
[DRfold2 subprocess] t=62 moved=1372
[DRfold2 subprocess] t=63 moved=1361
[DRfold2 subprocess] t=64 moved=1387
[DRfold2 subprocess] t=65 moved=1364
[DRfold2 subprocess] t=66 moved=1366
[DRfold2 subprocess] t=67 moved=1379
[DRfold2 subprocess] t=68 moved=1365
[DRfold2 subprocess] t=69 moved=1359
[DRfold2 subprocess] t=70 moved=1375
[DRfold2 subprocess] t=71 moved=1370
[DRfold2 subprocess] t=72 moved=1362
[DRfold2 subprocess] t=73 moved=1352
[DRfold2 subprocess] t=74 moved=1364
[DRfold2 subprocess] t=75 moved=1366
[DRfold2 subprocess] t=76 moved=1368
[DRfold2 subprocess] t=77 moved=1365
[DRfold2 subprocess] t=78 moved=1367
[DRfold2 subprocess] t=79 moved=1361
[DRfold2 subprocess] t=80 moved=1358
[DRfold2 subprocess] t=81 moved=1362
[DRfold2 subprocess] t=82 moved=1374
[DRfold2 subprocess] t=83 moved=1356
[DRfold2 subprocess] t=84 moved=1372
[DRfold2 subprocess] t=85 moved=1369
[DRfold2 subprocess] t=86 moved=1363
[DRfold2 subprocess] t=87 moved=1360
[DRfold2 subprocess] t=88 moved=1354
[DRfold2 subprocess] t=89 moved=1372
[DRfold2 subprocess] t=90 moved=1375
[DRfold2 subprocess] t=91 moved=1372
[DRfold2 subprocess] t=92 moved=1365
[DRfold2 subprocess] t=93 moved=1373
[DRfold2 subprocess] t=94 moved=1376
[DRfold2 subprocess] t=95 moved=1373
[DRfold2 subprocess] t=96 moved=1382
[DRfold2 subprocess] t=97 moved=1363
[DRfold2 subprocess] t=98 moved=1367
[DRfold2 subprocess] t=99 moved=1367
[DRfold2 subprocess] t=100 moved=1377
[DRfold2 subprocess] t=101 moved=1366
[DRfold2 subprocess] t=102 moved=1398
[DRfold2 subprocess] stage1 t=0 moved=1442
[DRfold2 subprocess] stage1 t=1 moved=1298
[DRfold2 subprocess] stage1 t=2 moved=1025
[DRfold2 subprocess] stage1 t=3 moved=874
[DRfold2 subprocess] stage1 t=4 moved=847
[DRfold2 subprocess] stage1 t=5 moved=706
[DRfold2 subprocess] stage1 t=6 moved=589
[DRfold2 subprocess] stage1 t=7 moved=553
[DRfold2 subprocess] stage1 t=8 moved=467
[DRfold2 subprocess] stage1 t=9 moved=467
[DRfold2 subprocess] stage1 t=10 moved=404
[DRfold2 subprocess] stage1 t=11 moved=410
[DRfold2 subprocess] stage1 t=12 moved=338
[DRfold2 subprocess] stage1 t=13 moved=352
[DRfold2 subprocess] stage1 t=14 moved=343
[DRfold2 subprocess] stage1 t=15 moved=257
[DRfold2 subprocess] stage1 t=16 moved=233
[DRfold2 subprocess] stage1 t=17 moved=238
[DRfold2 subprocess] stage1 t=18 moved=228
[DRfold2 subprocess] stage1 t=19 moved=203
[DRfold2 subprocess] stage1 t=20 moved=224
[DRfold2 subprocess] stage1 t=21 moved=184
[DRfold2 subprocess] stage1 t=22 moved=188
[DRfold2 subprocess] stage1 t=23 moved=187
[DRfold2 subprocess] stage1 t=24 moved=184
[DRfold2 subprocess] stage1 t=25 moved=200
[DRfold2 subprocess] stage1 t=26 moved=200
[DRfold2 subprocess] stage1 t=27 moved=154
[DRfold2 subprocess] stage1 t=28 moved=191
[DRfold2 subprocess] stage1 t=29 moved=140
[DRfold2 subprocess] stage1 t=30 moved=150
[DRfold2 subprocess] stage1 t=31 moved=155
[DRfold2 subprocess] stage1 t=32 moved=139
[DRfold2 subprocess] stage1 t=33 moved=133
[DRfold2 subprocess] stage1 t=34 moved=117
[DRfold2 subprocess] stage1 t=35 moved=105
[DRfold2 subprocess] stage1 t=36 moved=81
[DRfold2 subprocess] stage1 t=37 moved=74
[DRfold2 subprocess] stage1 t=38 moved=94
[DRfold2 subprocess] stage1 t=39 moved=86
[DRfold2 subprocess] stage1 t=40 moved=69
[DRfold2 subprocess] stage1 t=41 moved=73
[DRfold2 subprocess] stage1 t=42 moved=53
[DRfold2 subprocess] stage1 t=43 moved=38
[DRfold2 subprocess] stage1 t=44 moved=36
[DRfold2 subprocess] stage1 t=45 moved=36
[DRfold2 subprocess] stage1 t=46 moved=36
[DRfold2 subprocess] stage1 t=47 moved=34
[DRfold2 subprocess] stage1 t=48 moved=31
[DRfold2 subprocess] stage1 t=49 moved=32
[DRfold2 subprocess] stage1 t=50 moved=30
[DRfold2 subprocess] stage1 t=51 moved=28
[DRfold2 subprocess] stage1 t=52 moved=18
[DRfold2 subprocess] stage1 t=53 moved=45
[DRfold2 subprocess] stage1 t=54 moved=16
[DRfold2 subprocess] stage1 t=55 moved=18
[DRfold2 subprocess] stage1 t=56 moved=35
[DRfold2 subprocess] stage1 t=57 moved=53
[DRfold2 subprocess] stage1 t=58 moved=63
[DRfold2 subprocess] stage1 t=59 moved=68
[DRfold2 subprocess] stage1 t=60 moved=33
[DRfold2 subprocess] stage1 t=61 moved=32
[DRfold2 subprocess] stage1 t=62 moved=34
[DRfold2 subprocess] stage1 t=63 moved=38
[DRfold2 subprocess] stage1 t=64 moved=40
[DRfold2 subprocess] stage1 t=65 moved=42
[DRfold2 subprocess] stage1 t=66 moved=49
[DRfold2 subprocess] stage1 t=67 moved=47
[DRfold2 subprocess] stage1 t=68 moved=50
[DRfold2 subprocess] stage1 t=69 moved=35
[DRfold2 subprocess] stage1 t=70 moved=45
[DRfold2 subprocess] stage1 t=71 moved=27
[DRfold2 subprocess] stage1 t=72 moved=31
[DRfold2 subprocess] stage1 t=73 moved=33
[DRfold2 subprocess] stage1 t=74 moved=33
[DRfold2 subprocess] stage1 t=75 moved=14
[DRfold2 subprocess] stage1 t=76 moved=13
[DRfold2 subprocess] stage1 t=77 moved=13
[DRfold2 subprocess] stage1 t=78 moved=0
[DRfold2 subprocess] stage2 t=0 moved=389
[DRfold2 subprocess] stage2 t=1 moved=798
[DRfold2 subprocess] stage2 t=2 moved=724
[DRfold2 subprocess] stage2 t=3 moved=492
[DRfold2 subprocess] stage2 t=4 moved=461
[DRfold2 subprocess] stage2 t=5 moved=334
[DRfold2 subprocess] stage2 t=6 moved=305
[DRfold2 subprocess] stage2 t=7 moved=249
[DRfold2 subprocess] stage2 t=8 moved=210
[DRfold2 subprocess] stage2 t=9 moved=136
[DRfold2 subprocess] stage2 t=10 moved=119
[DRfold2 subprocess] stage2 t=11 moved=113
[DRfold2 subprocess] stage2 t=12 moved=149
[DRfold2 subprocess] stage2 t=13 moved=42
[DRfold2 subprocess] stage2 t=14 moved=15
[DRfold2 subprocess] stage2 t=15 moved=15
[DRfold2 subprocess] stage2 t=16 moved=33
[DRfold2 subprocess] stage2 t=17 moved=42
[DRfold2 subprocess] stage2 t=18 moved=31
[DRfold2 subprocess] stage2 t=19 moved=19
[DRfold2 subprocess] stage2 t=20 moved=13
[DRfold2 subprocess] stage2 t=21 moved=26
[DRfold2 subprocess] stage2 t=22 moved=27
[DRfold2 subprocess] stage2 t=23 moved=30
[DRfold2 subprocess] stage2 t=24 moved=16
[DRfold2 subprocess] stage2 t=25 moved=4
[DRfold2 subprocess] stage2 t=26 moved=20
[DRfold2 subprocess] stage2 t=27 moved=31
[DRfold2 subprocess] stage2 t=28 moved=29
[DRfold2 subprocess] stage2 t=29 moved=2
[DRfold2 subprocess] stage2 t=30 moved=0
[DRfold2 subprocess] stage3 t=0 moved=0
[DRfold2] Running refinement for cluster 1 completed successfully
[DRfold2] PROCESSING CLUSTER 2/4
[DRfold2] Cluster 2 Selection Process
[DRfold2] Found 3 return files for selection
[DRfold2] Selection output prefix: /kaggle/working/predictions/R1107/folds/sel_3
[DRfold2] Running selection for cluster 2
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Selection.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/DRfold2/cfg_for_selection.json /kaggle/working/predictions/R1107/folds/sel_3 /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_15.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_6.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_9.ret
[DRfold2 subprocess] ['/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_15.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_6.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_9.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] tensor(10468.2365, device='cuda:0', dtype=torch.float64) tensor(-2326278.0637, device='cuda:0', dtype=torch.float64) tensor(-2300849.5442, device='cuda:0', dtype=torch.float64) tensor(-2121421.7094, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(6302.4832, device='cuda:0', dtype=torch.float64) tensor(-2473428.5086, device='cuda:0', dtype=torch.float64) tensor(-2469426.6910, device='cuda:0', dtype=torch.float64) tensor(-2355528.8647, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(29842.4503, device='cuda:0', dtype=torch.float64) tensor(-2260729.7026, device='cuda:0', dtype=torch.float64) tensor(-2294774.3985, device='cuda:0', dtype=torch.float64) tensor(-2196877.0015, device='cuda:0', dtype=torch.float64)
[DRfold2] Running selection for cluster 2 completed successfully
[DRfold2] Cluster 2 Optimization Process
[DRfold2] Optimization output prefix: /kaggle/working/predictions/R1107/folds/opt_3
[DRfold2] Running optimization for cluster 2
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Optimization.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/folds/opt_3 /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/sel_3 /kaggle/working/DRfold2/cfg_for_folding.json /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] Before sort: ['cfg_97_model_6.ret', 'cfg_97_model_15.ret', 'cfg_97_model_9.ret']
[DRfold2 subprocess] After sort: ['cfg_97_model_15.ret', 'cfg_97_model_6.ret', 'cfg_97_model_9.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] [DRfold2] Loading AlphaFold3 prediction from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Loaded AlphaFold3 structure from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] AlphaFold3 alignment initialized (shape: torch.Size([69, 69, 3, 3]))
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2] Running optimization for cluster 2 completed successfully
[DRfold2] Cluster 2 Refinement Process
[DRfold2] Found optimized structure: /kaggle/working/predictions/R1107/folds/opt_3_from_cfg_97_model_6.ret.pdb
[DRfold2] Final output will be saved to: /kaggle/working/predictions/R1107/relax/model_3.pdb
[DRfold2] Running refinement for cluster 2
[DRfold2] Command: /kaggle/working/DRfold2/Arena/Arena /kaggle/working/predictions/R1107/folds/opt_3_from_cfg_97_model_6.ret.pdb /kaggle/working/predictions/R1107/relax/model_3.pdb 7
[DRfold2 subprocess] G1 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G1  with 15 missing atoms
[DRfold2 subprocess] G2 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G2  with 15 missing atoms
[DRfold2 subprocess] G3 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G3  with 15 missing atoms
[DRfold2 subprocess] G4 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G4  with 15 missing atoms
[DRfold2 subprocess] G5 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G5  with 15 missing atoms
[DRfold2 subprocess] C6 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C6  with 12 missing atoms
[DRfold2 subprocess] C7 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C7  with 12 missing atoms
[DRfold2 subprocess] A8 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A8  with 14 missing atoms
[DRfold2 subprocess] C9 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C9  with 12 missing atoms
[DRfold2 subprocess] A10 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A10  with 14 missing atoms
[DRfold2 subprocess] G11 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G11  with 15 missing atoms
[DRfold2 subprocess] C12 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C12  with 12 missing atoms
[DRfold2 subprocess] A13 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A13  with 14 missing atoms
[DRfold2 subprocess] G14 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G14  with 15 missing atoms
[DRfold2 subprocess] A15 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A15  with 14 missing atoms
[DRfold2 subprocess] A16 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A16  with 14 missing atoms
[DRfold2 subprocess] G17 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G17  with 15 missing atoms
[DRfold2 subprocess] C18 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C18  with 12 missing atoms
[DRfold2 subprocess] G19 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G19  with 15 missing atoms
[DRfold2 subprocess] U20 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U20  with 12 missing atoms
[DRfold2 subprocess] U21 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U21  with 12 missing atoms
[DRfold2 subprocess] C22 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C22  with 12 missing atoms
[DRfold2 subprocess] A23 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A23  with 14 missing atoms
[DRfold2 subprocess] C24 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C24  with 12 missing atoms
[DRfold2 subprocess] G25 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G25  with 15 missing atoms
[DRfold2 subprocess] U26 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U26  with 12 missing atoms
[DRfold2 subprocess] C27 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C27  with 12 missing atoms
[DRfold2 subprocess] G28 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G28  with 15 missing atoms
[DRfold2 subprocess] C29 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C29  with 12 missing atoms
[DRfold2 subprocess] A30 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A30  with 14 missing atoms
[DRfold2 subprocess] G31 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G31  with 15 missing atoms
[DRfold2 subprocess] C32 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C32  with 12 missing atoms
[DRfold2 subprocess] C33 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C33  with 12 missing atoms
[DRfold2 subprocess] C34 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C34  with 12 missing atoms
[DRfold2 subprocess] C35 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C35  with 12 missing atoms
[DRfold2 subprocess] U36 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U36  with 12 missing atoms
[DRfold2 subprocess] G37 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G37  with 15 missing atoms
[DRfold2 subprocess] U38 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U38  with 12 missing atoms
[DRfold2 subprocess] C39 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C39  with 12 missing atoms
[DRfold2 subprocess] A40 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A40  with 14 missing atoms
[DRfold2 subprocess] G41 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G41  with 15 missing atoms
[DRfold2 subprocess] C42 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C42  with 12 missing atoms
[DRfold2 subprocess] C43 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C43  with 12 missing atoms
[DRfold2 subprocess] A44 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A44  with 14 missing atoms
[DRfold2 subprocess] U45 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U45  with 12 missing atoms
[DRfold2 subprocess] U46 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U46  with 12 missing atoms
[DRfold2 subprocess] G47 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G47  with 15 missing atoms
[DRfold2 subprocess] C48 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C48  with 12 missing atoms
[DRfold2 subprocess] A49 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A49  with 14 missing atoms
[DRfold2 subprocess] C50 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C50  with 12 missing atoms
[DRfold2 subprocess] U51 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U51  with 12 missing atoms
[DRfold2 subprocess] C52 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C52  with 12 missing atoms
[DRfold2 subprocess] C53 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C53  with 12 missing atoms
[DRfold2 subprocess] G54 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G54  with 15 missing atoms
[DRfold2 subprocess] G55 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G55  with 15 missing atoms
[DRfold2 subprocess] C56 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C56  with 12 missing atoms
[DRfold2 subprocess] U57 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U57  with 12 missing atoms
[DRfold2 subprocess] G58 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G58  with 15 missing atoms
[DRfold2 subprocess] C59 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C59  with 12 missing atoms
[DRfold2 subprocess] G60 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G60  with 15 missing atoms
[DRfold2 subprocess] A61 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A61  with 14 missing atoms
[DRfold2 subprocess] A62 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A62  with 14 missing atoms
[DRfold2 subprocess] U63 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U63  with 12 missing atoms
[DRfold2 subprocess] U64 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U64  with 12 missing atoms
[DRfold2 subprocess] C65 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C65  with 12 missing atoms
[DRfold2 subprocess] U66 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U66  with 12 missing atoms
[DRfold2 subprocess] G67 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G67  with 15 missing atoms
[DRfold2 subprocess] C68 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C68  with 12 missing atoms
[DRfold2 subprocess] U69 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U69  with 12 missing atoms
[DRfold2 subprocess] t=0 moved=1731
[DRfold2 subprocess] t=1 moved=2464
[DRfold2 subprocess] t=2 moved=2350
[DRfold2 subprocess] t=3 moved=2265
[DRfold2 subprocess] t=4 moved=2228
[DRfold2 subprocess] t=5 moved=2180
[DRfold2 subprocess] t=6 moved=2168
[DRfold2 subprocess] t=7 moved=2132
[DRfold2 subprocess] t=8 moved=2105
[DRfold2 subprocess] t=9 moved=2083
[DRfold2 subprocess] t=10 moved=2071
[DRfold2 subprocess] t=11 moved=2078
[DRfold2 subprocess] t=12 moved=2061
[DRfold2 subprocess] t=13 moved=2025
[DRfold2 subprocess] t=14 moved=2028
[DRfold2 subprocess] t=15 moved=2028
[DRfold2 subprocess] t=16 moved=2023
[DRfold2 subprocess] t=17 moved=2029
[DRfold2 subprocess] t=18 moved=2018
[DRfold2 subprocess] t=19 moved=2042
[DRfold2 subprocess] t=20 moved=2027
[DRfold2 subprocess] t=21 moved=2021
[DRfold2 subprocess] t=22 moved=2008
[DRfold2 subprocess] t=23 moved=2026
[DRfold2 subprocess] t=24 moved=2019
[DRfold2 subprocess] t=25 moved=2011
[DRfold2 subprocess] t=26 moved=2017
[DRfold2 subprocess] t=27 moved=2008
[DRfold2 subprocess] t=28 moved=2004
[DRfold2 subprocess] t=29 moved=1991
[DRfold2 subprocess] t=30 moved=1987
[DRfold2 subprocess] t=31 moved=1976
[DRfold2 subprocess] t=32 moved=1970
[DRfold2 subprocess] t=33 moved=1994
[DRfold2 subprocess] t=34 moved=1982
[DRfold2 subprocess] t=35 moved=1959
[DRfold2 subprocess] t=36 moved=1982
[DRfold2 subprocess] t=37 moved=1966
[DRfold2 subprocess] t=38 moved=1971
[DRfold2 subprocess] t=39 moved=1965
[DRfold2 subprocess] t=40 moved=1956
[DRfold2 subprocess] t=41 moved=1983
[DRfold2 subprocess] t=42 moved=1983
[DRfold2 subprocess] t=43 moved=1997
[DRfold2 subprocess] t=44 moved=1954
[DRfold2 subprocess] t=45 moved=1976
[DRfold2 subprocess] t=46 moved=1953
[DRfold2 subprocess] t=47 moved=1964
[DRfold2 subprocess] t=48 moved=1971
[DRfold2 subprocess] t=49 moved=1971
[DRfold2 subprocess] t=50 moved=1979
[DRfold2 subprocess] t=51 moved=1976
[DRfold2 subprocess] t=52 moved=1955
[DRfold2 subprocess] t=53 moved=1940
[DRfold2 subprocess] t=54 moved=1956
[DRfold2 subprocess] t=55 moved=1957
[DRfold2 subprocess] t=56 moved=1931
[DRfold2 subprocess] t=57 moved=1946
[DRfold2 subprocess] t=58 moved=1932
[DRfold2 subprocess] t=59 moved=1940
[DRfold2 subprocess] t=60 moved=1944
[DRfold2 subprocess] t=61 moved=1928
[DRfold2 subprocess] t=62 moved=1938
[DRfold2 subprocess] t=63 moved=1934
[DRfold2 subprocess] t=64 moved=1918
[DRfold2 subprocess] t=65 moved=1935
[DRfold2 subprocess] t=66 moved=1932
[DRfold2 subprocess] t=67 moved=1928
[DRfold2 subprocess] t=68 moved=1908
[DRfold2 subprocess] t=69 moved=1927
[DRfold2 subprocess] t=70 moved=1939
[DRfold2 subprocess] t=71 moved=1939
[DRfold2 subprocess] t=72 moved=1952
[DRfold2 subprocess] t=73 moved=1920
[DRfold2 subprocess] t=74 moved=1915
[DRfold2 subprocess] t=75 moved=1957
[DRfold2 subprocess] t=76 moved=1913
[DRfold2 subprocess] t=77 moved=1915
[DRfold2 subprocess] t=78 moved=1948
[DRfold2 subprocess] t=79 moved=1956
[DRfold2 subprocess] t=80 moved=1947
[DRfold2 subprocess] t=81 moved=1933
[DRfold2 subprocess] t=82 moved=1945
[DRfold2 subprocess] t=83 moved=1915
[DRfold2 subprocess] t=84 moved=1919
[DRfold2 subprocess] t=85 moved=1937
[DRfold2 subprocess] t=86 moved=1953
[DRfold2 subprocess] t=87 moved=1929
[DRfold2 subprocess] t=88 moved=1949
[DRfold2 subprocess] t=89 moved=1920
[DRfold2 subprocess] t=90 moved=1929
[DRfold2 subprocess] t=91 moved=1933
[DRfold2 subprocess] t=92 moved=1915
[DRfold2 subprocess] t=93 moved=1937
[DRfold2 subprocess] t=94 moved=1917
[DRfold2 subprocess] t=95 moved=1933
[DRfold2 subprocess] t=96 moved=1950
[DRfold2 subprocess] t=97 moved=1945
[DRfold2 subprocess] t=98 moved=1937
[DRfold2 subprocess] t=99 moved=1941
[DRfold2 subprocess] t=100 moved=1934
[DRfold2 subprocess] t=101 moved=1921
[DRfold2 subprocess] t=102 moved=1925
[DRfold2 subprocess] stage1 t=0 moved=2182
[DRfold2 subprocess] stage1 t=1 moved=2107
[DRfold2 subprocess] stage1 t=2 moved=1773
[DRfold2 subprocess] stage1 t=3 moved=1595
[DRfold2 subprocess] stage1 t=4 moved=1423
[DRfold2 subprocess] stage1 t=5 moved=1264
[DRfold2 subprocess] stage1 t=6 moved=1169
[DRfold2 subprocess] stage1 t=7 moved=1070
[DRfold2 subprocess] stage1 t=8 moved=963
[DRfold2 subprocess] stage1 t=9 moved=942
[DRfold2 subprocess] stage1 t=10 moved=868
[DRfold2 subprocess] stage1 t=11 moved=840
[DRfold2 subprocess] stage1 t=12 moved=803
[DRfold2 subprocess] stage1 t=13 moved=676
[DRfold2 subprocess] stage1 t=14 moved=652
[DRfold2 subprocess] stage1 t=15 moved=704
[DRfold2 subprocess] stage1 t=16 moved=687
[DRfold2 subprocess] stage1 t=17 moved=612
[DRfold2 subprocess] stage1 t=18 moved=563
[DRfold2 subprocess] stage1 t=19 moved=529
[DRfold2 subprocess] stage1 t=20 moved=485
[DRfold2 subprocess] stage1 t=21 moved=473
[DRfold2 subprocess] stage1 t=22 moved=444
[DRfold2 subprocess] stage1 t=23 moved=435
[DRfold2 subprocess] stage1 t=24 moved=377
[DRfold2 subprocess] stage1 t=25 moved=330
[DRfold2 subprocess] stage1 t=26 moved=268
[DRfold2 subprocess] stage1 t=27 moved=223
[DRfold2 subprocess] stage1 t=28 moved=244
[DRfold2 subprocess] stage1 t=29 moved=235
[DRfold2 subprocess] stage1 t=30 moved=265
[DRfold2 subprocess] stage1 t=31 moved=246
[DRfold2 subprocess] stage1 t=32 moved=217
[DRfold2 subprocess] stage1 t=33 moved=171
[DRfold2 subprocess] stage1 t=34 moved=221
[DRfold2 subprocess] stage1 t=35 moved=228
[DRfold2 subprocess] stage1 t=36 moved=153
[DRfold2 subprocess] stage1 t=37 moved=139
[DRfold2 subprocess] stage1 t=38 moved=126
[DRfold2 subprocess] stage1 t=39 moved=118
[DRfold2 subprocess] stage1 t=40 moved=110
[DRfold2 subprocess] stage1 t=41 moved=106
[DRfold2 subprocess] stage1 t=42 moved=127
[DRfold2 subprocess] stage1 t=43 moved=79
[DRfold2 subprocess] stage1 t=44 moved=82
[DRfold2 subprocess] stage1 t=45 moved=67
[DRfold2 subprocess] stage1 t=46 moved=80
[DRfold2 subprocess] stage1 t=47 moved=83
[DRfold2 subprocess] stage1 t=48 moved=68
[DRfold2 subprocess] stage1 t=49 moved=46
[DRfold2 subprocess] stage1 t=50 moved=74
[DRfold2 subprocess] stage1 t=51 moved=98
[DRfold2 subprocess] stage1 t=52 moved=104
[DRfold2 subprocess] stage1 t=53 moved=90
[DRfold2 subprocess] stage1 t=54 moved=76
[DRfold2 subprocess] stage1 t=55 moved=69
[DRfold2 subprocess] stage1 t=56 moved=62
[DRfold2 subprocess] stage1 t=57 moved=49
[DRfold2 subprocess] stage1 t=58 moved=53
[DRfold2 subprocess] stage1 t=59 moved=67
[DRfold2 subprocess] stage1 t=60 moved=22
[DRfold2 subprocess] stage1 t=61 moved=21
[DRfold2 subprocess] stage1 t=62 moved=20
[DRfold2 subprocess] stage1 t=63 moved=0
[DRfold2 subprocess] stage2 t=0 moved=500
[DRfold2 subprocess] stage2 t=1 moved=1245
[DRfold2 subprocess] stage2 t=2 moved=1261
[DRfold2 subprocess] stage2 t=3 moved=1232
[DRfold2 subprocess] stage2 t=4 moved=1206
[DRfold2 subprocess] stage2 t=5 moved=1098
[DRfold2 subprocess] stage2 t=6 moved=977
[DRfold2 subprocess] stage2 t=7 moved=903
[DRfold2 subprocess] stage2 t=8 moved=930
[DRfold2 subprocess] stage2 t=9 moved=866
[DRfold2 subprocess] stage2 t=10 moved=807
[DRfold2 subprocess] stage2 t=11 moved=747
[DRfold2 subprocess] stage2 t=12 moved=630
[DRfold2 subprocess] stage2 t=13 moved=505
[DRfold2 subprocess] stage2 t=14 moved=680
[DRfold2 subprocess] stage2 t=15 moved=722
[DRfold2 subprocess] stage2 t=16 moved=510
[DRfold2 subprocess] stage2 t=17 moved=562
[DRfold2 subprocess] stage2 t=18 moved=719
[DRfold2 subprocess] stage2 t=19 moved=628
[DRfold2 subprocess] stage2 t=20 moved=568
[DRfold2 subprocess] stage2 t=21 moved=551
[DRfold2 subprocess] stage2 t=22 moved=533
[DRfold2 subprocess] stage2 t=23 moved=450
[DRfold2 subprocess] stage2 t=24 moved=296
[DRfold2 subprocess] stage2 t=25 moved=297
[DRfold2 subprocess] stage2 t=26 moved=309
[DRfold2 subprocess] stage2 t=27 moved=282
[DRfold2 subprocess] stage2 t=28 moved=234
[DRfold2 subprocess] stage2 t=29 moved=175
[DRfold2 subprocess] stage2 t=30 moved=260
[DRfold2 subprocess] stage2 t=31 moved=249
[DRfold2 subprocess] stage2 t=32 moved=237
[DRfold2 subprocess] stage2 t=33 moved=199
[DRfold2 subprocess] stage2 t=34 moved=136
[DRfold2 subprocess] stage2 t=35 moved=152
[DRfold2 subprocess] stage2 t=36 moved=215
[DRfold2 subprocess] stage2 t=37 moved=217
[DRfold2 subprocess] stage2 t=38 moved=158
[DRfold2 subprocess] stage2 t=39 moved=128
[DRfold2 subprocess] stage2 t=40 moved=160
[DRfold2 subprocess] stage2 t=41 moved=161
[DRfold2 subprocess] stage2 t=42 moved=134
[DRfold2 subprocess] stage2 t=43 moved=173
[DRfold2 subprocess] stage2 t=44 moved=109
[DRfold2 subprocess] stage2 t=45 moved=95
[DRfold2 subprocess] stage2 t=46 moved=117
[DRfold2 subprocess] stage2 t=47 moved=116
[DRfold2 subprocess] stage2 t=48 moved=88
[DRfold2 subprocess] stage2 t=49 moved=60
[DRfold2 subprocess] stage2 t=50 moved=111
[DRfold2 subprocess] stage2 t=51 moved=42
[DRfold2 subprocess] stage2 t=52 moved=75
[DRfold2 subprocess] stage2 t=53 moved=78
[DRfold2 subprocess] stage2 t=54 moved=99
[DRfold2 subprocess] stage2 t=55 moved=60
[DRfold2 subprocess] stage2 t=56 moved=50
[DRfold2 subprocess] stage2 t=57 moved=36
[DRfold2 subprocess] stage2 t=58 moved=48
[DRfold2 subprocess] stage2 t=59 moved=67
[DRfold2 subprocess] stage2 t=60 moved=74
[DRfold2 subprocess] stage2 t=61 moved=99
[DRfold2 subprocess] stage2 t=62 moved=99
[DRfold2 subprocess] stage2 t=63 moved=114
[DRfold2 subprocess] stage2 t=64 moved=114
[DRfold2 subprocess] stage2 t=65 moved=107
[DRfold2 subprocess] stage2 t=66 moved=115
[DRfold2 subprocess] stage2 t=67 moved=124
[DRfold2 subprocess] stage2 t=68 moved=243
[DRfold2 subprocess] stage2 t=69 moved=245
[DRfold2 subprocess] stage2 t=70 moved=219
[DRfold2 subprocess] stage2 t=71 moved=159
[DRfold2 subprocess] stage2 t=72 moved=152
[DRfold2 subprocess] stage2 t=73 moved=76
[DRfold2 subprocess] stage2 t=74 moved=73
[DRfold2 subprocess] stage2 t=75 moved=36
[DRfold2 subprocess] stage2 t=76 moved=38
[DRfold2 subprocess] stage2 t=77 moved=36
[DRfold2 subprocess] stage2 t=78 moved=0
[DRfold2 subprocess] stage3 t=0 moved=0
[DRfold2] Running refinement for cluster 2 completed successfully
[DRfold2] PROCESSING CLUSTER 3/4
[DRfold2] Cluster 3 Selection Process
[DRfold2] Found 2 return files for selection
[DRfold2] Selection output prefix: /kaggle/working/predictions/R1107/folds/sel_4
[DRfold2] Running selection for cluster 3
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Selection.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/DRfold2/cfg_for_selection.json /kaggle/working/predictions/R1107/folds/sel_4 /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_1.ret /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_2.ret
[DRfold2 subprocess] ['/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_1.ret', '/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_2.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] tensor(113426.3262, device='cuda:0', dtype=torch.float64) tensor(-4066270.7941, device='cuda:0', dtype=torch.float64) tensor(-4095329.9081, device='cuda:0', dtype=torch.float64) tensor(-4270029.8541, device='cuda:0', dtype=torch.float64)
[DRfold2 subprocess] tensor(471319.2023, device='cuda:0', dtype=torch.float64) tensor(-3899754.1397, device='cuda:0', dtype=torch.float64) tensor(-3896048.8682, device='cuda:0', dtype=torch.float64) tensor(-4006897.2968, device='cuda:0', dtype=torch.float64)
[DRfold2] Running selection for cluster 3 completed successfully
[DRfold2] Cluster 3 Optimization Process
[DRfold2] Optimization output prefix: /kaggle/working/predictions/R1107/folds/opt_4
[DRfold2] Running optimization for cluster 3
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Optimization.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/folds/opt_4 /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/sel_4 /kaggle/working/DRfold2/cfg_for_folding.json /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] Before sort: ['cfg_97_model_1.ret', 'cfg_97_model_2.ret']
[DRfold2 subprocess] After sort: ['cfg_97_model_1.ret', 'cfg_97_model_2.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] [DRfold2] Loading AlphaFold3 prediction from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Loaded AlphaFold3 structure from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] AlphaFold3 alignment initialized (shape: torch.Size([69, 69, 3, 3]))
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2] Running optimization for cluster 3 completed successfully
[DRfold2] Cluster 3 Refinement Process
[DRfold2] Found optimized structure: /kaggle/working/predictions/R1107/folds/opt_4_from_cfg_97_model_1.ret.pdb
[DRfold2] Final output will be saved to: /kaggle/working/predictions/R1107/relax/model_4.pdb
[DRfold2] Running refinement for cluster 3
[DRfold2] Command: /kaggle/working/DRfold2/Arena/Arena /kaggle/working/predictions/R1107/folds/opt_4_from_cfg_97_model_1.ret.pdb /kaggle/working/predictions/R1107/relax/model_4.pdb 7
[DRfold2 subprocess] G1 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G1  with 15 missing atoms
[DRfold2 subprocess] G2 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G2  with 15 missing atoms
[DRfold2 subprocess] G3 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G3  with 15 missing atoms
[DRfold2 subprocess] G4 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G4  with 15 missing atoms
[DRfold2 subprocess] G5 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G5  with 15 missing atoms
[DRfold2 subprocess] C6 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C6  with 12 missing atoms
[DRfold2 subprocess] C7 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C7  with 12 missing atoms
[DRfold2 subprocess] A8 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A8  with 14 missing atoms
[DRfold2 subprocess] C9 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C9  with 12 missing atoms
[DRfold2 subprocess] A10 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A10  with 14 missing atoms
[DRfold2 subprocess] G11 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G11  with 15 missing atoms
[DRfold2 subprocess] C12 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C12  with 12 missing atoms
[DRfold2 subprocess] A13 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A13  with 14 missing atoms
[DRfold2 subprocess] G14 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G14  with 15 missing atoms
[DRfold2 subprocess] A15 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A15  with 14 missing atoms
[DRfold2 subprocess] A16 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A16  with 14 missing atoms
[DRfold2 subprocess] G17 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G17  with 15 missing atoms
[DRfold2 subprocess] C18 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C18  with 12 missing atoms
[DRfold2 subprocess] G19 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G19  with 15 missing atoms
[DRfold2 subprocess] U20 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U20  with 12 missing atoms
[DRfold2 subprocess] U21 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U21  with 12 missing atoms
[DRfold2 subprocess] C22 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C22  with 12 missing atoms
[DRfold2 subprocess] A23 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A23  with 14 missing atoms
[DRfold2 subprocess] C24 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C24  with 12 missing atoms
[DRfold2 subprocess] G25 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G25  with 15 missing atoms
[DRfold2 subprocess] U26 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U26  with 12 missing atoms
[DRfold2 subprocess] C27 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C27  with 12 missing atoms
[DRfold2 subprocess] G28 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G28  with 15 missing atoms
[DRfold2 subprocess] C29 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C29  with 12 missing atoms
[DRfold2 subprocess] A30 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A30  with 14 missing atoms
[DRfold2 subprocess] G31 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G31  with 15 missing atoms
[DRfold2 subprocess] C32 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C32  with 12 missing atoms
[DRfold2 subprocess] C33 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C33  with 12 missing atoms
[DRfold2 subprocess] C34 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C34  with 12 missing atoms
[DRfold2 subprocess] C35 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C35  with 12 missing atoms
[DRfold2 subprocess] U36 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U36  with 12 missing atoms
[DRfold2 subprocess] G37 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G37  with 15 missing atoms
[DRfold2 subprocess] U38 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U38  with 12 missing atoms
[DRfold2 subprocess] C39 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C39  with 12 missing atoms
[DRfold2 subprocess] A40 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A40  with 14 missing atoms
[DRfold2 subprocess] G41 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G41  with 15 missing atoms
[DRfold2 subprocess] C42 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C42  with 12 missing atoms
[DRfold2 subprocess] C43 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C43  with 12 missing atoms
[DRfold2 subprocess] A44 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A44  with 14 missing atoms
[DRfold2 subprocess] U45 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U45  with 12 missing atoms
[DRfold2 subprocess] U46 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U46  with 12 missing atoms
[DRfold2 subprocess] G47 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G47  with 15 missing atoms
[DRfold2 subprocess] C48 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C48  with 12 missing atoms
[DRfold2 subprocess] A49 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A49  with 14 missing atoms
[DRfold2 subprocess] C50 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C50  with 12 missing atoms
[DRfold2 subprocess] U51 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U51  with 12 missing atoms
[DRfold2 subprocess] C52 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C52  with 12 missing atoms
[DRfold2 subprocess] C53 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C53  with 12 missing atoms
[DRfold2 subprocess] G54 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G54  with 15 missing atoms
[DRfold2 subprocess] G55 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G55  with 15 missing atoms
[DRfold2 subprocess] C56 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C56  with 12 missing atoms
[DRfold2 subprocess] U57 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U57  with 12 missing atoms
[DRfold2 subprocess] G58 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G58  with 15 missing atoms
[DRfold2 subprocess] C59 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C59  with 12 missing atoms
[DRfold2 subprocess] G60 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G60  with 15 missing atoms
[DRfold2 subprocess] A61 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A61  with 14 missing atoms
[DRfold2 subprocess] A62 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A62  with 14 missing atoms
[DRfold2 subprocess] U63 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U63  with 12 missing atoms
[DRfold2 subprocess] U64 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U64  with 12 missing atoms
[DRfold2 subprocess] C65 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C65  with 12 missing atoms
[DRfold2 subprocess] U66 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U66  with 12 missing atoms
[DRfold2 subprocess] G67 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G67  with 15 missing atoms
[DRfold2 subprocess] C68 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C68  with 12 missing atoms
[DRfold2 subprocess] U69 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U69  with 12 missing atoms
[DRfold2 subprocess] t=0 moved=1535
[DRfold2 subprocess] t=1 moved=2032
[DRfold2 subprocess] t=2 moved=1869
[DRfold2 subprocess] t=3 moved=1813
[DRfold2 subprocess] t=4 moved=1746
[DRfold2 subprocess] t=5 moved=1711
[DRfold2 subprocess] t=6 moved=1682
[DRfold2 subprocess] t=7 moved=1693
[DRfold2 subprocess] t=8 moved=1659
[DRfold2 subprocess] t=9 moved=1665
[DRfold2 subprocess] t=10 moved=1629
[DRfold2 subprocess] t=11 moved=1656
[DRfold2 subprocess] t=12 moved=1626
[DRfold2 subprocess] t=13 moved=1626
[DRfold2 subprocess] t=14 moved=1632
[DRfold2 subprocess] t=15 moved=1612
[DRfold2 subprocess] t=16 moved=1586
[DRfold2 subprocess] t=17 moved=1600
[DRfold2 subprocess] t=18 moved=1586
[DRfold2 subprocess] t=19 moved=1590
[DRfold2 subprocess] t=20 moved=1586
[DRfold2 subprocess] t=21 moved=1596
[DRfold2 subprocess] t=22 moved=1579
[DRfold2 subprocess] t=23 moved=1575
[DRfold2 subprocess] t=24 moved=1571
[DRfold2 subprocess] t=25 moved=1574
[DRfold2 subprocess] t=26 moved=1576
[DRfold2 subprocess] t=27 moved=1583
[DRfold2 subprocess] t=28 moved=1574
[DRfold2 subprocess] t=29 moved=1580
[DRfold2 subprocess] t=30 moved=1590
[DRfold2 subprocess] t=31 moved=1563
[DRfold2 subprocess] t=32 moved=1564
[DRfold2 subprocess] t=33 moved=1569
[DRfold2 subprocess] t=34 moved=1579
[DRfold2 subprocess] t=35 moved=1581
[DRfold2 subprocess] t=36 moved=1574
[DRfold2 subprocess] t=37 moved=1560
[DRfold2 subprocess] t=38 moved=1588
[DRfold2 subprocess] t=39 moved=1567
[DRfold2 subprocess] t=40 moved=1570
[DRfold2 subprocess] t=41 moved=1558
[DRfold2 subprocess] t=42 moved=1569
[DRfold2 subprocess] t=43 moved=1565
[DRfold2 subprocess] t=44 moved=1561
[DRfold2 subprocess] t=45 moved=1568
[DRfold2 subprocess] t=46 moved=1559
[DRfold2 subprocess] t=47 moved=1551
[DRfold2 subprocess] t=48 moved=1577
[DRfold2 subprocess] t=49 moved=1581
[DRfold2 subprocess] t=50 moved=1571
[DRfold2 subprocess] t=51 moved=1570
[DRfold2 subprocess] t=52 moved=1570
[DRfold2 subprocess] t=53 moved=1564
[DRfold2 subprocess] t=54 moved=1563
[DRfold2 subprocess] t=55 moved=1567
[DRfold2 subprocess] t=56 moved=1558
[DRfold2 subprocess] t=57 moved=1569
[DRfold2 subprocess] t=58 moved=1565
[DRfold2 subprocess] t=59 moved=1568
[DRfold2 subprocess] t=60 moved=1593
[DRfold2 subprocess] t=61 moved=1572
[DRfold2 subprocess] t=62 moved=1576
[DRfold2 subprocess] t=63 moved=1566
[DRfold2 subprocess] t=64 moved=1565
[DRfold2 subprocess] t=65 moved=1567
[DRfold2 subprocess] t=66 moved=1575
[DRfold2 subprocess] t=67 moved=1580
[DRfold2 subprocess] t=68 moved=1570
[DRfold2 subprocess] t=69 moved=1579
[DRfold2 subprocess] t=70 moved=1558
[DRfold2 subprocess] t=71 moved=1582
[DRfold2 subprocess] t=72 moved=1568
[DRfold2 subprocess] t=73 moved=1582
[DRfold2 subprocess] t=74 moved=1567
[DRfold2 subprocess] t=75 moved=1590
[DRfold2 subprocess] t=76 moved=1562
[DRfold2 subprocess] t=77 moved=1569
[DRfold2 subprocess] t=78 moved=1553
[DRfold2 subprocess] t=79 moved=1568
[DRfold2 subprocess] t=80 moved=1558
[DRfold2 subprocess] t=81 moved=1577
[DRfold2 subprocess] t=82 moved=1563
[DRfold2 subprocess] t=83 moved=1560
[DRfold2 subprocess] t=84 moved=1566
[DRfold2 subprocess] t=85 moved=1574
[DRfold2 subprocess] t=86 moved=1574
[DRfold2 subprocess] t=87 moved=1562
[DRfold2 subprocess] t=88 moved=1569
[DRfold2 subprocess] t=89 moved=1582
[DRfold2 subprocess] t=90 moved=1592
[DRfold2 subprocess] t=91 moved=1571
[DRfold2 subprocess] t=92 moved=1564
[DRfold2 subprocess] t=93 moved=1569
[DRfold2 subprocess] t=94 moved=1563
[DRfold2 subprocess] t=95 moved=1563
[DRfold2 subprocess] t=96 moved=1570
[DRfold2 subprocess] t=97 moved=1580
[DRfold2 subprocess] t=98 moved=1559
[DRfold2 subprocess] t=99 moved=1580
[DRfold2 subprocess] t=100 moved=1579
[DRfold2 subprocess] t=101 moved=1547
[DRfold2 subprocess] t=102 moved=1582
[DRfold2 subprocess] stage1 t=0 moved=1683
[DRfold2 subprocess] stage1 t=1 moved=1506
[DRfold2 subprocess] stage1 t=2 moved=1238
[DRfold2 subprocess] stage1 t=3 moved=1094
[DRfold2 subprocess] stage1 t=4 moved=883
[DRfold2 subprocess] stage1 t=5 moved=734
[DRfold2 subprocess] stage1 t=6 moved=654
[DRfold2 subprocess] stage1 t=7 moved=574
[DRfold2 subprocess] stage1 t=8 moved=500
[DRfold2 subprocess] stage1 t=9 moved=409
[DRfold2 subprocess] stage1 t=10 moved=402
[DRfold2 subprocess] stage1 t=11 moved=366
[DRfold2 subprocess] stage1 t=12 moved=342
[DRfold2 subprocess] stage1 t=13 moved=302
[DRfold2 subprocess] stage1 t=14 moved=273
[DRfold2 subprocess] stage1 t=15 moved=271
[DRfold2 subprocess] stage1 t=16 moved=259
[DRfold2 subprocess] stage1 t=17 moved=286
[DRfold2 subprocess] stage1 t=18 moved=234
[DRfold2 subprocess] stage1 t=19 moved=223
[DRfold2 subprocess] stage1 t=20 moved=182
[DRfold2 subprocess] stage1 t=21 moved=179
[DRfold2 subprocess] stage1 t=22 moved=174
[DRfold2 subprocess] stage1 t=23 moved=179
[DRfold2 subprocess] stage1 t=24 moved=156
[DRfold2 subprocess] stage1 t=25 moved=141
[DRfold2 subprocess] stage1 t=26 moved=140
[DRfold2 subprocess] stage1 t=27 moved=148
[DRfold2 subprocess] stage1 t=28 moved=134
[DRfold2 subprocess] stage1 t=29 moved=101
[DRfold2 subprocess] stage1 t=30 moved=92
[DRfold2 subprocess] stage1 t=31 moved=77
[DRfold2 subprocess] stage1 t=32 moved=91
[DRfold2 subprocess] stage1 t=33 moved=97
[DRfold2 subprocess] stage1 t=34 moved=77
[DRfold2 subprocess] stage1 t=35 moved=91
[DRfold2 subprocess] stage1 t=36 moved=52
[DRfold2 subprocess] stage1 t=37 moved=34
[DRfold2 subprocess] stage1 t=38 moved=28
[DRfold2 subprocess] stage1 t=39 moved=28
[DRfold2 subprocess] stage1 t=40 moved=32
[DRfold2 subprocess] stage1 t=41 moved=30
[DRfold2 subprocess] stage1 t=42 moved=31
[DRfold2 subprocess] stage1 t=43 moved=28
[DRfold2 subprocess] stage1 t=44 moved=24
[DRfold2 subprocess] stage1 t=45 moved=19
[DRfold2 subprocess] stage1 t=46 moved=14
[DRfold2 subprocess] stage1 t=47 moved=4
[DRfold2 subprocess] stage1 t=48 moved=1
[DRfold2 subprocess] stage1 t=49 moved=0
[DRfold2 subprocess] stage2 t=0 moved=438
[DRfold2 subprocess] stage2 t=1 moved=881
[DRfold2 subprocess] stage2 t=2 moved=911
[DRfold2 subprocess] stage2 t=3 moved=640
[DRfold2 subprocess] stage2 t=4 moved=535
[DRfold2 subprocess] stage2 t=5 moved=535
[DRfold2 subprocess] stage2 t=6 moved=474
[DRfold2 subprocess] stage2 t=7 moved=379
[DRfold2 subprocess] stage2 t=8 moved=462
[DRfold2 subprocess] stage2 t=9 moved=400
[DRfold2 subprocess] stage2 t=10 moved=241
[DRfold2 subprocess] stage2 t=11 moved=275
[DRfold2 subprocess] stage2 t=12 moved=232
[DRfold2 subprocess] stage2 t=13 moved=199
[DRfold2 subprocess] stage2 t=14 moved=112
[DRfold2 subprocess] stage2 t=15 moved=53
[DRfold2 subprocess] stage2 t=16 moved=0
[DRfold2 subprocess] stage3 t=0 moved=0
[DRfold2] Running refinement for cluster 3 completed successfully
[DRfold2] PROCESSING CLUSTER 4/4
[DRfold2] Cluster 4 Selection Process
[DRfold2] Found 1 return files for selection
[DRfold2] Selection output prefix: /kaggle/working/predictions/R1107/folds/sel_5
[DRfold2] Running selection for cluster 4
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Selection.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/DRfold2/cfg_for_selection.json /kaggle/working/predictions/R1107/folds/sel_5 /kaggle/working/predictions/R1107/rets_dir/cfg_97_model_3.ret
[DRfold2 subprocess] ['/kaggle/working/predictions/R1107/rets_dir/cfg_97_model_3.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] tensor(6.0508e-08, device='cuda:0', dtype=torch.float64) tensor(-4385736.0060, device='cuda:0', dtype=torch.float64) tensor(-4424327.7662, device='cuda:0', dtype=torch.float64) tensor(-4573153.6736, device='cuda:0', dtype=torch.float64)
[DRfold2] Running selection for cluster 4 completed successfully
[DRfold2] Cluster 4 Optimization Process
[DRfold2] Optimization output prefix: /kaggle/working/predictions/R1107/folds/opt_5
[DRfold2] Running optimization for cluster 4
[DRfold2] Command: python /kaggle/working/DRfold2/PotentialFold/Optimization.py /kaggle/working/fasta_files/R1107.fasta /kaggle/working/predictions/R1107/folds/opt_5 /kaggle/working/predictions/R1107/rets_dir /kaggle/working/predictions/R1107/folds/sel_5 /kaggle/working/DRfold2/cfg_for_folding.json /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Using AlphaFold3 structure: /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] Before sort: ['cfg_97_model_3.ret']
[DRfold2 subprocess] After sort: ['cfg_97_model_3.ret']
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] fitting cubic distance
[DRfold2 subprocess] [DRfold2] Loading AlphaFold3 prediction from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] Loaded AlphaFold3 structure from /kaggle/working/outputs_prediction/boltz_results_inputs_prediction/predictions/R1107/R1107_model_0.pdb
[DRfold2 subprocess] [DRfold2] AlphaFold3 alignment initialized (shape: torch.Size([69, 69, 3, 3]))
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2 subprocess] [DRfold2] Computing AlphaFold3 energy contribution (weight: 2.5)
[DRfold2] Running optimization for cluster 4 completed successfully
[DRfold2] Cluster 4 Refinement Process
[DRfold2] Found optimized structure: /kaggle/working/predictions/R1107/folds/opt_5_from_cfg_97_model_3.ret.pdb
[DRfold2] Final output will be saved to: /kaggle/working/predictions/R1107/relax/model_5.pdb
[DRfold2] Running refinement for cluster 4
[DRfold2] Command: /kaggle/working/DRfold2/Arena/Arena /kaggle/working/predictions/R1107/folds/opt_5_from_cfg_97_model_3.ret.pdb /kaggle/working/predictions/R1107/relax/model_5.pdb 7
[DRfold2 subprocess] G1 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G1  with 15 missing atoms
[DRfold2 subprocess] G2 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G2  with 15 missing atoms
[DRfold2 subprocess] G3 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G3  with 15 missing atoms
[DRfold2 subprocess] G4 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G4  with 15 missing atoms
[DRfold2 subprocess] G5 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G5  with 15 missing atoms
[DRfold2 subprocess] C6 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C6  with 12 missing atoms
[DRfold2 subprocess] C7 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C7  with 12 missing atoms
[DRfold2 subprocess] A8 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A8  with 14 missing atoms
[DRfold2 subprocess] C9 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C9  with 12 missing atoms
[DRfold2 subprocess] A10 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A10  with 14 missing atoms
[DRfold2 subprocess] G11 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G11  with 15 missing atoms
[DRfold2 subprocess] C12 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C12  with 12 missing atoms
[DRfold2 subprocess] A13 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A13  with 14 missing atoms
[DRfold2 subprocess] G14 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G14  with 15 missing atoms
[DRfold2 subprocess] A15 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A15  with 14 missing atoms
[DRfold2 subprocess] A16 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A16  with 14 missing atoms
[DRfold2 subprocess] G17 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G17  with 15 missing atoms
[DRfold2 subprocess] C18 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C18  with 12 missing atoms
[DRfold2 subprocess] G19 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G19  with 15 missing atoms
[DRfold2 subprocess] U20 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U20  with 12 missing atoms
[DRfold2 subprocess] U21 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U21  with 12 missing atoms
[DRfold2 subprocess] C22 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C22  with 12 missing atoms
[DRfold2 subprocess] A23 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A23  with 14 missing atoms
[DRfold2 subprocess] C24 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C24  with 12 missing atoms
[DRfold2 subprocess] G25 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G25  with 15 missing atoms
[DRfold2 subprocess] U26 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U26  with 12 missing atoms
[DRfold2 subprocess] C27 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C27  with 12 missing atoms
[DRfold2 subprocess] G28 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G28  with 15 missing atoms
[DRfold2 subprocess] C29 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C29  with 12 missing atoms
[DRfold2 subprocess] A30 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A30  with 14 missing atoms
[DRfold2 subprocess] G31 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G31  with 15 missing atoms
[DRfold2 subprocess] C32 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C32  with 12 missing atoms
[DRfold2 subprocess] C33 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C33  with 12 missing atoms
[DRfold2 subprocess] C34 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C34  with 12 missing atoms
[DRfold2 subprocess] C35 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C35  with 12 missing atoms
[DRfold2 subprocess] U36 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U36  with 12 missing atoms
[DRfold2 subprocess] G37 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G37  with 15 missing atoms
[DRfold2 subprocess] U38 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U38  with 12 missing atoms
[DRfold2 subprocess] C39 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C39  with 12 missing atoms
[DRfold2 subprocess] A40 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A40  with 14 missing atoms
[DRfold2 subprocess] G41 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G41  with 15 missing atoms
[DRfold2 subprocess] C42 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C42  with 12 missing atoms
[DRfold2 subprocess] C43 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C43  with 12 missing atoms
[DRfold2 subprocess] A44 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A44  with 14 missing atoms
[DRfold2 subprocess] U45 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U45  with 12 missing atoms
[DRfold2 subprocess] U46 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U46  with 12 missing atoms
[DRfold2 subprocess] G47 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G47  with 15 missing atoms
[DRfold2 subprocess] C48 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C48  with 12 missing atoms
[DRfold2 subprocess] A49 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A49  with 14 missing atoms
[DRfold2 subprocess] C50 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C50  with 12 missing atoms
[DRfold2 subprocess] U51 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U51  with 12 missing atoms
[DRfold2 subprocess] C52 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C52  with 12 missing atoms
[DRfold2 subprocess] C53 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C53  with 12 missing atoms
[DRfold2 subprocess] G54 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G54  with 15 missing atoms
[DRfold2 subprocess] G55 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G55  with 15 missing atoms
[DRfold2 subprocess] C56 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C56  with 12 missing atoms
[DRfold2 subprocess] U57 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U57  with 12 missing atoms
[DRfold2 subprocess] G58 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G58  with 15 missing atoms
[DRfold2 subprocess] C59 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C59  with 12 missing atoms
[DRfold2 subprocess] G60 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G60  with 15 missing atoms
[DRfold2 subprocess] A61 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A61  with 14 missing atoms
[DRfold2 subprocess] A62 	14 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N3   N6   N7   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   A62  with 14 missing atoms
[DRfold2 subprocess] U63 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U63  with 12 missing atoms
[DRfold2 subprocess] U64 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U64  with 12 missing atoms
[DRfold2 subprocess] C65 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C65  with 12 missing atoms
[DRfold2 subprocess] U66 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U66  with 12 missing atoms
[DRfold2 subprocess] G67 	15 missing atoms	 C2   C2'  C4   C5   C6   C8   N1   N2   N3   N7   O2'  O4'  O6   OP1  OP2
[DRfold2 subprocess] fill residue   G67  with 15 missing atoms
[DRfold2 subprocess] C68 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   N4   O2   O2'  O4'  OP1  OP2
[DRfold2 subprocess] fill residue   C68  with 12 missing atoms
[DRfold2 subprocess] U69 	12 missing atoms	 C2   C2'  C4   C5   C6   N3   O2   O2'  O4   O4'  OP1  OP2
[DRfold2 subprocess] fill residue   U69  with 12 missing atoms
[DRfold2 subprocess] t=0 moved=1446
[DRfold2 subprocess] t=1 moved=1920
[DRfold2 subprocess] t=2 moved=1798
[DRfold2 subprocess] t=3 moved=1768
[DRfold2 subprocess] t=4 moved=1691
[DRfold2 subprocess] t=5 moved=1651
[DRfold2 subprocess] t=6 moved=1646
[DRfold2 subprocess] t=7 moved=1614
[DRfold2 subprocess] t=8 moved=1571
[DRfold2 subprocess] t=9 moved=1562
[DRfold2 subprocess] t=10 moved=1578
[DRfold2 subprocess] t=11 moved=1580
[DRfold2 subprocess] t=12 moved=1554
[DRfold2 subprocess] t=13 moved=1580
[DRfold2 subprocess] t=14 moved=1590
[DRfold2 subprocess] t=15 moved=1574
[DRfold2 subprocess] t=16 moved=1549
[DRfold2 subprocess] t=17 moved=1547
[DRfold2 subprocess] t=18 moved=1553
[DRfold2 subprocess] t=19 moved=1531
[DRfold2 subprocess] t=20 moved=1534
[DRfold2 subprocess] t=21 moved=1539
[DRfold2 subprocess] t=22 moved=1521
[DRfold2 subprocess] t=23 moved=1529
[DRfold2 subprocess] t=24 moved=1534
[DRfold2 subprocess] t=25 moved=1529
[DRfold2 subprocess] t=26 moved=1527
[DRfold2 subprocess] t=27 moved=1540
[DRfold2 subprocess] t=28 moved=1517
[DRfold2 subprocess] t=29 moved=1520
[DRfold2 subprocess] t=30 moved=1525
[DRfold2 subprocess] t=31 moved=1521
[DRfold2 subprocess] t=32 moved=1532
[DRfold2 subprocess] t=33 moved=1511
[DRfold2 subprocess] t=34 moved=1524
[DRfold2 subprocess] t=35 moved=1518
[DRfold2 subprocess] t=36 moved=1507
[DRfold2 subprocess] t=37 moved=1521
[DRfold2 subprocess] t=38 moved=1493
[DRfold2 subprocess] t=39 moved=1513
[DRfold2 subprocess] t=40 moved=1517
[DRfold2 subprocess] t=41 moved=1522
[DRfold2 subprocess] t=42 moved=1521
[DRfold2 subprocess] t=43 moved=1498
[DRfold2 subprocess] t=44 moved=1504
[DRfold2 subprocess] t=45 moved=1509
[DRfold2 subprocess] t=46 moved=1521
[DRfold2 subprocess] t=47 moved=1526
[DRfold2 subprocess] t=48 moved=1520
[DRfold2 subprocess] t=49 moved=1500
[DRfold2 subprocess] t=50 moved=1498
[DRfold2 subprocess] t=51 moved=1519
[DRfold2 subprocess] t=52 moved=1502
[DRfold2 subprocess] t=53 moved=1490
[DRfold2 subprocess] t=54 moved=1513
[DRfold2 subprocess] t=55 moved=1502
[DRfold2 subprocess] t=56 moved=1505
[DRfold2 subprocess] t=57 moved=1506
[DRfold2 subprocess] t=58 moved=1485
[DRfold2 subprocess] t=59 moved=1499
[DRfold2 subprocess] t=60 moved=1494
[DRfold2 subprocess] t=61 moved=1495
[DRfold2 subprocess] t=62 moved=1490
[DRfold2 subprocess] t=63 moved=1507
[DRfold2 subprocess] t=64 moved=1483
[DRfold2 subprocess] t=65 moved=1507
[DRfold2 subprocess] t=66 moved=1502
[DRfold2 subprocess] t=67 moved=1499
[DRfold2 subprocess] t=68 moved=1504
[DRfold2 subprocess] t=69 moved=1490
[DRfold2 subprocess] t=70 moved=1486
[DRfold2 subprocess] t=71 moved=1487
[DRfold2 subprocess] t=72 moved=1491
[DRfold2 subprocess] t=73 moved=1495
[DRfold2 subprocess] t=74 moved=1489
[DRfold2 subprocess] t=75 moved=1517
[DRfold2 subprocess] t=76 moved=1507
[DRfold2 subprocess] t=77 moved=1497
[DRfold2 subprocess] t=78 moved=1498
[DRfold2 subprocess] t=79 moved=1503
[DRfold2 subprocess] t=80 moved=1500
[DRfold2 subprocess] t=81 moved=1505
[DRfold2 subprocess] t=82 moved=1496
[DRfold2 subprocess] t=83 moved=1495
[DRfold2 subprocess] t=84 moved=1500
[DRfold2 subprocess] t=85 moved=1483
[DRfold2 subprocess] t=86 moved=1502
[DRfold2 subprocess] t=87 moved=1497
[DRfold2 subprocess] t=88 moved=1487
[DRfold2 subprocess] t=89 moved=1485
[DRfold2 subprocess] t=90 moved=1501
[DRfold2 subprocess] t=91 moved=1491
[DRfold2 subprocess] t=92 moved=1499
[DRfold2 subprocess] t=93 moved=1494
[DRfold2 subprocess] t=94 moved=1495
[DRfold2 subprocess] t=95 moved=1495
[DRfold2 subprocess] t=96 moved=1500
[DRfold2 subprocess] t=97 moved=1494
[DRfold2 subprocess] t=98 moved=1488
[DRfold2 subprocess] t=99 moved=1485
[DRfold2 subprocess] t=100 moved=1497
[DRfold2 subprocess] t=101 moved=1489
[DRfold2 subprocess] t=102 moved=1510
[DRfold2 subprocess] stage1 t=0 moved=1590
[DRfold2 subprocess] stage1 t=1 moved=1464
[DRfold2 subprocess] stage1 t=2 moved=1195
[DRfold2 subprocess] stage1 t=3 moved=1049
[DRfold2 subprocess] stage1 t=4 moved=882
[DRfold2 subprocess] stage1 t=5 moved=785
[DRfold2 subprocess] stage1 t=6 moved=680
[DRfold2 subprocess] stage1 t=7 moved=684
[DRfold2 subprocess] stage1 t=8 moved=594
[DRfold2 subprocess] stage1 t=9 moved=616
[DRfold2 subprocess] stage1 t=10 moved=482
[DRfold2 subprocess] stage1 t=11 moved=478
[DRfold2 subprocess] stage1 t=12 moved=485
[DRfold2 subprocess] stage1 t=13 moved=422
[DRfold2 subprocess] stage1 t=14 moved=422
[DRfold2 subprocess] stage1 t=15 moved=420
[DRfold2 subprocess] stage1 t=16 moved=380
[DRfold2 subprocess] stage1 t=17 moved=357
[DRfold2 subprocess] stage1 t=18 moved=397
[DRfold2 subprocess] stage1 t=19 moved=391
[DRfold2 subprocess] stage1 t=20 moved=345
[DRfold2 subprocess] stage1 t=21 moved=347
[DRfold2 subprocess] stage1 t=22 moved=349
[DRfold2 subprocess] stage1 t=23 moved=314
[DRfold2 subprocess] stage1 t=24 moved=307
[DRfold2 subprocess] stage1 t=25 moved=284
[DRfold2 subprocess] stage1 t=26 moved=238
[DRfold2 subprocess] stage1 t=27 moved=248
[DRfold2 subprocess] stage1 t=28 moved=234
[DRfold2 subprocess] stage1 t=29 moved=235
[DRfold2 subprocess] stage1 t=30 moved=223
[DRfold2 subprocess] stage1 t=31 moved=252
[DRfold2 subprocess] stage1 t=32 moved=253
[DRfold2 subprocess] stage1 t=33 moved=241
[DRfold2 subprocess] stage1 t=34 moved=222
[DRfold2 subprocess] stage1 t=35 moved=209
[DRfold2 subprocess] stage1 t=36 moved=220
[DRfold2 subprocess] stage1 t=37 moved=257
[DRfold2 subprocess] stage1 t=38 moved=229
[DRfold2 subprocess] stage1 t=39 moved=231
[DRfold2 subprocess] stage1 t=40 moved=193
[DRfold2 subprocess] stage1 t=41 moved=165
[DRfold2 subprocess] stage1 t=42 moved=184
[DRfold2 subprocess] stage1 t=43 moved=180
[DRfold2 subprocess] stage1 t=44 moved=160
[DRfold2 subprocess] stage1 t=45 moved=180
[DRfold2 subprocess] stage1 t=46 moved=158
[DRfold2 subprocess] stage1 t=47 moved=149
[DRfold2 subprocess] stage1 t=48 moved=173
[DRfold2 subprocess] stage1 t=49 moved=125
[DRfold2 subprocess] stage1 t=50 moved=178
[DRfold2 subprocess] stage1 t=51 moved=132
[DRfold2 subprocess] stage1 t=52 moved=116
[DRfold2 subprocess] stage1 t=53 moved=124
[DRfold2 subprocess] stage1 t=54 moved=103
[DRfold2 subprocess] stage1 t=55 moved=130
[DRfold2 subprocess] stage1 t=56 moved=100
[DRfold2 subprocess] stage1 t=57 moved=96
[DRfold2 subprocess] stage1 t=58 moved=115
[DRfold2 subprocess] stage1 t=59 moved=92
[DRfold2 subprocess] stage1 t=60 moved=88
[DRfold2 subprocess] stage1 t=61 moved=121
[DRfold2 subprocess] stage1 t=62 moved=134
[DRfold2 subprocess] stage1 t=63 moved=127
[DRfold2 subprocess] stage1 t=64 moved=135
[DRfold2 subprocess] stage1 t=65 moved=100
[DRfold2 subprocess] stage1 t=66 moved=120
[DRfold2 subprocess] stage1 t=67 moved=120
[DRfold2 subprocess] stage1 t=68 moved=133
[DRfold2 subprocess] stage1 t=69 moved=115
[DRfold2 subprocess] stage1 t=70 moved=85
[DRfold2 subprocess] stage1 t=71 moved=90
[DRfold2 subprocess] stage1 t=72 moved=91
[DRfold2 subprocess] stage1 t=73 moved=69
[DRfold2 subprocess] stage1 t=74 moved=96
[DRfold2 subprocess] stage1 t=75 moved=64
[DRfold2 subprocess] stage1 t=76 moved=52
[DRfold2 subprocess] stage1 t=77 moved=60
[DRfold2 subprocess] stage1 t=78 moved=58
[DRfold2 subprocess] stage1 t=79 moved=47
[DRfold2 subprocess] stage1 t=80 moved=38
[DRfold2 subprocess] stage1 t=81 moved=79
[DRfold2 subprocess] stage1 t=82 moved=72
[DRfold2 subprocess] stage1 t=83 moved=97
[DRfold2 subprocess] stage1 t=84 moved=83
[DRfold2 subprocess] stage1 t=85 moved=101
[DRfold2 subprocess] stage1 t=86 moved=47
[DRfold2 subprocess] stage1 t=87 moved=42
[DRfold2 subprocess] stage1 t=88 moved=48
[DRfold2 subprocess] stage1 t=89 moved=43
[DRfold2 subprocess] stage1 t=90 moved=32
[DRfold2 subprocess] stage1 t=91 moved=29
[DRfold2 subprocess] stage1 t=92 moved=24
[DRfold2 subprocess] stage1 t=93 moved=16
[DRfold2 subprocess] stage1 t=94 moved=18
[DRfold2 subprocess] stage1 t=95 moved=19
[DRfold2 subprocess] stage1 t=96 moved=41
[DRfold2 subprocess] stage1 t=97 moved=18
[DRfold2 subprocess] stage1 t=98 moved=31
[DRfold2 subprocess] stage1 t=99 moved=31
[DRfold2 subprocess] stage1 t=100 moved=17
[DRfold2 subprocess] stage1 t=101 moved=15
[DRfold2 subprocess] stage1 t=102 moved=15
[DRfold2 subprocess] stage2 t=0 moved=422
[DRfold2 subprocess] stage2 t=1 moved=804
[DRfold2 subprocess] stage2 t=2 moved=812
[DRfold2 subprocess] stage2 t=3 moved=700
[DRfold2 subprocess] stage2 t=4 moved=632
[DRfold2 subprocess] stage2 t=5 moved=559
[DRfold2 subprocess] stage2 t=6 moved=489
[DRfold2 subprocess] stage2 t=7 moved=513
[DRfold2 subprocess] stage2 t=8 moved=470
[DRfold2 subprocess] stage2 t=9 moved=448
[DRfold2 subprocess] stage2 t=10 moved=382
[DRfold2 subprocess] stage2 t=11 moved=418
[DRfold2 subprocess] stage2 t=12 moved=326
[DRfold2 subprocess] stage2 t=13 moved=311
[DRfold2 subprocess] stage2 t=14 moved=321
[DRfold2 subprocess] stage2 t=15 moved=259
[DRfold2 subprocess] stage2 t=16 moved=287
[DRfold2 subprocess] stage2 t=17 moved=352
[DRfold2 subprocess] stage2 t=18 moved=272
[DRfold2 subprocess] stage2 t=19 moved=332
[DRfold2 subprocess] stage2 t=20 moved=263
[DRfold2 subprocess] stage2 t=21 moved=281
[DRfold2 subprocess] stage2 t=22 moved=375
[DRfold2 subprocess] stage2 t=23 moved=368
[DRfold2 subprocess] stage2 t=24 moved=317
[DRfold2 subprocess] stage2 t=25 moved=294
[DRfold2 subprocess] stage2 t=26 moved=343
[DRfold2 subprocess] stage2 t=27 moved=219
[DRfold2 subprocess] stage2 t=28 moved=224
[DRfold2 subprocess] stage2 t=29 moved=171
[DRfold2 subprocess] stage2 t=30 moved=198
[DRfold2 subprocess] stage2 t=31 moved=209
[DRfold2 subprocess] stage2 t=32 moved=203
[DRfold2 subprocess] stage2 t=33 moved=202
[DRfold2 subprocess] stage2 t=34 moved=251
[DRfold2 subprocess] stage2 t=35 moved=275
[DRfold2 subprocess] stage2 t=36 moved=290
[DRfold2 subprocess] stage2 t=37 moved=279
[DRfold2 subprocess] stage2 t=38 moved=268
[DRfold2 subprocess] stage2 t=39 moved=252
[DRfold2 subprocess] stage2 t=40 moved=203
[DRfold2 subprocess] stage2 t=41 moved=180
[DRfold2 subprocess] stage2 t=42 moved=222
[DRfold2 subprocess] stage2 t=43 moved=200
[DRfold2 subprocess] stage2 t=44 moved=212
[DRfold2 subprocess] stage2 t=45 moved=104
[DRfold2 subprocess] stage2 t=46 moved=84
[DRfold2 subprocess] stage2 t=47 moved=114
[DRfold2 subprocess] stage2 t=48 moved=121
[DRfold2 subprocess] stage2 t=49 moved=184
[DRfold2 subprocess] stage2 t=50 moved=111
[DRfold2 subprocess] stage2 t=51 moved=133
[DRfold2 subprocess] stage2 t=52 moved=161
[DRfold2 subprocess] stage2 t=53 moved=150
[DRfold2 subprocess] stage2 t=54 moved=176
[DRfold2 subprocess] stage2 t=55 moved=108
[DRfold2 subprocess] stage2 t=56 moved=158
[DRfold2 subprocess] stage2 t=57 moved=209
[DRfold2 subprocess] stage2 t=58 moved=153
[DRfold2 subprocess] stage2 t=59 moved=182
[DRfold2 subprocess] stage2 t=60 moved=140
[DRfold2 subprocess] stage2 t=61 moved=167
[DRfold2 subprocess] stage2 t=62 moved=143
[DRfold2 subprocess] stage2 t=63 moved=134
[DRfold2 subprocess] stage2 t=64 moved=99
[DRfold2 subprocess] stage2 t=65 moved=177
[DRfold2 subprocess] stage2 t=66 moved=172
[DRfold2 subprocess] stage2 t=67 moved=159
[DRfold2 subprocess] stage2 t=68 moved=114
[DRfold2 subprocess] stage2 t=69 moved=123
[DRfold2 subprocess] stage2 t=70 moved=95
[DRfold2 subprocess] stage2 t=71 moved=110
[DRfold2 subprocess] stage2 t=72 moved=109
[DRfold2 subprocess] stage2 t=73 moved=145
[DRfold2 subprocess] stage2 t=74 moved=160
[DRfold2 subprocess] stage2 t=75 moved=165
[DRfold2 subprocess] stage2 t=76 moved=182
[DRfold2 subprocess] stage2 t=77 moved=153
[DRfold2 subprocess] stage2 t=78 moved=102
[DRfold2 subprocess] stage2 t=79 moved=137
[DRfold2 subprocess] stage2 t=80 moved=116
[DRfold2 subprocess] stage2 t=81 moved=90
[DRfold2 subprocess] stage2 t=82 moved=86
[DRfold2 subprocess] stage2 t=83 moved=84
[DRfold2 subprocess] stage2 t=84 moved=95
[DRfold2 subprocess] stage2 t=85 moved=66
[DRfold2 subprocess] stage2 t=86 moved=106
[DRfold2 subprocess] stage2 t=87 moved=91
[DRfold2 subprocess] stage2 t=88 moved=72
[DRfold2 subprocess] stage2 t=89 moved=139
[DRfold2 subprocess] stage2 t=90 moved=153
[DRfold2 subprocess] stage2 t=91 moved=66
[DRfold2 subprocess] stage2 t=92 moved=86
[DRfold2 subprocess] stage2 t=93 moved=77
[DRfold2 subprocess] stage2 t=94 moved=62
[DRfold2 subprocess] stage2 t=95 moved=54
[DRfold2 subprocess] stage2 t=96 moved=43
[DRfold2 subprocess] stage2 t=97 moved=38
[DRfold2 subprocess] stage2 t=98 moved=11
[DRfold2 subprocess] stage2 t=99 moved=72
[DRfold2 subprocess] stage2 t=100 moved=25
[DRfold2 subprocess] stage2 t=101 moved=85
[DRfold2 subprocess] stage2 t=102 moved=60
[DRfold2 subprocess] stage3 t=0 moved=51
[DRfold2 subprocess] stage3 t=1 moved=50
[DRfold2 subprocess] stage3 t=2 moved=30
[DRfold2 subprocess] stage3 t=3 moved=51
[DRfold2 subprocess] stage3 t=4 moved=49
[DRfold2 subprocess] stage3 t=5 moved=59
[DRfold2 subprocess] stage3 t=6 moved=52
[DRfold2 subprocess] stage3 t=7 moved=67
[DRfold2 subprocess] stage3 t=8 moved=60
[DRfold2 subprocess] stage3 t=9 moved=45
[DRfold2 subprocess] stage3 t=10 moved=59
[DRfold2 subprocess] stage3 t=11 moved=62
[DRfold2 subprocess] stage3 t=12 moved=24
[DRfold2 subprocess] stage3 t=13 moved=37
[DRfold2 subprocess] stage3 t=14 moved=35
[DRfold2 subprocess] stage3 t=15 moved=16
[DRfold2 subprocess] stage3 t=16 moved=0
[DRfold2] Running refinement for cluster 4 completed successfully
[DRfold2] PREDICTION PIPELINE COMPLETED SUCCESSFULLY
Processing target 2/12: R1108 (69 nt), elapsed: 266.2s, est. remaining: 2928.4s, time left: 24880.8s
Using template approach for R1108 (index out of DRfold range)
Processing target 12/12: R1190 (118 nt), elapsed: 269.4s, est. remaining: 24.5s, time left: 24877.6s
Using template approach for R1190 (index out of DRfold range)
Processing target 11/12: R1189 (118 nt), elapsed: 276.3s, est. remaining: 55.3s, time left: 24870.7s
Using template approach for R1189 (index out of DRfold range)
Processing target 9/12: R1149 (124 nt), elapsed: 283.2s, est. remaining: 141.6s, time left: 24863.8s
Using template approach for R1149 (index out of DRfold range)
Processing target 10/12: R1156 (135 nt), elapsed: 290.1s, est. remaining: 96.7s, time left: 24857.0s
Using template approach for R1156 (index out of DRfold range)
Processing target 3/12: R1116 (157 nt), elapsed: 295.9s, est. remaining: 1479.6s, time left: 24851.1s
Using template approach for R1116 (index out of DRfold range)
Processing target 6/12: R1128 (238 nt), elapsed: 302.6s, est. remaining: 423.7s, time left: 24844.4s
Using template approach for R1128 (index out of DRfold range)
Processing target 5/12: R1126 (363 nt), elapsed: 307.7s, est. remaining: 615.4s, time left: 24839.3s
Using template approach for R1126 (index out of DRfold range)
Processing target 7/12: R1136 (374 nt), elapsed: 312.1s, est. remaining: 312.1s, time left: 24835.0s
Using template approach for R1136 (index out of DRfold range)
Processing target 8/12: R1138 (720 nt), elapsed: 316.6s, est. remaining: 226.1s, time left: 24830.5s
Using template approach for R1138 (index out of DRfold range)
Generated predictions for 12 RNA sequences
Used DRfold2 for 1 targets and template approach for 11 targets
Total runtime: 384.0 seconds
submission
ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
0	R1107_1	G	1	6.50434	-12.16070	-11.78114	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	R1107_2	G	2	6.28621	-7.02678	-13.83277	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2	R1107_3	G	3	4.02922	-1.92618	-14.61284	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
3	R1107_4	G	4	0.78219	2.16863	-12.73632	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
4	R1107_5	G	5	-1.83363	4.35521	-8.35424	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2510	R1189_114	U	114	-19.90772	-9.26861	10.50842	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2511	R1189_115	U	115	-21.97411	-9.46059	5.59679	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2512	R1189_116	U	116	-23.58990	-7.85262	0.52375	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2513	R1189_117	U	117	-23.07915	-4.46932	-3.61029	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2514	R1189_118	U	118	-20.21993	-0.85038	-6.12332	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2515 rows × 18 columns

submission_df
ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
0	R1117v2_1	U	1	-8.664709	-19.583431	7.126974	-6.547215	9.436258	6.500803	-1.000000e+18	-1.000000e+18	-1.000000e+18	8.650256	-6.907680	4.414479	38.903601	22.245334	62.388446
1	R1117v2_2	U	2	-14.527092	-19.694456	7.180366	-12.625456	9.547409	6.413443	-1.600000e+17	-1.600000e+17	-1.600000e+17	10.596878	-8.954947	-0.928671	37.301625	24.325268	66.375418
2	R1117v2_3	G	3	-17.788505	-19.152800	11.749251	-7.658676	11.079351	4.790899	-2.560000e+16	-2.560000e+16	-2.560000e+16	11.102577	-8.515493	-6.276471	33.214319	24.770646	71.335372
3	R1117v2_4	G	4	-20.690249	-15.619536	14.320118	-2.345248	11.182468	5.787371	-4.096000e+15	-4.096000e+15	-4.096000e+15	12.215663	-4.578998	-9.876159	30.185025	22.255375	75.036588
4	R1117v2_5	G	5	-22.435796	-10.659642	15.867694	2.166948	10.728666	8.333066	-6.553600e+14	-6.553600e+14	-6.553600e+14	13.326376	0.001679	-11.304471	28.603161	17.308122	77.238682
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2510	R1138_716	G	716	209.997237	158.867917	212.020732	167.326248	200.415491	176.966336	2.100186e+02	1.587842e+02	2.120004e+02	248.307198	278.315776	293.577825	160.065856	212.493885	131.380302
2511	R1138_717	A	717	206.465078	161.400042	208.916174	169.236752	196.013280	174.420871	2.063621e+02	1.613092e+02	2.089734e+02	249.208052	274.829362	296.941357	156.234882	211.027252	135.472738
2512	R1138_718	A	718	204.691701	165.830376	206.820732	168.994366	190.746508	172.381864	2.047219e+02	1.657395e+02	2.068234e+02	250.966021	269.683328	299.728486	156.192794	207.647136	139.696765
2513	R1138_719	U	719	205.262385	170.933876	205.665501	166.136190	186.448242	170.630504	2.051196e+02	1.709664e+02	2.055306e+02	252.677368	264.821691	304.109816	158.287361	203.949870	143.466904
2514	R1138_720	U	720	207.800732	175.425306	204.124851	161.422395	184.199749	168.782041	2.079572e+02	1.755582e+02	2.040918e+02	253.437145	259.917735	307.693913	160.575016	200.863072	145.335745
2515 rows × 18 columns

Post Processing:
%cd /kaggle/working/
/kaggle/working
!ls
boltz	     inference.py	 outputs_prediction    submission_dr.csv
DRfold2      inputs_prediction	 predictions
fasta_files  __notebook__.ipynb  submission_boltz.csv
import pandas as pd

# Load the data
submission_boltz = pd.read_csv('submission_boltz.csv')
submission_dr = pd.read_csv('submission_dr.csv')

# Function to extract target_id from ID column
def extract_target_id(id_string):
    """Extract target_id by removing the suffix (everything after the last underscore)"""
    return id_string.rsplit('_', 1)[0]

# Add target_id column to both dataframes
submission_boltz['target_id'] = submission_boltz['ID'].apply(extract_target_id)
submission_dr['target_id'] = submission_dr['ID'].apply(extract_target_id)

# Calculate sequence lengths
test_sequences['sequence_length'] = test_sequences['sequence'].str.len()

# Get target_ids with sequence length > 600
long_sequences = test_sequences[test_sequences['sequence_length'] > 600]['target_id'].tolist()

print(f"Found {len(long_sequences)} sequences with length > 600:")
for seq_id in long_sequences:
    seq_len = test_sequences[test_sequences['target_id'] == seq_id]['sequence_length'].iloc[0]
    print(f"  {seq_id}: {seq_len} nucleotides")

# Create a copy of submission_dr for modification
submission_processed = submission_dr.copy()

# Define a threshold for detecting placeholder values (e.g., values < -1e10)
PLACEHOLDER_THRESHOLD = -1e17

# For each row in submission_dr, check if it belongs to a long sequence
for idx, row in submission_dr.iterrows():
    target_id = row['target_id']

    if target_id in long_sequences:
        # Find corresponding row in submission_boltz
        boltz_row = submission_boltz[submission_boltz['ID'] == row['ID']]

        if not boltz_row.empty:
            boltz_x1 = boltz_row.iloc[0]['x_1']
            boltz_y1 = boltz_row.iloc[0]['y_1']
            boltz_z1 = boltz_row.iloc[0]['z_1']

            # Always replace the first conformation
            submission_processed.loc[idx, 'x_1'] = boltz_x1
            submission_processed.loc[idx, 'y_1'] = boltz_y1
            submission_processed.loc[idx, 'z_1'] = boltz_z1

            # Check and replace placeholder values in other conformations
            for conf_num in [2, 3, 4, 5]:
                x_col = f'x_{conf_num}'
                y_col = f'y_{conf_num}'
                z_col = f'z_{conf_num}'

                # If any coordinate in this conformation is a placeholder, replace all three
                if (row[x_col] < PLACEHOLDER_THRESHOLD or
                    row[y_col] < PLACEHOLDER_THRESHOLD or
                    row[z_col] < PLACEHOLDER_THRESHOLD):

                    submission_processed.loc[idx, x_col] = boltz_x1
                    submission_processed.loc[idx, y_col] = boltz_y1
                    submission_processed.loc[idx, z_col] = boltz_z1

# Remove the temporary target_id column
submission_processed = submission_processed.drop('target_id', axis=1)
Found 1 sequences with length > 600:
  R1138: 720 nucleotides
submission_processed
ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
0	R1117v2_1	U	1	-8.664709	-19.583431	7.126974	-6.547215	9.436258	6.500803	-1.000000e+18	-1.000000e+18	-1.000000e+18	8.650256	-6.907680	4.414479	38.903601	22.245334	62.388446
1	R1117v2_2	U	2	-14.527092	-19.694456	7.180366	-12.625456	9.547409	6.413443	-1.600000e+17	-1.600000e+17	-1.600000e+17	10.596878	-8.954947	-0.928671	37.301625	24.325268	66.375418
2	R1117v2_3	G	3	-17.788505	-19.152800	11.749251	-7.658676	11.079351	4.790899	-2.560000e+16	-2.560000e+16	-2.560000e+16	11.102577	-8.515493	-6.276471	33.214319	24.770646	71.335372
3	R1117v2_4	G	4	-20.690249	-15.619536	14.320118	-2.345248	11.182468	5.787371	-4.096000e+15	-4.096000e+15	-4.096000e+15	12.215663	-4.578998	-9.876159	30.185025	22.255375	75.036588
4	R1117v2_5	G	5	-22.435796	-10.659642	15.867694	2.166948	10.728666	8.333066	-6.553600e+14	-6.553600e+14	-6.553600e+14	13.326376	0.001679	-11.304471	28.603161	17.308122	77.238682
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2510	R1138_716	G	716	30.045710	-4.915490	-20.897400	167.326248	200.415491	176.966336	2.100186e+02	1.587842e+02	2.120004e+02	248.307198	278.315776	293.577825	160.065856	212.493885	131.380302
2511	R1138_717	A	717	26.133060	-2.656440	-17.677370	169.236752	196.013280	174.420871	2.063621e+02	1.613092e+02	2.089734e+02	249.208052	274.829362	296.941357	156.234882	211.027252	135.472738
2512	R1138_718	A	718	24.489350	-0.723280	-13.104610	168.994366	190.746508	172.381864	2.047219e+02	1.657395e+02	2.068234e+02	250.966021	269.683328	299.728486	156.192794	207.647136	139.696765
2513	R1138_719	U	719	25.277100	0.750210	-8.062750	166.136190	186.448242	170.630504	2.051196e+02	1.709664e+02	2.055306e+02	252.677368	264.821691	304.109816	158.287361	203.949870	143.466904
2514	R1138_720	U	720	28.056160	2.245200	-3.710250	161.422395	184.199749	168.782041	2.079572e+02	1.755582e+02	2.040918e+02	253.437145	259.917735	307.693913	160.575016	200.863072	145.335745
2515 rows × 18 columns

!ls
boltz	     inference.py	 outputs_prediction    submission_dr.csv
DRfold2      inputs_prediction	 predictions
fasta_files  __notebook__.ipynb  submission_boltz.csv
%cd /kaggle/working/
/kaggle/working
!rm -rf ./*
!ls
__notebook__.ipynb
# Save the processed sub
submission_processed.to_csv('submission.csv', index=False)

# Print summary of changes
total_rows_modified = 0
for target_id in long_sequences:
    rows_for_target = submission_dr[submission_dr['target_id'] == target_id].shape[0]
    total_rows_modified += rows_for_target
    print(f"Modified {rows_for_target} rows for {target_id}")

print(f"\nTotal rows modified: {total_rows_modified}")
print(f"Total rows in submission_dr: {len(submission_dr)}")
print(f"Processed submission_dr saved as 'submission.csv'")
Modified 720 rows for R1138

Total rows modified: 720
Total rows in submission_dr: 2515
Processed submission_dr saved as 'submission.csv'
!ls
__notebook__.ipynb  submission.csv
submission_processed
ID	resname	resid	x_1	y_1	z_1	x_2	y_2	z_2	x_3	y_3	z_3	x_4	y_4	z_4	x_5	y_5	z_5
0	R1117v2_1	U	1	-8.664709	-19.583431	7.126974	-6.547215	9.436258	6.500803	-1.000000e+18	-1.000000e+18	-1.000000e+18	8.650256	-6.907680	4.414479	38.903601	22.245334	62.388446
1	R1117v2_2	U	2	-14.527092	-19.694456	7.180366	-12.625456	9.547409	6.413443	-1.600000e+17	-1.600000e+17	-1.600000e+17	10.596878	-8.954947	-0.928671	37.301625	24.325268	66.375418
2	R1117v2_3	G	3	-17.788505	-19.152800	11.749251	-7.658676	11.079351	4.790899	-2.560000e+16	-2.560000e+16	-2.560000e+16	11.102577	-8.515493	-6.276471	33.214319	24.770646	71.335372
3	R1117v2_4	G	4	-20.690249	-15.619536	14.320118	-2.345248	11.182468	5.787371	-4.096000e+15	-4.096000e+15	-4.096000e+15	12.215663	-4.578998	-9.876159	30.185025	22.255375	75.036588
4	R1117v2_5	G	5	-22.435796	-10.659642	15.867694	2.166948	10.728666	8.333066	-6.553600e+14	-6.553600e+14	-6.553600e+14	13.326376	0.001679	-11.304471	28.603161	17.308122	77.238682
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2510	R1138_716	G	716	30.045710	-4.915490	-20.897400	167.326248	200.415491	176.966336	2.100186e+02	1.587842e+02	2.120004e+02	248.307198	278.315776	293.577825	160.065856	212.493885	131.380302
2511	R1138_717	A	717	26.133060	-2.656440	-17.677370	169.236752	196.013280	174.420871	2.063621e+02	1.613092e+02	2.089734e+02	249.208052	274.829362	296.941357	156.234882	211.027252	135.472738
2512	R1138_718	A	718	24.489350	-0.723280	-13.104610	168.994366	190.746508	172.381864	2.047219e+02	1.657395e+02	2.068234e+02	250.966021	269.683328	299.728486	156.192794	207.647136	139.696765
2513	R1138_719	U	719	25.277100	0.750210	-8.062750	166.136190	186.448242	170.630504	2.051196e+02	1.709664e+02	2.055306e+02	252.677368	264.821691	304.109816	158.287361	203.949870	143.466904
2514	R1138_720	U	720	28.056160	2.245200	-3.710250	161.422395	184.199749	168.782041	2.079572e+02	1.755582e+02	2.040918e+02	253.437145	259.917735	307.693913	160.575016	200.863072	145.335745