"""Download the competition data into the location used by this repository."""
from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
from dotenv import load_dotenv
from kagglehub.exceptions import UnauthenticatedError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "stanford-rna-3d-folding"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination directory (default: repo data directory)",
    )
    args = parser.parse_args()

    # Support both a repo-local .env and a shared workspace-level .env. Values
    # already present in the process environment always take precedence.
    for env_file in (REPO_ROOT / ".env", REPO_ROOT.parent / ".env"):
        if env_file.is_file():
            load_dotenv(env_file, override=False)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = kagglehub.competition_download(
            "stanford-rna-3d-folding",
            output_dir=str(output_dir),
        )
    except UnauthenticatedError as exc:
        raise SystemExit(
            "Kaggle authentication is required. Put KAGGLE_API_TOKEN in the "
            "repo's .env file or save the token at ~/.kaggle/access_token, "
            "then run this script again."
        ) from exc
    print(path)


if __name__ == "__main__":
    main()
