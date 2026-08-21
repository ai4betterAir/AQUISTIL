#!/usr/bin/env python3
"""Create a deterministic archive and checksum manifest for frozen AQUISTIL."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import config_spatial as config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _snapshot_files() -> list[Path]:
    files = []
    for path in CODE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name.startswith("X1_output_"):
            continue
        files.append(path)
    for pattern in ("README.md", "01_run.sl", "requirements*.txt"):
        files.extend(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def _write_deterministic_archive(archive_path: Path, files: list[Path], tag: str) -> None:
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(PROJECT_ROOT)
                    info = archive.gettarinfo(
                        str(path), arcname=(Path(tag) / relative).as_posix()
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=config.FROZEN_RELEASE_TAG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = PROJECT_ROOT / "Frozen_Releases" / args.tag
    archive_path = release_dir / f"{args.tag}.tar.gz"
    manifest_path = release_dir / "manifest.json"
    if release_dir.exists() and not args.force:
        raise FileExistsError(f"Release already exists: {release_dir}")
    release_dir.mkdir(parents=True, exist_ok=True)

    files = _snapshot_files()
    _write_deterministic_archive(archive_path, files, args.tag)
    file_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in files
    }
    core_files = (
        "code/config_spatial.py",
        "code/Model/AQUISTIL.py",
        "code/Model/LightGBM.py",
        "code/main.py",
        "code/regional_imputation.py",
        "code/missingness_regimes.py",
    )
    manifest = {
        "release_tag": args.tag,
        "git_base_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "snapshot_file_count": len(files),
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "core_file_sha256": {path: file_hashes[path] for path in core_files},
        "all_file_sha256": file_hashes,
        "protocol": {
            "development_regions": list(config.DEVELOPMENT_REGIONS),
            "held_out_validation_regions": list(config.HELD_OUT_VALIDATION_REGIONS),
            "held_out_regions_without_inputs": list(config.HELD_OUT_REGIONS_WITHOUT_INPUTS),
            "models": list(config.MODELS_TO_RUN),
            "targets": list(config.TARGET_COLUMNS),
            "missingness_regimes": list(config.MISSINGNESS_REGIMES),
            "missingness_levels": list(config.MISSINGNESS_LEVELS),
            "seeds": list(config.REGIONAL_EVALUATION_SEEDS),
            "gap_expert_min_run_length": config.AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH,
            "output_directory": config.FROZEN_OUTPUT_DIRECTORY,
        },
        "relevant_git_status": _git(
            "status", "--short", "--", "code", "01_run.sl", "README.md"
        ).splitlines(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen archive: {archive_path}")
    print(f"Archive SHA256: {manifest['archive_sha256']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
