"""
Weight resolution for hosted deployments.

Two failure modes bite when this project leaves a laptop, and both produce
confusing errors far from their cause:

1. **Unresolved LFS pointers.** `best_sar_model.pth` is 105 MB and must travel
   through Git LFS. Platforms that clone without LFS support (Streamlit
   Community Cloud among them) leave a ~130-byte text stub on disk with the
   right filename. `torch.load` then fails with an opaque unpickling error.
   We detect the stub and say plainly what happened.

2. **Weights deliberately kept out of the repo.** Set `POSEATSEA_WEIGHTS_REPO`
   to a Hugging Face Hub repo id and missing files are fetched at first use,
   which lets the same code deploy anywhere without a 105 MB blob in git.

When the files are simply present and valid -- the normal local case -- every
function here is a cheap stat() and nothing else happens.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

# Git LFS writes this ASCII header in place of the real blob.
_LFS_HEADER = b"version https://git-lfs"
_MAX_POINTER_BYTES = 1024


def is_lfs_pointer(path: Path) -> bool:
    """True if `path` is an unresolved Git LFS stub rather than real content."""
    try:
        if path.stat().st_size > _MAX_POINTER_BYTES:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_HEADER)) == _LFS_HEADER
    except OSError:
        return False


DEFAULT_WEIGHTS_REPO = "23f2003521/poseatsea-weights"


def _download(filename: str, dest_dir: Path) -> Path:
    repo = os.getenv("POSEATSEA_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if not repo:
        raise FileNotFoundError(
            f"{filename} is missing from {dest_dir} and POSEATSEA_WEIGHTS_REPO is not "
            f"set, so it cannot be fetched. Either commit the weights via Git LFS or "
            f"set POSEATSEA_WEIGHTS_REPO to a Hugging Face repo id holding them."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "POSEATSEA_WEIGHTS_REPO is set but huggingface_hub is not installed. "
            "Add `huggingface-hub` to requirements.txt."
        ) from exc

    repo_type = os.getenv("POSEATSEA_WEIGHTS_REPO_TYPE", "model")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    cached = hf_hub_download(repo_id=repo, filename=filename,
                             repo_type=repo_type, token=token)
    return Path(cached)


def resolve(path: Path) -> Path:
    """
    Return a usable path to one weight file, fetching it if need be.

    Raises with an actionable message rather than letting a pointer stub reach
    `torch.load` and surface as a pickle error.
    """
    if path.exists() and not is_lfs_pointer(path):
        return path

    repo = os.getenv("POSEATSEA_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if path.exists() and is_lfs_pointer(path):
        if repo:
            return _download(path.name, path.parent)
        raise RuntimeError(
            f"{path.name} is an unresolved Git LFS pointer, not the real weights "
            f"({path.stat().st_size} bytes on disk).\n"
            f"  - Cloning locally?  run `git lfs install && git lfs pull`.\n"
            f"  - On a host without LFS support (e.g. Streamlit Community Cloud), "
            f"set POSEATSEA_WEIGHTS_REPO to a Hugging Face repo holding the weights."
        )

    return _download(path.name, path.parent)


def check_all() -> List[str]:
    """Report problems with every required artefact. Empty list means healthy."""
    from .config import AIS_AE_WEIGHTS, AIS_SCALER, SAR_WEIGHTS, TRAJECTORY_WEIGHTS

    problems: List[str] = []
    for path in (SAR_WEIGHTS, TRAJECTORY_WEIGHTS, AIS_AE_WEIGHTS, AIS_SCALER):
        if not path.exists():
            problems.append(f"{path.name}: missing from {path.parent}")
        elif is_lfs_pointer(path):
            problems.append(f"{path.name}: unresolved Git LFS pointer -- run `git lfs pull`")
    return problems
