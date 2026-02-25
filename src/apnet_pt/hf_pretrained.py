import os
import sys
from importlib import resources

HF_REPO_ID = "awallace3/qcmlforge"
_DOWNLOAD_APPROVED = None


def _hf_hub_download(rel_path: str, local_files_only: bool) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load pretrained models. "
            "Install qcmlforge dependencies or `pip install huggingface_hub`."
        ) from exc

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=rel_path,
        local_files_only=local_files_only,
    )


def _packaged_model_path(rel_path: str) -> str | None:
    model_path = resources.files("apnet_pt").joinpath("models", *rel_path.split("/"))
    return str(model_path) if model_path.is_file() else None


def _allow_model_download(missing_paths: list[str]) -> bool:
    global _DOWNLOAD_APPROVED

    if _DOWNLOAD_APPROVED is not None:
        return _DOWNLOAD_APPROVED

    env_value = os.getenv("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        _DOWNLOAD_APPROVED = True
        return True
    if env_value in {"0", "false", "no", "n"}:
        _DOWNLOAD_APPROVED = False
        return False

    if not sys.stdin.isatty():
        _DOWNLOAD_APPROVED = False
        return False

    preview = ", ".join(missing_paths[:3])
    if len(missing_paths) > 3:
        preview += ", ..."
    answer = (
        input(
            "Pretrained model weights are not available locally and need to be "
            f"downloaded from https://huggingface.co/{HF_REPO_ID} "
            f"(missing: {preview}). Download now? [y/N]: "
        )
        .strip()
        .lower()
    )
    _DOWNLOAD_APPROVED = answer in {"y", "yes"}
    return _DOWNLOAD_APPROVED


def resolve_pretrained_paths(rel_paths: list[str]) -> dict[str, str]:
    resolved = {}
    missing = []

    for rel_path in rel_paths:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=True)
        except Exception:
            missing.append(rel_path)

    if not missing:
        return resolved

    if not _allow_model_download(missing):
        for rel_path in missing:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                "Missing pretrained model in local cache. "
                "Set QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1 to auto-download, "
                f"or run interactively and accept download for '{rel_path}'."
            )
        return resolved

    for rel_path in missing:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=False)
        except Exception as exc:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                f"Unable to load pretrained model '{rel_path}' from "
                f"https://huggingface.co/{HF_REPO_ID}."
            ) from exc

    return resolved


def resolve_pretrained_path(rel_path: str) -> str:
    return resolve_pretrained_paths([rel_path])[rel_path]
