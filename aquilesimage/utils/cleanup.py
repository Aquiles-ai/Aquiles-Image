import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def resolve_image_dir(image_dir: Optional[str] = None) -> Path:
    if image_dir:
        return Path(image_dir).expanduser().resolve()

    env_dir = os.getenv("AQUILES_IMAGE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return Path(tempfile.gettempdir(), "images").resolve()


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def total_size(paths: Iterable[Path]) -> int:
    size = 0
    for path in paths:
        try:
            size += path.stat().st_size
        except OSError:
            pass
    return size


def find_images(image_dir: Path, older_than_seconds: Optional[float] = None) -> List[Path]:
    if not image_dir.is_dir():
        return []

    now = time.time()
    images = []
    for path in image_dir.glob("img*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if older_than_seconds is not None and now - _mtime(path) < older_than_seconds:
            continue
        images.append(path)

    return sorted(images, key=_mtime)


def delete_images(paths: Iterable[Path]) -> Tuple[List[Path], List[Tuple[Path, str]], int]:
    deleted = []
    failed = []
    bytes_freed = 0

    for path in paths:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError as exc:
            failed.append((path, str(exc)))
            continue

        deleted.append(path)
        bytes_freed += size

    return deleted, failed, bytes_freed
