import os
from pathlib import Path

from typer.testing import CliRunner

from aquilesimage.cli.cli import app
from aquilesimage.utils.cleanup import (
    delete_images,
    find_images,
    resolve_image_dir,
    total_size,
)


def _touch(path: Path, age_seconds: float = 0) -> Path:
    path.write_bytes(b"image")
    if age_seconds:
        timestamp = path.stat().st_mtime - age_seconds
        os.utime(path, (timestamp, timestamp))
    return path


def test_find_images_filters_by_name_extension_and_age(tmp_path):
    recent = _touch(tmp_path / "imgrecent.png")
    old = _touch(tmp_path / "imgold.jpg", age_seconds=3600)
    _touch(tmp_path / "imgold.webp", age_seconds=3600)
    _touch(tmp_path / "other.png", age_seconds=3600)
    _touch(tmp_path / "imgold.txt", age_seconds=3600)
    (tmp_path / "imgdir.png").mkdir()

    found = find_images(tmp_path, older_than_seconds=60)

    assert {path.name for path in found} == {old.name, "imgold.webp"}
    assert recent not in found


def test_find_images_returns_all_when_age_is_none(tmp_path):
    _touch(tmp_path / "img1.png", age_seconds=3600)
    _touch(tmp_path / "img2.webp", age_seconds=3600)

    found = find_images(tmp_path, older_than_seconds=None)

    assert {path.name for path in found} == {"img1.png", "img2.webp"}


def test_delete_images_reports_deleted_and_missing(tmp_path):
    first = _touch(tmp_path / "img1.png")
    second = _touch(tmp_path / "img2.jpeg")
    missing = tmp_path / "imgmissing.png"

    deleted, failed, bytes_freed = delete_images([first, second, missing])

    assert {path.name for path in deleted} == {"img1.png", "img2.jpeg"}
    assert len(failed) == 1
    assert failed[0][0] == missing
    assert bytes_freed == len(b"image") * 2
    assert not first.exists()
    assert not second.exists()


def test_total_size_ignores_missing_files(tmp_path):
    first = _touch(tmp_path / "img1.png")
    second = _touch(tmp_path / "img2.png")

    assert total_size([first, second, tmp_path / "missing.png"]) == len(b"image") * 2


def test_resolve_image_dir_uses_env(tmp_path, monkeypatch):
    target = tmp_path / "env-images"
    monkeypatch.setenv("AQUILES_IMAGE_DIR", str(target))

    assert resolve_image_dir() == target.resolve()


def test_clean_images_dry_run_keeps_files(tmp_path):
    image = _touch(tmp_path / "img1.png")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["clean-images", "--image-dir", str(tmp_path), "--all", "--dry-run"],
    )

    assert result.exit_code == 0
    assert image.exists()
    assert "Dry run" in result.output


def test_clean_images_all_deletes_generated_files(tmp_path):
    image = _touch(tmp_path / "img1.png")
    unrelated = _touch(tmp_path / "notes.png")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["clean-images", "--image-dir", str(tmp_path), "--all"],
    )

    assert result.exit_code == 0
    assert not image.exists()
    assert unrelated.exists()
