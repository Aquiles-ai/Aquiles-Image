import gc
import io
import os

import pytest
import torch
from PIL import Image as PILImage

from aquilesimage.utils import Utils, create_dev_mode_response


@pytest.fixture()
def utils(tmp_path):
    app = Utils()
    app.image_dir = str(tmp_path)
    return app


@pytest.mark.parametrize(
    "output_format,expected_format,extension",
    [
        ("png", "PNG", ".png"),
        ("jpeg", "JPEG", ".jpg"),
        ("jpg", "JPEG", ".jpg"),
        ("webp", "WEBP", ".webp"),
    ],
)
def test_save_image_writes_requested_format(utils, output_format, expected_format, extension):
    image = PILImage.new("RGB", (64, 48), (10, 20, 30))

    filename = utils.save_image(image, output_format, output_compression=90)

    assert filename.endswith(extension)
    with PILImage.open(os.path.join(utils.image_dir, filename)) as saved:
        assert saved.format == expected_format
        assert saved.size == image.size


@pytest.mark.parametrize(
    "output_format,magic",
    [
        ("png", b"\x89PNG\r\n\x1a\n"),
        ("jpeg", b"\xff\xd8"),
        ("webp", b"RIFF"),
    ],
)
def test_encode_image_returns_requested_container(utils, output_format, magic):
    data = utils.encode_image(PILImage.new("RGB", (16, 16), "red"), output_format)

    assert data.startswith(magic)
    with PILImage.open(io.BytesIO(data)) as decoded:
        assert decoded.format == output_format.upper()


def test_png_is_lossless_at_fast_compression(utils):
    image = PILImage.new("RGB", (32, 32))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x * 8 % 256, y * 8 % 256, (x + y) * 4 % 256)

    data = utils.encode_image(image, "png", output_compression=10)

    with PILImage.open(io.BytesIO(data)) as decoded:
        assert decoded.convert("RGB").tobytes() == image.tobytes()


def test_png_preserves_alpha(utils):
    image = PILImage.new("RGBA", (8, 8), (255, 0, 0, 0))

    data = utils.encode_image(image, "png")

    with PILImage.open(io.BytesIO(data)) as decoded:
        assert decoded.mode == "RGBA"
        assert decoded.getpixel((0, 0))[3] == 0


def test_jpeg_composites_alpha_over_white(utils):
    image = PILImage.new("RGBA", (8, 8), (255, 0, 0, 0))

    data = utils.encode_image(image, "jpeg", output_compression=90)

    with PILImage.open(io.BytesIO(data)) as decoded:
        assert decoded.mode == "RGB"
        assert all(channel >= 250 for channel in decoded.getpixel((0, 0)))


def test_save_image_does_not_run_per_image_gc_or_cuda_cleanup(utils, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("per-image cleanup must not run")

    monkeypatch.setattr(gc, "collect", fail)
    monkeypatch.setattr(torch.cuda, "empty_cache", fail)

    filename = utils.save_image(PILImage.new("RGB", (8, 8), "blue"), "png")

    assert filename.endswith(".png")


def test_unknown_format_is_rejected(utils):
    with pytest.raises(ValueError):
        utils.encode_image(PILImage.new("RGB", (8, 8)), "gif")


def test_save_image_uses_unique_names(utils):
    image = PILImage.new("RGB", (8, 8))

    first = utils.save_image(image, "png")
    second = utils.save_image(image, "png")

    assert first != second


def test_utils_uses_custom_image_dir(tmp_path):
    target = tmp_path / "custom-images"

    app = Utils(image_dir=str(target))

    assert app.image_dir == str(target.resolve())
    assert target.is_dir()


def test_utils_uses_image_dir_env_var(tmp_path, monkeypatch):
    target = tmp_path / "env-images"
    monkeypatch.setenv("AQUILES_IMAGE_DIR", str(target))

    app = Utils()

    assert app.image_dir == str(target.resolve())
    assert target.is_dir()


def test_dev_mode_response_uses_requested_format(utils, tmp_path):
    source = tmp_path / "source.png"
    PILImage.new("RGB", (32, 32), "blue").save(source)

    response = create_dev_mode_response(
        str(source),
        None,
        n=1,
        response_format="url",
        output_format="webp",
        base_url="http://localhost:5500",
        utils_app=utils,
    )

    filename = response["data"][0]["url"].rsplit("/", 1)[-1]
    assert filename.endswith(".webp")
    with PILImage.open(os.path.join(utils.image_dir, filename)) as saved:
        assert saved.format == "WEBP"
