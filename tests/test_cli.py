import pytest
from click.testing import CliRunner
from lab1.cli import cli
import io
from PIL import Image


@pytest.fixture
def runner():
    return CliRunner()


def test_remove_missing_cli(runner):
    result = runner.invoke(cli, ["clean", "remove-missing", "1", "", "2"])
    assert "1" in result.output
    assert "2" in result.output


def test_fill_missing_cli(runner):
    result = runner.invoke(cli, ["clean", "fill-missing", "1", "", "2", "--fill", "5"])
    assert "5" in result.output


def test_normalize_cli(runner):
    result = runner.invoke(cli, ["numeric", "normalize", "1", "2", "3"])
    assert result.exit_code == 0


def test_tokenize_cli(runner):
    result = runner.invoke(cli, ["text", "tokenize", "Hello World!"])
    assert "hello" in result.output


def test_shuffle_cli(runner):
    result = runner.invoke(cli, ["struct", "shuffle", "1", "2", "3", "--seed", "1"])
    assert result.exit_code == 0


def make_image_bytes(size=(50, 50)):
    img = Image.new("RGB", size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_cli(tmp_path):
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(make_image_bytes().read())
    runner = CliRunner()
    res = runner.invoke(cli, ["predict", str(img_file)])
    assert res.exit_code == 0
    assert "predicted_class" in res.output


def test_resize_cli(tmp_path):
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(make_image_bytes(size=(80, 60)).read())
    out = tmp_path / "out.jpg"
    runner = CliRunner()
    res = runner.invoke(cli, ["resize", str(img_file), "32", "32", "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
