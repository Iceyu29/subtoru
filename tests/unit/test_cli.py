from __future__ import annotations

from typer.testing import CliRunner

from subtoru import __version__
from subtoru.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_commands():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["run", "--help"]).exit_code == 0
    assert runner.invoke(app, ["download", "--help"]).exit_code == 0
    assert runner.invoke(app, ["transcribe", "--help"]).exit_code == 0
    assert runner.invoke(app, ["translate", "--help"]).exit_code == 0
    assert runner.invoke(app, ["export", "--help"]).exit_code == 0
    assert runner.invoke(app, ["list-models", "--help"]).exit_code == 0


def test_transcribe_nonexistent():
    result = runner.invoke(app, ["transcribe", "nonexistent.wav"])
    assert result.exit_code != 0


def test_translate_nonexistent():
    result = runner.invoke(app, ["translate", "nonexistent.srt", "-t", "French"])
    assert result.exit_code != 0


def test_export_nonexistent():
    result = runner.invoke(app, ["export", "nonexistent.srt"])
    assert result.exit_code != 0
