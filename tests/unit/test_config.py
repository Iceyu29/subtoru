from __future__ import annotations

import sys

from subtoru.config import SubxtConfig


def test_default_values():
    config = SubxtConfig()
    assert config.whisper_model == "large-v2"
    assert config.device == "auto"
    assert config.export_formats == "srt"


def test_resolve_device_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    config = SubxtConfig(device="auto")
    assert config.resolve_device() == "cpu"


def test_resolve_compute_type():
    config1 = SubxtConfig(compute_type="float32")
    assert config1.resolve_compute_type() == "float32"

    config2 = SubxtConfig(device="cpu")
    assert config2.resolve_compute_type() == "int8"


def test_get_target_languages():
    config1 = SubxtConfig(target_language="fr, es,ja")
    assert config1.get_target_languages() == ["fr", "es", "ja"]

    config2 = SubxtConfig()
    assert config2.get_target_languages() == []


def test_get_export_formats():
    config = SubxtConfig(export_formats="SRT, vtt , json")
    assert config.get_export_formats() == ["srt", "vtt", "json"]


def test_load_overrides():
    config = SubxtConfig.load(whisper_model="small", verbose=True)
    assert config.whisper_model == "small"
    assert config.verbose is True


def test_env_var_loading(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    monkeypatch.setenv("SUBTORU_WHISPER_MODEL", "tiny")

    config = SubxtConfig.load()
    assert config.gemini_api_key == "test_key"
    assert config.whisper_model == "tiny"
