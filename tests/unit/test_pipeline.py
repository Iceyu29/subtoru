from __future__ import annotations

from subtoru.config import SubxtConfig
from subtoru.models import RunManifest
from subtoru.pipeline import _load_manifest, _resolve_plan, _save_manifest


def test_manifest_roundtrip(tmp_path):
    manifest = RunManifest(input_source="test.mp4", config_used={"whisper_model": "tiny"})
    manifest.media_path = "test.mp4"

    _save_manifest(manifest, tmp_path)
    assert (tmp_path / "manifest.json").exists()

    loaded = _load_manifest(tmp_path)
    assert loaded is not None
    assert loaded.input_source == "test.mp4"
    assert loaded.media_path == "test.mp4"
    assert loaded.config_used == {"whisper_model": "tiny"}


def test_resolve_plan_url():
    config = SubxtConfig(target_language="fr")
    plan = _resolve_plan(
        input_source="https://youtube.com/watch?v=123",
        config=config,
        skip_download=False,
        no_translate=False,
        resume=False,
        manifest=None,
    )

    assert plan["needs_download"] is True
    assert plan["target_languages"] == ["fr"]
    assert len(plan["stages"]) > 0
    assert plan["stages"][0]["name"] == "download"
    assert not plan["stages"][0]["skip"]


def test_resolve_plan_file():
    config = SubxtConfig(target_language="fr")
    plan = _resolve_plan(
        input_source="/path/to/local.mp4",
        config=config,
        skip_download=False,
        no_translate=False,
        resume=False,
        manifest=None,
    )

    assert plan["needs_download"] is False
    assert plan["target_languages"] == ["fr"]
    # No download stage
    assert plan["stages"][0]["name"] == "transcribe"


def test_resolve_plan_no_translate():
    config = SubxtConfig(target_language="fr")
    plan = _resolve_plan(
        input_source="/path/to/local.mp4",
        config=config,
        skip_download=False,
        no_translate=True,
        resume=False,
        manifest=None,
    )

    assert plan["no_translate"] is True
    assert plan["target_languages"] == []
    stage_names = [s["name"] for s in plan["stages"]]
    assert "transcribe" in stage_names
    assert not any(s.startswith("translate") for s in stage_names)


def test_resolve_plan_resume():
    config = SubxtConfig(target_language="fr")
    manifest = RunManifest(input_source="/path/to/local.mp4", config_used={})
    manifest.media_path = "/path/to/local.mp4"
    manifest.source_srt_path = "local.srt"

    plan = _resolve_plan(
        input_source="/path/to/local.mp4",
        config=config,
        skip_download=False,
        no_translate=False,
        resume=True,
        manifest=manifest,
    )

    transcribe_stage = next(s for s in plan["stages"] if s["name"] == "transcribe")
    assert transcribe_stage["skip"] is True
    assert transcribe_stage["reason"] == "already transcribed"

    translate_stage = next(s for s in plan["stages"] if s["name"] == "translate (fr)")
    assert translate_stage["skip"] is False
