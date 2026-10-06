"""Unit tests for subtoru.utils.naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtoru.utils.naming import (
    derive_output_path,
    is_url,
    make_output_dir,
    sanitize_filename,
    slugify,
)


class TestSlugify:
    """Tests for the slugify helper function."""

    def test_slugify_basic_text(self) -> None:
        assert slugify("Hello World") == "hello-world"
        assert slugify("Simple Test Case") == "simple-test-case"

    def test_slugify_unicode_normalization(self) -> None:
        assert slugify("Café & Restaurant") == "cafe-restaurant"
        assert slugify("Über naïve façade") == "uber-naive-facade"
        assert slugify("Crème Brûlée") == "creme-brulee"
        assert slugify("Español y Português") == "espanol-y-portugues"

    def test_slugify_special_characters(self) -> None:
        assert slugify("My Video — Episode 3!") == "my-video-episode-3"
        assert slugify("file (2024) [remastered] #1") == "file-2024-remastered-1"
        assert slugify("test@email.com: hello / world") == "testemailcom-hello-world"
        assert slugify("foo_bar-baz") == "foo_bar-baz"

    def test_slugify_whitespace_and_hyphen_collapsing(self) -> None:
        assert slugify("foo   ---   bar") == "foo-bar"
        assert slugify("  -- leading and trailing --  ") == "leading-and-trailing"
        assert slugify("multiple      spaces") == "multiple-spaces"

    def test_slugify_empty_and_symbols_only(self) -> None:
        assert slugify("") == ""
        assert slugify("   ") == ""
        assert slugify("!@#$%^&*()+={}[]|:;'<>,.?/") == ""

    def test_slugify_max_length(self) -> None:
        long_text = "a" * 120
        # Default max_length is 80
        assert slugify(long_text) == "a" * 80
        assert len(slugify(long_text)) == 80

        # Custom max_length
        assert (
            slugify("a-very-long-video-title-that-should-be-cut", max_length=15)
            == "a-very-long-vid"
        )
        assert len(slugify("a-very-long-video-title-that-should-be-cut", max_length=15)) == 15

        # Shorter than max_length is untouched
        assert slugify("short-title", max_length=50) == "short-title"

        # Edge case: max_length=0
        assert slugify("anything", max_length=0) == ""


class TestMakeOutputDir:
    """Tests for make_output_dir directory creation."""

    def test_make_output_dir_with_explicit_base(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "custom" / "output_dir"
        assert not base_dir.exists()

        result = make_output_dir(base=str(base_dir), source_name="ignored-source")
        assert result == base_dir
        assert result.is_dir()

    def test_make_output_dir_with_existing_base(self, tmp_path: Path) -> None:
        existing_dir = tmp_path / "already_exists"
        existing_dir.mkdir(parents=True)

        result = make_output_dir(base=str(existing_dir), source_name="video")
        assert result == existing_dir
        assert result.is_dir()

    def test_make_output_dir_default_root(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        result = make_output_dir(
            base=None,
            source_name="My Video — Episode 1!",
            default_root=str(root),
        )
        expected = root / "my-video-episode-1"
        assert result == expected
        assert result.is_dir()

    def test_make_output_dir_empty_slug_fallback_to_unnamed(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        # Input that slugifies to empty string
        result = make_output_dir(
            base=None,
            source_name="???!!!###",
            default_root=str(root),
        )
        expected = root / "unnamed"
        assert result == expected
        assert result.is_dir()

    def test_make_output_dir_empty_source_name(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        result = make_output_dir(
            base=None,
            source_name="",
            default_root=str(root),
        )
        expected = root / "unnamed"
        assert result == expected
        assert result.is_dir()


class TestDeriveOutputPath:
    """Tests for derive_output_path path generation."""

    def test_derive_output_path_basic(self) -> None:
        result = derive_output_path("video.wav", ".srt")
        assert result == Path("video.srt")

    def test_derive_output_path_preserves_parent(self) -> None:
        result = derive_output_path("path/to/media/audio.m4a", ".vtt")
        assert result == Path("path/to/media/audio.vtt")

    def test_derive_output_path_from_path_instance(self) -> None:
        source = Path("/var/media/sample.mp4")
        result = derive_output_path(source, ".json")
        assert result == Path("/var/media/sample.json")

    def test_derive_output_path_with_language(self) -> None:
        result = derive_output_path("video.wav", ".srt", language="fr")
        assert result == Path("video.fr.srt")

    def test_derive_output_path_with_tag(self) -> None:
        result = derive_output_path("video.wav", ".srt", tag="bilingual")
        assert result == Path("video.bilingual.srt")

    def test_derive_output_path_with_language_and_tag(self) -> None:
        result = derive_output_path("video.wav", ".srt", language="es", tag="aligned")
        assert result == Path("video.es.aligned.srt")

    def test_derive_output_path_with_output_dir(self, tmp_path: Path) -> None:
        result = derive_output_path("source/video.wav", ".srt", output_dir=tmp_path)
        assert result == tmp_path / "video.srt"

    def test_derive_output_path_with_output_dir_str_and_all_tags(self, tmp_path: Path) -> None:
        result = derive_output_path(
            "source/video.wav",
            ".ass",
            output_dir=str(tmp_path),
            language="de",
            tag="bilingual",
        )
        assert result == tmp_path / "video.de.bilingual.ass"


class TestIsUrl:
    """Tests for the is_url heuristic function."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://example.com/audio.mp3",
            "https://vimeo.com/123456789",
            "https://sub.domain.org/path/resource?arg=1&foo=bar",
            "http://localhost:8080/media",
        ],
    )
    def test_is_url_with_scheme(self, url: str) -> None:
        assert is_url(url) is True

    @pytest.mark.parametrize(
        "domain_url",
        [
            "youtu.be/dQw4w9WgXcQ",
            "youtube.com/watch?v=123",
            "example.com/audio.mp3",
            "subtoru.io",
            "my-domain.org/video",
        ],
    )
    def test_is_url_domain_pattern(self, domain_url: str) -> None:
        assert is_url(domain_url) is True

    @pytest.mark.parametrize(
        "non_url",
        [
            "/absolute/path/to/video.mp4",
            "./relative/path/to/audio.wav",
            "../parent/file.mp3",
            "just_a_filename_without_extension",
            "plain text with spaces",
            "",
            "   ",
            "C:\\Windows\\Media\\file.wav",
            "file.c",  # single letter extension does not match [a-z]{2,}
        ],
    )
    def test_is_url_false_for_local_paths_and_text(self, non_url: str) -> None:
        assert is_url(non_url) is False


class TestSanitizeFilename:
    """Tests for the sanitize_filename helper function."""

    def test_preserves_normal_filename(self) -> None:
        assert sanitize_filename("My Video.mp4") == "My Video.mp4"

    def test_removes_reserved_chars(self) -> None:
        assert sanitize_filename('file:name*"test?.mp4') == "filenametest.mp4"

    def test_removes_control_chars(self) -> None:
        assert sanitize_filename("file\x00\x1fname.mp4") == "filename.mp4"

    def test_collapses_whitespace(self) -> None:
        assert sanitize_filename("my   big   video.mp4") == "my big video.mp4"

    def test_strips_leading_trailing_dots(self) -> None:
        assert sanitize_filename("...hidden...") == "hidden"

    def test_strips_leading_trailing_spaces(self) -> None:
        assert sanitize_filename("  spaced  ") == "spaced"

    def test_preserves_non_ascii(self) -> None:
        assert sanitize_filename("日本語テスト.mp4") == "日本語テスト.mp4"

    def test_preserves_accented_chars(self) -> None:
        assert sanitize_filename("café résumé.txt") == "café résumé.txt"

    def test_preserves_extension(self) -> None:
        result = sanitize_filename("bad<>name.srt")
        assert result.endswith(".srt")

    def test_truncates_long_stem(self) -> None:
        long_name = "a" * 300 + ".mp4"
        result = sanitize_filename(long_name, max_length=50)
        assert len(Path(result).stem) <= 50
        assert result.endswith(".mp4")

    def test_empty_stem_fallback(self) -> None:
        assert sanitize_filename(":::.mp4") == "untitled.mp4"

    def test_completely_empty(self) -> None:
        assert sanitize_filename("") == "untitled"

    def test_removes_path_separators(self) -> None:
        # Note: Path("path/to\\file.mp4") sees path/to as dirs, stem="\\file"
        # so only the backslash in stem is removed
        assert sanitize_filename("path\\file.mp4") == "pathfile.mp4"

    def test_preserves_dashes_and_underscores(self) -> None:
        assert sanitize_filename("my_video-2024.mp4") == "my_video-2024.mp4"

    def test_emoji_preserved(self) -> None:
        # Emojis are non-ASCII but not control chars, should be kept
        assert sanitize_filename("🎬 Movie.mp4") == "🎬 Movie.mp4"

    def test_pipe_removed(self) -> None:
        # Pipe removed, then whitespace collapsed to single space
        assert sanitize_filename("title | subtitle.mp4") == "title subtitle.mp4"
