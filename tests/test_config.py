"""Tests for frankenbote.config — YAML loading with tmp_path fixtures."""

import pytest
import yaml

from frankenbote.config import load_sources
from frankenbote.models import Category


def _write_sources(tmp_path, sources_list: list[dict]) -> str:
    """Write a valid sources.yaml to tmp_path and return its path string."""
    path = tmp_path / "sources.yaml"
    path.write_text(
        yaml.dump({"sources": sources_list}),
        encoding="utf-8",
    )
    return str(path)


def _minimal_source(**overrides) -> dict:
    base = dict(
        id="test_src",
        name="Test Source",
        url="https://example.com/feed",
        category="local",
        enabled=True,
    )
    base.update(overrides)
    return base


class TestLoadSourcesValid:
    def test_returns_list_of_source_objects(self, tmp_path):
        path = _write_sources(tmp_path, [_minimal_source()])
        sources = load_sources(path)
        assert len(sources) == 1
        assert sources[0].id == "test_src"

    def test_multiple_sources_returned(self, tmp_path):
        path = _write_sources(tmp_path, [
            _minimal_source(id="src_one", url="https://one.example.com/feed"),
            _minimal_source(id="src_two", url="https://two.example.com/feed"),
        ])
        sources = load_sources(path)
        assert len(sources) == 2

    def test_category_is_parsed(self, tmp_path):
        path = _write_sources(tmp_path, [_minimal_source(category="national")])
        sources = load_sources(path)
        assert sources[0].category == Category.NATIONAL


class TestLoadSourcesFiltering:
    def test_disabled_sources_excluded(self, tmp_path):
        path = _write_sources(tmp_path, [
            _minimal_source(id="active_src", enabled=True),
            _minimal_source(id="inactive_src", url="https://inactive.example.com/feed", enabled=False),
        ])
        sources = load_sources(path)
        ids = [s.id for s in sources]
        assert "active_src" in ids
        assert "inactive_src" not in ids

    def test_all_disabled_returns_empty_list(self, tmp_path):
        path = _write_sources(tmp_path, [
            _minimal_source(id="s1", enabled=False),
            _minimal_source(id="s2", url="https://s2.example.com/feed", enabled=False),
        ])
        sources = load_sources(path)
        assert sources == []


class TestLoadSourcesErrors:
    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_sources(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        path = tmp_path / "sources.yaml"
        path.write_text("{ unclosed: [bracket", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_sources(str(path))

    def test_missing_sources_key_raises_value_error(self, tmp_path):
        path = tmp_path / "sources.yaml"
        path.write_text(yaml.dump({"not_sources": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="sources"):
            load_sources(str(path))

    def test_invalid_source_id_raises(self, tmp_path):
        path = _write_sources(tmp_path, [_minimal_source(id="Has Spaces")])
        with pytest.raises(Exception):
            load_sources(str(path))
