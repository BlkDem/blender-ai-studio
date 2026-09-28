"""Configuration that a person sets by hand.

The environment is the documented interface, so a variable that silently does
nothing is worse than a missing feature: `.env.example` showed the nested form
for months and none of those four variables did anything.
"""

from __future__ import annotations

import pytest

from app.core.settings import Settings


def test_a_nested_field_can_be_set_by_the_documented_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """STUDIO_AGENT__MAX_STEPS, which is what the README and .env.example show.

    Without an explicit nested delimiter pydantic-settings ignores the second
    underscore, and the variable parses as a field that does not exist -- so the
    default stands and the user believes they configured something.
    """
    monkeypatch.setenv("STUDIO_AGENT__MAX_STEPS", "99")
    monkeypatch.setenv("STUDIO_THREE_D__PROVIDER", "mock-3d")
    monkeypatch.setenv("STUDIO_THREE_D__DOWNLOAD_DIR", "out/assets")

    settings = Settings()

    assert settings.agent.max_steps == 99
    assert settings.three_d.provider == "mock-3d"
    assert settings.three_d.download_dir == "out/assets"


def test_the_json_form_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_AGENT", '{"max_steps": 7}')
    assert Settings().agent.max_steps == 7


def test_a_plain_variable_still_wins_over_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_LOG_LEVEL", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_the_database_overrides_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lowest to highest: defaults, then .env/environment, then the settings table."""
    monkeypatch.setenv("STUDIO_AGENT__MAX_STEPS", "10")
    settings = Settings()
    settings.apply_overrides({"agent.max_steps": 5})
    assert settings.agent.max_steps == 5


def test_an_unreadable_override_is_dropped_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A settings row written by a newer version must not stop the app starting."""
    monkeypatch.delenv("STUDIO_AGENT__MAX_STEPS", raising=False)
    settings = Settings()
    settings.apply_overrides({"agent.max_steps": "not a number", "log_level": "DEBUG"})
    assert settings.agent.max_steps == 30, "the default stands"
    assert settings.log_level == "DEBUG", "and the good one is applied"


def test_both_ways_of_naming_a_nested_field_can_be_mixed() -> None:
    """JSON for one, delimiter for another: neither is a special case."""
    settings = Settings()
    settings.apply_overrides({"three_d.provider": "mock-3d", "agent.max_3d_credits": 7})
    assert settings.three_d.provider == "mock-3d"
    assert settings.agent.max_3d_credits == 7


def test_the_data_directory_can_be_moved_by_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "studio"))
    settings = Settings()
    assert settings.resolved_database_path == tmp_path / "studio" / "studio.db"
    assert settings.resolved_secrets_path == tmp_path / "studio" / "secrets.json"


def test_an_explicit_database_path_wins_over_the_directory(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, database_path=tmp_path / "other.db")
    assert settings.resolved_database_path == tmp_path / "other.db"
