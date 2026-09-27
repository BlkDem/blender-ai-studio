"""Secrets: never logged, never in the database, never in an export."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from app.core.errors import ConfigurationError
from app.storage.secrets import SecretStore, key_name


def test_a_key_round_trips_through_the_file(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-test-1234567890")
    assert store.get("openai.api_key") == "sk-test-1234567890"


def test_the_file_is_not_world_readable(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-secret")
    mode = secrets_file.stat().st_mode
    assert not mode & (stat.S_IRGRP | stat.S_IROTH), "an API key readable by other users is a leak"


def test_a_missing_key_is_none_not_an_error(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    assert store.get("anthropic.api_key") is None
    assert store.has("anthropic.api_key") is False


def test_the_masked_form_does_not_leak_the_key(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-abcdefghijklmnop")
    masked = store.masked("openai.api_key")
    assert "abcdefghij" not in masked
    assert masked.endswith("mnop"), "the last few characters help recognise which key is set"
    assert "sk-abcdefghijklmnop" not in masked


def test_a_short_key_is_masked_completely(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("x", "short")
    assert store.masked("x") == "•" * 5


def test_deleting_removes_it(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-1")
    store.delete("openai.api_key")
    assert store.get("openai.api_key") is None
    assert json.loads(secrets_file.read_text(encoding="utf-8")) == {}


def test_an_empty_secret_is_refused(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    with pytest.raises(ConfigurationError):
        store.set("openai.api_key", "")


def test_a_corrupt_file_is_reported_not_ignored(secrets_file: Path) -> None:
    """Silently treating a corrupt file as empty would look like a lost key and
    send the user looking in the wrong place."""
    secrets_file.write_text("{not json", encoding="utf-8")
    store = SecretStore(secrets_file)
    with pytest.raises(ConfigurationError) as excinfo:
        store.get("openai.api_key")
    assert "secrets.json" in str(excinfo.value)


def test_names_lists_what_is_stored(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-1")
    store.set("tripo.api_key", "tsk-1")
    assert store.names() == ["openai.api_key", "tripo.api_key"]


def test_the_backend_is_reported(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    assert store.backend in {"file", "keyring"}


def test_the_key_name_is_namespaced_by_provider() -> None:
    assert key_name("openai") == "openai.api_key"
    assert key_name("tripo") == "tripo.api_key"


def test_nothing_in_the_repr_holds_a_key(secrets_file: Path) -> None:
    store = SecretStore(secrets_file)
    store.set("openai.api_key", "sk-do-not-print-me")
    assert "sk-do-not-print-me" not in repr(store)
    assert "sk-do-not-print-me" not in str(store.__dict__)
