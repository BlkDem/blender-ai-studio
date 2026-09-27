"""Secrets, kept out of the database and out of every log line.

An API key in a SQLite file is an API key in every backup, every project export
and every "send me your studio folder" support request. So keys live in their own
file, created with owner-only permissions, and :class:`SecretStore` is the only
thing in the application that is allowed to return one.

``keyring`` is used when it is importable, because a desktop app should not keep
plaintext keys on disk when the platform has a keychain. When it is not there,
the file is used and the fact is reported by :meth:`SecretStore.backend` rather
than hidden.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

from app.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

#: Keys are namespaced so two applications sharing a keyring do not collide.
KEYRING_SERVICE = "blender-ai-studio"


class _SecretCache:
    """A cache that refuses to print itself.

    Anything can end up in a log: an exception's ``vars()``, a debugger, a
    ``print(store.__dict__)`` left in a debug branch. A plain dict of keys would
    put every API key in every one of those, so the values live behind a repr that
    says only that there are some.
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def put(self, key: str, value: str) -> None:
        self._values[key] = value

    def drop(self, key: str) -> None:
        self._values.pop(key, None)

    def keys(self) -> list[str]:
        return sorted(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"<secret cache: {len(self._values)} entr{'y' if len(self._values) == 1 else 'ies'}>"

    __str__ = __repr__


class SecretStore:
    """Read and write API keys. Never log a value from here."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._memory = _SecretCache()
        self._keyring: Any = None
        self._load_keyring()

    def _load_keyring(self) -> None:
        try:
            import keyring  # type: ignore[import-not-found]
            from keyring.errors import NoKeyringError  # type: ignore[import-not-found]
        except ImportError:
            logger.info("keyring is not installed; secrets will be stored in %s", self._path)
            return
        try:
            keyring.get_keyring()
        except NoKeyringError:  # pragma: no cover - platform dependent
            logger.info("no usable system keyring; secrets will be stored in %s", self._path)
            return
        self._keyring = keyring

    @property
    def backend(self) -> str:
        return "keyring" if self._keyring is not None else "file"

    # --- file backend ------------------------------------------------------

    def _read_file(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigurationError(
                f"The secrets file at {self._path} could not be read: {exc}",
                hint="Delete it to start over, or repair the JSON by hand.",
            ) from exc
        return {str(k): str(v) for k, v in data.items()}

    def _write_file(self, values: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 rather than chmod after: there should be no window in
        # which the file exists and is world-readable.
        descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
        self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # --- public API --------------------------------------------------------

    def get(self, key: str) -> str | None:
        """A key, or ``None``. The only method that returns a secret."""
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        if self._keyring is not None:
            try:
                value = self._keyring.get_password(KEYRING_SERVICE, key)
            except Exception:  # pragma: no cover - keyring implementations vary
                logger.debug("keyring read failed for %s", key, exc_info=True)
                value = None
            if value:
                self._memory.put(key, value)
                return value
            return None
        value = self._read_file().get(key)
        if value is not None:
            self._memory.put(key, value)
        return value

    def set(self, key: str, value: str) -> None:
        if not value:
            raise ConfigurationError("Refusing to store an empty secret")
        self._memory.put(key, value)
        if self._keyring is not None:
            try:
                self._keyring.set_password(KEYRING_SERVICE, key, value)
                return
            except Exception:  # pragma: no cover - keyring implementations vary
                logger.warning("keyring write failed; falling back to the secrets file")
        values = self._read_file()
        values[key] = value
        self._write_file(values)

    def delete(self, key: str) -> None:
        self._memory.drop(key)
        if self._keyring is not None:
            try:
                self._keyring.delete_password(KEYRING_SERVICE, key)
            except Exception:  # pragma: no cover
                logger.debug("keyring delete failed for %s", key, exc_info=True)
        values = self._read_file()
        if values.pop(key, None) is not None:
            self._write_file(values)

    def has(self, key: str) -> bool:
        return bool(self.get(key))

    def masked(self, key: str) -> str:
        """What the GUI shows. A key's last four characters are enough to
        recognise which one is set, and not enough to use it."""
        value = self.get(key)
        if not value:
            return ""
        return f"{'•' * 8}{value[-4:]}" if len(value) > 8 else "•" * len(value)

    def names(self) -> list[str]:
        if self._keyring is not None:
            return self._memory.keys()
        return sorted(self._read_file())


def key_name(provider: str) -> str:
    """One naming scheme, so the GUI and the .env file agree on what to call a key."""
    return f"{provider}.api_key"
