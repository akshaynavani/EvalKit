"""Versioned prompt templates."""

from __future__ import annotations

import hashlib
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class PromptError(ValueError):
    pass


class _StrictFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):  # type: ignore[override]
        if isinstance(key, str) and key not in kwargs:
            raise PromptError(f"prompt references missing variable {{{key}}}")
        return super().get_value(key, args, kwargs)


_FORMATTER = _StrictFormatter()


@dataclass
class PromptTemplate:
    """A system + user template pair rendered with ``str.format`` semantics.

    Literal braces must be doubled (``{{`` / ``}}``) as with ``str.format``.
    """

    name: str
    user: str
    system: str = ""
    version: int = 1
    parent: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{self.system}\n---\n{self.user}".encode()).hexdigest()
        return digest[:12]

    @property
    def label(self) -> str:
        return f"{self.name}@v{self.version}"

    def variables(self) -> set[str]:
        names = set()
        for text in (self.system, self.user):
            for _, field_name, _, _ in _FORMATTER.parse(text):
                if field_name:
                    names.add(field_name.split(".")[0].split("[")[0])
        return names

    def render(self, variables: dict[str, Any]) -> tuple[str, str]:
        """Return ``(system, user)`` strings."""
        return (
            _FORMATTER.format(self.system, **variables) if self.system else "",
            _FORMATTER.format(self.user, **variables),
        )

    def derive(self, *, system: str | None = None, user: str | None = None, notes: str = "") -> "PromptTemplate":
        """Create the next version of this template."""
        return PromptTemplate(
            name=self.name,
            system=self.system if system is None else system,
            user=self.user if user is None else user,
            version=self.version + 1,
            parent=self.label,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        if "user" not in data:
            raise PromptError("prompt template requires a 'user' field")
        return cls(
            name=str(data.get("name", "prompt")),
            user=data["user"],
            system=data.get("system", ""),
            version=int(data.get("version", 1)),
            parent=data.get("parent"),
            notes=data.get("notes", ""),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def load_prompt(path: str | Path) -> PromptTemplate:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PromptError(f"{path}: prompt file must be a mapping")
    data.setdefault("name", path.stem)
    return PromptTemplate.from_dict(data)
