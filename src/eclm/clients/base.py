"""Model client interface and errors (ARCHITECTURE §5)."""

from __future__ import annotations

from typing import Literal, Protocol

from eclm.types import Generation


class ClientError(Exception):
    """A single call failed. Recorded on the result row; the run continues."""


class ClientUnavailable(Exception):
    """The backend can't be used at all. Stop before the first question; message says how to fix."""


class ModelClient(Protocol):
    name: str
    ran_on: Literal["edge", "cloud"]

    def check(self) -> None:
        """Raise ClientUnavailable if the backend can't be used."""
        ...

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation: ...
