"""Model client interface and errors (ARCHITECTURE §5)."""

from __future__ import annotations

from typing import Literal, Protocol

from eclm.types import Generation


class ClientError(Exception):
    """A single call failed. Recorded on the result row; the run continues."""


class ClientUnavailable(Exception):
    """The backend can't be used at all. Stop (before the first question, or mid-run keeping the rows already
    written); message says how to fix."""


class BudgetExceeded(Exception):
    """The run's cloud-call budget is used up. Raised BEFORE the call; the runner exits 2 (resumable)."""


class ModelClient(Protocol):
    name: str
    ran_on: Literal["edge", "cloud"]

    def check(self) -> None:
        """Raise ClientUnavailable if the backend can't be used."""
        ...

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation: ...


class BudgetedClient:
    """Wraps the cloud client and counts calls; refuses the call that would go over `max_calls` (§7).

    The count covers this object only, i.e. one process: a restarted run starts a fresh budget.
    Every attempted call counts, including one that then fails with ClientError.
    """

    def __init__(self, inner: ModelClient, max_calls: int) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.calls = 0
        self.name = inner.name
        self.ran_on = inner.ran_on

    def check(self) -> None:
        self.inner.check()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"cloud-call budget of {self.max_calls} calls for this run is used up")
        self.calls += 1
        return self.inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
