"""Client factory."""

from __future__ import annotations

from typing import Literal

from eclm.clients.base import BudgetedClient, BudgetExceeded, ClientError, ClientUnavailable, ModelClient
from eclm.clients.cloud import CloudClient
from eclm.clients.mock import MockClient, mock_accuracy
from eclm.clients.ollama import OllamaClient
from eclm.config import Settings

__all__ = [
    "BudgetExceeded",
    "BudgetedClient",
    "ClientError",
    "CloudClient",
    "ClientUnavailable",
    "MockClient",
    "ModelClient",
    "OllamaClient",
    "make_client",
    "mock_accuracy",
]


def make_client(
    kind: Literal["edge", "cloud"],
    settings: Settings,
    *,
    model: str | None = None,
    mock: bool = False,
) -> ModelClient:
    """Build the client for one side. With mock=True the client is named after the requested model."""
    if kind == "edge":
        name = model or settings.edge.hybrid_model
    else:
        name = model or settings.cloud.model
    if mock:
        return MockClient(name, kind, mock_accuracy(name, kind), settings.seed)
    if kind == "edge":
        return OllamaClient(name, settings)
    return CloudClient(name, settings)
