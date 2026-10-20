"""
Generated documents: the `documents` table.
"""

from __future__ import annotations

import hashlib

from pydantic import AwareDatetime, Field

from assay.corpus import CommitSha, Unit
from assay.extract import SymbolKind
from assay.llm import Provider
from assay.records import Record
from assay.storage import TableSpec


class Document(Record):
    """
    One unit documented by one subject model: a `documents` row.

    `document_id` is derived from everything that determines the output (the
    unit's content, the prompt, the exact model weights and the sampling
    settings), so generating again with the same inputs is skipped and any
    change produces a new document rather than overwriting an old one.
    """

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str
    unit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo: str
    commit: CommitSha
    qualified_name: str
    kind: SymbolKind
    subject: str
    """The configured name of the subject model, such as `qwen-coder-3b`."""
    provider: Provider
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    temperature: float = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    text: str
    truncated: bool
    """The reply stopped at `max_output_tokens`."""
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    generated_at: AwareDatetime


DOCUMENTS = TableSpec(name="documents", model=Document, key=("document_id",))


def document_id(
    unit: Unit,
    *,
    prompt_hash: str,
    provider: Provider,
    model_id: str,
    model_revision: str,
    seed: int,
    temperature: float,
    max_output_tokens: int,
) -> str:
    parts = (
        unit.unit_hash,
        prompt_hash,
        provider.value,
        model_id,
        model_revision,
        str(seed),
        repr(temperature),
        str(max_output_tokens),
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
