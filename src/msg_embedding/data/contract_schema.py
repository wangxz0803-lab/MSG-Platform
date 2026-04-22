"""JSON-Schema export for :class:`msg_embedding.data.contract.ChannelSample`.

Kept in a dedicated module so downstream services (platform DB migrations,
OpenAPI docs, data-validation CLIs) can import just the schema without pulling
in numpy / pydantic model machinery.
"""

from __future__ import annotations

from typing import Any

from .contract import ChannelSample


def get_channel_sample_schema() -> dict[str, Any]:
    """Return the JSON Schema for :class:`ChannelSample`.

    Uses pydantic v2's :meth:`BaseModel.model_json_schema` under the hood.
    ``arbitrary_types_allowed=True`` means numpy array fields appear with a
    generic ``"type": "object"`` placeholder — consumers that want a stricter
    schema (e.g. shape/dtype) should layer on top.
    """
    schema = ChannelSample.model_json_schema()
    # Annotate for platform tooling: this schema is for the MSG ChannelSample v1.
    schema.setdefault("title", "ChannelSample")
    schema["$id"] = "https://msg-embedding.local/schemas/channel_sample/v1.json"
    schema["x-contract-version"] = 1
    return schema


__all__ = ["get_channel_sample_schema"]
