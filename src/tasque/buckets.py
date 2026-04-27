"""Bucket definitions — the nine canonical life-area buckets used across tasque."""

from typing import Literal, get_args

Bucket = Literal[
    "health",
    "relationships",
    "education",
    "career",
    "finance",
    "creative",
    "home",
    "personal",
    "recreation",
]

ALL_BUCKETS: tuple[Bucket, ...] = get_args(Bucket)


def is_bucket(value: str) -> bool:
    return value in ALL_BUCKETS
