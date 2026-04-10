"""Shared utilities for AttentionDB Python modules."""

import hashlib
import struct


def hash_u64(s: str) -> int:
    """Hash a string to a uint64 using first 8 bytes of SHA-256."""
    h = hashlib.sha256(s.encode()).digest()[:8]
    return struct.unpack("<Q", h)[0]
