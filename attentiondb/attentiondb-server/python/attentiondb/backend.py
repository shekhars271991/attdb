"""LMCache StorageBackendInterface adapter for AttentionDB."""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import _attentiondb
except ImportError:
    _attentiondb = None
    logger.warning("attentiondb native module not found, backend will not function")

# Entry type constants matching C++ EntryType enum
ENTRY_TYPE_CONVERSATION = 0
ENTRY_TYPE_SYSTEM_PROMPT = 1
ENTRY_TYPE_RAG_CONTEXT = 2


def _hash_u64(s: str) -> int:
    """Hash a string to a uint64 using first 8 bytes of SHA-256."""
    h = hashlib.sha256(s.encode()).digest()[:8]
    return struct.unpack("<Q", h)[0]


class AttentionDBBackend:
    """
    Implements LMCache's StorageBackendInterface using AttentionDB as the
    storage engine. Drop-in replacement for LocalCPUBackend / LocalDiskBackend.
    """

    def __init__(self, dst_device: str = "cpu", config: object = None):
        if _attentiondb is None:
            raise ImportError("attentiondb native module not available")

        self.dst_device = dst_device
        config_path = ""
        self._tenant_id = 0
        self._chunk_size = 256
        self._cost_per_token = 1

        if config is not None:
            config_path = getattr(config, "attentiondb_config", "")
            self._tenant_id = getattr(config, "tenant_id", 0)
            self._chunk_size = getattr(config, "chunk_size", 256)
            self._cost_per_token = getattr(config, "cost_per_token", 1)

        self._engine = _attentiondb.Engine(config_path)

    def contains(self, key, pin: bool = False) -> bool:
        storage_key = self._map_key(key)
        return self._engine.contains(storage_key)

    def batched_contains(
        self,
        keys: list,
        search_range: Optional[Tuple[int, int]] = None,
        pin: bool = False,
    ) -> int:
        """
        Check how many consecutive keys (from the start) are cached.
        Follows LMCache prefix-match semantics: stop at first miss.
        Returns the count of consecutive hits.
        """
        if search_range is not None:
            start, end = search_range
            keys = keys[start:end]

        count = 0
        for key in keys:
            storage_key = self._map_key(key)
            if self._engine.contains(storage_key):
                count += 1
            else:
                break
        return count

    def get_blocking(self, key) -> Optional[object]:
        """
        Blocking get. Returns a MemoryObj-compatible object or None on miss.
        """
        storage_key = self._map_key(key)
        blob = self._engine.get(storage_key)
        if blob is None:
            return None

        # Wrap the raw bytes in a MemoryObj-like wrapper.
        # In production, LMCache provides MemoryObj; here we return raw bytes
        # for the adapter to work standalone, and wrap if MemoryObj is available.
        try:
            from lmcache.storage_backend.abstract_backend import MemoryObj
            return MemoryObj.from_buffer(blob, device=self.dst_device)
        except ImportError:
            return _MemoryObjStub(blob, self.dst_device)

    def batched_submit_put_task(
        self,
        keys: list,
        memory_objs: list,
        **kwargs,
    ) -> None:
        """Submit a batch of put operations."""
        for key, mem in zip(keys, memory_objs):
            storage_key = self._map_key(key)
            opts = self._compute_put_opts(key)
            data = self._extract_bytes(mem)
            try:
                self._engine.put(storage_key, data, opts)
            except RuntimeError:
                # Admission/backpressure rejection — acceptable
                pass

    def remove(self, key) -> None:
        storage_key = self._map_key(key)
        self._engine.delete(storage_key)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def _map_key(self, key) -> "_attentiondb.StorageKey":
        """Map LMCache CacheEngineKey to AttentionDB StorageKey."""
        model_name = getattr(key, "model_name", "")
        chunk_hash = getattr(key, "chunk_hash", 0)
        worker_id = getattr(key, "worker_id", 0)

        return _attentiondb.StorageKey(
            model_id=_hash_u64(model_name) if isinstance(model_name, str) else model_name,
            tenant_id=self._tenant_id,
            chunk_hash=chunk_hash if isinstance(chunk_hash, int) else _hash_u64(str(chunk_hash)),
            layer_group_id=worker_id,
            chunk_index=0,
        )

    def _compute_put_opts(self, key) -> "_attentiondb.PutOpts":
        """Compute eviction metadata from the key."""
        entry_type = self._classify_entry(key)
        return _attentiondb.PutOpts(
            num_tokens=self._chunk_size,
            recompute_cost=self._chunk_size * self._cost_per_token,
            entry_type=entry_type,
            ttl_seconds=0,
        )

    def _classify_entry(self, key) -> int:
        """Classify an entry type from LMCache key metadata."""
        # Default to conversation; override if metadata is available
        request_configs = getattr(key, "request_configs", None)
        if request_configs and hasattr(request_configs, "get"):
            kind = request_configs.get("entry_type", "conversation")
            if kind == "system_prompt":
                return ENTRY_TYPE_SYSTEM_PROMPT
            if kind == "rag_context":
                return ENTRY_TYPE_RAG_CONTEXT
        return ENTRY_TYPE_CONVERSATION

    def _extract_bytes(self, mem) -> bytes:
        """Extract raw bytes from a MemoryObj or similar."""
        if isinstance(mem, bytes):
            return mem
        if hasattr(mem, "data"):
            data = mem.data
            if isinstance(data, bytes):
                return data
            if hasattr(data, "tobytes"):
                return data.tobytes()
        if hasattr(mem, "tobytes"):
            return mem.tobytes()
        return bytes(mem)


class _MemoryObjStub:
    """Minimal stand-in for LMCache MemoryObj when lmcache is not installed."""

    def __init__(self, data: bytes, device: str):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def tobytes(self) -> bytes:
        return self.data
