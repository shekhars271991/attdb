"""LMCache remote storage plugin for AttentionDB.

Implements the ConnectorAdapter + RemoteConnector pattern required by
LMCache's remote_storage_plugins system.

LMCache config:
    remote_storage_plugins: ["attentiondb"]
    extra_config:
      remote_storage_plugin.attentiondb.module_path: attentiondb.lmcache_adapter
      remote_storage_plugin.attentiondb.class_name: AttentionDBConnectorAdapter
      save_chunk_meta: true
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PLUGIN_TYPE = "attentiondb"


class AttentionDBConnectorAdapter:
    """ConnectorAdapter for AttentionDB.

    LMCache instantiates this with no arguments, then calls
    ``create_connector(context)`` to get the actual RemoteConnector.
    """

    def __init__(self) -> None:
        from lmcache.v1.storage_backend.connector import ConnectorAdapter

        ConnectorAdapter.__init__(self, "attentiondb://")

    def can_parse(self, url: str) -> bool:
        if url.startswith("attentiondb://"):
            return True
        if url.startswith("plugin://"):
            from lmcache.v1.storage_backend.connector import extract_plugin_type

            pname = url[len("plugin://"):]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context):
        from lmcache.v1.storage_backend.connector.base_connector import (
            RemoteConnector,
        )

        logger.info("Creating AttentionDB connector")

        config_path = _parse_config_path(context.url)

        return AttentionDBConnector(
            context.loop,
            context.local_cpu_backend,
            context.config,
            context.metadata,
            config_path=config_path,
            plugin_name=getattr(context, "plugin_name", None),
        )


class AttentionDBConnector:
    """RemoteConnector implementation backed by the AttentionDB C++ engine.

    Stores KV cache chunks as raw blobs. When ``save_chunk_meta`` is
    enabled, each blob is prefixed with LMCache's RemoteMetadata so
    partial-chunk recovery works on cold restart.
    """

    def __init__(
        self,
        loop,
        local_cpu_backend,
        config,
        metadata,
        config_path: str,
        plugin_name=None,
    ):
        from lmcache.v1.storage_backend.connector.base_connector import (
            RemoteConnector,
        )

        RemoteConnector.__init__(self, config, metadata)

        try:
            import _attentiondb
        except ImportError:
            raise ImportError("attentiondb native module (_attentiondb) not found")

        self._engine = _attentiondb.Engine(config_path)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        logger.info(
            "AttentionDB connector initialised  config=%s  save_chunk_meta=%s",
            config_path,
            self.save_chunk_meta,
        )

    # ── Key mapping ──────────────────────────────────────────────

    @staticmethod
    def _map_key(key):
        """Map a CacheEngineKey to an AttentionDB StorageKey."""
        import _attentiondb

        model_name = getattr(key, "model_name", "")
        chunk_hash = getattr(key, "chunk_hash", 0)
        worker_id = getattr(key, "worker_id", 0)

        model_id = _hash_u64(model_name) if isinstance(model_name, str) else model_name

        return _attentiondb.StorageKey(
            model_id=model_id,
            tenant_id=0,
            chunk_hash=chunk_hash if isinstance(chunk_hash, int) else _hash_u64(str(chunk_hash)),
            layer_group_id=worker_id,
            chunk_index=0,
        )

    # ── Required abstract methods ────────────────────────────────

    async def exists(self, key) -> bool:
        return await self.loop.run_in_executor(None, self._exists_sync, key)

    def exists_sync(self, key) -> bool:
        return self._exists_sync(key)

    def _exists_sync(self, key) -> bool:
        sk = self._map_key(key)
        return self._engine.contains(sk)

    async def get(self, key):
        return await self.loop.run_in_executor(None, self._get_sync, key)

    def _get_sync(self, key):
        from lmcache.v1.protocol import RemoteMetadata

        sk = self._map_key(key)
        blob = self._engine.get(sk)
        if blob is None:
            return None

        try:
            if self.save_chunk_meta:
                md_size = self.remote_metadata_bytes
                if len(blob) < md_size:
                    logger.warning("Blob too small for metadata: %d < %d", len(blob), md_size)
                    return None
                md = RemoteMetadata.deserialize(blob[:md_size])
                data_bytes = blob[md_size:]
                memory_obj = self.local_cpu_backend.allocate(
                    md.shapes, md.dtypes, md.fmt
                )
            else:
                data_bytes = blob
                memory_obj = self.local_cpu_backend.allocate(
                    self.meta_shapes, self.meta_dtypes, self.meta_fmt
                )

            if memory_obj is None:
                logger.debug("Memory allocation failed during AttentionDB get")
                return None

            buf = memory_obj.byte_array
            actual = min(len(data_bytes), len(buf))
            buf[:actual] = data_bytes[:actual]
            memory_obj = self.reshape_partial_chunk(memory_obj, actual)
            return memory_obj

        except Exception as e:
            logger.error("AttentionDB get failed for key %s: %s", sk, e)
            return None

    async def put(self, key, memory_obj):
        await self.loop.run_in_executor(None, self._put_sync, key, memory_obj)

    def _put_sync(self, key, memory_obj):
        import _attentiondb
        from lmcache.v1.protocol import RemoteMetadata

        sk = self._map_key(key)

        try:
            data_buffer = bytes(memory_obj.byte_array)

            if self.save_chunk_meta:
                md = RemoteMetadata(
                    len(data_buffer),
                    memory_obj.get_shapes(),
                    memory_obj.get_dtypes(),
                    memory_obj.get_memory_format(),
                )
                blob = md.serialize() + data_buffer
            else:
                blob = data_buffer

            opts = _attentiondb.PutOpts(
                num_tokens=0,
                recompute_cost=0,
                entry_type=0,
                ttl_seconds=0,
            )
            self._engine.put(sk, blob, opts)

        except RuntimeError:
            pass  # admission/backpressure rejection
        except Exception as e:
            logger.error("AttentionDB put failed for key %s: %s", sk, e)

    async def list(self) -> List[str]:
        return []

    async def close(self):
        logger.info("Closing AttentionDB connector")
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    # ── Optional overrides ───────────────────────────────────────

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        return 0 if self._engine is not None else 1


# ── Helpers ──────────────────────────────────────────────────────

def _hash_u64(s: str) -> int:
    h = hashlib.sha256(s.encode()).digest()[:8]
    return struct.unpack("<Q", h)[0]


def _parse_config_path(remote_url: str) -> str:
    parsed = urlparse(remote_url)
    path = parsed.path
    if not path:
        raise ValueError(f"remote_url must contain a config path: {remote_url}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"AttentionDB config not found at '{path}' "
            f"(parsed from remote_url={remote_url})"
        )
    return path


# Register as a ConnectorAdapter subclass at import time so that
# isinstance() checks pass even though we avoid a direct import
# at class-definition time (which would fail if lmcache isn't installed).
try:
    from lmcache.v1.storage_backend.connector import ConnectorAdapter
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

    ConnectorAdapter.register(AttentionDBConnectorAdapter)
    RemoteConnector.register(AttentionDBConnector)
except ImportError:
    pass
