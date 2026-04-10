"""LMCache remote storage plugin for AttentionDB.

Implements the ConnectorAdapter + RemoteConnector pattern required by
LMCache v0.4.2's remote_storage_plugins system.

LMCache config (config.yaml):
    chunk_size: 8
    local_cpu: true
    max_local_cpu_size: 1.0
    remote_url: "attentiondb:///etc/attentiondb/attentiondb.yaml"
    remote_serde: "naive"
    remote_storage_plugins: ["attentiondb"]
    extra_config:
      remote_storage_plugin.attentiondb.module_path: attentiondb.lmcache_adapter
      remote_storage_plugin.attentiondb.class_name: AttentionDBConnectorAdapter
      save_chunk_meta: true
"""

from __future__ import annotations

import logging
import os
from typing import List
from urllib.parse import urlparse

from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

from attentiondb._util import hash_u64

logger = logging.getLogger(__name__)


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


class AttentionDBConnectorAdapter(ConnectorAdapter):
    """ConnectorAdapter for AttentionDB.

    LMCache instantiates this with no arguments, then calls
    ``create_connector(context)`` to get the actual RemoteConnector.
    """

    def __init__(self) -> None:
        super().__init__("attentiondb://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        logger.info("Creating AttentionDB connector for url=%s", context.url)
        config_path = _parse_config_path(context.url)
        return AttentionDBConnector(
            config=context.config,
            metadata=context.metadata,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config_path=config_path,
        )


class AttentionDBConnector(RemoteConnector):
    """RemoteConnector backed by the AttentionDB C++ engine.

    Stores KV cache chunks as raw blobs keyed by a hash of the
    CacheEngineKey fields. When ``save_chunk_meta`` is enabled (default),
    each blob is prefixed with LMCache's RemoteMetadata header so that
    partial-chunk recovery works on cold restart.
    """

    def __init__(
        self,
        config,
        metadata,
        loop,
        local_cpu_backend,
        config_path: str,
    ):
        super().__init__(config, metadata)

        import _attentiondb

        self._adb = _attentiondb
        self._engine = _attentiondb.Engine(config_path)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        logger.info(
            "AttentionDB connector initialised  config=%s  save_chunk_meta=%s",
            config_path,
            self.save_chunk_meta,
        )

    # ── Key mapping ──────────────────────────────────────────────

    def _map_key(self, key):
        model_name = getattr(key, "model_name", "")
        chunk_hash = getattr(key, "chunk_hash", 0)
        worker_id = getattr(key, "worker_id", 0)

        model_id = hash_u64(model_name) if isinstance(model_name, str) else model_name

        if isinstance(chunk_hash, int):
            chunk_hash_u64 = chunk_hash & 0xFFFFFFFFFFFFFFFF
        else:
            chunk_hash_u64 = hash_u64(str(chunk_hash))

        return self._adb.StorageKey(
            model_id=model_id & 0xFFFFFFFFFFFFFFFF,
            tenant_id=0,
            chunk_hash=chunk_hash_u64,
            layer_group_id=worker_id & 0xFFFFFFFFFFFFFFFF,
            chunk_index=0,
        )

    # ── Abstract method implementations ──────────────────────────

    async def exists(self, key) -> bool:
        return await self.loop.run_in_executor(None, self.exists_sync, key)

    def exists_sync(self, key) -> bool:
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

            buf = memoryview(memory_obj.byte_array)
            if buf.format != 'B':
                buf = buf.cast('B')
            actual = min(len(data_bytes), len(buf))
            buf[:actual] = data_bytes[:actual]
            memory_obj = self.reshape_partial_chunk(memory_obj, actual)
            return memory_obj

        except (RuntimeError, ValueError, OverflowError) as e:
            logger.error("AttentionDB get failed for key %s: %s", sk, e)
            return None

    async def put(self, key, memory_obj):
        await self.loop.run_in_executor(None, self._put_sync, key, memory_obj)

    def _put_sync(self, key, memory_obj):
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

            num_tokens = len(data_buffer) // self.single_token_size if self.single_token_size else 0
            opts = self._adb.PutOpts(
                num_tokens=num_tokens,
                recompute_cost=max(num_tokens * 100, 1000),
                entry_type=0,
                ttl_seconds=0,
            )
            self._engine.put(sk, blob, opts)

        except RuntimeError as e:
            logger.debug("AttentionDB put rejected for key %s: %s", sk, e)
        except (ValueError, OverflowError) as e:
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
