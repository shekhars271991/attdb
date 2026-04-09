"""LMCache remote storage plugin adapter for AttentionDB.

Plugs AttentionDBBackend into LMCache's remote_storage_plugin system,
mirroring lmc_aerospike_backend.adapter.AerospikeConnectorAdapter.

LMCache config:
    remote_storage_plugins: ["attentiondb"]
    extra_config:
      remote_storage_plugin.attentiondb.module_path: attentiondb.lmcache_adapter
      remote_storage_plugin.attentiondb.class_name: AttentionDBConnectorAdapter
      save_chunk_meta: true
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple
from urllib.parse import urlparse

from attentiondb.backend import AttentionDBBackend

logger = logging.getLogger(__name__)


class _BridgeConfig:
    """Bridges LMCache config fields to what AttentionDBBackend expects."""

    __slots__ = ("attentiondb_config", "tenant_id", "chunk_size", "cost_per_token")

    def __init__(self, config_path: str, chunk_size: int):
        self.attentiondb_config = config_path
        self.tenant_id = 0
        self.chunk_size = chunk_size
        self.cost_per_token = 1


class AttentionDBConnectorAdapter:
    """LMCache remote storage plugin that delegates to AttentionDBBackend.

    Constructor signature matches what LMCache's plugin loader expects:
    ``cls(config, metadata)`` where *config* is an LMCacheEngineConfig and
    *metadata* is an LMCacheEngineMetadata.

    The AttentionDB YAML config path is extracted from ``config.remote_url``:
        attentiondb:///opt/dlami/nvme/attentiondb/config.yaml
    """

    def __init__(self, config, metadata):
        remote_url = getattr(config, "remote_url", "") or ""
        chunk_size = getattr(config, "chunk_size", 8)

        config_path = self._parse_config_path(remote_url)

        dst_device = str(getattr(metadata, "device", "cpu"))
        bridge = _BridgeConfig(config_path, chunk_size)

        self._backend = AttentionDBBackend(dst_device=dst_device, config=bridge)
        logger.info(
            "AttentionDB adapter initialised  config=%s  chunk_size=%d  device=%s",
            config_path,
            chunk_size,
            dst_device,
        )

    # ── StorageBackendInterface delegation ───────────────────────

    def contains(self, key, pin: bool = False) -> bool:
        return self._backend.contains(key, pin=pin)

    def batched_contains(
        self,
        keys: list,
        search_range: Optional[Tuple[int, int]] = None,
        pin: bool = False,
    ) -> int:
        return self._backend.batched_contains(
            keys, search_range=search_range, pin=pin
        )

    def get_blocking(self, key):
        return self._backend.get_blocking(key)

    def batched_submit_put_task(self, keys, memory_objs, **kwargs):
        self._backend.batched_submit_put_task(keys, memory_objs, **kwargs)

    def remove(self, key):
        self._backend.remove(key)

    def close(self):
        logger.info("AttentionDB adapter closing")
        self._backend.close()

    # ── Stats (not part of the interface, but useful for benchmarking) ──

    def stats(self):
        """Expose AttentionDB engine stats for verification scripts."""
        return self._backend._engine.stats() if self._backend._engine else None

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _parse_config_path(remote_url: str) -> str:
        """Extract the AttentionDB YAML path from ``attentiondb:///<path>``."""
        parsed = urlparse(remote_url)
        path = parsed.path
        if not path:
            raise ValueError(
                f"remote_url must contain a config path: {remote_url}"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"AttentionDB config not found at '{path}' "
                f"(parsed from remote_url={remote_url})"
            )
        return path
