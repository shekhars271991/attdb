#include "core/config.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>

namespace attentiondb {

namespace {

size_t ParseSize(const std::string& s) {
    size_t val = 0;
    size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        size_t next = val * 10 + static_cast<size_t>(s[i] - '0');
        if (next < val) return SIZE_MAX;  // overflow
        val = next;
        ++i;
    }
    std::string suffix = s.substr(i);
    size_t multiplier = 1;
    if (suffix == "KB" || suffix == "kb") multiplier = 1024;
    else if (suffix == "MB" || suffix == "mb") multiplier = 1024ULL * 1024;
    else if (suffix == "GB" || suffix == "gb") multiplier = 1024ULL * 1024 * 1024;
    else if (suffix == "TB" || suffix == "tb") multiplier = 1024ULL * 1024 * 1024 * 1024;
    if (val > SIZE_MAX / multiplier) return SIZE_MAX;
    return val * multiplier;
}

}  // namespace

Config Config::Default() {
    return Config{};
}

Config Config::LoadFromString(const std::string& yaml_content) {
    YAML::Node root = YAML::Load(yaml_content);
    Config cfg;

    // Support two YAML layouts:
    //   1. Nested under "attentiondb:" key  (legacy / programmatic)
    //   2. Flat top-level keys              (human-friendly config files)
    YAML::Node adb = root["attentiondb"];
    bool flat = !adb;
    if (flat) adb = root;

    if (adb["mode"]) cfg.mode = adb["mode"].as<std::string>();

    // --- Legacy nested "local" section ---
    if (auto local = adb["local"]) {
        if (local["t1_dram_size"])
            cfg.dram_cache.size_bytes = ParseSize(local["t1_dram_size"].as<std::string>());
        if (local["t2_nvme_path"])
            cfg.nvme_store.paths = {local["t2_nvme_path"].as<std::string>()};
        if (local["t2_nvme_size"])
            cfg.nvme_store.max_size_per_drive = ParseSize(local["t2_nvme_size"].as<std::string>());
        if (local["t2_use_gds"])
            cfg.nvme_store.io_engine = local["t2_use_gds"].as<bool>() ? "gds" : cfg.nvme_store.io_engine;
        if (local["write_block_size_mb"])
            cfg.nvme_store.write_block_size_mb = local["write_block_size_mb"].as<uint32_t>();
        if (local["write_block_pool_size"])
            cfg.nvme_store.write_block_pool_size = local["write_block_pool_size"].as<uint32_t>();
    }

    // --- Flat "dram_cache" section ---
    if (auto dc = adb["dram_cache"]) {
        if (dc["enabled"]) cfg.dram_cache.enabled = dc["enabled"].as<bool>();
        if (dc["size_bytes"]) cfg.dram_cache.size_bytes = dc["size_bytes"].as<size_t>();
        if (dc["hugepages"]) cfg.dram_cache.hugepages = dc["hugepages"].as<bool>();
        if (dc["pinned_memory"]) cfg.dram_cache.pinned_memory = dc["pinned_memory"].as<bool>();
        if (dc["size_classes"]) {
            cfg.dram_cache.size_classes.clear();
            for (const auto& sc : dc["size_classes"])
                cfg.dram_cache.size_classes.push_back(sc.as<size_t>());
        }
    }

    // --- Flat "nvme_store" section ---
    if (auto nv = adb["nvme_store"]) {
        if (nv["enabled"]) cfg.nvme_store.enabled = nv["enabled"].as<bool>();
        if (nv["paths"]) {
            cfg.nvme_store.paths.clear();
            for (const auto& p : nv["paths"])
                cfg.nvme_store.paths.push_back(p.as<std::string>());
        }
        if (nv["segment_size"]) cfg.nvme_store.segment_size = nv["segment_size"].as<size_t>();
        if (nv["io_engine"]) cfg.nvme_store.io_engine = nv["io_engine"].as<std::string>();
        if (nv["gc_enabled"]) cfg.nvme_store.gc_enabled = nv["gc_enabled"].as<bool>();
        if (nv["write_block_size_mb"])
            cfg.nvme_store.write_block_size_mb = nv["write_block_size_mb"].as<uint32_t>();
        if (nv["write_block_pool_size"])
            cfg.nvme_store.write_block_pool_size = nv["write_block_pool_size"].as<uint32_t>();
    }

    // --- admission ---
    if (auto adm = adb["admission"]) {
        if (adm["min_recompute_cost"].IsDefined())
            cfg.admission.min_recompute_cost = adm["min_recompute_cost"].as<uint32_t>();
        if (adm["base_threshold"].IsDefined())
            cfg.admission.base_threshold = adm["base_threshold"].as<uint32_t>();
    }

    // --- eviction ---
    if (auto ev = adb["eviction"]) {
        if (ev["policy"]) cfg.eviction.policy = ev["policy"].as<std::string>();
        if (ev["protected_ratio"]) cfg.eviction.protected_ratio = ev["protected_ratio"].as<double>();
    }

    if (auto to = adb["timeouts"]) {
        if (to["get_ms"]) cfg.get_timeout_ms = to["get_ms"].as<uint32_t>();
    }

    // --- checkpoint (flat "checkpoint" or legacy "index") ---
    if (auto ckpt = adb["checkpoint"]) {
        if (ckpt["interval_s"]) cfg.checkpoint.interval_s = ckpt["interval_s"].as<uint32_t>();
    }
    if (auto idx = adb["index"]) {
        if (idx["checkpoint_interval_s"])
            cfg.checkpoint.interval_s = idx["checkpoint_interval_s"].as<uint32_t>();
    }

    // --- logging ---
    if (auto log = adb["logging"]) {
        if (log["level"]) cfg.logging.level = log["level"].as<std::string>();
        if (log["format"]) cfg.logging.format = log["format"].as<std::string>();
        if (log["periodic_summary_interval_s"])
            cfg.logging.periodic_summary_interval_s = log["periodic_summary_interval_s"].as<uint32_t>();
    }

    return cfg;
}

Config Config::LoadFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return Config::Default();
    std::ostringstream ss;
    ss << f.rdbuf();
    return LoadFromString(ss.str());
}

}  // namespace attentiondb
