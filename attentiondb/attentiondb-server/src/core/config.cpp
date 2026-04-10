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
        val = val * 10 + (s[i] - '0');
        ++i;
    }
    std::string suffix = s.substr(i);
    if (suffix == "KB" || suffix == "kb") return val * 1024;
    if (suffix == "MB" || suffix == "mb") return val * 1024 * 1024;
    if (suffix == "GB" || suffix == "gb") return val * 1024ULL * 1024 * 1024;
    if (suffix == "TB" || suffix == "tb") return val * 1024ULL * 1024 * 1024 * 1024;
    return val;
}

}  // namespace

Config Config::Default() {
    return Config{};
}

Config Config::LoadFromString(const std::string& yaml_content) {
    YAML::Node root = YAML::Load(yaml_content);
    Config cfg;

    auto adb = root["attentiondb"];
    if (!adb) return cfg;

    if (adb["mode"]) cfg.mode = adb["mode"].as<std::string>();

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

    if (auto adm = adb["admission"]) {
        if (adm["min_recompute_cost"].IsDefined())
            cfg.admission.min_recompute_cost = adm["min_recompute_cost"].as<uint32_t>();
        if (adm["base_threshold"].IsDefined())
            cfg.admission.base_threshold = adm["base_threshold"].as<uint32_t>();
    }

    if (auto ev = adb["eviction"]) {
        if (ev["policy"])
            cfg.eviction.policy = ev["policy"].as<std::string>();
        if (ev["protected_ratio"])
            cfg.eviction.protected_ratio = ev["protected_ratio"].as<double>();
    }

    if (auto to = adb["timeouts"]) {
        if (to["get_ms"])
            cfg.get_timeout_ms = to["get_ms"].as<uint32_t>();
    }

    if (auto idx = adb["index"]) {
        if (idx["checkpoint_interval_s"])
            cfg.checkpoint.interval_s = idx["checkpoint_interval_s"].as<uint32_t>();
    }

    if (auto log = adb["logging"]) {
        if (log["level"])
            cfg.logging.level = log["level"].as<std::string>();
        if (log["format"])
            cfg.logging.format = log["format"].as<std::string>();
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
