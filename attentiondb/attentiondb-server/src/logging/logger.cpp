#include "logging/logger.h"

#include <cstdio>
#include <ctime>

namespace attentiondb {

LogLevel ParseLogLevel(const std::string& s) {
    if (s == "debug") return LogLevel::kDebug;
    if (s == "info")  return LogLevel::kInfo;
    if (s == "warn")  return LogLevel::kWarn;
    if (s == "error") return LogLevel::kError;
    return LogLevel::kInfo;
}

static const char* level_str(LogLevel l) {
    switch (l) {
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kInfo:  return "INFO";
        case LogLevel::kWarn:  return "WARN";
        case LogLevel::kError: return "ERROR";
    }
    return "UNKNOWN";
}

Logger::Logger(const LoggingConfig& config)
    : config_(config), min_level_(ParseLogLevel(config.level)) {}

Logger::~Logger() {
    stop();
}

void Logger::debug(const std::string& event, const std::string& json_fields) {
    log(LogLevel::kDebug, event, json_fields);
}

void Logger::info(const std::string& event, const std::string& json_fields) {
    log(LogLevel::kInfo, event, json_fields);
}

void Logger::warn(const std::string& event, const std::string& json_fields) {
    log(LogLevel::kWarn, event, json_fields);
}

void Logger::error(const std::string& event, const std::string& json_fields) {
    log(LogLevel::kError, event, json_fields);
}

void Logger::log(LogLevel level, const std::string& event,
                  const std::string& json_fields) {
    if (level < min_level_) return;

    std::lock_guard<std::mutex> lock(write_mu_);
    fprintf(stderr,
            "{\"ts\":\"%s\",\"level\":\"%s\",\"event\":\"%s\"%s%s}\n",
            timestamp_now().c_str(),
            level_str(level),
            event.c_str(),
            json_fields.empty() ? "" : ",",
            json_fields.c_str());
}

std::string Logger::timestamp_now() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count() % 1000000;

    char buf[64];
    struct tm tm;
    gmtime_r(&time_t, &tm);
    int n = strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(buf + n, sizeof(buf) - n, ".%06ldZ", static_cast<long>(us));
    return buf;
}

void Logger::start_periodic_summary() {
    if (config_.periodic_summary_interval_s == 0) return;
    running_ = true;
    summary_thread_ = std::thread(&Logger::periodic_summary_fn, this);
}

void Logger::stop() {
    running_ = false;
    if (summary_thread_.joinable()) summary_thread_.join();
}

void Logger::periodic_summary_fn() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.periodic_summary_interval_s));
        if (!running_) break;

        auto& s = period_stats_;
        uint64_t gets = s.gets.exchange(0);
        uint64_t t1_hits = s.t1_hits.exchange(0);
        uint64_t t2_hits = s.t2_hits.exchange(0);
        uint64_t misses = s.misses.exchange(0);
        uint64_t puts = s.puts.exchange(0);
        uint64_t puts_rejected = s.puts_rejected.exchange(0);
        uint64_t evictions = s.evictions.exchange(0);
        uint64_t lat_sum = s.get_latency_sum_us.exchange(0);
        uint64_t lat_max = s.get_latency_max_us.exchange(0);

        double total = static_cast<double>(t1_hits + t2_hits + misses);
        double t1_rate = total > 0 ? t1_hits / total : 0;
        double t2_rate = total > 0 ? t2_hits / total : 0;
        double miss_rate = total > 0 ? misses / total : 0;
        uint64_t avg_lat = gets > 0 ? lat_sum / gets : 0;

        char buf[512];
        snprintf(buf, sizeof(buf),
                 "\"interval_s\":%u,"
                 "\"t1_hit_rate\":%.3f,\"t2_hit_rate\":%.3f,\"miss_rate\":%.3f,"
                 "\"gets\":%llu,\"puts\":%llu,\"puts_rejected\":%llu,"
                 "\"evictions\":%llu,"
                 "\"avg_get_latency_us\":%llu,\"p99_get_latency_us\":%llu",
                 config_.periodic_summary_interval_s,
                 t1_rate, t2_rate, miss_rate,
                 (unsigned long long)gets, (unsigned long long)puts,
                 (unsigned long long)puts_rejected,
                 (unsigned long long)evictions,
                 (unsigned long long)avg_lat, (unsigned long long)lat_max);

        info("periodic_summary", buf);
    }
}

}  // namespace attentiondb
