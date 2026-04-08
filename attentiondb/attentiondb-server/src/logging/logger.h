#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "core/config.h"

namespace attentiondb {

enum class LogLevel : uint8_t {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
};

LogLevel ParseLogLevel(const std::string& s);

class Logger {
public:
    explicit Logger(const LoggingConfig& config);
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void debug(const std::string& event, const std::string& json_fields);
    void info(const std::string& event, const std::string& json_fields);
    void warn(const std::string& event, const std::string& json_fields);
    void error(const std::string& event, const std::string& json_fields);

    // Periodic summary stats (call from engine to update counters)
    struct PeriodStats {
        std::atomic<uint64_t> gets{0};
        std::atomic<uint64_t> t1_hits{0};
        std::atomic<uint64_t> t2_hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> puts{0};
        std::atomic<uint64_t> puts_rejected{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<uint64_t> get_latency_sum_us{0};
        std::atomic<uint64_t> get_latency_max_us{0};
    };

    PeriodStats& period_stats() { return period_stats_; }

    void start_periodic_summary();
    void stop();

private:
    void log(LogLevel level, const std::string& event, const std::string& json_fields);
    void periodic_summary_fn();
    std::string timestamp_now() const;

    LoggingConfig config_;
    LogLevel min_level_;
    std::mutex write_mu_;

    PeriodStats period_stats_;
    std::thread summary_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace attentiondb
