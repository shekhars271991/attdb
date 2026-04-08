#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "core/config.h"
#include "core/index_entry.h"
#include "core/status.h"

namespace attentiondb {

struct PutOpts {
    uint32_t num_tokens = 0;
    uint32_t recompute_cost = 0;
    EntryType entry_type = EntryType::kConversation;
    uint32_t ttl_seconds = 0;
};

enum class AdmissionResult : uint8_t {
    kAdmit,
    kRejectLowCost,
    kRejectDynamicThreshold,
};

class AdmissionControl {
public:
    explicit AdmissionControl(const AdmissionConfig& config);

    // Evaluate whether a put should be admitted.
    // `storage_utilization` is the fraction of total storage used (0.0–1.0).
    AdmissionResult evaluate(const PutOpts& opts, double storage_utilization) const;

    // Get the current dynamic threshold given storage utilization.
    uint32_t dynamic_threshold(double utilization) const;

    // Stats
    uint64_t total_evaluated() const { return total_evaluated_.load(); }
    uint64_t total_rejected() const { return total_rejected_.load(); }

    void record_evaluation(AdmissionResult result);

private:
    AdmissionConfig config_;
    mutable std::atomic<uint64_t> total_evaluated_{0};
    mutable std::atomic<uint64_t> total_rejected_{0};
};

}  // namespace attentiondb
