#include "admission/admission.h"

namespace attentiondb {

AdmissionControl::AdmissionControl(const AdmissionConfig& config)
    : config_(config) {}

AdmissionResult AdmissionControl::evaluate(const PutOpts& opts,
                                            double storage_utilization) const {
    // Rule 1: Always admit system prompts
    if (opts.entry_type == EntryType::kSystemPrompt) {
        return AdmissionResult::kAdmit;
    }

    // Rule 2: Reject if recompute_cost below minimum threshold
    if (opts.recompute_cost < config_.min_recompute_cost) {
        return AdmissionResult::kRejectLowCost;
    }

    // Rule 3: Dynamic threshold when storage > 70% full
    uint32_t threshold = dynamic_threshold(storage_utilization);
    if (threshold > 0 && opts.recompute_cost < threshold) {
        return AdmissionResult::kRejectDynamicThreshold;
    }

    return AdmissionResult::kAdmit;
}

uint32_t AdmissionControl::dynamic_threshold(double utilization) const {
    if (utilization < 0.7) return 0;

    if (utilization > 0.9) {
        return config_.base_threshold * 2;
    }

    // Linear scale between 70% and 90%
    double fraction = (utilization - 0.7) / 0.2;
    return static_cast<uint32_t>(config_.base_threshold * fraction);
}

void AdmissionControl::record_evaluation(AdmissionResult result) {
    total_evaluated_.fetch_add(1, std::memory_order_relaxed);
    if (result != AdmissionResult::kAdmit) {
        total_rejected_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace attentiondb
