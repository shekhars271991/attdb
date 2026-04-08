#include <gtest/gtest.h>

#include "admission/admission.h"

using namespace attentiondb;

TEST(AdmissionControlTest, SystemPromptAlwaysAdmitted) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 100;
    cfg.base_threshold = 200;
    AdmissionControl ac(cfg);

    PutOpts opts;
    opts.entry_type = EntryType::kSystemPrompt;
    opts.recompute_cost = 0;  // Even with zero cost

    EXPECT_EQ(ac.evaluate(opts, 0.99), AdmissionResult::kAdmit);
}

TEST(AdmissionControlTest, RejectLowCost) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 100;
    AdmissionControl ac(cfg);

    PutOpts opts;
    opts.entry_type = EntryType::kConversation;
    opts.recompute_cost = 50;  // Below minimum

    EXPECT_EQ(ac.evaluate(opts, 0.0), AdmissionResult::kRejectLowCost);
}

TEST(AdmissionControlTest, AdmitAboveMinCost) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 100;
    AdmissionControl ac(cfg);

    PutOpts opts;
    opts.entry_type = EntryType::kConversation;
    opts.recompute_cost = 150;

    EXPECT_EQ(ac.evaluate(opts, 0.0), AdmissionResult::kAdmit);
}

TEST(AdmissionControlTest, DynamicThresholdBelow70Percent) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 10;
    cfg.base_threshold = 100;
    AdmissionControl ac(cfg);

    EXPECT_EQ(ac.dynamic_threshold(0.0), 0u);
    EXPECT_EQ(ac.dynamic_threshold(0.5), 0u);
    EXPECT_EQ(ac.dynamic_threshold(0.69), 0u);
}

TEST(AdmissionControlTest, DynamicThresholdScales) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 10;
    cfg.base_threshold = 100;
    AdmissionControl ac(cfg);

    // At 80% (midpoint of 70-90 range), threshold should be ~50
    uint32_t t = ac.dynamic_threshold(0.8);
    EXPECT_GT(t, 0u);
    EXPECT_LT(t, 100u);
}

TEST(AdmissionControlTest, DynamicThresholdAbove90Percent) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 10;
    cfg.base_threshold = 100;
    AdmissionControl ac(cfg);

    EXPECT_EQ(ac.dynamic_threshold(0.95), 200u);
}

TEST(AdmissionControlTest, RejectDynamicThreshold) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 10;
    cfg.base_threshold = 100;
    AdmissionControl ac(cfg);

    PutOpts opts;
    opts.entry_type = EntryType::kConversation;
    opts.recompute_cost = 50;

    // At 95% utilization, threshold is 200
    EXPECT_EQ(ac.evaluate(opts, 0.95), AdmissionResult::kRejectDynamicThreshold);
}

TEST(AdmissionControlTest, StatsTracking) {
    AdmissionConfig cfg;
    cfg.min_recompute_cost = 100;
    AdmissionControl ac(cfg);

    PutOpts opts;
    opts.recompute_cost = 50;

    auto result = ac.evaluate(opts, 0.0);
    ac.record_evaluation(result);

    EXPECT_EQ(ac.total_evaluated(), 1u);
    EXPECT_EQ(ac.total_rejected(), 1u);
}
