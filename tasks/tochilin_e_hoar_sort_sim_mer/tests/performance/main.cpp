#include <gtest/gtest.h>

#include <random>

#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"
#include "tochilin_e_hoar_sort_sim_mer/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

class TochilinEHoarSortSimMerRunPerfTestsSEQ
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int kCount_ = 50000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(static_cast<std::size_t>(kCount_));
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(-10000, 10000);

    for (int i = 0; i < kCount_; ++i) {
      input_data_[static_cast<std::size_t>(i)] = dis(gen);
    }
  }

  bool CheckTestOutputData(OutType& output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TochilinEHoarSortSimMerRunPerfTestsSEQ, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEHoarSortSimMerSEQ>(
        PPC_SETTINGS_tochilin_e_hoar_sort_sim_mer);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName =
    TochilinEHoarSortSimMerRunPerfTestsSEQ::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(
    RunModeTests,
    TochilinEHoarSortSimMerRunPerfTestsSEQ,
    kGtestValues,
    kPerfTestName);

}  // namespace

}  // namespace tochilin_e_hoar_sort_sim_mer
