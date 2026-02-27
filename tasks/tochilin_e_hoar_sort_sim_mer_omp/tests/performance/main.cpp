#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "tochilin_e_hoar_sort_sim_mer_omp/common/include/common.hpp"
#include "tochilin_e_hoar_sort_sim_mer_omp/omp/include/ops_omp.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_hoar_sort_sim_mer_omp {

class TochilinEHoarSortSimMerRunPerfTestsOMP : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_count = 2000000;
  InType input_data;

  void SetUp() override {
    input_data.resize(static_cast<std::size_t>(k_count));
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(-10000, 10000);

    for (int i = 0; i < k_count; ++i) {
      input_data[static_cast<std::size_t>(i)] = dis(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(TochilinEHoarSortSimMerRunPerfTestsOMP, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEHoarSortSimMerSEQ>(PPC_SETTINGS_tochilin_e_hoar_sort_sim_mer_omp);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TochilinEHoarSortSimMerRunPerfTestsOMP::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TochilinEHoarSortSimMerRunPerfTestsOMP, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tochilin_e_hoar_sort_sim_mer_omp
