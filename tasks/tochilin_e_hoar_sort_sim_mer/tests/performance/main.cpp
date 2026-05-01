#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "tochilin_e_hoar_sort_sim_mer/all/include/ops_all.hpp"
#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"
#include "tochilin_e_hoar_sort_sim_mer/omp/include/ops_omp.hpp"
#include "tochilin_e_hoar_sort_sim_mer/seq/include/ops_seq.hpp"
#include "tochilin_e_hoar_sort_sim_mer/stl/include/ops_stl.hpp"
#include "tochilin_e_hoar_sort_sim_mer/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

namespace {

constexpr std::uint32_t kPerfSeed = 0x5A17C3D2U;

}  // namespace

class TochilinEHoarSortSimMerRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_count = 2000000;
  InType input_data;

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    const auto &perf_test_param = GetParam();
    const auto &test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(perf_test_param);

    if (test_name.find("_all_") != std::string::npos || test_name.find("_mpi_") != std::string::npos) {
      const double t0 = ppc::util::GetTimeMPI();
      perf_attrs.current_timer = [t0] { return ppc::util::GetTimeMPI() - t0; };
    } else if (test_name.find("_omp_") != std::string::npos) {
      const double t0 = omp_get_wtime();
      perf_attrs.current_timer = [t0] { return omp_get_wtime() - t0; };
    } else {
      const auto t0 = std::chrono::high_resolution_clock::now();
      perf_attrs.current_timer = [t0] {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        return static_cast<double>(ns) * 1e-9;
      };
    }
  }

  void SetUp() override {
    input_data.resize(static_cast<std::size_t>(k_count));
    std::seed_seq seed_seq{static_cast<std::uint32_t>(k_count), kPerfSeed};
    std::mt19937 gen(seed_seq);
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

TEST_P(TochilinEHoarSortSimMerRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEHoarSortSimMerSEQ, TochilinEHoarSortSimMerOMP,
                                TochilinEHoarSortSimMerTBB, TochilinEHoarSortSimMerSTL, TochilinEHoarSortSimMerALL>(
        PPC_SETTINGS_tochilin_e_hoar_sort_sim_mer);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TochilinEHoarSortSimMerRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TochilinEHoarSortSimMerRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tochilin_e_hoar_sort_sim_mer
