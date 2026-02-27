#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

#include "maryin_l_mark_components_seq/common/include/common.hpp"
#include "maryin_l_mark_components_seq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maryin_l_mark_components_seq {

namespace {

Image MakeRandomBinaryImage(int h, int w, double fill_prob) {
  Image img(static_cast<std::size_t>(h), std::vector<int>(static_cast<std::size_t>(w), 0));

  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = (dist(gen) < fill_prob) ? 1 : 0;
    }
  }
  return img;
}

}  // namespace

class MaryinLRunPerfTestComponents : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_width = 2048;
  const int k_height = 2048;
  InType input_data_{};

  void SetUp() override {
    input_data_.binary = MakeRandomBinaryImage(k_height, k_width, 0.3);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &labels = output_data.labels;

    if (labels.size() != input_data_.binary.size()) {
      return false;
    }
    if (!labels.empty() && labels[0].size() != input_data_.binary[0].size()) {
      return false;
    }

    const int h = static_cast<int>(labels.size());
    const int w = h ? static_cast<int>(labels[0].size()) : 0;

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (input_data_.binary[y][x] == 0 && labels[y][x] != 0) {
          return false;
        }
        if (input_data_.binary[y][x] == 1 && labels[y][x] <= 0) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MaryinLRunPerfTestComponents, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MaryinLMarkComponentsSEQ>(PPC_SETTINGS_maryin_l_mark_components_seq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaryinLRunPerfTestComponents::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ComponentLabelingPerf, MaryinLRunPerfTestComponents, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace maryin_l_mark_components_seq
