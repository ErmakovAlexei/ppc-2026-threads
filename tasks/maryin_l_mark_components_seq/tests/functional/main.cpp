#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "maryin_l_mark_components_seq/common/include/common.hpp"
#include "maryin_l_mark_components_seq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace maryin_l_mark_components_seq {

namespace {

using Image = std::vector<std::vector<int>>;
using Labels = std::vector<std::vector<int>>;

Image MakeImage(int h, int w, int fill = 0) {
  return Image(static_cast<std::size_t>(h), std::vector<int>(static_cast<std::size_t>(w), fill));
}

bool SameComponentStructure(const Labels &a, const Labels &b) {
  if (a.size() != b.size() || a.empty() || a[0].size() != b[0].size()) {
    return false;
  }

  const int h = static_cast<int>(a.size());
  const int w = static_cast<int>(a[0].size());

  auto normalize = [h, w](const Labels &src) -> Labels {
    Labels res = src;
    std::vector<int> remap(100000, 0);
    int next_id = 1;

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int lbl = res[y][x];
        if (lbl == 0) {
          continue;
        }

        if (lbl >= static_cast<int>(remap.size())) {
          remap.resize(static_cast<std::size_t>(lbl + 1000), 0);
        }
        if (remap[static_cast<std::size_t>(lbl)] == 0) {
          remap[static_cast<std::size_t>(lbl)] = next_id++;
        }
        res[y][x] = remap[static_cast<std::size_t>(lbl)];
      }
    }
    return res;
  };

  return normalize(a) == normalize(b);
}

Labels ComputeReferenceLabels(const Image &binary) {
  const int h = static_cast<int>(binary.size());
  const int w = h ? static_cast<int>(binary[0].size()) : 0;

  Labels result(static_cast<std::size_t>(h), std::vector<int>(static_cast<std::size_t>(w), 0));
  int label = 0;

  const auto inside = [h, w](int y, int x) { return y >= 0 && y < h && x >= 0 && x < w; };

  std::vector<std::pair<int, int>> stack;
  stack.reserve(static_cast<std::size_t>(h * w / 4));

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (binary[y][x] == 1 && result[y][x] == 0) {
        ++label;

        stack.clear();
        stack.emplace_back(y, x);
        result[y][x] = label;

        while (!stack.empty()) {
          auto [cy, cx] = stack.back();
          stack.pop_back();

          const int dy[] = {-1, 1, 0, 0};
          const int dx[] = {0, 0, -1, 1};

          for (int d = 0; d < 4; ++d) {
            int ny = cy + dy[d];
            int nx = cx + dx[d];

            if (inside(ny, nx) && binary[ny][nx] == 1 && result[ny][nx] == 0) {
              result[ny][nx] = label;
              stack.emplace_back(ny, nx);
            }
          }
        }
      }
    }
  }
  return result;
}

}  // namespace

class MaryinLRunFuncTestComponents : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_" +
           std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    const auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const int w = std::get<0>(params);
    const int h = std::get<1>(params);
    const std::string &scenario = std::get<2>(params);

    input_data_.binary = MakeImage(h, w);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    if (scenario == "SingleBlob") {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          input_data_.binary[y][x] = 1;
        }
      }
    } else if (scenario == "TwoBlocks") {
      for (int y = 1; y < h / 2; ++y) {
        for (int x = 1; x < w / 2; ++x) {
          input_data_.binary[y][x] = 1;
        }
      }
      for (int y = h / 2 + 1; y < h - 1; ++y) {
        for (int x = w / 2 + 1; x < w - 1; ++x) {
          input_data_.binary[y][x] = 1;
        }
      }
    } else if (scenario == "Checker") {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          input_data_.binary[y][x] = ((y + x) % 2);
        }
      }
    } else if (scenario == "Diagonal") {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          input_data_.binary[y][x] = ((y % 3 == x % 3) ? 1 : 0);
        }
      }
    } else if (scenario == "Random") {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          input_data_.binary[y][x] = (dist_prob(gen) < 0.25) ? 1 : 0;
        }
      }
    }

    expected_output_.labels = ComputeReferenceLabels(input_data_.binary);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return SameComponentStructure(expected_output_.labels, output_data.labels);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_{};
};

namespace {

TEST_P(MaryinLRunFuncTestComponents, MarkComponentsSeq) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParams = {
    std::make_tuple(5, 5, "SingleBlob"),  // 1 компонента
    std::make_tuple(8, 10, "TwoBlocks"),  // 2 компоненты
    std::make_tuple(7, 7, "Checker"),     // ~25 компонент
    std::make_tuple(12, 12, "Diagonal"),  // диагональные полосы
    std::make_tuple(15, 10, "Random"),    // случайное
    std::make_tuple(4, 6, "TwoBlocks")    // маленький тест
};

const auto kTestTasksList =
    ppc::util::AddFuncTask<MaryinLMarkComponentsSEQ, InType>(kTestParams, PPC_SETTINGS_maryin_l_mark_components_seq);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = MaryinLRunFuncTestComponents::PrintFuncTestName<MaryinLRunFuncTestComponents>;

INSTANTIATE_TEST_SUITE_P(ComponentLabelingTests, MaryinLRunFuncTestComponents, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace maryin_l_mark_components_seq
