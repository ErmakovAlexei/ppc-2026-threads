#include "tochilin_e_hoar_sort_sim_mer/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "oneapi/tbb/parallel_invoke.h"
#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

namespace {

constexpr int kParallelDepthLimit = 3;

}  // namespace

TochilinEHoarSortSimMerTBB::TochilinEHoarSortSimMerTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TochilinEHoarSortSimMerTBB::ValidationImpl() {
  return !GetInput().empty();
}

bool TochilinEHoarSortSimMerTBB::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

std::pair<int, int> TochilinEHoarSortSimMerTBB::Partition(std::vector<int> &arr, int l, int r) {
  int i = l;
  int j = r;
  const int pivot = arr[(l + r) / 2];

  while (i <= j) {
    while (arr[i] < pivot) {
      ++i;
    }
    while (arr[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      ++i;
      --j;
    }
  }

  return {i, j};
}

void TochilinEHoarSortSimMerTBB::QuickSortSequential(std::vector<int> &arr, int low, int high) {
  if (low >= high) {
    return;
  }

  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(low, high);

  while (!stack.empty()) {
    const auto [l, r] = stack.back();
    stack.pop_back();

    if (l >= r) {
      continue;
    }

    const auto [i, j] = Partition(arr, l, r);

    if (l < j) {
      stack.emplace_back(l, j);
    }
    if (i < r) {
      stack.emplace_back(i, r);
    }
  }
}

void TochilinEHoarSortSimMerTBB::QuickSortTBB(std::vector<int> &arr, int low, int high, int depth_limit) {
  if (low >= high) {
    return;
  }

  if (depth_limit <= 0) {
    QuickSortSequential(arr, low, high);
    return;
  }

  const auto [i, j] = Partition(arr, low, high);

  tbb::parallel_invoke(
      [&] {
        if (low < j) {
          QuickSortTBB(arr, low, j, depth_limit - 1);
        }
      },
      [&] {
        if (i < high) {
          QuickSortTBB(arr, i, high, depth_limit - 1);
        }
      });
}

std::vector<int> TochilinEHoarSortSimMerTBB::MergeSortedVectors(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::ranges::merge(a, b, std::back_inserter(result));
  return result;
}

bool TochilinEHoarSortSimMerTBB::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

  const auto mid = static_cast<std::vector<int>::difference_type>(data.size() / 2);
  std::vector<int> left(data.begin(), data.begin() + mid);
  std::vector<int> right(data.begin() + mid, data.end());

  tbb::parallel_invoke(
      [&] {
        QuickSortTBB(left, 0, static_cast<int>(left.size()) - 1, kParallelDepthLimit);
      },
      [&] {
        QuickSortTBB(right, 0, static_cast<int>(right.size()) - 1, kParallelDepthLimit);
      });

  data = MergeSortedVectors(left, right);
  return true;
}

bool TochilinEHoarSortSimMerTBB::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
