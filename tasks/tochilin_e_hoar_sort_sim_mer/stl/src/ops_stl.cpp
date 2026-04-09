#include "tochilin_e_hoar_sort_sim_mer/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

namespace {

constexpr std::size_t kMinPartSize = 4096;

}  // namespace

TochilinEHoarSortSimMerSTL::TochilinEHoarSortSimMerSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TochilinEHoarSortSimMerSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool TochilinEHoarSortSimMerSTL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

std::pair<int, int> TochilinEHoarSortSimMerSTL::Partition(std::vector<int> &arr, int l, int r) {
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

void TochilinEHoarSortSimMerSTL::QuickSortSequential(std::vector<int> &arr, int low, int high) {
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

std::vector<int> TochilinEHoarSortSimMerSTL::MergeSortedVectors(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::ranges::merge(a, b, std::back_inserter(result));
  return result;
}

bool TochilinEHoarSortSimMerSTL::ShouldRunParallel(std::size_t size) {
  return size >= (2 * kMinPartSize) && ppc::util::GetNumThreads() > 1;
}

bool TochilinEHoarSortSimMerSTL::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

  const auto mid = static_cast<std::vector<int>::difference_type>(data.size() / 2);

  std::vector<int> left(data.begin(), data.begin() + mid);
  std::vector<int> right(data.begin() + mid, data.end());

  if (ShouldRunParallel(data.size())) {
    std::thread left_worker([&left] { QuickSortSequential(left, 0, static_cast<int>(left.size()) - 1); });
    std::thread right_worker([&right] { QuickSortSequential(right, 0, static_cast<int>(right.size()) - 1); });
    left_worker.join();
    right_worker.join();
  } else {
    QuickSortSequential(left, 0, static_cast<int>(left.size()) - 1);
    QuickSortSequential(right, 0, static_cast<int>(right.size()) - 1);
  }

  data = MergeSortedVectors(left, right);
  return true;
}

bool TochilinEHoarSortSimMerSTL::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
