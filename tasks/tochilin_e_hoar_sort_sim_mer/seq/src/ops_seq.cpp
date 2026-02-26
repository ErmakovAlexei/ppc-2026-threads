#include "tochilin_e_hoar_sort_sim_mer/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

TochilinEHoarSortSimMerSEQ::TochilinEHoarSortSimMerSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TochilinEHoarSortSimMerSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TochilinEHoarSortSimMerSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

void TochilinEHoarSortSimMerSEQ::QuickSort(std::vector<int> &arr, int low, int high) {
  if (low >= high) {
    return;
  }

  int pivot = arr[(low + high) / 2];
  int i = low;
  int j = high;

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

  if (low < j) {
    QuickSort(arr, low, j);
  }
  if (i < high) {
    QuickSort(arr, i, high);
  }
}

std::vector<int> TochilinEHoarSortSimMerSEQ::MergeSortedVectors(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
  return result;
}

bool TochilinEHoarSortSimMerSEQ::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

  QuickSort(data, 0, static_cast<int>(data.size()) - 1);

  std::vector<int> left(data.begin(), data.begin() + data.size() / 2);
  std::vector<int> right(data.begin() + data.size() / 2, data.end());
  data = MergeSortedVectors(left, right);

  return true;
}

bool TochilinEHoarSortSimMerSEQ::PostProcessingImpl() {
  return std::is_sorted(GetOutput().begin(), GetOutput().end());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
