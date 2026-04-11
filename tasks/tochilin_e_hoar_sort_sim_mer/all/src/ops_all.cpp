#include "tochilin_e_hoar_sort_sim_mer/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <vector>

#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

TochilinEHoarSortSimMerALL::TochilinEHoarSortSimMerALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TochilinEHoarSortSimMerALL::ValidationImpl() {
  return !GetInput().empty();
}

bool TochilinEHoarSortSimMerALL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

std::pair<int, int> TochilinEHoarSortSimMerALL::Partition(std::vector<int> &arr, int l, int r) {
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

void TochilinEHoarSortSimMerALL::QuickSortOMP(std::vector<int> &arr, int low, int high, int depth_limit) {
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(low, high);

  while (!stack.empty()) {
    const auto [l0, r0] = stack.back();
    stack.pop_back();

    int l = l0;
    int r = r0;

    if (l >= r) {
      continue;
    }

    const std::pair<int, int> bounds = Partition(arr, l, r);
    int i = bounds.first;
    int j = bounds.second;

    if (depth_limit > 0) {
#pragma omp task default(none) shared(arr) firstprivate(l, j, depth_limit)
      QuickSortOMP(arr, l, j, depth_limit - 1);

#pragma omp task default(none) shared(arr) firstprivate(i, r, depth_limit)
      QuickSortOMP(arr, i, r, depth_limit - 1);
    } else {
      if (l < j) {
        stack.emplace_back(l, j);
      }
      if (i < r) {
        stack.emplace_back(i, r);
      }
    }
  }

#pragma omp taskwait
}

bool TochilinEHoarSortSimMerALL::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

  int proc_count = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (!data.empty()) {
#pragma omp parallel default(none) shared(data)
    {
#pragma omp single
      QuickSortOMP(data, 0, static_cast<int>(data.size()) - 1, 3);
    }
  }

  const int local_sorted = std::ranges::is_sorted(data) ? 1 : 0;
  int sorted_sum = 0;
  MPI_Allreduce(&local_sorted, &sorted_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return sorted_sum == proc_count;
}

bool TochilinEHoarSortSimMerALL::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
