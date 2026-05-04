#include "tochilin_e_hoar_sort_sim_mer/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/parallel_for.h"
#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

namespace {

constexpr std::size_t kMinPartSize = 4096;
constexpr int kOversubscription = 4;

int ResolveConcurrency() {
  return std::max(1, ppc::util::GetNumThreads());
}

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

int TochilinEHoarSortSimMerTBB::ResolvePartCount(std::size_t size) {
  if (size < (2 * kMinPartSize)) {
    return 1;
  }

  const int concurrency = ResolveConcurrency();
  const int preferred_parts = concurrency * kOversubscription;
  const int max_parts_by_size = static_cast<int>(size / kMinPartSize);
  return std::max(1, std::min(preferred_parts, max_parts_by_size));
}

std::size_t TochilinEHoarSortSimMerTBB::ResolveGrainSize(std::size_t task_count) {
  if (task_count <= 1) {
    return 1;
  }

  const auto concurrency = static_cast<std::size_t>(ResolveConcurrency());
  return std::max<std::size_t>(1, task_count / (concurrency * kOversubscription));
}

std::vector<std::size_t> TochilinEHoarSortSimMerTBB::BuildBoundaries(std::size_t size, int part_count) {
  std::vector<std::size_t> boundaries(static_cast<std::size_t>(part_count) + 1);
  for (int i = 0; i <= part_count; ++i) {
    boundaries[static_cast<std::size_t>(i)] = (static_cast<std::size_t>(i) * size) / part_count;
  }
  return boundaries;
}

void TochilinEHoarSortSimMerTBB::SortParts(std::vector<int> &data, const std::vector<std::size_t> &boundaries) {
  const int part_count = static_cast<int>(boundaries.size()) - 1;
  const auto grain_size = static_cast<int>(ResolveGrainSize(static_cast<std::size_t>(part_count)));

  tbb::parallel_for(tbb::blocked_range<int>(0, part_count, grain_size), [&](const tbb::blocked_range<int> &range) {
    for (int part = range.begin(); part != range.end(); ++part) {
      const std::size_t begin = boundaries[static_cast<std::size_t>(part)];
      const std::size_t end = boundaries[static_cast<std::size_t>(part) + 1];
      if (begin < end) {
        QuickSortSequential(data, static_cast<int>(begin), static_cast<int>(end - 1));
      }
    }
  });
}

void TochilinEHoarSortSimMerTBB::MergeRanges(const std::vector<int> &src, std::vector<int> &dst, std::size_t left,
                                             std::size_t mid, std::size_t right) {
  auto out = dst.begin() + static_cast<std::ptrdiff_t>(left);
  std::ranges::merge(src.begin() + static_cast<std::ptrdiff_t>(left), src.begin() + static_cast<std::ptrdiff_t>(mid),
                     src.begin() + static_cast<std::ptrdiff_t>(mid), src.begin() + static_cast<std::ptrdiff_t>(right),
                     out);
}

std::vector<std::size_t> TochilinEHoarSortSimMerTBB::MergePass(const std::vector<int> &src, std::vector<int> &dst,
                                                               const std::vector<std::size_t> &current_boundaries) {
  const std::size_t current_parts = current_boundaries.size() - 1;
  const std::size_t merge_pairs = current_parts / 2;
  const auto grain_size = ResolveGrainSize(merge_pairs);

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, merge_pairs, grain_size),
                    [&](const tbb::blocked_range<std::size_t> &range) {
    for (std::size_t pair_idx = range.begin(); pair_idx != range.end(); ++pair_idx) {
      const std::size_t left = current_boundaries[pair_idx * 2];
      const std::size_t mid = current_boundaries[(pair_idx * 2) + 1];
      const std::size_t right = current_boundaries[(pair_idx * 2) + 2];
      MergeRanges(src, dst, left, mid, right);
    }
  });

  if ((current_parts % 2) != 0U) {
    const std::size_t tail_begin = current_boundaries[current_parts - 1];
    std::ranges::copy(src.begin() + static_cast<std::ptrdiff_t>(tail_begin), src.end(),
                      dst.begin() + static_cast<std::ptrdiff_t>(tail_begin));
  }

  std::vector<std::size_t> next_boundaries;
  next_boundaries.reserve((current_parts / 2) + 2);
  next_boundaries.push_back(0);
  for (std::size_t i = 2; i < current_boundaries.size(); i += 2) {
    next_boundaries.push_back(current_boundaries[i]);
  }
  if ((current_parts % 2) != 0U) {
    next_boundaries.push_back(current_boundaries.back());
  }

  return next_boundaries;
}

bool TochilinEHoarSortSimMerTBB::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

  const tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                    static_cast<std::size_t>(ResolveConcurrency()));

  const int part_count = ResolvePartCount(data.size());

  if (part_count == 1) {
    QuickSortSequential(data, 0, static_cast<int>(data.size()) - 1);
    return true;
  }

  const std::vector<std::size_t> boundaries = BuildBoundaries(data.size(), part_count);
  SortParts(data, boundaries);

  std::vector<int> buffer(data.size());
  std::vector<std::size_t> current_boundaries = boundaries;
  bool data_is_source = true;

  while ((current_boundaries.size() - 1) > 1) {
    const auto &src = data_is_source ? data : buffer;
    auto &dst = data_is_source ? buffer : data;
    current_boundaries = MergePass(src, dst, current_boundaries);
    data_is_source = !data_is_source;
  }

  if (!data_is_source) {
    data = std::move(buffer);
  }

  return true;
}

bool TochilinEHoarSortSimMerTBB::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
