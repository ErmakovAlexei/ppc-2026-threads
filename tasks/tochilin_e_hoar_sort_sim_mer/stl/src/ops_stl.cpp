#include "tochilin_e_hoar_sort_sim_mer/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

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

int TochilinEHoarSortSimMerSTL::ResolvePartCount(std::size_t size) {
  if (size < (2 * kMinPartSize)) {
    return 1;
  }

  const int concurrency = ResolveConcurrency();
  const int preferred_parts = concurrency * kOversubscription;
  const int max_parts_by_size = static_cast<int>(size / kMinPartSize);
  return std::max(1, std::min(preferred_parts, max_parts_by_size));
}

std::vector<std::size_t> TochilinEHoarSortSimMerSTL::BuildBoundaries(std::size_t size, int part_count) {
  std::vector<std::size_t> boundaries(static_cast<std::size_t>(part_count) + 1);
  for (int i = 0; i <= part_count; ++i) {
    boundaries[static_cast<std::size_t>(i)] = (static_cast<std::size_t>(i) * size) / part_count;
  }
  return boundaries;
}

void TochilinEHoarSortSimMerSTL::SortParts(std::vector<int> &data, const std::vector<std::size_t> &boundaries) {
  const int part_count = static_cast<int>(boundaries.size()) - 1;
  const int worker_count = std::max(1, std::min(ResolveConcurrency(), part_count));

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(worker_count));

  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&, worker_idx] {
      for (int part = worker_idx; part < part_count; part += worker_count) {
        const std::size_t begin = boundaries[static_cast<std::size_t>(part)];
        const std::size_t end = boundaries[static_cast<std::size_t>(part) + 1];
        if (begin < end) {
          QuickSortSequential(data, static_cast<int>(begin), static_cast<int>(end - 1));
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
}

void TochilinEHoarSortSimMerSTL::MergeRanges(const std::vector<int> &src, std::vector<int> &dst, std::size_t left,
                                             std::size_t mid, std::size_t right) {
  std::ranges::merge(src.begin() + static_cast<std::ptrdiff_t>(left), src.begin() + static_cast<std::ptrdiff_t>(mid),
                     src.begin() + static_cast<std::ptrdiff_t>(mid), src.begin() + static_cast<std::ptrdiff_t>(right),
                     dst.begin() + static_cast<std::ptrdiff_t>(left));
}

std::vector<std::size_t> TochilinEHoarSortSimMerSTL::MergePass(const std::vector<int> &src, std::vector<int> &dst,
                                                               const std::vector<std::size_t> &current_boundaries) {
  const std::size_t current_parts = current_boundaries.size() - 1;
  const std::size_t merge_pairs = current_parts / 2;
  const int worker_count = std::max(1, std::min(ResolveConcurrency(), static_cast<int>(merge_pairs)));

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(worker_count));

  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&, worker_idx] {
      for (auto pair_idx = static_cast<std::size_t>(worker_idx); pair_idx < merge_pairs;
           pair_idx += static_cast<std::size_t>(worker_count)) {
        const std::size_t left = current_boundaries[pair_idx * 2];
        const std::size_t mid = current_boundaries[(pair_idx * 2) + 1];
        const std::size_t right = current_boundaries[(pair_idx * 2) + 2];
        MergeRanges(src, dst, left, mid, right);
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

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

bool TochilinEHoarSortSimMerSTL::RunImpl() {
  auto &data = GetOutput();
  if (data.empty()) {
    return false;
  }

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

bool TochilinEHoarSortSimMerSTL::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace tochilin_e_hoar_sort_sim_mer
