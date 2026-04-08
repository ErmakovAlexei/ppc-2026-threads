#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "task/include/task.hpp"
#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

class TochilinEHoarSortSimMerTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit TochilinEHoarSortSimMerTBB(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void QuickSortSequential(std::vector<int> &arr, int low, int high);
  static std::pair<int, int> Partition(std::vector<int> &arr, int l, int r);
  static int ResolvePartCount(std::size_t size);
  static std::size_t ResolveGrainSize(std::size_t task_count);
  static std::vector<std::size_t> BuildBoundaries(std::size_t size, int part_count);
  static void SortParts(std::vector<int> &data, const std::vector<std::size_t> &boundaries);
  static void MergeRanges(const std::vector<int> &src, std::vector<int> &dst, std::size_t left, std::size_t mid,
                          std::size_t right);
  static std::vector<std::size_t> MergePass(const std::vector<int> &src, std::vector<int> &dst,
                                            const std::vector<std::size_t> &current_boundaries);
};

}  // namespace tochilin_e_hoar_sort_sim_mer
