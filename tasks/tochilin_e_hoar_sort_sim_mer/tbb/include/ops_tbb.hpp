#pragma once

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
  static std::vector<int> MergeSortedVectors(const std::vector<int> &a, const std::vector<int> &b);
  static int ResolvePartCount(std::size_t size);
  static void MergeRanges(const std::vector<int> &src, std::vector<int> &dst, std::size_t left, std::size_t mid,
                          std::size_t right);
};

}  // namespace tochilin_e_hoar_sort_sim_mer
