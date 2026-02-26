#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tochilin_e_hoar_sort_sim_mer/common/include/common.hpp"

namespace tochilin_e_hoar_sort_sim_mer {

class TochilinEHoarSortSimMerSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TochilinEHoarSortSimMerSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void QuickSort(std::vector<int> &arr, int low, int high);
  std::vector<int> MergeSortedVectors(const std::vector<int> &a, const std::vector<int> &b);
};

}  // namespace tochilin_e_hoar_sort_sim_mer
