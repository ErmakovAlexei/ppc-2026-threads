#pragma once

#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_mark_components {

class MarinLMarkComponentsOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit MarinLMarkComponentsOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool IsBinary(const Image &img);

  void FirstPassOMP();
  void MergeStripeBorders();
  void SecondPassOMP();

  Image binary_;
  Labels labels_;
  std::vector<int> parent_;
  int max_label_id_ = 0;
};

}  // namespace marin_l_mark_components
