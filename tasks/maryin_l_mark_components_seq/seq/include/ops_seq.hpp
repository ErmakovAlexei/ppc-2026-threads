#pragma once

#include "maryin_l_mark_components_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maryin_l_mark_components_seq {

class MaryinLMarkComponentsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit MaryinLMarkComponentsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool IsBinary(const Image &img);

  void FirstPass();
  void SecondPass();

  Image binary_;
  Labels labels_;
};

}  // namespace maryin_l_mark_components_seq
