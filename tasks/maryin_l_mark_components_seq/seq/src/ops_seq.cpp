#include "maryin_l_mark_components_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "maryin_l_mark_components_seq/common/include/common.hpp"

namespace maryin_l_mark_components_seq {

namespace detail {
std::vector<int> &GetParentStorage() {
  thread_local std::vector<int> parent_storage;
  return parent_storage;
}
}  // namespace detail

MaryinLMarkComponentsSEQ::MaryinLMarkComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaryinLMarkComponentsSEQ::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel_value : row) {
      if (pixel_value != 0 && pixel_value != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MaryinLMarkComponentsSEQ::ValidationImpl() {
  const auto &img = GetInput().binary;

  if (img.empty()) {
    return false;
  }
  const std::size_t img_width = img[0].size();
  if (img_width == 0) {
    return false;
  }
  for (const auto &row : img) {
    if (row.size() != img_width) {
      return false;
    }
  }

  return IsBinary(img);
}

bool MaryinLMarkComponentsSEQ::PreProcessingImpl() {
  binary_ = GetInput().binary;

  const int height = static_cast<int>(binary_.size());
  if (height <= 0) {
    return false;
  }

  const int width = static_cast<int>(binary_[0].size());
  if (width <= 0) {
    return false;
  }

  if (static_cast<unsigned long long>(height) * width > 100000000ULL) {
    return false;
  }

  labels_.clear();
  labels_.reserve(static_cast<std::size_t>(height));

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    labels_.emplace_back(static_cast<std::size_t>(width), 0);
  }

  return true;
}

bool MaryinLMarkComponentsSEQ::RunImpl() {
  FirstPass();
  SecondPass();
  return true;
}

namespace {
int FindRoot(std::vector<int> &parent, int x) {
  while (parent[static_cast<std::size_t>(x)] != x) {
    parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    x = parent[static_cast<std::size_t>(x)];
  }
  return x;
}

void UnionLabels(std::vector<int> &parent, int a, int b) {
  const int root_a = FindRoot(parent, a);
  const int root_b = FindRoot(parent, b);
  if (root_a == root_b) {
    return;
  }
  if (root_a < root_b) {
    parent[static_cast<std::size_t>(root_b)] = root_a;
  } else {
    parent[static_cast<std::size_t>(root_a)] = root_b;
  }
}
}  // namespace

void MaryinLMarkComponentsSEQ::FirstPass() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_[0].size());

  const int max_labels = height * width;

  auto &parent = detail::GetParentStorage();
  parent.resize(static_cast<std::size_t>(max_labels + 1));
  for (int idx = 0; idx <= max_labels; ++idx) {
    parent[static_cast<std::size_t>(idx)] = idx;
  }

  int next_label = 1;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (binary_[row][col] == 0) {
        labels_[row][col] = 0;
        continue;
      }

      int left_label = 0;
      int top_label = 0;

      if (col > 0 && binary_[row][col - 1] == 1) {
        left_label = labels_[row][col - 1];
      }
      if (row > 0 && binary_[row - 1][col] == 1) {
        top_label = labels_[row - 1][col];
      }

      if (left_label == 0 && top_label == 0) {
        labels_[row][col] = next_label;
        ++next_label;
      } else if (left_label != 0 && top_label == 0) {
        labels_[row][col] = left_label;
      } else if (left_label == 0 && top_label != 0) {
        labels_[row][col] = top_label;
      } else {
        const int min_label = std::min(left_label, top_label);
        labels_[row][col] = min_label;
        if (left_label != top_label) {
          UnionLabels(parent, left_label, top_label);
        }
      }
    }
  }

  for (int label_id = 1; label_id < next_label; ++label_id) {
    parent[static_cast<std::size_t>(label_id)] = FindRoot(parent, label_id);
  }
}

void MaryinLMarkComponentsSEQ::SecondPass() {
  const int height = static_cast<int>(labels_.size());
  const int width = height != 0 ? static_cast<int>(labels_[0].size()) : 0;

  auto &parent = detail::GetParentStorage();

  auto FindRootLocal = [&parent](int x) {
    int root = x;
    while (parent[static_cast<std::size_t>(root)] != root) {
      root = parent[static_cast<std::size_t>(root)];
    }
    return root;
  };

  int max_label = 0;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      max_label = std::max(max_label, labels_[row][col]);
    }
  }

  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  int next_id = 1;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label == 0) {
        continue;
      }
      const int root = FindRootLocal(label);

      if (root >= static_cast<int>(root_to_compact.size())) {
        root_to_compact.resize(static_cast<std::size_t>(root) + 1, 0);
      }

      if (root_to_compact[static_cast<std::size_t>(root)] == 0) {
        root_to_compact[static_cast<std::size_t>(root)] = next_id++;
      }
      labels_[row][col] = root_to_compact[static_cast<std::size_t>(root)];
    }
  }
}

bool MaryinLMarkComponentsSEQ::PostProcessingImpl() {
  OutType out;
  out.labels = labels_;
  GetOutput() = out;
  return true;
}

}  // namespace maryin_l_mark_components_seq
