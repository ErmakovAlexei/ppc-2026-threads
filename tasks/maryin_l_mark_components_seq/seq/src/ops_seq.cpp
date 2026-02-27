#include "maryin_l_mark_components_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "maryin_l_mark_components_seq/common/include/common.hpp"

namespace maryin_l_mark_components_seq {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

std::vector<int> &GetParentStorage() {
  thread_local std::vector<int> parent_storage;
  return parent_storage;
}

int FindRoot(std::vector<int> &parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
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
    parent[root_b] = root_a;
  } else {
    parent[root_a] = root_b;
  }
}

}  // namespace

MaryinLMarkComponentsSEQ::MaryinLMarkComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaryinLMarkComponentsSEQ::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (const int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MaryinLMarkComponentsSEQ::ValidationImpl() {
  const auto &img = GetInput().binary;

  if (img.empty() || img.front().empty()) {
    return false;
  }

  const std::size_t width = img.front().size();

  for (const auto &row : img) {
    if (row.size() != width) {
      return false;
    }
  }

  return IsBinary(img);
}

bool MaryinLMarkComponentsSEQ::PreProcessingImpl() {
  binary_ = GetInput().binary;

  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());

  if (height <= 0 || width <= 0) {
    return false;
  }

  const std::uint64_t pixels = static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(width);

  if (pixels > kMaxPixels) {
    return false;
  }

  labels_.assign(static_cast<std::size_t>(height), std::vector<int>(static_cast<std::size_t>(width), 0));

  return true;
}

bool MaryinLMarkComponentsSEQ::RunImpl() {
  FirstPass();
  SecondPass();
  return true;
}

void MaryinLMarkComponentsSEQ::FirstPass() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());

  const int max_labels = height * width;

  auto &parent = GetParentStorage();
  parent.assign(static_cast<std::size_t>(max_labels + 1), 0);

  for (int i = 0; i <= max_labels; ++i) {
    parent[i] = i;
  }

  int next_label = 1;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (binary_[row][col] == 0) {
        continue;
      }

      int left_label = 0;
      int top_label = 0;

      if (col > 0) {
        left_label = labels_[row][col - 1];
      }

      if (row > 0) {
        top_label = labels_[row - 1][col];
      }

      if (left_label == 0 && top_label == 0) {
        labels_[row][col] = next_label;
        ++next_label;
        continue;
      }

      if (left_label != 0 && top_label == 0) {
        labels_[row][col] = left_label;
        continue;
      }

      if (left_label == 0 && top_label != 0) {
        labels_[row][col] = top_label;
        continue;
      }

      const int min_label = std::min(left_label, top_label);
      labels_[row][col] = min_label;

      if (left_label != top_label) {
        UnionLabels(parent, left_label, top_label);
      }
    }
  }

  for (int label = 1; label < next_label; ++label) {
    parent[label] = FindRoot(parent, label);
  }
}

void MaryinLMarkComponentsSEQ::SecondPass() {
  const int height = static_cast<int>(labels_.size());
  const int width = height > 0 ? static_cast<int>(labels_.front().size()) : 0;

  auto &parent = GetParentStorage();

  int max_label = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      max_label = std::max(max_label, labels_[i][j]);
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  int next_id = 1;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int label = labels_[i][j];
      if (label == 0) {
        continue;
      }

      const int root = FindRoot(parent, label);

      if (root_to_compact[root] == 0) {
        root_to_compact[root] = next_id++;
      }

      labels_[i][j] = root_to_compact[root];
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
