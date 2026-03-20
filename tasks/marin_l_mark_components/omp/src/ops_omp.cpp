#include "marin_l_mark_components/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
constexpr int kBlockSize = 64;

int FindRoot(std::vector<int> &parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void UnionLabels(std::vector<int> &parent, int a, int b) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);
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

MarinLMarkComponentsOMP::MarinLMarkComponentsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsOMP::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsOMP::ValidationImpl() {
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

bool MarinLMarkComponentsOMP::PreProcessingImpl() {
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
  parent_.assign(static_cast<std::size_t>(height * width) + 1ULL, 0);
  max_label_id_ = 0;

  return true;
}

bool MarinLMarkComponentsOMP::RunImpl() {
  FirstPassOMP();
  MergeStripeBorders();  // Используем для resolve границ блоков
  SecondPassOMP();
  return true;
}

void MarinLMarkComponentsOMP::FirstPassOMP() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());
  const int num_blocks_h = (height + kBlockSize - 1) / kBlockSize;
  const int num_blocks_w = (width + kBlockSize - 1) / kBlockSize;
  const int num_blocks = num_blocks_h * num_blocks_w;
  std::vector<int> block_offsets(static_cast<std::size_t>(num_blocks) + 1ULL, 0);

  for (int bh = 0; bh < num_blocks_h; ++bh) {
    for (int bw = 0; bw < num_blocks_w; ++bw) {
      const int block_id = (bh * num_blocks_w) + bw;
      const int block_height = std::min(kBlockSize, height - (bh * kBlockSize));
      const int block_width = std::min(kBlockSize, width - (bw * kBlockSize));
      block_offsets[static_cast<std::size_t>(block_id) + 1ULL] = block_height * block_width;
    }
  }
  std::partial_sum(block_offsets.begin(), block_offsets.end(), block_offsets.begin());
  max_label_id_ = (num_blocks > 0) ? block_offsets[static_cast<std::size_t>(num_blocks)] : 0;

#pragma omp parallel for schedule(static)
  for (int i = 0; i <= max_label_id_; ++i) {
    parent_[static_cast<std::size_t>(i)] = i;
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int bh = 0; bh < num_blocks_h; ++bh) {
    for (int bw = 0; bw < num_blocks_w; ++bw) {
      const int row_start = bh * kBlockSize;
      const int row_end = std::min(row_start + kBlockSize, height);
      const int col_start = bw * kBlockSize;
      const int col_end = std::min(col_start + kBlockSize, width);
      const int block_id = (bh * num_blocks_w) + bw;
      const int base_label = 1 + block_offsets[static_cast<std::size_t>(block_id)];
      const int block_capacity =
          block_offsets[static_cast<std::size_t>(block_id) + 1ULL] - block_offsets[static_cast<std::size_t>(block_id)];
      int next_label = base_label;

      for (int row = row_start; row < row_end; ++row) {
        for (int col = col_start; col < col_end; ++col) {
          if (binary_[row][col] == 0) {
            continue;
          }

          int left_label = 0;
          int top_label = 0;
          if (col > col_start) {
            left_label = labels_[row][col - 1];
          }
          if (row > row_start) {
            top_label = labels_[row - 1][col];
          }

          if (left_label == 0 && top_label == 0) {
            if (next_label < base_label + block_capacity) {
              labels_[row][col] = next_label;
              ++next_label;
            }
          } else if (left_label != 0 && top_label == 0) {
            labels_[row][col] = left_label;
          } else if (left_label == 0 && top_label != 0) {
            labels_[row][col] = top_label;
          } else {
            const int min_label = std::min(left_label, top_label);
            labels_[row][col] = min_label;
            if (left_label != top_label) {
              UnionLabels(parent_, left_label, top_label);
            }
          }
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int label = 1; label <= max_label_id_; ++label) {
    parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
  }
}

void MarinLMarkComponentsOMP::MergeStripeBorders() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());
  const int num_blocks_h = (height + kBlockSize - 1) / kBlockSize;
  const int num_blocks_w = (width + kBlockSize - 1) / kBlockSize;

  // Resolve горизонтальных границ блоков
  for (int bh = 0; bh < num_blocks_h - 1; ++bh) {
    const int border_row = (bh + 1) * kBlockSize;
    if (border_row >= height) {
      continue;
    }

    for (int col = 0; col < width; ++col) {
      if (binary_[border_row - 1][col] && binary_[border_row][col]) {
        int l1 = labels_[border_row - 1][col];
        int l2 = labels_[border_row][col];
        if (l1 > 0 && l2 > 0 && l1 != l2) {
          UnionLabels(parent_, l1, l2);
        }
      }
    }
  }

  // Resolve вертикальных границ блоков
  for (int bw = 0; bw < num_blocks_w - 1; ++bw) {
    const int border_col = (bw + 1) * kBlockSize;
    if (border_col >= width) {
      continue;
    }

    for (int row = 0; row < height; ++row) {
      if (binary_[row][border_col - 1] && binary_[row][border_col]) {
        int l1 = labels_[row][border_col - 1];
        int l2 = labels_[row][border_col];
        if (l1 > 0 && l2 > 0 && l1 != l2) {
          UnionLabels(parent_, l1, l2);
        }
      }
    }
  }

  // Финальное сжатие путей
#pragma omp parallel for schedule(static)
  for (int label = 1; label <= max_label_id_; ++label) {
    parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
  }
}

void MarinLMarkComponentsOMP::SecondPassOMP() {
  const int height = static_cast<int>(labels_.size());
  const int width = height > 0 ? static_cast<int>(labels_.front().size()) : 0;

  if (height == 0 || width == 0) {
    return;
  }

  int max_label = 0;
#pragma omp parallel for reduction(max : max_label) schedule(static, 256)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      max_label = std::max(max_label, labels_[i][j]);
    }
  }

  if (max_label == 0) {
    return;
  }

  // Параллельная замена на корни
#pragma omp parallel for schedule(static, 256)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int label = labels_[i][j];
      if (label != 0) {
        labels_[i][j] = FindRoot(parent_, label);
      }
    }
  }

  // Компактация (однопоточная)
  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  int next_id = 1;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int root = labels_[i][j];
      if (root == 0) {
        continue;
      }
      if (root_to_compact[static_cast<std::size_t>(root)] == 0) {
        root_to_compact[static_cast<std::size_t>(root)] = next_id++;
      }
      labels_[i][j] = root_to_compact[static_cast<std::size_t>(root)];
    }
  }
}

bool MarinLMarkComponentsOMP::PostProcessingImpl() {
  OutType out;
  out.labels = labels_;
  GetOutput() = out;
  return true;
}

}  // namespace marin_l_mark_components
