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
constexpr int kMinRowsPerStripe = 64;

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
  const auto &input_binary = GetInput().binary;
  height_ = static_cast<int>(input_binary.size());
  width_ = static_cast<int>(input_binary.front().size());

  if (height_ <= 0 || width_ <= 0) {
    return false;
  }

  const std::uint64_t pixels = static_cast<std::uint64_t>(height_) * static_cast<std::uint64_t>(width_);
  if (pixels > kMaxPixels) {
    return false;
  }

  binary_.assign(static_cast<std::size_t>(pixels), 0);
  labels_flat_.assign(static_cast<std::size_t>(pixels), 0);
  labels_.clear();
  parent_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  max_label_id_ = 0;

#pragma omp parallel for schedule(static)
  for (int row = 0; row < height_; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
    for (int col = 0; col < width_; ++col) {
      binary_[row_offset + static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(input_binary[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
    }
  }

  return true;
}

bool MarinLMarkComponentsOMP::RunImpl() {
  FirstPassOMP();
  MergeStripeBorders();  // Используем для resolve границ блоков
  SecondPassOMP();
  return true;
}

void MarinLMarkComponentsOMP::FirstPassOMP() {
  stripe_count_ = std::max(1, omp_get_max_threads());
  stripe_count_ = std::min(stripe_count_, height_);
  stripe_count_ = std::min(stripe_count_, std::max(1, height_ / kMinRowsPerStripe));

  std::vector<int> stripe_offsets(static_cast<std::size_t>(stripe_count_) + 1ULL, 0);
  for (int stripe = 0; stripe < stripe_count_; ++stripe) {
    const int row_start = (stripe * height_) / stripe_count_;
    const int row_end = ((stripe + 1) * height_) / stripe_count_;
    stripe_offsets[static_cast<std::size_t>(stripe) + 1ULL] = (row_end - row_start) * width_;
  }
  std::partial_sum(stripe_offsets.begin(), stripe_offsets.end(), stripe_offsets.begin());
  max_label_id_ = stripe_offsets[static_cast<std::size_t>(stripe_count_)];

#pragma omp parallel for schedule(static)
  for (int i = 0; i <= max_label_id_; ++i) {
    parent_[static_cast<std::size_t>(i)] = i;
  }

#pragma omp parallel for schedule(static)
  for (int stripe = 0; stripe < stripe_count_; ++stripe) {
    const int row_start = (stripe * height_) / stripe_count_;
    const int row_end = ((stripe + 1) * height_) / stripe_count_;
    int next_label = 1 + stripe_offsets[static_cast<std::size_t>(stripe)];

    for (int row = row_start; row < row_end; ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      const bool has_top = row > row_start;
      const std::size_t top_row_offset = has_top ? row_offset - static_cast<std::size_t>(width_) : 0;

      for (int col = 0; col < width_; ++col) {
        const std::size_t idx = row_offset + static_cast<std::size_t>(col);
        if (binary_[idx] == 0) {
          continue;
        }

        const int left_label = (col > 0) ? labels_flat_[idx - 1ULL] : 0;
        const int top_label = has_top ? labels_flat_[top_row_offset + static_cast<std::size_t>(col)] : 0;

        if (left_label == 0 && top_label == 0) {
          labels_flat_[idx] = next_label++;
        } else if (left_label != 0 && top_label == 0) {
          labels_flat_[idx] = left_label;
        } else if (left_label == 0 && top_label != 0) {
          labels_flat_[idx] = top_label;
        } else {
          const int min_label = std::min(left_label, top_label);
          labels_flat_[idx] = min_label;
          if (left_label != top_label) {
            UnionLabels(parent_, left_label, top_label);
          }
        }
      }
    }
  }
}

void MarinLMarkComponentsOMP::MergeStripeBorders() {
  for (int stripe = 0; stripe < stripe_count_ - 1; ++stripe) {
    const int border_row = ((stripe + 1) * height_) / stripe_count_;
    const std::size_t top_row_offset = static_cast<std::size_t>(border_row - 1) * static_cast<std::size_t>(width_);
    const std::size_t bottom_row_offset = static_cast<std::size_t>(border_row) * static_cast<std::size_t>(width_);
    for (int col = 0; col < width_; ++col) {
      const std::size_t top_idx = top_row_offset + static_cast<std::size_t>(col);
      const std::size_t bottom_idx = bottom_row_offset + static_cast<std::size_t>(col);
      if (binary_[top_idx] && binary_[bottom_idx]) {
        const int top_label = labels_flat_[top_idx];
        const int bottom_label = labels_flat_[bottom_idx];
        if (top_label > 0 && bottom_label > 0 && top_label != bottom_label) {
          UnionLabels(parent_, top_label, bottom_label);
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int label = 1; label <= max_label_id_; ++label) {
    parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
  }
}

void MarinLMarkComponentsOMP::SecondPassOMP() {
  if (height_ == 0 || width_ == 0) {
    return;
  }

  if (max_label_id_ == 0) {
    return;
  }

  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label_id_ + 1), 0);
  int next_id = 1;
  const std::size_t pixels = static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_);
  for (std::size_t idx = 0; idx < pixels; ++idx) {
    const int label = labels_flat_[idx];
    if (label == 0) {
      continue;
    }

    const int root = parent_[static_cast<std::size_t>(label)];
    if (root_to_compact[static_cast<std::size_t>(root)] == 0) {
      root_to_compact[static_cast<std::size_t>(root)] = next_id++;
    }
    labels_flat_[idx] = root_to_compact[static_cast<std::size_t>(root)];
  }
}

bool MarinLMarkComponentsOMP::PostProcessingImpl() {
  ConvertLabelsToOutput();
  OutType out;
  out.labels = labels_;
  GetOutput() = out;
  return true;
}

void MarinLMarkComponentsOMP::ConvertLabelsToOutput() {
  labels_.clear();
  labels_.resize(static_cast<std::size_t>(height_));

#pragma omp parallel for schedule(static)
  for (int row = 0; row < height_; ++row) {
    labels_[static_cast<std::size_t>(row)].resize(static_cast<std::size_t>(width_));
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
    for (int col = 0; col < width_; ++col) {
      labels_[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
          labels_flat_[row_offset + static_cast<std::size_t>(col)];
    }
  }
}

}  // namespace marin_l_mark_components
