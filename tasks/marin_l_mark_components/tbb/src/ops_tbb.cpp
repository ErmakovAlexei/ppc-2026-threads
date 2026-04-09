#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
constexpr int kMinRowsPerStripe = 64;

int FindRoot(std::vector<int> &parent, int x) {
  int root = x;
  while (parent[root] != root) {
    root = parent[root];
  }

  int current = x;
  while (current != root) {
    const int next = parent[current];
    parent[current] = root;
    current = next;
  }

  return root;
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

MarinLMarkComponentsTBB::MarinLMarkComponentsTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsTBB::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
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

bool MarinLMarkComponentsTBB::PreProcessingImpl() {
  const auto &img = GetInput().binary;
  height_ = static_cast<int>(img.size());
  width_ = static_cast<int>(img.front().size());

  if (height_ <= 0 || width_ <= 0) {
    return false;
  }

  const std::uint64_t total_pixels = static_cast<std::uint64_t>(height_) * static_cast<std::uint64_t>(width_);
  if (total_pixels > kMaxPixels) {
    return false;
  }

  binary_flat_.assign(static_cast<std::size_t>(total_pixels), 0);
  labels_flat_.assign(static_cast<std::size_t>(total_pixels), 0);
  labels_out_.clear();

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      for (int col = 0; col < width_; ++col) {
        binary_flat_[row_offset + static_cast<std::size_t>(col)] =
            static_cast<std::uint8_t>(img[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::RunImpl() {
  if (height_ == 0 || width_ == 0) {
    return true;
  }

  int num_stripes = std::min(height_, oneapi::tbb::this_task_arena::max_concurrency() * 2);
  if (num_stripes > 0 && height_ / num_stripes < kMinRowsPerStripe) {
    num_stripes = std::max(1, height_ / kMinRowsPerStripe);
  }
  num_stripes = std::max(1, num_stripes);

  std::vector<int> stripe_bounds(static_cast<std::size_t>(num_stripes) + 1ULL, 0);
  for (int i = 0; i <= num_stripes; ++i) {
    stripe_bounds[static_cast<std::size_t>(i)] = (i * height_) / num_stripes;
  }

  std::vector<int> stripe_base_label(static_cast<std::size_t>(num_stripes), 0);
  int total_max_labels = 1;
  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    stripe_base_label[static_cast<std::size_t>(stripe)] = total_max_labels;
    const int stripe_height =
        stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL] - stripe_bounds[static_cast<std::size_t>(stripe)];
    total_max_labels += ((stripe_height * width_) / 2) + 1;
  }

  std::vector<int> parent(static_cast<std::size_t>(total_max_labels), 0);
  tbb::parallel_for(tbb::blocked_range<int>(0, total_max_labels), [&](const tbb::blocked_range<int> &range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      parent[static_cast<std::size_t>(i)] = i;
    }
  });

  std::vector<int> stripe_max_used(static_cast<std::size_t>(num_stripes), 0);

  tbb::parallel_for(tbb::blocked_range<int>(0, num_stripes), [&](const tbb::blocked_range<int> &range) {
    for (int stripe = range.begin(); stripe != range.end(); ++stripe) {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
      int next_label = stripe_base_label[static_cast<std::size_t>(stripe)];

      for (int row = start_row; row < end_row; ++row) {
        const int row_offset = row * width_;
        const int prev_row_offset = (row - 1) * width_;

        for (int col = 0; col < width_; ++col) {
          const std::size_t idx = static_cast<std::size_t>(row_offset + col);
          if (binary_flat_[idx] == 0U) {
            continue;
          }

          const int left_label = (col > 0) ? labels_flat_[idx - 1ULL] : 0;
          const int top_label = (row > start_row) ? labels_flat_[static_cast<std::size_t>(prev_row_offset + col)] : 0;

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
              UnionLabels(parent, left_label, top_label);
            }
          }
        }
      }

      stripe_max_used[static_cast<std::size_t>(stripe)] = next_label;
    }
  });

  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    const int boundary_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
    const int row_offset = boundary_row * width_;
    const int prev_row_offset = (boundary_row - 1) * width_;

    for (int col = 0; col < width_; ++col) {
      const std::size_t bottom_idx = static_cast<std::size_t>(row_offset + col);
      const std::size_t top_idx = static_cast<std::size_t>(prev_row_offset + col);
      if (binary_flat_[bottom_idx] == 1U && binary_flat_[top_idx] == 1U) {
        UnionLabels(parent, labels_flat_[bottom_idx], labels_flat_[top_idx]);
      }
    }
  }

  std::vector<int> compacted(static_cast<std::size_t>(total_max_labels), 0);
  int next_compact_id = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    for (int label = stripe_base_label[static_cast<std::size_t>(stripe)];
         label < stripe_max_used[static_cast<std::size_t>(stripe)]; ++label) {
      const int root = FindRoot(parent, label);
      if (compacted[static_cast<std::size_t>(root)] == 0) {
        compacted[static_cast<std::size_t>(root)] = next_compact_id++;
      }
      compacted[static_cast<std::size_t>(label)] = compacted[static_cast<std::size_t>(root)];
    }
  }

  tbb::parallel_for(tbb::blocked_range<int>(0, num_stripes), [&](const tbb::blocked_range<int> &range) {
    for (int stripe = range.begin(); stripe != range.end(); ++stripe) {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];

      for (int row = start_row; row < end_row; ++row) {
        const int row_offset = row * width_;
        for (int col = 0; col < width_; ++col) {
          const std::size_t idx = static_cast<std::size_t>(row_offset + col);
          const int label = labels_flat_[idx];
          if (label != 0) {
            labels_flat_[idx] = compacted[static_cast<std::size_t>(label)];
          }
        }
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  labels_out_.assign(static_cast<std::size_t>(height_), std::vector<int>(static_cast<std::size_t>(width_), 0));

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      std::copy(labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset),
                labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset) + width_,
                labels_out_[static_cast<std::size_t>(row)].begin());
    }
  });

  GetOutput().labels = std::move(labels_out_);
  return true;
}

}  // namespace marin_l_mark_components
