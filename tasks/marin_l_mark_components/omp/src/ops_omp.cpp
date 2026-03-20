#include "marin_l_mark_components/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

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

  return true;
}

bool MarinLMarkComponentsOMP::RunImpl() {
  FirstPassOMP();
  MergeStripeBorders();
  SecondPassOMP();
  return true;
}

// Первый проход: каждый поток обрабатывает свой диапазон строк строго последовательно
void MarinLMarkComponentsOMP::FirstPassOMP() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());
  const int max_labels = height * width;

  // Инициализация UF
  for (int i = 0; i <= max_labels; ++i) {
    parent_[static_cast<std::size_t>(i)] = i;
  }

  const int num_threads = std::max(1, omp_get_max_threads());
  const int stripe_height = (height + num_threads - 1) / num_threads;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int stripe_start = tid * stripe_height;
    const int stripe_end = std::min(stripe_start + stripe_height, height);

    // Убрали return, заменили условием
    if (stripe_start < height) {
      int next_label = stripe_start * width + 1;

      for (int row = stripe_start; row < stripe_end; ++row) {
        for (int col = 0; col < width; ++col) {
          if (binary_[row][col] == 0) {
            continue;
          }

          int left_label = 0;
          int top_label = 0;

          if (col > 0) {
            left_label = labels_[row][col - 1];
          }
          if (row > stripe_start) {  // верхний сосед внутри той же полосы
            top_label = labels_[row - 1][col];
          }

          if (left_label == 0 && top_label == 0) {
            const int label = next_label++;
            labels_[row][col] = label;
            parent_[static_cast<std::size_t>(label)] = label;
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
          const int max_label_local = std::max(left_label, top_label);
          labels_[row][col] = min_label;

          if (min_label != max_label_local) {
            UnionLabels(parent_, min_label, max_label_local);
          }
        }
      }
    }
  }
}

// Слияние компонент на границах полос
void MarinLMarkComponentsOMP::MergeStripeBorders() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());

  const int num_threads = std::max(1, omp_get_max_threads());
  const int stripe_height = (height + num_threads - 1) / num_threads;

  // Обрабатываем строки, которые являются нижней границей полосы:
  // row = stripe_end - 1 и row + 1 — первая строка следующей полосы.
#pragma omp parallel for schedule(static)
  for (int stripe = 0; stripe < num_threads; ++stripe) {
    const int stripe_start = stripe * stripe_height;
    const int stripe_end = std::min(stripe_start + stripe_height, height);
    const int border_row = stripe_end - 1;
    const int next_row = stripe_end;

    if (border_row < 0 || next_row >= height) {
      continue;
    }

    for (int col = 0; col < width; ++col) {
      if (binary_[border_row][col] == 0 || binary_[next_row][col] == 0) {
        continue;
      }

      const int label_top = labels_[border_row][col];
      const int label_bottom = labels_[next_row][col];

      if (label_top == 0 || label_bottom == 0 || label_top == label_bottom) {
        continue;
      }

      UnionLabels(parent_, label_top, label_bottom);
    }
  }

  const int max_labels = height * width;

  // Финальное сжатие путей
#pragma omp parallel for schedule(static)
  for (int label = 1; label <= max_labels; ++label) {
    parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
  }
}

// Второй проход: параллельная замена на корни + компактация
void MarinLMarkComponentsOMP::SecondPassOMP() {
  const int height = static_cast<int>(labels_.size());
  const int width = height > 0 ? static_cast<int>(labels_.front().size()) : 0;

  if (height == 0 || width == 0) {
    return;
  }

  int max_label = 0;
#pragma omp parallel for reduction(max : max_label) schedule(static)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      max_label = std::max(max_label, labels_[i][j]);
    }
  }

  if (max_label == 0) {
    return;
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int label = labels_[i][j];
      if (label == 0) {
        continue;
      }
      labels_[i][j] = FindRoot(parent_, label);
    }
  }

  // Компактация (однопоточная, K сильно меньше количества пикселей)
  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  int next_id = 1;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int label = labels_[i][j];
      if (label == 0) {
        continue;
      }

      const int root = label;
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
