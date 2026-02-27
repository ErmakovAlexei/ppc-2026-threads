#include "maryin_l_mark_components_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "maryin_l_mark_components_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace maryin_l_mark_components_seq {

namespace {
std::vector<int> g_parent;
}  // namespace

MaryinLMarkComponentsSEQ::MaryinLMarkComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaryinLMarkComponentsSEQ::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int v : row) {
      if (v != 0 && v != 1) {
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
  const std::size_t w = img[0].size();
  if (w == 0) {
    return false;
  }
  for (const auto &row : img) {
    if (row.size() != w) {
      return false;
    }
  }

  if (!IsBinary(img)) {
    return false;
  }

  return true;
}

bool MaryinLMarkComponentsSEQ::PreProcessingImpl() {
  binary_ = GetInput().binary;
  const int h = static_cast<int>(binary_.size());
  const int w = static_cast<int>(binary_[0].size());

  labels_.assign(static_cast<std::size_t>(h), std::vector<int>(static_cast<std::size_t>(w), 0));

  return true;
}

bool MaryinLMarkComponentsSEQ::RunImpl() {
  FirstPass();
  SecondPass();
  return true;
}

void MaryinLMarkComponentsSEQ::FirstPass() {
  const int h = static_cast<int>(binary_.size());
  const int w = static_cast<int>(binary_[0].size());

  const int max_labels = h * w;
  g_parent.resize(static_cast<std::size_t>(max_labels + 1));
  for (int i = 0; i <= max_labels; ++i) {
    g_parent[static_cast<std::size_t>(i)] = i;
  }

  auto find_root = [](int x) {
    while (g_parent[static_cast<std::size_t>(x)] != x) {
      g_parent[static_cast<std::size_t>(x)] = g_parent[static_cast<std::size_t>(g_parent[static_cast<std::size_t>(x)])];
      x = g_parent[static_cast<std::size_t>(x)];
    }
    return x;
  };

  auto unite = [&find_root](int a, int b) {
    const int ra = find_root(a);
    const int rb = find_root(b);
    if (ra == rb) {
      return;
    }
    if (ra < rb) {
      g_parent[static_cast<std::size_t>(rb)] = ra;
    } else {
      g_parent[static_cast<std::size_t>(ra)] = rb;
    }
  };

  int next_label = 1;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (binary_[y][x] == 0) {
        labels_[y][x] = 0;
        continue;
      }

      int left_label = 0;
      int top_label = 0;

      if (x > 0 && binary_[y][x - 1] == 1) {
        left_label = labels_[y][x - 1];
      }
      if (y > 0 && binary_[y - 1][x] == 1) {
        top_label = labels_[y - 1][x];
      }

      if (left_label == 0 && top_label == 0) {
        labels_[y][x] = next_label;
        ++next_label;
      } else if (left_label != 0 && top_label == 0) {
        labels_[y][x] = left_label;
      } else if (left_label == 0 && top_label != 0) {
        labels_[y][x] = top_label;
      } else {
        const int m = std::min(left_label, top_label);
        labels_[y][x] = m;
        if (left_label != top_label) {
          unite(left_label, top_label);
        }
      }
    }
  }

  for (int l = 1; l < next_label; ++l) {
    g_parent[static_cast<std::size_t>(l)] = find_root(l);
  }
}

void MaryinLMarkComponentsSEQ::SecondPass() {
  const int h = static_cast<int>(labels_.size());
  const int w = h ? static_cast<int>(labels_[0].size()) : 0;

  auto find_root = [](int x) {
    int r = x;
    while (g_parent[static_cast<std::size_t>(r)] != r) {
      r = g_parent[static_cast<std::size_t>(r)];
    }
    return r;
  };

  int max_label = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (labels_[y][x] > max_label) {
        max_label = labels_[y][x];
      }
    }
  }

  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  int next_id = 1;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int lbl = labels_[y][x];
      if (lbl == 0) {
        continue;
      }
      const int r = find_root(lbl);
      if (r >= static_cast<int>(root_to_compact.size())) {
        root_to_compact.resize(static_cast<std::size_t>(r) + 1, 0);
      }
      if (root_to_compact[static_cast<std::size_t>(r)] == 0) {
        root_to_compact[static_cast<std::size_t>(r)] = next_id++;
      }
      labels_[y][x] = root_to_compact[static_cast<std::size_t>(r)];
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
