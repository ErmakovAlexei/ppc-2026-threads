#include "ermakov_a_spar_mat_mult_tbb/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ranges>
#include <vector>

#include "ermakov_a_spar_mat_mult_tbb/common/include/common.hpp"

namespace ermakov_a_spar_mat_mult_tbb {

ErmakovASparMatMultTBB::ErmakovASparMatMultTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovASparMatMultTBB::ValidateMatrix(const MatrixCRS &m) {
  if (m.rows < 0 || m.cols < 0) {
    return false;
  }

  if (m.row_ptr.size() != static_cast<std::size_t>(m.rows) + 1ULL) {
    return false;
  }

  if (m.col_index.size() != m.values.size()) {
    return false;
  }

  if (!std::ranges::is_sorted(m.row_ptr)) {
    return false;
  }

  if (!m.row_ptr.empty() && !std::cmp_equal(m.row_ptr.back(), m.values.size())) {
    return false;
  }

  if (!std::ranges::all_of(m.col_index, [&](int idx) { return idx >= 0 && idx < m.cols; })) {
    return false;
  }

  return true;
}

bool ErmakovASparMatMultTBB::ValidationImpl() {
  const auto &in = GetInput();
  if (!ValidateMatrix(in.A) || !ValidateMatrix(in.B)) {
    return false;
  }
  if (in.A.cols != in.B.rows) {
    return false;
  }
  return true;
}

bool ErmakovASparMatMultTBB::PreProcessingImpl() {
  const auto &in = GetInput();
  a_ = in.A;
  b_ = in.B;

  c_.rows = a_.rows;
  c_.cols = b_.cols;

  c_.values.clear();
  c_.col_index.clear();
  c_.row_ptr.assign(static_cast<std::size_t>(c_.rows) + 1ULL, 0);

  return true;
}

void ErmakovASparMatMultTBB::ProcessRow(int i, std::vector<std::complex<double>> &row_vals, std::vector<int> &row_mark,
                                        std::vector<int> &used_cols,
                                        std::vector<std::vector<std::complex<double>>> &row_values,
                                        std::vector<std::vector<int>> &row_cols) {
  used_cols.clear();

  const std::size_t row_i = static_cast<std::size_t>(i);

  const int a_start = a_.row_ptr[row_i];
  const int a_end = a_.row_ptr[row_i + 1];

  for (int ak = a_start; ak < a_end; ++ak) {
    const int j = a_.col_index[static_cast<std::size_t>(ak)];
    const auto a_ij = a_.values[static_cast<std::size_t>(ak)];

    const std::size_t row_j = static_cast<std::size_t>(j);

    const int b_start = b_.row_ptr[row_j];
    const int b_end = b_.row_ptr[row_j + 1];

    for (int bk = b_start; bk < b_end; ++bk) {
      const int k = b_.col_index[static_cast<std::size_t>(bk)];
      const auto b_jk = b_.values[static_cast<std::size_t>(bk)];

      const std::size_t col_k = static_cast<std::size_t>(k);

      if (row_mark[col_k] != i) {
        row_mark[col_k] = i;
        row_vals[col_k] = a_ij * b_jk;
        used_cols.push_back(k);
      } else {
        row_vals[col_k] += a_ij * b_jk;
      }
    }
  }

  std::ranges::sort(used_cols);

  auto &cols = row_cols[row_i];
  auto &vals = row_values[row_i];

  cols.clear();
  vals.clear();

  cols.reserve(used_cols.size());
  vals.reserve(used_cols.size());

  for (int k : used_cols) {
    const std::size_t col_k = static_cast<std::size_t>(k);
    const auto &v = row_vals[col_k];

    if (v != std::complex<double>(0.0, 0.0)) {
      cols.push_back(k);
      vals.push_back(v);
    }
  }
}

bool ErmakovASparMatMultTBB::RunImpl() {
  const int m = a_.rows;
  const int p = b_.cols;

  if (a_.cols != b_.rows) {
    return false;
  }

  c_.values.clear();
  c_.col_index.clear();
  std::ranges::fill(c_.row_ptr, 0);

  std::vector<std::vector<std::complex<double>>> row_values(static_cast<std::size_t>(m));
  std::vector<std::vector<int>> row_cols(static_cast<std::size_t>(m));

  tbb::parallel_for(tbb::blocked_range<int>(0, m, 32), [&](const tbb::blocked_range<int> &range) {
    std::vector<std::complex<double>> row_vals(static_cast<std::size_t>(p), std::complex<double>(0.0, 0.0));

    std::vector<int> row_mark(static_cast<std::size_t>(p), -1);

    std::vector<int> used_cols;
    used_cols.reserve(256);

    for (int i = range.begin(); i < range.end(); ++i) {
      ProcessRow(i, row_vals, row_mark, used_cols, row_values, row_cols);
    }
  });

  int nnz = 0;

  for (int i = 0; i < m; ++i) {
    const std::size_t row_i = static_cast<std::size_t>(i);
    c_.row_ptr[row_i] = nnz;
    nnz += static_cast<int>(row_values[row_i].size());
  }

  c_.row_ptr[static_cast<std::size_t>(m)] = nnz;

  c_.values.reserve(static_cast<std::size_t>(nnz));
  c_.col_index.reserve(static_cast<std::size_t>(nnz));

  for (int i = 0; i < m; ++i) {
    const std::size_t row_i = static_cast<std::size_t>(i);

    c_.values.insert(c_.values.end(), row_values[row_i].begin(), row_values[row_i].end());

    c_.col_index.insert(c_.col_index.end(), row_cols[row_i].begin(), row_cols[row_i].end());
  }

  return true;
}

bool ErmakovASparMatMultTBB::PostProcessingImpl() {
  GetOutput() = c_;
  return true;
}

}  // namespace ermakov_a_spar_mat_mult_tbb
