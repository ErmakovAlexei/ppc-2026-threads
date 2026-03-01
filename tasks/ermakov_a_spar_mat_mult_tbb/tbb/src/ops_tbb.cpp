#include "ermakov_a_spar_mat_mult_tbb/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <vector>

#include "ermakov_a_spar_mat_mult_tbb/common/include/common.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace ermakov_a_spar_mat_mult_tbb {

ErmakovASparMatMultTBB::ErmakovASparMatMultTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovASparMatMultTBB::ValidateMatrix(const MatrixCRS &m) {
  if (m.rows < 0 || m.cols < 0) {
    return false;
  }
  if (m.row_ptr.size() != static_cast<std::size_t>(m.rows + 1)) {
    return false;
  }
  if (m.col_index.size() != m.values.size()) {
    return false;
  }
  if (!std::is_sorted(m.row_ptr.begin(), m.row_ptr.end())) {
    return false;
  }
  if (!m.row_ptr.empty() && m.row_ptr.back() != static_cast<int>(m.values.size())) {
    return false;
  }
  for (int idx : m.col_index) {
    if (idx < 0 || idx >= m.cols) {
      return false;
    }
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
  c_.row_ptr.assign(static_cast<std::size_t>(c_.rows + 1), 0);
  return true;
}

void ErmakovASparMatMultTBB::ProcessRow(int row_index, std::vector<std::complex<double>> &row_vals,
                                        std::vector<int> &row_mark, std::vector<int> &used_cols,
                                        std::vector<std::vector<std::complex<double>>> &row_values,
                                        std::vector<std::vector<int>> &row_cols) {
  const int a_row_start = a_.row_ptr[static_cast<std::size_t>(row_index)];
  const int a_row_end = a_.row_ptr[static_cast<std::size_t>(row_index + 1)];

  row_vals.assign(static_cast<std::size_t>(b_.cols), std::complex<double>(0.0, 0.0));
  std::fill(row_mark.begin(), row_mark.end(), -1);
  used_cols.clear();

  for (int idx = a_row_start; idx < a_row_end; ++idx) {
    const int a_col = a_.col_index[static_cast<std::size_t>(idx)];
    const std::complex<double> a_val = a_.values[static_cast<std::size_t>(idx)];

    const int b_row_start = b_.row_ptr[static_cast<std::size_t>(a_col)];
    const int b_row_end = b_.row_ptr[static_cast<std::size_t>(a_col + 1)];

    for (int j = b_row_start; j < b_row_end; ++j) {
      const int b_col = b_.col_index[static_cast<std::size_t>(j)];
      const std::complex<double> b_val = b_.values[static_cast<std::size_t>(j)];

      if (row_mark[static_cast<std::size_t>(b_col)] == -1) {
        row_mark[static_cast<std::size_t>(b_col)] = 1;
        used_cols.push_back(b_col);
        row_vals[static_cast<std::size_t>(b_col)] = a_val * b_val;
      } else {
        row_vals[static_cast<std::size_t>(b_col)] += a_val * b_val;
      }
    }
  }

  auto &local_vals = row_values[static_cast<std::size_t>(row_index)];
  auto &local_cols = row_cols[static_cast<std::size_t>(row_index)];
  local_vals.clear();
  local_cols.clear();

  std::sort(used_cols.begin(), used_cols.end());
  for (int col : used_cols) {
    const auto val = row_vals[static_cast<std::size_t>(col)];
    if (val != std::complex<double>(0.0, 0.0)) {
      local_vals.push_back(val);
      local_cols.push_back(col);
    }
  }
}

bool ErmakovASparMatMultTBB::RunImpl() {
  const int rows = a_.rows;
  const int cols = b_.cols;

  if (rows == 0 || cols == 0) {
    c_.values.clear();
    c_.col_index.clear();
    std::fill(c_.row_ptr.begin(), c_.row_ptr.end(), 0);
    return true;
  }

  std::vector<std::vector<std::complex<double>>> row_values(static_cast<std::size_t>(rows));
  std::vector<std::vector<int>> row_cols(static_cast<std::size_t>(rows));

  tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int> &range) {
    std::vector<std::complex<double>> row_vals(static_cast<std::size_t>(cols), std::complex<double>(0.0, 0.0));
    std::vector<int> row_mark(static_cast<std::size_t>(cols), -1);
    std::vector<int> used_cols;
    used_cols.reserve(static_cast<std::size_t>(cols / 8 + 1));

    for (int i = range.begin(); i < range.end(); ++i) {
      ProcessRow(i, row_vals, row_mark, used_cols, row_values, row_cols);
    }
  });

  // Сборка CRS результата
  c_.row_ptr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    c_.row_ptr[static_cast<std::size_t>(i + 1)] =
        c_.row_ptr[static_cast<std::size_t>(i)] + static_cast<int>(row_values[static_cast<std::size_t>(i)].size());
  }

  const std::size_t nnz = static_cast<std::size_t>(c_.row_ptr[static_cast<std::size_t>(rows)]);
  c_.values.resize(nnz);
  c_.col_index.resize(nnz);

  for (int i = 0; i < rows; ++i) {
    const int row_start = c_.row_ptr[static_cast<std::size_t>(i)];
    const auto &vals = row_values[static_cast<std::size_t>(i)];
    const auto &cols_vec = row_cols[static_cast<std::size_t>(i)];
    const std::size_t row_nnz = vals.size();

    for (std::size_t k = 0; k < row_nnz; ++k) {
      const std::size_t idx = static_cast<std::size_t>(row_start) + k;
      c_.values[idx] = vals[k];
      c_.col_index[idx] = cols_vec[k];
    }
  }

  return true;
}

bool ErmakovASparMatMultTBB::PostProcessingImpl() {
  GetOutput() = c_;
  return true;
}

}  // namespace ermakov_a_spar_mat_mult_tbb
