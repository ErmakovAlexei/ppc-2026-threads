#include "ermakov_a_spar_mat_mult/omp/include/ops_omp.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ranges>
#include <vector>

#include "ermakov_a_spar_mat_mult/common/include/common.hpp"

namespace ermakov_a_spar_mat_mult {

ErmakovASparMatMultOMP::ErmakovASparMatMultOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovASparMatMultOMP::ValidateMatrix(const MatrixCRS &m) {
  if (m.rows < 0 || m.cols < 0) {
    return false;
  }

  if (m.row_ptr.size() != static_cast<std::size_t>(m.rows) + 1) {
    return false;
  }
  if (m.values.size() != m.col_index.size()) {
    return false;
  }

  const int nnz = static_cast<int>(m.values.size());

  if (m.row_ptr.empty()) {
    return false;
  }
  if (m.row_ptr.front() != 0 || m.row_ptr.back() != nnz) {
    return false;
  }

  for (int i = 0; i < m.rows; ++i) {
    if (m.row_ptr[i] > m.row_ptr[i + 1]) {
      return false;
    }
  }

  for (int k = 0; k < nnz; ++k) {
    if (m.col_index[k] < 0 || m.col_index[k] >= m.cols) {
      return false;
    }
  }

  return true;
}

bool ErmakovASparMatMultOMP::ValidationImpl() {
  const auto &a = GetInput().A;
  const auto &b = GetInput().B;

  if (a.cols != b.rows) {
    return false;
  }
  if (!ValidateMatrix(a)) {
    return false;
  }
  if (!ValidateMatrix(b)) {
    return false;
  }

  return true;
}

bool ErmakovASparMatMultOMP::PreProcessingImpl() {
  a_ = GetInput().A;
  b_ = GetInput().B;

  c_.rows = a_.rows;
  c_.cols = b_.cols;

  c_.values.clear();
  c_.col_index.clear();
  c_.row_ptr.assign(static_cast<std::size_t>(c_.rows) + 1, 0);

  return true;
}

void ErmakovASparMatMultOMP::ProcessRow(int i, std::vector<std::complex<double>> &row_vals, std::vector<int> &row_mark,
                                        std::vector<int> &used_cols,
                                        std::vector<std::vector<std::complex<double>>> &row_values,
                                        std::vector<std::vector<int>> &row_cols) {
  used_cols.clear();

  const int a_start = a_.row_ptr[i];
  const int a_end = a_.row_ptr[i + 1];

  for (int ak = a_start; ak < a_end; ++ak) {
    const int j = a_.col_index[ak];
    const auto a_ij = a_.values[ak];

    const int b_start = b_.row_ptr[j];
    const int b_end = b_.row_ptr[j + 1];

    for (int bk = b_start; bk < b_end; ++bk) {
      const int k = b_.col_index[bk];
      const auto b_jk = b_.values[bk];

      if (row_mark[k] != i) {
        row_mark[k] = i;
        row_vals[k] = a_ij * b_jk;
        used_cols.push_back(k);
      } else {
        row_vals[k] += a_ij * b_jk;
      }
    }
  }

  std::ranges::sort(used_cols);

  auto &cols = row_cols[static_cast<std::size_t>(i)];
  auto &vals = row_values[static_cast<std::size_t>(i)];

  cols.reserve(used_cols.size());
  vals.reserve(used_cols.size());

  for (int k : used_cols) {
    const auto v = row_vals[k];
    if (v != std::complex<double>(0.0, 0.0)) {
      cols.push_back(k);
      vals.push_back(v);
    }
  }
}

void ErmakovASparMatMultOMP::AccumulateRowProducts(int row_index, std::vector<std::complex<double>> &row_vals,
                                                   std::vector<int> &row_mark, std::vector<int> &used_cols) {
  used_cols.clear();

  const int a_start = a_.row_ptr[row_index];
  const int a_end = a_.row_ptr[row_index + 1];

  for (int ak = a_start; ak < a_end; ++ak) {
    const int j = a_.col_index[static_cast<std::size_t>(ak)];
    const auto a_ij = a_.values[static_cast<std::size_t>(ak)];

    const int b_start = b_.row_ptr[j];
    const int b_end = b_.row_ptr[j + 1];

    for (int bk = b_start; bk < b_end; ++bk) {
      const int k = b_.col_index[static_cast<std::size_t>(bk)];
      const auto b_jk = b_.values[static_cast<std::size_t>(bk)];

      if (row_mark[static_cast<std::size_t>(k)] != row_index) {
        row_mark[static_cast<std::size_t>(k)] = row_index;
        row_vals[static_cast<std::size_t>(k)] = a_ij * b_jk;
        used_cols.push_back(k);
      } else {
        row_vals[static_cast<std::size_t>(k)] += a_ij * b_jk;
      }
    }
  }
}

void ermakov_a_spar_mat_mult::ErmakovASparMatMultOMP::SortUsedCols(std::vector<int> &used_cols) {
  std::ranges::sort(used_cols);
}

void ErmakovASparMatMultOMP::CollectRowValues(const std::vector<std::complex<double>> &row_vals,
                                              const std::vector<int> &used_cols, std::vector<int> &cols,
                                              std::vector<std::complex<double>> &vals) {
  cols.clear();
  vals.clear();

  cols.reserve(used_cols.size());
  vals.reserve(used_cols.size());

  for (int col : used_cols) {
    const auto &v = row_vals[static_cast<std::size_t>(col)];
    if (v != std::complex<double>(0.0, 0.0)) {
      cols.push_back(col);
      vals.push_back(v);
    }
  }
}

bool ErmakovASparMatMultOMP::RunImpl() {
  const int m = a_.rows;
  const int p = b_.cols;

  if (a_.cols != b_.rows) {
    return false;
  }

  if (m == 0 || p == 0) {
    return true;
  }

  std::vector<std::vector<std::complex<double>>> row_values(static_cast<std::size_t>(m));
  std::vector<std::vector<int>> row_cols(static_cast<std::size_t>(m));

#pragma omp parallel
  {
    std::vector<std::complex<double>> row_vals(static_cast<std::size_t>(p), std::complex<double>(0.0, 0.0));
    std::vector<int> row_mark(static_cast<std::size_t>(p), -1);
    std::vector<int> used_cols;
    used_cols.reserve(256);

#pragma omp for
    for (int i = 0; i < m; ++i) {
      used_cols.clear();

      const auto row_i = static_cast<std::size_t>(i);
      const int a_start = a_.row_ptr[row_i];
      const int a_end = a_.row_ptr[row_i + 1];

      for (int ak = a_start; ak < a_end; ++ak) {
        const auto ak_i = static_cast<std::size_t>(ak);

        const int j = a_.col_index[ak_i];
        const auto a_ij = a_.values[ak_i];

        const auto row_j = static_cast<std::size_t>(j);
        const int b_start = b_.row_ptr[row_j];
        const int b_end = b_.row_ptr[row_j + 1];

        for (int bk = b_start; bk < b_end; ++bk) {
          const auto bk_i = static_cast<std::size_t>(bk);

          const int k = b_.col_index[bk_i];
          const auto b_jk = b_.values[bk_i];

          const auto col_k = static_cast<std::size_t>(k);

          if (row_mark[col_k] != i) {
            row_mark[col_k] = i;
            row_vals[col_k] = a_ij * b_jk;
            used_cols.push_back(k);
          } else {
            row_vals[col_k] += a_ij * b_jk;
          }
        }
      }

      SortUsedCols(used_cols);

      CollectRowValues(row_vals, used_cols, row_cols[row_i], row_values[row_i]);
    }
  }

  int nnz = 0;

  for (int i = 0; i < m; ++i) {
    const auto row_i = static_cast<std::size_t>(i);
    c_.row_ptr[row_i] = nnz;
    nnz += static_cast<int>(row_values[row_i].size());
  }

  c_.row_ptr[static_cast<std::size_t>(m)] = nnz;

  c_.values.reserve(static_cast<std::size_t>(nnz));
  c_.col_index.reserve(static_cast<std::size_t>(nnz));

  for (int i = 0; i < m; ++i) {
    const auto row_i = static_cast<std::size_t>(i);

    c_.values.insert(c_.values.end(), row_values[row_i].begin(), row_values[row_i].end());

    c_.col_index.insert(c_.col_index.end(), row_cols[row_i].begin(), row_cols[row_i].end());
  }

  return true;
}

bool ErmakovASparMatMultOMP::PostProcessingImpl() {
  GetOutput() = c_;
  return true;
}

}  // namespace ermakov_a_spar_mat_mult
