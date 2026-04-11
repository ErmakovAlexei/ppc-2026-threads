#include "ermakov_a_spar_mat_mult/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "ermakov_a_spar_mat_mult/common/include/common.hpp"

namespace ermakov_a_spar_mat_mult {

namespace {

struct LocalRowData {
  std::vector<int> cols;
  std::vector<std::complex<double>> vals;
};

void AccumulateRowProducts(const MatrixCRS &a, const MatrixCRS &b, int row_index,
                           std::vector<std::complex<double>> &row_vals, std::vector<int> &row_mark,
                           std::vector<int> &used_cols) {
  used_cols.clear();

  for (int ak = a.row_ptr[static_cast<std::size_t>(row_index)]; ak < a.row_ptr[static_cast<std::size_t>(row_index + 1)];
       ++ak) {
    const int b_row = a.col_index[static_cast<std::size_t>(ak)];
    const auto a_val = a.values[static_cast<std::size_t>(ak)];

    for (int bk = b.row_ptr[static_cast<std::size_t>(b_row)]; bk < b.row_ptr[static_cast<std::size_t>(b_row + 1)];
         ++bk) {
      const int col = b.col_index[static_cast<std::size_t>(bk)];
      const auto product = a_val * b.values[static_cast<std::size_t>(bk)];

      if (row_mark[static_cast<std::size_t>(col)] != row_index) {
        row_mark[static_cast<std::size_t>(col)] = row_index;
        row_vals[static_cast<std::size_t>(col)] = product;
        used_cols.push_back(col);
      } else {
        row_vals[static_cast<std::size_t>(col)] += product;
      }
    }
  }
}

void CollectRowValues(const std::vector<std::complex<double>> &row_vals, std::vector<int> &used_cols,
                      LocalRowData &row) {
  std::ranges::sort(used_cols);
  row.cols.clear();
  row.vals.clear();
  row.cols.reserve(used_cols.size());
  row.vals.reserve(used_cols.size());

  for (int col : used_cols) {
    const auto &value = row_vals[static_cast<std::size_t>(col)];
    if (value != std::complex<double>(0.0, 0.0)) {
      row.cols.push_back(col);
      row.vals.push_back(value);
    }
  }
}

MatrixCRS MultiplyLocalOMP(const MatrixCRS &a, const MatrixCRS &b) {
  MatrixCRS result;
  result.rows = a.rows;
  result.cols = b.cols;
  result.row_ptr.assign(static_cast<std::size_t>(result.rows) + 1ULL, 0);

  if (a.rows == 0 || b.cols == 0) {
    return result;
  }

  std::vector<LocalRowData> rows_data(static_cast<std::size_t>(a.rows));

#pragma omp parallel default(none) shared(a, b, rows_data)
  {
    std::vector<std::complex<double>> row_vals(static_cast<std::size_t>(b.cols), std::complex<double>(0.0, 0.0));
    std::vector<int> row_mark(static_cast<std::size_t>(b.cols), -1);
    std::vector<int> used_cols;
    used_cols.reserve(256);

#pragma omp for
    for (int row = 0; row < a.rows; ++row) {
      AccumulateRowProducts(a, b, row, row_vals, row_mark, used_cols);
      CollectRowValues(row_vals, used_cols, rows_data[static_cast<std::size_t>(row)]);
    }
  }

  int total_nnz = 0;
  for (int row = 0; row < result.rows; ++row) {
    result.row_ptr[static_cast<std::size_t>(row)] = total_nnz;
    total_nnz += static_cast<int>(rows_data[static_cast<std::size_t>(row)].vals.size());
  }
  result.row_ptr[static_cast<std::size_t>(result.rows)] = total_nnz;

  result.values.reserve(static_cast<std::size_t>(total_nnz));
  result.col_index.reserve(static_cast<std::size_t>(total_nnz));

  for (int row = 0; row < result.rows; ++row) {
    const auto &row_data = rows_data[static_cast<std::size_t>(row)];
    result.col_index.insert(result.col_index.end(), row_data.cols.begin(), row_data.cols.end());
    result.values.insert(result.values.end(), row_data.vals.begin(), row_data.vals.end());
  }

  return result;
}

}  // namespace

ErmakovASparMatMultALL::ErmakovASparMatMultALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovASparMatMultALL::ValidateMatrix(const MatrixCRS &m) {
  if (m.rows < 0 || m.cols < 0) {
    return false;
  }
  if (m.row_ptr.size() != static_cast<std::size_t>(m.rows) + 1ULL) {
    return false;
  }
  if (m.values.size() != m.col_index.size()) {
    return false;
  }
  if (m.row_ptr.empty()) {
    return false;
  }

  const int nnz = static_cast<int>(m.values.size());
  if (m.row_ptr.front() != 0 || m.row_ptr.back() != nnz) {
    return false;
  }

  for (int row = 0; row < m.rows; ++row) {
    if (m.row_ptr[static_cast<std::size_t>(row)] > m.row_ptr[static_cast<std::size_t>(row + 1)]) {
      return false;
    }
  }

  for (int idx = 0; idx < nnz; ++idx) {
    if (m.col_index[static_cast<std::size_t>(idx)] < 0 || m.col_index[static_cast<std::size_t>(idx)] >= m.cols) {
      return false;
    }
  }

  return true;
}

bool ErmakovASparMatMultALL::ValidationImpl() {
  const auto &a = GetInput().A;
  const auto &b = GetInput().B;
  return a.cols == b.rows && ValidateMatrix(a) && ValidateMatrix(b);
}

bool ErmakovASparMatMultALL::PreProcessingImpl() {
  a_ = GetInput().A;
  b_ = GetInput().B;
  c_.rows = a_.rows;
  c_.cols = b_.cols;
  c_.values.clear();
  c_.col_index.clear();
  c_.row_ptr.assign(static_cast<std::size_t>(c_.rows) + 1ULL, 0);
  return true;
}

bool ErmakovASparMatMultALL::RunImpl() {
  if (a_.cols != b_.rows) {
    return false;
  }

  c_ = MultiplyLocalOMP(a_, b_);

  const int local_nnz = static_cast<int>(c_.values.size());
  int total_nnz = 0;
  MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return total_nnz >= local_nnz;
}

bool ErmakovASparMatMultALL::PostProcessingImpl() {
  GetOutput() = c_;
  return true;
}

}  // namespace ermakov_a_spar_mat_mult
