#include "ermakov_a_spar_mat_mult/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numeric>
#include <vector>

#include "ermakov_a_spar_mat_mult/common/include/common.hpp"

namespace ermakov_a_spar_mat_mult {

namespace {

struct RowChunk {
  int row_begin = 0;
  int row_end = 0;
};

struct LocalRowData {
  std::vector<int> cols;
  std::vector<std::complex<double>> vals;
};

RowChunk ResolveRowChunk(int rank, int size, int total_rows) {
  const int row_begin = (rank * total_rows) / size;
  const int row_end = ((rank + 1) * total_rows) / size;
  return {.row_begin = row_begin, .row_end = row_end};
}

MatrixCRS SliceRows(const MatrixCRS &matrix, int row_begin, int row_end) {
  MatrixCRS local;
  local.rows = row_end - row_begin;
  local.cols = matrix.cols;
  local.row_ptr.assign(static_cast<std::size_t>(local.rows) + 1ULL, 0);

  const int nnz_begin = matrix.row_ptr[static_cast<std::size_t>(row_begin)];
  const int nnz_end = matrix.row_ptr[static_cast<std::size_t>(row_end)];

  local.values.assign(matrix.values.begin() + nnz_begin, matrix.values.begin() + nnz_end);
  local.col_index.assign(matrix.col_index.begin() + nnz_begin, matrix.col_index.begin() + nnz_end);

  for (int row = 0; row < local.rows; ++row) {
    local.row_ptr[static_cast<std::size_t>(row)] =
        matrix.row_ptr[static_cast<std::size_t>(row_begin + row)] - nnz_begin;
  }
  local.row_ptr[static_cast<std::size_t>(local.rows)] = nnz_end - nnz_begin;

  return local;
}

void BroadcastMatrix(MatrixCRS &matrix, int rank) {
  int dims[3] = {matrix.rows, matrix.cols, static_cast<int>(matrix.values.size())};
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.rows = dims[0];
    matrix.cols = dims[1];
    matrix.values.resize(static_cast<std::size_t>(dims[2]));
    matrix.col_index.resize(static_cast<std::size_t>(dims[2]));
    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows) + 1ULL);
  }

  MPI_Bcast(matrix.col_index.data(), dims[2], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(matrix.row_ptr.data(), matrix.rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> real_parts(static_cast<std::size_t>(dims[2]), 0.0);
  std::vector<double> imag_parts(static_cast<std::size_t>(dims[2]), 0.0);

  if (rank == 0) {
    for (int i = 0; i < dims[2]; ++i) {
      real_parts[static_cast<std::size_t>(i)] = matrix.values[static_cast<std::size_t>(i)].real();
      imag_parts[static_cast<std::size_t>(i)] = matrix.values[static_cast<std::size_t>(i)].imag();
    }
  }

  MPI_Bcast(real_parts.data(), dims[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(imag_parts.data(), dims[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    for (int i = 0; i < dims[2]; ++i) {
      matrix.values[static_cast<std::size_t>(i)] =
          std::complex<double>(real_parts[static_cast<std::size_t>(i)], imag_parts[static_cast<std::size_t>(i)]);
    }
  }
}

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

void GatherMatrix(const MatrixCRS &local, MatrixCRS &global, int rank, int size, int total_rows) {
  std::vector<int> row_counts(static_cast<std::size_t>(size), 0);
  const int local_rows = local.rows;
  MPI_Gather(&local_rows, 1, MPI_INT, row_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> nnz_counts(static_cast<std::size_t>(size), 0);
  const int local_nnz = static_cast<int>(local.values.size());
  MPI_Gather(&local_nnz, 1, MPI_INT, nnz_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_row_lengths(static_cast<std::size_t>(local.rows), 0);
  for (int row = 0; row < local.rows; ++row) {
    local_row_lengths[static_cast<std::size_t>(row)] =
        local.row_ptr[static_cast<std::size_t>(row + 1)] - local.row_ptr[static_cast<std::size_t>(row)];
  }

  std::vector<int> row_displs;
  std::vector<int> nnz_displs;
  std::vector<int> gathered_row_lengths;
  std::vector<int> gathered_cols;
  std::vector<double> gathered_real;
  std::vector<double> gathered_imag;

  if (rank == 0) {
    row_displs.resize(static_cast<std::size_t>(size), 0);
    nnz_displs.resize(static_cast<std::size_t>(size), 0);
    for (int proc = 1; proc < size; ++proc) {
      row_displs[static_cast<std::size_t>(proc)] =
          row_displs[static_cast<std::size_t>(proc - 1)] + row_counts[static_cast<std::size_t>(proc - 1)];
      nnz_displs[static_cast<std::size_t>(proc)] =
          nnz_displs[static_cast<std::size_t>(proc - 1)] + nnz_counts[static_cast<std::size_t>(proc - 1)];
    }

    gathered_row_lengths.resize(static_cast<std::size_t>(total_rows), 0);
    gathered_cols.resize(static_cast<std::size_t>(std::accumulate(nnz_counts.begin(), nnz_counts.end(), 0)), 0);
    gathered_real.resize(gathered_cols.size(), 0.0);
    gathered_imag.resize(gathered_cols.size(), 0.0);
  }

  MPI_Gatherv(local_row_lengths.data(), local_rows, MPI_INT, gathered_row_lengths.data(), row_counts.data(),
              row_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local.col_index.data(), local_nnz, MPI_INT, gathered_cols.data(), nnz_counts.data(), nnz_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_real(static_cast<std::size_t>(local_nnz), 0.0);
  std::vector<double> local_imag(static_cast<std::size_t>(local_nnz), 0.0);
  for (int i = 0; i < local_nnz; ++i) {
    local_real[static_cast<std::size_t>(i)] = local.values[static_cast<std::size_t>(i)].real();
    local_imag[static_cast<std::size_t>(i)] = local.values[static_cast<std::size_t>(i)].imag();
  }

  MPI_Gatherv(local_real.data(), local_nnz, MPI_DOUBLE, gathered_real.data(), nnz_counts.data(), nnz_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local_imag.data(), local_nnz, MPI_DOUBLE, gathered_imag.data(), nnz_counts.data(), nnz_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    return;
  }

  global.row_ptr.assign(static_cast<std::size_t>(total_rows) + 1ULL, 0);
  int prefix = 0;
  for (int row = 0; row < total_rows; ++row) {
    global.row_ptr[static_cast<std::size_t>(row)] = prefix;
    prefix += gathered_row_lengths[static_cast<std::size_t>(row)];
  }
  global.row_ptr[static_cast<std::size_t>(total_rows)] = prefix;

  global.col_index = std::move(gathered_cols);
  global.values.resize(static_cast<std::size_t>(prefix));
  for (int i = 0; i < prefix; ++i) {
    global.values[static_cast<std::size_t>(i)] =
        std::complex<double>(gathered_real[static_cast<std::size_t>(i)], gathered_imag[static_cast<std::size_t>(i)]);
  }
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
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0 && a_.cols != b_.rows) {
    return false;
  }

  BroadcastMatrix(b_, rank);
  BroadcastMatrix(a_, rank);

  c_.rows = a_.rows;
  c_.cols = b_.cols;
  c_.values.clear();
  c_.col_index.clear();
  c_.row_ptr.assign(static_cast<std::size_t>(c_.rows) + 1ULL, 0);

  const RowChunk chunk = ResolveRowChunk(rank, size, a_.rows);
  const MatrixCRS local_a = SliceRows(a_, chunk.row_begin, chunk.row_end);
  const MatrixCRS local_c = MultiplyLocalOMP(local_a, b_);

  GatherMatrix(local_c, c_, rank, size, a_.rows);
  BroadcastMatrix(c_, rank);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ErmakovASparMatMultALL::PostProcessingImpl() {
  GetOutput() = c_;
  return true;
}

}  // namespace ermakov_a_spar_mat_mult
