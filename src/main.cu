#include <iostream>
#include <cublas_v2.h>
#include <chrono>
#include <type_traits>
#include <string>

#include <mateval/cuda/comparison.hpp>
#include <mateval/cuda/utils.hpp>

#include <cutf/curand.hpp>
#include <curand_fp16/curand_fp16.hpp>

constexpr unsigned num_test = 128;

template <class T>
inline cudaDataType get_cuda_data_type();
template <>
inline cudaDataType get_cuda_data_type<float>() {return CUDA_R_32F;};
template <>
inline cudaDataType get_cuda_data_type<cuComplex>() {return CUDA_C_32F;};
template <>
inline cudaDataType get_cuda_data_type<double>() {return CUDA_R_64F;};
template <>
inline cudaDataType get_cuda_data_type<cuDoubleComplex>() {return CUDA_C_32F;};
template <>
inline cudaDataType get_cuda_data_type<half>() {return CUDA_R_16F;};

template <class T>
inline T one() {return 1;}
template <>
inline cuComplex one<cuComplex>() {return make_cuComplex(1, 0);}
template <>
inline cuDoubleComplex one<cuDoubleComplex>() {return make_cuDoubleComplex(1, 0);}

inline std::string get_op_str(const cublasOperation_t op) {
  switch(op) {
    case CUBLAS_OP_N:
      return "N";
    case CUBLAS_OP_T:
      return "T";
    case CUBLAS_OP_C:
      return "C";
    default:
      return "Unknown";
  }
}

struct run_gemm_base {
  virtual void operator()(
      cublasOperation_t op_a,
      cublasOperation_t op_b,
      const std::size_t m,
      const std::size_t n,
      const std::size_t k,
      const std::size_t lda,
      const std::size_t ldb,
      const std::size_t ldc
      ) = 0;
};

template <class T>
struct real_type {
  using type = T;
};

template <>
struct real_type<cuDoubleComplex> {
  using type = double;
};

template <>
struct real_type<cuComplex> {
  using type = float;
};

template <class T>
using real_type_v = typename real_type<T>::type;

template <class T>
struct run_gemm : run_gemm_base {
  void operator() (
      cublasOperation_t op_a,
      cublasOperation_t op_b,
      const std::size_t m,
      const std::size_t n,
      const std::size_t k,
      const std::size_t lda,
      const std::size_t ldb,
      const std::size_t ldc
      ) override {
    const auto mat_a_size = lda * (op_a == CUBLAS_OP_N ? k : m);
    const auto mat_b_size = ldb * (op_b == CUBLAS_OP_N ? n : k);
    const auto mat_c_size = ldc * n;

    T *mat_a, *mat_b, *mat_c, *mat_d;
    cudaMalloc(&mat_a, mat_a_size * sizeof(T));
    cudaMalloc(&mat_b, mat_b_size * sizeof(T));
    cudaMalloc(&mat_c, mat_c_size * sizeof(T));
    cudaMalloc(&mat_d, mat_c_size * sizeof(T));

    if constexpr (!std::is_same_v<T, half>) {
      auto cgen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_XORWOW);
      cutf::curand::generate_normal(*cgen.get(), reinterpret_cast<real_type_v<T>*>(mat_a), mat_a_size, 0, 1);
      cutf::curand::generate_normal(*cgen.get(), reinterpret_cast<real_type_v<T>*>(mat_b), mat_b_size, 0, 1);
      cutf::curand::generate_normal(*cgen.get(), reinterpret_cast<real_type_v<T>*>(mat_c), mat_c_size, 0, 1);
    } else {
      mtk::curand_fp16::generator_t cugen;
      mtk::curand_fp16::create(cugen, CURAND_RNG_PSEUDO_XORWOW);

      mtk::curand_fp16::normal(cugen, reinterpret_cast<real_type_v<T>*>(mat_a), mat_a_size, 0, 1);
      mtk::curand_fp16::normal(cugen, reinterpret_cast<real_type_v<T>*>(mat_b), mat_b_size, 0, 1);
      mtk::curand_fp16::normal(cugen, reinterpret_cast<real_type_v<T>*>(mat_c), mat_c_size, 0, 1);
    }

    cudaMemset(mat_a, 0x0, mat_a_size * sizeof(T));
    cudaMemset(mat_b, 0x0, mat_b_size * sizeof(T));
    cudaMemset(mat_c, 0x0, mat_c_size * sizeof(T));
    cudaMemcpy(mat_d, mat_c, mat_c_size * sizeof(T), cudaMemcpyDefault);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cudaDeviceSynchronize();
    const auto start_clock = std::chrono::system_clock::now();

    const T alpha = one<T>(), beta = one<T>();

    cublasGemmEx(
        cublas_handle,
        op_a,
        op_b,
        m, n, k,
        &alpha,
        mat_a, get_cuda_data_type<T>(), lda,
        mat_b, get_cuda_data_type<T>(), ldb,
        &beta,
        mat_d, get_cuda_data_type<T>(), ldc,
        get_cuda_data_type<T>(),
        CUBLAS_GEMM_DEFAULT
        );
    const auto error = mtk::mateval::cuda::get_error_GEMM(
        mtk::mateval::relative_residual,
        m, n, k,
        mtk::mateval::utils::get_mateval_layout(op_a),
        mtk::mateval::utils::get_mateval_layout(op_b),
        mtk::mateval::col_major,
        mtk::mateval::col_major,
        alpha,
        mat_a, lda,
        mat_b, ldb,
        beta,
        mat_c, ldc,
        mat_d, ldc
        );

    for (unsigned t = 0; t < num_test; t++) {
      cublasGemmEx(
          cublas_handle,
          op_a,
          op_b,
          m, n, k,
          &alpha,
          mat_a, get_cuda_data_type<T>(), lda,
          mat_b, get_cuda_data_type<T>(), ldb,
          &beta,
          mat_c, get_cuda_data_type<T>(), ldc,
          get_cuda_data_type<T>(),
          CUBLAS_GEMM_DEFAULT
          );
    }

    cudaDeviceSynchronize();
    const auto end_clock = std::chrono::system_clock::now();

    const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
    auto complexity = 2lu * m * n * k;

    if (std::is_same_v<cuComplex, T> || std::is_same_v<cuDoubleComplex, T>) {
      complexity *= 4;
    }

    std::printf("op_A=%s, op_B=%s, shape=(%lu, %lu, %lu), ld=(%lu, %lu, %lu), throughput=%e TFlop/s, relative_error=%e\n",
                get_op_str(op_a).c_str(),
                get_op_str(op_b).c_str(),
                m, n, k,
                lda, ldb, ldc,
                complexity * num_test / elapsed_time * 1e-12,
                error.at(mtk::mateval::relative_residual)
               );

    cublasDestroy(cublas_handle);

    cudaFree(mat_a);
    cudaFree(mat_b);
    cudaFree(mat_c);
    cudaFree(mat_d);
  }
};

inline cublasOperation_t get_op(const std::string s) {
  if (s == "N" || s == "n") {
    return CUBLAS_OP_N;
  } else if (s == "T" || s == "t") {
    return CUBLAS_OP_T;
  } else if (s == "C" || s == "c") {
    return CUBLAS_OP_C;
  } else {
    throw std::runtime_error("Unknown op: " + s);
  }
}

// ./cublas.out sgemm m n k
int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "%s [gemm type] [op_A] [op_B] [m] [n] [k] (optional: [lda] [ldb] [ldc])\n"
                 " - [gemm type] : s | c | d | z | h\n"
                 " - [op_A/B] : N | T | C\n",
                 argv[0]
                );
    return 1;
  }

  const std::string gemm_mode_str = argv[1];

  const auto op_a = get_op(argv[2]);
  const auto op_b = get_op(argv[3]);
  const auto m = std::stoul(argv[4]);
  const auto n = std::stoul(argv[5]);
  const auto k = std::stoul(argv[6]);

  const auto lda = (argc >= 8) ? std::stoul(argv[7]) : (op_a == CUBLAS_OP_N ? m : k);
  const auto ldb = (argc >= 9) ? std::stoul(argv[8]) : (op_b == CUBLAS_OP_N ? k : n);
  const auto ldc = (argc >= 10) ? std::stoul(argv[9]) : m;

  run_gemm_base* gemm = nullptr;
  if (gemm_mode_str == "s") {
    gemm = new run_gemm<float>;
  } else if (gemm_mode_str == "c") {
    gemm = new run_gemm<cuComplex>;
  } else if (gemm_mode_str == "d") {
    gemm = new run_gemm<double>;
  } else if (gemm_mode_str == "z") {
    gemm = new run_gemm<cuDoubleComplex>;
  } else if (gemm_mode_str == "h") {
    gemm = new run_gemm<half>;
  } else {
    std::fprintf(stderr, "Unknown gemm type : %s\n", gemm_mode_str.c_str());
    return 1;
  }

  (*gemm)(
      op_a,
      op_b,
      m, n, k,
      lda,
      ldb,
      ldc
      );

  delete gemm;
}
