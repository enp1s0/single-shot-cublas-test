#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>
#include <chrono>

constexpr unsigned num_test = 128;

enum gemm_mode_t {
	sgemm,
	cgemm,
	unknown
};

// ./cublas.out sgemm m n k
int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr,
				"%s [gemm type] [m] [n] [k]\n"
				" - [gemm type] : sgemm / cgemm\n",
				argv[0]
				);
		return 1;
	}

	gemm_mode_t gemm_mode = unknown;
	std::string gemm_mode_str = argv[1];
	if (gemm_mode_str == "sgemm") {
		gemm_mode = sgemm;
	} else if (gemm_mode_str == "cgemm") {
		gemm_mode = cgemm;
	}

	if (gemm_mode == unknown) {
		std::fprintf(stderr, "Unknown gemm type : %s\n", gemm_mode_str.c_str());
		return 1;
	}

	const auto m = std::stoul(argv[2]);
	const auto n = std::stoul(argv[3]);
	const auto k = std::stoul(argv[4]);

	auto mat_a_size = m * k;
	auto mat_b_size = k * n;
	auto mat_c_size = m * n;

	if (gemm_mode == cgemm) {
		mat_a_size *= 2;
		mat_b_size *= 2;
		mat_c_size *= 2;
	}

	float *mat_a, *mat_b, *mat_c;
	cudaMalloc(&mat_a, mat_a_size * sizeof(float));
	cudaMalloc(&mat_b, mat_b_size * sizeof(float));
	cudaMalloc(&mat_c, mat_c_size * sizeof(float));

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	const auto sgemm_func = [&]() {
		const float alpha = 1.0f, beta = 0.0f;
		cublasSgemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_T,
				m, n, k,
				&alpha,
				mat_a, k,
				mat_b, k,
				&beta,
				mat_c, m
				);
	};
	const auto cgemm_func = [&]() {
		const cuComplex alpha = make_cuComplex(1.0f, 00.f), beta = make_cuComplex(0.0f, 0.0f);
		cublasCgemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_T,
				m, n, k,
				&alpha,
				reinterpret_cast<cuComplex*>(mat_a), k,
				reinterpret_cast<cuComplex*>(mat_b), k,
				&beta,
				reinterpret_cast<cuComplex*>(mat_c), m
				);
	};

	if (gemm_mode == sgemm) {
		sgemm_func();
	} else {
		cgemm_func();
	}

	cudaDeviceSynchronize();
	const auto start_clock = std::chrono::system_clock::now();

	for (unsigned t = 0; t < num_test; t++) {
		if (gemm_mode == sgemm) {
			sgemm_func();
		} else {
			cgemm_func();
		}
	}

	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
	auto complexity = 2lu * m * n * k;
	if (gemm_mode == cgemm) {
		complexity *= 4;
	}

	std::printf("shape=(%lu, %lu, %lu), throughput=%e TFlop/s\n",
			m, n, k,
			complexity * num_test / elapsed_time * 1e-12
			);

	cublasDestroy(cublas_handle);

	cudaFree(mat_a);
	cudaFree(mat_b);
	cudaFree(mat_c);
}
