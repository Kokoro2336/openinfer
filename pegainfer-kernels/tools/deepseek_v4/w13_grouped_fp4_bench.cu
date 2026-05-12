#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" int deepseek_tilelang_act_quant_k4096(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

namespace {

constexpr int kInDim = 4096;
constexpr int kOutDim = 2048;
constexpr int kActScaleCols = kInDim / 128;
constexpr int kWeightScaleCols = kInDim / 32;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define TK_CHECK(expr)                                                         \
  do {                                                                         \
    int _err = (expr);                                                         \
    if (_err != 0) {                                                           \
      std::fprintf(stderr, "TileLang launcher error %s:%d: %d\n", __FILE__,    \
                   __LINE__, _err);                                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  int rows = 128;
  int experts = 8;
  int warmup = 20;
  int iters = 200;
  int seed = 42;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto read_int = [&](const char* name, int* out) {
      if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        *out = std::atoi(argv[++i]);
        return true;
      }
      return false;
    };
    if (read_int("--rows", &args.rows) || read_int("--experts", &args.experts) ||
        read_int("--warmup", &args.warmup) || read_int("--iters", &args.iters) ||
        read_int("--seed", &args.seed)) {
      continue;
    }
    std::fprintf(stderr,
                 "usage: %s [--rows N] [--experts N] [--warmup N] [--iters N] "
                 "[--seed N]\n",
                 argv[0]);
    std::exit(2);
  }
  if (args.rows <= 0 || args.experts <= 0 || args.warmup < 0 || args.iters <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(2);
  }
  return args;
}

std::vector<int> make_indptr(int rows, int experts) {
  std::vector<int> counts(experts, 0);
  int remaining = rows;
  for (int e = 0; e < experts; ++e) {
    int left = experts - e;
    int count = (e % 5 == 0) ? 0 : std::max(1, remaining / left);
    count = std::min(count, remaining);
    counts[e] = count;
    remaining -= count;
  }
  counts.back() += remaining;

  std::vector<int> indptr(experts + 1, 0);
  for (int e = 0; e < experts; ++e) {
    indptr[e + 1] = indptr[e] + counts[e];
  }
  return indptr;
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return ptr;
}

void fill_ptrs(
    unsigned char* base,
    size_t stride,
    int experts,
    void*** out_device_ptrs) {
  std::vector<const void*> host(experts);
  for (int e = 0; e < experts; ++e) {
    host[e] = base + e * stride;
  }
  void** device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, experts * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), experts * sizeof(void*), cudaMemcpyHostToDevice));
  *out_device_ptrs = device;
}

float time_ms(cudaStream_t stream, int iters, const std::function<void()>& fn) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int compare_u16(
    const std::vector<uint16_t>& expected,
    const std::vector<uint16_t>& got,
    const char* name) {
  int mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != got[i]) {
      if (mismatches < 8) {
        std::fprintf(stderr, "%s mismatch[%zu]: expected=0x%04x got=0x%04x\n",
                     name, i, expected[i], got[i]);
      }
      ++mismatches;
    }
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> x_dist(-3.0f, 3.0f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> scale_dist(120, 132);

  const size_t x_elems = static_cast<size_t>(args.rows) * kInDim;
  const size_t act_bytes = x_elems;
  const size_t act_scale_bytes = static_cast<size_t>(args.rows) * kActScaleCols;
  const size_t weight_bytes_per_expert = static_cast<size_t>(kOutDim) * kInDim / 2;
  const size_t weight_scale_bytes_per_expert = static_cast<size_t>(kOutDim) * kWeightScaleCols;
  const size_t out_elems = static_cast<size_t>(args.rows) * kOutDim;

  std::vector<__nv_bfloat16> x_host(x_elems);
  for (auto& value : x_host) {
    value = __float2bfloat16(x_dist(rng));
  }
  auto* x = device_copy(x_host);

  unsigned char* act = nullptr;
  unsigned char* act_scale = nullptr;
  CUDA_CHECK(cudaMalloc(&act, act_bytes));
  CUDA_CHECK(cudaMalloc(&act_scale, act_scale_bytes));
  TK_CHECK(deepseek_tilelang_act_quant_k4096(x, act, act_scale, args.rows, stream));

  const size_t all_weight_bytes = weight_bytes_per_expert * args.experts;
  const size_t all_scale_bytes = weight_scale_bytes_per_expert * args.experts;
  std::vector<unsigned char> w1_host(all_weight_bytes);
  std::vector<unsigned char> w3_host(all_weight_bytes);
  std::vector<unsigned char> s1_host(all_scale_bytes);
  std::vector<unsigned char> s3_host(all_scale_bytes);
  for (auto& value : w1_host) value = static_cast<unsigned char>(byte_dist(rng));
  for (auto& value : w3_host) value = static_cast<unsigned char>(byte_dist(rng));
  for (auto& value : s1_host) value = static_cast<unsigned char>(scale_dist(rng));
  for (auto& value : s3_host) value = static_cast<unsigned char>(scale_dist(rng));

  auto* w1 = device_copy(w1_host);
  auto* w3 = device_copy(w3_host);
  auto* s1 = device_copy(s1_host);
  auto* s3 = device_copy(s3_host);
  void** w1_ptrs = nullptr;
  void** w3_ptrs = nullptr;
  void** s1_ptrs = nullptr;
  void** s3_ptrs = nullptr;
  fill_ptrs(w1, weight_bytes_per_expert, args.experts, &w1_ptrs);
  fill_ptrs(w3, weight_bytes_per_expert, args.experts, &w3_ptrs);
  fill_ptrs(s1, weight_scale_bytes_per_expert, args.experts, &s1_ptrs);
  fill_ptrs(s3, weight_scale_bytes_per_expert, args.experts, &s3_ptrs);

  std::vector<int> indptr_host = make_indptr(args.rows, args.experts);
  auto* indptr = device_copy(indptr_host);

  __nv_bfloat16* gate_ref = nullptr;
  __nv_bfloat16* up_ref = nullptr;
  __nv_bfloat16* gate_w13 = nullptr;
  __nv_bfloat16* up_w13 = nullptr;
  CUDA_CHECK(cudaMalloc(&gate_ref, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&up_ref, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gate_w13, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&up_w13, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemsetAsync(gate_ref, 0x11, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(up_ref, 0x22, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(gate_w13, 0x33, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(up_w13, 0x44, out_elems * sizeof(__nv_bfloat16), stream));

  auto run_baseline = [&]() {
    TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
        act, reinterpret_cast<const void* const*>(w1_ptrs), gate_ref, act_scale,
        reinterpret_cast<const void* const*>(s1_ptrs), indptr, args.rows, args.experts, stream));
    TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
        act, reinterpret_cast<const void* const*>(w3_ptrs), up_ref, act_scale,
        reinterpret_cast<const void* const*>(s3_ptrs), indptr, args.rows, args.experts, stream));
  };
  auto run_w13 = [&]() {
    TK_CHECK(deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
        act, reinterpret_cast<const void* const*>(w1_ptrs),
        reinterpret_cast<const void* const*>(w3_ptrs), gate_w13, up_w13, act_scale,
        reinterpret_cast<const void* const*>(s1_ptrs),
        reinterpret_cast<const void* const*>(s3_ptrs), indptr, args.rows, args.experts, stream));
  };

  run_baseline();
  run_w13();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint16_t> gate_ref_host(out_elems);
  std::vector<uint16_t> up_ref_host(out_elems);
  std::vector<uint16_t> gate_w13_host(out_elems);
  std::vector<uint16_t> up_w13_host(out_elems);
  CUDA_CHECK(cudaMemcpy(gate_ref_host.data(), gate_ref, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(up_ref_host.data(), up_ref, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gate_w13_host.data(), gate_w13, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(up_w13_host.data(), up_w13, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  int gate_mismatches = compare_u16(gate_ref_host, gate_w13_host, "gate");
  int up_mismatches = compare_u16(up_ref_host, up_w13_host, "up");
  if (gate_mismatches || up_mismatches) {
    std::fprintf(stderr, "FUZZ FAIL gate_mismatches=%d up_mismatches=%d\n",
                 gate_mismatches, up_mismatches);
    return 1;
  }

  for (int i = 0; i < args.warmup; ++i) {
    run_baseline();
    run_w13();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float baseline_ms = time_ms(stream, args.iters, run_baseline);
  float w13_ms = time_ms(stream, args.iters, run_w13);

  std::printf("W13 grouped FP4 fuzz: PASS rows=%d experts=%d seed=%d\n",
              args.rows, args.experts, args.seed);
  std::printf("expert_indptr:");
  for (int value : indptr_host) std::printf(" %d", value);
  std::printf("\n");
  std::printf("baseline_two_gemm_ms=%.6f\n", baseline_ms);
  std::printf("w13_one_gemm_ms=%.6f\n", w13_ms);
  std::printf("speedup=%.3fx\n", baseline_ms / w13_ms);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
