// Inkling 2 attention on Xe2, shared by its images; geometry from -DHEADS= -DKVH= -DEXTENT= (Mini's by default).
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel
#define HD 128  // a sub-group of 16 lanes holds a head vector as 8 values each, the DPAS chain is HD / 16 steps
#define REL 16  // one lane a rel logit
#ifndef HEADS
#define HEADS 24
#endif
#ifndef KVH
#define KVH 8
#endif
#ifndef EXTENT
#define EXTENT 1024  // positions the relative bias spans
#endif
#if HEADS % KVH != 0
#error "HEADS must be a multiple of KVH"
#endif
#define GROUP (HEADS / KVH)
#define LOG2E 1.4426950408889634f

// One stream's rows in a call: rows row0.. sit at positions p0.., its caches `off` elements in.
typedef struct {
  int row0, n, p0, pad;
  long off;
} Seg;

static inline float bf(ushort v) { return as_float((uint)v << 16); }
static inline ushort to_bf(float f) {
  uint u = as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (ushort)(u >> 16);
}
static inline float8 ld8(global const ushort *p) { return as_float8(convert_uint8(vload8(0, p)) << 16); }
static inline ushort8 st8(float8 x) {
  return (ushort8)(to_bf(x.s0), to_bf(x.s1), to_bf(x.s2), to_bf(x.s3), to_bf(x.s4), to_bf(x.s5), to_bf(x.s6), to_bf(x.s7));
}
static inline float sq8(float8 x) {
  return ((x.s0 * x.s0 + x.s1 * x.s1) + (x.s2 * x.s2 + x.s3 * x.s3)) + ((x.s4 * x.s4 + x.s5 * x.s5) + (x.s6 * x.s6 + x.s7 * x.s7));
}
// L2 clip factor of a head vector (fmin drops a NaN norm, as fminf): scaled down to `thr` when longer.
static inline float clipf(float ss, float thr) { return fmin(thr / (sqrt(ss) + 1e-6f), 1.0f); }
