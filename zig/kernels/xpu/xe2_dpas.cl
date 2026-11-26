// One DPAS call per 8 x 16 output tile on a sub-group of 16: the validator for core/contract/xe2_dpas.zig.
#define SG __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))

SG kernel void xe2_dpas_bf16(global const ushort *A, global const ushort *B, global const float *C, global float *O) {
  size_t t = get_group_id(0);
  uint l = get_sub_group_local_id();
  short8 a;
  ushort *ap = (ushort *)&a;
  for (int m = 0; m < 8; m++) ap[m] = A[(t * 8 + m) * 16 + l];
  int8 b;
  int *bp = (int *)&b;
  for (int j = 0; j < 8; j++) bp[j] = (int)((uint)B[(t * 16 + 2 * j) * 16 + l] | ((uint)B[(t * 16 + 2 * j + 1) * 16 + l] << 16));
  float8 acc;
  float *cp = (float *)&acc;
  for (int m = 0; m < 8; m++) cp[m] = C[(t * 8 + m) * 16 + l];
  float8 r = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
  float *rp = (float *)&r;
  for (int m = 0; m < 8; m++) O[(t * 8 + m) * 16 + l] = rp[m];
}

SG kernel void xe2_dpas_f16(global const ushort *A, global const ushort *B, global const float *C, global float *O) {
  size_t t = get_group_id(0);
  uint l = get_sub_group_local_id();
  short8 a;
  ushort *ap = (ushort *)&a;
  for (int m = 0; m < 8; m++) ap[m] = A[(t * 8 + m) * 16 + l];
  int8 b;
  int *bp = (int *)&b;
  for (int j = 0; j < 8; j++) bp[j] = (int)((uint)B[(t * 16 + 2 * j) * 16 + l] | ((uint)B[(t * 16 + 2 * j + 1) * 16 + l] << 16));
  float8 acc;
  float *cp = (float *)&acc;
  for (int m = 0; m < 8; m++) cp[m] = C[(t * 8 + m) * 16 + l];
  float8 r = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
  float *rp = (float *)&r;
  for (int m = 0; m < 8; m++) O[(t * 8 + m) * 16 + l] = rp[m];
}

// tf32: A is [tile][8 rows][8 k], B [tile][8 k][16 columns]; element i of lane l is A row 2i + (l >= 8), col l % 8.
SG kernel void xe2_dpas_tf32(global const float *A, global const float *B, global const float *C, global float *O) {
  size_t t = get_group_id(0);
  uint l = get_sub_group_local_id();
  float4 a;
  float *ap = (float *)&a;
  for (int i = 0; i < 4; i++) ap[i] = A[(t * 8 + 2 * i + (l >= 8)) * 8 + (l % 8)];
  float8 b;
  float *bp = (float *)&b;
  for (int k = 0; k < 8; k++) bp[k] = B[(t * 8 + k) * 16 + l];
  float8 acc;
  float *cp = (float *)&acc;
  for (int m = 0; m < 8; m++) cp[m] = C[(t * 8 + m) * 16 + l];
  float8 r = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, acc);
  float *rp = (float *)&r;
  for (int m = 0; m < 8; m++) O[(t * 8 + m) * 16 + l] = rp[m];
}
