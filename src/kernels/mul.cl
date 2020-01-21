inline float inline_mul(const float a, const float b) { return a * b; }

OPENCLDEV_KERNEL_FW_X_CONST(mul_const, px[i] * k)
OPENCLDEV_KERNEL_BW_X_CONST(mul_const, k * pgy[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(mul_scalar, inline_mul)
OPENCLDEV_KERNEL_FW_AB(mul, inline_mul)

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_bw_a_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    atomic_add_float(pga + a_ofs, gy * pb[b_ofs]);
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_bw_b_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    atomic_add_float(pgb + b_ofs, gy * pa[a_ofs]);
  }
}
