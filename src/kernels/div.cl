inline float inline_div(const float a, const float b) { return a / b; }

OPENCLDEV_KERNEL_FW_X_CONST(div_const_r, px[i] / k)
OPENCLDEV_KERNEL_FW_X_CONST(div_const_l, k / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(div_const_r, pgy[i] / k)
OPENCLDEV_KERNEL_BW_X_CONST(div_const_l, -py[i] * pgy[i] / px[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(div_scalar_r, inline_div)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(div_scalar_l, inline_div)
OPENCLDEV_KERNEL_FW_AB(div, inline_div)

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void div_bw_a_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    atomic_add_float(pga + i + mba * shift, k);
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void div_bw_b_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    atomic_add_float(pgb + b_ofs, -k * py[y_ofs]);
  }
}
