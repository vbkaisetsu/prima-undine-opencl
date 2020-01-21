OPENCLDEV_KERNEL_FW_X_CONST(powf_const_r, pow(px[i], k))
OPENCLDEV_KERNEL_FW_X_CONST(powf_const_l, pow(k, px[i]))
OPENCLDEV_KERNEL_BW_X_CONST(powf_const_r, pgy[i] * k * py[i] / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(powf_const_l, pgy[i] * log(k) * py[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(powf_scalar_r, pow)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(powf_scalar_l, pow)
OPENCLDEV_KERNEL_FW_AB(powf, pow)

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void powf_bw_a_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    atomic_add_float(pga + a_ofs, k * pb[b_ofs] / pa[a_ofs]);
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void powf_bw_b_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    atomic_add_float(pgb + b_ofs, k * log(pa[a_ofs]));
  }
}
