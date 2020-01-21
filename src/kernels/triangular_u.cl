kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void triangular_u_fw_kernel(
    const global float *px, unsigned k, unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * size * size;
  if (i < size && j < size) py[ofs + i + j * size] = px[ofs + i + j * size] * (j >= i + k);
}

kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void triangular_u_bw_kernel(
    const global float *py, unsigned k, unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * size * size;
  if (j < size && j >= i + k) px[ofs + i + j * size] += py[ofs + i + j * size];
}
