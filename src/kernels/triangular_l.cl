kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void triangular_l_fw_kernel(
    const global float *px, unsigned k, unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * size * size;
  if (i < size && j < size) py[ofs + i + j * size] = px[ofs + i + j * size] * (i >= j + k);
}

kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void triangular_l_bw_kernel(
    const global float *py, unsigned k, unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * size * size;
  if (i < size && i >= j + k) px[ofs + i + j * size] += py[ofs + i + j * size];
}
