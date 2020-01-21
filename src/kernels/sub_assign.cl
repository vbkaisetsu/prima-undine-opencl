kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sub_assign_kernel(
    const global float *px, const unsigned size,
    const unsigned mbx, const unsigned mby, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, -px[i + mbx * shift]);
}
