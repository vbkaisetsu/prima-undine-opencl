kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_assign_const_kernel(
    const float k, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] *= k;
}
