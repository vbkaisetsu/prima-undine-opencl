kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void batch_concat_fw_kernel(
    const global float *px, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[i + shift] = px[i];
}
