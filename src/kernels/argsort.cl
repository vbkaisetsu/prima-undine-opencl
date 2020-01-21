kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void init_argsort_kernel(
    const unsigned size, global unsigned *py) {
  const unsigned gid = get_global_id(0);
  if (gid < size) py[gid] = gid;
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void argsort_kernel(
    const global float *px, const unsigned block_size, const unsigned dist,
    const unsigned skip, const unsigned len, const unsigned idx_len, const unsigned size,
    const unsigned idx_size, global unsigned *py) {
  const unsigned gid = get_global_id(0);
  const unsigned i = gid % skip;
  const unsigned j = gid / skip;
  const unsigned j1 = j * 2 - j % dist;
  const unsigned j2 = j1 + dist;
  if (j2 * skip < idx_size) {
    const unsigned batch = j1 / idx_len;
    const unsigned block_idx = (j1 - batch * idx_len) / block_size;
    const unsigned inner_trans = batch * len * skip;
    const unsigned outer_trans = size + (batch * (idx_len - len) - len) * skip;
    const unsigned d1 = j1 % idx_len;
    const unsigned d2 = j2 % idx_len;
    const unsigned p = i + (d1 < len ? inner_trans : outer_trans) + d1 * skip;
    const unsigned q = i + (d2 < len ? inner_trans : outer_trans) + d2 * skip;
    const unsigned y1 = py[p];
    const unsigned y2 = py[q];
    if (block_idx % 2 == 0 ? (y1 >= size || (y2 < size && px[y1] > px[y2])) : (y2 >= size || (y1 < size && px[y2] > px[y1]))) {
      py[p] = y2;
      py[q] = y1;
    }
  }
}
