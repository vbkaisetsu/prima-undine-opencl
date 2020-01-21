kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void initialize_xorshift_kernel(
    const global unsigned *seeds, global uint4 *state) {
  const unsigned i = get_global_id(0);
  state[i].x = 123456789;
  state[i].y = 362436069;
  state[i].z = 521288629;
  state[i].w = seeds[i];
}

inline void update_xorshift(global uint4 *state) {
  const unsigned lid = get_local_id(0);
  const unsigned x = state[lid].x;
  const unsigned w = state[lid].w;
  const unsigned t = x ^ (x << 11);
  state[lid].xyz = state[lid].yzw;
  state[lid].w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void xorshift_bernoulli_kernel(
    global uint4 *state, float p, unsigned size, global float *py) {
  const unsigned gid = get_global_id(0);
  const unsigned lid = get_local_id(0);
  update_xorshift(state);
  if (gid < size) {
    py[gid] = (float) ((float) state[lid].w < p * 4294967296);
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void xorshift_uniform_kernel(
    global uint4 *state, float lower, float upper, unsigned size, global float *py) {
  const unsigned gid = get_global_id(0);
  const unsigned lid = get_local_id(0);
  update_xorshift(state);
  if (gid < size) {
    const float p = (float) state[lid].w / (float) 4294967296;
    py[gid] = p * (upper - lower) + lower;
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void xorshift_normal_kernel(
    global uint4 *state, float mean, float sd, unsigned size, global float *py) {
  const unsigned gid = get_global_id(0);
  const unsigned lid = get_local_id(0);
  update_xorshift(state);
  const unsigned lid_f = (lid / 2) * 2;
  if (gid < size) {
    const float p = (float) state[lid_f + 0].w / 4294967296;
    const float q = (float) state[lid_f + 1].w / 4294967296;
    const float a = p > 0.0 ? sqrt(-2.0 * log(p)) : 0.0;
    const float b = gid % 2 == 0 ? cos(2.0 * M_PI * q) : sin(2.0 * M_PI * q);
    py[gid] = a * b * sd + mean;
  }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void xorshift_log_normal_kernel(
    global uint4 *state, float mean, float sd, unsigned size, global float *py) {
  const unsigned gid = get_global_id(0);
  const unsigned lid = get_local_id(0);
  update_xorshift(state);
  const unsigned lid_f = (lid / 2) * 2;
  if (gid < size) {
    const float p = (float) state[lid_f + 0].w / 4294967296;
    const float q = (float) state[lid_f + 1].w / 4294967296;
    const float a = p > 0.0 ? sqrt(-2.0 * log(p)) : 0.0;
    const float b = gid % 2 == 0 ? cos(2.0 * M_PI * q) : sin(2.0 * M_PI * q);
    py[gid] = exp(a * b * sd + mean);
  }
}
