// Inkling 2 attention on Xe2: prep, bias table, DPAS attention partials and the ordered merge (attention_common.h).
#include "attention_common.h"

// Prep a row and head: q, rel = clip(x * scale) (* tau on global layers), k = clip(k), v = rms(v); k, v go to the ring.
KERNEL(16) void ink2_prep(global const ushort *q_raw, global const ushort *k_raw, global const ushort *v_raw, global const ushort *rel_raw,
                          global const ushort *q_scale, global const ushort *rel_scale, global const int *rpos, global const int *rseg,
                          global const Seg *segs, global ushort *q_out, global ushort *rel_out, global ushort *kc, global ushort *vc,
                          int ring, int global_layer, float qk_clip, float rel_clip, float floor_pos, float alpha, float eps) {
  const int row = get_group_id(0), y = get_group_id(1), lane = get_sub_group_local_id();
  const int pos = rpos[row];
  const float tau = global_layer ? 1.0f + alpha * log(fmax((float)(pos + 1) / floor_pos, 1.0f)) : 1.0f;
  if (y < HEADS) {
    global const ushort *src = q_raw + ((long)row * HEADS + y) * HD + lane * 8;
    float8 x = ld8(src) * ld8(q_scale + y * HD + lane * 8);
    x *= clipf(sub_group_reduce_add(sq8(x)), qk_clip);
    float8 t = x;
    if (global_layer) t = as_float8(convert_uint8(st8(x)) << 16) * tau;
    vstore8(st8(t), 0, q_out + ((long)row * HEADS + y) * HD + lane * 8);
    float r = bf(rel_raw[((long)row * HEADS + y) * REL + lane]) * bf(rel_scale[y * REL + lane]);
    r *= clipf(sub_group_reduce_add(r * r), rel_clip);
    if (global_layer) r = bf(to_bf(r)) * tau;
    rel_out[((long)row * HEADS + y) * REL + lane] = to_bf(r);
    return;
  }
  const bool key = y < HEADS + KVH;
  const int h = key ? y - HEADS : y - HEADS - KVH;
  float8 x = ld8((key ? k_raw : v_raw) + ((long)row * KVH + h) * HD + lane * 8);
  if (key) {
    x *= clipf(sub_group_reduce_add(sq8(x)), qk_clip);
    // A non-finite value row makes its key NaN and is stored as 0, so a masked key's 0 x NaN never reaches other rows.
    const float8 vr = ld8(v_raw + ((long)row * KVH + h) * HD + lane * 8);
    if (sub_group_any(any(!isfinite(vr)))) x = (float8)(NAN);
  } else {
    x *= rsqrt(sub_group_reduce_add(sq8(x)) / HD + eps);
    x = as_float8(as_uint8(x) & as_uint8(isfinite(x)));
  }
  const Seg s = segs[rseg[row]];
  vstore8(st8(x), 0, (key ? kc : vc) + s.off + ((long)h * ring + pos % ring) * HD + lane * 8);
}

// tab[row, head, d] = bf16(rel[row, head, :] . proj[:, d]) for d < 1024, fp32 sums in tap order (rel_logits_proj).
KERNEL(256) void ink2_rel_table(global const ushort *rel, global const ushort *proj, global ushort *tab) {
  const long rh = get_group_id(0);
  float r[REL];
  for (int j = 0; j < REL; j++) r[j] = bf(rel[rh * REL + j]);
  for (int d = get_local_id(0); d < EXTENT; d += 256) {
    float acc = 0.0f;
    for (int j = 0; j < REL; j++) acc += r[j] * bf(proj[j * EXTENT + d]);
    tab[rh * EXTENT + d] = to_bf(acc);
  }
}

// DPAS attention: chunk partials (ink2_merge folds them) for decode and verify, 32 / GROUP rows a work-group.
#define NAME ink2_attn
#define NT 2
#define KU 1
#include "attention_body.h"
#undef NAME
#undef NT
#undef KU
// The prompt's attention, the arithmetic of ink2_attn: 64 / GROUP rows a work-group, q in local memory.
#define NAME ink2_attn_pq
#define NT 4
#define KU 1
#define QSLM 1
#include "attention_body.h"
#undef NAME
#undef NT
#undef KU
#undef QSLM

// Per (row, head): its chunks folded in order, then the sink (one logit, no value), then the sigmoid gate.
KERNEL(128) void ink2_merge(global const float *PO, global const float2 *PML, global const ushort *sink, global const ushort *gate,
                            global const int *rpos, global ushort *out, int window, int chunk, int c0, int slots) {
  const long rh = get_group_id(0);
  const int row = (int)(rh / HEADS), d = get_local_id(0);
  const int pos = rpos[row];
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  float mm = -INFINITY, ll = 0.0f, o = 0.0f;
  for (int c = lo / chunk; c <= pos / chunk; c++) {
    const long at = rh * slots + (c - (window > 0 ? lo / chunk : c0));
    const float2 ml = PML[at];
    const float mn = fmax(mm, ml.x), a = exp2(mm - mn), b = exp2(ml.x - mn);
    ll = ll * a + ml.y * b;
    o = o * a + PO[at * HD + d] * b;
    mm = mn;
  }
  const float sk = bf(sink[rh]) * LOG2E, mf = fmax(mm, sk), a = exp2(mm - mf);
  const float y = o * a / (ll * a + exp2(sk - mf));
  out[rh * HD + d] = to_bf(y * (1.0f / (1.0f + exp(-bf(gate[rh * HD + d])))));
}
