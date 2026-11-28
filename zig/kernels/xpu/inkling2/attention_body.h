#ifndef PV_BLOCK
#define PV_BLOCK(i) (i)  // the order of a tile's 16-key blocks in P.V: a contract control plants another
#endif
#if QSLM
#define QV(t, ds) Qs[((t) * (HD / 16) + (ds)) * 16 + lane]
#else
#define QV(t, ds) qb[t][ds]
#endif
// DPAS attention body, once a variant: NAME, NT (16-pair tiles), KU (8-key sets a sub-group), QSLM (q in local memory).
#define NP (16 * NT)
#define TOKS (NP / GROUP)
#define NPG (NP / 8)
#define TILE (KU * 64)

// Work-group (TOKS rows, key head, chunk): S = K Q^T on DPAS, base-2 online softmax, P in bf16 via SLM, O = P V.
KERNEL(128) void NAME(global const ushort *Q,
                      global const ushort *KC, global const ushort *VC,
                      global const ushort *TAB,
                      global const Seg *segs, global const int2 *tiles, global float *PO, global float2 *PML,
                      int ring, int window, int chunk, int c0, int slots, float scale
#if QSLM
                      , int rbase  // the first row of this launch: its partials start at the buffer's start
#endif
                      ) {
  local ushort Pl[NP * TILE];
#if QSLM
  local int8 Qs[NT * (HD / 16) * 16];  // q by (slot tile, DPAS step, lane), read where used
#endif
  local float MX[8 * NP], SM[8 * NP];
  const int2 tile = tiles[get_group_id(0)];
  const Seg s = segs[tile.x];
  const int kvh = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nr = min(TOKS, s.n - tile.y);
  const int pos_first = s.p0 + tile.y, pos_last = pos_first + nr - 1;
  const int tile_lo = window > 0 ? max(0, pos_first - window + 1) : 0;
  const int c = (window > 0 ? tile_lo / chunk : c0) + (int)get_group_id(2);
  const int b_lo = c * chunk, b_hi = b_lo + chunk - 1;
  if (b_lo > pos_last || b_hi < tile_lo) return;
#if !QSLM
  int8 qb[NT][HD / 16];
#endif
  bool act[NT], live[NT];
  int qpos[NT], lo[NT];
  long prow[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) {
    const int slot = t * 16 + lane, tok = slot / GROUP;
    act[t] = slot < TOKS * GROUP && tok < nr;
    live[t] = t * 16 < nr * GROUP;
    qpos[t] = pos_first + (act[t] ? tok : 0);
    lo[t] = window > 0 ? max(0, qpos[t] - window + 1) : 0;
    prow[t] = (long)(s.row0 + tile.y + (act[t] ? tok : 0)) * HEADS + kvh * GROUP + (act[t] ? slot % GROUP : 0);
    global const uint *qp = (global const uint *)(Q + prow[t] * HD);
#if QSLM
    Qs[(t * (HD / 16) + sg) * 16 + lane] = act[t] ? as_int8(vload8(0, qp + sg * 8)) : (int8)(0);
#else
    for (int ds = 0; ds < HD / 16; ds++) qb[t][ds] = act[t] ? as_int8(vload8(0, qp + ds * 8)) : (int8)(0);
#endif
  }
#if QSLM
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  global const void *kbase = (global const void *)(KC + s.off + (long)kvh * ring * HD);
  global const void *vbase = (global const void *)(VC + s.off + (long)kvh * ring * HD);
  const int wbytes = HD * 2, pitch = HD * 2;
  float m_run[NT], l_run[NT];
  float8 o[NT][2];
#pragma unroll
  for (int t = 0; t < NT; t++) {
    m_run[t] = -INFINITY;
    l_run[t] = 0.0f;
    o[t][0] = o[t][1] = (float8)(0.0f);
  }
  const int lo_last = window > 0 ? max(0, pos_last - window + 1) : 0;
  const int k_first = max(b_lo, tile_lo) & ~(TILE - 1), k_last = min(b_hi, pos_last);
  for (int kt = k_first; kt <= k_last; kt += TILE) {
    const int kb = kt + sg * 8 * KU;
    const bool nomask = kb >= lo_last && kb + 8 * KU - 1 <= pos_first;  // every key live for every row
    const bool nobias = nomask && window == 0 && kb + 8 * KU - 1 + EXTENT <= pos_first;  // and past the bias
    float8 sc[KU][NT];
#pragma unroll
    for (int u = 0; u < KU; u++)
#pragma unroll
      for (int t = 0; t < NT; t++) sc[u][t] = (float8)(0.0f);
#pragma unroll
    for (int ds = 0; ds < HD / 16; ds++) {
#pragma unroll
      for (int u = 0; u < KU; u++) {
        ushort8 ka;
        intel_sub_group_2d_block_read_16b_8r16x1c((global void *)kbase, wbytes, ring, pitch, (int2)(ds * 16, (kb + 8 * u) % ring), (private ushort *)&ka);
#pragma unroll
        for (int t = 0; t < NT; t++)
          if (live[t]) sc[u][t] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(ka), QV(t, ds), sc[u][t]);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (!live[t]) continue;
      float mx = -INFINITY;
#pragma unroll
      for (int u = 0; u < KU; u++) {
        if (nobias) {
#pragma unroll
          for (int r = 0; r < 8; r++) {
            sc[u][t][r] = (sc[u][t][r] * scale) * LOG2E;
            mx = fmax(mx, sc[u][t][r]);
          }
        } else {
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const int kk = kb + 8 * u + r, d = qpos[t] - kk;
            const bool in = nomask || (act[t] && kk >= lo[t] && kk <= qpos[t]);
            const float bias = in && d < EXTENT ? bf(TAB[prow[t] * EXTENT + d]) : 0.0f;
            sc[u][t][r] = in ? (sc[u][t][r] * scale + bias) * LOG2E : -INFINITY;
            mx = fmax(mx, sc[u][t][r]);
          }
        }
      }
      MX[sg * NP + t * 16 + lane] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float alpha[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (!live[t]) continue;
      float mt = -INFINITY;
      for (int i = 0; i < 8; i++) mt = fmax(mt, MX[i * NP + t * 16 + lane]);
      const float mn = fmax(m_run[t], mt), mu = mn == -INFINITY ? 0.0f : mn;
      alpha[t] = exp2(m_run[t] - mu);
      m_run[t] = mn;
      float sum = 0.0f;
      const int pg = (t * 16 + lane) >> 3, pi = lane & 7;
#pragma unroll
      for (int u = 0; u < KU; u++)
#pragma unroll
        for (int r = 0; r < 8; r++) {
          const ushort p = intel_convert_bfloat16_as_ushort(exp2(sc[u][t][r] - mu));
          const int kk = sg * 8 * KU + u * 8 + r;  // a sub-group's keys fill whole 16-key blocks of the tile
          Pl[(((kk >> 4) * NPG + pg) * 16 + (kk & 15)) * 8 + pi] = p;
          sum += bf(p);
        }
      SM[sg * NP + t * 16 + lane] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (!live[t]) continue;
      float lt = 0.0f;
      for (int i = 0; i < 8; i++) lt += SM[i * NP + t * 16 + lane];
      l_run[t] = l_run[t] * alpha[t] + lt;
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int i = 0; i < 8; i++) o[t][h][i] *= sub_group_broadcast(alpha[t], 8 * h + i);
    }
    for (int ci = 0; ci < TILE / 16; ci++) {
      const int cb = PV_BLOCK(ci);
      if (kt + cb * 16 > pos_last || kt + cb * 16 + 15 < tile_lo) continue;
      uint8 vw;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)vbase, wbytes, ring, pitch, (int2)(sg * 16, (kt + cb * 16) % ring), (private uint *)&vw);
#pragma unroll
      for (int t = 0; t < NT; t++) {
        if (!live[t]) continue;
        const short8 a0 = as_short8(vload8(0, Pl + ((cb * NPG + 2 * t) * 16 + lane) * 8));
        const short8 a1 = as_short8(vload8(0, Pl + ((cb * NPG + 2 * t + 1) * 16 + lane) * 8));
        o[t][0] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), o[t][0]);
        o[t][1] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), o[t][1]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // slot p = 8h + i of tile t's accumulators is lane p's (row, head) of that tile; its dims are sg * 16 + lane
#pragma unroll
  for (int t = 0; t < NT; t++) {
    if (!live[t]) continue;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int p = 8 * h + i, slot = t * 16 + p, tk = slot / GROUP;
        if (slot >= TOKS * GROUP || tk >= nr) continue;
        const int pq = pos_first + tk, plo = window > 0 ? max(0, pq - window + 1) : 0;
        if (c < plo / chunk || c > pq / chunk || b_lo > pq) continue;  // the merge folds a row's chunks up to its own
        const long rh = (long)(s.row0 + tile.y + tk) * HEADS + kvh * GROUP + slot % GROUP;
        const float m = sub_group_broadcast(m_run[t], p), l = sub_group_broadcast(l_run[t], p);
#if QSLM
        const long at = (rh - (long)rbase * HEADS) * slots + (c - (window > 0 ? plo / chunk : c0));
#else
        const long at = rh * slots + (c - (window > 0 ? plo / chunk : c0));
#endif
        if (lane == 0) PML[at] = (float2)(m, l);
        PO[at * HD + sg * 16 + lane] = o[t][h][i];
      }
  }
}
#undef NP
#undef TOKS
#undef NPG
#undef TILE
#undef QV
