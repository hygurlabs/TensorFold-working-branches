// Planted faults of the Inkling 2 attention contract: the served body with one rounding or one order changed.
#include "attention_common.h"

// P.V over a tile's 16-key blocks last to first: the same products, summed in another order.
#define PV_BLOCK(i) (TILE / 16 - 1 - (i))
#define NAME tf_ctl_ink2_attn_kblock
#define NT 2
#define KU 1
#include "attention_body.h"
#undef NAME
#undef NT
#undef KU
#undef PV_BLOCK

// P cut to bf16 by dropping its low half instead of rounding to nearest even.
#define intel_convert_bfloat16_as_ushort(x) ((ushort)(as_uint(x) >> 16))
#define NAME tf_ctl_ink2_attn_ptrunc
#define NT 2
#define KU 1
#include "attention_body.h"
#undef NAME
#undef NT
#undef KU
#undef intel_convert_bfloat16_as_ushort
