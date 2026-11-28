//! Inkling 2's attention on the Intel GPU (attention.cl): the launch plan's geometry, over a device (see Attention).
const std = @import("std");
const config = @import("config.zig");

pub const head_dim = config.head_dim;
pub const d_rel = config.d_rel;
/// Key chunks of the baseline (INT8 excluded): a row folds its chunks in order, so its bits never depend on its call.
pub const global_chunk = 256;
pub const sliding_chunk = 128;
/// Keys a K/V tile holds (attention_body.h TILE); a ring holds whole tiles.
pub const tile_keys = 64;
/// (row, head) pairs a work-group holds: 2 tiles of 16 for ink2_attn, 4 for ink2_attn_pq.
pub const decode_pairs = 32;
pub const prompt_pairs = 64;
/// Most rows a call takes the decode kernel for (the verify windows); more rows run the prompt kernel in windows.
pub const decode_rows = 32;
/// Most rows of one prompt window: the partials of a window are all the buffer holds at once.
pub const window_rows = 512;
/// The kernel's softmax scale, 1 / sqrt(head_dim) in f32.
pub const scale: f32 = 0.08838834764831845;

pub const Ptr = ?*anyopaque;

/// `p` moved `bytes` on.
pub fn at(p: Ptr, bytes: usize) Ptr {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// One stream's rows in a call (attention.cl Seg): rows row0.. at positions p0.., its caches `off` elements in.
pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
/// A work-group's rows: segment `seg`, rows `row`.. of it (attention_body.h's int2 tile).
pub const Tile = extern struct { seg: i32, row: i32 };

/// What the image is built for: -DHEADS, -DKVH and -DEXTENT of attention.cl (the head and rel widths are fixed).
pub const Geometry = struct {
    heads: u32,
    kv_heads: u32,
    extent: u32,

    /// The geometry of a config's attention; refuses one the single image cannot serve.
    pub fn fromText(t: config.Text) error{Inkling2Geometry}!Geometry {
        const l = t.local;
        if (l.q != t.global.q or l.kv != t.global.kv or l.dim != head_dim or t.global.dim != head_dim) return error.Inkling2Geometry;
        if (t.d_rel != d_rel or t.window != t.rel_extent or l.kv == 0 or l.q % l.kv != 0) return error.Inkling2Geometry;
        return .{ .heads = l.q, .kv_heads = l.kv, .extent = t.window };
    }

    /// Query heads a key head.
    pub fn group(g: Geometry) u32 {
        return g.heads / g.kv_heads;
    }

    /// Rows a work-group holds: decode (ink2_attn) or prompt (ink2_attn_pq).
    pub fn tileRows(g: Geometry, prompt: bool) u32 {
        const pairs: u32 = if (prompt) prompt_pairs else decode_pairs;
        return pairs / g.group();
    }

    /// The compiler options that make the image for this geometry.
    pub fn flags(g: Geometry, buf: []u8) error{NoSpaceLeft}![]const u8 {
        return std.fmt.bufPrint(buf, "-DHEADS={d} -DKVH={d} -DEXTENT={d}", .{ g.heads, g.kv_heads, g.extent });
    }

    /// Bytes of the bias table of `rows` rows (bf16) and of their partials of `stride` chunks.
    pub fn tableBytes(g: Geometry, rows: usize) usize {
        return rows * g.heads * g.extent * 2;
    }

    pub fn partialBytes(g: Geometry, rows: usize, stride: u32) usize {
        return rows * g.heads * stride * (head_dim * 4 + 8);
    }
};

/// A layer's cache shape: `window` 0 is a global layer; `ring` slots a key head (a multiple of 64).
pub const Layer = struct {
    window: u32,
    ring: u32,
    chunk: u32,

    pub fn sliding(window: u32, rows: u32) Layer {
        return .{ .window = window, .ring = ringFor(window, rows), .chunk = sliding_chunk };
    }

    pub fn global(capacity: u32) Layer {
        return .{ .window = 0, .ring = @intCast(std.mem.alignForward(usize, capacity, tile_keys)), .chunk = global_chunk };
    }

    /// Bytes of one stream's K (or V) cache with `kv` key heads: [kv][ring][128] bf16.
    pub fn cacheBytes(l: Layer, kv: usize) usize {
        return @as(usize, l.ring) * kv * head_dim * 2;
    }
};

/// Sliding ring slots: the window plus a call's rows less one, so a chunk never overwrites keys it still reads.
pub fn ringFor(window: u32, rows: u32) u32 {
    return @intCast(std.mem.alignForward(usize, @as(usize, window) + rows - 1, tile_keys));
}

/// The ring the tiles assume: whole 64-key tiles (a tile read never wraps), and a window plus the rows less one.
pub fn ringFits(l: Layer, rows: u32) error{ Inkling2RingTiles, Inkling2RingShort }!void {
    if (l.ring == 0 or l.ring % tile_keys != 0) return error.Inkling2RingTiles;
    if (@as(u64, l.window) + rows > @as(u64, l.ring) + 1) return error.Inkling2RingShort;
}

/// Partials a (row, head) needs for positions below `context`: a sliding row's window, every global chunk.
pub fn slotsFor(l: Layer, context: usize) u32 {
    if (l.window > 0) return (l.window - 1) / l.chunk + 2;
    return @intCast((context - 1) / l.chunk + 1);
}

/// Chunk blocks (grid z) of a tile of `tile_rows` rows: sliding spans its rows and window, a global one the stride.
pub fn chunkBlocks(l: Layer, tile_rows: u32, stride: u32) u32 {
    if (l.window == 0) return stride;
    return (tile_rows - 1 + l.window) / l.chunk + 2;
}

/// The partial slot chunk `c` of a row at `pos` is stored in; null outside the row's keys (stores nothing).
pub fn partialSlot(l: Layer, c: usize, pos: usize) ?usize {
    const lo = if (l.window > 0) (pos + 1) -| l.window else 0;
    if (c < lo / l.chunk or c > pos / l.chunk) return null;
    return c - lo / l.chunk;
}

/// The work-groups' tiles of a call's segments, `tile_rows` rows each; returns how many were written.
pub fn tiles(segs: []const Seg, tile_rows: u32, out: []Tile) error{TooManyTiles}!usize {
    var n: usize = 0;
    for (segs, 0..) |s, i| {
        var row: usize = 0;
        while (row < @as(usize, @intCast(s.n))) : (row += tile_rows) {
            if (n == out.len) return error.TooManyTiles;
            out[n] = .{ .seg = @intCast(i), .row = @intCast(row) };
            n += 1;
        }
    }
    return n;
}

/// Tiles of `rows` rows of one segment.
pub fn tileCount(rows: usize, tile_rows: u32) usize {
    return std.math.divCeil(usize, rows, tile_rows) catch unreachable;
}

/// A call's kernel: the decode kernel up to `decode_rows`, else the prompt kernel, whose tiles are `prompt_pairs` wide.
pub fn isPrompt(rows: u32) bool {
    return rows > decode_rows;
}

/// Prompt tiles a window holds: as many as the partials hold at `stride` chunks, within `window_rows` rows; 0 if none.
pub fn windowTiles(g: Geometry, entries: usize, stride: u32) usize {
    const t = g.tileRows(true);
    const fit = entries / (@as(usize, g.heads) * stride) / t;
    return @min(fit, window_rows / t);
}

/// The constants ink2_prep takes from config.json (Mini: qk 22.63, rel 8, floor 8,192, alpha 0.1).
pub const Prep = struct {
    qk_clip: f32,
    rel_clip: f32,
    floor: f32,
    alpha: f32,
    eps: f32,

    pub fn fromText(t: config.Text) Prep {
        return .{ .qk_clip = t.qk_clip, .rel_clip = t.rel_clip, .floor = @floatFromInt(t.log_floor), .alpha = t.log_alpha, .eps = t.eps };
    }
};

/// A layer's caches (device addresses): K and V rows.
pub const Cache = struct { k: Ptr, v: Ptr };

/// The call's rows: row arrays are `rows` long, `segs` and `tiles` the tables of `Seg` and `Tile`.
pub const Rows = struct {
    rows: u32,
    pos: Ptr,
    seg_of: Ptr,
    segs: Ptr,
    tiles: Ptr,
    n_tiles: u32,
    last_pos: u32, // the call's latest row position (a global layer's chunks run to it)
    streams: u32, // segments in `segs`, stated by every caller: only one stream's rows take the prompt kernel
};

/// The chunk partials: `stride` chunks a (row, head), in a buffer of `entries` of them (rows x heads x stride at most).
pub const Partials = struct { o: Ptr, ml: Ptr, stride: u32, entries: usize };

/// Dev: Context, Module, Kernel, Stream; load, unload, kernel, deinit, dispatch(kernel, stream, grid, args tuple).
pub fn Attention(comptime Dev: type) type {
    return struct {
        pub const Kernels = struct {
            geometry: Geometry,
            mod: Dev.Module,
            prep: Dev.Kernel,
            rel_table: Dev.Kernel,
            attn: Dev.Kernel,
            attn_pq: Dev.Kernel,
            merge: Dev.Kernel,

            pub fn load(ctx: *const Dev.Context, image: []const u8, geometry: Geometry) !Kernels {
                if (image.len == 0) return error.BuiltWithoutKernels;
                var mod = try Dev.load(ctx, image);
                errdefer Dev.unload(&mod);
                var prep_k = try Dev.kernel(mod, "ink2_prep", .{ 16, 1, 1 });
                errdefer Dev.deinit(&prep_k);
                var rel_k = try Dev.kernel(mod, "ink2_rel_table", .{ 256, 1, 1 });
                errdefer Dev.deinit(&rel_k);
                var attn_k = try Dev.kernel(mod, "ink2_attn", .{ 128, 1, 1 });
                errdefer Dev.deinit(&attn_k);
                var pq_k = try Dev.kernel(mod, "ink2_attn_pq", .{ 128, 1, 1 });
                errdefer Dev.deinit(&pq_k);
                const merge_k = try Dev.kernel(mod, "ink2_merge", .{ 128, 1, 1 });
                return .{ .geometry = geometry, .mod = mod, .prep = prep_k, .rel_table = rel_k, .attn = attn_k, .attn_pq = pq_k, .merge = merge_k };
            }

            pub fn unload(k: *Kernels) void {
                inline for (.{ "prep", "rel_table", "attn", "attn_pq", "merge" }) |name| Dev.deinit(&@field(k, name));
                Dev.unload(&k.mod);
                k.* = undefined;
            }
        };

        /// q, rel (per head) and k, v (per key head, at slot pos % ring of the row's segment) from the projections.
        pub fn prep(k: *const Kernels, s: Dev.Stream, r: Rows, l: Layer, c: Prep, raw: [4]Ptr, scales: [2]Ptr, q: Ptr, rel: Ptr, kv: Cache) !void {
            const g = k.geometry;
            try Dev.dispatch(k.prep, s, .{ r.rows, g.heads + 2 * g.kv_heads, 1 }, .{
                raw[0],  raw[1], raw[2], raw[3], scales[0], scales[1],                             r.pos,     r.seg_of,   r.segs,
                q,       rel,    kv.k,   kv.v,   l.ring,    @as(u32, @intFromBool(l.window == 0)), c.qk_clip, c.rel_clip, c.floor,
                c.alpha, c.eps,
            });
        }

        /// The bias table of each (row, head): its rel vector against the layer's rel_logits_proj [16][extent].
        pub fn relTable(k: *const Kernels, s: Dev.Stream, r: Rows, rel: Ptr, proj: Ptr, tab: Ptr) !void {
            try Dev.dispatch(k.rel_table, s, .{ r.rows * k.geometry.heads, 1, 1 }, .{ rel, proj, tab });
        }

        /// Chunk partials, then the merge: up to `decode_rows` rows run ink2_attn, more run ink2_attn_pq in windows.
        pub fn attend(k: *const Kernels, s: Dev.Stream, r: Rows, l: Layer, p: Partials, q: Ptr, kv: Cache, tab: Ptr, sink: Ptr, gate: Ptr, out: Ptr) !void {
            const g = k.geometry;
            try ringFits(l, r.rows);
            if (r.streams == 0) return error.Inkling2Streams;
            if (slotsFor(l, @as(usize, r.last_pos) + 1) > p.stride) return error.Inkling2Partials;
            const prompt = isPrompt(r.rows);
            if (prompt and r.streams != 1) return error.Inkling2Streams;
            const tile_rows = g.tileRows(prompt);
            if (prompt) {
                if (r.n_tiles != tileCount(r.rows, tile_rows)) return error.Inkling2Tiles;
                if (windowTiles(g, p.entries, p.stride) == 0) return error.Inkling2PartialRoom;
            } else if (p.entries < @as(usize, r.rows) * g.heads * p.stride) return error.Inkling2PartialRoom;
            const z = chunkBlocks(l, tile_rows, p.stride);
            if (!prompt) {
                try Dev.dispatch(k.attn, s, .{ r.n_tiles, g.kv_heads, z }, .{
                    q, kv.k, kv.v, tab, r.segs, r.tiles, p.o, p.ml, l.ring, l.window, l.chunk, @as(u32, 0), p.stride, scale,
                });
                return merge(k, s, g.heads, l, p, r.rows, r.pos, sink, gate, out);
            }
            const per = windowTiles(g, p.entries, p.stride);
            var t0: usize = 0;
            while (t0 < r.n_tiles) : (t0 += per) {
                const nt: u32 = @intCast(@min(per, r.n_tiles - t0));
                const r0: usize = t0 * tile_rows;
                const rn: u32 = @intCast(@min(@as(usize, nt) * tile_rows, r.rows - r0));
                try Dev.dispatch(k.attn_pq, s, .{ nt, g.kv_heads, z }, .{
                    q,      kv.k,     kv.v,    tab,         r.segs,   at(r.tiles, t0 * @sizeOf(Tile)), p.o,                    p.ml,
                    l.ring, l.window, l.chunk, @as(u32, 0), p.stride, scale,                           @as(u32, @intCast(r0)),
                });
                try merge(k, s, g.heads, l, p, rn, at(r.pos, r0 * 4), at(sink, r0 * g.heads * 2), at(gate, r0 * g.heads * head_dim * 2), at(out, r0 * g.heads * head_dim * 2));
            }
        }

        fn merge(k: *const Kernels, s: Dev.Stream, heads: u32, l: Layer, p: Partials, rows: u32, pos: Ptr, sink: Ptr, gate: Ptr, out: Ptr) !void {
            try Dev.dispatch(k.merge, s, .{ rows * heads, 1, 1 }, .{ p.o, p.ml, sink, gate, pos, out, l.window, l.chunk, @as(u32, 0), p.stride });
        }
    };
}

test {
    _ = @import("xpu_attention_test.zig");
}
