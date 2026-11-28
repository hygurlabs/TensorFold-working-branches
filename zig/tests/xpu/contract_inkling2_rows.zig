//! Inkling 2's XPU attention under the contract at Mini's 24/8 heads: decode, verify and prompt launches, their merge.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const X = @import("inkling2_xpu");
const rig_ = @import("contract_rig.zig");
const shim = @import("contract_inkling2_dev.zig");
const fill_ = @import("contract_inkling2_fill.zig");

const ref = core.inkling2_attn_ref;
const c = core.spec;
const g = core.gates;
const D = ref.D;
const heads = fill_.heads;
const kv = fill_.kv;
const n_out = heads * D;
const A = X.Attention(shim);

/// The layer: sliding (window 1,024, 128-key chunks) or global (256-key chunks, a 4,096-slot ring that wraps).
pub const Kind = enum { sliding, global };
/// What `replaced` runs: nothing (float64 decides), the frozen baseline kernels, each row alone, each stream alone.
pub const Mode = enum { reference, frozen, one_row, apart };
/// A control's planted fault: a kernel's own, a launch's, or an input's.
pub const Plant = enum { none, p_trunc, k_block, fold_1024, no_sink, no_gate };
pub const Options = struct { kind: Kind, mode: Mode = .reference, plant: Plant = .none, exact_p: bool = false };

/// Every registration: float64 against the plan's rows, then the byte comparisons, the exact cases, the five controls.
pub const registrations = [_]Options{
    .{ .kind = .sliding },
    .{ .kind = .global },
    .{ .kind = .sliding, .mode = .frozen },
    .{ .kind = .global, .mode = .frozen },
    .{ .kind = .sliding, .mode = .one_row },
    .{ .kind = .global, .mode = .one_row },
    .{ .kind = .sliding, .mode = .apart },
    .{ .kind = .global, .mode = .apart },
    .{ .kind = .sliding, .exact_p = true },
    .{ .kind = .global, .exact_p = true },
    .{ .kind = .global, .exact_p = true, .plant = .p_trunc },
    .{ .kind = .global, .mode = .frozen, .plant = .k_block },
    .{ .kind = .global, .mode = .frozen, .plant = .fold_1024 },
    .{ .kind = .sliding, .plant = .no_sink },
    .{ .kind = .global, .plant = .no_gate },
};

const geometry: X.Geometry = .{ .heads = heads, .kv_heads = kv, .extent = ref.extent };
/// Partials a prompt window may hold at most: past this a window is a few tiles (the 512-row cap needs short contexts).
const partial_budget = 1 << 17;

fn depth(_: c.Shape) usize {
    return 4 * 17 + 2 * (4096 + 9) + 8; // the bounds come from the reference's own chains (Chain.xpu)
}

fn sh(rows: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n_out, .k = k };
}

// Contexts at the chunk and window edges, decode windows, 600 prompt rows (past the 512-row cap), 1M keys on a ring.
const sliding_served = [_]c.Shape{
    sh(8, 2048),    sh(1, 1048576), sh(1, 127),    sh(1, 128),  sh(1, 129),  sh(1, 255),   sh(1, 256),
    sh(1, 257),     sh(5, 1023),    sh(5, 1024),   sh(5, 1025), sh(7, 1030), sh(32, 5000), sh(33, 1030),
    sh(130, 20000), sh(600, 1100),  sh(600, 3000),
};
const global_served = [_]c.Shape{
    sh(8, 1100),   sh(1, 1048576), sh(1, 255),    sh(1, 256),    sh(1, 257),   sh(5, 1023),  sh(5, 1024),
    sh(5, 1025),   sh(4, 16390),   sh(2, 300000), sh(16, 20000), sh(32, 2000), sh(33, 1100), sh(130, 20000),
    sh(600, 1100),
};
// three streams in one launch: at least a row each, none over a decode window
const apart_sliding = [_]c.Shape{ sh(9, 2048), sh(16, 20000), sh(3, 129), sh(5, 1025), sh(32, 5000) };
const apart_global = [_]c.Shape{ sh(9, 1100), sh(16, 20000), sh(3, 300000), sh(4, 257), sh(5, 1025), sh(32, 2000) };
// exact P: rows below position 64, one tile each (the prompt kernel's too); the outputs equal float64 bit for bit
const exact_shapes = [_]c.Shape{ sh(16, 64), sh(1, 64), sh(40, 64), sh(3, 40) };

fn served(o: Options) []const c.Shape {
    if (o.exact_p) return &exact_shapes;
    if (o.mode == .apart) return if (o.kind == .sliding) &apart_sliding else &apart_global;
    return if (o.kind == .sliding) &sliding_served else &global_served;
}

fn layerOf(kind: Kind) X.Layer {
    return if (kind == .sliding) X.Layer.sliding(1024, 2048) else X.Layer.global(fill_.global_ring);
}

fn baseName(o: Options) []const u8 {
    if (o.exact_p) return if (o.kind == .sliding)
        "inkling2 xpu attn sliding rows at bf16 P's own bits (q 0, no sink, gate 1, integer v)"
    else
        "inkling2 xpu attn global rows at bf16 P's own bits (q 0, no sink, gate 1, integer v)";
    const sliding = o.kind == .sliding;
    return switch (o.mode) {
        .reference => if (sliding) "inkling2 xpu attn ink2_attn, ink2_attn_pq + ink2_merge (sliding layers: window 1,024, 128-key chunks)" else "inkling2 xpu attn ink2_attn, ink2_attn_pq + ink2_merge (global layers: 256-key chunks)",
        .frozen => if (sliding) "inkling2 xpu attn sliding layers against the frozen ink-xpu-baseline kernels" else "inkling2 xpu attn global layers against the frozen ink-xpu-baseline kernels",
        .one_row => if (sliding) "inkling2 xpu attn sliding verify rows and prompt windows against one-row decode" else "inkling2 xpu attn global verify rows and prompt windows against one-row decode",
        .apart => if (sliding) "inkling2 xpu attn sliding: three streams' rows and caches in one launch" else "inkling2 xpu attn global: three streams' rows and caches in one launch",
    };
}

pub fn spec(o: Options) c.Spec {
    var x: c.Spec = .{ .name = baseName(o), .accum = .f32, .depth = depth, .ops = 12, .out = .bf16, .served = served(o) };
    x.tiles = .{ .{ .size = geometry.tileRows(false) }, .{}, .{} };
    if (o.exact_p) {
        const why = "64 keys of gate 4's values: neither fp16 nor bf16 provably misses here; the 1M-key registrations prove fp32";
        x.sums = .{ .f16 = 0, .bf16 = 0, .why = why };
        return x;
    }
    x.replaces = switch (o.mode) {
        .reference => null,
        .frozen => "ink-xpu-baseline's ink2_attn, ink2_attn_pq and ink2_merge (KV8 off), frozen in contract_inkling2_upstream.cl",
        .one_row => "the same rows one at a time on ink2_attn and ink2_merge",
        .apart => "the same launches one stream at a time, at each stream's row and cache offsets",
    };
    return x;
}

/// The kernel the options name; a control's name says its fault.
fn named(o: Options) []const u8 {
    return switch (o.plant) {
        .none => baseName(o),
        .p_trunc => "control: inkling2 xpu attn rows with P cut to bf16 by truncation instead of rounded to nearest even",
        .k_block => "control: inkling2 xpu attn global rows with a tile's 16-key blocks summed last to first in P.V",
        .fold_1024 => "control: inkling2 xpu attn global rows folded in 1,024-key chunks instead of 256",
        .no_sink => "control: inkling2 xpu attn sliding rows merged without their sink",
        .no_gate => "control: inkling2 xpu attn global rows merged without their gate",
    };
}

/// A baseline kernel set: the frozen image's five (prep and table are the prep file's).
const Upstream = struct { attn: xpu.Kernel, pq: xpu.Kernel, merge: xpu.Kernel };

pub const Attn = struct {
    rig: rig_.Rig,
    o: Options,
    kern: A.Kernels,
    up: Upstream,
    ctl: ?xpu.Kernel = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    lay: fill_.Layout = undefined,
    data: fill_.Data = .{},
    tol: []f64 = &.{},
    dev: Dev = .{},
    spans: []Span = &.{},
    alone_segs: []X.Seg = &.{},
    alone_tiles: []X.Tile = &.{},

    const Ptr = X.Ptr;
    const Dev = struct { q: Ptr = null, kc: Ptr = null, vc: Ptr = null, tab: Ptr = null, sink: Ptr = null, sink_run: Ptr = null, gate: Ptr = null, gate_run: Ptr = null, pos: Ptr = null, segs: Ptr = null, tiles: Ptr = null, n_tiles: u32 = 0, po: Ptr = null, pml: Ptr = null, out: Ptr = null, base: Ptr = null, alone_segs: Ptr = null, alone_tiles: Ptr = null, po1: Ptr = null, pml1: Ptr = null };
    /// Rows `row0`.. of the call that one launch of the baseline takes alone (one row, or one stream's rows).
    const Span = struct { row0: usize, n: usize, tile0: usize, tiles: usize };

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, o: Options) !Attn {
        var kern = try A.Kernels.load(&dev.ctx, rig_.blob(.inkling2), geometry);
        errdefer kern.unload();
        var attn = try dev.kernel(.inkling2_upstream, "ink2_attn", .{ 128, 1, 1 });
        errdefer attn.deinit();
        var pq = try dev.kernel(.inkling2_upstream, "ink2_attn_pq", .{ 128, 1, 1 });
        errdefer pq.deinit();
        var merge = try dev.kernel(.inkling2_upstream, "ink2_merge", .{ 128, 1, 1 });
        errdefer merge.deinit();
        const ctl: ?xpu.Kernel = switch (o.plant) {
            .p_trunc => try dev.kernel(.inkling2_controls, "tf_ctl_ink2_attn_ptrunc", .{ 128, 1, 1 }),
            .k_block => try dev.kernel(.inkling2_controls, "tf_ctl_ink2_attn_kblock", .{ 128, 1, 1 }),
            else => null,
        };
        return .{ .rig = .{ .a = a, .dev = dev }, .o = o, .kern = kern, .up = .{ .attn = attn, .pq = pq, .merge = merge }, .ctl = ctl };
    }

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
        if (self.ctl) |*k| k.deinit();
        self.up.merge.deinit();
        self.up.pq.deinit();
        self.up.attn.deinit();
        self.kern.unload();
    }

    fn free(self: *Attn) void {
        const a = self.rig.a;
        self.data.free(a);
        a.free(self.tol);
        self.tol = &.{};
        a.free(self.spans);
        a.free(self.alone_segs);
        a.free(self.alone_tiles);
        self.spans = &.{};
        self.alone_segs = &.{};
        self.alone_tiles = &.{};
    }

    pub fn kernel(self: *Attn) g.Kernel {
        var s = spec(self.o);
        s.name = named(self.o);
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.o.plant == .none) null else .shapes };
    }

    const vt = g.hooks(Attn);

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * n_out;
    }

    fn call(self: *const Attn) ref.Call {
        const d = &self.data;
        return .{ .q = d.q, .k = d.kc, .v = d.vc, .tab = d.tab, .sink = d.sink, .gate = d.gate, .pos = d.pos, .stream = d.stream, .ring = self.lay.layer.ring, .window = self.lay.layer.window, .exact = self.fill == .stress, .heads = heads, .kv = kv, .chain = .xpu };
    }

    fn stride(self: *const Attn) u32 {
        return X.slotsFor(self.lay.layer, self.shape.k);
    }

    /// Partials a launch holds: every row's for a decode window, a few prompt tiles' past the budget.
    fn entries(self: *const Attn) usize {
        const per_row = @as(usize, heads) * self.stride();
        const full = self.shape.rows * per_row;
        if (!X.isPrompt(@intCast(self.shape.rows))) return full;
        return @max(geometry.tileRows(true) * per_row, @min(full, partial_budget));
    }

    /// The rows' streams (fill_.Layout), their inputs, and the device's buffers; false for a shape the plan refuses.
    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        if (shape.n != n_out or rows == 0 or shape.k < rows or rows > 2048) return false;
        if (self.o.mode == .apart and (rows < 3 or rows > X.decode_rows)) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.a;
        const layer = layerOf(self.o.kind);
        const streams: usize = if (self.o.mode == .apart) 3 else if (rows >= 2 and rows <= X.decode_rows) 2 else 1;
        self.lay = fill_.Layout.of(layer, rows, shape.k, streams);
        if (self.o.exact_p and fill == .random) try self.data.makeExactP(a, self.lay, seed) else try self.data.make(a, self.lay, fill, seed);
        const d = &self.dev;
        const td = &self.data;
        d.q = try self.rig.input(std.mem.sliceAsBytes(td.q));
        d.kc = try self.rig.input(std.mem.sliceAsBytes(td.kc));
        d.vc = try self.rig.input(std.mem.sliceAsBytes(td.vc));
        d.tab = try self.rig.input(std.mem.sliceAsBytes(td.tab));
        // the sink outweighs every key (y is ~0), so dropping it moves y by its whole size
        if (self.o.plant == .no_sink and fill == .random) @memset(td.sink, ref.bits(40));
        d.sink = try self.rig.input(std.mem.sliceAsBytes(td.sink));
        d.gate = try self.rig.input(std.mem.sliceAsBytes(td.gate));
        d.sink_run = d.sink;
        d.gate_run = d.gate;
        if (self.o.plant == .no_sink) {
            const t = try a.alloc(u16, td.sink.len);
            defer a.free(t);
            @memset(t, 0xff80); // -inf: exp2 of it is 0
            d.sink_run = try self.rig.input(std.mem.sliceAsBytes(t));
        } else if (self.o.plant == .no_gate) {
            const t = try a.alloc(u16, td.gate.len);
            defer a.free(t);
            @memset(t, ref.bits(32)); // sigmoid(32) is 1 in fp32
            d.gate_run = try self.rig.input(std.mem.sliceAsBytes(t));
        }
        const pos32 = try a.alloc(i32, rows);
        defer a.free(pos32);
        for (pos32, td.pos) |*p, v| p.* = @intCast(v);
        d.pos = try self.rig.input(std.mem.sliceAsBytes(pos32));
        const segs = self.lay.segs[0..streams];
        d.segs = try self.rig.input(std.mem.sliceAsBytes(segs));
        const tile_rows = geometry.tileRows(X.isPrompt(@intCast(rows)));
        const tiles = try a.alloc(X.Tile, X.tileCount(rows, tile_rows) + streams);
        defer a.free(tiles);
        const n = try X.tiles(segs, tile_rows, tiles);
        d.n_tiles = @intCast(n);
        d.tiles = try self.rig.input(std.mem.sliceAsBytes(tiles[0..n]));
        try self.alone(streams);
        const parts = self.entries();
        d.po = try self.rig.scratch(parts * D * 4);
        d.pml = try self.rig.scratch(parts * 8);
        d.out = try self.rig.output(rows * n_out * 2);
        if (self.o.mode != .reference) d.base = try self.rig.scratch(rows * n_out * 2);
        return true;
    }

    /// The baseline's launches for `one_row` and `apart`: a segment from row 0 and its tiles for each span of rows.
    fn alone(self: *Attn, streams: usize) !void {
        if (self.o.mode != .one_row and self.o.mode != .apart) return;
        const a = self.rig.a;
        const rows = self.shape.rows;
        const n_spans = if (self.o.mode == .apart) streams else rows;
        self.spans = try a.alloc(Span, n_spans);
        self.alone_segs = try a.alloc(X.Seg, n_spans);
        var tiles: std.ArrayList(X.Tile) = .empty;
        defer tiles.deinit(a);
        var most: usize = 1;
        for (self.spans, self.alone_segs, 0..) |*sp, *sg, i| {
            const s = if (self.o.mode == .apart) self.lay.segs[i] else self.lay.segs[self.data.stream[i]];
            const row0: usize = if (self.o.mode == .apart) @intCast(s.row0) else i;
            const n: usize = if (self.o.mode == .apart) @intCast(s.n) else 1;
            sg.* = .{ .row0 = 0, .n = @intCast(n), .p0 = @intCast(self.data.pos[row0]), .off = s.off };
            const t0 = tiles.items.len;
            const room = try tiles.addManyAsSlice(a, X.tileCount(n, geometry.tileRows(false)));
            _ = try X.tiles(sg[0..1], geometry.tileRows(false), room);
            sp.* = .{ .row0 = row0, .n = n, .tile0 = t0, .tiles = room.len };
            most = @max(most, n);
        }
        self.alone_tiles = try tiles.toOwnedSlice(a);
        const d = &self.dev;
        d.alone_segs = try self.rig.input(std.mem.sliceAsBytes(self.alone_segs));
        d.alone_tiles = try self.rig.input(std.mem.sliceAsBytes(self.alone_tiles));
        const parts = most * heads * self.stride();
        d.po1 = try self.rig.scratch(parts * D * 4);
        d.pml1 = try self.rig.scratch(parts * 8);
    }

    fn rowsAll(self: *const Attn) X.Rows {
        const d = self.dev;
        return .{ .rows = @intCast(self.shape.rows), .pos = d.pos, .seg_of = null, .segs = d.segs, .tiles = d.tiles, .n_tiles = d.n_tiles, .last_pos = @intCast(self.shape.k - 1), .streams = @intCast(self.lay.streams) };
    }

    /// The plan as the forward calls it, on the kernels the options name (a control swaps the decode kernel).
    pub fn run(self: *Attn) !void {
        const d = self.dev;
        var k = self.kern;
        if (self.ctl) |ctl| k.attn = ctl;
        var layer = self.lay.layer;
        if (self.o.plant == .fold_1024) layer.chunk = 1024;
        const p: X.Partials = .{ .o = d.po, .ml = d.pml, .stride = X.slotsFor(layer, self.shape.k), .entries = self.entries() };
        try A.attend(&k, self.rig.dev.s, self.rowsAll(), layer, p, d.q, .{ .k = d.kc, .v = d.vc }, d.tab, d.sink_run, d.gate_run, d.out);
        try self.rig.dev.s.synchronize();
    }

    fn launch(self: *Attn, k: xpu.Kernel, grid: [3]u32, args: anytype) !void {
        try shim.dispatch(k, self.rig.dev.s, grid, args);
    }

    /// The baseline's attend: the frozen kernels' argument lists (sink, gate, out unread) over the plan's windows.
    fn frozen(self: *Attn) !void {
        const d = self.dev;
        const l = self.lay.layer;
        const rows: u32 = @intCast(self.shape.rows);
        const st = self.stride();
        const prompt = X.isPrompt(rows);
        const tile_rows = geometry.tileRows(prompt);
        const z = X.chunkBlocks(l, tile_rows, st);
        const zero: u32 = 0;
        if (!prompt) {
            try self.launch(self.up.attn, .{ d.n_tiles, kv, z }, .{ d.q, d.kc, d.vc, d.tab, d.segs, d.tiles, d.po, d.pml, d.sink, d.gate, d.base, l.ring, l.window, l.chunk, zero, st, X.scale });
            return self.launch(self.up.merge, .{ rows * heads, 1, 1 }, .{ d.po, d.pml, d.sink, d.gate, d.pos, d.base, l.window, l.chunk, zero, st });
        }
        const per = X.windowTiles(geometry, self.entries(), st);
        var t0: usize = 0;
        while (t0 < d.n_tiles) : (t0 += per) {
            const nt: u32 = @intCast(@min(per, d.n_tiles - t0));
            const r0: usize = t0 * tile_rows;
            const rn: u32 = @intCast(@min(@as(usize, nt) * tile_rows, rows - r0));
            try self.launch(self.up.pq, .{ nt, kv, z }, .{ d.q, d.kc, d.vc, d.tab, d.segs, X.at(d.tiles, t0 * @sizeOf(X.Tile)), d.po, d.pml, d.sink, d.gate, d.base, l.ring, l.window, l.chunk, zero, st, X.scale, @as(u32, @intCast(r0)) });
            const out = X.at(d.base, r0 * n_out * 2);
            try self.launch(self.up.merge, .{ rn * heads, 1, 1 }, .{ d.po, d.pml, X.at(d.sink, r0 * heads * 2), X.at(d.gate, r0 * n_out * 2), X.at(d.pos, r0 * 4), out, l.window, l.chunk, zero, st });
        }
    }

    /// The new plan on each span alone: its segment from row 0, its rows' inputs and outputs by offset.
    fn spansAlone(self: *Attn) !void {
        const d = self.dev;
        for (self.spans, 0..) |sp, i| {
            const r: X.Rows = .{ .rows = @intCast(sp.n), .pos = X.at(d.pos, sp.row0 * 4), .seg_of = null, .segs = X.at(d.alone_segs, i * @sizeOf(X.Seg)), .tiles = X.at(d.alone_tiles, sp.tile0 * @sizeOf(X.Tile)), .n_tiles = @intCast(sp.tiles), .last_pos = @intCast(self.data.pos[sp.row0 + sp.n - 1]), .streams = 1 };
            const p: X.Partials = .{ .o = d.po1, .ml = d.pml1, .stride = self.stride(), .entries = sp.n * heads * self.stride() };
            const b = sp.row0 * heads;
            try A.attend(&self.kern, self.rig.dev.s, r, self.lay.layer, p, X.at(d.q, b * D * 2), .{ .k = d.kc, .v = d.vc }, X.at(d.tab, b * ref.extent * 2), X.at(d.sink, b * 2), X.at(d.gate, b * D * 2), X.at(d.base, b * D * 2));
        }
    }

    /// The baseline on the true inputs: the frozen kernels, each row alone, or each stream alone.
    pub fn replaced(self: *Attn) ![]const u8 {
        if (self.o.mode == .frozen) try self.frozen() else try self.spansAlone();
        try self.rig.dev.s.synchronize();
        return self.rig.bytes(self.dev.base, self.count() * 2);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.a;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        const cl = self.call();
        if (self.o.exact_p and self.fill == .random) {
            // bit for bit, one bf16 ulp only within 4 fp32 ulp of a tie (the device's `/` is inexact)
            for (0..self.shape.rows * heads) |rh| {
                fill_.exactP(&cl, rh / heads, rh % heads, want[rh * D ..][0..D], self.tol[rh * D ..][0..D]);
            }
        } else try fill_.Pool.run(a, .{ .call = &cl, .want = want, .tol = self.tol }, self.shape.rows * heads);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    /// Gate 4 (stress rows only): q = 0 gives every head of a key head the same row, so one replay serves its group.
    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        std.debug.assert(self.fill == .stress);
        const cl = self.call();
        try fill_.Pool.run(self.rig.a, .{ .call = &cl, .want = out, .tol = &.{}, .sim = .{ .acc = acc, .order = order } }, self.shape.rows * kv);
    }
};
