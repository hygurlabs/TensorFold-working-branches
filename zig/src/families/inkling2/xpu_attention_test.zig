//! Inkling 2's XPU attention plan without a GPU: chunks, slots, tiles, refusals and the recorded launches.
const std = @import("std");
const config = @import("config.zig");
const fixtures = @import("config_test.zig");
const k = @import("xpu_attention.zig");
const t = std.testing;

const mini: k.Geometry = .{ .heads = 24, .kv_heads = 8, .extent = 1024 };

fn miniText(a: std.mem.Allocator) !config.Text {
    var why: config.Why = .{};
    return (try config.parse(a, fixtures.mini, &why)).text;
}

test "inkling2 xpu attention: the baseline's chunks, tiles and partial slots at Mini's shapes" {
    try t.expectEqual(@as(u32, 256), k.global_chunk);
    try t.expectEqual(@as(u32, 128), k.sliding_chunk);
    try t.expectEqual(@as(u32, 10), mini.tileRows(false));
    try t.expectEqual(@as(u32, 21), mini.tileRows(true));
    try t.expectEqual(@as(usize, 24), @sizeOf(k.Seg));
    try t.expectEqual(@as(usize, 8), @sizeOf(k.Tile));
    const g = k.Layer.global(1 << 20);
    try t.expectEqual(@as(u32, 4096), k.slotsFor(g, 1 << 20));
    try t.expectEqual(@as(u32, 1), k.slotsFor(g, 1));
    const s = k.Layer.sliding(1024, 2048);
    try t.expectEqual(@as(u32, 9), k.slotsFor(s, 1 << 20));
    try t.expectEqual(@as(u32, 3072), s.ring);
    try t.expectEqual(@as(u32, 1088), k.Layer.sliding(1024, 65).ring);
    try t.expectEqual(@as(usize, 3072 * 8 * 256), s.cacheBytes(8));
    try t.expectEqual(@as(u32, 131136), k.Layer.global(131073).ring);
    try t.expectEqual(@as(usize, 24 * 4096 * 520), mini.partialBytes(1, 4096));
    try t.expectEqual(@as(usize, 100663296), mini.tableBytes(2048));
    try t.expectEqual(@as(f32, 1.0 / @sqrt(@as(f32, k.head_dim))), k.scale);
}

test "inkling2 xpu attention: every chunk a row reads has a partial slot, and a tile's blocks reach its last row's chunk" {
    const l = k.Layer.sliding(1024, 2048);
    for ([_]usize{ 0, 1, 127, 128, 1023, 1024, 1025, 5000, 70001 }) |pos| {
        const lo = (pos + 1) -| l.window;
        for (lo / l.chunk..pos / l.chunk + 1) |c| try t.expect(k.partialSlot(l, c, pos).? < k.slotsFor(l, pos + 1));
        try t.expectEqual(@as(?usize, null), k.partialSlot(l, pos / l.chunk + 1, pos));
    }
    // every first row and tile height: the blocks from the tile's first chunk reach the last row's chunk
    for ([_]u32{ 10, 21 }) |rows| {
        for (0..3000) |first| {
            const lo_chunk = ((first + 1) -| l.window) / l.chunk;
            try t.expect((first + rows - 1) / l.chunk < lo_chunk + k.chunkBlocks(l, rows, 9));
        }
    }
    try t.expectEqual(@as(u32, 17), k.chunkBlocks(k.Layer.global(4096), 10, 17));
}

test "inkling2 xpu attention: tiles hold the rows of one segment and cover each segment" {
    var out: [16]k.Tile = undefined;
    const segs = [_]k.Seg{ .{ .row0 = 0, .n = 11, .p0 = 0, .off = 0 }, .{ .row0 = 11, .n = 1, .p0 = 900, .off = 1 << 20 } };
    const n = try k.tiles(&segs, 10, &out);
    try t.expectEqual(@as(usize, 3), n);
    try t.expectEqualSlices(k.Tile, &.{ .{ .seg = 0, .row = 0 }, .{ .seg = 0, .row = 10 }, .{ .seg = 1, .row = 0 } }, out[0..3]);
    try t.expectError(error.TooManyTiles, k.tiles(&segs, 10, out[0..2]));
    try t.expectEqual(@as(usize, 98), k.tileCount(2048, 21));
    try t.expect(!k.isPrompt(32) and k.isPrompt(33));
}

test "inkling2 xpu attention: a prompt window holds the partials the buffer has, at most 24 tiles of 21 rows (504 rows), none if one tile does not fit" {
    try t.expectEqual(@as(usize, 24), k.windowTiles(mini, 1 << 30, 9)); // 24 tiles of 21 rows = 504, not 98
    try t.expectEqual(@as(usize, 2), k.windowTiles(mini, 2 * 21 * 24 * 4096, 4096));
    try t.expectEqual(@as(usize, 2), k.windowTiles(mini, 2 * 21 * 24 * 4096 + 21 * 24 * 4095, 4096));
    try t.expectEqual(@as(usize, 0), k.windowTiles(mini, 21 * 24 * 4096 - 1, 4096));
}

test "inkling2 xpu attention: the geometry comes from the config, a config other than the compiled image's is refused" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var text = try miniText(arena.allocator());
    const g = try k.Geometry.fromText(text);
    try t.expectEqual(mini, g);
    const keep = text;
    text.global.q += 8;
    try t.expectError(error.Inkling2Geometry, k.Geometry.fromText(text));
    text = keep;
    text.rel_extent = 2048;
    try t.expectError(error.Inkling2Geometry, k.Geometry.fromText(text));
    text = keep;
    text.local.kv = 7;
    text.global.kv = 7;
    try t.expectError(error.Inkling2Geometry, k.Geometry.fromText(text));
    // Small (q 48, kv 8) parses as a config but the image is compiled for Mini's 24 heads: refused, so are its loads
    text = keep;
    text.local.q = 48;
    text.global.q = 48;
    try t.expectError(error.Inkling2Geometry, k.Geometry.fromText(text));
    const small: k.Geometry = .{ .heads = 48, .kv_heads = 8, .extent = 1024 };
    const ctx: u8 = 0;
    try t.expectError(error.Inkling2Geometry, A.Kernels.load(&ctx, "image", small));
    // a group wider than a decode tile's 32 pairs would make tileRows 0
    try t.expectError(error.Inkling2Geometry, (k.Geometry{ .heads = 264, .kv_heads = 8, .extent = 1024 }).check());
    try t.expectError(error.Inkling2Geometry, (k.Geometry{ .heads = 24, .kv_heads = 0, .extent = 1024 }).check());
    try mini.check();
    // the prep constants of the baseline's launch
    const c = k.Prep.fromText(keep);
    try t.expectEqual(k.Prep{ .qk_clip = 22.627416997969522, .rel_clip = 8.0, .floor = 8192.0, .alpha = 0.1, .eps = 1e-6 }, c);
}

const Arg = union(enum) { ptr: usize, int: u64, float: f32 };
const Launch = struct { id: usize, grid: [3]u32, args: [24]Arg = undefined, n: usize = 0 };

/// A device that records what it is asked: kernels are numbered in creation order.
const Mock = struct {
    pub const Context = u8;
    pub const Module = struct { live: bool };
    pub const Kernel = struct { id: usize };
    pub const Stream = u8;
    var names: [8][]const u8 = undefined;
    var locals: [8][3]u32 = undefined;
    var made: usize = 0;
    var live_kernels: usize = 0;
    var live_modules: usize = 0;
    var fail_at: usize = 99;
    var launches: [16]Launch = undefined;
    var count: usize = 0;

    fn reset() void {
        made = 0;
        live_kernels = 0;
        live_modules = 0;
        fail_at = 99;
        count = 0;
    }

    pub fn load(_: *const Context, _: []const u8) !Module {
        live_modules += 1;
        return .{ .live = true };
    }

    pub fn unload(_: *Module) void {
        live_modules -= 1;
    }

    pub fn kernel(_: Module, name: [:0]const u8, local: [3]u32) !Kernel {
        if (made == fail_at) return error.Invalid;
        names[made] = name;
        locals[made] = local;
        made += 1;
        live_kernels += 1;
        return .{ .id = made };
    }

    pub fn deinit(_: *Kernel) void {
        live_kernels -= 1;
    }

    pub fn dispatch(kern: Kernel, _: Stream, grid: [3]u32, args: anytype) !void {
        var l: Launch = .{ .id = kern.id, .grid = grid };
        inline for (args) |v| {
            l.args[l.n] = switch (@typeInfo(@TypeOf(v))) {
                .optional, .pointer => .{ .ptr = @intFromPtr(v) },
                .int, .comptime_int => .{ .int = @intCast(v) },
                .float => .{ .float = v },
                else => @compileError("argument type"),
            };
            l.n += 1;
        }
        launches[count] = l;
        count += 1;
    }
};
const A = k.Attention(Mock);

fn loaded() !A.Kernels {
    Mock.reset();
    const ctx: u8 = 0;
    return A.Kernels.load(&ctx, "image", mini);
}

fn ptr(n: usize) k.Ptr {
    return @ptrFromInt(n);
}

fn call(rows: u32, streams: u32, last_pos: u32) k.Rows {
    return .{ .rows = rows, .pos = ptr(0x1000), .seg_of = ptr(0x2000), .segs = ptr(0x3000), .tiles = ptr(0x4000), .n_tiles = @intCast(k.tileCount(rows, mini.tileRows(k.isPrompt(rows)))), .last_pos = last_pos, .streams = streams, .row0 = 0 };
}

const no_cache: k.Cache = .{ .k = ptr(0x5000), .v = ptr(0x6000) };

fn part(stride: u32, entries: usize) k.Partials {
    return .{ .o = ptr(0x7000), .ml = ptr(0x8000), .stride = stride, .entries = entries };
}

fn attend(kern: *const A.Kernels, r: k.Rows, l: k.Layer, p: k.Partials) !void {
    try A.attend(kern, 0, r, l, p, ptr(0x9000), no_cache, ptr(0xa000), ptr(0xb000), ptr(0xc000), ptr(0xd000));
}

test "inkling2 xpu attention: the loader refuses an empty image and names the five kernels with their group sizes" {
    Mock.reset();
    const ctx: u8 = 0;
    try t.expectError(error.BuiltWithoutKernels, A.Kernels.load(&ctx, "", mini));
    try t.expectEqual(@as(usize, 0), Mock.live_modules);
    var kern = try loaded();
    const want = [_][]const u8{ "ink2_prep", "ink2_rel_table", "ink2_attn", "ink2_attn_pq", "ink2_merge" };
    for (want, 0..) |name, i| try t.expectEqualStrings(name, Mock.names[i]);
    try t.expectEqualSlices(u32, &.{ 16, 256, 128, 128, 128 }, &.{ Mock.locals[0][0], Mock.locals[1][0], Mock.locals[2][0], Mock.locals[3][0], Mock.locals[4][0] });
    kern.unload();
    try t.expectEqual(@as(usize, 0), Mock.live_kernels);
    try t.expectEqual(@as(usize, 0), Mock.live_modules);
}

test "inkling2 xpu attention: a kernel the module lacks releases what was made" {
    Mock.reset();
    Mock.fail_at = 3;
    const ctx: u8 = 0;
    try t.expectError(error.Invalid, A.Kernels.load(&ctx, "image", mini));
    try t.expectEqual(@as(usize, 0), Mock.live_kernels);
    try t.expectEqual(@as(usize, 0), Mock.live_modules);
}

test "inkling2 xpu attention: prep and the bias table launch their grids with the arguments in the kernels' order" {
    const kern = try loaded();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const prep_c = k.Prep.fromText(try miniText(arena.allocator()));
    const g = k.Layer.global(8192);
    try A.prep(&kern, 0, call(5, 1, 4), g, prep_c, .{ ptr(1), ptr(2), ptr(3), ptr(4) }, .{ ptr(5), ptr(6) }, ptr(7), ptr(8), no_cache);
    try A.relTable(&kern, 0, call(5, 1, 4), ptr(9), ptr(10), ptr(11));
    try t.expectEqual(@as(usize, 2), Mock.count);
    const p = Mock.launches[0];
    try t.expectEqual(@as(usize, 1), p.id);
    try t.expectEqual([3]u32{ 5, 40, 1 }, p.grid);
    try t.expectEqual(@as(usize, 20), p.n);
    for (0..6) |i| try t.expectEqual(Arg{ .ptr = i + 1 }, p.args[i]);
    try t.expectEqual(Arg{ .ptr = 0x1000 }, p.args[6]); // rpos, rseg, segs
    try t.expectEqual(Arg{ .ptr = 0x3000 }, p.args[8]);
    try t.expectEqual(Arg{ .ptr = 7 }, p.args[9]); // q, rel, kc, vc
    try t.expectEqual(Arg{ .ptr = 0x6000 }, p.args[12]);
    try t.expectEqual(Arg{ .int = 8192 }, p.args[13]); // ring
    try t.expectEqual(Arg{ .int = 1 }, p.args[14]); // a global layer
    try t.expectEqual(Arg{ .float = 8.0 }, p.args[16]);
    try t.expectEqual(Arg{ .float = 8192.0 }, p.args[17]);
    try t.expectEqual(@as(usize, 2), Mock.launches[1].id);
    try t.expectEqual([3]u32{ 120, 1, 1 }, Mock.launches[1].grid);
    Mock.count = 0;
    try A.prep(&kern, 0, call(5, 1, 4), k.Layer.sliding(1024, 5), prep_c, .{ ptr(1), ptr(2), ptr(3), ptr(4) }, .{ ptr(5), ptr(6) }, ptr(7), ptr(8), no_cache);
    try t.expectEqual(Arg{ .int = 0 }, Mock.launches[0].args[14]);
}

test "inkling2 xpu attention: a decode call is one ink2_attn over the tiles, then the merge" {
    const kern = try loaded();
    const sl = k.Layer.sliding(1024, 8);
    const stride = k.slotsFor(sl, 1);
    Mock.count = 0;
    try attend(&kern, call(8, 1, 5000), sl, part(stride, 8 * 24 * stride));
    try t.expectEqual(@as(usize, 2), Mock.count);
    const a = Mock.launches[0];
    try t.expectEqual(@as(usize, 3), a.id);
    try t.expectEqual([3]u32{ 1, 8, 10 }, a.grid); // 8 rows: one tile of 10; (9 + 1024) / 128 + 2 blocks
    const want = [_]Arg{ .{ .ptr = 0x9000 }, .{ .ptr = 0x5000 }, .{ .ptr = 0x6000 }, .{ .ptr = 0xa000 }, .{ .ptr = 0x3000 }, .{ .ptr = 0x4000 }, .{ .ptr = 0x7000 }, .{ .ptr = 0x8000 }, .{ .int = 1088 }, .{ .int = 1024 }, .{ .int = 128 }, .{ .int = 0 }, .{ .int = 9 }, .{ .float = k.scale } };
    try t.expectEqualSlices(Arg, &want, a.args[0..a.n]);
    const m = Mock.launches[1];
    try t.expectEqual(@as(usize, 5), m.id);
    try t.expectEqual([3]u32{ 192, 1, 1 }, m.grid);
    const mw = [_]Arg{ .{ .ptr = 0x7000 }, .{ .ptr = 0x8000 }, .{ .ptr = 0xb000 }, .{ .ptr = 0xc000 }, .{ .ptr = 0x1000 }, .{ .ptr = 0xd000 }, .{ .int = 1024 }, .{ .int = 128 }, .{ .int = 0 }, .{ .int = 9 } };
    try t.expectEqualSlices(Arg, &mw, m.args[0..m.n]);
    // a global layer takes a block for every chunk of the stride
    Mock.count = 0;
    const g = k.Layer.global(8192);
    try attend(&kern, call(32, 3, 8191), g, part(32, 32 * 24 * 32));
    try t.expectEqual([3]u32{ 4, 8, 32 }, Mock.launches[0].grid);
    try t.expectEqual(@as(usize, 3), Mock.launches[0].id);
}

test "inkling2 xpu attention: a prompt runs ink2_attn_pq in windows, each merged at its own rows" {
    const kern = try loaded();
    const g = k.Layer.global(8192);
    const stride: u32 = 32;
    // room for two tiles of 21 rows: 100 rows are 5 tiles, so three windows of 42, 42 and 16 rows
    Mock.count = 0;
    try attend(&kern, call(100, 1, 8191), g, part(stride, 2 * 21 * 24 * stride));
    try t.expectEqual(@as(usize, 6), Mock.count);
    const grids = [_][3]u32{ .{ 2, 8, 32 }, .{ 1008, 1, 1 }, .{ 2, 8, 32 }, .{ 1008, 1, 1 }, .{ 1, 8, 32 }, .{ 384, 1, 1 } };
    const ids = [_]usize{ 4, 5, 4, 5, 4, 5 };
    for (grids, ids, 0..) |grid, id, i| {
        try t.expectEqual(grid, Mock.launches[i].grid);
        try t.expectEqual(id, Mock.launches[i].id);
    }
    // the second window starts at tile 2 (16 bytes in), row 42, and its rows' sink, gate, pos and out
    const a = Mock.launches[2];
    try t.expectEqual(Arg{ .ptr = 0x4000 + 16 }, a.args[5]);
    try t.expectEqual(Arg{ .int = 42 }, a.args[14]);
    try t.expectEqual(@as(usize, 15), a.n);
    const m = Mock.launches[3];
    try t.expectEqual(Arg{ .ptr = 0xb000 + 42 * 24 * 2 }, m.args[2]);
    try t.expectEqual(Arg{ .ptr = 0xc000 + 42 * 24 * 128 * 2 }, m.args[3]);
    try t.expectEqual(Arg{ .ptr = 0x1000 + 42 * 4 }, m.args[4]);
    try t.expectEqual(Arg{ .ptr = 0xd000 + 42 * 24 * 128 * 2 }, m.args[5]);
    // the partials start at the buffer's start for every window
    try t.expectEqual(Arg{ .ptr = 0x7000 }, m.args[0]);
    // a window never passes 24 tiles = 504 rows (window_rows 512 is the cap that count comes from)
    Mock.count = 0;
    const sl = k.Layer.sliding(1024, 2048);
    try attend(&kern, call(1100, 1, 1099), sl, part(9, 1 << 30));
    try t.expectEqual(@as(usize, 6), Mock.count); // 53 tiles in windows of 24, 24 and 5
    try t.expectEqual(@as(u32, 24), Mock.launches[0].grid[0]);
    try t.expectEqual(@as(u32, 504 * 24), Mock.launches[1].grid[0]);
    try t.expectEqual(@as(u32, 5), Mock.launches[4].grid[0]);
    try t.expectEqual(@as(u32, 92 * 24), Mock.launches[5].grid[0]); // the rest: 1100 - 2 x 504 rows
}

test "inkling2 xpu attention: attend refuses a bad ring, zero streams and short partials, before any launch" {
    const kern = try loaded();
    var g = k.Layer.global(8192);
    g.ring = 8191;
    Mock.count = 0;
    try t.expectError(error.Inkling2RingTiles, attend(&kern, call(4, 1, 5000), g, part(32, 4 * 24 * 32)));
    var l = k.Layer.sliding(1024, 4);
    l.ring = 1024; // whole tiles, but four rows' new keys overwrite keys the first row still reads
    try t.expectError(error.Inkling2RingShort, attend(&kern, call(4, 1, 5000), l, part(9, 4 * 24 * 9)));
    l.ring = 1088;
    try t.expectError(error.Inkling2Streams, attend(&kern, call(4, 0, 5000), l, part(9, 4 * 24 * 9)));
    try t.expectError(error.Inkling2Partials, attend(&kern, call(4, 1, 5000), l, part(8, 4 * 24 * 9)));
    try t.expectError(error.Inkling2PartialRoom, attend(&kern, call(4, 1, 5000), l, part(9, 4 * 24 * 9 - 1)));
    // a prompt: several streams, a tile table that is not the rows', and a buffer that cannot hold one tile
    const big = k.Layer.sliding(1024, 100);
    try t.expectError(error.Inkling2Streams, attend(&kern, call(100, 2, 5000), big, part(9, 1 << 30)));
    var r = call(100, 1, 5000);
    r.n_tiles += 1;
    try t.expectError(error.Inkling2Tiles, attend(&kern, r, big, part(9, 1 << 30)));
    try t.expectError(error.Inkling2PartialRoom, attend(&kern, call(100, 1, 5000), big, part(9, 21 * 24 * 9 - 1)));
    try t.expectEqual(@as(usize, 0), Mock.count);
    // a prompt whose segment does not start at row 0 would index partials the merge does not read
    r = call(100, 1, 5000);
    r.row0 = 50;
    try t.expectError(error.Inkling2PromptRow0, attend(&kern, r, big, part(9, 1 << 30)));
    // a global ring of 64 holds 64 rows, not 65 (positions 0 and 64 share slot 0); 64 rows: 2 launches
    var g64 = k.Layer.global(64);
    try t.expectError(error.Inkling2RingShort, attend(&kern, call(65, 1, 64), g64, part(1, 1 << 30)));
    g64.chunk = k.global_chunk;
    try attend(&kern, call(64, 1, 63), g64, part(1, 1 << 30));
    try attend(&kern, call(100, 1, 5000), big, part(9, 21 * 24 * 9));
    try attend(&kern, call(4, 1, 5000), l, part(9, 4 * 24 * 9));
    try t.expectEqual(@as(usize, 2 * 5 + 2 + 2), Mock.count);
}
