//! Tiles of operands for the Xe2 DPAS model: hand-built adversarial cases, slot sweeps and seeded random classes.
const std = @import("std");
const dpas = @import("xe2_dpas.zig");

pub const Kind = dpas.Kind;

/// An 8 x 16 output tile: A is 8 rows of K operands, B K rows of 16 columns, C the accumulators; operands are raw bits.
pub const Tile = struct { a: [8][16]u32 = @splat(@splat(0)), b: [16][16]u32 = @splat(@splat(0)), c: [8][16]f32 = @splat(@splat(0)) };

pub const Class = enum { tight, mid, wide, subnormal, ties, cancel, nearcancel };

/// The random classes a format is checked with.
pub fn classes(kind: Kind) []const Class {
    return switch (kind) {
        .bf16 => &.{ .tight, .mid, .wide, .subnormal, .ties, .cancel, .nearcancel },
        .f16, .tf32 => &.{ .tight, .mid, .subnormal, .cancel, .nearcancel },
    };
}

const Layout = struct { mant: u5, exp_bits: u4, center: i32 };

fn layout(kind: Kind) Layout {
    return switch (kind) {
        .bf16 => .{ .mant = 7, .exp_bits = 8, .center = 127 },
        .f16 => .{ .mant = 10, .exp_bits = 5, .center = 15 },
        .tf32 => .{ .mant = 23, .exp_bits = 8, .center = 127 },
    };
}

fn signBit(kind: Kind) u32 {
    const l = layout(kind);
    return @as(u32, 1) << (@as(u5, l.mant) + l.exp_bits);
}

/// The exact value of an operand word, as the unit sees it (tf32 cut to ten mantissa bits).
pub fn valueOf(kind: Kind, w: u32) f64 {
    return switch (kind) {
        .bf16 => @as(f32, @bitCast(w << 16)),
        .f16 => @as(f16, @bitCast(@as(u16, @intCast(w)))),
        .tf32 => @as(f32, @bitCast(w & 0xffffe000)),
    };
}

/// `x` as an operand word when the format holds it exactly (an infinity, or finite with nothing cut), else null.
pub fn operand(kind: Kind, x: f32) ?u32 {
    const w: u32 = switch (kind) {
        .bf16 => @as(u32, @bitCast(x)) >> 16,
        .f16 => @as(u16, @bitCast(@as(f16, @floatCast(x)))),
        .tf32 => @as(u32, @bitCast(x)),
    };
    if (kind == .tf32 and w & 0x1fff != 0) return null;
    if (std.math.isNan(x) or valueOf(kind, w) != x) return null;
    return w;
}

const Rand = struct {
    r: std.Random,
    kind: Kind,

    fn word(s: Rand, spread: i32, center: i32, zero_p: f64, sub_p: f64) u32 {
        const l = layout(s.kind);
        const top: i32 = (@as(i32, 1) << l.exp_bits) - 2;
        var e = std.math.clamp(center + s.r.intRangeAtMost(i32, -spread, spread), 0, top);
        if (sub_p > 0 and s.r.float(f64) < sub_p) e = 0;
        const m = s.r.uintLessThan(u32, @as(u32, 1) << l.mant);
        const v = (if (s.r.boolean()) signBit(s.kind) else 0) | (@as(u32, @intCast(e)) << l.mant) | m;
        return if (s.r.float(f64) < zero_p) 0 else v;
    }

    fn f32w(s: Rand, spread: i32, center: i32) f32 {
        const e: u32 = @intCast(std.math.clamp(center + s.r.intRangeAtMost(i32, -spread, spread), 0, 254));
        const sign: u32 = if (s.r.boolean()) 0x80000000 else 0;
        return @bitCast(sign | (e << 23) | s.r.uintLessThan(u32, 1 << 23));
    }

    fn fillA(s: Rand, t: *Tile, k: usize, spread: i32, center: i32, zero_p: f64, sub_p: f64) void {
        for (&t.a) |*row| for (row[0..k]) |*x| {
            x.* = s.word(spread, center, zero_p, sub_p);
        };
    }

    fn fillB(s: Rand, t: *Tile, k: usize, spread: i32, center: i32, zero_p: f64, sub_p: f64) void {
        for (t.b[0..k]) |*row| for (row) |*x| {
            x.* = s.word(spread, center, zero_p, sub_p);
        };
    }

    fn fillC(s: Rand, t: *Tile, spread: i32, center: i32, zero_p: f64) void {
        for (&t.c) |*row| for (row) |*x| {
            x.* = if (s.r.float(f64) < zero_p) 0 else s.f32w(spread, center);
        };
    }
};

/// A tile whose products pair off to cancel; `near` sets each accumulator to the negated exact sum, jittered.
fn cancelTile(s: Rand, kind: Kind, near: bool) Tile {
    const k = dpas.depth(kind);
    const half = k / 2;
    var t: Tile = .{};
    const c = layout(kind).center;
    const spread = s.r.intRangeAtMost(i32, 2, 29);
    s.fillA(&t, k, spread, c, 0.1, 0);
    s.fillB(&t, k, spread, c, 0.1, 0);
    var perm: [8]usize = undefined;
    for (perm[0..half], 0..) |*p, i| p.* = i;
    s.r.shuffle(usize, perm[0..half]);
    for (0..half) |i| {
        for (&t.a) |*row| row[i + half] = row[perm[i]] ^ signBit(kind);
        t.b[i + half] = t.b[perm[i]];
    }
    for (&t.a) |*row| for (row[0..k]) |*x| {
        if (s.r.float(f64) < 0.3) x.* ^= s.r.uintLessThan(u32, 8);
    };
    for (&t.c, 0..) |*row, m| for (row, 0..) |*x, n| {
        if (!near) {
            x.* = s.f32w(10, 127);
            continue;
        }
        var sum: f64 = 0;
        for (0..k) |i| sum += valueOf(kind, t.a[m][i]) * valueOf(kind, t.b[i][n]);
        var f: f32 = @floatCast(-sum);
        if (s.r.boolean()) f = @bitCast(@as(u32, @bitCast(f)) +% @as(u32, @bitCast(s.r.intRangeAtMost(i32, -3, 3))));
        x.* = if (std.math.isFinite(f)) f else 0;
    };
    return t;
}

/// A tile of one random class with every output element an independent draw of its row and column.
pub fn randomTile(kind: Kind, class: Class, r: std.Random) Tile {
    const s: Rand = .{ .r = r, .kind = kind };
    const k = dpas.depth(kind);
    const c = layout(kind).center;
    var t: Tile = .{};
    switch (class) {
        .tight => {
            const sp: i32 = if (kind == .f16) 2 else 3;
            s.fillA(&t, k, sp, c, 0.15, 0);
            s.fillB(&t, k, sp, c, 0.15, 0);
            s.fillC(&t, if (kind == .f16) 8 else 10, 127, 0);
        },
        .mid => {
            const sp: i32 = if (kind == .f16) r.intRangeAtMost(i32, 2, 11) else r.intRangeAtMost(i32, 4, 39);
            s.fillA(&t, k, sp, c, 0.15, 0);
            s.fillB(&t, k, sp, c, 0.15, 0);
            s.fillC(&t, 3 * sp, 127, 0.2);
        },
        .wide => {
            s.fillA(&t, k, 60, c, 0.15, 0);
            s.fillB(&t, k, 60, c, 0.15, 0);
            s.fillC(&t, 100, 127, 0);
        },
        .subnormal => {
            const ca = [_]i32{ 2, 5, 60, 200, 250 };
            const cb = [_]i32{ 2, 5, 60, 127, 200 };
            const cc = [_]i32{ 0, 2, 60, 127, 200 };
            const small = [_]i32{ 2, 4, 8, 15 };
            const sub = kind == .f16;
            s.fillA(&t, k, 4, if (sub) small[r.uintLessThan(usize, 4)] else ca[r.uintLessThan(usize, 5)], 0.15, 0.3);
            s.fillB(&t, k, 4, if (sub) small[r.uintLessThan(usize, 4)] else cb[r.uintLessThan(usize, 5)], 0.15, 0.3);
            s.fillC(&t, 4, if (sub) 100 else cc[r.uintLessThan(usize, 5)], 0.3);
        },
        .ties => {
            s.fillA(&t, k, 7, 127, 0.3, 0);
            s.fillB(&t, k, 7, 127, 0.3, 0);
            const drop = ~((@as(u32, 1) << (layout(kind).mant - 2)) - 1);
            for (&t.a) |*row| for (row[0..k]) |*x| {
                x.* &= drop;
            };
            for (t.b[0..k]) |*row| for (row) |*x| {
                x.* &= drop;
            };
            s.fillC(&t, 12, 127, 0.3);
        },
        .cancel => return cancelTile(s, kind, false),
        .nearcancel => return cancelTile(s, kind, true),
    }
    return t;
}

/// A tile whose 128 outputs are one case: row 0 of A, column 0 of B and C[0][0] copied to every row and column.
fn replicate(a: []const u32, b: []const u32, acc: f32) Tile {
    var t: Tile = .{};
    for (&t.a) |*row| @memcpy(row[0..a.len], a);
    for (b, 0..) |x, k| t.b[k] = @splat(x);
    t.c = @splat(@splat(acc));
    return t;
}

const P = [2]f32;
const Case = struct { pairs: []const P, acc: f32 };

fn u(comptime e: comptime_int) f32 {
    if (e >= -126) return @bitCast(@as(u32, e + 127) << 23);
    return @bitCast(@as(u32, 1) << (e + 149));
}

/// Hand-built cases, each aimed at one rule of the model; the host test requires each to catch a planted fault.
pub const handmade = [_]Case{
    .{ .pairs = &.{ .{ u(-12), u(-12) }, .{ u(-12), u(-12) } }, .acc = 1 },
    .{ .pairs = &.{ .{ u(-12), u(-12) }, .{ u(-14), u(-9) } }, .acc = 1 },
    .{ .pairs = &.{ .{ u(-12), u(-12) }, .{ u(-12), u(-12) }, .{ u(-13), u(-13) } }, .acc = 1 },
    .{ .pairs = &.{.{ u(-12), u(-12) }}, .acc = 1 },
    .{ .pairs = &.{.{ u(-12), u(-12) }}, .acc = 1 + u(-23) },
    .{ .pairs = &.{.{ u(-12), -u(-12) }}, .acc = 1 + u(-23) },
    .{ .pairs = &.{.{ u(-13), -u(-13) }}, .acc = 1 },
    .{ .pairs = &.{ .{ 1, 1 }, .{ -u(-14), u(-14) } }, .acc = 0 },
    .{ .pairs = &.{ .{ 1, 1 }, .{ -u(-14), u(-14) }, .{ -u(-14), u(-14) } }, .acc = 0 },
    .{ .pairs = &.{ .{ 1, 1 }, .{ -1, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ u(-14), -u(-14) } }, .acc = 0 },
    .{ .pairs = &.{ .{ 1024, 1 }, .{ -1016, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ u(-10), u(-11) }, .{ u(-11), u(-11) } }, .acc = 0 },
    .{ .pairs = &.{ .{ 1024, 1 }, .{ -1016, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ -u(-10), u(-11) }, .{ -u(-11), u(-11) } }, .acc = 0 },
    .{ .pairs = &.{.{ u(-60), 1 }}, .acc = -1 },
    .{ .pairs = &.{.{ u(60), u(60) }}, .acc = 3.4028235e38 },
    .{ .pairs = &.{.{ u(-80), -u(-80) }}, .acc = 0 },
    .{ .pairs = &.{.{ u(-70), -u(-70) }}, .acc = 0 },
    .{ .pairs = &.{.{ u(-70), -u(-70) }}, .acc = -u(-130) },
    .{ .pairs = &.{.{ 0, 1 }}, .acc = u(-130) },
    // a subnormal product keys the window (field exponent, or f16's own); it floors or keeps the other's half-ulp tail
    .{ .pairs = &.{ .{ u(-133), u(100) }, .{ 1.125 * u(-30), u(-27) } }, .acc = 0 },
    .{ .pairs = &.{ .{ u(-136), u(100) }, .{ 1.125 * u(-30), u(-30) } }, .acc = 0 },
    .{ .pairs = &.{ .{ u(-24), u(15) }, .{ 1.125 * u(-14), u(-19) } }, .acc = 0 },
    .{ .pairs = &.{ .{ 1.9921875, 1.9921875 }, .{ -1.9921875, 1.9921875 } }, .acc = u(-30) },
    .{ .pairs = &.{ .{ 1.9921875, 1.9921875 }, .{ u(-14), u(-14) }, .{ u(-14), u(-14) } }, .acc = -4 },
};

/// The `i`-th hand-built case as a tile of `kind`, or null when the format cannot hold its values.
pub fn handmadeTile(kind: Kind, i: usize) ?Tile {
    const case = handmade[i];
    var a: [16]u32 = @splat(0);
    var b: [16]u32 = @splat(0);
    for (case.pairs, 0..) |p, k| {
        a[k] = operand(kind, p[0]) orelse return null;
        b[k] = operand(kind, p[1]) orelse return null;
    }
    return replicate(a[0..dpas.depth(kind)], b[0..dpas.depth(kind)], case.acc);
}

/// The count of slot-sweep tiles: one large term at slot i and two small ones at the slots j < l, all distinct.
pub fn sweepCount(kind: Kind) usize {
    const k = dpas.depth(kind);
    return k * (k - 1) * (k - 2) / 2;
}

/// Sweep tile `n`: 1 and two products 2^-25 + 0.75 x 2^-27 (1.0 if floored beside the 1, 1 + 2^-23 if summed first).
pub fn sweepTile(kind: Kind, n: usize, negative: bool) Tile {
    const k = dpas.depth(kind);
    var at: usize = 0;
    for (0..k) |i| for (0..k) |j| for (j + 1..k) |l| {
        if (i == j or i == l) continue;
        if (at == n) {
            var a: [16]u32 = @splat(0);
            var b: [16]u32 = @splat(0);
            const small: f32 = 1.1875 * u(-12);
            a[i] = operand(kind, 1).?;
            b[i] = operand(kind, 1).?;
            for ([_]usize{ j, l }) |x| {
                a[x] = operand(kind, if (negative) -small else small).?;
                b[x] = operand(kind, u(-13)).?;
            }
            return replicate(a[0..k], b[0..k], 0);
        }
        at += 1;
    };
    unreachable;
}

/// Operand sets outside the finite model, as raw words (tf32 as f32 bits): NaNs, infinities, inf x 0, overflow.
pub const Special = struct { kind: Kind, a: []const u32, b: []const u32, acc: u32 };

pub const specials = [_]Special{
    .{ .kind = .bf16, .a = &.{0x3f80}, .b = &.{0x3f80}, .acc = 0x7f800000 },
    .{ .kind = .bf16, .a = &.{0x3f80}, .b = &.{0x3f80}, .acc = 0xff800000 },
    .{ .kind = .bf16, .a = &.{0x3f80}, .b = &.{0x3f80}, .acc = 0x7fa12345 },
    .{ .kind = .bf16, .a = &.{0}, .b = &.{0}, .acc = 0x7f800000 },
    .{ .kind = .bf16, .a = &.{ 0xff7f, 0xff7f }, .b = &.{ 0x7f7f, 0x7f7f }, .acc = 0x7f800000 },
    .{ .kind = .bf16, .a = &.{ 0x7f7f, 0x7f7f }, .b = &.{ 0x7f7f, 0x7f7f }, .acc = 0xff7fffff },
    .{ .kind = .bf16, .a = &.{ 0x7f7f, 0, 0, 0, 0xff7f }, .b = &.{ 0x7f7f, 0, 0, 0, 0x7f7f }, .acc = 0x3f800000 },
    .{ .kind = .bf16, .a = &.{0x7f80}, .b = &.{0x3f80}, .acc = 0 },
    .{ .kind = .bf16, .a = &.{0x7f80}, .b = &.{0x0001}, .acc = 0 },
    .{ .kind = .bf16, .a = &.{0x7f80}, .b = &.{0x8000}, .acc = 0 },
    .{ .kind = .bf16, .a = &.{ 0x7f80, 0xff80 }, .b = &.{ 0x3f80, 0x3f80 }, .acc = 0 },
    .{ .kind = .bf16, .a = &.{0x7f80}, .b = &.{0x3f80}, .acc = 0xff800000 },
    .{ .kind = .bf16, .a = &.{ 0xff80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7f80 }, .b = &.{ 0x3f80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3f80 }, .acc = 0 },
    .{ .kind = .bf16, .a = &.{ 0x7f80, 0xff7f }, .b = &.{ 0x3f80, 0x7f7f }, .acc = 0 },
    .{ .kind = .bf16, .a = &.{0x7f81}, .b = &.{0x3f80}, .acc = 0 },
    .{ .kind = .bf16, .a = &.{0xffa5}, .b = &.{0x3f80}, .acc = 0x3f800000 },
    .{ .kind = .bf16, .a = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7fc0 }, .b = &.{0}, .acc = 0x3f800000 },
    .{ .kind = .bf16, .a = &.{0x8000}, .b = &.{0x3f80}, .acc = 0x80000000 },
    .{ .kind = .bf16, .a = &@as([16]u32, @splat(0x8000)), .b = &@as([16]u32, @splat(0x3f80)), .acc = 0x80000000 },
    .{ .kind = .f16, .a = &@as([16]u32, @splat(0x8000)), .b = &@as([16]u32, @splat(0x3c00)), .acc = 0x80000000 },
    .{ .kind = .tf32, .a = &@as([8]u32, @splat(0x80000000)), .b = &@as([8]u32, @splat(0x3f800000)), .acc = 0x80000000 },
    .{ .kind = .f16, .a = &.{0x3c00}, .b = &.{0x3c00}, .acc = 0x7f800000 },
    .{ .kind = .f16, .a = &.{0x3c00}, .b = &.{0x3c00}, .acc = 0x7fa12345 },
    .{ .kind = .f16, .a = &.{0x7c00}, .b = &.{0x0001}, .acc = 0 },
    .{ .kind = .f16, .a = &.{0x7c00}, .b = &.{0}, .acc = 0 },
    .{ .kind = .f16, .a = &.{0xfc00}, .b = &.{0x3c00}, .acc = 0xff800000 },
    .{ .kind = .f16, .a = &.{0x7c01}, .b = &.{0x3c00}, .acc = 0 },
    .{ .kind = .tf32, .a = &.{0x3f800000}, .b = &.{0x3f800000}, .acc = 0x7f800000 },
    .{ .kind = .tf32, .a = &.{0x3f800000}, .b = &.{0x3f800000}, .acc = 0x7fa12345 },
    .{ .kind = .tf32, .a = &.{0x7f800000}, .b = &.{0x00002000}, .acc = 0 },
    .{ .kind = .tf32, .a = &.{0x7f800000}, .b = &.{0x00001fff}, .acc = 0 },
    .{ .kind = .tf32, .a = &.{0x7f800001}, .b = &.{0x3f800000}, .acc = 0 },
    .{ .kind = .tf32, .a = &.{0x7fa02000}, .b = &.{0x3f800000}, .acc = 0 },
    .{ .kind = .tf32, .a = &.{0x3f800000}, .b = &.{0x00001fff}, .acc = 0x80000000 },
    .{ .kind = .tf32, .a = &.{ 0x7f7fe000, 0x7f7fe000 }, .b = &.{ 0x7f7fe000, 0x7f7fe000 }, .acc = 0 },
};

/// The `i`-th special case as a tile, or null when it is another format's.
pub fn specialTile(kind: Kind, i: usize) ?Tile {
    const c = specials[i];
    if (c.kind != kind) return null;
    var a: [16]u32 = @splat(0);
    var b: [16]u32 = @splat(0);
    @memcpy(a[0..c.a.len], c.a);
    @memcpy(b[0..c.b.len], c.b);
    return replicate(a[0..dpas.depth(kind)], b[0..dpas.depth(kind)], @bitCast(c.acc));
}

/// The corpus in four parts: hand-built cases, the special values, the slot sweep and the seeded random classes.
pub const Part = enum { handmade, special, sweep, random };

/// The tiles of one part for `kind`; `per_class` random tiles of each class from `seed`.
pub fn tiles(a: std.mem.Allocator, kind: Kind, part: Part, seed: u64, per_class: usize) ![]Tile {
    var list: std.ArrayList(Tile) = .empty;
    errdefer list.deinit(a);
    switch (part) {
        .handmade => for (0..handmade.len) |i| if (handmadeTile(kind, i)) |t| try list.append(a, t),
        .special => for (0..specials.len) |i| if (specialTile(kind, i)) |t| try list.append(a, t),
        .sweep => for ([_]bool{ false, true }) |neg| for (0..sweepCount(kind)) |n| try list.append(a, sweepTile(kind, n, neg)),
        .random => for (classes(kind)) |class| {
            var prng = std.Random.DefaultPrng.init(seed ^ (@as(u64, @backingInt(kind)) << 8) ^ @backingInt(class));
            for (0..per_class) |_| try list.append(a, randomTile(kind, class, prng.random()));
        },
    }
    return list.toOwnedSlice(a);
}

/// The faults a part must catch on every format, the rules its cases aim at; the random classes have none of their own.
pub fn required(part: Part) []const dpas.Fault {
    return switch (part) {
        .handmade => &.{ .window26, .window28, .reverse, .truncate, .acc_first, .normalise_flip, .no_reset, .renormalise, .toward_zero, .ties_away, .flush, .chain, .exact },
        .special => &.{ .nan_positive, .ieee_zero },
        .sweep => &.{ .window26, .window28, .group_half, .group_double, .reverse },
        .random => &.{},
    };
}

/// The outputs of tile `i` of `part` that the vectors file records: one of a replicated tile, two of a random one.
pub fn recorded(part: Part, i: usize) []const [2]u8 {
    return switch (part) {
        .handmade, .special => &.{.{ 0, 0 }},
        .sweep => if (i % 32 == 0) &.{.{ 0, 0 }} else &.{},
        .random => &.{ .{ 0, 0 }, .{ 5, 9 } },
    };
}

/// Operand words of output (m, n) of `t`: row m of A and column n of B.
pub fn operands(kind: Kind, t: *const Tile, m: usize, n: usize, a: *[16]u32, b: *[16]u32) void {
    for (0..dpas.depth(kind)) |i| {
        a[i] = t.a[m][i];
        b[i] = t.b[i][n];
    }
}
