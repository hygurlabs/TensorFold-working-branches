//! Intel Xe2 DPAS (bf16, f16, tf32) as a host function: a windowed fixed-point sum of products, the accumulator last.
const std = @import("std");

pub const Kind = enum { bf16, f16, tf32 };

pub const Error = error{BadLength};

/// Bits of the window below the largest product exponent that a group's products keep.
pub const window = 27;

/// Every NaN result, whatever NaNs went in: the sign set, quiet, no payload.
pub const nan: u32 = 0xffc00000;

/// The rules of one DPAS call: `measured` is the card's; each other value is a planted fault (`faulty`).
pub const Model = struct {
    window: u6 = window,
    /// products a group sums; null: the format's, 4 for bf16 and f16, 2 for tf32
    group: ?usize = null,
    /// groups taken from the last slot down instead of the first up
    reverse: bool = false,
    /// products and the running sum floored into the window; false: cut toward zero
    floor: bool = true,
    /// the accumulator added after the groups with the one rounding; false: it enters the window first
    acc_last: bool = true,
    /// subnormal operands normalised for the window exponent; null: the format's, f16 only
    normalise: ?bool = null,
    /// a running sum that cancels to zero restarts the window at the next group; false: it keeps its exponent
    reset: bool = true,
    /// true: after a partial cancellation the window narrows to the running sum's own exponent
    renormalise: bool = false,
    /// the result rounded to nearest even; false: toward zero, overflow to the largest finite
    nearest: bool = true,
    /// true: a result below the smallest normal is a zero of its sign
    flush: bool = false,
    /// true: tf32 operands rounded to nearest even instead of cut to ten mantissa bits
    tf32_nearest: bool = false,
    /// false: a NaN result is the positive quiet NaN
    nan_negative: bool = true,
    /// true: a tie of the final rounding goes away from zero
    ties_away: bool = false,
    /// true: a zero result takes IEEE's sign (-0 when every term is a negative zero); the card's is +0
    ieee_zero: bool = false,
    /// true: the products added one at a time into the accumulator by fused f32 multiply-adds, in slot order
    chain: bool = false,
    /// true: the products and the accumulator summed exactly, rounded once
    exact: bool = false,
};

pub const measured: Model = .{};

const Fmt = struct { mant: u5, exp_bits: u4, bias: i32, k: usize, group: usize, normalise: bool };

fn fmt(kind: Kind) Fmt {
    return switch (kind) {
        .bf16 => .{ .mant = 7, .exp_bits = 8, .bias = 127, .k = 16, .group = 4, .normalise = false },
        .f16 => .{ .mant = 10, .exp_bits = 5, .bias = 15, .k = 16, .group = 4, .normalise = true },
        .tf32 => .{ .mant = 10, .exp_bits = 8, .bias = 127, .k = 8, .group = 2, .normalise = false },
    };
}

/// Products one DPAS call sums (the reduction depth of one instruction).
pub fn depth(kind: Kind) usize {
    return fmt(kind).k;
}

/// Products a group of `kind` sums before the window moves on.
pub fn group(kind: Kind) usize {
    return fmt(kind).group;
}

/// A planted fault: one rule of the model changed to a plausible other hardware's.
pub const Fault = enum { window26, window28, group_half, group_double, reverse, truncate, acc_first, normalise_flip, no_reset, renormalise, toward_zero, ties_away, flush, ieee_zero, tf32_nearest, nan_positive, chain, exact };

/// The model with `fault` planted, or null when the fault does not apply to `kind` (`tf32_nearest` outside tf32).
pub fn faulty(kind: Kind, fault: Fault) ?Model {
    const f = fmt(kind);
    return switch (fault) {
        .window26 => .{ .window = 26 },
        .window28 => .{ .window = 28 },
        .group_half => .{ .group = f.group / 2 },
        .group_double => .{ .group = f.group * 2 },
        .reverse => .{ .reverse = true },
        .truncate => .{ .floor = false },
        .acc_first => .{ .acc_last = false },
        .normalise_flip => .{ .normalise = !f.normalise },
        .no_reset => .{ .reset = false },
        .renormalise => .{ .renormalise = true },
        .toward_zero => .{ .nearest = false },
        .ties_away => .{ .ties_away = true },
        .flush => .{ .flush = true },
        .ieee_zero => .{ .ieee_zero = true },
        .tf32_nearest => if (kind == .tf32) .{ .tf32_nearest = true } else null,
        .nan_positive => .{ .nan_negative = false },
        .chain => .{ .chain = true },
        .exact => .{ .exact = true },
    };
}

/// An operand word as the unit reads it: bf16 and f16 as they are, a tf32 operand's f32 bits cut to ten mantissa bits.
fn decode(kind: Kind, model: Model, w: u32) u32 {
    if (kind != .tf32) return w;
    if (!model.tf32_nearest or (w >> 23) & 255 == 255) return w >> 13;
    const low = w & 0x1fff;
    var q = w >> 13;
    if (low > 0x1000 or (low == 0x1000 and q & 1 == 1)) q += 1; // a carry into the exponent is the next binade
    return q;
}

const Class = enum { zero, finite, inf, nan };

fn classOf(f: Fmt, w: u32) Class {
    const e = (w >> f.mant) & ((@as(u32, 1) << f.exp_bits) - 1);
    const m = w & ((@as(u32, 1) << f.mant) - 1);
    if (e == (@as(u32, 1) << f.exp_bits) - 1) return if (m == 0) .inf else .nan;
    return if (e == 0 and m == 0) .zero else .finite;
}

fn negative(f: Fmt, w: u32) bool {
    return (w >> (@as(u5, f.mant) + f.exp_bits)) & 1 == 1;
}

const Product = struct { nz: bool, neg: bool, m: i64, e: i32, ew: i32 };

fn msb(x: i64) i32 {
    return 63 - @as(i32, @clz(@as(u64, @intCast(x))));
}

/// A finite product: its integer mantissa `m`, its value m * 2^(e - 2 mant), `ew` the exponent its window keys on.
fn product(f: Fmt, normalise: bool, ua: u32, ub: u32) Product {
    const exp_mask: u32 = (@as(u32, 1) << f.exp_bits) - 1;
    const man_mask: u32 = (@as(u32, 1) << f.mant) - 1;
    const ea = (ua >> f.mant) & exp_mask;
    const eb = (ub >> f.mant) & exp_mask;
    var ma: i64 = ua & man_mask;
    var mb: i64 = ub & man_mask;
    if (ea != 0) ma |= @as(i64, 1) << f.mant;
    if (eb != 0) mb |= @as(i64, 1) << f.mant;
    const fa: i32 = @max(@as(i32, @intCast(ea)), 1);
    const fb: i32 = @max(@as(i32, @intCast(eb)), 1);
    const e = fa + fb - 2 * f.bias;
    var ew = e;
    if (normalise) {
        const la: i32 = if (ea != 0) @intCast(ea) else if (ma != 0) 1 - (@as(i32, f.mant) - msb(ma)) else 1;
        const lb: i32 = if (eb != 0) @intCast(eb) else if (mb != 0) 1 - (@as(i32, f.mant) - msb(mb)) else 1;
        ew = la + lb - 2 * f.bias;
    }
    return .{ .nz = ma != 0 and mb != 0, .neg = negative(f, ua) != negative(f, ub), .m = ma * mb, .e = e, .ew = ew };
}

/// `v` shifted right by `n` bits, floored (two's complement) or cut toward zero.
fn shiftDown(v: i64, n: i32, floor: bool) i64 {
    if (!floor and v < 0) return -shiftDown(-v, n, true);
    if (n >= 63) return if (v < 0) -1 else 0;
    return v >> @intCast(n);
}

/// `v * 2^lo` as an f32: gradual underflow, signed zeros kept, to nearest even or toward zero as `model` says.
fn round(comptime T: type, v: T, lo: i32, model: Model) f32 {
    const U = @Int(.unsigned, @bitSizeOf(T));
    if (v == 0) return 0.0;
    const neg = v < 0;
    const a: U = if (neg) @intCast(-v) else @intCast(v);
    const n: i32 = @bitSizeOf(T) - @as(i32, @clz(a));
    const top = n - 1 + lo;
    if (model.flush and top < -126) return if (neg) -0.0 else 0.0;
    const ulp: i32 = if (top < -126) -149 else top - 23;
    const sh = ulp - lo;
    if (sh >= @bitSizeOf(T)) return if (neg) -0.0 else 0.0; // under half the smallest subnormal: a < 2^(bits - 1)
    const m: U = if (sh <= 0) a << @intCast(-sh) else blk: {
        const s: std.math.Log2Int(U) = @intCast(sh);
        var q = a >> s;
        const rem = a & ((@as(U, 1) << s) - 1);
        const half = @as(U, 1) << (s - 1);
        if (model.nearest and (rem > half or (rem == half and (model.ties_away or q & 1 == 1)))) q += 1;
        break :blk q;
    };
    var d = std.math.ldexp(@as(f64, @floatFromInt(@as(u32, @intCast(m)))), ulp); // m: 24 bits, 25 on a carry
    if (d >= 0x1p128) d = if (model.nearest) std.math.inf(f64) else std.math.floatMax(f32);
    const f: f32 = @floatCast(d);
    return if (neg) -f else f;
}

/// The planted `chain` fault: each product fused into the running f32 in slot order, the accumulator first.
fn chained(kind: Kind, ua: []const u32, ub: []const u32, acc: f32) f32 {
    var r = acc;
    for (ua, ub) |x, y| r = @mulAdd(f32, value(kind, x), value(kind, y), r);
    return r;
}

/// The planted `exact` fault: every term exact on one i1024 from 2^-320 (the least product is 2^-272), rounded once.
fn summed(f: Fmt, p: []const Product, acc_m: i64, acc_lo: i32, model: Model) f32 {
    const lo = -320;
    var s: i1024 = @as(i1024, acc_m) << @intCast(acc_lo - lo);
    for (p) |x| if (x.nz) {
        const v: i1024 = if (x.neg) -@as(i1024, x.m) else x.m;
        s += v << @intCast(x.e - 2 * @as(i32, f.mant) - lo);
    };
    return round(i1024, s, lo, model);
}

/// The exact value of a decoded operand word (tf32 already cut).
fn value(kind: Kind, w: u32) f32 {
    return switch (kind) {
        .bf16 => @bitCast(w << 16),
        .f16 => @as(f16, @bitCast(@as(u16, @intCast(w)))),
        .tf32 => @bitCast(w << 13),
    };
}

/// One DPAS call under `model`: `a` and `b` hold `depth(kind)` operand words (bf16 or f16 bits, tf32 as f32 bits).
pub fn madModel(kind: Kind, model: Model, a: []const u32, b: []const u32, acc: f32) Error!f32 {
    const f = fmt(kind);
    if (a.len != f.k or b.len != f.k) return error.BadLength;
    var ua: [16]u32 = undefined;
    var ub: [16]u32 = undefined;
    for (0..f.k) |i| {
        ua[i] = decode(kind, model, a[i]);
        ub[i] = decode(kind, model, b[i]);
    }
    // IEEE first: a NaN, inf x 0 or inf - inf is NaN; an infinity otherwise wins over every finite term
    const nan_result: f32 = @bitCast(if (model.nan_negative) nan else @as(u32, 0x7fc00000));
    var plus_inf = false;
    var minus_inf = false;
    for (ua[0..f.k], ub[0..f.k]) |x, y| {
        const cx = classOf(f, x);
        const cy = classOf(f, y);
        if (cx == .nan or cy == .nan) return nan_result;
        if (cx != .inf and cy != .inf) continue;
        if (cx == .zero or cy == .zero) return nan_result;
        if (negative(f, x) != negative(f, y)) minus_inf = true else plus_inf = true;
    }
    if (std.math.isNan(acc)) return nan_result;
    if (std.math.isInf(acc)) {
        if (acc > 0) plus_inf = true else minus_inf = true;
    }
    if (plus_inf and minus_inf) return nan_result;
    if (plus_inf) return std.math.inf(f32);
    if (minus_inf) return -std.math.inf(f32);
    if (model.chain) return chained(kind, ua[0..f.k], ub[0..f.k], acc);
    const win: i32 = model.window;
    const fr: i32 = 2 * @as(i32, f.mant);
    const normalise = model.normalise orelse f.normalise;
    var p: [16]Product = undefined;
    for (0..f.k) |i| p[i] = product(f, normalise, ua[i], ub[i]);
    const bits: u32 = @bitCast(acc);
    const ea: i32 = @intCast((bits >> 23) & 255);
    var ma: i64 = bits & 0x7fffff;
    if (ea != 0) ma |= 0x800000;
    const lea: i32 = if (ea != 0) ea - 150 else -149;
    if (bits >> 31 == 1) ma = -ma;
    if (model.exact) return summed(f, p[0..f.k], ma, lea, model);
    var s: i64 = 0;
    var ereg: i32 = 0;
    var have = false;
    if (!model.acc_last and ma != 0) { // the accumulator as the window's first term, its leading bit at the window top
        ereg = lea + 23;
        s = ma << @intCast(win - 23);
        have = true;
    }
    const size = model.group orelse f.group;
    for (0..f.k / size) |gi| {
        const g = (if (model.reverse) f.k / size - 1 - gi else gi) * size;
        var em: i32 = std.math.minInt(i32);
        var any = false;
        for (p[g..][0..size]) |x| if (x.nz) {
            any = true;
            em = @max(em, x.ew);
        };
        if (!any) continue;
        if (have) {
            em = @max(em, ereg);
            if (em > ereg) s = shiftDown(s, em - ereg, model.floor);
        }
        for (p[g..][0..size]) |x| if (x.nz) {
            const v: i64 = if (x.neg) -x.m else x.m;
            const sh = x.e - em + win - fr;
            s += if (sh >= 0) v << @intCast(sh) else shiftDown(v, -sh, model.floor);
        };
        ereg = em;
        have = s != 0 or !model.reset; // the card closes the window when the sum cancels
        if (model.renormalise and s != 0) { // the sum's top bit where a product's would sit: [2^e, 2^(e + 2))
            const e = ereg - win + msb(if (s < 0) -s else s) - 1;
            if (e < ereg) {
                s <<= @intCast(ereg - e);
                ereg = e;
            }
        }
    }
    if (!model.acc_last) return round(i64, s, ereg - win, model);
    if (!have or s == 0) { // a zero sum has no magnitude, even under a window kept open (`no_reset`)
        if (ma != 0) return if (model.flush and ea == 0) (if (ma < 0) -0.0 else 0.0) else acc;
        var all_negative = bits >> 31 == 1;
        for (p[0..f.k]) |x| all_negative = all_negative and x.neg;
        return if (model.ieee_zero and all_negative) -0.0 else 0.0;
    }
    const lq = ereg - win;
    if (ma == 0) return round(i64, s, lq, model);
    const lo = @min(lea, lq);
    if (@max(lea, lq) - lo > 60) { // the smaller term only decides a tie: a sticky bit below the larger one
        if (lea < lq) return round(i128, (@as(i128, s) << 60) + (if (ma < 0) @as(i128, -1) else 1), lq - 60, model);
        return round(i128, (@as(i128, ma) << 60) + (if (s < 0) @as(i128, -1) else 1), lea - 60, model);
    }
    return round(i128, (@as(i128, ma) << @intCast(lea - lo)) + (@as(i128, s) << @intCast(lq - lo)), lo, model);
}

/// One DPAS call as the card computes it.
pub fn mad(kind: Kind, a: []const u32, b: []const u32, acc: f32) Error!f32 {
    return madModel(kind, measured, a, b, acc);
}

/// A K-blocked dot product: DPAS calls of `depth(kind)` products in order, each taking the last result as accumulator.
pub fn dot(kind: Kind, a: []const u32, b: []const u32, acc: f32) Error!f32 {
    const k = depth(kind);
    if (a.len != b.len or a.len % k != 0) return error.BadLength;
    var r = acc;
    var at: usize = 0;
    while (at < a.len) : (at += k) r = try mad(kind, a[at..][0..k], b[at..][0..k], r);
    return r;
}
