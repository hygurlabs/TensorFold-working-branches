//! The Xe2 DPAS model against the card's results on its corpus, and every planted fault caught by those results.
const std = @import("std");
const dpas = @import("xe2_dpas.zig");
const cases = @import("xe2_dpas_cases.zig");

const vectors = @embedFile("xe2_dpas_vectors.txt");
const gpa = std.testing.allocator;

fn bits(x: f32) u32 {
    return @bitCast(x);
}

fn words(text: []const u8, out: *[16]u32) !usize {
    var it = std.mem.splitScalar(u8, text, ',');
    var n: usize = 0;
    while (it.next()) |w| : (n += 1) out[n] = try std.fmt.parseInt(u32, w, 16);
    return n;
}

const Vector = struct { part: cases.Part, kind: dpas.Kind, a: [16]u32, b: [16]u32, acc: f32, out: u32 };

fn parse(line: []const u8) !Vector {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    var v: Vector = undefined;
    v.part = std.meta.stringToEnum(cases.Part, it.next().?) orelse return error.BadPart;
    v.kind = std.meta.stringToEnum(dpas.Kind, it.next().?) orelse return error.BadKind;
    try std.testing.expectEqual(dpas.depth(v.kind), try words(it.next().?, &v.a));
    try std.testing.expectEqual(dpas.depth(v.kind), try words(it.next().?, &v.b));
    v.acc = @bitCast(try std.fmt.parseInt(u32, it.next().?, 16));
    v.out = try std.fmt.parseInt(u32, it.next().?, 16);
    return v;
}

/// The file's records after its comments and its `seed S tiles N` line, which it returns.
fn records(lines: *std.mem.TokenIterator(u8, .scalar)) !struct { seed: u64, tiles: usize } {
    while (lines.next()) |line| {
        if (line[0] == '#') continue;
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        try std.testing.expectEqualStrings("seed", it.next().?);
        const seed = try std.fmt.parseInt(u64, it.next().?, 16);
        try std.testing.expectEqualStrings("tiles", it.next().?);
        return .{ .seed = seed, .tiles = try std.fmt.parseInt(usize, it.next().?, 10) };
    }
    return error.NoHeader;
}

fn all() !std.ArrayList(Vector) {
    var lines = std.mem.tokenizeScalar(u8, vectors, '\n');
    _ = try records(&lines);
    var list: std.ArrayList(Vector) = .empty;
    while (lines.next()) |line| try list.append(gpa, try parse(line));
    return list;
}

fn result(v: *const Vector, model: dpas.Model) !u32 {
    const k = dpas.depth(v.kind);
    return bits(try dpas.madModel(v.kind, model, v.a[0..k], v.b[0..k], v.acc));
}

test "xe2 dpas: the vectors are this corpus's operands, in order, and the model gives the card's every result" {
    var lines = std.mem.tokenizeScalar(u8, vectors, '\n');
    const h = try records(&lines);
    var n: usize = 0;
    for (std.enums.values(dpas.Kind)) |kind| for (std.enums.values(cases.Part)) |part| {
        const list = try cases.tiles(gpa, kind, part, h.seed, h.tiles);
        defer gpa.free(list);
        for (list, 0..) |*t, i| for (cases.recorded(part, i)) |mn| {
            const v = try parse(lines.next() orelse return error.VectorsEndEarly);
            var a: [16]u32 = undefined;
            var b: [16]u32 = undefined;
            cases.operands(kind, t, mn[0], mn[1], &a, &b);
            const k = dpas.depth(kind);
            try std.testing.expectEqual(part, v.part);
            try std.testing.expectEqual(kind, v.kind);
            try std.testing.expectEqualSlices(u32, a[0..k], v.a[0..k]);
            try std.testing.expectEqualSlices(u32, b[0..k], v.b[0..k]);
            try std.testing.expectEqual(bits(t.c[mn[0]][mn[1]]), bits(v.acc));
            try std.testing.expectEqual(v.out, try result(&v, dpas.measured));
            n += 1;
        };
    };
    try std.testing.expectEqual(@as(?[]const u8, null), lines.next());
    try std.testing.expect(n >= 500);
}

test "xe2 dpas: on the card's results every planted fault is caught, each part its own, each hand-built case one" {
    var list = try all();
    defer list.deinit(gpa);
    for (std.enums.values(dpas.Kind)) |kind| for (std.enums.values(dpas.Fault)) |fault| {
        const model = dpas.faulty(kind, fault) orelse continue;
        var parts: [4]bool = @splat(false);
        for (list.items) |*v| if (v.kind == kind and try result(v, model) != v.out) {
            parts[@backingInt(v.part)] = true;
        };
        errdefer std.debug.print("{t} {t}: caught by {any}\n", .{ kind, fault, parts });
        try std.testing.expect(std.mem.indexOfScalar(bool, &parts, true) != null);
        for (std.enums.values(cases.Part)) |part| for (cases.required(part)) |f| if (f == fault) try std.testing.expect(parts[@backingInt(part)]);
    };
    for (list.items) |*v| if (v.part == .handmade) {
        var caught = false;
        for (std.enums.values(dpas.Fault)) |fault| if (dpas.faulty(v.kind, fault)) |model| {
            caught = caught or try result(v, model) != v.out;
        };
        errdefer std.debug.print("{t} handmade {any}: no fault caught\n", .{ v.kind, v.a[0..dpas.depth(v.kind)] });
        try std.testing.expect(caught);
    };
}

test "xe2 dpas: two products of 2^-24 on 1.0 add up in one group, where a serial chain rounds each away" {
    const q = cases.operand(.bf16, std.math.ldexp(@as(f32, 1), -12)).?;
    var x: [16]u32 = @splat(0);
    x[0] = q;
    x[1] = q;
    try std.testing.expectEqual(@as(u32, 0x3f800001), bits(try dpas.mad(.bf16, &x, &x, 1)));
    try std.testing.expectEqual(@as(u32, 0x3f800000), bits(try dpas.madModel(.bf16, dpas.faulty(.bf16, .chain).?, &x, &x, 1)));
}

test "xe2 dpas: NaN, inf x 0 and inf - inf give the one NaN; an infinity beats every finite term" {
    const one: [16]u32 = .{0x3f80} ++ @as([15]u32, @splat(0));
    var inf = one;
    inf[0] = 0x7f80;
    var zero: [16]u32 = @splat(0);
    try std.testing.expectEqual(dpas.nan, bits(try dpas.mad(.bf16, &inf, &zero, 1)));
    try std.testing.expectEqual(dpas.nan, bits(try dpas.mad(.bf16, &inf, &one, -std.math.inf(f32))));
    try std.testing.expectEqual(dpas.nan, bits(try dpas.mad(.bf16, &one, &one, std.math.nan(f32))));
    try std.testing.expectEqual(bits(std.math.inf(f32)), bits(try dpas.mad(.bf16, &inf, &one, -3.4028235e38)));
    zero[0] = 0x7f7f;
    var big = zero;
    big[0] = 0xff7f;
    try std.testing.expectEqual(bits(std.math.inf(f32)), bits(try dpas.mad(.bf16, &big, &zero, std.math.inf(f32))));
}

test "xe2 dpas: a K-blocked dot is the chain of its calls, and a block that overflows carries its infinity on" {
    var a: [32]u32 = undefined;
    var b: [32]u32 = undefined;
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = cases.operand(.bf16, @floatFromInt(i + 1)).?;
        y.* = cases.operand(.bf16, 0.5).?;
    }
    const first = try dpas.mad(.bf16, a[0..16], b[0..16], 0.25);
    try std.testing.expectEqual(bits(try dpas.mad(.bf16, a[16..], b[16..], first)), bits(try dpas.dot(.bf16, &a, &b, 0.25)));
    @memset(&a, 0);
    @memset(&b, 0);
    a[0] = 0x7f7f;
    b[0] = 0x7f7f;
    try std.testing.expectEqual(bits(std.math.inf(f32)), bits(try dpas.dot(.bf16, &a, &b, 0)));
    a[16] = 0xff7f;
    b[16] = 0x7f7f;
    try std.testing.expectEqual(bits(std.math.inf(f32)), bits(try dpas.dot(.bf16, &a, &b, 0)));
}

test "xe2 dpas: wrong lengths are refused" {
    const z: [16]u32 = @splat(0);
    try std.testing.expectError(error.BadLength, dpas.mad(.bf16, z[0..8], z[0..8], 0));
    try std.testing.expectError(error.BadLength, dpas.dot(.tf32, z[0..12], z[0..12], 0));
}
