//! tf-xpu-xe2-dpas-check: the card's DPAS against the host model, byte for byte, and every planted fault caught.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const image = @import("xe2_dpas_image");

const dpas = core.xe2_dpas;
const cases = core.xe2_dpas_cases;
const Kind = dpas.Kind;
const Part = cases.Part;
const Tile = cases.Tile;
const Fault = dpas.Fault;
const faults = std.enums.values(Fault);

const batch = 2048;

/// A format's outputs and model mismatches, and per part the first output on which each planted fault left the card.
const Stats = struct {
    outputs: usize = 0,
    bad: usize = 0,
    caught: [4][faults.len]usize = @splat(@splat(0)),

    fn has(st: *const Stats, part: Part, fault: Fault) bool {
        return st.caught[@backingInt(part)][@backingInt(fault)] > 0;
    }
};

const Options = struct { tiles: usize = 4000, seed: u64 = 0x58e2, emit: ?[]const u8 = null, device: ?u32 = null };

const Dev = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    s: xpu.Stream,
    mod: xpu.Module,
};

fn symbol(kind: Kind) [:0]const u8 {
    return switch (kind) {
        .bf16 => "xe2_dpas_bf16",
        .f16 => "xe2_dpas_f16",
        .tf32 => "xe2_dpas_tf32",
    };
}

fn bits(x: f32) u32 {
    return @bitCast(x);
}

const Out = std.Io.Writer;

fn emitLine(w: *Out, kind: Kind, part: Part, t: *const Tile, m: usize, n: usize, got: f32) !void {
    const k = dpas.depth(kind);
    try w.print("{t} {t} ", .{ part, kind });
    for (0..k) |i| try w.print("{x}{s}", .{ t.a[m][i], if (i + 1 < k) "," else " " });
    for (0..k) |i| try w.print("{x}{s}", .{ t.b[i][n], if (i + 1 < k) "," else " " });
    try w.print("{x} {x}\n", .{ bits(t.c[m][n]), bits(got) });
}

/// Output (m, n) of `t` against the model; a fault is tried until its part has caught it once (one is enough).
fn compare(kind: Kind, part: Part, t: *const Tile, m: usize, n: usize, got: f32, st: *Stats) !void {
    const k = dpas.depth(kind);
    var a: [16]u32 = undefined;
    var b: [16]u32 = undefined;
    cases.operands(kind, t, m, n, &a, &b);
    const want = try dpas.mad(kind, a[0..k], b[0..k], t.c[m][n]);
    st.outputs += 1;
    if (bits(got) != bits(want)) {
        if (st.bad < 3) std.debug.print("MISMATCH {t} {t} ({d},{d}): card {x}, model {x}\n", .{ kind, part, m, n, bits(got), bits(want) });
        st.bad += 1;
    }
    for (faults, 0..) |fault, fi| {
        if (st.caught[@backingInt(part)][fi] > 0) continue;
        const model = dpas.faulty(kind, fault) orelse continue;
        if (bits(try dpas.madModel(kind, model, a[0..k], b[0..k], t.c[m][n])) != bits(got)) st.caught[@backingInt(part)][fi] = st.outputs;
    }
}

fn run(a: std.mem.Allocator, dev: *Dev, kind: Kind, part: Part, tiles: []const Tile, st: *Stats, w: ?*Out) !void {
    var kernel = try dev.mod.kernel(symbol(kind), .{ 16, 1, 1 });
    defer kernel.deinit();
    const kd = dpas.depth(kind);
    const word: usize = if (kind == .tf32) 4 else 2;
    var ba = try xpu.DeviceBuffer.alloc(&dev.ctx, batch * 8 * kd * word);
    defer ba.free();
    var bb = try xpu.DeviceBuffer.alloc(&dev.ctx, batch * kd * 16 * word);
    defer bb.free();
    var bc = try xpu.DeviceBuffer.alloc(&dev.ctx, batch * 128 * 4);
    defer bc.free();
    var bo = try xpu.DeviceBuffer.alloc(&dev.ctx, batch * 128 * 4);
    defer bo.free();
    const host_a = try a.alloc(u8, ba.len);
    defer a.free(host_a);
    const host_b = try a.alloc(u8, bb.len);
    defer a.free(host_b);
    const host_c = try a.alloc(f32, batch * 128);
    defer a.free(host_c);
    const host_o = try a.alloc(f32, batch * 128);
    defer a.free(host_o);
    var at: usize = 0;
    while (at < tiles.len) : (at += batch) {
        const chunk = tiles[at..@min(at + batch, tiles.len)];
        for (chunk, 0..) |*t, ti| {
            for (0..8) |m| for (0..kd) |i| put(host_a, kind, (ti * 8 + m) * kd + i, t.a[m][i]);
            for (0..kd) |i| for (0..16) |n| put(host_b, kind, (ti * kd + i) * 16 + n, t.b[i][n]);
            for (0..8) |m| for (0..16) |n| {
                host_c[(ti * 8 + m) * 16 + n] = t.c[m][n];
            };
        }
        try ba.upload(dev.s, 0, host_a[0 .. chunk.len * 8 * kd * word]);
        try bb.upload(dev.s, 0, host_b[0 .. chunk.len * kd * 16 * word]);
        try bc.upload(dev.s, 0, std.mem.sliceAsBytes(host_c[0 .. chunk.len * 128]));
        try xpu.launch.dispatch(kernel, dev.s, .{ @intCast(chunk.len), 1, 1 }, .{ ba, bb, bc, bo });
        try dev.s.synchronize();
        try bo.download(dev.s, 0, std.mem.sliceAsBytes(host_o[0 .. chunk.len * 128]));
        for (chunk, 0..) |*t, ti| {
            for (0..8) |m| for (0..16) |n| try compare(kind, part, t, m, n, host_o[(ti * 8 + m) * 16 + n], st);
            if (w) |out| for (cases.recorded(part, at + ti)) |mn| try emitLine(out, kind, part, t, mn[0], mn[1], host_o[(ti * 8 + mn[0]) * 16 + mn[1]]);
        }
    }
}

fn put(buf: []u8, kind: Kind, i: usize, v: u32) void {
    if (kind == .tf32) std.mem.writeInt(u32, buf[i * 4 ..][0..4], v, .little) else std.mem.writeInt(u16, buf[i * 2 ..][0..2], @intCast(v), .little);
}

fn usage() u8 {
    std.debug.print("usage: tf-xpu-xe2-dpas-check [--device N] [--tiles N] [--seed N] [--emit FILE]\n", .{});
    return 2;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var o: Options = .{};
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--emit")) {
            o.emit = v;
        } else if (std.mem.eql(u8, args[i], "--device")) {
            o.device = std.fmt.parseInt(u32, v, 10) catch return usage();
        } else if (std.mem.eql(u8, args[i], "--tiles")) {
            o.tiles = std.fmt.parseInt(usize, v, 10) catch return usage();
        } else if (std.mem.eql(u8, args[i], "--seed")) {
            o.seed = std.fmt.parseInt(u64, v, 0) catch return usage();
        } else return usage();
    }
    if (i != args.len) return usage();
    var dev: Dev = undefined;
    dev.drv = try xpu.Driver.open();
    defer dev.drv.close();
    dev.ctx = try xpu.Context.init(&dev.drv, o.device);
    defer dev.ctx.deinit();
    dev.s = try xpu.Stream.init(&dev.ctx);
    defer dev.s.deinit();
    dev.mod = try xpu.Module.loadNative(&dev.ctx, &image.zebin);
    defer dev.mod.unload();
    var out_buf: [1 << 16]u8 = undefined;
    var file: ?std.Io.File = null;
    defer if (file) |f| f.close(init.io);
    var fw: std.Io.File.Writer = undefined;
    var w: ?*Out = null;
    if (o.emit) |path| {
        file = try std.Io.Dir.cwd().createFile(init.io, path, .{});
        fw = file.?.writer(init.io, &out_buf);
        w = &fw.interface;
        try w.?.print("# The card's own DPAS on the corpus of core/contract/xe2_dpas_cases.zig: {s}.\n", .{dev.ctx.name()});
        try w.?.print("# Made by tf-xpu-xe2-dpas-check --emit; part, kind, A and B words (tf32 as f32 bits), accumulator, result.\n", .{});
        try w.?.print("seed {x} tiles {d}\n", .{ o.seed, o.tiles });
    }
    var uncaught: usize = 0;
    var per: [3]Stats = .{ .{}, .{}, .{} };
    for (std.enums.values(Kind), &per) |kind, *st| {
        for (std.enums.values(Part)) |part| {
            const list = try cases.tiles(a, kind, part, o.seed, o.tiles);
            defer a.free(list);
            try run(a, &dev, kind, part, list, st, w);
            for (cases.required(part)) |fault| if (!st.has(part, fault)) {
                std.debug.print("NOT CAUGHT {t}: the {t} part never separates the planted {t}\n", .{ kind, part, fault });
                uncaught += 1;
            };
        }
        for (faults) |fault| {
            if (dpas.faulty(kind, fault) == null) continue;
            var required = false;
            for (std.enums.values(Part)) |part| for (cases.required(part)) |f| {
                required = required or f == fault;
            };
            if (required) continue; // a required fault missed is already counted, once, above
            var any = false;
            for (std.enums.values(Part)) |part| any = any or st.has(part, fault);
            if (!any) {
                std.debug.print("NOT CAUGHT {t}: no part separates the planted {t}\n", .{ kind, fault });
                uncaught += 1;
            }
        }
    }
    if (w) |x| try x.flush();
    const bad = per[0].bad + per[1].bad + per[2].bad;
    std.debug.print("SUMMARY xe2-dpas {s}: bf16 {d}, f16 {d}, tf32 {d} outputs: {d} mismatches; planted faults not caught: {d}\n", .{ dev.ctx.name(), per[0].outputs, per[1].outputs, per[2].outputs, bad, uncaught });
    for (std.enums.values(Kind), &per) |kind, *st| {
        std.debug.print("  {t}, the parts that caught each fault (h handmade, p special, s sweep, r random):", .{kind});
        for (faults) |fault| {
            if (dpas.faulty(kind, fault) == null) continue;
            std.debug.print(" {t}=", .{fault});
            for (std.enums.values(Part), "hpsr") |part, c| std.debug.print("{c}", .{if (st.has(part, fault)) c else '.'});
        }
        std.debug.print("\n", .{});
    }
    return if (bad != 0 or uncaught != 0) 1 else 0;
}
