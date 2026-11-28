//! The Dev that Inkling 2's attention plan (xpu_attention.zig Attention) takes, over the xpu package's own calls.
const xpu = @import("xpu");

pub const Context = xpu.Context;
pub const Module = xpu.Module;
pub const Kernel = xpu.Kernel;
pub const Stream = xpu.Stream;

pub fn load(ctx: *const Context, image: []const u8) !Module {
    return Module.loadNative(ctx, image);
}

pub fn unload(m: *Module) void {
    m.unload();
}

pub fn kernel(m: Module, name: [:0]const u8, local: [3]u32) !Kernel {
    return m.kernel(name, local);
}

pub fn deinit(k: *Kernel) void {
    k.deinit();
}

pub fn dispatch(k: Kernel, s: Stream, grid: [3]u32, args: anytype) !void {
    return xpu.launch.dispatch(k, s, grid, args);
}
