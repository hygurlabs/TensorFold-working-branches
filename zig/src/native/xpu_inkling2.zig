//! The Inkling 2 Mini entry of the Intel GPU lane host: a table row that refuses to open until the engine exists.
const std = @import("std");
const api = @import("engine_api");
const inkling2 = @import("inkling2_xpu");
const Allocator = std.mem.Allocator;

pub const model_type = inkling2.model_type;

/// Reads the checkpoint's config.json through the family package; this row has no Engine, so no lane is built.
pub fn refuse(a: Allocator, io: std.Io, o: api.Open) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ o.dir, "config.json" }) catch return null;
    const json = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch
        return std.fmt.allocPrint(a, "{s}: cannot read {s}", .{ model_type, path }) catch null;
    return inkling2.refusal(a, json) catch null;
}
