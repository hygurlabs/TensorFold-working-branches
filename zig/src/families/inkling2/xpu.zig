//! Inkling 2 Mini on the Intel GPU (XPU backend): what its open says until the engine's kernels land.
const std = @import("std");
pub const config = @import("config.zig");
/// The fixtures the open's refusal tests read.
pub const fixtures = if (@import("builtin").is_test) @import("config_test.zig") else struct {};

/// config.json's model_type for the looped Mini, the same as the CUDA and Metal columns.
pub const model_type = "inkling2_loop";

/// The refusal for a config.json: the host parser's own reason, else that no XPU engine exists yet.
pub fn refusal(a: std.mem.Allocator, json: []const u8) std.mem.Allocator.Error![]const u8 {
    var why: config.Why = .{};
    const c = config.parse(a, json, &why) catch return std.fmt.allocPrint(a, "{s}: {s}", .{ model_type, why.text() });
    if (c.text.loop == null) return std.fmt.allocPrint(a, "{s}: config.json is not a looped Inkling 2 Mini", .{model_type});
    return std.fmt.allocPrint(a, "{s}: the Intel GPU engine for Inkling 2 Mini is not available yet (its kernels are still being ported); serve it on CUDA or Metal", .{model_type});
}

test "refusal: a Mini config is refused by name as not yet ported" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const text = try refusal(arena.allocator(), fixtures.mini);
    try std.testing.expect(std.mem.startsWith(u8, text, "inkling2_loop: the Intel GPU engine for Inkling 2 Mini is not available yet"));
}

test "refusal: a config that is not Mini's keeps the host parser's reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("inkling2_loop: config.json is not JSON", try refusal(arena.allocator(), "{"));
    for ([_][]const u8{ fixtures.real, fixtures.tiny }) |json| {
        try std.testing.expectEqualStrings("inkling2_loop: config.json is not a looped Inkling 2 Mini", try refusal(arena.allocator(), json));
    }
}
