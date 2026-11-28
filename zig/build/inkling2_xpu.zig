//! Inkling 2 Mini's XPU package in the build: its module and its host regressions.
const std = @import("std");

/// The family's XPU package (config and the open's refusal) for the lane host.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// The package's own regressions, with no device and no Level Zero.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .Debug) }));
    b.step("test-inkling2-xpu-host", "Inkling 2 Mini XPU package: the open's refusal, without devices").dependOn(&run.step);
    step.dependOn(&run.step);
    const attention = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu_attention.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true }) }));
    b.step("test-inkling2-xpu-attention-host", "Inkling 2 XPU attention plan: chunks, tiles, refusals and the recorded launches, without devices").dependOn(&attention.step);
    step.dependOn(&attention.step);
}
