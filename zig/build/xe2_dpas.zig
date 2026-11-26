//! The Xe2 DPAS validator in the build: its kernel image and tf-xpu-xe2-dpas-check, under -Dxpu-contract.
const std = @import("std");
const kernel_contract = @import("kernel_contract.zig");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8) void {
    const core = kernel_contract.module(b, target, optimize);
    const ze = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    options.addOption(bool, "dpas", true);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    xpu.addImport("ze", ze);
    xpu.addImport("core", core);
    const run_cl = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    run_cl.addFileArg(b.path("zig/kernels/xpu/xe2_dpas.cl"));
    run_cl.addArgs(&.{ "-device", device, "-options", "-cl-std=CL3.0", "-output_no_suffix", "-output", "xe2_dpas", "-out_dir" });
    run_cl.setCwd(b.path("."));
    const dir = run_cl.addOutputDirectoryArg("ocloc-xe2-dpas");
    const files = b.addWriteFiles();
    _ = files.addCopyFile(dir.path(b, "xe2_dpas.bin"), "xe2_dpas.bin");
    const image = b.createModule(.{ .root_source_file = files.add("xe2_dpas_image.zig", "pub const zebin align(8) = @embedFile(\"xe2_dpas.bin\").*;\n"), .target = target, .optimize = optimize });
    const exe = b.addExecutable(.{ .name = "tf-xpu-xe2-dpas-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/xe2_dpas_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "xe2_dpas_image", .module = image },
        },
    }) });
    b.step("tf-xpu-xe2-dpas-check", "Build the Xe2 DPAS validator: the card's DPAS against the host model").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run = b.addRunArtifact(exe);
    b.step("xpu-xe2-dpas-check", "Run the Xe2 DPAS validator on this card (ZE_AFFINITY_MASK picks it)").dependOn(&run.step);
}
