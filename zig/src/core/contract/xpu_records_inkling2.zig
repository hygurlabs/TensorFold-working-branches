//! Inkling 2's XPU attention launches: its image, the frozen baseline's and the planted kernels, as ocloc builds them.
const limits = @import("xpu_limits.zig");
const Record = @import("xpu_records.zig").Record;

const native = limits.Format.native;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "inkling2 ink2_prep", .symbol = "ink2_prep", .image = "inkling2_attention", .format = native, .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 ink2_rel_table", .symbol = "ink2_rel_table", .image = "inkling2_attention", .format = native, .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 ink2_attn", .symbol = "ink2_attn", .image = "inkling2_attention", .format = native, .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 ink2_attn_pq", .symbol = "ink2_attn_pq", .image = "inkling2_attention", .format = native, .threads = .{ 128, 1, 1 }, .slm = 28672, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 28672, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 ink2_merge", .symbol = "ink2_merge", .image = "inkling2_attention", .format = native, .threads = .{ 128, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 upstream ink2_prep", .symbol = "ink2_prep", .image = "contract_inkling2_upstream", .format = native, .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 upstream ink2_rel_table", .symbol = "ink2_rel_table", .image = "contract_inkling2_upstream", .format = native, .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 upstream ink2_attn", .symbol = "ink2_attn", .image = "contract_inkling2_upstream", .format = native, .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 upstream ink2_attn_pq", .symbol = "ink2_attn_pq", .image = "contract_inkling2_upstream", .format = native, .threads = .{ 128, 1, 1 }, .slm = 28672, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 28672, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 upstream ink2_merge", .symbol = "ink2_merge", .image = "contract_inkling2_upstream", .format = native, .threads = .{ 128, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 planted ink2_attn with P cut to bf16 by truncation", .symbol = "tf_ctl_ink2_attn_ptrunc", .image = "contract_inkling2_controls", .format = native, .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 planted ink2_attn summing a tile's 16-key blocks last to first", .symbol = "tf_ctl_ink2_attn_kblock", .image = "contract_inkling2_controls", .format = native, .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 6144, .sub_group = 16, .grf = 256 }},
    },
};
