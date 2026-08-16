// Fixed P4 inference kernel for the LLMCRAFT v2 deployment artifact.
// Storage, tasking, DMA, and sampling remain outside this file.

const builtin = @import("builtin");
const build_options = @import("llmm_options");

const Vocab = 32_768;
const Width = 192;
const Layers = 12;
const Heads = 6;
const KvHeads = 2;
const HeadDim = 32;
const Ffn = 512;
const Experts = 29;
const PleDim = 176;
const Context = 1_024;
const RecordBytes = 32;
const HeaderBytes = 96;
const RecordCount = 2_276;
const PleRowElements = Layers * PleDim;
const PleRowBytes = (PleRowElements + 4) / 5;
const AttentionScale: f32 = 0.1767766952966369;
const RmsEpsilon: f32 = 0.00001;
const RopeTheta: f32 = 10_000.0;
const TopLevelPleProjection: u16 = 1_131;
const OutputNorm: u16 = 2_272;
const TopLevelPleNorm: u16 = 2_273;
const TopLevelPleProjectionNorm: u16 = 2_274;
const TernaryScaleTensor: u16 = 2_275;
const KvCacheElements = Context * KvHeads * HeadDim;
const KvScaleElements = Context * KvHeads;
const TokenizerHeaderBytes = 64;
const TokenizerByteIdsBytes = 256 * 2;
const TokenizerMergeBytes = 8;
const BaseVocab = 32_753;
const MaxTopK = 64;
const ProfilePhases = 2;
const ProfileStages = 11;
const ProfileAccelKinds = 3;
const ProfileCores = 2;
const ProfileCacheGroups = 8;
const ProfileCacheEvents = 5;
const ProfileRegions = 2;
const ProfileTrafficKinds = 3;
const ProfileUsbDirections = 2;

const flash_region: u8 = 1;
const psram_region: u8 = 2;
const ternary_base3: u8 = 1;
const fp16: u8 = 2;
const q8: u8 = 3;

pub const Reader = *const fn (context: ?*anyopaque, region: u8, offset: usize, destination: [*]u8, bytes: usize) callconv(.c) c_int;

pub const Handle = extern struct {
    artifact: [*]const u8,
    bytes: usize,
    index_offset: usize,
    flash_offset: usize,
    psram_offset: usize,
    flash_bytes: usize,
    psram_bytes: usize,
    position: u32,
    reader: ?Reader,
    reader_context: ?*anyopaque,
};

pub const Tokenizer = extern struct {
    asset: [*]const u8,
    bytes: usize,
    byte_ids_offset: usize,
    token_offsets_offset: usize,
    token_bytes_offset: usize,
    token_bytes: usize,
    merge_offset: usize,
    merge_count: u32,
    max_token_bytes: u32,
    eos_token: u32,
};

pub const Candidate = extern struct {
    token: u32,
    logit: f32,
};

pub const Profile = extern struct {
    stage_cycles: [ProfilePhases][ProfileStages]u64,
    stage_max_cycles: [ProfilePhases][ProfileStages]u64,
    stage_calls: [ProfilePhases][ProfileStages]u64,
    step_cycles: [ProfilePhases]u64,
    step_max_cycles: [ProfilePhases]u64,
    step_min_cycles: [ProfilePhases]u64,
    step_calls: [ProfilePhases]u64,
    owner_wait_cycles: [ProfileAccelKinds]u64,
    owner_wait_max_cycles: [ProfileAccelKinds]u64,
    owner_wait_calls: [ProfileAccelKinds]u64,
    worker_busy_cycles: [ProfileAccelKinds]u64,
    worker_busy_max_cycles: [ProfileAccelKinds]u64,
    worker_busy_calls: [ProfileAccelKinds]u64,
    dispatch_cycles: [ProfileAccelKinds]u64,
    dispatch_max_cycles: [ProfileAccelKinds]u64,
    dispatch_calls: [ProfileAccelKinds]u64,
    worker_idle_cycles: u64,
    worker_idle_max_cycles: u64,
    worker_idle_calls: u64,
    cpu_cycles: [ProfileCores]u64,
    cpu_instructions: [ProfileCores]u64,
    cpu_branch_misses: [ProfileCores]u64,
    cpu_conditional_branches: [ProfileCores]u64,
    cpu_stores: [ProfileCores]u64,
    cache_events: [ProfileCacheGroups][ProfileCacheEvents]u64,
    reader_bytes: [ProfileRegions]u64,
    reader_cycles: [ProfileRegions]u64,
    reader_max_cycles: [ProfileRegions]u64,
    reader_calls: [ProfileRegions]u64,
    traffic_bytes: [ProfileTrafficKinds]u64,
    traffic_cycles: [ProfileTrafficKinds]u64,
    traffic_max_cycles: [ProfileTrafficKinds]u64,
    traffic_calls: [ProfileTrafficKinds]u64,
    usb_bytes: [ProfileUsbDirections]u64,
    usb_cycles: [ProfileUsbDirections]u64,
    usb_max_cycles: [ProfileUsbDirections]u64,
    usb_calls: [ProfileUsbDirections]u64,
    total_cycles: u64,
    ttft_cycles: u64,
    token_steps: [ProfilePhases]u64,
    scored_steps: [ProfilePhases]u64,
    kv_reused_tokens: u64,
    kv_appended_tokens: u64,
    kv_occupancy_tokens: u64,
    kv_evictions: u64,
    expert_route_same: u64,
    expert_route_total: u64,
    rope_same: u64,
    rope_sequential: u64,
    rope_rebuild: u64,
    internal_free_bytes: u64,
    internal_largest_bytes: u64,
    internal_min_free_bytes: u64,
    psram_free_bytes: u64,
    psram_largest_bytes: u64,
    psram_min_free_bytes: u64,
    stack_free_bytes: [ProfileCores]u64,
};

const use_p4_acceleration = builtin.target.cpu.arch == .riscv32 and builtin.target.os.tag == .freestanding;
const enable_debug = build_options.debug;
var active_profile: ?*Profile = null;
var active_profile_phase: usize = 0;

extern fn llmm_p4_base3_matvec(
    model: *const Handle,
    tensor_id: u16,
    input: [*]const f32,
    output: [*]f32,
    row_start: usize,
    rows: usize,
    columns: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
) callconv(.c) c_int;
extern fn llmm_p4_rmsnorm(
    model: *const Handle,
    norm_id: u16,
    input: [*]const f32,
    output: [*]f32,
    elements: usize,
    epsilon: f32,
) callconv(.c) c_int;
extern fn llmm_p4_rope(vector: [*]f32, heads: usize, position: usize) callconv(.c) c_int;
extern fn llmm_p4_attention_gqa(
    query: [*]const f32,
    keys: [*]const i8,
    key_scales: [*]const u16,
    values: [*]const i8,
    value_scales: [*]const u16,
    tokens: usize,
    output: [*]f32,
) callconv(.c) c_int;
extern fn llmm_p4_router_top1(logits: [*]const f32, experts: usize, selected: *u32, probability: *f32) callconv(.c) c_int;
extern fn llmm_p4_silu_mul(gate: [*]f32, up: [*]const f32, elements: usize) callconv(.c) c_int;
extern fn llmm_p4_gelu_mul(value: [*]f32, multiplier: [*]const f32, elements: usize) callconv(.c) c_int;
extern fn llmm_p4_output_head_topk(
    model: *const Handle,
    normalized: [*]const f32,
    candidates: [*]Candidate,
    candidate_capacity: usize,
    requested_top_k: usize,
    candidate_count: *usize,
) callconv(.c) c_int;
extern fn llmm_p4_cycle_count() callconv(.c) u32;

const ProfileStage = enum {
    embedding,
    ple_embedding,
    qkv,
    attention,
    attention_output,
    router,
    expert,
    ple_adapter,
    output_head,
    sampling,
};

inline fn profileStart() u32 {
    if (comptime !enable_debug or !use_p4_acceleration) return 0;
    if (active_profile == null) return 0;
    return llmm_p4_cycle_count();
}

inline fn profileFinish(started: u32, stage: ProfileStage) void {
    if (comptime !enable_debug or !use_p4_acceleration) return;
    const profile = active_profile orelse return;
    const elapsed = @as(u64, llmm_p4_cycle_count() -% started);
    const stage_index: usize = @intFromEnum(stage);
    profile.stage_cycles[active_profile_phase][stage_index] += elapsed;
    profile.stage_calls[active_profile_phase][stage_index] += 1;
    if (elapsed > profile.stage_max_cycles[active_profile_phase][stage_index]) {
        profile.stage_max_cycles[active_profile_phase][stage_index] = elapsed;
    }
}

fn profileBegin(profile: *Profile) callconv(.c) void {
    active_profile = profile;
    active_profile_phase = 0;
}

fn profileEnd() callconv(.c) void {
    active_profile = null;
}

fn profileSetPhase(phase: u32) callconv(.c) void {
    active_profile_phase = if (phase < ProfilePhases) phase else 0;
}

comptime {
    if (enable_debug) {
        @export(&profileBegin, .{ .name = "llmm_profile_begin" });
        @export(&profileEnd, .{ .name = "llmm_profile_end" });
        @export(&profileSetPhase, .{ .name = "llmm_profile_set_phase" });
    }
}

fn le16(bytes: [*]const u8, offset: usize) u16 {
    return @as(u16, bytes[offset]) | (@as(u16, bytes[offset + 1]) << 8);
}

fn le32(bytes: [*]const u8, offset: usize) u32 {
    return @as(u32, bytes[offset]) |
        (@as(u32, bytes[offset + 1]) << 8) |
        (@as(u32, bytes[offset + 2]) << 16) |
        (@as(u32, bytes[offset + 3]) << 24);
}

fn le64(bytes: [*]const u8, offset: usize) u64 {
    return @as(u64, le32(bytes, offset)) | (@as(u64, le32(bytes, offset + 4)) << 32);
}

fn validRange(offset: u64, size: u64, total: usize) bool {
    return offset <= total and size <= total - offset;
}

fn tokenizerRange(offset: u32, size: u32, total: usize) bool {
    return validRange(@as(u64, offset), @as(u64, size), total);
}

const Merge = struct {
    result: u16,
    rank: u16,
};

const SpecialTokens = [_][]const u8{
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<think>",
    "</think>",
};

fn asciiLower(value: u8) u8 {
    return if (value >= 'A' and value <= 'Z') value + ('a' - 'A') else value;
}

fn isAsciiLetter(value: u8) bool {
    return (value >= 'a' and value <= 'z') or (value >= 'A' and value <= 'Z');
}

fn isAsciiDigit(value: u8) bool {
    return value >= '0' and value <= '9';
}

fn isAsciiWhitespace(value: u8) bool {
    return value == ' ' or (value >= 0x09 and value <= 0x0d);
}

fn isNewline(value: u8) bool {
    return value == '\r' or value == '\n';
}

fn isAsciiSymbol(value: u8) bool {
    return !isAsciiWhitespace(value) and !isAsciiLetter(value) and !isAsciiDigit(value);
}

fn textMatches(text: [*]const u8, text_bytes: usize, start: usize, pattern: []const u8) bool {
    if (start > text_bytes or pattern.len > text_bytes - start) return false;
    var index: usize = 0;
    while (index < pattern.len) : (index += 1) if (text[start + index] != pattern[index]) return false;
    return true;
}

fn specialTokenAt(text: [*]const u8, text_bytes: usize, start: usize) ?struct { token: u32, bytes: usize } {
    inline for (SpecialTokens, 0..) |pattern, index| {
        if (textMatches(text, text_bytes, start, pattern)) {
            return .{ .token = BaseVocab + @as(u32, @intCast(index)), .bytes = pattern.len };
        }
    }
    return null;
}

fn contractionBytes(text: [*]const u8, text_bytes: usize, start: usize) usize {
    if (start >= text_bytes or text[start] != '\'') return 0;
    const remaining = text_bytes - start;
    if (remaining >= 3 and asciiLower(text[start + 1]) == 'r' and asciiLower(text[start + 2]) == 'e') return 3;
    if (remaining >= 3 and asciiLower(text[start + 1]) == 'v' and asciiLower(text[start + 2]) == 'e') return 3;
    if (remaining >= 3 and asciiLower(text[start + 1]) == 'l' and asciiLower(text[start + 2]) == 'l') return 3;
    if (remaining >= 2) {
        const suffix = asciiLower(text[start + 1]);
        if (suffix == 's' or suffix == 't' or suffix == 'm' or suffix == 'd') return 2;
    }
    return 0;
}

// This is the ASCII specialization of the locked Qwen ByteLevel pre-tokenizer
// regex. The runtime accepts English byte-level tokenizer input here.
fn preTokenEnd(text: [*]const u8, text_bytes: usize, start: usize) usize {
    const contraction = contractionBytes(text, text_bytes, start);
    if (contraction != 0) return start + contraction;

    var cursor = start;
    if (isAsciiLetter(text[cursor])) {
        cursor += 1;
        while (cursor < text_bytes and isAsciiLetter(text[cursor])) : (cursor += 1) {}
        return cursor;
    }
    if (!isNewline(text[cursor]) and !isAsciiLetter(text[cursor]) and !isAsciiDigit(text[cursor]) and
        cursor + 1 < text_bytes and isAsciiLetter(text[cursor + 1]))
    {
        cursor += 2;
        while (cursor < text_bytes and isAsciiLetter(text[cursor])) : (cursor += 1) {}
        return cursor;
    }
    if (isAsciiDigit(text[cursor])) return cursor + 1;

    if (text[cursor] == ' ' and cursor + 1 < text_bytes and isAsciiSymbol(text[cursor + 1])) cursor += 1;
    if (isAsciiSymbol(text[cursor])) {
        cursor += 1;
        while (cursor < text_bytes and isAsciiSymbol(text[cursor])) : (cursor += 1) {}
        while (cursor < text_bytes and isNewline(text[cursor])) : (cursor += 1) {}
        return cursor;
    }

    if (isAsciiWhitespace(text[cursor])) {
        // Match \s*[\r\n]+ before the two generic whitespace alternatives.
        cursor = start;
        while (cursor < text_bytes and isAsciiWhitespace(text[cursor]) and !isNewline(text[cursor])) : (cursor += 1) {}
        if (cursor < text_bytes and isNewline(text[cursor])) {
            while (cursor < text_bytes and isNewline(text[cursor])) : (cursor += 1) {}
            return cursor;
        }

        // \s+(?!\S) leaves one whitespace byte before a following non-space
        // character, allowing the optional-space letter/symbol alternatives to
        // consume it on the next pre-token.
        cursor = start;
        while (cursor < text_bytes and isAsciiWhitespace(text[cursor])) : (cursor += 1) {}
        if (cursor > start + 1 and cursor < text_bytes) return cursor - 1;
        return cursor;
    }
    return start + 1;
}

fn tokenizerByteId(tokenizer: *const Tokenizer, value: u8) u16 {
    return le16(tokenizer.asset, tokenizer.byte_ids_offset + @as(usize, value) * 2);
}

fn findMerge(tokenizer: *const Tokenizer, left: u16, right: u16) ?Merge {
    var first: usize = 0;
    var end: usize = tokenizer.merge_count;
    while (first < end) {
        const middle = first + (end - first) / 2;
        const offset = tokenizer.merge_offset + middle * TokenizerMergeBytes;
        const record_left = le16(tokenizer.asset, offset);
        const record_right = le16(tokenizer.asset, offset + 2);
        if (record_left < left or (record_left == left and record_right < right)) {
            first = middle + 1;
        } else {
            end = middle;
        }
    }
    if (first >= tokenizer.merge_count) return null;
    const offset = tokenizer.merge_offset + first * TokenizerMergeBytes;
    if (le16(tokenizer.asset, offset) != left or le16(tokenizer.asset, offset + 2) != right) return null;
    return .{ .result = le16(tokenizer.asset, offset + 4), .rank = le16(tokenizer.asset, offset + 6) };
}

fn bpeEncode(
    tokenizer: *const Tokenizer,
    text: [*]const u8,
    start: usize,
    text_bytes: usize,
    output: [*]u32,
    output_capacity: usize,
    output_count: *usize,
    pieces: [*]u16,
    piece_capacity: usize,
) c_int {
    if (text_bytes == 0 or text_bytes > piece_capacity) return -1;
    var index: usize = 0;
    while (index < text_bytes) : (index += 1) pieces[index] = tokenizerByteId(tokenizer, text[start + index]);
    var count = text_bytes;
    while (count > 1) {
        var best_rank: u16 = 0xffff;
        var best_result: u16 = 0;
        var best_index: usize = count;
        index = 0;
        while (index + 1 < count) : (index += 1) {
            if (findMerge(tokenizer, pieces[index], pieces[index + 1])) |merge| {
                if (merge.rank < best_rank) {
                    best_rank = merge.rank;
                    best_result = merge.result;
                    best_index = index;
                }
            }
        }
        if (best_index == count) break;
        pieces[best_index] = best_result;
        index = best_index + 1;
        while (index + 1 < count) : (index += 1) pieces[index] = pieces[index + 1];
        count -= 1;
    }
    if (output_count.* > output_capacity or count > output_capacity - output_count.*) return -2;
    index = 0;
    while (index < count) : (index += 1) output[output_count.* + index] = pieces[index];
    output_count.* += count;
    return 0;
}

pub export fn llmm_tokenizer_init(tokenizer: *Tokenizer, asset: [*]const u8, bytes: usize) c_int {
    if (bytes < TokenizerHeaderBytes) return -1;
    if (asset[0] != 'L' or asset[1] != 'L' or asset[2] != 'M' or asset[3] != 'T' or
        asset[4] != 'O' or asset[5] != 'K' or asset[6] != '0' or asset[7] != '1') return -2;
    if (le16(asset, 8) != 1 or le16(asset, 10) != TokenizerHeaderBytes) return -3;
    const total_bytes = le32(asset, 12);
    const vocab = le32(asset, 16);
    const merge_count = le32(asset, 20);
    const byte_ids_offset = le32(asset, 24);
    const token_offsets_offset = le32(asset, 28);
    const token_bytes_offset = le32(asset, 32);
    const token_bytes = le32(asset, 36);
    const merge_offset = le32(asset, 40);
    const max_token_bytes = le32(asset, 44);
    const eos_token = le32(asset, 48);
    if (total_bytes > bytes or vocab != Vocab or merge_count == 0 or eos_token >= Vocab) return -4;
    const total: usize = total_bytes;
    if (!tokenizerRange(byte_ids_offset, TokenizerByteIdsBytes, total) or
        !tokenizerRange(token_offsets_offset, (Vocab + 1) * 4, total) or
        !tokenizerRange(token_bytes_offset, token_bytes, total) or
        !tokenizerRange(merge_offset, merge_count * TokenizerMergeBytes, total)) return -5;
    if (le32(asset, @as(usize, token_offsets_offset) + Vocab * 4) != token_bytes) return -6;
    tokenizer.* = .{
        .asset = asset,
        .bytes = total,
        .byte_ids_offset = byte_ids_offset,
        .token_offsets_offset = token_offsets_offset,
        .token_bytes_offset = token_bytes_offset,
        .token_bytes = token_bytes,
        .merge_offset = merge_offset,
        .merge_count = merge_count,
        .max_token_bytes = max_token_bytes,
        .eos_token = eos_token,
    };
    return 0;
}

pub export fn llmm_tokenizer_encode(
    tokenizer: *const Tokenizer,
    text: [*]const u8,
    text_bytes: usize,
    output: [*]u32,
    output_capacity: usize,
    output_count: *usize,
    pieces: [*]u16,
    piece_capacity: usize,
) c_int {
    if (text_bytes == 0) return -1;
    var index: usize = 0;
    while (index < text_bytes) : (index += 1) if (text[index] >= 0x80) return -2;
    var count: usize = 0;
    index = 0;
    while (index < text_bytes) {
        if (specialTokenAt(text, text_bytes, index)) |special| {
            if (count >= output_capacity) return -3;
            output[count] = special.token;
            count += 1;
            index += special.bytes;
            continue;
        }
        const end = preTokenEnd(text, text_bytes, index);
        if (end <= index) return -4;
        const status = bpeEncode(tokenizer, text, index, end - index, output, output_capacity, &count, pieces, piece_capacity);
        if (status != 0) return -10 + status;
        index = end;
    }
    output_count.* = count;
    return 0;
}

pub export fn llmm_tokenizer_decode(
    tokenizer: *const Tokenizer,
    token: u32,
    output: [*]u8,
    output_capacity: usize,
    output_bytes: *usize,
) c_int {
    if (token >= Vocab) return -1;
    const start = le32(tokenizer.asset, tokenizer.token_offsets_offset + @as(usize, token) * 4);
    const end = le32(tokenizer.asset, tokenizer.token_offsets_offset + @as(usize, token + 1) * 4);
    if (end < start or end > tokenizer.token_bytes or end - start > output_capacity) return -2;
    var index: usize = 0;
    while (index < end - start) : (index += 1) output[index] = tokenizer.asset[tokenizer.token_bytes_offset + start + index];
    output_bytes.* = end - start;
    return 0;
}

pub export fn llmm_tokenizer_eos(tokenizer: *const Tokenizer) u32 {
    return tokenizer.eos_token;
}

pub export fn llmm_tokenizer_max_token_bytes(tokenizer: *const Tokenizer) usize {
    return tokenizer.max_token_bytes;
}

// Every record is fixed-width and indexed by its tensor id. No tensor-name
// strings are stored in the deployment artifact.
fn recordOffset(model: *const Handle, tensor_id: u16) usize {
    return model.index_offset + @as(usize, tensor_id) * RecordBytes;
}

fn recordRegion(model: *const Handle, tensor_id: u16) u8 {
    return model.artifact[recordOffset(model, tensor_id) + 2];
}

fn recordStorage(model: *const Handle, tensor_id: u16) u8 {
    return model.artifact[recordOffset(model, tensor_id) + 3];
}

fn recordScale(model: *const Handle, tensor_id: u16) u16 {
    return le16(model.artifact, recordOffset(model, tensor_id) + 4);
}

fn recordElements(model: *const Handle, tensor_id: u16) usize {
    return @intCast(le64(model.artifact, recordOffset(model, tensor_id) + 6));
}

fn recordPayload(model: *const Handle, tensor_id: u16) [*]const u8 {
    const offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, tensor_id) + 14)));
    const base = if (recordRegion(model, tensor_id) == flash_region) model.flash_offset else model.psram_offset;
    return model.artifact + base + offset;
}

fn recordBytes(model: *const Handle, tensor_id: u16) usize {
    return @intCast(le64(model.artifact, recordOffset(model, tensor_id) + 22));
}

// The exporter emits all rank-two tensors first, in lexical state_dict order,
// then appends all FP16 norm vectors. A block therefore contributes 94 Base-3
// matrix records: attention (4), experts (29 * 3), router (1), PLE (2).
fn blockBase(layer: u32) u16 {
    // Lexical state_dict order is 0, 1, 10, 11, 2 ... 9.
    const lexical_layer: u32 = if (layer < 2) layer else if (layer < 10) layer + 2 else layer - 8;
    return @intCast(3 + lexical_layer * 94);
}

fn normBlockBase(layer: u32) u16 {
    return @intCast(1_132 + (if (layer < 2) layer else if (layer < 10) layer + 2 else layer - 8) * 95);
}

pub fn tensorIdAttentionWeight(layer: u32, which: u32) u16 {
    // which: key=0, output=1, query=2, value=3.
    return blockBase(layer) + @as(u16, @intCast(which));
}

pub fn tensorIdExpertWeight(layer: u32, expert: u32, which: u32) u16 {
    // Packed checkpoints store contiguous down, gate, and up weight banks.
    return blockBase(layer) + 4 + @as(u16, @intCast(expert + which * Experts));
}

pub fn tensorIdRouterWeight(layer: u32) u16 {
    return blockBase(layer) + 91;
}

pub fn tensorIdPleWeight(layer: u32, which: u32) u16 {
    // which: gate=0, projection=1.
    return switch (which) {
        0 => blockBase(layer) + 92,
        1 => blockBase(layer) + 93,
        else => unreachable,
    };
}

pub fn tensorIdAttentionNorm(layer: u32, which: u32) u16 {
    return normBlockBase(layer) + @as(u16, @intCast(which));
}

pub fn tensorIdExpertNorm(layer: u32, expert: u32, which: u32) u16 {
    // Packed checkpoints store contiguous down, gate, and up norm banks.
    return normBlockBase(layer) + 4 + @as(u16, @intCast(expert + which * Experts));
}

pub fn tensorIdRouterNorm(layer: u32) u16 {
    return normBlockBase(layer) + 91;
}

pub fn tensorIdPleNorm(layer: u32, which: u32) u16 {
    // which: gate input=0, block output=1, projection input=2.
    return normBlockBase(layer) + 92 + @as(u16, @intCast(which));
}

pub fn tensorIdPleGateNorm(layer: u32) u16 {
    return normBlockBase(layer) + 92;
}

pub fn tensorIdPleOutputNorm(layer: u32) u16 {
    return normBlockBase(layer) + 93;
}

pub fn tensorIdPleProjectionNorm(layer: u32) u16 {
    return normBlockBase(layer) + 94;
}

fn decodeFp16Bits(bits: u16) f32 {
    const sign: f32 = if ((bits & 0x8000) == 0) 1.0 else -1.0;
    const exponent: i32 = @intCast((bits >> 10) & 0x1f);
    const fraction: f32 = @as(f32, @floatFromInt(bits & 0x03ff));
    if (exponent == 0) return sign * fraction * 0.000000059604644775390625;
    if (exponent == 31) return if (fraction == 0) sign * 65504.0 else 0.0;
    var value = 1.0 + fraction / 1024.0;
    var shift = exponent - 15;
    while (shift > 0) : (shift -= 1) value *= 2.0;
    while (shift < 0) : (shift += 1) value *= 0.5;
    return sign * value;
}

fn decodeFp16(bytes: [*]const u8) f32 {
    return decodeFp16Bits(le16(bytes, 0));
}

pub export fn llmm_init(model: *Handle, artifact: [*]const u8, bytes: usize) c_int {
    if (bytes < HeaderBytes + RecordCount * RecordBytes) return -1;
    if (artifact[0] != 'L' or artifact[1] != 'L' or artifact[2] != 'M' or artifact[3] != 'C' or
        artifact[4] != 'R' or artifact[5] != 'A' or artifact[6] != 'F' or artifact[7] != 'T') return -2;
    if (le16(artifact, 8) != 2 or le16(artifact, 10) != HeaderBytes) return -3;
    const expected = [_]u32{ Vocab, Width, Layers, Heads, KvHeads, Ffn, Experts, PleDim, Context };
    inline for (expected, 0..) |value, index| if (le32(artifact, 12 + index * 4) != value) return -4;
    if (le32(artifact, 48) != RecordCount) return -5;
    const index_offset: usize = @intCast(le64(artifact, 52));
    const flash_offset: usize = @intCast(le64(artifact, 60));
    const flash_bytes: usize = @intCast(le64(artifact, 68));
    const psram_offset: usize = @intCast(le64(artifact, 76));
    const psram_bytes: usize = @intCast(le64(artifact, 84));
    if (index_offset != HeaderBytes or flash_offset != HeaderBytes + RecordCount * RecordBytes) return -6;
    if (psram_offset != flash_offset + flash_bytes or !validRange(flash_offset, flash_bytes, bytes) or !validRange(psram_offset, psram_bytes, bytes)) return -7;
    model.* = .{ .artifact = artifact, .bytes = bytes, .index_offset = index_offset, .flash_offset = flash_offset, .psram_offset = psram_offset, .flash_bytes = flash_bytes, .psram_bytes = psram_bytes, .position = 0, .reader = null, .reader_context = null };
    var tensor_id: u16 = 0;
    while (tensor_id < RecordCount) : (tensor_id += 1) {
        const offset = recordOffset(model, tensor_id);
        if (le16(artifact, offset) != tensor_id) return -8;
        const region = recordRegion(model, tensor_id);
        const storage = recordStorage(model, tensor_id);
        const payload_offset = le64(artifact, offset + 14);
        const payload_bytes = le64(artifact, offset + 22);
        const limit: usize = if (region == flash_region) flash_bytes else if (region == psram_region) psram_bytes else return -9;
        if ((storage != ternary_base3 and storage != fp16 and storage != q8) or !validRange(payload_offset, payload_bytes, limit)) return -10;
    }
    return 0;
}

pub export fn llmm_init_manifest(model: *Handle, manifest: [*]const u8, bytes: usize) c_int {
    if (bytes < HeaderBytes + RecordCount * RecordBytes) return -1;
    if (manifest[0] != 'L' or manifest[1] != 'L' or manifest[2] != 'M' or manifest[3] != 'C' or manifest[4] != 'R' or manifest[5] != 'A' or manifest[6] != 'F' or manifest[7] != 'T') return -2;
    if (le16(manifest, 8) != 2 or le16(manifest, 10) != HeaderBytes or le32(manifest, 48) != RecordCount) return -3;
    const index_offset: usize = @intCast(le64(manifest, 52));
    const flash_offset: usize = @intCast(le64(manifest, 60));
    const flash_bytes: usize = @intCast(le64(manifest, 68));
    const psram_offset: usize = @intCast(le64(manifest, 76));
    const psram_bytes: usize = @intCast(le64(manifest, 84));
    if (index_offset != HeaderBytes or flash_offset != HeaderBytes + RecordCount * RecordBytes or bytes < flash_offset) return -4;
    if (psram_offset != flash_offset + flash_bytes) return -5;
    const expected = [_]u32{ Vocab, Width, Layers, Heads, KvHeads, Ffn, Experts, PleDim, Context };
    inline for (expected, 0..) |value, index| if (le32(manifest, 12 + index * 4) != value) return -6;
    model.* = .{ .artifact = manifest, .bytes = bytes, .index_offset = index_offset, .flash_offset = flash_offset, .psram_offset = psram_offset, .flash_bytes = flash_bytes, .psram_bytes = psram_bytes, .position = 0, .reader = null, .reader_context = null };
    var tensor_id: u16 = 0;
    while (tensor_id < RecordCount) : (tensor_id += 1) {
        const offset = recordOffset(model, tensor_id);
        if (le16(manifest, offset) != tensor_id) return -7;
        const region = recordRegion(model, tensor_id);
        const storage = recordStorage(model, tensor_id);
        const payload_offset = le64(manifest, offset + 14);
        const payload_bytes = le64(manifest, offset + 22);
        const limit: usize = if (region == flash_region) flash_bytes else if (region == psram_region) psram_bytes else return -8;
        if ((storage != ternary_base3 and storage != fp16 and storage != q8) or !validRange(payload_offset, payload_bytes, limit)) return -9;
    }
    return 0;
}

pub export fn llmm_set_reader(model: *Handle, context: ?*anyopaque, reader: ?Reader) void {
    model.reader_context = context;
    model.reader = reader;
}

pub export fn llmm_reset(model: *Handle) void {
    model.position = 0;
}

// Decode the i-th ternary element directly from the packed Base-3 payload.
// Codes are {-1, 0, +1}; no complete matrix expansion is permitted.
pub export fn llmm_ternary_at(model: *const Handle, tensor_id: u16, index: usize) c_int {
    if (tensor_id >= RecordCount or recordStorage(model, tensor_id) != ternary_base3 or index >= recordElements(model, tensor_id)) return -2;
    const byte = recordPayload(model, tensor_id)[index / 5];
    var power: u8 = 1;
    var count: usize = 0;
    while (count < index % 5) : (count += 1) power *= 3;
    return @as(c_int, (byte / power) % 3) - 1;
}

pub export fn llmm_tensor_bytes(model: *const Handle, tensor_id: u16) usize {
    if (tensor_id >= RecordCount) return 0;
    return recordBytes(model, tensor_id);
}

pub export fn llmm_tensor_scale_index(model: *const Handle, tensor_id: u16) u16 {
    if (tensor_id >= RecordCount) return 0xffff;
    return recordScale(model, tensor_id);
}

pub export fn llmm_embedding_row(model: *const Handle, token: u32, output: [*]f32, output_len: usize) c_int {
    if (token >= Vocab or output_len < Width or recordStorage(model, 1) != q8 or recordStorage(model, 2) != fp16) return -1;
    const weights = recordPayload(model, 1) + @as(usize, token) * Width;
    const scales = recordPayload(model, 2) + @as(usize, token) * 2;
    const scale = decodeFp16(scales);
    var index: usize = 0;
    while (index < Width) : (index += 1) output[index] = @as(f32, @floatFromInt(@as(i8, @bitCast(weights[index])))) * scale;
    return 0;
}

pub export fn llmm_embedding_row_stream(model: *const Handle, token: u32, output: [*]f32, output_len: usize) c_int {
    if (model.reader == null or token >= Vocab or output_len < Width) return -1;
    var weights: [Width]u8 = undefined;
    var scale_bytes: [2]u8 = undefined;
    const read = model.reader.?;
    const weight_id: u16 = 1;
    const scale_id: u16 = 2;
    const weight_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, weight_id) + 14))) + @as(usize, token) * Width;
    const scale_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, scale_id) + 14))) + @as(usize, token) * 2;
    if (read(model.reader_context, recordRegion(model, weight_id), weight_offset, &weights, Width) != 0) return -2;
    if (read(model.reader_context, recordRegion(model, scale_id), scale_offset, &scale_bytes, 2) != 0) return -3;
    const scale = decodeFp16(&scale_bytes);
    var index: usize = 0;
    while (index < Width) : (index += 1) output[index] = @as(f32, @floatFromInt(@as(i8, @bitCast(weights[index])))) * scale;
    return 0;
}

// Read one token's PLE slice directly from the Flash-mapped Base-3 table.
// A slice needs at most 36 packed bytes; no row or table is expanded.
pub export fn llmm_ple_slice_stream(
    model: *const Handle,
    token: u32,
    layer: u32,
    output: [*]f32,
    output_len: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    const ple_id: u16 = 0;
    const scale_id: u16 = 2_275;
    if (model.reader == null or token >= Vocab or layer >= Layers or output_len < PleDim) return -1;
    if (recordStorage(model, ple_id) != ternary_base3 or recordScale(model, ple_id) == 0xffff) return -2;
    const start: usize = @as(usize, layer) * PleDim;
    const first_byte: usize = start / 5;
    const slice_bytes: usize = (start + PleDim + 4) / 5 - first_byte;
    if (scratch_bytes < slice_bytes) return -3;
    const read = model.reader.?;
    const payload_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, ple_id) + 14)));
    const row_offset = payload_offset + @as(usize, token) * PleRowBytes + first_byte;
    if (read(model.reader_context, recordRegion(model, ple_id), row_offset, scratch, slice_bytes) != 0) return -4;

    const scale_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, scale_id) + 14))) + @as(usize, recordScale(model, ple_id)) * 2;
    var scale_bytes: [2]u8 = undefined;
    if (read(model.reader_context, recordRegion(model, scale_id), scale_offset, &scale_bytes, 2) != 0) return -5;
    const scale = decodeFp16(&scale_bytes);
    var column: usize = 0;
    while (column < PleDim) : (column += 1) {
        const index = start + column;
        const packed_byte = scratch[index / 5 - first_byte];
        var power: u8 = 1;
        var digit: usize = 0;
        while (digit < index % 5) : (digit += 1) power *= 3;
        const trit: c_int = @as(c_int, (packed_byte / power) % 3) - 1;
        output[column] = @as(f32, @floatFromInt(trit)) * scale;
    }
    return 0;
}

pub export fn llmm_matvec_slice_stream(
    model: *const Handle,
    tensor_id: u16,
    input: [*]const f32,
    output: [*]f32,
    row_start: usize,
    rows: usize,
    columns: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    if (model.reader == null or tensor_id >= RecordCount or recordStorage(model, tensor_id) != ternary_base3 or columns == 0) return -1;
    const elements = recordElements(model, tensor_id);
    if (elements % columns != 0) return -2;
    const total_rows = elements / columns;
    if (row_start > total_rows or rows > total_rows - row_start) return -2;
    const minimum_bytes = (columns + 4) / 5;
    if (scratch_bytes < minimum_bytes) return -3;

    if (comptime use_p4_acceleration) {
        return llmm_p4_base3_matvec(model, tensor_id, input, output, row_start, rows, columns, scratch, scratch_bytes);
    }

    const read = model.reader.?;
    const scale_index = recordScale(model, tensor_id);
    if (scale_index == 0xffff) return -4;
    const scale_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, TernaryScaleTensor) + 14))) + @as(usize, scale_index) * 2;
    var scale_bytes: [2]u8 = undefined;
    if (read(model.reader_context, recordRegion(model, TernaryScaleTensor), scale_offset, &scale_bytes, 2) != 0) return -5;
    const weight_scale = decodeFp16(&scale_bytes);

    var maximum: f32 = 0.0;
    var column: usize = 0;
    while (column < columns) : (column += 1) {
        const value = input[column];
        const absolute = if (value < 0.0) -value else value;
        if (absolute > maximum) maximum = absolute;
    }
    const activation_scale: f32 = if (maximum == 0.0) 1.0 else maximum / 127.0;
    const payload_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, tensor_id) + 14)));
    const maximum_tile_rows = (scratch_bytes * 5) / columns;
    if (maximum_tile_rows == 0 and rows != 0) return -3;

    var first_row: usize = row_start;
    const end_row = row_start + rows;
    while (first_row < end_row) {
        var tile_rows = if (maximum_tile_rows < end_row - first_row) maximum_tile_rows else end_row - first_row;
        const packed_start = (first_row * columns) / 5;
        while (tile_rows > 1 and (((first_row + tile_rows) * columns + 4) / 5 - packed_start > scratch_bytes)) tile_rows -= 1;
        const packed_end = ((first_row + tile_rows) * columns + 4) / 5;
        const packed_bytes = packed_end - packed_start;
        if (read(model.reader_context, recordRegion(model, tensor_id), payload_offset + packed_start, scratch, packed_bytes) != 0) return -6;

        var row: usize = 0;
        while (row < tile_rows) : (row += 1) {
            var sum: f32 = 0.0;
            column = 0;
            while (column < columns) : (column += 1) {
                const index = (first_row + row) * columns + column;
                const packed_byte = scratch[index / 5 - packed_start];
                var power: u8 = 1;
                var digit: usize = 0;
                while (digit < index % 5) : (digit += 1) power *= 3;
                const trit = @as(f32, @floatFromInt(@as(c_int, (packed_byte / power) % 3) - 1));
                var quantized = @round(input[column] / activation_scale);
                if (quantized > 127.0) quantized = 127.0;
                if (quantized < -127.0) quantized = -127.0;
                sum += (quantized * activation_scale) * trit * weight_scale;
            }
            output[first_row + row - row_start] = sum;
        }
        first_row += tile_rows;
    }
    return 0;
}

pub export fn llmm_matvec_stream(
    model: *const Handle,
    tensor_id: u16,
    input: [*]const f32,
    output: [*]f32,
    rows: usize,
    columns: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    return llmm_matvec_slice_stream(model, tensor_id, input, output, 0, rows, columns, scratch, scratch_bytes);
}

pub export fn llmm_rmsnorm_stream(
    model: *const Handle,
    norm_id: u16,
    input: [*]const f32,
    output: [*]f32,
    elements: usize,
    epsilon: f32,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    if (model.reader == null or norm_id >= RecordCount or recordStorage(model, norm_id) != fp16 or recordElements(model, norm_id) != elements or elements == 0) return -1;
    if (comptime use_p4_acceleration) {
        return llmm_p4_rmsnorm(model, norm_id, input, output, elements, epsilon);
    }
    const tile_elements = scratch_bytes / 2;
    if (tile_elements == 0) return -2;
    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index < elements) : (index += 1) sum += input[index] * input[index];
    const safe_epsilon = if (epsilon > 0.0) epsilon else RmsEpsilon;
    const inverse_rms = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(elements)) + safe_epsilon);
    const payload_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, norm_id) + 14)));
    const read = model.reader.?;
    index = 0;
    while (index < elements) {
        const count = if (tile_elements < elements - index) tile_elements else elements - index;
        if (read(model.reader_context, recordRegion(model, norm_id), payload_offset + index * 2, scratch, count * 2) != 0) return -3;
        var tile: usize = 0;
        while (tile < count) : (tile += 1) output[index + tile] = input[index + tile] * inverse_rms * decodeFp16(scratch + tile * 2);
        index += count;
    }
    return 0;
}

pub export fn llmm_matvec_norm_stream(
    model: *const Handle,
    tensor_id: u16,
    norm_id: u16,
    input: [*]const f32,
    output: [*]f32,
    rows: usize,
    columns: usize,
    normalized: [*]f32,
    normalized_len: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    if (normalized_len < columns) return -1;
    const norm_status = llmm_rmsnorm_stream(model, norm_id, input, normalized, columns, RmsEpsilon, scratch, scratch_bytes);
    if (norm_status != 0) return -10 + norm_status;
    return llmm_matvec_stream(model, tensor_id, normalized, output, rows, columns, scratch, scratch_bytes);
}

pub export fn llmm_apply_rope(vector: [*]f32, heads: usize, position: usize) c_int {
    if (heads == 0 or heads * HeadDim > 8 * Width) return -1;
    if (comptime use_p4_acceleration) {
        return llmm_p4_rope(vector, heads, position);
    }
    const position_value = @as(f32, @floatFromInt(position));
    var head: usize = 0;
    while (head < heads) : (head += 1) {
        var pair: usize = 0;
        while (pair < HeadDim / 2) : (pair += 1) {
            const exponent = @as(f32, @floatFromInt(pair * 2)) / @as(f32, @floatFromInt(HeadDim));
            const frequency = @exp(-exponent * @log(RopeTheta));
            const angle = position_value * frequency;
            const cosine = @cos(angle);
            const sine = @sin(angle);
            const offset = head * HeadDim + pair * 2;
            const first = vector[offset];
            const second = vector[offset + 1];
            vector[offset] = first * cosine - second * sine;
            vector[offset + 1] = first * sine + second * cosine;
        }
    }
    return 0;
}

fn encodeFp16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const sign: u16 = @intCast((bits >> 16) & 0x8000);
    const fraction = bits & 0x007fffff;
    var exponent: i32 = @intCast((bits >> 23) & 0xff);
    if (exponent == 0xff) {
        const nan_payload: u16 = if (fraction == 0) 0 else 0x0200;
        return sign | 0x7c00 | nan_payload;
    }
    exponent = exponent - 127 + 15;
    if (exponent <= 0) return sign;
    if (exponent >= 31) return sign | 0x7c00;
    return sign | (@as(u16, @intCast(exponent)) << 10) | @as(u16, @intCast(fraction >> 13));
}

pub export fn llmm_quantize_i8(input: [*]const f32, output: [*]i8, elements: usize, scale: *u16) c_int {
    if (elements == 0) return -1;
    var maximum: f32 = 0.0;
    var index: usize = 0;
    while (index < elements) : (index += 1) {
        const absolute = if (input[index] < 0.0) -input[index] else input[index];
        if (absolute > maximum) maximum = absolute;
    }
    const value_scale: f32 = if (maximum == 0.0) 1.0 else maximum / 127.0;
    scale.* = encodeFp16(value_scale);
    index = 0;
    while (index < elements) : (index += 1) {
        var quantized = @round(input[index] / value_scale);
        if (quantized > 127.0) quantized = 127.0;
        if (quantized < -127.0) quantized = -127.0;
        output[index] = @intCast(@as(i32, @intFromFloat(quantized)));
    }
    return 0;
}

pub export fn llmm_attention_gqa(
    query: [*]const f32,
    keys: [*]const i8,
    key_scales: [*]const u16,
    values: [*]const i8,
    value_scales: [*]const u16,
    tokens: usize,
    output: [*]f32,
) c_int {
    if (tokens == 0) return -1;
    if (comptime use_p4_acceleration) {
        return llmm_p4_attention_gqa(query, keys, key_scales, values, value_scales, tokens, output);
    }
    var head: usize = 0;
    while (head < Heads) : (head += 1) {
        var accumulator: [HeadDim]f32 = [_]f32{0.0} ** HeadDim;
        var maximum: f32 = -3.402823466e38;
        var normalizer: f32 = 0.0;
        const kv_head = head / (Heads / KvHeads);
        var token: usize = 0;
        while (token < tokens) : (token += 1) {
            const cache_offset = (token * KvHeads + kv_head) * HeadDim;
            const key_scale = decodeFp16Bits(key_scales[token * KvHeads + kv_head]);
            const value_scale = decodeFp16Bits(value_scales[token * KvHeads + kv_head]);
            var score: f32 = 0.0;
            var dimension: usize = 0;
            while (dimension < HeadDim) : (dimension += 1) score += query[head * HeadDim + dimension] * @as(f32, @floatFromInt(keys[cache_offset + dimension])) * key_scale;
            score *= AttentionScale;
            var weight: f32 = undefined;
            if (score > maximum) {
                if (normalizer != 0.0) {
                    const factor = @exp(maximum - score);
                    normalizer *= factor;
                    dimension = 0;
                    while (dimension < HeadDim) : (dimension += 1) accumulator[dimension] *= factor;
                }
                maximum = score;
                weight = 1.0;
            } else {
                weight = @exp(score - maximum);
            }
            normalizer += weight;
            dimension = 0;
            while (dimension < HeadDim) : (dimension += 1) accumulator[dimension] += weight * @as(f32, @floatFromInt(values[cache_offset + dimension])) * value_scale;
        }
        var dimension: usize = 0;
        while (dimension < HeadDim) : (dimension += 1) output[head * HeadDim + dimension] = accumulator[dimension] / normalizer;
    }
    return 0;
}

pub export fn llmm_router_top1(logits: [*]const f32, experts: usize, selected: *u32, probability: *f32) c_int {
    if (experts == 0 or experts > Experts) return -1;
    if (comptime use_p4_acceleration) {
        return llmm_p4_router_top1(logits, experts, selected, probability);
    }
    var maximum = logits[0];
    var index: usize = 1;
    while (index < experts) : (index += 1) {
        if (logits[index] > maximum) maximum = logits[index];
    }
    var normalizer: f32 = 0.0;
    index = 0;
    while (index < experts) : (index += 1) normalizer += @exp(logits[index] - maximum);
    var best: usize = 0;
    var best_probability: f32 = 0.0;
    index = 0;
    while (index < experts) : (index += 1) {
        const current = @exp(logits[index] - maximum) / normalizer;
        if (current > best_probability) {
            best_probability = current;
            best = index;
        }
    }
    selected.* = @intCast(best);
    probability.* = best_probability;
    return 0;
}

pub export fn llmm_silu_mul(gate: [*]f32, up: [*]const f32, elements: usize) c_int {
    if (comptime use_p4_acceleration) {
        return llmm_p4_silu_mul(gate, up, elements);
    }
    var index: usize = 0;
    while (index < elements) : (index += 1) gate[index] = (gate[index] / (1.0 + @exp(-gate[index]))) * up[index];
    return 0;
}

pub export fn llmm_gelu_mul(value: [*]f32, multiplier: [*]const f32, elements: usize) c_int {
    if (comptime use_p4_acceleration) {
        return llmm_p4_gelu_mul(value, multiplier, elements);
    }
    var index: usize = 0;
    while (index < elements) : (index += 1) {
        const x = value[index];
        const cube = x * x * x;
        const argument = 0.7978845608028654 * (x + 0.044715 * cube);
        const tanh_argument: f32 = if (argument > 10.0)
            1.0
        else if (argument < -10.0)
            -1.0
        else blk: {
            const exponential = @exp(2.0 * argument);
            break :blk (exponential - 1.0) / (exponential + 1.0);
        };
        const gelu = 0.5 * x * (1.0 + tanh_argument);
        value[index] = gelu * multiplier[index];
    }
    return 0;
}

fn pleEmbeddingSliceNormalized(
    model: *const Handle,
    token: u32,
    layer: u32,
    normalized_source: [*]const f32,
    projected: [*]f32,
    table: [*]f32,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    if (layer >= Layers) return -1;
    var status = llmm_matvec_slice_stream(model, TopLevelPleProjection, normalized_source, projected, @as(usize, layer) * PleDim, PleDim, Width, scratch, scratch_bytes);
    if (status != 0) return -20 + status;
    const projection_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Width)));
    var index: usize = 0;
    while (index < PleDim) : (index += 1) projected[index] *= projection_scale;
    status = llmm_rmsnorm_stream(model, TopLevelPleProjectionNorm, projected, projected, PleDim, RmsEpsilon, scratch, scratch_bytes);
    if (status != 0) return -30 + status;
    status = llmm_ple_slice_stream(model, token, layer, table, PleDim, scratch, scratch_bytes);
    if (status != 0) return -40 + status;
    const table_scale = @sqrt(@as(f32, @floatFromInt(PleDim)));
    const combine_scale: f32 = 0.7071067811865475;
    index = 0;
    while (index < PleDim) : (index += 1) projected[index] = (projected[index] + table[index] * table_scale) * combine_scale;
    return 0;
}

pub export fn llmm_ple_embedding_slice(
    model: *const Handle,
    token: u32,
    layer: u32,
    source: [*]const f32,
    projected: [*]f32,
    table: [*]f32,
    normalized: [*]f32,
    scratch: [*]u8,
    scratch_bytes: usize,
) c_int {
    const status = llmm_rmsnorm_stream(model, TopLevelPleNorm, source, normalized, Width, RmsEpsilon, scratch, scratch_bytes);
    if (status != 0) return -10 + status;
    return pleEmbeddingSliceNormalized(model, token, layer, normalized, projected, table, scratch, scratch_bytes);
}

pub export fn llmm_layer_step(
    model: *const Handle,
    layer: u32,
    position: usize,
    hidden: [*]f32,
    ple_vector: [*]const f32,
    query: [*]f32,
    key: [*]f32,
    value: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    selected_expert: *u32,
    selected_probability: *f32,
) c_int {
    if (layer >= Layers or position >= cache_capacity) return -1;
    var status: c_int = undefined;
    var phase_started = profileStart();

    status = llmm_matvec_norm_stream(model, tensorIdAttentionWeight(layer, 2), tensorIdAttentionNorm(layer, 2), hidden, query, Width, Width, attended, Width, scratch, scratch_bytes);
    if (status != 0) return -10 + status;
    status = llmm_matvec_norm_stream(model, tensorIdAttentionWeight(layer, 0), tensorIdAttentionNorm(layer, 0), hidden, key, KvHeads * HeadDim, Width, attended, Width, scratch, scratch_bytes);
    if (status != 0) return -20 + status;
    if (llmm_apply_rope(query, Heads, position) != 0 or llmm_apply_rope(key, KvHeads, position) != 0) return -40;

    var kv_head: usize = 0;
    while (kv_head < KvHeads) : (kv_head += 1) {
        const cache_index = (position * KvHeads + kv_head) * HeadDim;
        if (llmm_quantize_i8(key + kv_head * HeadDim, keys + cache_index, HeadDim, &key_scales[position * KvHeads + kv_head]) != 0) return -41;
    }
    status = llmm_matvec_norm_stream(model, tensorIdAttentionWeight(layer, 3), tensorIdAttentionNorm(layer, 3), hidden, value, KvHeads * HeadDim, Width, attended, Width, scratch, scratch_bytes);
    if (status != 0) return -30 + status;
    kv_head = 0;
    while (kv_head < KvHeads) : (kv_head += 1) {
        const cache_index = (position * KvHeads + kv_head) * HeadDim;
        if (llmm_quantize_i8(value + kv_head * HeadDim, values + cache_index, HeadDim, &value_scales[position * KvHeads + kv_head]) != 0) return -42;
    }
    profileFinish(phase_started, .qkv);

    phase_started = profileStart();
    status = llmm_attention_gqa(query, keys, key_scales, values, value_scales, position + 1, attended);
    if (status != 0) return -50 + status;
    profileFinish(phase_started, .attention);

    phase_started = profileStart();
    status = llmm_matvec_norm_stream(model, tensorIdAttentionWeight(layer, 1), tensorIdAttentionNorm(layer, 1), attended, query, Width, Width, attended, Width, scratch, scratch_bytes);
    if (status != 0) return -60 + status;
    var index: usize = 0;
    while (index < Width) : (index += 1) hidden[index] += query[index];
    profileFinish(phase_started, .attention_output);

    phase_started = profileStart();
    status = llmm_matvec_norm_stream(model, tensorIdRouterWeight(layer), tensorIdRouterNorm(layer), hidden, query, Experts, Width, attended, Width, scratch, scratch_bytes);
    if (status != 0) return -70 + status;
    if (llmm_router_top1(query, Experts, selected_expert, selected_probability) != 0) return -71;
    profileFinish(phase_started, .router);

    phase_started = profileStart();
    const selected_weight = selected_probability.*;
    const expert = selected_expert.*;
    status = llmm_matvec_norm_stream(model, tensorIdExpertWeight(layer, expert, 1), tensorIdExpertNorm(layer, expert, 1), hidden, expert_gate, Ffn, Width, query, Width, scratch, scratch_bytes);
    if (status != 0) return -80 + status;
    status = llmm_matvec_norm_stream(model, tensorIdExpertWeight(layer, expert, 2), tensorIdExpertNorm(layer, expert, 2), hidden, expert_up, Ffn, Width, query, Width, scratch, scratch_bytes);
    if (status != 0) return -90 + status;
    if (llmm_silu_mul(expert_gate, expert_up, Ffn) != 0) return -91;
    status = llmm_matvec_norm_stream(model, tensorIdExpertWeight(layer, expert, 0), tensorIdExpertNorm(layer, expert, 0), expert_gate, attended, Width, Ffn, expert_up, Ffn, scratch, scratch_bytes);
    if (status != 0) return -100 + status;
    index = 0;
    while (index < Width) : (index += 1) {
        attended[index] *= selected_weight;
        hidden[index] += attended[index];
    }
    profileFinish(phase_started, .expert);

    phase_started = profileStart();
    status = llmm_matvec_norm_stream(model, tensorIdPleWeight(layer, 0), tensorIdPleGateNorm(layer), hidden, ple_gate, PleDim, Width, query, Width, scratch, scratch_bytes);
    if (status != 0) return -110 + status;
    if (llmm_gelu_mul(ple_gate, ple_vector, PleDim) != 0) return -111;
    status = llmm_matvec_norm_stream(model, tensorIdPleWeight(layer, 1), tensorIdPleProjectionNorm(layer), ple_gate, attended, Width, PleDim, query, PleDim, scratch, scratch_bytes);
    if (status != 0) return -120 + status;
    status = llmm_rmsnorm_stream(model, tensorIdPleOutputNorm(layer), attended, attended, Width, RmsEpsilon, scratch, scratch_bytes);
    if (status != 0) return -130 + status;
    index = 0;
    while (index < Width) : (index += 1) hidden[index] += attended[index];
    profileFinish(phase_started, .ple_adapter);
    return 0;
}

fn insertTopCandidate(candidates: [*]Candidate, count: *usize, capacity: usize, candidate: Candidate) void {
    if (count.* == capacity and candidate.logit <= candidates[capacity - 1].logit) return;
    var slot: usize = if (count.* < capacity) count.* else capacity - 1;
    if (count.* < capacity) count.* += 1;
    while (slot > 0 and candidate.logit > candidates[slot - 1].logit) : (slot -= 1) {
        candidates[slot] = candidates[slot - 1];
    }
    candidates[slot] = candidate;
}

// Scores the tied Q8 embedding rows after the model's final RMSNorm. Only the
// requested top candidates survive the scan; no vocabulary-sized logits exist.
pub export fn llmm_output_head_topk(
    model: *const Handle,
    hidden: [*]const f32,
    normalized: [*]f32,
    scratch: [*]u8,
    scratch_bytes: usize,
    candidates: [*]Candidate,
    candidate_capacity: usize,
    requested_top_k: usize,
    candidate_count: *usize,
) c_int {
    if (model.reader == null or recordStorage(model, 1) != q8 or recordStorage(model, 2) != fp16 or
        recordElements(model, 1) != Vocab * Width or recordElements(model, 2) != Vocab) return -1;
    if (candidate_capacity == 0 or candidate_capacity > MaxTopK or requested_top_k == 0 or requested_top_k > candidate_capacity) return -2;
    const norm_status = llmm_rmsnorm_stream(model, OutputNorm, hidden, normalized, Width, RmsEpsilon, scratch, scratch_bytes);
    if (norm_status != 0) return -20 + norm_status;

    if (comptime use_p4_acceleration) {
        return llmm_p4_output_head_topk(model, normalized, candidates, candidate_capacity, requested_top_k, candidate_count);
    }

    const rows_per_tile = scratch_bytes / (Width + 2);
    if (rows_per_tile == 0) return -3;
    const read = model.reader.?;
    const weights_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, 1) + 14)));
    const scales_offset = @as(usize, @intCast(le64(model.artifact, recordOffset(model, 2) + 14)));
    var count: usize = 0;
    var row_start: usize = 0;
    while (row_start < Vocab) {
        const rows = if (rows_per_tile < Vocab - row_start) rows_per_tile else Vocab - row_start;
        const weights_bytes = rows * Width;
        if (read(model.reader_context, recordRegion(model, 1), weights_offset + row_start * Width, scratch, weights_bytes) != 0) return -4;
        const scale_bytes = scratch + weights_bytes;
        if (read(model.reader_context, recordRegion(model, 2), scales_offset + row_start * 2, scale_bytes, rows * 2) != 0) return -5;
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const scale = decodeFp16(scale_bytes + row * 2);
            var logit: f32 = 0.0;
            var column: usize = 0;
            while (column < Width) : (column += 1) {
                const weight: i8 = @bitCast(scratch[row * Width + column]);
                logit += normalized[column] * @as(f32, @floatFromInt(weight)) * scale;
            }
            insertTopCandidate(candidates, &count, requested_top_k, .{ .token = @intCast(row_start + row), .logit = logit });
        }
        row_start += rows;
    }
    candidate_count.* = count;
    return 0;
}

pub export fn llmm_output_head_argmax(
    model: *const Handle,
    hidden: [*]const f32,
    normalized: [*]f32,
    scratch: [*]u8,
    scratch_bytes: usize,
    next_token: *u32,
    next_logit: *f32,
) c_int {
    var candidate: [1]Candidate = undefined;
    var count: usize = 0;
    const status = llmm_output_head_topk(model, hidden, normalized, scratch, scratch_bytes, candidate[0..].ptr, 1, 1, &count);
    if (status != 0 or count != 1) return -10 + status;
    next_token.* = candidate[0].token;
    next_logit.* = candidate[0].logit;
    return 0;
}

fn nextRandom(state: *u32) u32 {
    var value = state.*;
    if (value == 0) value = 0x6d2b79f5;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    state.* = value;
    return value;
}

pub export fn llmm_sample_topk(
    candidates: [*]const Candidate,
    candidate_count: usize,
    temperature: f32,
    random_state: *u32,
    next_token: *u32,
    next_logit: *f32,
) c_int {
    if (candidate_count == 0 or candidate_count > MaxTopK) return -1;
    if (temperature == 0.0) {
        next_token.* = candidates[0].token;
        next_logit.* = candidates[0].logit;
        return 0;
    }
    if (!(temperature > 0.0)) return -2;
    const inverse_temperature = 1.0 / temperature;
    const maximum = candidates[0].logit * inverse_temperature;
    var normalizer: f32 = 0.0;
    var index: usize = 0;
    while (index < candidate_count) : (index += 1) normalizer += @exp(candidates[index].logit * inverse_temperature - maximum);
    const random = @as(f32, @floatFromInt(nextRandom(random_state))) * 0.00000000023283064365386963;
    const threshold = random * normalizer;
    var cumulative: f32 = 0.0;
    index = 0;
    while (index < candidate_count) : (index += 1) {
        cumulative += @exp(candidates[index].logit * inverse_temperature - maximum);
        if (threshold < cumulative or index + 1 == candidate_count) {
            next_token.* = candidates[index].token;
            next_logit.* = candidates[index].logit;
            return 0;
        }
    }
    return -3;
}

// Runs one causal token through all twelve layers. The immutable embedding copy
// is normalized once because every layer consumes the same PLE projection input.
const Sampler = struct {
    candidates: [*]Candidate,
    candidate_capacity: usize,
    top_k: usize,
    temperature: f32,
    random_state: *u32,
};

// prepare_ple: 0 = embedding already RMSNormed, 1 = embed token into embedding only
// (worker keeps the inbound residual in hidden), 2 = embed token into hidden+embedding
// (coordinator / full token step).
fn hiddenLayersStep(
    model: *const Handle,
    token: u32,
    position: usize,
    layer_begin: u32,
    layer_end: u32,
    prepare_ple: u32,
    hidden: [*]f32,
    embedding: [*]f32,
    ple_vector: [*]f32,
    query: [*]f32,
    key: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    routes: [*]u32,
    routes_len: usize,
    score_output: u32,
    layer_trace: ?[*]f32,
    next_token: *u32,
    next_logit: *f32,
    sampler: ?Sampler,
) c_int {
    if (token >= Vocab or position >= Context or cache_capacity < Context) return -1;
    if (layer_end > Layers or layer_begin >= layer_end) return -2;
    if (comptime enable_debug) {
        if (routes_len < Layers) return -1;
    }
    if (comptime enable_debug) {
        if (active_profile) |profile| {
            if (prepare_ple == 2) profile.token_steps[active_profile_phase] += 1;
            if (score_output != 0) profile.scored_steps[active_profile_phase] += 1;
        }
    }
    var status: c_int = 0;
    if (prepare_ple == 1 or prepare_ple == 2) {
        var phase_started = profileStart();
        const embed_out = if (prepare_ple == 2) hidden else embedding;
        status = llmm_embedding_row_stream(model, token, embed_out, Width);
        if (status != 0) return -10 + status;
        if (prepare_ple == 2) {
            var index: usize = 0;
            while (index < Width) : (index += 1) embedding[index] = hidden[index];
        }
        profileFinish(phase_started, .embedding);
        phase_started = profileStart();
        status = llmm_rmsnorm_stream(model, TopLevelPleNorm, embedding, embedding, Width, RmsEpsilon, scratch, scratch_bytes);
        if (status != 0) return -900 + status;
        profileFinish(phase_started, .ple_embedding);
    }

    var layer: u32 = layer_begin;
    while (layer < layer_end) : (layer += 1) {
        const phase_started = profileStart();
        status = pleEmbeddingSliceNormalized(model, token, layer, embedding, ple_vector, ple_gate, scratch, scratch_bytes);
        if (status != 0) return -1_000 - @as(c_int, @intCast(layer)) * 100 + status;
        profileFinish(phase_started, .ple_embedding);
        var selected_expert: u32 = 0;
        var previous_expert: u32 = 0;
        if (comptime enable_debug) {
            if (position != 0) previous_expert = routes[@as(usize, layer)];
        }
        const selected_expert_out = if (comptime enable_debug)
            &routes[@as(usize, layer)]
        else
            &selected_expert;
        status = llmm_layer_step(
            model,
            layer,
            position,
            hidden,
            ple_vector,
            query,
            key,
            key,
            attended,
            ple_gate,
            expert_gate,
            expert_up,
            keys + @as(usize, layer) * KvCacheElements,
            key_scales + @as(usize, layer) * KvScaleElements,
            values + @as(usize, layer) * KvCacheElements,
            value_scales + @as(usize, layer) * KvScaleElements,
            cache_capacity,
            scratch,
            scratch_bytes,
            selected_expert_out,
            next_logit,
        );
        if (status != 0) return -40_000 - @as(c_int, @intCast(layer)) * 1_000 + status;
        if (comptime enable_debug) {
            if (position != 0) {
                if (active_profile) |profile| {
                    profile.expert_route_total += 1;
                    if (previous_expert == selected_expert_out.*) profile.expert_route_same += 1;
                }
            }
            if (layer_trace) |trace| {
                var index: usize = 0;
                while (index < Width) : (index += 1) trace[@as(usize, layer) * Width + index] = hidden[index];
            }
        }
    }
    if (score_output == 0) return 0;
    if (sampler) |value| {
        var candidate_count: usize = 0;
        var phase_started = profileStart();
        status = llmm_output_head_topk(model, hidden, query, scratch, scratch_bytes, value.candidates, value.candidate_capacity, value.top_k, &candidate_count);
        if (status != 0) return -500 + status;
        profileFinish(phase_started, .output_head);
        phase_started = profileStart();
        status = llmm_sample_topk(value.candidates, candidate_count, value.temperature, value.random_state, next_token, next_logit);
        if (status != 0) return -600 + status;
        profileFinish(phase_started, .sampling);
        return 0;
    }
    const phase_started = profileStart();
    status = llmm_output_head_argmax(model, hidden, query, scratch, scratch_bytes, next_token, next_logit);
    if (status != 0) return -500 + status;
    profileFinish(phase_started, .output_head);
    return 0;
}

fn tokenStep(
    model: *const Handle,
    token: u32,
    position: usize,
    hidden: [*]f32,
    embedding: [*]f32,
    ple_vector: [*]f32,
    query: [*]f32,
    key: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    routes: [*]u32,
    routes_len: usize,
    score_output: u32,
    layer_trace: ?[*]f32,
    next_token: *u32,
    next_logit: *f32,
    sampler: ?Sampler,
) c_int {
    return hiddenLayersStep(
        model,
        token,
        position,
        0,
        Layers,
        2,
        hidden,
        embedding,
        ple_vector,
        query,
        key,
        attended,
        ple_gate,
        expert_gate,
        expert_up,
        keys,
        key_scales,
        values,
        value_scales,
        cache_capacity,
        scratch,
        scratch_bytes,
        routes,
        routes_len,
        score_output,
        layer_trace,
        next_token,
        next_logit,
        sampler,
    );
}

pub export fn llmm_token_step(
    model: *const Handle,
    token: u32,
    position: usize,
    hidden: [*]f32,
    embedding: [*]f32,
    ple_vector: [*]f32,
    query: [*]f32,
    key: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    routes: [*]u32,
    routes_len: usize,
    score_output: u32,
    layer_trace: ?[*]f32,
    next_token: *u32,
    next_logit: *f32,
) c_int {
    return tokenStep(model, token, position, hidden, embedding, ple_vector, query, key, attended, ple_gate, expert_gate, expert_up, keys, key_scales, values, value_scales, cache_capacity, scratch, scratch_bytes, routes, routes_len, score_output, layer_trace, next_token, next_logit, null);
}

pub export fn llmm_token_step_sampled(
    model: *const Handle,
    token: u32,
    position: usize,
    hidden: [*]f32,
    embedding: [*]f32,
    ple_vector: [*]f32,
    query: [*]f32,
    key: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    routes: [*]u32,
    routes_len: usize,
    score_output: u32,
    candidates: [*]Candidate,
    candidate_capacity: usize,
    top_k: usize,
    temperature: f32,
    random_state: *u32,
    layer_trace: ?[*]f32,
    next_token: *u32,
    next_logit: *f32,
) c_int {
    const sampler = Sampler{
        .candidates = candidates,
        .candidate_capacity = candidate_capacity,
        .top_k = top_k,
        .temperature = temperature,
        .random_state = random_state,
    };
    return tokenStep(model, token, position, hidden, embedding, ple_vector, query, key, attended, ple_gate, expert_gate, expert_up, keys, key_scales, values, value_scales, cache_capacity, scratch, scratch_bytes, routes, routes_len, score_output, layer_trace, next_token, next_logit, sampler);
}

pub export fn llmm_hidden_layers_step(
    model: *const Handle,
    token: u32,
    position: usize,
    layer_begin: u32,
    layer_end: u32,
    prepare_ple: u32,
    hidden: [*]f32,
    embedding: [*]f32,
    ple_vector: [*]f32,
    query: [*]f32,
    key: [*]f32,
    attended: [*]f32,
    ple_gate: [*]f32,
    expert_gate: [*]f32,
    expert_up: [*]f32,
    keys: [*]i8,
    key_scales: [*]u16,
    values: [*]i8,
    value_scales: [*]u16,
    cache_capacity: usize,
    scratch: [*]u8,
    scratch_bytes: usize,
    routes: [*]u32,
    routes_len: usize,
    score_output: u32,
    layer_trace: ?[*]f32,
    next_token: *u32,
    next_logit: *f32,
) c_int {
    return hiddenLayersStep(
        model,
        token,
        position,
        layer_begin,
        layer_end,
        prepare_ple,
        hidden,
        embedding,
        ple_vector,
        query,
        key,
        attended,
        ple_gate,
        expert_gate,
        expert_up,
        keys,
        key_scales,
        values,
        value_scales,
        cache_capacity,
        scratch,
        scratch_bytes,
        routes,
        routes_len,
        score_output,
        layer_trace,
        next_token,
        next_logit,
        null,
    );
}
