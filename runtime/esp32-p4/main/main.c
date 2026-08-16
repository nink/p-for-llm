#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_attr.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_private/esp_clk.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "esp_log.h"
#if LLMM_DEBUG
#include "esp_async_memcpy.h"
#include "riscv/csr.h"
#include "soc/cache_struct.h"
#endif
#if LLMM_HOST_UART
#include "driver/uart.h"
#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#else
#include "driver/usb_serial_jtag.h"
#endif
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "llmm_compress.h"
#include "llmm_eth.h"
#include "llmm_pack.h"

#ifndef LLMM_HOST_UART
#define LLMM_HOST_UART 0
#endif

#if LLMM_HOST_UART
#define LLMM_HOST_UART_PORT UART_NUM_0
#define LLMM_HOST_UART_TX_GPIO 37
#define LLMM_HOST_UART_RX_GPIO 38
#define LLMM_HOST_UART_BAUD 460800
#define LLMM_SD_MOUNT_POINT "/sdcard"
#define LLMM_SD_PAYLOAD_PATH "/sdcard/pfor-psram.bin"
#define LLMM_SD_PACK_PATH "/sdcard/canada.kpack"
#define LLMM_SD_PWR_GPIO GPIO_NUM_45
#define LLMM_SD_LDO_CHANNEL_ID 4
#define LLMM_SD_PUT_CHUNK 32768U
#endif

#define LLMM_CHAT_END_TOKEN 32755U

#if LLMM_DEBUG
extern float zig_calculate_pi(uint32_t iterations);
extern uint32_t zig_cpu_benchmark(uint32_t iterations);
extern uint32_t zig_divide_benchmark(uint32_t iterations);
extern float zig_float_benchmark(uint32_t iterations);
#endif

typedef struct {
    const uint8_t *artifact;
    size_t bytes, index_offset, flash_offset, psram_offset, flash_bytes, psram_bytes;
    uint32_t position;
    void *reader;
    void *reader_context;
} llmm_handle_t;

typedef struct {
    const uint8_t *asset;
    size_t bytes;
    size_t byte_ids_offset;
    size_t token_offsets_offset;
    size_t token_bytes_offset;
    size_t token_bytes;
    size_t merge_offset;
    uint32_t merge_count;
    uint32_t max_token_bytes;
    uint32_t eos_token;
} llmm_tokenizer_t;

typedef struct {
    uint32_t token;
    float logit;
} llmm_candidate_t;

#if LLMM_DEBUG
#define LLMM_PROFILE_PHASES 2U
#define LLMM_PROFILE_STAGES 11U
#define LLMM_PROFILE_ACCEL_KINDS 3U
#define LLMM_PROFILE_CORES 2U
#define LLMM_PROFILE_CACHE_GROUPS 8U
#define LLMM_PROFILE_CACHE_EVENTS 5U
#define LLMM_PROFILE_REGIONS 2U
#define LLMM_PROFILE_TRAFFIC_KINDS 3U
#define LLMM_PROFILE_USB_DIRECTIONS 2U

typedef struct {
    uint64_t stage_cycles[LLMM_PROFILE_PHASES][LLMM_PROFILE_STAGES];
    uint64_t stage_max_cycles[LLMM_PROFILE_PHASES][LLMM_PROFILE_STAGES];
    uint64_t stage_calls[LLMM_PROFILE_PHASES][LLMM_PROFILE_STAGES];
    uint64_t step_cycles[LLMM_PROFILE_PHASES];
    uint64_t step_max_cycles[LLMM_PROFILE_PHASES];
    uint64_t step_min_cycles[LLMM_PROFILE_PHASES];
    uint64_t step_calls[LLMM_PROFILE_PHASES];
    uint64_t owner_wait_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t owner_wait_max_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t owner_wait_calls[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t worker_busy_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t worker_busy_max_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t worker_busy_calls[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t dispatch_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t dispatch_max_cycles[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t dispatch_calls[LLMM_PROFILE_ACCEL_KINDS];
    uint64_t worker_idle_cycles;
    uint64_t worker_idle_max_cycles;
    uint64_t worker_idle_calls;
    uint64_t cpu_cycles[LLMM_PROFILE_CORES];
    uint64_t cpu_instructions[LLMM_PROFILE_CORES];
    uint64_t cpu_branch_misses[LLMM_PROFILE_CORES];
    uint64_t cpu_conditional_branches[LLMM_PROFILE_CORES];
    uint64_t cpu_stores[LLMM_PROFILE_CORES];
    uint64_t cache_events[LLMM_PROFILE_CACHE_GROUPS][LLMM_PROFILE_CACHE_EVENTS];
    uint64_t reader_bytes[LLMM_PROFILE_REGIONS];
    uint64_t reader_cycles[LLMM_PROFILE_REGIONS];
    uint64_t reader_max_cycles[LLMM_PROFILE_REGIONS];
    uint64_t reader_calls[LLMM_PROFILE_REGIONS];
    uint64_t traffic_bytes[LLMM_PROFILE_TRAFFIC_KINDS];
    uint64_t traffic_cycles[LLMM_PROFILE_TRAFFIC_KINDS];
    uint64_t traffic_max_cycles[LLMM_PROFILE_TRAFFIC_KINDS];
    uint64_t traffic_calls[LLMM_PROFILE_TRAFFIC_KINDS];
    uint64_t usb_bytes[LLMM_PROFILE_USB_DIRECTIONS];
    uint64_t usb_cycles[LLMM_PROFILE_USB_DIRECTIONS];
    uint64_t usb_max_cycles[LLMM_PROFILE_USB_DIRECTIONS];
    uint64_t usb_calls[LLMM_PROFILE_USB_DIRECTIONS];
    uint64_t total_cycles;
    uint64_t ttft_cycles;
    uint64_t token_steps[LLMM_PROFILE_PHASES];
    uint64_t scored_steps[LLMM_PROFILE_PHASES];
    uint64_t kv_reused_tokens;
    uint64_t kv_appended_tokens;
    uint64_t kv_occupancy_tokens;
    uint64_t kv_evictions;
    uint64_t expert_route_same;
    uint64_t expert_route_total;
    uint64_t rope_same;
    uint64_t rope_sequential;
    uint64_t rope_rebuild;
    uint64_t internal_free_bytes;
    uint64_t internal_largest_bytes;
    uint64_t internal_min_free_bytes;
    uint64_t psram_free_bytes;
    uint64_t psram_largest_bytes;
    uint64_t psram_min_free_bytes;
    uint64_t stack_free_bytes[LLMM_PROFILE_CORES];
} llmm_profile_t;

_Static_assert(sizeof(llmm_profile_t) == 1640U, "profile wire layout changed");
#endif

typedef int (*llmm_reader_fn)(void *, uint8_t, size_t, uint8_t *, size_t);

extern int llmm_init_manifest(llmm_handle_t *model, const uint8_t *manifest, size_t bytes);
extern void llmm_set_reader(llmm_handle_t *model, void *context, llmm_reader_fn reader);
extern int llmm_tokenizer_init(llmm_tokenizer_t *tokenizer, const uint8_t *asset, size_t bytes);
extern int llmm_tokenizer_encode(const llmm_tokenizer_t *tokenizer, const uint8_t *text, size_t text_bytes,
                                 uint32_t *output, size_t output_capacity, size_t *output_count,
                                 uint16_t *pieces, size_t piece_capacity);
extern int llmm_tokenizer_decode(const llmm_tokenizer_t *tokenizer, uint32_t token,
                                 uint8_t *output, size_t output_capacity, size_t *output_bytes);
extern uint32_t llmm_tokenizer_eos(const llmm_tokenizer_t *tokenizer);
extern size_t llmm_tokenizer_max_token_bytes(const llmm_tokenizer_t *tokenizer);
extern int llmm_embedding_row_stream(const llmm_handle_t *model, uint32_t token, float *output, size_t output_len);
extern int llmm_ple_embedding_slice(const llmm_handle_t *model, uint32_t token, uint32_t layer, const float *source,
                                    float *projected, float *table, float *normalized, uint8_t *scratch, size_t scratch_bytes);
extern int llmm_ple_slice_stream(const llmm_handle_t *model, uint32_t token, uint32_t layer, float *output,
                                 size_t output_len, uint8_t *scratch, size_t scratch_bytes);
extern int llmm_matvec_stream(const llmm_handle_t *model, uint16_t tensor_id, const float *input, float *output,
                              size_t rows, size_t columns, uint8_t *scratch, size_t scratch_bytes);
extern int llmm_layer_step(const llmm_handle_t *model, uint32_t layer, size_t position, float *hidden,
                           const float *ple_vector, float *query, float *key, float *value, float *attended,
                           float *ple_gate, float *expert_gate, float *expert_up, int8_t *keys,
                           uint16_t *key_scales, int8_t *values,
                           uint16_t *value_scales, size_t cache_capacity, uint8_t *scratch, size_t scratch_bytes,
                           uint32_t *selected_expert, float *selected_probability);
extern int llmm_token_step(const llmm_handle_t *model, uint32_t token, size_t position, float *hidden,
                           float *embedding, float *ple_vector, float *query, float *key, float *attended,
                           float *ple_gate, float *expert_gate, float *expert_up, int8_t *keys,
                           uint16_t *key_scales, int8_t *values, uint16_t *value_scales,
                           size_t cache_capacity, uint8_t *scratch, size_t scratch_bytes,
                           uint32_t *routes, size_t routes_len, uint32_t score_output,
                           float *layer_trace, uint32_t *next_token, float *next_logit);
extern int llmm_token_step_sampled(const llmm_handle_t *model, uint32_t token, size_t position, float *hidden,
                                   float *embedding, float *ple_vector, float *query, float *key, float *attended,
                                   float *ple_gate, float *expert_gate, float *expert_up, int8_t *keys,
                                   uint16_t *key_scales, int8_t *values, uint16_t *value_scales,
                                   size_t cache_capacity, uint8_t *scratch, size_t scratch_bytes,
                                   uint32_t *routes, size_t routes_len, uint32_t score_output,
                                   llmm_candidate_t *candidates, size_t candidate_capacity, size_t top_k,
                                   float temperature, uint32_t *random_state, float *layer_trace,
                                   uint32_t *next_token, float *next_logit);
extern int llmm_hidden_layers_step(const llmm_handle_t *model, uint32_t token, size_t position,
                                   uint32_t layer_begin, uint32_t layer_end, uint32_t prepare_ple, float *hidden,
                                   float *embedding, float *ple_vector, float *query, float *key, float *attended,
                                   float *ple_gate, float *expert_gate, float *expert_up, int8_t *keys,
                                   uint16_t *key_scales, int8_t *values, uint16_t *value_scales,
                                   size_t cache_capacity, uint8_t *scratch, size_t scratch_bytes,
                                   uint32_t *routes, size_t routes_len, uint32_t score_output,
                                   float *layer_trace, uint32_t *next_token, float *next_logit);
#if LLMM_DEBUG
extern void llmm_profile_begin(llmm_profile_t *profile);
extern void llmm_profile_end(void);
extern void llmm_profile_set_phase(uint32_t phase);

uint32_t llmm_p4_cycle_count(void)
{
    return (uint32_t)esp_cpu_get_cycle_count();
}

typedef struct {
    uint32_t cycles;
    uint32_t instructions;
    uint32_t branch_misses;
    uint32_t conditional_branches;
    uint32_t stores;
} llmm_hpm_snapshot_t;

static llmm_profile_t *volatile llmm_active_profile;
static llmm_profile_t llmm_debug_profile;
static uint32_t llmm_profile_started_us;
static uint32_t llmm_step_started;
static uint32_t llmm_step_phase;
static volatile uint32_t llmm_worker_last_finished_us;
static llmm_hpm_snapshot_t llmm_step_hpm;
static uint32_t llmm_cache_last[LLMM_PROFILE_CACHE_GROUPS][LLMM_PROFILE_CACHE_EVENTS];

static inline uint64_t llmm_profile_us_to_cycles(uint32_t elapsed_us)
{
    return (uint64_t)elapsed_us * CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
}

static inline void llmm_profile_stat(uint64_t *cycles, uint64_t *maximum,
                                     uint64_t *calls, uint64_t elapsed)
{
    *cycles += elapsed;
    if (elapsed > *maximum) *maximum = elapsed;
    *calls += 1U;
}

static inline void llmm_hpm_configure(void)
{
    const uint32_t enabled = (1U << 0U) | (1U << 2U) | (1U << 8U) |
                             (1U << 9U) | (1U << 13U);
    RV_WRITE_CSR(mcountinhibit, RV_READ_CSR(mcountinhibit) & ~enabled);
    RV_WRITE_CSR(mhpmevent8, 0x6U);
    RV_WRITE_CSR(mhpmevent9, 0x7U);
    RV_WRITE_CSR(mhpmevent13, 0xBU);
}

static inline llmm_hpm_snapshot_t llmm_hpm_read(void)
{
    return (llmm_hpm_snapshot_t) {
        .cycles = (uint32_t)RV_READ_CSR(mcycle),
        .instructions = (uint32_t)RV_READ_CSR(minstret),
        .branch_misses = (uint32_t)RV_READ_CSR(mhpmcounter8),
        .conditional_branches = (uint32_t)RV_READ_CSR(mhpmcounter9),
        .stores = (uint32_t)RV_READ_CSR(mhpmcounter13),
    };
}

static void llmm_cache_read(uint32_t values[LLMM_PROFILE_CACHE_GROUPS][LLMM_PROFILE_CACHE_EVENTS])
{
    memset(values, 0, LLMM_PROFILE_CACHE_GROUPS * LLMM_PROFILE_CACHE_EVENTS * sizeof(uint32_t));
#define LLMM_CACHE_I(GROUP, LEVEL, BUS) \
    values[GROUP][0] = CACHE.LEVEL##_##BUS##_acs_hit_cnt.val; \
    values[GROUP][1] = CACHE.LEVEL##_##BUS##_acs_miss_cnt.val; \
    values[GROUP][2] = CACHE.LEVEL##_##BUS##_acs_conflict_cnt.val; \
    values[GROUP][3] = CACHE.LEVEL##_##BUS##_acs_nxtlvl_rd_cnt.val
#define LLMM_CACHE_D(GROUP, LEVEL, BUS) \
    LLMM_CACHE_I(GROUP, LEVEL, BUS); \
    values[GROUP][4] = CACHE.LEVEL##_##BUS##_acs_nxtlvl_wr_cnt.val
    LLMM_CACHE_I(0, l1, ibus0);
    LLMM_CACHE_D(1, l1, dbus0);
    LLMM_CACHE_I(2, l2, ibus0);
    LLMM_CACHE_D(3, l2, dbus0);
    LLMM_CACHE_I(4, l1, ibus1);
    LLMM_CACHE_D(5, l1, dbus1);
    LLMM_CACHE_I(6, l2, ibus1);
    LLMM_CACHE_D(7, l2, dbus1);
#undef LLMM_CACHE_D
#undef LLMM_CACHE_I
}

static void llmm_cache_start(void)
{
    CACHE.l1_cache_acs_cnt_ctrl.val = 0x00330000U;
    CACHE.l2_cache_acs_cnt_ctrl.val = 0x33000000U;
    CACHE.l1_cache_acs_cnt_ctrl.val = 0x00000033U;
    CACHE.l2_cache_acs_cnt_ctrl.val = 0x00003300U;
    llmm_cache_read(llmm_cache_last);
}

static void llmm_cache_sample(llmm_profile_t *profile)
{
    uint32_t current[LLMM_PROFILE_CACHE_GROUPS][LLMM_PROFILE_CACHE_EVENTS];
    llmm_cache_read(current);
    for (size_t group = 0; group < LLMM_PROFILE_CACHE_GROUPS; ++group) {
        for (size_t event = 0; event < LLMM_PROFILE_CACHE_EVENTS; ++event) {
            profile->cache_events[group][event] +=
                (uint32_t)(current[group][event] - llmm_cache_last[group][event]);
            llmm_cache_last[group][event] = current[group][event];
        }
    }
}

static void llmm_debug_profile_begin(llmm_profile_t *profile)
{
    memset(profile, 0, sizeof(*profile));
    profile->step_min_cycles[0] = UINT64_MAX;
    profile->step_min_cycles[1] = UINT64_MAX;
    llmm_hpm_configure();
    llmm_profile_begin(profile);
    llmm_active_profile = profile;
    __sync_synchronize();
    llmm_cache_start();
    llmm_profile_started_us = (uint32_t)esp_timer_get_time();
    llmm_worker_last_finished_us = llmm_profile_started_us;
}

static void llmm_debug_profile_end(llmm_profile_t *profile, TaskHandle_t worker_task)
{
    llmm_cache_sample(profile);
    const uint32_t finished_us = (uint32_t)esp_timer_get_time();
    const uint64_t idle = llmm_profile_us_to_cycles(finished_us - llmm_worker_last_finished_us);
    llmm_profile_stat(&profile->worker_idle_cycles, &profile->worker_idle_max_cycles,
                      &profile->worker_idle_calls, idle);
    profile->total_cycles = llmm_profile_us_to_cycles(finished_us - llmm_profile_started_us);
    for (size_t phase = 0; phase < LLMM_PROFILE_PHASES; ++phase) {
        if (profile->step_calls[phase] == 0U) profile->step_min_cycles[phase] = 0U;
    }
    profile->internal_free_bytes = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    profile->internal_largest_bytes = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    profile->internal_min_free_bytes = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    profile->psram_free_bytes = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    profile->psram_largest_bytes = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    profile->psram_min_free_bytes = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    profile->stack_free_bytes[0] = uxTaskGetStackHighWaterMark(NULL);
    profile->stack_free_bytes[1] = worker_task != NULL ? uxTaskGetStackHighWaterMark(worker_task) : 0U;
    __sync_synchronize();
    llmm_active_profile = NULL;
    llmm_profile_end();
}

static inline void llmm_debug_step_begin(uint32_t phase)
{
    llmm_profile_t *profile = llmm_active_profile;
    if (profile == NULL) return;
    llmm_step_phase = phase < LLMM_PROFILE_PHASES ? phase : 0U;
    llmm_profile_set_phase(llmm_step_phase);
    llmm_step_hpm = llmm_hpm_read();
    llmm_step_started = llmm_p4_cycle_count();
}

static inline void llmm_debug_step_end(void)
{
    llmm_profile_t *profile = llmm_active_profile;
    if (profile == NULL) return;
    const llmm_hpm_snapshot_t current = llmm_hpm_read();
    const uint32_t elapsed = llmm_p4_cycle_count() - llmm_step_started;
    const size_t phase = llmm_step_phase;
    profile->step_cycles[phase] += elapsed;
    if (elapsed > profile->step_max_cycles[phase]) profile->step_max_cycles[phase] = elapsed;
    if (elapsed < profile->step_min_cycles[phase]) profile->step_min_cycles[phase] = elapsed;
    profile->step_calls[phase] += 1U;
    profile->cpu_cycles[0] += current.cycles - llmm_step_hpm.cycles;
    profile->cpu_instructions[0] += current.instructions - llmm_step_hpm.instructions;
    profile->cpu_branch_misses[0] += current.branch_misses - llmm_step_hpm.branch_misses;
    profile->cpu_conditional_branches[0] +=
        current.conditional_branches - llmm_step_hpm.conditional_branches;
    profile->cpu_stores[0] += current.stores - llmm_step_hpm.stores;
    llmm_cache_sample(profile);
}

static inline void llmm_debug_stage(uint32_t phase, uint32_t stage, uint32_t elapsed)
{
    llmm_profile_t *profile = llmm_active_profile;
    if (profile == NULL || phase >= LLMM_PROFILE_PHASES || stage >= LLMM_PROFILE_STAGES) return;
    llmm_profile_stat(&profile->stage_cycles[phase][stage],
                      &profile->stage_max_cycles[phase][stage],
                      &profile->stage_calls[phase][stage], elapsed);
}

static inline void llmm_debug_io(uint32_t kind, size_t bytes, uint32_t elapsed,
                                 uint64_t totals[LLMM_PROFILE_REGIONS],
                                 uint64_t cycles[LLMM_PROFILE_REGIONS],
                                 uint64_t maximum[LLMM_PROFILE_REGIONS],
                                 uint64_t calls[LLMM_PROFILE_REGIONS])
{
    if (kind >= LLMM_PROFILE_REGIONS) return;
    totals[kind] += bytes;
    llmm_profile_stat(&cycles[kind], &maximum[kind], &calls[kind], elapsed);
}
#endif

static int llmm_accel_init(void);
static TaskHandle_t llmm_accel_worker_task;

#if LLMM_DEBUG
#define CPU_ITERATIONS 10000000U
#define DIVIDE_ITERATIONS 1000000U
#define FLOAT_ITERATIONS 10000000U
#define MEMORY_BYTES (64U * 1024U)
#define MEMORY_PASSES 100U
#define XESPV_ITERATIONS 10000000U
#define Q4_ELEMENTS 4096U
#define MATRIX_SIZE 32U
#define MATRIX_REPETITIONS 200U
#define VECTOR_SIZE 128U
#define VECTOR_REPETITIONS 1000U
#define PSRAM_BYTES (4U * 1024U * 1024U)
#define PSRAM_STRIDE 32U
#define PSRAM_RANDOM_ACCESSES 16384U
#define DUAL_DEBUG_ITERATIONS 1000000U
#define GEMV_K 256U
#define GEMV_N 128U
#define GEMV_REPETITIONS 100U
#define PREFETCH_PASSES 128U
#define END_TO_END_REPETITIONS 100U
#define RANS_SCALE_BITS 12U
#define RANS_SCALE (1U << RANS_SCALE_BITS)
#define RANS_BYTE_L (1U << 23U)
#define RANS_PACKED_BYTES (64U * 1024U)
#define RANS_TRITS (RANS_PACKED_BYTES * 5U)
#define RANS_REPETITIONS 20U
#endif
#define LLMM_MANIFEST_FLASH_OFFSET 0x255000U
#define LLMM_MANIFEST_FLASH_BYTES 0x1B000U
#define LLMM_TOKENIZER_FLASH_OFFSET 0x110000U
#define LLMM_TOKENIZER_FLASH_BYTES 0x145000U
#define LLMM_PLE_FLASH_OFFSET 0x270000U
#define LLMM_PLE_FLASH_BYTES 13860864U
#define LLMM_TILE_BYTES 2048U
#define LLMM_KV_CONTEXT 1024U
#define LLMM_LAYERS 12U
#define LLMM_TEXT_MAX_BYTES 1024U
/* Long raw prompts arrive over the wire; on-device compress fits TEXT_MAX. */
#define LLMM_RAW_PROMPT_MAX_BYTES LLMM_RAW_MAX_BYTES
#define LLMM_TOP_K_MAX 64U
#define LLMM_WIDTH 192U
#define LLMM_SPLIT_LAYER 6U
#define LLMM_HOP_HEADER_BYTES 24U
#define LLMM_HOP_HIDDEN_BYTES (LLMM_WIDTH * sizeof(float))
#define LLMM_HOP_FLAG_SESSION 0x1U
#define LLMM_VOCAB 32768U
#define LLMM_HEADS 6U
#define LLMM_KV_HEADS 2U
#define LLMM_HEAD_DIM 32U
#define LLMM_MAX_COLUMNS 512U
#define LLMM_TERNARY_SCALE_TENSOR 2275U
#define LLMM_ACCEL_WORKER_STACK_BYTES 3072U
#define LLMM_KV_LAYER_VECTOR_BYTES (LLMM_KV_CONTEXT * 2U * 32U)
#define LLMM_KV_LAYER_SCALE_BYTES (LLMM_KV_CONTEXT * 2U * sizeof(uint16_t))
#define LLMM_KV_STORAGE_BYTES \
    (2U * LLMM_LAYERS * LLMM_KV_LAYER_VECTOR_BYTES + 2U * LLMM_LAYERS * LLMM_KV_LAYER_SCALE_BYTES)
#if LLMM_DEBUG
#define LLMM_TRACE_HIDDEN 0x1U
#define LLMM_TRACE_LAYERS 0x2U
#endif
#define LLMM_TEXT_FLAG_SESSION_EVICTED 0x1U
#define LLMM_USB_PAYLOAD_CHUNK_BYTES (16U * 1024U)
#define LLMM_USB_PAYLOAD_TIMEOUT_MS 5000U

typedef struct {
    float hidden[192];
    float embedding[192];
    float ple_vector[176];
    float query[192];
    float key[64];
    float attended[192];
    float ple_gate[176];
    float expert_gate[512];
    float expert_up[512];
    uint32_t tokens[LLMM_KV_CONTEXT];
    uint16_t bpe_pieces[LLMM_TEXT_MAX_BYTES];
    llmm_candidate_t candidates[LLMM_TOP_K_MAX];
    uint8_t text[LLMM_TEXT_MAX_BYTES];
    uint8_t scratch[LLMM_TILE_BYTES] __attribute__((aligned(16)));
} llmm_layer_workspace_t;

typedef struct {
    const uint8_t *manifest_xip;
    size_t manifest_xip_bytes;
    const uint8_t *ple_xip;
    size_t ple_xip_bytes;
    const uint8_t *tokenizer_xip;
    size_t tokenizer_xip_bytes;
    const uint8_t *psram_payload;
    size_t psram_payload_bytes;
} llmm_storage_t;

typedef struct {
    int8_t *keys;
    int8_t *values;
    uint16_t *key_scales;
    uint16_t *value_scales;
    size_t position;
    uint32_t pending_token;
    uint32_t has_pending;
#if LLMM_DEBUG
    uint32_t routes[LLMM_LAYERS];
#endif
} llmm_inference_state_t;

typedef struct {
    llmm_storage_t storage;
    llmm_handle_t model;
    llmm_tokenizer_t tokenizer;
    llmm_layer_workspace_t *workspace;
    llmm_inference_state_t state;
    uint8_t *psram_payload;
    uint8_t *kv_storage;
    uint8_t *raw_prompt;
    esp_partition_mmap_handle_t manifest_map_handle;
    esp_partition_mmap_handle_t ple_map_handle;
    esp_partition_mmap_handle_t tokenizer_map_handle;
    int init_status;
    uint32_t payload_id;
    uint32_t loaded;
} llmm_runtime_t;

static inline int llmm_usb_read_once(void *buffer, uint32_t bytes, TickType_t wait)
{
#if LLMM_DEBUG
    llmm_profile_t *profile = llmm_active_profile;
    const uint32_t started = profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
#if LLMM_HOST_UART
    int count;
    if (llmm_eth_client_connected()) {
        const int timeout_ms = wait == portMAX_DELAY ? -1 : (int)(wait * portTICK_PERIOD_MS);
        count = llmm_eth_recv(buffer, bytes, timeout_ms);
    } else {
        TickType_t uart_wait = wait;
        if (wait == portMAX_DELAY) uart_wait = pdMS_TO_TICKS(50);
        count = uart_read_bytes(LLMM_HOST_UART_PORT, buffer, bytes, uart_wait);
    }
#else
    const int count = usb_serial_jtag_read_bytes(buffer, bytes, wait);
#endif
#if LLMM_DEBUG
    if (profile != NULL) {
        if (count > 0) profile->usb_bytes[0] += (uint32_t)count;
        llmm_profile_stat(&profile->usb_cycles[0], &profile->usb_max_cycles[0],
                          &profile->usb_calls[0], llmm_p4_cycle_count() - started);
    }
#endif
    return count;
}

static inline int llmm_usb_write_once(const void *buffer, size_t bytes, TickType_t wait)
{
#if LLMM_DEBUG
    llmm_profile_t *profile = llmm_active_profile;
    const uint32_t started = profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
#if LLMM_HOST_UART
    (void)wait;
    int count;
    if (llmm_eth_client_connected()) {
        count = llmm_eth_send(buffer, bytes);
    } else {
        count = uart_write_bytes(LLMM_HOST_UART_PORT, buffer, bytes);
    }
#else
    const int count = usb_serial_jtag_write_bytes(buffer, bytes, wait);
#endif
#if LLMM_DEBUG
    if (profile != NULL) {
        if (count > 0) profile->usb_bytes[1] += (uint32_t)count;
        llmm_profile_stat(&profile->usb_cycles[1], &profile->usb_max_cycles[1],
                          &profile->usb_calls[1], llmm_p4_cycle_count() - started);
    }
#endif
    return count;
}

static int llmm_usb_read_exact(void *buffer, size_t bytes)
{
    uint8_t *cursor = buffer;
    while (bytes != 0) {
        const int count = llmm_usb_read_once(cursor, (uint32_t)bytes, portMAX_DELAY);
        if (count < 0) return -1;
        if (count == 0) continue;
        cursor += count;
        bytes -= (size_t)count;
    }
    return 0;
}

static int llmm_usb_write_exact(const void *buffer, size_t bytes)
{
    const uint8_t *cursor = buffer;
    while (bytes != 0) {
        const int count = llmm_usb_write_once(cursor, bytes, portMAX_DELAY);
        if (count <= 0) return -1;
        cursor += count;
        bytes -= (size_t)count;
    }
    return 0;
}

static int llmm_usb_read_payload(uint8_t *buffer, size_t bytes, uint32_t *payload_crc)
{
    size_t offset = 0;
    uint32_t crc = 0;
    while (offset < bytes) {
        const size_t remaining = bytes - offset;
        const uint32_t requested = remaining < LLMM_USB_PAYLOAD_CHUNK_BYTES
            ? (uint32_t)remaining : LLMM_USB_PAYLOAD_CHUNK_BYTES;
        const int count = llmm_usb_read_once(buffer + offset, requested,
                                             pdMS_TO_TICKS(LLMM_USB_PAYLOAD_TIMEOUT_MS));
        if (count <= 0) return -1;
        crc = esp_rom_crc32_le(crc, buffer + offset, (uint32_t)count);
        offset += (size_t)count;
    }
    *payload_crc = crc;
    return 0;
}

static int llmm_local_reader(void *context, uint8_t region, size_t offset, uint8_t *destination, size_t bytes)
{
    const llmm_storage_t *storage = context;
    if (region == 1U && storage->ple_xip != NULL && offset <= storage->ple_xip_bytes &&
        bytes <= storage->ple_xip_bytes - offset) {
#if LLMM_DEBUG
        llmm_profile_t *profile = llmm_active_profile;
        const uint32_t started = profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
        memcpy(destination, storage->ple_xip + offset, bytes);
#if LLMM_DEBUG
        if (profile != NULL) {
            llmm_debug_io(0U, bytes, llmm_p4_cycle_count() - started,
                          profile->reader_bytes, profile->reader_cycles,
                          profile->reader_max_cycles, profile->reader_calls);
        }
#endif
        return 0;
    }
    if (region == 2U && storage->psram_payload != NULL && offset <= storage->psram_payload_bytes &&
        bytes <= storage->psram_payload_bytes - offset) {
#if LLMM_DEBUG
        llmm_profile_t *profile = llmm_active_profile;
        const uint32_t started = profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
        memcpy(destination, storage->psram_payload + offset, bytes);
#if LLMM_DEBUG
        if (profile != NULL) {
            llmm_debug_io(1U, bytes, llmm_p4_cycle_count() - started,
                          profile->reader_bytes, profile->reader_cycles,
                          profile->reader_max_cycles, profile->reader_calls);
        }
#endif
        return 0;
    }
    return -1;
}

#if LLMM_DEBUG
static uint32_t llmm_f32_checksum(const float *values, size_t count)
{
    uint32_t checksum = 2166136261U;
    for (size_t index = 0; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        checksum = (checksum ^ bits) * 16777619U;
    }
    return checksum;
}
#endif

static void llmm_usb_ready(int status, uint32_t psram_bytes, uint32_t loaded,
                           uint32_t payload_id, uint32_t session_tokens)
{
    uint8_t frame[32] = {'L','L','M','R','D','Y','0','5'};
    const int32_t wire_status = status;
    uint8_t eth_octets[4] = {0, 0, 0, 0};
#if LLMM_HOST_UART
    llmm_eth_ipv4_octets(eth_octets);
#endif
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &psram_bytes, sizeof(psram_bytes));
    memcpy(frame + 16, &loaded, sizeof(loaded));
    memcpy(frame + 20, &payload_id, sizeof(payload_id));
    memcpy(frame + 24, &session_tokens, sizeof(session_tokens));
    memcpy(frame + 28, eth_octets, 4);
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}

static void llmm_usb_loaded(int status, uint32_t psram_bytes, uint32_t payload_id)
{
    uint8_t frame[20] = {'L','L','M','L','O','A','D','5'};
    const int32_t wire_status = status;
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &psram_bytes, sizeof(psram_bytes));
    memcpy(frame + 16, &payload_id, sizeof(payload_id));
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}

static void llmm_usb_control_status(const char magic[8], int status)
{
    uint8_t frame[12];
    const int32_t wire_status = status;
    memcpy(frame, magic, 8);
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}

#if LLMM_DEBUG
static void llmm_usb_done(int status, uint32_t generated, uint32_t checksum, uint32_t elapsed_us)
{
    uint8_t frame[24] = {'L','L','M','D','O','N','E','3'};
    const int32_t wire_status = status;
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &generated, sizeof(generated));
    memcpy(frame + 16, &checksum, sizeof(checksum));
    memcpy(frame + 20, &elapsed_us, sizeof(elapsed_us));
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}
#endif

static void llmm_usb_text_done(int status, uint32_t generated, uint32_t checksum, uint32_t elapsed_us)
{
    uint8_t frame[24] = {'L','L','M','D','O','N','E','5'};
    const int32_t wire_status = status;
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &generated, sizeof(generated));
    memcpy(frame + 16, &checksum, sizeof(checksum));
    memcpy(frame + 20, &elapsed_us, sizeof(elapsed_us));
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}

#if LLMM_DEBUG
static void llmm_usb_profile(const llmm_profile_t *profile)
{
    uint8_t frame[16] = {'L','L','M','P','R','F','0','6'};
    const uint32_t cpu_hz = (uint32_t)esp_clk_cpu_freq();
    const uint32_t profile_bytes = sizeof(*profile);
    memcpy(frame + 8, &cpu_hz, sizeof(cpu_hz));
    memcpy(frame + 12, &profile_bytes, sizeof(profile_bytes));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        (void)llmm_usb_write_exact(profile, sizeof(*profile));
    }
}
#endif

static int llmm_usb_text_tokens(const uint32_t *tokens, uint32_t count, uint32_t flags)
{
    uint8_t frame[16] = {'L','L','M','T','O','K','0','5'};
    memcpy(frame + 8, &count, sizeof(count));
    memcpy(frame + 12, &flags, sizeof(flags));
    if (llmm_usb_write_exact(frame, sizeof(frame)) != 0) return -1;
    return llmm_usb_write_exact(tokens, count * sizeof(*tokens));
}

static int llmm_usb_text_chunk(const uint8_t *bytes, uint32_t count)
{
    uint8_t frame[12] = {'L','L','M','C','H','N','0','5'};
    memcpy(frame + 8, &count, sizeof(count));
    if (llmm_usb_write_exact(frame, sizeof(frame)) != 0) return -1;
    return llmm_usb_write_exact(bytes, count);
}

#if LLMM_DEBUG
static void llmm_usb_routes(const uint32_t *routes)
{
    uint8_t frame[12] = {'L','L','M','R','O','U','T','3'};
    const uint32_t count = LLMM_LAYERS;
    memcpy(frame + 8, &count, sizeof(count));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        (void)llmm_usb_write_exact(routes, count * sizeof(*routes));
    }
}

static void llmm_usb_generated(const uint32_t *tokens, uint32_t count)
{
    uint8_t frame[12] = {'L','L','M','G','E','N','0','3'};
    memcpy(frame + 8, &count, sizeof(count));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        (void)llmm_usb_write_exact(tokens, count * sizeof(*tokens));
    }
}

static void llmm_usb_hidden(const float *hidden, size_t count)
{
    uint8_t frame[12] = {'L','L','M','O','U','T','0','3'};
    const uint32_t wire_count = (uint32_t)count;
    memcpy(frame + 8, &wire_count, sizeof(wire_count));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        (void)llmm_usb_write_exact(hidden, count * sizeof(*hidden));
    }
}

static void llmm_usb_layers(const float *layers)
{
    uint8_t frame[12] = {'L','L','M','L','A','Y','0','3'};
    const uint32_t count = LLMM_LAYERS;
    memcpy(frame + 8, &count, sizeof(count));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        for (uint32_t layer = 0; layer < count; ++layer) {
            if (llmm_usb_write_exact(layers + layer * 192U, 192U * sizeof(*layers)) != 0) break;
        }
    }
}
#endif

static int llmm_run_step(const llmm_handle_t *model, llmm_layer_workspace_t *workspace,
                         llmm_inference_state_t *state, uint32_t token, uint32_t score_output,
                         uint32_t sampled, uint32_t top_k, float temperature, uint32_t *random_state,
                         float *layer_trace, uint32_t *next_token, float *next_logit)
{
    if (sampled != 0U) {
        return llmm_token_step_sampled(model, token, state->position, workspace->hidden, workspace->embedding,
                                       workspace->ple_vector, workspace->query, workspace->key, workspace->attended,
                                       workspace->ple_gate, workspace->expert_gate, workspace->expert_up,
                                       state->keys, state->key_scales, state->values, state->value_scales,
                                       LLMM_KV_CONTEXT, workspace->scratch, sizeof(workspace->scratch),
#if LLMM_DEBUG
                                       state->routes, LLMM_LAYERS, score_output, workspace->candidates,
#else
                                       NULL, 0U, score_output, workspace->candidates,
#endif
                                       LLMM_TOP_K_MAX, top_k, temperature, random_state, layer_trace,
                                       next_token, next_logit);
    }
    return llmm_token_step(model, token, state->position, workspace->hidden, workspace->embedding,
                           workspace->ple_vector, workspace->query, workspace->key, workspace->attended,
                           workspace->ple_gate, workspace->expert_gate, workspace->expert_up,
                           state->keys, state->key_scales, state->values, state->value_scales,
                           LLMM_KV_CONTEXT, workspace->scratch, sizeof(workspace->scratch),
#if LLMM_DEBUG
                           state->routes, LLMM_LAYERS, score_output, layer_trace, next_token, next_logit);
#else
                           NULL, 0U, score_output, layer_trace, next_token, next_logit);
#endif
}

static int llmm_generate(const llmm_handle_t *model, llmm_layer_workspace_t *workspace,
                         llmm_inference_state_t *state, const uint32_t *prompt, uint32_t prompt_count,
                         uint32_t requested_tokens, uint32_t sampled, uint32_t top_k, float temperature,
                         uint32_t *random_state, const llmm_tokenizer_t *tokenizer, uint32_t *generated,
                         uint32_t *generated_count, float *layer_trace, float *next_logit
#if LLMM_DEBUG
                         , llmm_profile_t *profile
#endif
                         )
{
    int status = 0;
    uint32_t next_token = 0;
    state->has_pending = 0;
    for (uint32_t index = 0; index < prompt_count; ++index) {
#if LLMM_DEBUG
        llmm_debug_step_begin(0U);
#endif
        status = llmm_run_step(model, workspace, state, prompt[index], index + 1U == prompt_count ? 1U : 0U,
                               sampled, top_k, temperature, random_state, layer_trace, &next_token, next_logit);
#if LLMM_DEBUG
        llmm_debug_step_end();
#endif
        if (status != 0) return status;
        state->position += 1U;
    }

    uint32_t count = 0;
    for (; count < requested_tokens; ++count) {
        if (tokenizer != NULL &&
            (next_token == llmm_tokenizer_eos(tokenizer) || next_token == LLMM_CHAT_END_TOKEN)) break;
        generated[count] = next_token;
        state->pending_token = next_token;
        state->has_pending = 1U;
        if (tokenizer != NULL) {
#if LLMM_DEBUG
            const uint32_t stream_started = profile != NULL ? llmm_p4_cycle_count() : 0U;
            const uint32_t stream_phase = count == 0U ? 0U : 1U;
#endif
            size_t text_bytes = 0;
            status = llmm_tokenizer_decode(tokenizer, next_token, workspace->text, sizeof(workspace->text), &text_bytes);
            if (status != 0) return -100 + status;
            if (text_bytes != 0 && llmm_usb_text_chunk(workspace->text, (uint32_t)text_bytes) != 0) return -110;
#if LLMM_DEBUG
            if (profile != NULL) {
                llmm_debug_stage(stream_phase, 10U, llmm_p4_cycle_count() - stream_started);
                if (profile->ttft_cycles == 0U && text_bytes != 0U) {
                    profile->ttft_cycles = llmm_profile_us_to_cycles(
                        (uint32_t)esp_timer_get_time() - llmm_profile_started_us);
                }
            }
#endif
        }
        if (count + 1U == requested_tokens) {
            count += 1U;
            break;
        }
#if LLMM_DEBUG
        llmm_debug_step_begin(1U);
#endif
        status = llmm_run_step(model, workspace, state, next_token, 1U, sampled, top_k, temperature,
                               random_state, layer_trace, &next_token, next_logit);
#if LLMM_DEBUG
        llmm_debug_step_end();
#endif
        if (status != 0) return status;
        state->position += 1U;
        state->has_pending = 0U;
    }
    *generated_count = count;
    return 0;
}

static int llmm_map_asset(uint8_t subtype, uint32_t expected_offset, uint32_t expected_bytes,
                          const char *name, const uint8_t **asset, size_t *asset_bytes,
                          esp_partition_mmap_handle_t *map_handle)
{
    const esp_partition_t *partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, subtype, NULL);
    if (partition == NULL || partition->address != expected_offset || partition->size != expected_bytes) {
        ESP_LOGE("llmm-flash", "%s partition mismatch: expected offset=0x%08" PRIx32 " bytes=%" PRIu32,
                 name, expected_offset, expected_bytes);
        return -1;
    }
    const void *mapped = NULL;
    if (esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA,
                           &mapped, map_handle) != ESP_OK) {
        ESP_LOGE("llmm-flash", "%s XIP map failed", name);
        return -1;
    }
    *asset = mapped;
    *asset_bytes = partition->size;
    ESP_LOGI("llmm-flash", "%s XIP mapped at %p bytes=%" PRIu32, name, mapped, partition->size);
    return 0;
}

static void llmm_session_clear(llmm_runtime_t *runtime)
{
    runtime->state.position = 0;
    runtime->state.pending_token = 0;
    runtime->state.has_pending = 0;
#if LLMM_DEBUG
    memset(runtime->state.routes, 0, sizeof(runtime->state.routes));
#endif
}

static uint32_t llmm_session_tokens(const llmm_runtime_t *runtime)
{
    return (uint32_t)runtime->state.position + runtime->state.has_pending;
}

#if LLMM_HOST_UART
static sdmmc_card_t *s_sd_card;
static sd_pwr_ctrl_handle_t s_sd_pwr;
static SemaphoreHandle_t s_sd_mu;
static int s_sd_mounted;

static int llmm_sd_ensure_mounted(void)
{
    if (s_sd_mu == NULL) {
        s_sd_mu = xSemaphoreCreateMutex();
        if (s_sd_mu == NULL) return -1;
    }
    if (xSemaphoreTake(s_sd_mu, pdMS_TO_TICKS(20000)) != pdTRUE) return -1;
    int err = 0;
    if (!s_sd_mounted) {
        gpio_config_t power_cfg = {
            .pin_bit_mask = 1ULL << LLMM_SD_PWR_GPIO,
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        if (gpio_config(&power_cfg) != ESP_OK) {
            err = -2;
        } else {
            gpio_set_level(LLMM_SD_PWR_GPIO, 0);
        }
        if (err == 0 && s_sd_pwr == NULL) {
            sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = LLMM_SD_LDO_CHANNEL_ID};
            if (sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &s_sd_pwr) != ESP_OK) err = -3;
        }
        if (err == 0) {
            esp_vfs_fat_sdmmc_mount_config_t mount_config = {
                .format_if_mount_failed = false,
                .max_files = 4,
                .allocation_unit_size = 16 * 1024,
            };
            sdmmc_host_t host = SDMMC_HOST_DEFAULT();
            host.slot = SDMMC_HOST_SLOT_0;
            host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
            host.pwr_ctrl_handle = s_sd_pwr;
            sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
            slot_config.width = 4;
            slot_config.clk = GPIO_NUM_43;
            slot_config.cmd = GPIO_NUM_44;
            slot_config.d0 = GPIO_NUM_39;
            slot_config.d1 = GPIO_NUM_40;
            slot_config.d2 = GPIO_NUM_41;
            slot_config.d3 = GPIO_NUM_42;
            slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
            sdmmc_card_t *card = NULL;
            if (esp_vfs_fat_sdmmc_mount(LLMM_SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card) !=
                ESP_OK) {
                err = -4;
            } else {
                s_sd_card = card;
                s_sd_mounted = 1;
            }
        }
    }
    if (err != 0) xSemaphoreGive(s_sd_mu);
    return err;
}

static void llmm_sd_unlock(void)
{
    if (s_sd_mu != NULL) xSemaphoreGive(s_sd_mu);
}

static void llmm_sd_release_card(void)
{
    if (s_sd_mounted && s_sd_card != NULL) {
        esp_vfs_fat_sdcard_unmount(LLMM_SD_MOUNT_POINT, s_sd_card);
        s_sd_card = NULL;
        s_sd_mounted = 0;
    }
}

static FILE *s_pack_put_fp;
static uint32_t s_pack_put_total;
static uint8_t s_pack_chunk[LLMM_SD_PUT_CHUNK];

static FILE *s_pack_fp;

static void llmm_pack_close(void)
{
    if (s_pack_fp != NULL) {
        fclose(s_pack_fp);
        s_pack_fp = NULL;
    }
    llmm_pack_index_reset();
}

static void llmm_usb_pack_done(int status, const char *text)
{
    uint8_t frame[16] = {'L', 'L', 'M', 'P', 'A', 'K', 'D', '5'};
    const int32_t wire_status = status;
    uint32_t alen = 0;
    if (status == 0 && text != NULL) alen = (uint32_t)strlen(text);
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &alen, sizeof(alen));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0 && alen > 0) {
        (void)llmm_usb_write_exact(text, alen);
    }
}

static void llmm_handle_pack(void)
{
    uint8_t req[4];
    if (llmm_usb_read_exact(req, sizeof(req)) != 0) {
        llmm_usb_pack_done(-300, NULL);
        return;
    }
    uint32_t qlen = 0;
    memcpy(&qlen, req, sizeof(qlen));
    if (qlen == 0 || qlen > LLMM_PACK_Q_MAX) {
        llmm_usb_pack_done(-301, NULL);
        return;
    }
    char question[LLMM_PACK_Q_MAX + 1];
    if (llmm_usb_read_exact(question, qlen) != 0) {
        llmm_usb_pack_done(-302, NULL);
        return;
    }
    question[qlen] = 0;

    int mount = llmm_sd_ensure_mounted();
    if (mount != 0) {
        llmm_usb_pack_done(-310 + mount, NULL);
        return;
    }
    FILE *fp = s_pack_fp;
    if (fp == NULL) {
        fp = fopen(LLMM_SD_PACK_PATH, "rb");
        s_pack_fp = fp;
    }
    if (fp == NULL) {
        llmm_sd_unlock();
        llmm_usb_pack_done(-304, NULL);
        return;
    }
    char answer[LLMM_PACK_A_MAX];
    int status = llmm_pack_answer(fp, question, answer, sizeof(answer));
    llmm_sd_unlock();
    if (status != 0) {
        llmm_usb_pack_done(status, NULL);
        return;
    }
    llmm_usb_pack_done(0, answer);
}

static void llmm_handle_pack_put(void)
{
    uint8_t req[12];
    if (llmm_usb_read_exact(req, sizeof(req)) != 0) {
        llmm_usb_control_status("LLMPUTD5", -320);
        return;
    }
    uint32_t total = 0, offset = 0, nbytes = 0;
    memcpy(&total, req, 4);
    memcpy(&offset, req + 4, 4);
    memcpy(&nbytes, req + 8, 4);
    if (nbytes == 0 || nbytes > LLMM_SD_PUT_CHUNK || total == 0 || offset + nbytes > total) {
        if (nbytes > 0 && nbytes <= LLMM_SD_PUT_CHUNK) {
            (void)llmm_usb_read_exact(s_pack_chunk, nbytes);
        }
        llmm_usb_control_status("LLMPUTD5", -321);
        return;
    }
    if (llmm_usb_read_exact(s_pack_chunk, nbytes) != 0) {
        llmm_usb_control_status("LLMPUTD5", -322);
        return;
    }
    int mount = llmm_sd_ensure_mounted();
    if (mount != 0) {
        llmm_usb_control_status("LLMPUTD5", -330 + mount);
        return;
    }
    int status = 0;
    if (offset == 0U) {
        llmm_pack_close();
        if (s_pack_put_fp != NULL) {
            fclose(s_pack_put_fp);
            s_pack_put_fp = NULL;
        }
        s_pack_put_fp = fopen(LLMM_SD_PACK_PATH, "wb");
        s_pack_put_total = total;
    }
    if (s_pack_put_fp == NULL) {
        status = -324;
    } else if (s_pack_put_total != total) {
        status = -327;
    } else if ((uint32_t)ftell(s_pack_put_fp) != offset &&
               fseek(s_pack_put_fp, (long)offset, SEEK_SET) != 0) {
        status = -325;
    } else if (fwrite(s_pack_chunk, 1, nbytes, s_pack_put_fp) != nbytes) {
        status = -326;
    }
    if (s_pack_put_fp != NULL && (status != 0 || offset + nbytes >= total)) {
        fclose(s_pack_put_fp);
        s_pack_put_fp = NULL;
    }
    llmm_sd_unlock();
    llmm_usb_control_status("LLMPUTD5", status);
}

static int llmm_try_load_sd(llmm_runtime_t *runtime)
{
    if (runtime->psram_payload == NULL || runtime->model.psram_bytes == 0) {
        return -1;
    }

    int mount = llmm_sd_ensure_mounted();
    if (mount != 0) return mount;

    FILE *file = fopen(LLMM_SD_PAYLOAD_PATH, "rb");
    if (file == NULL) {
        llmm_sd_unlock();
        return -5;
    }

    uint8_t header[16];
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fclose(file);
        llmm_sd_unlock();
        return -6;
    }
    if (memcmp(header, "P4SD", 4) != 0) {
        fclose(file);
        llmm_sd_unlock();
        return -7;
    }

    uint32_t version = 0;
    uint32_t payload_bytes = 0;
    uint32_t expected_crc = 0;
    memcpy(&version, header + 4, sizeof(version));
    memcpy(&payload_bytes, header + 8, sizeof(payload_bytes));
    memcpy(&expected_crc, header + 12, sizeof(expected_crc));
    if (version != 1U || payload_bytes != (uint32_t)runtime->model.psram_bytes) {
        fclose(file);
        llmm_sd_unlock();
        return -8;
    }

    size_t got = fread(runtime->psram_payload, 1, payload_bytes, file);
    fclose(file);
    if (got != payload_bytes) {
        llmm_sd_unlock();
        return -9;
    }

    uint32_t actual_crc = esp_rom_crc32_le(0, runtime->psram_payload, payload_bytes);
    llmm_sd_unlock();
    if (actual_crc != expected_crc) {
        return -10;
    }

    runtime->loaded = 1U;
    runtime->payload_id = actual_crc;
    llmm_session_clear(runtime);
    ESP_LOGI("llmm-sd", "loaded %s bytes=%" PRIu32 " crc=0x%08" PRIx32,
             LLMM_SD_PAYLOAD_PATH, payload_bytes, actual_crc);
    return 0;
}
#endif

static int llmm_runtime_init(llmm_runtime_t *runtime)
{
    memset(runtime, 0, sizeof(*runtime));
#if LLMM_HOST_UART
    const uart_config_t uart_config = {
        .baud_rate = LLMM_HOST_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(LLMM_HOST_UART_PORT, 16384, 4096, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(LLMM_HOST_UART_PORT, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(LLMM_HOST_UART_PORT, LLMM_HOST_UART_TX_GPIO, LLMM_HOST_UART_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI("llmm", "host protocol on UART0 @ %d baud (GPIO%d TX / GPIO%d RX)",
             LLMM_HOST_UART_BAUD, LLMM_HOST_UART_TX_GPIO, LLMM_HOST_UART_RX_GPIO);
#else
    const usb_serial_jtag_driver_config_t config = {
        .rx_buffer_size = 4096,
        .tx_buffer_size = 4096,
    };
    if (!usb_serial_jtag_is_driver_installed()) {
        ESP_ERROR_CHECK(usb_serial_jtag_driver_install((usb_serial_jtag_driver_config_t *)&config));
    }
#endif
    if (llmm_map_asset(0x41, LLMM_MANIFEST_FLASH_OFFSET, LLMM_MANIFEST_FLASH_BYTES, "manifest",
                       &runtime->storage.manifest_xip, &runtime->storage.manifest_xip_bytes,
                       &runtime->manifest_map_handle) != 0) return -190;
    if (llmm_map_asset(0x42, LLMM_PLE_FLASH_OFFSET, LLMM_PLE_FLASH_BYTES, "PLE",
                       &runtime->storage.ple_xip, &runtime->storage.ple_xip_bytes,
                       &runtime->ple_map_handle) != 0) return -191;
    if (llmm_map_asset(0x40, LLMM_TOKENIZER_FLASH_OFFSET, LLMM_TOKENIZER_FLASH_BYTES, "tokenizer",
                       &runtime->storage.tokenizer_xip, &runtime->storage.tokenizer_xip_bytes,
                       &runtime->tokenizer_map_handle) != 0) return -192;

    int status = llmm_init_manifest(&runtime->model, runtime->storage.manifest_xip,
                                    runtime->storage.manifest_xip_bytes);
    if (status != 0) return -199 + status;
    if (runtime->model.flash_bytes != LLMM_PLE_FLASH_BYTES || runtime->model.psram_bytes > UINT32_MAX) return -200;
    status = llmm_tokenizer_init(&runtime->tokenizer, runtime->storage.tokenizer_xip,
                                 runtime->storage.tokenizer_xip_bytes);
    if (status != 0 || llmm_tokenizer_max_token_bytes(&runtime->tokenizer) > LLMM_TEXT_MAX_BYTES) {
        return status != 0 ? -230 + status : -240;
    }

    runtime->psram_payload = heap_caps_malloc(
        runtime->model.psram_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_SIMD);
    if (runtime->psram_payload == NULL) return -201;
    runtime->kv_storage = heap_caps_malloc(
        LLMM_KV_STORAGE_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_SIMD);
    if (runtime->kv_storage == NULL) return -202;
    runtime->workspace = heap_caps_malloc(sizeof(*runtime->workspace), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (runtime->workspace == NULL) return -203;
    runtime->raw_prompt = heap_caps_malloc(LLMM_RAW_PROMPT_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (runtime->raw_prompt == NULL) return -203;
    if (llmm_accel_init() != 0) return -204;

    runtime->storage.psram_payload = runtime->psram_payload;
    runtime->storage.psram_payload_bytes = runtime->model.psram_bytes;
    llmm_set_reader(&runtime->model, &runtime->storage, llmm_local_reader);
    runtime->state.keys = (int8_t *)runtime->kv_storage;
    runtime->state.values = (int8_t *)runtime->kv_storage + LLMM_LAYERS * LLMM_KV_LAYER_VECTOR_BYTES;
    runtime->state.key_scales =
        (uint16_t *)(runtime->kv_storage + 2U * LLMM_LAYERS * LLMM_KV_LAYER_VECTOR_BYTES);
    runtime->state.value_scales = runtime->state.key_scales + LLMM_LAYERS * LLMM_KV_CONTEXT * 2U;
    llmm_session_clear(runtime);
    ESP_LOGI("llmm", "resident runtime ready: psram-payload=%u kv=%u psram-free=%u loaded=%u",
             (unsigned)runtime->model.psram_bytes, (unsigned)LLMM_KV_STORAGE_BYTES,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)runtime->loaded);
    return 0;
}

static void llmm_handle_load(llmm_runtime_t *runtime)
{
    uint8_t request[8];
    int status = runtime->init_status;
    if (llmm_usb_read_exact(request, sizeof(request)) != 0) {
        status = -204;
    }
    uint32_t payload_bytes = 0;
    uint32_t expected_crc = 0;
    if (status == 0) {
        memcpy(&payload_bytes, request, sizeof(payload_bytes));
        memcpy(&expected_crc, request + 4, sizeof(expected_crc));
        if (payload_bytes != runtime->model.psram_bytes) status = -205;
    }
    if (status == 0) {
        runtime->loaded = 0;
        runtime->payload_id = 0;
        llmm_session_clear(runtime);
        uint32_t actual_crc = 0;
        if (llmm_usb_read_payload(runtime->psram_payload, payload_bytes, &actual_crc) != 0) {
            status = -206;
        } else if (actual_crc != expected_crc) {
            status = -207;
            ESP_LOGE("llmm-cache", "payload CRC mismatch: expected=0x%08" PRIx32 " actual=0x%08" PRIx32,
                     expected_crc, actual_crc);
        } else {
            runtime->loaded = 1U;
            runtime->payload_id = actual_crc;
            ESP_LOGI("llmm-cache", "payload cache filled: id=0x%08" PRIx32 " bytes=%" PRIu32,
                     actual_crc, payload_bytes);
        }
    }
    llmm_usb_loaded(status, status == 0 ? payload_bytes : 0U,
                    status == 0 ? runtime->payload_id : 0U);
}

#if LLMM_DEBUG
static void llmm_handle_numeric(llmm_runtime_t *runtime)
{
    uint8_t request[12];
    int status = runtime->loaded != 0U ? 0 : -212;
    if (llmm_usb_read_exact(request, sizeof(request)) != 0) {
        llmm_usb_done(-207, 0, 0, 0);
        return;
    }
    uint32_t prompt_count;
    uint32_t requested_tokens;
    uint32_t flags;
    memcpy(&prompt_count, request, sizeof(prompt_count));
    memcpy(&requested_tokens, request + 4, sizeof(requested_tokens));
    memcpy(&flags, request + 8, sizeof(flags));
    if (status == 0 && (prompt_count == 0 || requested_tokens == 0 || prompt_count > LLMM_KV_CONTEXT ||
                        requested_tokens > LLMM_KV_CONTEXT - prompt_count)) status = -208;
    float *layer_trace = NULL;
    if (status == 0 && (flags & LLMM_TRACE_LAYERS) != 0U) {
        layer_trace = heap_caps_malloc(LLMM_LAYERS * 192U * sizeof(*layer_trace),
                                       MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (layer_trace == NULL) status = -209;
    }
    if (status == 0 && llmm_usb_read_exact(runtime->workspace->tokens,
                                            prompt_count * sizeof(*runtime->workspace->tokens)) != 0) status = -211;
    if (status != 0) {
        llmm_usb_done(status, 0, 0, 0);
        heap_caps_free(layer_trace);
        return;
    }

    llmm_session_clear(runtime);
    uint32_t generated_count = 0;
    uint32_t random_state = 0;
    float next_logit = 0.0f;
    const int64_t started_us = esp_timer_get_time();
    status = llmm_generate(&runtime->model, runtime->workspace, &runtime->state,
                           runtime->workspace->tokens, prompt_count, requested_tokens,
                           0U, 1U, 0.0f, &random_state, NULL, runtime->workspace->tokens,
                           &generated_count, layer_trace, &next_logit, NULL);
    const uint32_t elapsed_us = (uint32_t)(esp_timer_get_time() - started_us);
    const uint32_t checksum = status == 0 ? llmm_f32_checksum(runtime->workspace->query, 192U) : 0U;
    llmm_usb_done(status, status == 0 ? generated_count : 0U, checksum, elapsed_us);
    if (status == 0) {
        llmm_usb_routes(runtime->state.routes);
        llmm_usb_generated(runtime->workspace->tokens, generated_count);
        if (layer_trace != NULL) llmm_usb_layers(layer_trace);
        if ((flags & LLMM_TRACE_HIDDEN) != 0U) llmm_usb_hidden(runtime->workspace->query, 192U);
    }
    heap_caps_free(layer_trace);
    llmm_session_clear(runtime);
}
#endif

static void llmm_handle_text(llmm_runtime_t *runtime
#if LLMM_DEBUG
                             , uint32_t profile_enabled
#endif
                             )
{
    uint8_t request[20];
    if (llmm_usb_read_exact(request, sizeof(request)) != 0) {
        llmm_usb_text_done(-220, 0, 0, 0);
        return;
    }
    uint32_t prompt_bytes;
    uint32_t requested_tokens;
    float temperature;
    uint32_t top_k;
    uint32_t random_state;
    memcpy(&prompt_bytes, request, sizeof(prompt_bytes));
    memcpy(&requested_tokens, request + 4, sizeof(requested_tokens));
    memcpy(&temperature, request + 8, sizeof(temperature));
    memcpy(&top_k, request + 12, sizeof(top_k));
    memcpy(&random_state, request + 16, sizeof(random_state));

    int status = runtime->loaded != 0U ? 0 : -221;
    if (status == 0 && (prompt_bytes == 0 || prompt_bytes > LLMM_RAW_PROMPT_MAX_BYTES ||
                        runtime->raw_prompt == NULL ||
                        requested_tokens == 0 || top_k == 0 || top_k > LLMM_TOP_K_MAX ||
                        !isfinite(temperature) || temperature < 0.0f)) status = -240;
    if (status != 0) {
        llmm_usb_text_done(status, 0, 0, 0);
        return;
    }
    if (llmm_usb_read_exact(runtime->raw_prompt, prompt_bytes) != 0) {
        llmm_usb_text_done(-241, 0, 0, 0);
        return;
    }

    size_t fitted_bytes = 0;
    llmm_compress_stats_t compress_stats;
    const int compress_status = llmm_compress_prompt_to_fit(
        runtime->raw_prompt, prompt_bytes, runtime->workspace->text, sizeof(runtime->workspace->text),
        &fitted_bytes, &compress_stats);
    if (compress_status != 0 || fitted_bytes == 0 || fitted_bytes > sizeof(runtime->workspace->text)) {
        llmm_usb_text_done(-242, 0, 0, 0);
        return;
    }
    if (compress_stats.compressed != 0U) {
        ESP_LOGI("llmm-compress",
                 "on-device ratio~%.2f:1 raw=%" PRIu32 " fitted=%" PRIu32 " kept=%" PRIu32
                 " dropped=%" PRIu32,
                 compress_stats.packet_bytes > 0
                     ? (double)compress_stats.source_bytes / (double)compress_stats.packet_bytes
                     : 0.0,
                 compress_stats.source_bytes, compress_stats.packet_bytes, compress_stats.kept_sentences,
                 compress_stats.dropped_sentences);
    }

    uint32_t prefix_count = runtime->state.has_pending;
    if (prefix_count != 0U) runtime->workspace->tokens[0] = runtime->state.pending_token;
    size_t prompt_count = 0;
    const int tokenizer_status = llmm_tokenizer_encode(
        &runtime->tokenizer, runtime->workspace->text, fitted_bytes,
        runtime->workspace->tokens + prefix_count, LLMM_KV_CONTEXT - prefix_count, &prompt_count,
        runtime->workspace->bpe_pieces,
        sizeof(runtime->workspace->bpe_pieces) / sizeof(*runtime->workspace->bpe_pieces));
    if (tokenizer_status != 0 || prompt_count == 0) status = -250 + tokenizer_status;

    uint32_t text_flags = 0;
    if (status == 0 && runtime->state.position + prefix_count + prompt_count + requested_tokens > LLMM_KV_CONTEXT) {
        if (llmm_session_tokens(runtime) != 0U) {
            ESP_LOGI("llmm-cache", "KV cache evicted at %" PRIu32 " tokens", llmm_session_tokens(runtime));
            llmm_session_clear(runtime);
            text_flags |= LLMM_TEXT_FLAG_SESSION_EVICTED;
            if (prefix_count != 0U) {
                memmove(runtime->workspace->tokens, runtime->workspace->tokens + prefix_count,
                        prompt_count * sizeof(*runtime->workspace->tokens));
                prefix_count = 0;
            }
        }
        if (prompt_count + requested_tokens > LLMM_KV_CONTEXT) status = -260;
    }
    if (status != 0) {
        llmm_usb_text_done(status, 0, 0, 0);
        return;
    }
    if (llmm_usb_text_tokens(runtime->workspace->tokens + prefix_count,
                             (uint32_t)prompt_count, text_flags) != 0) {
        llmm_session_clear(runtime);
        llmm_usb_text_done(-261, 0, 0, 0);
        return;
    }

    uint32_t generated_count = 0;
    float next_logit = 0.0f;
#if LLMM_DEBUG
    llmm_profile_t *profile = profile_enabled != 0U ? &llmm_debug_profile : NULL;
    if (profile != NULL) {
        llmm_debug_profile_begin(profile);
        profile->kv_reused_tokens = runtime->state.position;
        profile->kv_evictions = (text_flags & LLMM_TEXT_FLAG_SESSION_EVICTED) != 0U;
    }
#endif
    const int64_t started_us = esp_timer_get_time();
    status = llmm_generate(&runtime->model, runtime->workspace, &runtime->state,
                           runtime->workspace->tokens, prefix_count + (uint32_t)prompt_count,
                           requested_tokens, 1U, top_k, temperature, &random_state,
                           &runtime->tokenizer, runtime->workspace->tokens, &generated_count,
                           NULL, &next_logit
#if LLMM_DEBUG
                           , profile
#endif
                           );
    const uint32_t elapsed_us = (uint32_t)(esp_timer_get_time() - started_us);
#if LLMM_DEBUG
    if (profile != NULL) {
        profile->kv_appended_tokens = runtime->state.position - profile->kv_reused_tokens;
        profile->kv_occupancy_tokens = llmm_session_tokens(runtime);
        llmm_debug_profile_end(profile, llmm_accel_worker_task);
    }
    const uint32_t checksum = status == 0 ? llmm_f32_checksum(runtime->workspace->query, 192U) : 0U;
#else
    const uint32_t checksum = 0U;
#endif
    if (status != 0) llmm_session_clear(runtime);
    llmm_usb_text_done(status, status == 0 ? generated_count : 0U, checksum, elapsed_us);
#if LLMM_DEBUG
    if (profile != NULL) llmm_usb_profile(profile);
#endif
    ESP_LOGI("llmm", "text status=%d prompt=%u generated=%" PRIu32 " session=%" PRIu32
             " elapsed=%" PRIu32 " us psram-free=%u", status, (unsigned)prompt_count,
             generated_count, llmm_session_tokens(runtime), elapsed_us,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static int llmm_run_hidden_layers(llmm_runtime_t *runtime, uint32_t token, size_t position,
                                  uint32_t layer_begin, uint32_t layer_end, uint32_t prepare_ple,
                                  uint32_t score_output, uint32_t *next_token, float *next_logit)
{
    llmm_layer_workspace_t *workspace = runtime->workspace;
    llmm_inference_state_t *state = &runtime->state;
    return llmm_hidden_layers_step(
        &runtime->model, token, position, layer_begin, layer_end, prepare_ple,
        workspace->hidden, workspace->embedding, workspace->ple_vector, workspace->query,
        workspace->key, workspace->attended, workspace->ple_gate, workspace->expert_gate,
        workspace->expert_up, state->keys, state->key_scales, state->values, state->value_scales,
        LLMM_KV_CONTEXT, workspace->scratch, sizeof(workspace->scratch),
#if LLMM_DEBUG
        state->routes, LLMM_LAYERS,
#else
        NULL, 0U,
#endif
        score_output, NULL, next_token, next_logit);
}

static void llmm_usb_hop_done(int status, uint32_t next_token, float next_logit,
                              uint32_t elapsed_us, uint32_t position, const float *hidden)
{
    uint8_t frame[28] = {'L','L','M','H','O','K','0','5'};
    const int32_t wire_status = status;
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &next_token, sizeof(next_token));
    memcpy(frame + 16, &next_logit, sizeof(next_logit));
    memcpy(frame + 20, &elapsed_us, sizeof(elapsed_us));
    memcpy(frame + 24, &position, sizeof(position));
    if (llmm_usb_write_exact(frame, sizeof(frame)) == 0) {
        (void)llmm_usb_write_exact(hidden, LLMM_HOP_HIDDEN_BYTES);
    }
}

static void llmm_handle_hop(llmm_runtime_t *runtime)
{
    uint8_t request[LLMM_HOP_HEADER_BYTES];
    if (llmm_usb_read_exact(request, sizeof(request)) != 0) {
        float zeros[LLMM_WIDTH] = {0};
        llmm_usb_hop_done(-270, 0, 0.0f, 0, 0, zeros);
        return;
    }

    uint32_t token;
    uint32_t position;
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t score_output;
    uint32_t flags;
    memcpy(&token, request, sizeof(token));
    memcpy(&position, request + 4, sizeof(position));
    memcpy(&layer_begin, request + 8, sizeof(layer_begin));
    memcpy(&layer_end, request + 12, sizeof(layer_end));
    memcpy(&score_output, request + 16, sizeof(score_output));
    memcpy(&flags, request + 20, sizeof(flags));

    int status = runtime->loaded != 0U ? 0 : -271;
    if (status == 0 && llmm_usb_read_exact(runtime->workspace->hidden, LLMM_HOP_HIDDEN_BYTES) != 0) {
        status = -272;
    }
    if (status == 0 && (flags & LLMM_HOP_FLAG_SESSION) != 0U) {
        position = (uint32_t)runtime->state.position;
    }
    if (status == 0 && (token >= LLMM_VOCAB || position >= LLMM_KV_CONTEXT ||
                        layer_begin >= layer_end || layer_end > LLMM_LAYERS ||
                        score_output > 1U)) {
        status = -273;
    }

    uint32_t next_token = 0;
    float next_logit = 0.0f;
    const int64_t started_us = esp_timer_get_time();
    if (status == 0) {
        const uint32_t prepare_ple = layer_begin == 0U ? 2U : 1U;
        status = llmm_run_hidden_layers(runtime, token, position, layer_begin, layer_end,
                                        prepare_ple, score_output, &next_token, &next_logit);
        if (status == 0 && (flags & LLMM_HOP_FLAG_SESSION) != 0U) {
            runtime->state.position += 1U;
        }
    }
    const uint32_t elapsed_us = (uint32_t)(esp_timer_get_time() - started_us);
    llmm_usb_hop_done(status, next_token, next_logit, elapsed_us, position, runtime->workspace->hidden);
}

static void llmm_handle_split_loopback(llmm_runtime_t *runtime)
{
    uint8_t request[8];
    uint8_t frame[32] = {'L','L','M','H','S','L','D','5'};
    if (llmm_usb_read_exact(request, sizeof(request)) != 0) {
        const int32_t wire_status = -280;
        memcpy(frame + 8, &wire_status, sizeof(wire_status));
        (void)llmm_usb_write_exact(frame, sizeof(frame));
        return;
    }

    uint32_t token;
    uint32_t split_layer;
    memcpy(&token, request, sizeof(token));
    memcpy(&split_layer, request + 4, sizeof(split_layer));

    int status = runtime->loaded != 0U ? 0 : -281;
    if (status == 0 && (token >= LLMM_VOCAB || split_layer == 0U || split_layer >= LLMM_LAYERS)) {
        status = -282;
    }

    uint32_t split_token = 0;
    uint32_t full_token = 0;
    uint32_t tokens_match = 0;
    float max_abs_diff = 0.0f;
    float split_hidden[LLMM_WIDTH];
    float next_logit = 0.0f;
    const int64_t started_us = esp_timer_get_time();
    if (status == 0) {
        llmm_session_clear(runtime);
        status = llmm_run_hidden_layers(runtime, token, 0U, 0U, split_layer, 2U, 0U,
                                        &split_token, &next_logit);
        if (status == 0) {
            status = llmm_run_hidden_layers(runtime, token, 0U, split_layer, LLMM_LAYERS, 0U, 1U,
                                            &split_token, &next_logit);
        }
        if (status == 0) {
            memcpy(split_hidden, runtime->workspace->hidden, sizeof(split_hidden));
            llmm_session_clear(runtime);
            status = llmm_run_hidden_layers(runtime, token, 0U, 0U, LLMM_LAYERS, 2U, 1U,
                                            &full_token, &next_logit);
        }
        if (status == 0) {
            for (uint32_t index = 0; index < LLMM_WIDTH; ++index) {
                const float diff = fabsf(split_hidden[index] - runtime->workspace->hidden[index]);
                if (diff > max_abs_diff) max_abs_diff = diff;
            }
            tokens_match = split_token == full_token ? 1U : 0U;
        }
        llmm_session_clear(runtime);
    }
    const uint32_t elapsed_us = (uint32_t)(esp_timer_get_time() - started_us);
    const int32_t wire_status = status;
    memcpy(frame + 8, &wire_status, sizeof(wire_status));
    memcpy(frame + 12, &tokens_match, sizeof(tokens_match));
    memcpy(frame + 16, &split_token, sizeof(split_token));
    memcpy(frame + 20, &full_token, sizeof(full_token));
    memcpy(frame + 24, &max_abs_diff, sizeof(max_abs_diff));
    memcpy(frame + 28, &elapsed_us, sizeof(elapsed_us));
    (void)llmm_usb_write_exact(frame, sizeof(frame));
}

#if LLMM_HOST_UART
static void llmm_eth_start_task(void *arg)
{
    (void)arg;
    if (llmm_eth_start() != 0) {
        ESP_LOGE("llmm-eth", "ethernet start failed (UART host still available)");
    }
    vTaskDelete(NULL);
}

static void llmm_sd_load_task(void *arg)
{
    (void)llmm_try_load_sd(arg);
    vTaskDelete(NULL);
}
#endif

static void run_llmm_usb_inference(void)
{
    llmm_runtime_t runtime;
    runtime.init_status = llmm_runtime_init(&runtime);
    if (runtime.init_status != 0) ESP_LOGE("llmm", "resident runtime init failed: %d", runtime.init_status);
#if LLMM_HOST_UART
    if (xTaskCreate(llmm_eth_start_task, "llmm-eth", 8192, NULL, 4, NULL) != pdPASS) {
        ESP_LOGE("llmm-eth", "ethernet task create failed (UART host still available)");
    }
    if (runtime.init_status == 0 &&
        xTaskCreate(llmm_sd_load_task, "llmm-sd", 8192, &runtime, 5, NULL) != pdPASS) {
        ESP_LOGE("llmm-sd", "SD load task create failed");
    }
#endif

    for (;;) {
        uint8_t command[8];
        if (llmm_usb_read_exact(command, sizeof(command)) != 0) {
            ESP_LOGE("llmm-usb", "command read failed");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (memcmp(command, "LLMHOST5", 8) == 0) {
            llmm_usb_ready(runtime.init_status,
                           runtime.init_status == 0 ? (uint32_t)runtime.model.psram_bytes : 0U,
                           runtime.loaded, runtime.payload_id, llmm_session_tokens(&runtime));
        } else if (memcmp(command, "LLMPSR05", 8) == 0) {
            llmm_handle_load(&runtime);
        } else if (memcmp(command, "LLMCLR05", 8) == 0) {
            llmm_session_clear(&runtime);
            llmm_usb_control_status("LLMCLRD5", runtime.init_status);
        } else if (memcmp(command, "LLMBYE05", 8) == 0) {
            llmm_usb_control_status("LLMBYED5", 0);
#if LLMM_DEBUG
        } else if (memcmp(command, "LLMINF03", 8) == 0) {
            llmm_handle_numeric(&runtime);
#endif
        } else if (memcmp(command, "LLMTXT05", 8) == 0) {
#if LLMM_DEBUG
            llmm_handle_text(&runtime, 0U);
        } else if (memcmp(command, "LLMPRQ05", 8) == 0) {
            llmm_handle_text(&runtime, 1U);
#else
            llmm_handle_text(&runtime);
#endif
        } else if (memcmp(command, "LLMHOP05", 8) == 0) {
            llmm_handle_hop(&runtime);
        } else if (memcmp(command, "LLMHSL05", 8) == 0) {
            llmm_handle_split_loopback(&runtime);
#if LLMM_HOST_UART
        } else if (memcmp(command, "LLMPAK05", 8) == 0) {
            llmm_handle_pack();
        } else if (memcmp(command, "LLMPUT05", 8) == 0) {
            llmm_handle_pack_put();
#endif
        } else {
            ESP_LOGE("llmm-usb", "unknown command");
            llmm_usb_control_status("LLMERR05", -207);
        }
    }
}

#if LLMM_DEBUG
static volatile uint8_t memory_source[MEMORY_BYTES];
static volatile uint8_t memory_destination[MEMORY_BYTES];
static TaskHandle_t benchmark_main_task;
static volatile int16_t xespv_input[16] __attribute__((aligned(16))) = {
    1, 2, 3, 4, 5, 6, 7, 8,
    1, 2, 3, 4, 5, 6, 7, 8,
};
static volatile int16_t xespv_weights[8] __attribute__((aligned(16))) = {2, 4, 6, 8, 10, 12, 14, 16};
static volatile int8_t xespv_i8_input[32] __attribute__((aligned(16))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static volatile int8_t xespv_i8_weights[16] __attribute__((aligned(16))) = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};
static int8_t q4_activations[Q4_ELEMENTS];
static uint8_t q4_weights[Q4_ELEMENTS / 2U];
static int8_t matrix_a[MATRIX_SIZE * MATRIX_SIZE];
static int8_t matrix_b[MATRIX_SIZE * MATRIX_SIZE];
static int32_t matrix_c[MATRIX_SIZE * MATRIX_SIZE];
static float vector_data[VECTOR_SIZE];
static int8_t gemv_input[GEMV_K + 16U] __attribute__((aligned(16)));
static uint8_t gemv_q4_weights[GEMV_N * GEMV_K / 2U];
static int8_t gemv_unpacked[GEMV_K] __attribute__((aligned(16)));
static int8_t gemv_unpacked_core1[GEMV_K] __attribute__((aligned(16)));
static int32_t gemv_output[GEMV_N];
static float residual[VECTOR_SIZE];
static TaskHandle_t dma_wait_task;

__attribute__((target("arch=+xespv"))) static int32_t xespv_dot8(uint32_t iterations)
{
    int32_t result = 0;

    while (iterations-- > 0) {
        const int16_t *input = (const int16_t *)xespv_input;
        const int16_t *weights = (const int16_t *)xespv_weights;
        uint32_t shift = 0;

        __asm__ volatile(
            "esp.vld.128.ip q0, %[input], 16\n"
            "esp.vld.128.ip q1, %[weights], 16\n"
            "esp.zero.xacc\n"
            "esp.vmulas.s16.xacc.ld.ip q0, %[input], 16, q0, q1\n"
            "esp.srs.s.xacc %[result], %[shift]\n"
            : [result] "=r"(result), [input] "+r"(input), [weights] "+r"(weights), [shift] "+r"(shift)
            :
            : "memory");
    }

    return result;
}

__attribute__((target("arch=+xespv"))) static int32_t xespv_dot16_i8(uint32_t iterations)
{
    int32_t result = 0;

    while (iterations-- > 0) {
        const int8_t *input = (const int8_t *)xespv_i8_input;
        const int8_t *weights = (const int8_t *)xespv_i8_weights;
        uint32_t shift = 0;

        __asm__ volatile(
            "esp.vld.128.ip q0, %[input], 16\n"
            "esp.vld.128.ip q1, %[weights], 16\n"
            "esp.zero.xacc\n"
            "esp.vmulas.s8.xacc.ld.ip q0, %[input], 16, q0, q1\n"
            "esp.srs.s.xacc %[result], %[shift]\n"
            : [result] "=r"(result), [input] "+r"(input), [weights] "+r"(weights), [shift] "+r"(shift)
            :
            : "memory");
    }

    return result;
}

__attribute__((target("arch=+xespv"))) static int32_t xespv_dot16_i8_ptr(const int8_t *input, const int8_t *weights)
{
    int32_t result;
    uint32_t shift = 0;

    __asm__ volatile(
        "esp.vld.128.ip q0, %[input], 16\n"
        "esp.vld.128.ip q1, %[weights], 16\n"
        "esp.zero.xacc\n"
        "esp.vmulas.s8.xacc.ld.ip q0, %[input], 16, q0, q1\n"
        "esp.srs.s.xacc %[result], %[shift]\n"
        : [result] "=r"(result), [input] "+r"(input), [weights] "+r"(weights), [shift] "+r"(shift)
        :
        : "memory");
    return result;
}
#endif

typedef enum {
    LLMM_ACCEL_IDLE = 0,
    LLMM_ACCEL_BASE3 = 1,
    LLMM_ACCEL_HEAD = 2,
    LLMM_ACCEL_ATTENTION = 3,
} llmm_accel_kind_t;

typedef struct {
    volatile llmm_accel_kind_t kind;
    const uint8_t *weights;
    const uint8_t *scales;
    float *output;
    size_t base_row;
    size_t first_row;
    size_t last_row;
    size_t columns;
    float output_scale;
    size_t top_k;
    llmm_candidate_t candidates[LLMM_TOP_K_MAX];
    size_t candidate_count;
    const int8_t *attention_keys;
    const uint16_t *attention_key_scales;
    const int8_t *attention_values;
    const uint16_t *attention_value_scales;
    size_t attention_tokens;
    int status;
#if LLMM_DEBUG
    uint32_t dispatch_started_us;
#endif
} llmm_accel_job_t;

static SPM_DRAM_ATTR uint64_t llmm_base3_lut8[243] __attribute__((aligned(16)));
static int8_t llmm_accel_input[LLMM_MAX_COLUMNS] __attribute__((aligned(16)));
static int8_t llmm_base3_tiles[2][LLMM_MAX_COLUMNS + 16U] __attribute__((aligned(16)));
static int8_t llmm_attention_query[LLMM_WIDTH] __attribute__((aligned(16)));
static float llmm_attention_query_scales[LLMM_HEADS];
static float llmm_attention_accumulators[LLMM_HEADS][LLMM_HEAD_DIM] __attribute__((aligned(64)));
static float llmm_attention_maximums[LLMM_KV_HEADS][16] __attribute__((aligned(64)));
static float llmm_attention_normalizers[LLMM_KV_HEADS][16] __attribute__((aligned(64)));
static TaskHandle_t llmm_accel_owner_task;
static llmm_accel_job_t llmm_accel_job;

static void llmm_attention_kv_range(
    const int8_t *keys, const uint16_t *key_scales,
    const int8_t *values, const uint16_t *value_scales,
    size_t tokens, size_t first_kv_head, size_t last_kv_head);

#if LLMM_DEBUG
static inline size_t llmm_accel_profile_index(llmm_accel_kind_t kind)
{
    if (kind == LLMM_ACCEL_BASE3) return 0U;
    if (kind == LLMM_ACCEL_ATTENTION) return 1U;
    return 2U;
}

static inline void llmm_debug_mark_dispatch(void)
{
    llmm_accel_job.dispatch_started_us =
        llmm_active_profile != NULL ? (uint32_t)esp_timer_get_time() : 0U;
}

static inline void llmm_debug_owner_wait(llmm_accel_kind_t kind, uint32_t started)
{
    llmm_profile_t *profile = llmm_active_profile;
    if (profile == NULL) return;
    const size_t index = llmm_accel_profile_index(kind);
    llmm_profile_stat(&profile->owner_wait_cycles[index], &profile->owner_wait_max_cycles[index],
                      &profile->owner_wait_calls[index], llmm_p4_cycle_count() - started);
}

static inline void llmm_debug_traffic(uint32_t kind, size_t bytes, uint32_t elapsed)
{
    llmm_profile_t *profile = llmm_active_profile;
    if (profile == NULL || kind >= LLMM_PROFILE_TRAFFIC_KINDS) return;
    profile->traffic_bytes[kind] += bytes;
    llmm_profile_stat(&profile->traffic_cycles[kind], &profile->traffic_max_cycles[kind],
                      &profile->traffic_calls[kind], elapsed);
}
#endif

static const float llmm_rope_step_cos[LLMM_HEAD_DIM / 2U] = {
    0.540302277f, 0.846009135f, 0.950415254f, 0.984230220f,
    0.995004177f, 0.998419285f, 0.999500036f, 0.999841869f,
    0.999949992f, 0.999984205f, 0.999994993f, 0.999998391f,
    0.999999523f, 0.999999821f, 0.999999940f, 1.000000000f,
};
static const float llmm_rope_step_sin[LLMM_HEAD_DIM / 2U] = {
    0.841470957f, 0.533168435f, 0.310983598f, 0.176892191f,
    0.099833414f, 0.056204498f, 0.031617507f, 0.017781857f,
    0.009999833f, 0.005623383f, 0.003162272f, 0.001778279f,
    0.001000000f, 0.000562341f, 0.000316228f, 0.000177828f,
};
static float llmm_rope_cos[LLMM_HEAD_DIM / 2U];
static float llmm_rope_sin[LLMM_HEAD_DIM / 2U];
static uint32_t llmm_rope_position = UINT32_MAX;

static inline uint16_t llmm_le16(const uint8_t *bytes)
{
    return (uint16_t)bytes[0] | (uint16_t)((uint16_t)bytes[1] << 8U);
}

static inline uint32_t llmm_le32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) |
           ((uint32_t)bytes[2] << 16U) | ((uint32_t)bytes[3] << 24U);
}

static inline const uint8_t *llmm_record(const llmm_handle_t *model, uint16_t tensor_id)
{
    return model->artifact + model->index_offset + (size_t)tensor_id * 32U;
}

static const uint8_t *llmm_record_address(const llmm_handle_t *model, uint16_t tensor_id)
{
    if (model == NULL || model->reader_context == NULL) return NULL;
    const uint8_t *record = llmm_record(model, tensor_id);
    if (llmm_le32(record + 18U) != 0U || llmm_le32(record + 26U) != 0U) return NULL;
    const size_t offset = llmm_le32(record + 14U);
    const size_t bytes = llmm_le32(record + 22U);
    const llmm_storage_t *storage = model->reader_context;
    if (record[2] == 1U && offset <= storage->ple_xip_bytes &&
        bytes <= storage->ple_xip_bytes - offset) {
        return storage->ple_xip + offset;
    }
    if (record[2] == 2U && offset <= storage->psram_payload_bytes &&
        bytes <= storage->psram_payload_bytes - offset) {
        return storage->psram_payload + offset;
    }
    return NULL;
}

static inline float llmm_fp16_to_f32(uint16_t bits)
{
    const float sign = (bits & 0x8000U) == 0U ? 1.0f : -1.0f;
    const uint32_t exponent = (bits >> 10U) & 0x1fU;
    const uint32_t fraction = bits & 0x03ffU;
    if (exponent == 0U) return sign * (float)fraction * 0.000000059604644775390625f;
    if (exponent == 31U) return fraction == 0U ? sign * 65504.0f : 0.0f;
    union {
        uint32_t bits;
        float value;
    } converted = {
        .bits = ((bits & 0x8000U) != 0U ? 0x80000000U : 0U) |
                ((exponent + 112U) << 23U) | (fraction << 13U),
    };
    return converted.value;
}

static float llmm_quantize_a8(const float *input, size_t elements, int8_t *output)
{
    float maximum = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        const float absolute = input[index] < 0.0f ? -input[index] : input[index];
        if (absolute > maximum) maximum = absolute;
    }
    const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    for (size_t index = 0; index < elements; ++index) {
        const float raw = input[index] / scale;
        int32_t quantized = raw >= 0.0f ? (int32_t)(raw + 0.5f) : (int32_t)(raw - 0.5f);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[index] = (int8_t)quantized;
    }
    return scale;
}

static SPM_IRAM_ATTR __attribute__((target("arch=+xespv"), optimize("O3"), noinline))
int32_t llmm_xespv_dot_i8(const int8_t *input, const int8_t *weights, size_t elements)
{
    const uint32_t blocks = (uint32_t)(elements >> 4U);
    int32_t result;
    uint32_t shift = 0U;
    if (blocks == 1U) {
        __asm__ volatile(
            "esp.vld.128.ip q0, %[input], 16\n"
            "esp.vld.128.ip q1, %[weights], 16\n"
            "esp.zero.xacc\n"
            "esp.vmulas.s8.xacc q0, q1\n"
            "esp.srs.s.xacc %[result], %[shift]\n"
            : [result] "=r"(result), [input] "+r"(input), [weights] "+r"(weights),
              [shift] "+r"(shift)
            :
            : "memory");
        return result;
    }

    uint32_t pipelined_blocks = blocks - 1U;
    __asm__ volatile(
        "mv a0, %[input]\n"
        "mv a1, %[weights]\n"
        "esp.zero.xacc\n"
        "esp.vld.128.ip q0, a0, 16\n"
        "esp.vld.128.ip q1, a1, 16\n"
        "esp.lp.setup 0, %[blocks], 1f\n"
        "esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1\n"
        "1: esp.vld.128.ip q1, a1, 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.srs.s.xacc %[result], %[shift]\n"
        : [result] "=r"(result), [blocks] "+r"(pipelined_blocks), [shift] "+r"(shift)
        : [input] "r"(input), [weights] "r"(weights)
        : "a0", "a1", "memory");
    return result;
}

static SPM_IRAM_ATTR __attribute__((target("arch=+xespv"), optimize("O3"), noinline))
int32_t llmm_xespv_dot32_i8(const int8_t *input, const int8_t *weights)
{
    int32_t result;
    uint32_t shift = 0U;
    __asm__ volatile(
        "mv a0, %[input]\n"
        "mv a1, %[weights]\n"
        "esp.zero.xacc\n"
        "esp.vld.128.ip q0, a0, 16\n"
        "esp.vld.128.ip q1, a1, 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.vld.128.ip q0, a0, 16\n"
        "esp.vld.128.ip q1, a1, 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.srs.s.xacc %[result], %[shift]\n"
        : [result] "=r"(result), [shift] "+r"(shift)
        : [input] "r"(input), [weights] "r"(weights)
        : "a0", "a1", "memory");
    return result;
}

static __attribute__((optimize("O3"))) void llmm_decode_base3_row(
    const uint8_t *weights, size_t element_offset, size_t elements, int8_t *decoded)
{
    const uint8_t *packed = weights + element_offset / 5U;
    const size_t first_lane = element_offset % 5U;
    size_t written = 0U;
    if (first_lane != 0U) {
        const size_t count = elements < 5U - first_lane ? elements : 5U - first_lane;
        memcpy(decoded, (const int8_t *)&llmm_base3_lut8[*packed++] + first_lane, count);
        written = count;
    }
    while (written < elements) {
        memcpy(decoded + written, &llmm_base3_lut8[*packed++], sizeof(uint64_t));
        written += 5U;
    }
}

static __attribute__((target("arch=+xespv"), optimize("O3"))) void llmm_base3_range(
    const uint8_t *weights, float *output, size_t base_row, size_t first_row,
    size_t last_row, size_t columns, float output_scale, unsigned core)
{
    int8_t *decoded = llmm_base3_tiles[core];
    for (size_t row = first_row; row < last_row; ++row) {
        llmm_decode_base3_row(weights, row * columns, columns, decoded);
        const int32_t sum = llmm_xespv_dot_i8(llmm_accel_input, decoded, columns);
        output[row - base_row] = (float)sum * output_scale;
    }
}

static int llmm_base3_parallel_matvec(
    const uint8_t *weights, float *output, size_t row_start, size_t rows,
    size_t columns, float output_scale)
{
    if (rows == 1U || llmm_accel_worker_task == NULL) {
        llmm_base3_range(weights, output, row_start, row_start, row_start + rows,
                         columns, output_scale, 0U);
        return 0;
    }

    const size_t split = row_start + (rows + 1U) / 2U;
    llmm_accel_job.kind = LLMM_ACCEL_BASE3;
    llmm_accel_job.weights = weights;
    llmm_accel_job.output = output;
    llmm_accel_job.base_row = row_start;
    llmm_accel_job.first_row = split;
    llmm_accel_job.last_row = row_start + rows;
    llmm_accel_job.columns = columns;
    llmm_accel_job.output_scale = output_scale;
    llmm_accel_job.status = -1;
#if LLMM_DEBUG
    llmm_debug_mark_dispatch();
#endif
    __sync_synchronize();
    xTaskNotifyGive(llmm_accel_worker_task);
    llmm_base3_range(
        weights, output, row_start, row_start, split, columns, output_scale, 0U);
#if LLMM_DEBUG
    const uint32_t wait_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#if LLMM_DEBUG
    llmm_debug_owner_wait(LLMM_ACCEL_BASE3, wait_started);
#endif
    __sync_synchronize();
    llmm_accel_job.kind = LLMM_ACCEL_IDLE;
    return llmm_accel_job.status;
}

static inline void llmm_insert_candidate(llmm_candidate_t *candidates, size_t *count,
                                         size_t capacity, llmm_candidate_t candidate)
{
    if (*count == capacity && candidate.logit <= candidates[capacity - 1U].logit) return;
    size_t slot = *count < capacity ? *count : capacity - 1U;
    if (*count < capacity) ++*count;
    while (slot > 0U && candidate.logit > candidates[slot - 1U].logit) {
        candidates[slot] = candidates[slot - 1U];
        --slot;
    }
    candidates[slot] = candidate;
}

static __attribute__((target("arch=+xespv"), optimize("O3"))) void llmm_head_range(
    const int8_t *weights, const uint8_t *scales, size_t first_row, size_t last_row,
    float activation_scale, size_t top_k, llmm_candidate_t *candidates, size_t *count)
{
    *count = 0U;
    for (size_t row = first_row; row < last_row; ++row) {
        const int32_t sum = llmm_xespv_dot_i8(llmm_accel_input, weights + row * LLMM_WIDTH, LLMM_WIDTH);
        const float row_scale = llmm_fp16_to_f32(llmm_le16(scales + row * 2U));
        const llmm_candidate_t candidate = {
            .token = (uint32_t)row,
            .logit = (float)sum * activation_scale * row_scale,
        };
        llmm_insert_candidate(candidates, count, top_k, candidate);
    }
}

static __attribute__((optimize("O3"))) float llmm_fast_exp_negative(float value)
{
    if (value <= -20.0f) return 0.0f;
    const float scaled = value * 1.4426950408889634f;
    const int32_t exponent = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    const float remainder = __builtin_fmaf((float)-exponent, 0.6931471805599453f, value);
    float polynomial = 0.001388888888888889f;
    polynomial = __builtin_fmaf(polynomial, remainder, 0.008333333333333333f);
    polynomial = __builtin_fmaf(polynomial, remainder, 0.041666666666666664f);
    polynomial = __builtin_fmaf(polynomial, remainder, 0.166666666666666667f);
    polynomial = __builtin_fmaf(polynomial, remainder, 0.5f);
    polynomial = __builtin_fmaf(polynomial, remainder, 1.0f);
    polynomial = __builtin_fmaf(polynomial, remainder, 1.0f);
    union {
        uint32_t bits;
        float value;
    } power_of_two = { .bits = (uint32_t)(exponent + 127) << 23U };
    return polynomial * power_of_two.value;
}

static inline float llmm_fast_sigmoid(float value)
{
    const float exponential = llmm_fast_exp_negative(value < 0.0f ? value : -value);
    return value >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
}

static void llmm_accel_worker(void *argument)
{
    (void)argument;
#if LLMM_DEBUG
    llmm_hpm_configure();
#endif
    xTaskNotifyGive(llmm_accel_owner_task);
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        __sync_synchronize();
#if LLMM_DEBUG
        llmm_profile_t *profile = llmm_active_profile;
        const llmm_accel_kind_t profile_kind = llmm_accel_job.kind;
        const size_t profile_index = llmm_accel_profile_index(profile_kind);
        const uint32_t worker_started = profile != NULL ? llmm_p4_cycle_count() : 0U;
        const uint32_t worker_started_us = profile != NULL ? (uint32_t)esp_timer_get_time() : 0U;
        llmm_hpm_snapshot_t worker_hpm = {0};
        if (profile != NULL) {
            const uint64_t idle = llmm_profile_us_to_cycles(
                worker_started_us - llmm_worker_last_finished_us);
            llmm_profile_stat(&profile->worker_idle_cycles, &profile->worker_idle_max_cycles,
                              &profile->worker_idle_calls, idle);
            llmm_profile_stat(&profile->dispatch_cycles[profile_index],
                              &profile->dispatch_max_cycles[profile_index],
                              &profile->dispatch_calls[profile_index],
                              llmm_profile_us_to_cycles(
                                  worker_started_us - llmm_accel_job.dispatch_started_us));
            worker_hpm = llmm_hpm_read();
        }
#endif
        if (llmm_accel_job.kind == LLMM_ACCEL_BASE3) {
            llmm_base3_range(llmm_accel_job.weights, llmm_accel_job.output,
                             llmm_accel_job.base_row, llmm_accel_job.first_row,
                             llmm_accel_job.last_row, llmm_accel_job.columns,
                             llmm_accel_job.output_scale, 1U);
            llmm_accel_job.status = 0;
        } else if (llmm_accel_job.kind == LLMM_ACCEL_HEAD) {
            llmm_head_range((const int8_t *)llmm_accel_job.weights, llmm_accel_job.scales,
                            llmm_accel_job.first_row, llmm_accel_job.last_row,
                            llmm_accel_job.output_scale, llmm_accel_job.top_k,
                            llmm_accel_job.candidates, &llmm_accel_job.candidate_count);
            llmm_accel_job.status = 0;
        } else if (llmm_accel_job.kind == LLMM_ACCEL_ATTENTION) {
            llmm_attention_kv_range(
                llmm_accel_job.attention_keys, llmm_accel_job.attention_key_scales,
                llmm_accel_job.attention_values, llmm_accel_job.attention_value_scales,
                llmm_accel_job.attention_tokens, llmm_accel_job.first_row,
                llmm_accel_job.last_row);
            llmm_accel_job.status = 0;
        } else {
            llmm_accel_job.status = -1;
        }
#if LLMM_DEBUG
        if (profile != NULL) {
            const llmm_hpm_snapshot_t current = llmm_hpm_read();
            const uint32_t worker_finished = llmm_p4_cycle_count();
            llmm_profile_stat(&profile->worker_busy_cycles[profile_index],
                              &profile->worker_busy_max_cycles[profile_index],
                              &profile->worker_busy_calls[profile_index],
                              worker_finished - worker_started);
            profile->cpu_cycles[1] += current.cycles - worker_hpm.cycles;
            profile->cpu_instructions[1] += current.instructions - worker_hpm.instructions;
            profile->cpu_branch_misses[1] += current.branch_misses - worker_hpm.branch_misses;
            profile->cpu_conditional_branches[1] +=
                current.conditional_branches - worker_hpm.conditional_branches;
            profile->cpu_stores[1] += current.stores - worker_hpm.stores;
            llmm_worker_last_finished_us = (uint32_t)esp_timer_get_time();
        }
#endif
        __sync_synchronize();
        xTaskNotifyGive(llmm_accel_owner_task);
    }
}

static void llmm_base3_lut_init(void)
{
    for (uint32_t packed = 0; packed < 243U; ++packed) {
        uint32_t trits = packed;
        uint64_t expanded = 0U;
        for (uint32_t lane = 0; lane < 5U; ++lane) {
            const int8_t value = (int8_t)(trits % 3U) - 1;
            expanded |= (uint64_t)(uint8_t)value << (lane * 8U);
            trits /= 3U;
        }
        llmm_base3_lut8[packed] = expanded;
    }
}

static int llmm_accel_init(void)
{
    if (llmm_accel_worker_task != NULL) return 0;
    llmm_base3_lut_init();
    llmm_accel_owner_task = xTaskGetCurrentTaskHandle();
    const BaseType_t created = xTaskCreatePinnedToCore(
        llmm_accel_worker, "llmm_core1", LLMM_ACCEL_WORKER_STACK_BYTES, NULL,
        5, &llmm_accel_worker_task, 1);
    if (created != pdPASS) return -1;
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    ESP_LOGI("llmm-accel", "XespV pipeline worker ready: stack=%u B, base3-LUT=%u B",
             LLMM_ACCEL_WORKER_STACK_BYTES, (unsigned)sizeof(llmm_base3_lut8));
    return 0;
}

int llmm_p4_base3_matvec(const llmm_handle_t *model, uint16_t tensor_id,
                         const float *input, float *output, size_t row_start,
                         size_t rows, size_t columns, uint8_t *scratch,
                         size_t scratch_bytes)
{
    (void)scratch;
    (void)scratch_bytes;
    if (columns == 0U || columns > LLMM_MAX_COLUMNS || (columns & 15U) != 0U) return -1;
    const uint8_t *weights = llmm_record_address(model, tensor_id);
    const uint8_t *scale_record = llmm_record_address(model, LLMM_TERNARY_SCALE_TENSOR);
    if (weights == NULL || scale_record == NULL) return -2;
    const uint16_t scale_index = llmm_le16(llmm_record(model, tensor_id) + 4U);
    if (scale_index == UINT16_MAX) return -3;
    if (rows == 0U) return 0;

#if LLMM_DEBUG
    const uint32_t traffic_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
    const float activation_scale = llmm_quantize_a8(input, columns, llmm_accel_input);
    const float weight_scale = llmm_fp16_to_f32(llmm_le16(scale_record + (size_t)scale_index * 2U));
    const float output_scale = activation_scale * weight_scale;
    const int status = llmm_base3_parallel_matvec(
        weights, output, row_start, rows, columns, output_scale);
#if LLMM_DEBUG
    const size_t first_byte = row_start * columns / 5U;
    const size_t last_byte = ((row_start + rows) * columns + 4U) / 5U;
    llmm_debug_traffic(0U, last_byte - first_byte,
                       llmm_p4_cycle_count() - traffic_started);
#endif
    return status;
}

__attribute__((optimize("O3"))) int llmm_p4_rmsnorm(
    const llmm_handle_t *model, uint16_t norm_id, const float *input,
    float *output, size_t elements, float epsilon)
{
    const uint8_t *weights = llmm_record_address(model, norm_id);
    if (weights == NULL || elements == 0U) return -1;
    float sum = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        sum += input[index] * input[index];
    }
    const float safe_epsilon = epsilon > 0.0f ? epsilon : 0.00001f;
    const float inverse_rms = 1.0f / sqrtf(sum / (float)elements + safe_epsilon);
    for (size_t index = 0; index < elements; ++index) {
        output[index] = input[index] * inverse_rms *
                        llmm_fp16_to_f32(llmm_le16(weights + index * 2U));
    }
    return 0;
}

static __attribute__((optimize("O3"))) void llmm_rope_set_position(uint32_t position)
{
    if (llmm_rope_position == position) {
#if LLMM_DEBUG
        if (llmm_active_profile != NULL) llmm_active_profile->rope_same += 1U;
#endif
        return;
    }
#if LLMM_DEBUG
    if (llmm_active_profile != NULL) {
        if (llmm_rope_position != UINT32_MAX && position == llmm_rope_position + 1U) {
            llmm_active_profile->rope_sequential += 1U;
        } else {
            llmm_active_profile->rope_rebuild += 1U;
        }
    }
#endif
    if (position == 0U) {
        for (size_t pair = 0; pair < LLMM_HEAD_DIM / 2U; ++pair) {
            llmm_rope_cos[pair] = 1.0f;
            llmm_rope_sin[pair] = 0.0f;
        }
    } else if (llmm_rope_position != UINT32_MAX && position == llmm_rope_position + 1U) {
        for (size_t pair = 0; pair < LLMM_HEAD_DIM / 2U; ++pair) {
            const float cosine = llmm_rope_cos[pair];
            const float sine = llmm_rope_sin[pair];
            llmm_rope_cos[pair] = __builtin_fmaf(cosine, llmm_rope_step_cos[pair],
                                                  -sine * llmm_rope_step_sin[pair]);
            llmm_rope_sin[pair] = __builtin_fmaf(sine, llmm_rope_step_cos[pair],
                                                  cosine * llmm_rope_step_sin[pair]);
        }
    } else {
        for (size_t pair = 0; pair < LLMM_HEAD_DIM / 2U; ++pair) {
            float result_cos = 1.0f;
            float result_sin = 0.0f;
            float base_cos = llmm_rope_step_cos[pair];
            float base_sin = llmm_rope_step_sin[pair];
            uint32_t exponent = position;
            while (exponent != 0U) {
                if ((exponent & 1U) != 0U) {
                    const float next_cos = __builtin_fmaf(result_cos, base_cos, -result_sin * base_sin);
                    result_sin = __builtin_fmaf(result_sin, base_cos, result_cos * base_sin);
                    result_cos = next_cos;
                }
                const float next_base_cos = __builtin_fmaf(base_cos, base_cos, -base_sin * base_sin);
                base_sin = 2.0f * base_sin * base_cos;
                base_cos = next_base_cos;
                exponent >>= 1U;
            }
            llmm_rope_cos[pair] = result_cos;
            llmm_rope_sin[pair] = result_sin;
        }
    }
    llmm_rope_position = position;
}

__attribute__((optimize("O3"))) int llmm_p4_rope(float *vector, size_t heads, size_t position)
{
    if (heads == 0U || heads > LLMM_HEADS || position > UINT32_MAX) return -1;
    llmm_rope_set_position((uint32_t)position);
    for (size_t head = 0; head < heads; ++head) {
        for (size_t pair = 0; pair < LLMM_HEAD_DIM / 2U; ++pair) {
            const size_t offset = head * LLMM_HEAD_DIM + pair * 2U;
            const float first = vector[offset];
            const float second = vector[offset + 1U];
            vector[offset] = __builtin_fmaf(first, llmm_rope_cos[pair],
                                             -second * llmm_rope_sin[pair]);
            vector[offset + 1U] = __builtin_fmaf(first, llmm_rope_sin[pair],
                                                  second * llmm_rope_cos[pair]);
        }
    }
    return 0;
}

static __attribute__((target("arch=+xespv"), optimize("O3"))) void llmm_attention_kv_range(
    const int8_t *keys, const uint16_t *key_scales,
    const int8_t *values, const uint16_t *value_scales,
    size_t tokens, size_t first_kv_head, size_t last_kv_head)
{
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t kv_head = first_kv_head; kv_head < last_kv_head; ++kv_head) {
            const size_t cache_offset = (token * LLMM_KV_HEADS + kv_head) * LLMM_HEAD_DIM;
            const int8_t *key = keys + cache_offset;
            const int8_t *value = values + cache_offset;
            const float key_scale = llmm_fp16_to_f32(key_scales[token * LLMM_KV_HEADS + kv_head]);
            const float value_scale = llmm_fp16_to_f32(value_scales[token * LLMM_KV_HEADS + kv_head]);
            for (size_t local_head = 0; local_head < LLMM_HEADS / LLMM_KV_HEADS; ++local_head) {
                const size_t head = kv_head * (LLMM_HEADS / LLMM_KV_HEADS) + local_head;
                const int32_t dot = llmm_xespv_dot32_i8(
                    llmm_attention_query + head * LLMM_HEAD_DIM, key);
                const float score = (float)dot * llmm_attention_query_scales[head] *
                                    key_scale * 0.1767766952966369f;
                float weight;
                if (score > llmm_attention_maximums[kv_head][local_head]) {
                    if (llmm_attention_normalizers[kv_head][local_head] != 0.0f) {
                        const float factor = llmm_fast_exp_negative(
                            llmm_attention_maximums[kv_head][local_head] - score);
                        llmm_attention_normalizers[kv_head][local_head] *= factor;
                        for (size_t dimension = 0; dimension < LLMM_HEAD_DIM; ++dimension) {
                            llmm_attention_accumulators[head][dimension] *= factor;
                        }
                    }
                    llmm_attention_maximums[kv_head][local_head] = score;
                    weight = 1.0f;
                } else {
                    weight = llmm_fast_exp_negative(
                        score - llmm_attention_maximums[kv_head][local_head]);
                }
                llmm_attention_normalizers[kv_head][local_head] += weight;
                for (size_t dimension = 0; dimension < LLMM_HEAD_DIM; ++dimension) {
                    const float addition = weight * (float)value[dimension] * value_scale;
                    llmm_attention_accumulators[head][dimension] += addition;
                }
            }
        }
    }
}

__attribute__((target("arch=+xespv"), optimize("O3"))) int llmm_p4_attention_gqa(
    const float *query, const int8_t *keys, const uint16_t *key_scales,
    const int8_t *values, const uint16_t *value_scales, size_t tokens, float *output)
{
    if (tokens == 0U) return -1;
#if LLMM_DEBUG
    const uint32_t traffic_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
    for (size_t head = 0; head < LLMM_HEADS; ++head) {
        const size_t kv_head = head / (LLMM_HEADS / LLMM_KV_HEADS);
        const size_t local_head = head % (LLMM_HEADS / LLMM_KV_HEADS);
        llmm_attention_query_scales[head] = llmm_quantize_a8(
            query + head * LLMM_HEAD_DIM, LLMM_HEAD_DIM,
            llmm_attention_query + head * LLMM_HEAD_DIM);
        llmm_attention_maximums[kv_head][local_head] = -3.402823466e38f;
        llmm_attention_normalizers[kv_head][local_head] = 0.0f;
        memset(llmm_attention_accumulators[head], 0,
               LLMM_HEAD_DIM * sizeof(llmm_attention_accumulators[head][0]));
    }

    if (llmm_accel_worker_task != NULL) {
        llmm_accel_job.kind = LLMM_ACCEL_ATTENTION;
        llmm_accel_job.attention_keys = keys;
        llmm_accel_job.attention_key_scales = key_scales;
        llmm_accel_job.attention_values = values;
        llmm_accel_job.attention_value_scales = value_scales;
        llmm_accel_job.attention_tokens = tokens;
        llmm_accel_job.first_row = 1U;
        llmm_accel_job.last_row = LLMM_KV_HEADS;
        llmm_accel_job.status = -1;
#if LLMM_DEBUG
        llmm_debug_mark_dispatch();
#endif
        __sync_synchronize();
        xTaskNotifyGive(llmm_accel_worker_task);
        llmm_attention_kv_range(keys, key_scales, values, value_scales, tokens, 0U, 1U);
#if LLMM_DEBUG
        const uint32_t wait_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#if LLMM_DEBUG
        llmm_debug_owner_wait(LLMM_ACCEL_ATTENTION, wait_started);
#endif
        __sync_synchronize();
        if (llmm_accel_job.status != 0) {
#if LLMM_DEBUG
            const size_t bytes = tokens * LLMM_KV_HEADS *
                (2U * LLMM_HEAD_DIM * (LLMM_HEADS / LLMM_KV_HEADS) + 2U * sizeof(uint16_t));
            llmm_debug_traffic(2U, bytes, llmm_p4_cycle_count() - traffic_started);
#endif
            return llmm_accel_job.status;
        }
        llmm_accel_job.kind = LLMM_ACCEL_IDLE;
    } else {
        llmm_attention_kv_range(
            keys, key_scales, values, value_scales, tokens, 0U, LLMM_KV_HEADS);
    }

    for (size_t head = 0; head < LLMM_HEADS; ++head) {
        const size_t kv_head = head / (LLMM_HEADS / LLMM_KV_HEADS);
        const size_t local_head = head % (LLMM_HEADS / LLMM_KV_HEADS);
        for (size_t dimension = 0; dimension < LLMM_HEAD_DIM; ++dimension) {
            output[head * LLMM_HEAD_DIM + dimension] =
                llmm_attention_accumulators[head][dimension] /
                llmm_attention_normalizers[kv_head][local_head];
        }
    }
#if LLMM_DEBUG
    const size_t bytes = tokens * LLMM_KV_HEADS *
        (2U * LLMM_HEAD_DIM * (LLMM_HEADS / LLMM_KV_HEADS) + 2U * sizeof(uint16_t));
    llmm_debug_traffic(2U, bytes, llmm_p4_cycle_count() - traffic_started);
#endif
    return 0;
}

__attribute__((optimize("O3"))) int llmm_p4_router_top1(
    const float *logits, size_t experts, uint32_t *selected, float *probability)
{
    if (experts == 0U || experts > 29U) return -1;
    size_t best = 0U;
    float maximum = logits[0];
    for (size_t expert = 1U; expert < experts; ++expert) {
        if (logits[expert] > maximum) {
            maximum = logits[expert];
            best = expert;
        }
    }
    float normalizer = 0.0f;
    for (size_t expert = 0U; expert < experts; ++expert) {
        normalizer += llmm_fast_exp_negative(logits[expert] - maximum);
    }
    *selected = (uint32_t)best;
    *probability = 1.0f / normalizer;
    return 0;
}

__attribute__((optimize("O3"))) int llmm_p4_silu_mul(
    float *gate, const float *up, size_t elements)
{
    for (size_t index = 0; index < elements; ++index) {
        gate[index] = gate[index] * llmm_fast_sigmoid(gate[index]) * up[index];
    }
    return 0;
}

__attribute__((optimize("O3"))) int llmm_p4_gelu_mul(
    float *value, const float *multiplier, size_t elements)
{
    for (size_t index = 0; index < elements; ++index) {
        const float input = value[index];
        const float gelu = 0.5f * input * (1.0f + erff(input * 0.7071067811865475f));
        value[index] = gelu * multiplier[index];
    }
    return 0;
}

int llmm_p4_output_head_topk(const llmm_handle_t *model, const float *normalized,
                             llmm_candidate_t *candidates, size_t candidate_capacity,
                             size_t requested_top_k, size_t *candidate_count)
{
    if (candidate_capacity == 0U || requested_top_k == 0U ||
        requested_top_k > candidate_capacity || requested_top_k > LLMM_TOP_K_MAX) return -1;
    const int8_t *weights = (const int8_t *)llmm_record_address(model, 1U);
    const uint8_t *scales = llmm_record_address(model, 2U);
    if (weights == NULL || scales == NULL) return -2;
#if LLMM_DEBUG
    const uint32_t traffic_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
    const float activation_scale = llmm_quantize_a8(normalized, LLMM_WIDTH, llmm_accel_input);
    size_t local_count = 0U;

    llmm_accel_job.kind = LLMM_ACCEL_HEAD;
    llmm_accel_job.weights = (const uint8_t *)weights;
    llmm_accel_job.scales = scales;
    llmm_accel_job.first_row = LLMM_VOCAB / 2U;
    llmm_accel_job.last_row = LLMM_VOCAB;
    llmm_accel_job.output_scale = activation_scale;
    llmm_accel_job.top_k = requested_top_k;
    llmm_accel_job.status = -1;
#if LLMM_DEBUG
    llmm_debug_mark_dispatch();
#endif
    __sync_synchronize();
    xTaskNotifyGive(llmm_accel_worker_task);
    llmm_head_range(weights, scales, 0U, LLMM_VOCAB / 2U, activation_scale,
                    requested_top_k, candidates, &local_count);
#if LLMM_DEBUG
    const uint32_t wait_started = llmm_active_profile != NULL ? llmm_p4_cycle_count() : 0U;
#endif
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#if LLMM_DEBUG
    llmm_debug_owner_wait(LLMM_ACCEL_HEAD, wait_started);
#endif
    __sync_synchronize();
    if (llmm_accel_job.status != 0) {
#if LLMM_DEBUG
        llmm_debug_traffic(1U, LLMM_VOCAB * (LLMM_WIDTH + sizeof(uint16_t)),
                           llmm_p4_cycle_count() - traffic_started);
#endif
        return llmm_accel_job.status;
    }
    llmm_accel_job.kind = LLMM_ACCEL_IDLE;
    for (size_t index = 0; index < llmm_accel_job.candidate_count; ++index) {
        llmm_insert_candidate(candidates, &local_count, requested_top_k,
                              llmm_accel_job.candidates[index]);
    }
    *candidate_count = local_count;
#if LLMM_DEBUG
    llmm_debug_traffic(1U, LLMM_VOCAB * (LLMM_WIDTH + sizeof(uint16_t)),
                       llmm_p4_cycle_count() - traffic_started);
#endif
    return 0;
}

#if LLMM_DEBUG
static void gemv_q4_range(uint32_t first, uint32_t last, int8_t *unpacked)
{
    for (uint32_t output = first; output < last; ++output) {
        const uint8_t *packed = &gemv_q4_weights[output * GEMV_K / 2U];
        for (uint32_t index = 0; index < GEMV_K / 2U; ++index) {
            const uint8_t value = packed[index];
            unpacked[index * 2U] = (int8_t)((value & 0x0fU) ^ 0x08U) - 8;
            unpacked[index * 2U + 1U] = (int8_t)(((value >> 4U) & 0x0fU) ^ 0x08U) - 8;
        }
        int32_t sum = 0;
        for (uint32_t index = 0; index < GEMV_K; index += 16U) {
            sum += xespv_dot16_i8_ptr(&gemv_input[index], &unpacked[index]);
        }
        gemv_output[output] = sum;
    }
}

static void gemv_q4_packed_range(const uint8_t *packed_weights, uint32_t first, uint32_t last,
                                 int8_t *unpacked)
{
    for (uint32_t output = first; output < last; ++output) {
        const uint8_t *packed = &packed_weights[(output - first) * GEMV_K / 2U];
        for (uint32_t index = 0; index < GEMV_K / 2U; ++index) {
            const uint8_t value = packed[index];
            unpacked[index * 2U] = (int8_t)((value & 0x0fU) ^ 0x08U) - 8;
            unpacked[index * 2U + 1U] = (int8_t)(((value >> 4U) & 0x0fU) ^ 0x08U) - 8;
        }
        int32_t sum = 0;
        for (uint32_t index = 0; index < GEMV_K; index += 16U) {
            sum += xespv_dot16_i8_ptr(&gemv_input[index], &unpacked[index]);
        }
        gemv_output[output] = sum;
    }
}

static uint32_t gemv_q4_staged(const uint8_t *psram_weights, uint8_t *stage0, uint8_t *stage1,
                               uint32_t stage_bytes, int8_t *unpacked)
{
    const uint32_t rows_per_stage = stage_bytes / (GEMV_K / 2U);

    for (uint32_t first = 0, stage = 0; first < GEMV_N; first += rows_per_stage, stage ^= 1U) {
        const uint32_t rows = (GEMV_N - first < rows_per_stage) ? GEMV_N - first : rows_per_stage;
        uint8_t *destination = stage == 0 ? stage0 : stage1;
        const uint32_t bytes = rows * GEMV_K / 2U;
        memcpy(destination, &psram_weights[first * GEMV_K / 2U], bytes);
        gemv_q4_packed_range(destination, first, first + rows, unpacked);
    }
    uint32_t checksum = 2166136261U;
    for (uint32_t output = 0; output < GEMV_N; ++output) {
        checksum = (checksum ^ (uint32_t)gemv_output[output]) * 16777619U;
    }
    return checksum;
}

static uint32_t gemv_output_checksum(void)
{
    uint32_t checksum = 2166136261U;
    for (uint32_t output = 0; output < GEMV_N; ++output) {
        checksum = (checksum ^ (uint32_t)gemv_output[output]) * 16777619U;
    }
    return checksum;
}

static bool dma_copy_done(async_memcpy_handle_t handle, async_memcpy_event_t *event, void *argument)
{
    (void)handle;
    (void)event;
    BaseType_t higher_priority_task_woken = pdFALSE;
    vTaskNotifyGiveFromISR((TaskHandle_t)argument, &higher_priority_task_woken);
    return higher_priority_task_woken == pdTRUE;
}

static uint32_t gemv_q4_dma_pipelined(async_memcpy_handle_t dma, uint8_t *psram_weights,
                                      uint8_t *stage0, uint8_t *stage1, uint32_t stage_bytes,
                                      int8_t *unpacked)
{
    const uint32_t rows_per_stage = stage_bytes / (GEMV_K / 2U);
    const uint32_t stage_count = (GEMV_N + rows_per_stage - 1U) / rows_per_stage;
    dma_wait_task = xTaskGetCurrentTaskHandle();
    if (esp_async_memcpy(dma, stage0, psram_weights, rows_per_stage * GEMV_K / 2U, dma_copy_done, dma_wait_task) != ESP_OK) {
        return 0;
    }
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000)) == 0) return 0;

    for (uint32_t stage = 0; stage < stage_count; ++stage) {
        const uint32_t first = stage * rows_per_stage;
        const uint32_t rows = (GEMV_N - first < rows_per_stage) ? GEMV_N - first : rows_per_stage;
        uint8_t *current = (stage & 1U) == 0 ? stage0 : stage1;
        if (stage + 1U < stage_count) {
            const uint32_t next_first = (stage + 1U) * rows_per_stage;
            const uint32_t next_rows = (GEMV_N - next_first < rows_per_stage) ? GEMV_N - next_first : rows_per_stage;
            uint8_t *next = (stage & 1U) == 0 ? stage1 : stage0;
            if (esp_async_memcpy(dma, next, &psram_weights[next_first * GEMV_K / 2U],
                                 next_rows * GEMV_K / 2U, dma_copy_done, dma_wait_task) != ESP_OK) {
                return 0;
            }
        }
        gemv_q4_packed_range(current, first, first + rows, unpacked);
        if (stage + 1U < stage_count) {
            if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000)) == 0) return 0;
        }
    }
    return gemv_output_checksum();
}

typedef struct {
    uint32_t first;
    uint32_t last;
    uint32_t repetitions;
    int8_t *unpacked;
    int64_t start_us;
    int64_t end_us;
} gemv_worker_t;

static void run_gemv_worker(void *argument)
{
    gemv_worker_t *worker = argument;
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    worker->start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < worker->repetitions; ++repeat) {
        gemv_q4_range(worker->first, worker->last, worker->unpacked);
    }
    worker->end_us = esp_timer_get_time();
    xTaskNotifyGive(benchmark_main_task);
    vTaskDelete(NULL);
}

static int32_t scalar_dot8(uint32_t iterations)
{
    int32_t result = 0;

    for (uint32_t count = 0; count < iterations; ++count) {
        int32_t sum = 0;
        for (uint32_t index = 0; index < 8; ++index) {
            sum += (int32_t)xespv_input[index] * xespv_weights[index];
        }
        result = sum;
    }

    return result;
}

static int32_t q4_dot_product(void)
{
    int32_t sum = 0;

    for (uint32_t index = 0; index < Q4_ELEMENTS / 2U; ++index) {
        const uint8_t packed = q4_weights[index];
        const int8_t low = (int8_t)((packed & 0x0fU) ^ 0x08U) - 8;
        const int8_t high = (int8_t)(((packed >> 4) & 0x0fU) ^ 0x08U) - 8;

        sum += (int32_t)q4_activations[index * 2U] * low;
        sum += (int32_t)q4_activations[index * 2U + 1U] * high;
    }

    return sum;
}

static void matrix_multiply_i8(void)
{
    for (uint32_t row = 0; row < MATRIX_SIZE; ++row) {
        for (uint32_t column = 0; column < MATRIX_SIZE; ++column) {
            int32_t sum = 0;
            for (uint32_t index = 0; index < MATRIX_SIZE; ++index) {
                sum += (int32_t)matrix_a[row * MATRIX_SIZE + index] * matrix_b[index * MATRIX_SIZE + column];
            }
            matrix_c[row * MATRIX_SIZE + column] = sum;
        }
    }
}

static float rms_norm(void)
{
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        square_sum += vector_data[index] * vector_data[index];
    }
    const float scale = 1.0f / sqrtf(square_sum / VECTOR_SIZE + 1e-5f);
    float checksum = 0.0f;
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] *= scale;
        checksum += vector_data[index];
    }
    return checksum;
}

static float softmax(void)
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] = expf(vector_data[index]);
        sum += vector_data[index];
    }
    float checksum = 0.0f;
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] /= sum;
        checksum += vector_data[index];
    }
    return checksum;
}

static float rope(void)
{
    float checksum = 0.0f;
    for (uint32_t index = 0; index < VECTOR_SIZE; index += 2U) {
        const float angle = (float)index * 0.01f;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float left = vector_data[index];
        const float right = vector_data[index + 1U];
        vector_data[index] = left * cosine - right * sine;
        vector_data[index + 1U] = left * sine + right * cosine;
        checksum += vector_data[index] + vector_data[index + 1U];
    }
    return checksum;
}

static float silu(float value)
{
    return value / (1.0f + expf(-value));
}

static uint32_t run_token_operator_chain(void)
{
    rms_norm();
    for (uint32_t index = 0; index < GEMV_K; ++index) {
        const float source = vector_data[index % VECTOR_SIZE];
        gemv_input[index] = (int8_t)(source * 16.0f);
    }
    gemv_q4_range(0, GEMV_N, gemv_unpacked);
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        const float activated = silu((float)gemv_output[index] * 0.01f);
        residual[index] += activated;
        gemv_input[index] = (int8_t)(activated > 127.0f ? 127 : (activated < -128.0f ? -128 : activated));
    }
    for (uint32_t index = VECTOR_SIZE; index < GEMV_K; ++index) {
        gemv_input[index] = gemv_input[index % VECTOR_SIZE];
    }
    gemv_q4_range(0, GEMV_N, gemv_unpacked);
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] = (float)gemv_output[index] * 0.01f + residual[index];
    }
    softmax();

    uint32_t checksum = 0;
    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        checksum += (uint32_t)(vector_data[index] * 1000000.0f);
    }
    return checksum;
}

static int32_t q8_dot(const int8_t *input, const int8_t *weights, uint32_t elements)
{
    int32_t sum = 0;
    for (uint32_t index = 0; index < elements; index += 16U) {
        sum += xespv_dot16_i8_ptr(&input[index], &weights[index]);
    }
    return sum;
}

static uint32_t ternary_lut[256];
static int8_t ternary_base3_lut5[243][5] __attribute__((aligned(16)));
static uint64_t ternary_base3_lut8[243] __attribute__((aligned(16)));

static void initialize_ternary_lut(void)
{
    for (uint32_t packed = 0; packed < 256U; ++packed) {
        uint32_t expanded = 0;
        for (uint32_t lane = 0; lane < 4U; ++lane) {
            const uint32_t code = (packed >> (lane * 2U)) & 0x03U;
            const int8_t value = code == 1U ? 1 : (code == 2U ? -1 : 0);
            expanded |= (uint32_t)(uint8_t)value << (lane * 8U);
        }
        ternary_lut[packed] = expanded;
    }

    for (uint32_t packed = 0; packed < 243U; ++packed) {
        uint32_t trits = packed;
        uint64_t expanded = 0;
        for (uint32_t lane = 0; lane < 5U; ++lane) {
            const uint32_t code = trits % 3U;
            const int8_t value = code == 1U ? 1 : (code == 2U ? -1 : 0);
            ternary_base3_lut5[packed][lane] = value;
            expanded |= (uint64_t)(uint8_t)value << (lane * 8U);
            trits /= 3U;
        }
        ternary_base3_lut8[packed] = expanded;
    }
}

static int32_t ternary_dot_lut(const int8_t *input, const uint8_t *weights, uint32_t elements)
{
    int8_t expanded[16] __attribute__((aligned(16)));
    int32_t sum = 0;
    for (uint32_t index = 0; index < elements; index += 16U) {
        for (uint32_t packed = 0; packed < 4U; ++packed) {
            memcpy(&expanded[packed * 4U], &ternary_lut[weights[index / 4U + packed]], 4U);
        }
        sum += xespv_dot16_i8_ptr(&input[index], expanded);
    }
    return sum;
}

static int32_t ternary_dot_base3_lut5(const int8_t *input, const uint8_t *weights, uint32_t elements)
{
    int8_t expanded[272] __attribute__((aligned(16)));
    const uint32_t groups = (elements + 4U) / 5U;
    for (uint32_t group = 0; group < groups; ++group) {
        memcpy(&expanded[group * 5U], ternary_base3_lut5[weights[group]], 5U);
    }
    return q8_dot(input, expanded, elements);
}

static int32_t ternary_dot_base3_lut8(const int8_t *input, const uint8_t *weights, uint32_t elements)
{
    int8_t expanded[272] __attribute__((aligned(16)));
    const uint32_t groups = (elements + 4U) / 5U;
    for (uint32_t group = 0; group < groups; ++group) {
        memcpy(&expanded[group * 5U], &ternary_base3_lut8[weights[group]], sizeof(uint64_t));
    }
    return q8_dot(input, expanded, elements);
}

static void benchmark_ternary_xespv(void)
{
    const uint32_t rows = 512U;
    const uint32_t columns = 256U;
    const uint32_t repetitions = 10U;
    const uint32_t base3_row_bytes = (columns + 4U) / 5U;
    int8_t input[272] __attribute__((aligned(16)));
    uint8_t *ternary_psram = heap_caps_malloc(rows * columns / 4U, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *ternary_base3_psram = heap_caps_malloc(rows * base3_row_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *expanded_sram = heap_caps_malloc(rows * columns, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (ternary_psram == NULL || ternary_base3_psram == NULL || expanded_sram == NULL) {
        ESP_LOGE("w1.58", "PSRAM ternary/base3 or SRAM expanded allocation failed");
        heap_caps_free(ternary_psram);
        heap_caps_free(ternary_base3_psram);
        heap_caps_free(expanded_sram);
        return;
    }
    initialize_ternary_lut();
    for (uint32_t index = 0; index < columns; ++index) input[index] = (int8_t)((index % 15U) - 7);
    for (uint32_t index = 0; index < rows * columns / 4U; ++index) {
        const uint8_t code0 = (uint8_t)((index * 4U) % 3U);
        const uint8_t code1 = (uint8_t)((index * 4U + 1U) % 3U);
        const uint8_t code2 = (uint8_t)((index * 4U + 2U) % 3U);
        const uint8_t code3 = (uint8_t)((index * 4U + 3U) % 3U);
        ternary_psram[index] = code0 | (code1 << 2U) | (code2 << 4U) | (code3 << 6U);
        memcpy(&expanded_sram[index * 4U], &ternary_lut[ternary_psram[index]], 4U);
    }
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t group = 0; group < base3_row_bytes; ++group) {
            uint32_t packed = 0;
            uint32_t factor = 1U;
            for (uint32_t lane = 0; lane < 5U; ++lane) {
                const uint32_t column = group * 5U + lane;
                const uint32_t code = column < columns ? (row * columns + column) % 3U : 0U;
                packed += code * factor;
                factor *= 3U;
            }
            ternary_base3_psram[row * base3_row_bytes + group] = (uint8_t)packed;
        }
    }
    int32_t lut_checksum = 0;
    const int64_t lut_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        for (uint32_t row = 0; row < rows; ++row) {
            lut_checksum += ternary_dot_lut(input, &ternary_psram[row * columns / 4U], columns);
        }
    }
    const int64_t lut_elapsed_us = esp_timer_get_time() - lut_start_us;
    int32_t base3_lut5_checksum = 0;
    const int64_t base3_lut5_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        for (uint32_t row = 0; row < rows; ++row) {
            base3_lut5_checksum += ternary_dot_base3_lut5(
                input, &ternary_base3_psram[row * base3_row_bytes], columns);
        }
    }
    const int64_t base3_lut5_elapsed_us = esp_timer_get_time() - base3_lut5_start_us;
    int32_t base3_lut8_checksum = 0;
    const int64_t base3_lut8_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        for (uint32_t row = 0; row < rows; ++row) {
            base3_lut8_checksum += ternary_dot_base3_lut8(
                input, &ternary_base3_psram[row * base3_row_bytes], columns);
        }
    }
    const int64_t base3_lut8_elapsed_us = esp_timer_get_time() - base3_lut8_start_us;
    int32_t expanded_checksum = 0;
    const int64_t expanded_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        for (uint32_t row = 0; row < rows; ++row) {
            expanded_checksum += q8_dot(input, &expanded_sram[row * columns], columns);
        }
    }
    const int64_t expanded_elapsed_us = esp_timer_get_time() - expanded_start_us;
    const uint64_t operations = (uint64_t)rows * columns * repetitions;
    ESP_LOGI("w1.58", "2-bit LUT decode + XespV PSRAM: %" PRIu64 " ternary MAC/s, checksum: 0x%08" PRIx32,
             operations * 1000000ULL / (uint64_t)lut_elapsed_us, (uint32_t)lut_checksum);
    ESP_LOGI("w1.58", "base3 LUT5 decode + XespV PSRAM: %" PRIu64 " ternary MAC/s, checksum: 0x%08" PRIx32,
             operations * 1000000ULL / (uint64_t)base3_lut5_elapsed_us, (uint32_t)base3_lut5_checksum);
    ESP_LOGI("w1.58", "base3 LUT8 decode + XespV PSRAM: %" PRIu64 " ternary MAC/s, checksum: 0x%08" PRIx32,
             operations * 1000000ULL / (uint64_t)base3_lut8_elapsed_us, (uint32_t)base3_lut8_checksum);
    ESP_LOGI("w1.58", "pre-expanded XespV SRAM upper bound: %" PRIu64 " MAC/s, checksum: 0x%08" PRIx32,
             operations * 1000000ULL / (uint64_t)expanded_elapsed_us, (uint32_t)expanded_checksum);
    ESP_LOGI("w1.58", "weights: 32 KiB 2-bit, %" PRIu32 " KiB base3, 128 KiB int8; match: %s",
             rows * base3_row_bytes / 1024U,
             lut_checksum == expanded_checksum && base3_lut5_checksum == expanded_checksum &&
                     base3_lut8_checksum == expanded_checksum
                 ? "yes"
                 : "no");
    heap_caps_free(ternary_psram);
    heap_caps_free(ternary_base3_psram);
    heap_caps_free(expanded_sram);
}

static size_t rans_encode(const uint8_t *symbols, uint32_t symbol_count,
                          const uint16_t frequencies[3], uint32_t lanes,
                          uint8_t *output, size_t output_capacity)
{
    uint32_t cumulative[3] = {0U, frequencies[0], frequencies[0] + frequencies[1]};
    uint32_t states[4] = {RANS_BYTE_L, RANS_BYTE_L, RANS_BYTE_L, RANS_BYTE_L};
    uint8_t *cursor = output + output_capacity;

    for (uint32_t index = symbol_count; index-- > 0U;) {
        const uint32_t lane = index & (lanes - 1U);
        const uint32_t symbol = symbols[index];
        const uint32_t frequency = frequencies[symbol];
        uint32_t state = states[lane];
        const uint64_t renormalize_at =
            ((uint64_t)(RANS_BYTE_L >> RANS_SCALE_BITS) << 8U) * frequency;
        while (state >= renormalize_at) {
            if (cursor == output) return 0U;
            *--cursor = (uint8_t)state;
            state >>= 8U;
        }
        states[lane] = (state / frequency) * RANS_SCALE +
                       (state % frequency) + cumulative[symbol];
    }

    const size_t state_bytes = lanes * sizeof(uint32_t);
    if ((size_t)(cursor - output) < state_bytes) return 0U;
    cursor -= state_bytes;
    memcpy(cursor, states, state_bytes);
    const size_t encoded_size = (size_t)(output + output_capacity - cursor);
    memmove(output, cursor, encoded_size);
    return encoded_size;
}

static uint32_t rans_decode_base3(const uint8_t *encoded, size_t encoded_size,
                                  const uint16_t frequencies[3], const uint8_t *lookup,
                                  uint32_t lanes, uint8_t *packed, size_t *consumed)
{
    uint32_t cumulative[3] = {0U, frequencies[0], frequencies[0] + frequencies[1]};
    uint32_t states[4];
    const size_t state_bytes = lanes * sizeof(uint32_t);
    memcpy(states, encoded, state_bytes);
    const uint8_t *cursor = encoded + state_bytes;
    const uint8_t *end = encoded + encoded_size;
    uint32_t checksum = 0U;

    for (uint32_t group = 0; group < RANS_PACKED_BYTES; ++group) {
        uint32_t packed_value = 0U;
        uint32_t factor = 1U;
        for (uint32_t lane_in_group = 0; lane_in_group < 5U; ++lane_in_group) {
            const uint32_t index = group * 5U + lane_in_group;
            const uint32_t lane = index & (lanes - 1U);
            uint32_t state = states[lane];
            const uint32_t slot = state & (RANS_SCALE - 1U);
            const uint32_t symbol = lookup[slot];
            state = frequencies[symbol] * (state >> RANS_SCALE_BITS) +
                    slot - cumulative[symbol];
            while (state < RANS_BYTE_L && cursor < end) {
                state = (state << 8U) | *cursor++;
            }
            states[lane] = state;
            packed_value += symbol * factor;
            factor *= 3U;
        }
        packed[group] = (uint8_t)packed_value;
        checksum += packed_value;
    }

    *consumed = (size_t)(cursor - encoded);
    return checksum;
}

static void benchmark_rans_case(const char *name, const uint16_t frequencies[3])
{
    uint8_t *symbols = heap_caps_malloc(RANS_TRITS, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *encoded = heap_caps_malloc(RANS_TRITS + 32U, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *expected = heap_caps_malloc(RANS_PACKED_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *decoded = heap_caps_malloc(RANS_PACKED_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *lookup = heap_caps_malloc(RANS_SCALE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (symbols == NULL || encoded == NULL || expected == NULL || decoded == NULL || lookup == NULL) {
        ESP_LOGE("rans", "%s allocation failed", name);
        heap_caps_free(symbols);
        heap_caps_free(encoded);
        heap_caps_free(expected);
        heap_caps_free(decoded);
        heap_caps_free(lookup);
        return;
    }

    uint32_t cumulative = 0U;
    for (uint32_t symbol = 0; symbol < 3U; ++symbol) {
        memset(&lookup[cumulative], (int)symbol, frequencies[symbol]);
        cumulative += frequencies[symbol];
    }
    uint32_t random_state = 0x5eed1234U;
    for (uint32_t index = 0; index < RANS_TRITS; ++index) {
        random_state = random_state * 1664525U + 1013904223U;
        symbols[index] = lookup[random_state & (RANS_SCALE - 1U)];
    }
    for (uint32_t group = 0; group < RANS_PACKED_BYTES; ++group) {
        uint32_t value = 0U;
        uint32_t factor = 1U;
        for (uint32_t lane = 0; lane < 5U; ++lane) {
            value += symbols[group * 5U + lane] * factor;
            factor *= 3U;
        }
        expected[group] = (uint8_t)value;
    }

    for (uint32_t lanes = 1U; lanes <= 4U; lanes *= 4U) {
        const size_t encoded_size = rans_encode(symbols, RANS_TRITS, frequencies, lanes,
                                                encoded, RANS_TRITS + 32U);
        size_t consumed = 0U;
        const uint32_t expected_checksum = rans_decode_base3(
            encoded, encoded_size, frequencies, lookup, lanes, decoded, &consumed);
        const bool matches = encoded_size > 0U && consumed == encoded_size &&
                             memcmp(expected, decoded, RANS_PACKED_BYTES) == 0;
        const int64_t start_us = esp_timer_get_time();
        uint32_t checksum = 0U;
        for (uint32_t repeat = 0; repeat < RANS_REPETITIONS; ++repeat) {
            checksum += rans_decode_base3(
                encoded, encoded_size, frequencies, lookup, lanes, decoded, &consumed);
        }
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        const uint64_t packed_mib_per_second_x100 =
            (uint64_t)RANS_PACKED_BYTES * RANS_REPETITIONS * 100000000ULL /
            ((uint64_t)elapsed_us * 1024ULL * 1024ULL);
        const uint64_t million_trits_per_second_x100 =
            (uint64_t)RANS_TRITS * RANS_REPETITIONS * 100ULL /
            (uint64_t)elapsed_us;
        const uint32_t framed_size = (uint32_t)encoded_size + 16U;
        const uint32_t saved_basis_points =
            10000U - framed_size * 10000U / RANS_PACKED_BYTES;
        ESP_LOGI("rans", "%s %" PRIu32 "-state: %u -> %" PRIu32
                 " B, saved %" PRIu32 ".%02" PRIu32 "%%, %" PRIu64 ".%02" PRIu64
                 " MiB/s packed, %" PRIu64 ".%02" PRIu64 " Mtrit/s, %" PRIu64
                 " us/page, match: %s, checksum: 0x%08" PRIx32 "/0x%08" PRIx32,
                 name, lanes, RANS_PACKED_BYTES, framed_size,
                 saved_basis_points / 100U, saved_basis_points % 100U,
                 packed_mib_per_second_x100 / 100U, packed_mib_per_second_x100 % 100U,
                 million_trits_per_second_x100 / 100U, million_trits_per_second_x100 % 100U,
                 (uint64_t)elapsed_us / RANS_REPETITIONS,
                 matches ? "yes" : "no", checksum, expected_checksum * RANS_REPETITIONS);
    }

    heap_caps_free(symbols);
    heap_caps_free(encoded);
    heap_caps_free(expected);
    heap_caps_free(decoded);
    heap_caps_free(lookup);
}

static void benchmark_rans(void)
{
    const uint16_t checkpoint_frequencies[3] = {1413U, 1270U, 1413U};
    const uint16_t zero50_frequencies[3] = {1024U, 2048U, 1024U};
    const uint16_t zero70_frequencies[3] = {614U, 2868U, 614U};
    benchmark_rans_case("checkpoint-z31", checkpoint_frequencies);
    benchmark_rans_case("synthetic-z50", zero50_frequencies);
    benchmark_rans_case("synthetic-z70", zero70_frequencies);
}

static void benchmark_gemv_shapes(void)
{
    const uint32_t shapes[][2] = {{128U, 256U}, {256U, 256U}, {256U, 512U}, {1024U, 256U}};
    for (uint32_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        const uint32_t rows = shapes[shape][0];
        const uint32_t columns = shapes[shape][1];
        const uint32_t repetitions = (rows * columns <= 65536U) ? 40U : 10U;
        int8_t *input = heap_caps_malloc(columns + 16U, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        int8_t *weights = heap_caps_malloc(rows * columns, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        const char *location = "SRAM";
        if (weights == NULL) {
            weights = heap_caps_malloc(rows * columns, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            location = "PSRAM";
        }
        if (input == NULL || weights == NULL) {
            ESP_LOGE("gemv-shape", "allocation failed for %" PRIu32 "x%" PRIu32, rows, columns);
            heap_caps_free(input);
            heap_caps_free(weights);
            continue;
        }
        for (uint32_t index = 0; index < columns; ++index) input[index] = (int8_t)((index % 15U) - 7);
        for (uint32_t index = 0; index < rows * columns; ++index) weights[index] = (int8_t)((index % 13U) - 6);
        int32_t checksum = 0;
        const int64_t start_us = esp_timer_get_time();
        for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
            for (uint32_t row = 0; row < rows; ++row) checksum += q8_dot(input, &weights[row * columns], columns);
        }
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        ESP_LOGI("gemv-shape", "int8 %" PRIu32 "x%" PRIu32 " %s: %" PRIu64 " MAC/s, checksum: 0x%08" PRIx32,
                 rows, columns, location, ((uint64_t)rows * columns * repetitions * 1000000ULL) / (uint64_t)elapsed_us,
                 (uint32_t)checksum);
        heap_caps_free(input);
        heap_caps_free(weights);
    }
}

static void benchmark_quant_formats(void)
{
    int8_t input[144] __attribute__((aligned(16)));
    int8_t q8_weights[128] __attribute__((aligned(16)));
    uint8_t q4_weights[64] = {0};
    float reference = 0.0f;
    for (uint32_t index = 0; index < 128U; ++index) {
        const float activation = (float)((int32_t)(index % 15U) - 7) * 0.25f;
        const float weight = (float)((int32_t)(index % 23U) - 11) * 0.10f;
        input[index] = (int8_t)roundf(activation / 0.25f);
        q8_weights[index] = (int8_t)roundf(weight / 0.01f);
        const int8_t q4 = (int8_t)roundf(weight / 0.20f);
        q4_weights[index / 2U] |= (uint8_t)((uint8_t)q4 & 0x0fU) << ((index & 1U) * 4U);
        reference += activation * weight;
    }
    const int64_t int8_start_us = esp_timer_get_time();
    int32_t int8_sum = 0;
    for (uint32_t repeat = 0; repeat < 10000U; ++repeat) int8_sum = q8_dot(input, q8_weights, 128U);
    const int64_t int8_elapsed_us = esp_timer_get_time() - int8_start_us;
    const float q8_result = (float)int8_sum * 0.0025f;
    const int64_t q4_start_us = esp_timer_get_time();
    int32_t q4_sum = 0;
    for (uint32_t repeat = 0; repeat < 10000U; ++repeat) {
        q4_sum = 0;
        for (uint32_t index = 0; index < 64U; ++index) {
            const uint8_t packed = q4_weights[index];
            q4_sum += input[index * 2U] * ((int8_t)((packed & 0x0fU) ^ 0x08U) - 8);
            q4_sum += input[index * 2U + 1U] * ((int8_t)(((packed >> 4U) & 0x0fU) ^ 0x08U) - 8);
        }
    }
    const int64_t q4_elapsed_us = esp_timer_get_time() - q4_start_us;
    const float q4_result = (float)q4_sum * 0.05f;
    ESP_LOGI("quant", "INT8: %" PRIu64 " MAC/s, checksum %" PRId32 "; Q8(scale): %" PRIu64
             " MAC/s, abs error %.5f; Q4: %" PRIu64 " MAC/s, abs error %.5f",
             (uint64_t)128U * 10000U * 1000000ULL / (uint64_t)int8_elapsed_us, int8_sum,
             (uint64_t)128U * 10000U * 1000000ULL / (uint64_t)int8_elapsed_us, (double)fabsf(q8_result - reference),
             (uint64_t)128U * 10000U * 1000000ULL / (uint64_t)q4_elapsed_us, (double)fabsf(q4_result - reference));
}

static void benchmark_kv_attention(void)
{
    const uint32_t lengths[] = {128U, 256U, 512U, 1024U};
    int8_t query[144] __attribute__((aligned(16)));
    for (uint32_t index = 0; index < 128U; ++index) query[index] = (int8_t)((index % 11U) - 5);
    for (uint32_t length_index = 0; length_index < sizeof(lengths) / sizeof(lengths[0]); ++length_index) {
        const uint32_t length = lengths[length_index];
        int8_t *kv = heap_caps_malloc(length * 256U, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        float *scores = heap_caps_malloc(length * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (kv == NULL || scores == NULL) {
            ESP_LOGE("kv-attn", "allocation failed for context %" PRIu32, length);
            heap_caps_free(kv); heap_caps_free(scores); continue;
        }
        for (uint32_t index = 0; index < length * 256U; ++index) kv[index] = (int8_t)((index % 13U) - 6);
        const int64_t start_us = esp_timer_get_time();
        float checksum = 0.0f;
        for (uint32_t token = 0; token < length; ++token) scores[token] = (float)q8_dot(query, &kv[token * 256U], 128U) * 0.01f;
        float max_score = scores[0];
        for (uint32_t token = 1; token < length; ++token) if (scores[token] > max_score) max_score = scores[token];
        float sum = 0.0f;
        for (uint32_t token = 0; token < length; ++token) { scores[token] = expf(scores[token] - max_score); sum += scores[token]; }
        for (uint32_t dimension = 0; dimension < 128U; ++dimension) {
            float value = 0.0f;
            for (uint32_t token = 0; token < length; ++token) value += scores[token] / sum * kv[token * 256U + 128U + dimension];
            checksum += value;
        }
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        ESP_LOGI("kv-attn", "context %" PRIu32 ": %" PRId64 " us/layer, checksum: %.4f", length, elapsed_us, (double)checksum);
        heap_caps_free(kv); heap_caps_free(scores);
    }
}

static int32_t output_head_max(const int8_t *input, const int8_t *weights, uint32_t rows)
{
    int32_t maximum = INT32_MIN;
    for (uint32_t row = 0; row < rows; ++row) {
        const int32_t logit = q8_dot(input, &weights[row * 128U], 128U);
        if (logit > maximum) maximum = logit;
    }
    return maximum;
}

IRAM_ATTR static uint32_t iram_mix(uint32_t iterations)
{
    uint32_t value = 0x12345678;
    for (uint32_t index = 0; index < iterations; ++index) {
        value = value * 1664525U + 1013904223U;
        value = ((value << 13U) | (value >> 19U)) ^ index;
    }
    return value;
}

typedef struct {
    uint32_t checksum;
    uint32_t iterations;
    BaseType_t core_id;
    int64_t start_us;
    int64_t end_us;
} cpu_worker_result_t;

static void run_cpu_worker(void *argument)
{
    cpu_worker_result_t *result = argument;

    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    result->core_id = xPortGetCoreID();
    result->start_us = esp_timer_get_time();
    result->checksum = iram_mix(result->iterations);
    result->end_us = esp_timer_get_time();
    xTaskNotifyGive(benchmark_main_task);
    vTaskDelete(NULL);
}
#endif

void app_main(void)
{
    run_llmm_usb_inference();
#if LLMM_DEBUG
    return;
    const uint32_t iterations = 10000;
    const float pi = zig_calculate_pi(iterations);

    ESP_LOGI("zig-pi", "Nilakantha iterations: %" PRIu32, iterations);
    ESP_LOGI("zig-pi", "pi = %.7f", (double)pi);

    cpu_worker_result_t debug_core0_result = {0};
    cpu_worker_result_t debug_core1_result = {0};
    TaskHandle_t debug_core0_task;
    TaskHandle_t debug_core1_task;
    debug_core0_result.iterations = DUAL_DEBUG_ITERATIONS;
    debug_core1_result.iterations = DUAL_DEBUG_ITERATIONS;
    benchmark_main_task = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(run_cpu_worker, "bench_core0", 2048, &debug_core0_result, 5, &debug_core0_task, 0);
    xTaskCreatePinnedToCore(run_cpu_worker, "bench_core1", 2048, &debug_core1_result, 5, &debug_core1_task, 1);
    const int64_t debug_dual_start_us = esp_timer_get_time();
    xTaskNotifyGive(debug_core1_task);
    xTaskNotifyGive(debug_core0_task);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    const int64_t debug_dual_elapsed_us = esp_timer_get_time() - debug_dual_start_us;
    const uint64_t debug_dual_throughput =
        ((uint64_t)DUAL_DEBUG_ITERATIONS * 2ULL * 1000000ULL) / (uint64_t)debug_dual_elapsed_us;
    ESP_LOGI("dual-debug", "throughput: %" PRIu64 " iterations/s, elapsed: %" PRId64 " us",
             debug_dual_throughput, debug_dual_elapsed_us);
    ESP_LOGI("dual-debug", "cores: %" PRId32 ", %" PRId32 "; durations: %" PRId64 " us, %" PRId64 " us; starts delta: %" PRId64 " us",
             (int32_t)debug_core0_result.core_id, (int32_t)debug_core1_result.core_id,
             debug_core0_result.end_us - debug_core0_result.start_us, debug_core1_result.end_us - debug_core1_result.start_us,
             debug_core1_result.start_us - debug_core0_result.start_us);

    for (uint32_t index = 0; index < GEMV_K; ++index) {
        gemv_input[index] = (int8_t)((index % 15U) - 7);
    }
    for (uint32_t index = 0; index < sizeof(gemv_q4_weights); ++index) {
        gemv_q4_weights[index] = (uint8_t)(index * 37U + 11U);
    }
    const int64_t gemv_single_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < GEMV_REPETITIONS; ++repeat) {
        gemv_q4_range(0, GEMV_N, gemv_unpacked);
    }
    const int64_t gemv_single_elapsed_us = esp_timer_get_time() - gemv_single_start_us;
    ESP_LOGI("q4-xespv-gemv", "single: %" PRIu64 " MAC/s, checksum: %" PRId32,
             ((uint64_t)GEMV_N * GEMV_K * GEMV_REPETITIONS * 1000000ULL) / (uint64_t)gemv_single_elapsed_us,
             gemv_output[GEMV_N - 1U]);

    gemv_worker_t gemv_core0 = {.first = 0, .last = GEMV_N / 2U, .repetitions = GEMV_REPETITIONS, .unpacked = gemv_unpacked};
    gemv_worker_t gemv_core1 = {.first = GEMV_N / 2U, .last = GEMV_N, .repetitions = GEMV_REPETITIONS, .unpacked = gemv_unpacked_core1};
    TaskHandle_t gemv_core0_task;
    TaskHandle_t gemv_core1_task;
    benchmark_main_task = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(run_gemv_worker, "gemv_core0", 4096, &gemv_core0, 5, &gemv_core0_task, 0);
    xTaskCreatePinnedToCore(run_gemv_worker, "gemv_core1", 4096, &gemv_core1, 5, &gemv_core1_task, 1);
    const int64_t gemv_dual_start_us = esp_timer_get_time();
    xTaskNotifyGive(gemv_core1_task);
    xTaskNotifyGive(gemv_core0_task);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    const int64_t gemv_dual_elapsed_us = esp_timer_get_time() - gemv_dual_start_us;
    ESP_LOGI("q4-xespv-gemv", "dual: %" PRIu64 " MAC/s, starts delta: %" PRId64 " us, checksum: %" PRId32,
             ((uint64_t)GEMV_N * GEMV_K * GEMV_REPETITIONS * 1000000ULL) / (uint64_t)gemv_dual_elapsed_us,
             gemv_core1.start_us - gemv_core0.start_us, gemv_output[GEMV_N - 1U]);
    const uint32_t benchmark_iterations = CPU_ITERATIONS;
    const int64_t start_us = esp_timer_get_time();
    const uint32_t checksum = zig_cpu_benchmark(benchmark_iterations);
    const int64_t elapsed_us = esp_timer_get_time() - start_us;
    const uint64_t iterations_per_second =
        ((uint64_t)benchmark_iterations * 1000000ULL) / (uint64_t)elapsed_us;

    ESP_LOGI("zig-cpu", "iterations: %" PRIu32, benchmark_iterations);
    ESP_LOGI("zig-cpu", "elapsed: %" PRId64 " us", elapsed_us);
    ESP_LOGI("zig-cpu", "throughput: %" PRIu64 " iterations/s", iterations_per_second);
    ESP_LOGI("zig-cpu", "checksum: 0x%08" PRIx32, checksum);

    const int64_t divide_start_us = esp_timer_get_time();
    const uint32_t divide_checksum = zig_divide_benchmark(DIVIDE_ITERATIONS);
    const int64_t divide_elapsed_us = esp_timer_get_time() - divide_start_us;
    ESP_LOGI("zig-div", "throughput: %" PRIu64 " divisions/s, checksum: 0x%08" PRIx32,
             ((uint64_t)DIVIDE_ITERATIONS * 1000000ULL) / (uint64_t)divide_elapsed_us,
             divide_checksum);

    const int64_t float_start_us = esp_timer_get_time();
    const float float_checksum = zig_float_benchmark(FLOAT_ITERATIONS);
    const int64_t float_elapsed_us = esp_timer_get_time() - float_start_us;
    ESP_LOGI("zig-f32", "throughput: %" PRIu64 " iterations/s, checksum: %.7f",
             ((uint64_t)FLOAT_ITERATIONS * 1000000ULL) / (uint64_t)float_elapsed_us,
             (double)float_checksum);

    const int64_t scalar_dot_start_us = esp_timer_get_time();
    const int32_t scalar_dot_checksum = scalar_dot8(XESPV_ITERATIONS);
    const int64_t scalar_dot_elapsed_us = esp_timer_get_time() - scalar_dot_start_us;
    const int64_t xespv_dot_start_us = esp_timer_get_time();
    const int32_t xespv_dot_checksum = xespv_dot8(XESPV_ITERATIONS);
    const int64_t xespv_dot_elapsed_us = esp_timer_get_time() - xespv_dot_start_us;
    ESP_LOGI("xespv", "scalar: %" PRIu64 " MAC/s, checksum: %" PRId32,
             ((uint64_t)XESPV_ITERATIONS * 8ULL * 1000000ULL) / (uint64_t)scalar_dot_elapsed_us,
             scalar_dot_checksum);
    ESP_LOGI("xespv", "vector: %" PRIu64 " MAC/s, checksum: %" PRId32,
             ((uint64_t)XESPV_ITERATIONS * 8ULL * 1000000ULL) / (uint64_t)xespv_dot_elapsed_us,
             xespv_dot_checksum);
    const int64_t xespv_i8_start_us = esp_timer_get_time();
    const int32_t xespv_i8_checksum = xespv_dot16_i8(XESPV_ITERATIONS);
    const int64_t xespv_i8_elapsed_us = esp_timer_get_time() - xespv_i8_start_us;
    ESP_LOGI("xespv-i8", "vector: %" PRIu64 " MAC/s, checksum: %" PRId32,
             ((uint64_t)XESPV_ITERATIONS * 16ULL * 1000000ULL) / (uint64_t)xespv_i8_elapsed_us,
             xespv_i8_checksum);

    for (uint32_t index = 0; index < Q4_ELEMENTS; ++index) {
        q4_activations[index] = (int8_t)((index % 15U) - 7);
        if ((index & 1U) == 0) {
            q4_weights[index / 2U] = (uint8_t)(index % 16U);
        } else {
            q4_weights[index / 2U] |= (uint8_t)((index % 16U) << 4U);
        }
    }
    const int64_t q4_start_us = esp_timer_get_time();
    int32_t q4_checksum = 0;
    for (uint32_t repeat = 0; repeat < 1000U; ++repeat) {
        q4_checksum = q4_dot_product();
    }
    const int64_t q4_elapsed_us = esp_timer_get_time() - q4_start_us;
    ESP_LOGI("q4-dot", "throughput: %" PRIu64 " elements/s, checksum: %" PRId32,
             ((uint64_t)Q4_ELEMENTS * 1000ULL * 1000000ULL) / (uint64_t)q4_elapsed_us, q4_checksum);

    for (uint32_t index = 0; index < MATRIX_SIZE * MATRIX_SIZE; ++index) {
        matrix_a[index] = (int8_t)((index % 7U) - 3);
        matrix_b[index] = (int8_t)((index % 11U) - 5);
    }
    const int64_t matrix_start_us = esp_timer_get_time();
    for (uint32_t repeat = 0; repeat < MATRIX_REPETITIONS; ++repeat) {
        matrix_multiply_i8();
    }
    const int64_t matrix_elapsed_us = esp_timer_get_time() - matrix_start_us;
    ESP_LOGI("matmul", "throughput: %" PRIu64 " MAC/s, checksum: %" PRId32,
             ((uint64_t)MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE * MATRIX_REPETITIONS * 1000000ULL) /
                 (uint64_t)matrix_elapsed_us,
             matrix_c[MATRIX_SIZE * MATRIX_SIZE - 1U]);

    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] = (float)(index % 13U) * 0.01f;
    }
    const int64_t rms_start_us = esp_timer_get_time();
    float rms_checksum = 0.0f;
    for (uint32_t repeat = 0; repeat < VECTOR_REPETITIONS; ++repeat) {
        rms_checksum = rms_norm();
    }
    const int64_t rms_elapsed_us = esp_timer_get_time() - rms_start_us;
    ESP_LOGI("rmsnorm", "throughput: %" PRIu64 " vectors/s, checksum: %.6f",
             ((uint64_t)VECTOR_REPETITIONS * 1000000ULL) / (uint64_t)rms_elapsed_us, (double)rms_checksum);

    const int64_t softmax_start_us = esp_timer_get_time();
    float softmax_checksum = 0.0f;
    for (uint32_t repeat = 0; repeat < VECTOR_REPETITIONS; ++repeat) {
        softmax_checksum = softmax();
    }
    const int64_t softmax_elapsed_us = esp_timer_get_time() - softmax_start_us;
    ESP_LOGI("softmax", "throughput: %" PRIu64 " vectors/s, checksum: %.6f",
             ((uint64_t)VECTOR_REPETITIONS * 1000000ULL) / (uint64_t)softmax_elapsed_us, (double)softmax_checksum);

    const int64_t rope_start_us = esp_timer_get_time();
    float rope_checksum = 0.0f;
    for (uint32_t repeat = 0; repeat < VECTOR_REPETITIONS; ++repeat) {
        rope_checksum = rope();
    }
    const int64_t rope_elapsed_us = esp_timer_get_time() - rope_start_us;
    ESP_LOGI("rope", "throughput: %" PRIu64 " vectors/s, checksum: %.6f",
             ((uint64_t)VECTOR_REPETITIONS * 1000000ULL) / (uint64_t)rope_elapsed_us, (double)rope_checksum);

    for (uint32_t index = 0; index < VECTOR_SIZE; ++index) {
        vector_data[index] = (float)((int32_t)(index % 17U) - 8) * 0.05f;
        residual[index] = vector_data[index];
    }
    const int64_t token_chain_start_us = esp_timer_get_time();
    uint32_t token_chain_checksum = 0;
    for (uint32_t repeat = 0; repeat < END_TO_END_REPETITIONS; ++repeat) {
        token_chain_checksum = run_token_operator_chain();
    }
    const int64_t token_chain_elapsed_us = esp_timer_get_time() - token_chain_start_us;
    ESP_LOGI("token-chain", "rmsnorm->q4-gemv->silu->q4-gemv->residual->softmax: %" PRIu64
             " us/token, %" PRIu64 " tokens/s, checksum: 0x%08" PRIx32,
             (uint64_t)token_chain_elapsed_us / END_TO_END_REPETITIONS,
             ((uint64_t)END_TO_END_REPETITIONS * 1000000ULL) / (uint64_t)token_chain_elapsed_us,
             token_chain_checksum);

    benchmark_quant_formats();
    benchmark_gemv_shapes();
    benchmark_ternary_xespv();
    benchmark_rans();
    benchmark_kv_attention();

    const esp_partition_t *factory = esp_partition_find_first(ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_APP_FACTORY, NULL);
    uint8_t flash_buffer[512];
    uint32_t flash_checksum = 0;
    const int64_t flash_sequential_start_us = esp_timer_get_time();
    for (uint32_t offset = 0; offset < 256U * 1024U; offset += sizeof(flash_buffer)) {
        esp_partition_read(factory, offset, flash_buffer, sizeof(flash_buffer));
        flash_checksum += flash_buffer[0];
    }
    const int64_t flash_sequential_elapsed_us = esp_timer_get_time() - flash_sequential_start_us;
    uint32_t random_state = 0xdeadbeef;
    const int64_t flash_random_start_us = esp_timer_get_time();
    for (uint32_t access = 0; access < PSRAM_RANDOM_ACCESSES; ++access) {
        random_state = random_state * 1664525U + 1013904223U;
        const uint32_t offset = (random_state % (factory->size - 32U)) & ~31U;
        esp_partition_read(factory, offset, flash_buffer, 32U);
        flash_checksum += flash_buffer[0];
    }
    const int64_t flash_random_elapsed_us = esp_timer_get_time() - flash_random_start_us;
    ESP_LOGI("flash", "sequential: %" PRIu64 " MiB/s, random-32B: %" PRIu64 " us/read, checksum: 0x%08" PRIx32,
             ((uint64_t)256U * 1000000ULL) / ((uint64_t)flash_sequential_elapsed_us * 1024ULL),
             (uint64_t)flash_random_elapsed_us / PSRAM_RANDOM_ACCESSES, flash_checksum);

    const void *flash_mmap;
    esp_partition_mmap_handle_t flash_mmap_handle;
    if (esp_partition_mmap(factory, 0, 256U * 1024U, ESP_PARTITION_MMAP_DATA,
                           &flash_mmap, &flash_mmap_handle) == ESP_OK) {
        const volatile uint8_t *mapped = flash_mmap;
        uint32_t mmap_checksum = 0;
        const int64_t mmap_sequential_start_us = esp_timer_get_time();
        for (uint32_t offset = 0; offset < 256U * 1024U; offset += 32U) {
            mmap_checksum += mapped[offset];
        }
        const int64_t mmap_sequential_elapsed_us = esp_timer_get_time() - mmap_sequential_start_us;
        random_state = 0x1234abcd;
        const int64_t mmap_random_start_us = esp_timer_get_time();
        for (uint32_t access = 0; access < PSRAM_RANDOM_ACCESSES; ++access) {
            random_state = random_state * 1664525U + 1013904223U;
            mmap_checksum += mapped[(random_state % (256U * 1024U - 32U)) & ~31U];
        }
        const int64_t mmap_random_elapsed_us = esp_timer_get_time() - mmap_random_start_us;
        ESP_LOGI("flash-xip", "sequential: %" PRIu64 " MiB/s, random-32B: %" PRIu64 " ns/read, checksum: 0x%08" PRIx32,
                 ((uint64_t)256U * 1000000ULL) / ((uint64_t)mmap_sequential_elapsed_us * 1024ULL),
                 ((uint64_t)mmap_random_elapsed_us * 1000ULL) / PSRAM_RANDOM_ACCESSES, mmap_checksum);
        esp_partition_munmap(flash_mmap_handle);
    } else {
        ESP_LOGE("flash-xip", "partition mmap failed");
    }

    int8_t output_input[144] __attribute__((aligned(16)));
    for (uint32_t index = 0; index < 128U; ++index) output_input[index] = (int8_t)((index % 15U) - 7);
    int8_t *output_psram = heap_caps_malloc(32768U * 128U, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *output_sram = heap_caps_malloc(1024U * 128U, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (output_psram != NULL) {
        for (uint32_t index = 0; index < 32768U * 128U; ++index) output_psram[index] = (int8_t)((index % 17U) - 8);
        const int64_t output_psram_start_us = esp_timer_get_time();
        const int32_t output_psram_max = output_head_max(output_input, output_psram, 32768U);
        const int64_t output_psram_elapsed_us = esp_timer_get_time() - output_psram_start_us;
        ESP_LOGI("output-head", "int8 32768x128 PSRAM: %" PRId64 " us, max: %" PRId32,
                 output_psram_elapsed_us, output_psram_max);
    } else {
        ESP_LOGE("output-head", "PSRAM allocation failed for 32768x128");
    }
    if (output_sram != NULL) {
        for (uint32_t index = 0; index < 1024U * 128U; ++index) output_sram[index] = (int8_t)((index % 17U) - 8);
        const int64_t output_sram_start_us = esp_timer_get_time();
        const int32_t output_sram_max = output_head_max(output_input, output_sram, 1024U);
        const int64_t output_sram_elapsed_us = esp_timer_get_time() - output_sram_start_us;
        ESP_LOGI("output-head", "int8 1024x128 SRAM: %" PRId64 " us, max: %" PRId32,
                 output_sram_elapsed_us, output_sram_max);
    } else {
        ESP_LOGE("output-head", "SRAM allocation failed for 1024x128");
    }
    const void *head_xip;
    esp_partition_mmap_handle_t head_xip_handle;
    if (esp_partition_mmap(factory, 0, 1024U * 128U, ESP_PARTITION_MMAP_DATA, &head_xip, &head_xip_handle) == ESP_OK) {
        const int64_t output_xip_start_us = esp_timer_get_time();
        const int32_t output_xip_max = output_head_max(output_input, head_xip, 1024U);
        const int64_t output_xip_elapsed_us = esp_timer_get_time() - output_xip_start_us;
        ESP_LOGI("output-head", "int8 1024x128 Flash XIP: %" PRId64 " us, max: %" PRId32,
                 output_xip_elapsed_us, output_xip_max);
        esp_partition_munmap(head_xip_handle);
    }
    heap_caps_free(output_psram);
    heap_caps_free(output_sram);

    volatile uint8_t *psram = heap_caps_malloc(PSRAM_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    for (uint32_t index = 0; index < PSRAM_BYTES; ++index) {
        psram[index] = (uint8_t)index;
    }
    uint32_t psram_checksum = 0;
    const int64_t psram_sequential_start_us = esp_timer_get_time();
    for (uint32_t offset = 0; offset < PSRAM_BYTES; offset += PSRAM_STRIDE) {
        psram_checksum += psram[offset];
    }
    const int64_t psram_sequential_elapsed_us = esp_timer_get_time() - psram_sequential_start_us;
    random_state = 0x31415926;
    const int64_t psram_random_start_us = esp_timer_get_time();
    for (uint32_t access = 0; access < PSRAM_RANDOM_ACCESSES; ++access) {
        random_state = random_state * 1664525U + 1013904223U;
        psram_checksum += psram[random_state % PSRAM_BYTES];
    }
    const int64_t psram_random_elapsed_us = esp_timer_get_time() - psram_random_start_us;
    ESP_LOGI("psram", "sequential: %" PRIu64 " MiB/s, random: %" PRIu64 " ns/read, checksum: 0x%08" PRIx32,
             ((uint64_t)PSRAM_BYTES * 1000000ULL) / ((uint64_t)psram_sequential_elapsed_us * 1024ULL * 1024ULL),
             ((uint64_t)psram_random_elapsed_us * 1000ULL) / PSRAM_RANDOM_ACCESSES, psram_checksum);

    uint8_t *stage0 = heap_caps_malloc(16U * 1024U, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    uint8_t *stage1 = heap_caps_malloc(16U * 1024U, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    const uint32_t prefetch_sizes[] = {1024U, 2048U, 4096U, 8192U, 16384U};
    if (stage0 != NULL && stage1 != NULL) {
        for (uint32_t size_index = 0; size_index < sizeof(prefetch_sizes) / sizeof(prefetch_sizes[0]); ++size_index) {
            const uint32_t block_bytes = prefetch_sizes[size_index];
            uint32_t block_checksum = 0;
            const int64_t block_start_us = esp_timer_get_time();
            for (uint32_t pass = 0; pass < PREFETCH_PASSES; ++pass) {
                const uint32_t offset = (pass * block_bytes) % (PSRAM_BYTES - block_bytes);
                memcpy(stage0, (const uint8_t *)psram + offset, block_bytes);
                block_checksum += stage0[(pass * 37U) % block_bytes];
            }
            const int64_t block_elapsed_us = esp_timer_get_time() - block_start_us;
            ESP_LOGI("psram-stage", "%" PRIu32 " KiB: %" PRIu64 " MiB/s, checksum: 0x%08" PRIx32,
                     block_bytes / 1024U,
                     ((uint64_t)block_bytes * PREFETCH_PASSES * 1000000ULL) /
                         ((uint64_t)block_elapsed_us * 1024ULL * 1024ULL), block_checksum);
        }

        memcpy((uint8_t *)psram, gemv_q4_weights, sizeof(gemv_q4_weights));
        gemv_q4_range(0, GEMV_N, gemv_unpacked);
        const uint32_t direct_gemv_checksum = gemv_output_checksum();
        const int64_t staged_gemv_start_us = esp_timer_get_time();
        uint32_t staged_gemv_checksum = 0;
        for (uint32_t repeat = 0; repeat < GEMV_REPETITIONS; ++repeat) {
            staged_gemv_checksum = gemv_q4_staged((const uint8_t *)psram, stage0, stage1, 4096U, gemv_unpacked);
        }
        const int64_t staged_gemv_elapsed_us = esp_timer_get_time() - staged_gemv_start_us;
        ESP_LOGI("psram-gemv", "4 KiB CPU ping-pong (no DMA overlap): %" PRIu64
                 " MAC/s, staged: 0x%08" PRIx32 ", direct: 0x%08" PRIx32,
                 ((uint64_t)GEMV_N * GEMV_K * GEMV_REPETITIONS * 1000000ULL) / (uint64_t)staged_gemv_elapsed_us,
                 staged_gemv_checksum, direct_gemv_checksum);

        async_memcpy_handle_t dma;
        const async_memcpy_config_t dma_config = ASYNC_MEMCPY_DEFAULT_CONFIG();
        if (esp_async_memcpy_install(&dma_config, &dma) == ESP_OK) {
            const int64_t dma_gemv_start_us = esp_timer_get_time();
            uint32_t dma_gemv_checksum = 0;
            for (uint32_t repeat = 0; repeat < GEMV_REPETITIONS; ++repeat) {
                dma_gemv_checksum = gemv_q4_dma_pipelined(dma, (uint8_t *)psram, stage0, stage1,
                                                           4096U, gemv_unpacked);
                if (dma_gemv_checksum == 0) break;
            }
            const int64_t dma_gemv_elapsed_us = esp_timer_get_time() - dma_gemv_start_us;
            if (dma_gemv_checksum == 0) {
                ESP_LOGE("psram-gemv", "4 KiB GDMA transfer failed or timed out");
            } else {
                ESP_LOGI("psram-gemv", "4 KiB GDMA overlap: %" PRIu64
                         " MAC/s, staged: 0x%08" PRIx32 ", direct: 0x%08" PRIx32,
                         ((uint64_t)GEMV_N * GEMV_K * GEMV_REPETITIONS * 1000000ULL) / (uint64_t)dma_gemv_elapsed_us,
                         dma_gemv_checksum, direct_gemv_checksum);
            }
            esp_async_memcpy_uninstall(dma);
        } else {
            ESP_LOGE("psram-gemv", "GDMA async memcpy install failed");
        }
    } else {
        ESP_LOGE("psram-stage", "internal SRAM stage allocation failed");
    }
    heap_caps_free(stage0);
    heap_caps_free(stage1);

    for (uint32_t index = 0; index < MEMORY_BYTES; ++index) {
        memory_source[index] = (uint8_t)index;
    }
    const int64_t memory_start_us = esp_timer_get_time();
    for (uint32_t pass = 0; pass < MEMORY_PASSES; ++pass) {
        for (uint32_t index = 0; index < MEMORY_BYTES; ++index) {
            memory_destination[index] = memory_source[index];
        }
    }
    const int64_t memory_elapsed_us = esp_timer_get_time() - memory_start_us;
    const uint32_t memory_checksum = memory_destination[0] + memory_destination[MEMORY_BYTES - 1];
    const uint64_t memory_mib_per_second =
        ((uint64_t)MEMORY_BYTES * MEMORY_PASSES * 1000000ULL) /
        ((uint64_t)memory_elapsed_us * 1024ULL * 1024ULL);
    ESP_LOGI("memory", "copy: %" PRIu64 " MiB/s, checksum: 0x%08" PRIx32,
             memory_mib_per_second, memory_checksum);

    cpu_worker_result_t core0_result = {0};
    cpu_worker_result_t core1_result = {0};
    TaskHandle_t core0_task;
    TaskHandle_t core1_task;
    core0_result.iterations = CPU_ITERATIONS;
    core1_result.iterations = CPU_ITERATIONS;
    const int64_t iram_single_start_us = esp_timer_get_time();
    const uint32_t iram_single_checksum = iram_mix(CPU_ITERATIONS);
    const int64_t iram_single_elapsed_us = esp_timer_get_time() - iram_single_start_us;
    const uint64_t iram_single_throughput =
        ((uint64_t)CPU_ITERATIONS * 1000000ULL) / (uint64_t)iram_single_elapsed_us;
    ESP_LOGI("iram", "single-core: %" PRIu64 " iterations/s, checksum: 0x%08" PRIx32,
             iram_single_throughput, iram_single_checksum);
    benchmark_main_task = xTaskGetCurrentTaskHandle();
    xTaskCreatePinnedToCore(run_cpu_worker, "bench_core0", 2048, &core0_result, 5, &core0_task, 0);
    xTaskCreatePinnedToCore(run_cpu_worker, "bench_core1", 2048, &core1_result, 5, &core1_task, 1);
    const int64_t dual_start_us = esp_timer_get_time();
    xTaskNotifyGive(core1_task);
    xTaskNotifyGive(core0_task);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
    const int64_t dual_elapsed_us = esp_timer_get_time() - dual_start_us;
    const uint64_t dual_throughput =
        ((uint64_t)CPU_ITERATIONS * 2ULL * 1000000ULL) / (uint64_t)dual_elapsed_us;
    ESP_LOGI("dual-core-iram", "throughput: %" PRIu64 " iterations/s, cores: %" PRId32 ", %" PRId32 ", checksums: 0x%08" PRIx32 ", 0x%08" PRIx32,
             dual_throughput, (int32_t)core0_result.core_id, (int32_t)core1_result.core_id,
             core0_result.checksum, core1_result.checksum);
    ESP_LOGI("dual-core-iram", "durations: %" PRId64 " us, %" PRId64 " us; starts delta: %" PRId64 " us",
             core0_result.end_us - core0_result.start_us, core1_result.end_us - core1_result.start_us,
             core1_result.start_us - core0_result.start_us);
#endif
}
