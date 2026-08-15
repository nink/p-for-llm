#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wire may carry a long raw ChatML (or CONTEXT/QUESTION) prompt; on-device
 * extractive compress shrinks it into out_cap before tokenization. */
#ifndef LLMM_RAW_MAX_BYTES
#define LLMM_RAW_MAX_BYTES 49152U
#endif
/* Prompts that already fit the native window skip extractive trim. */
#ifndef LLMM_COMPRESS_BYPASS_BYTES
#define LLMM_COMPRESS_BYPASS_BYTES 1024U
#endif
/* Prefill-speed cap: smaller fitted packet >> 8:1 fill-to-1024. */
#ifndef LLMM_COMPRESS_FITTED_MAX_BYTES
#define LLMM_COMPRESS_FITTED_MAX_BYTES 400U
#endif
#ifndef LLMM_COMPRESS_MAX_TOKENS
#define LLMM_COMPRESS_MAX_TOKENS 80U
#endif

typedef struct {
    uint32_t source_bytes;
    uint32_t packet_bytes;
    uint32_t kept_sentences;
    uint32_t dropped_sentences;
    uint32_t compressed; /* 1 if trim ran, 0 if passthrough */
} llmm_compress_stats_t;

/* Fit ``in`` into ``out`` (must be <= out_cap). If already small, copies.
 * Recognizes ChatML user blocks and CONTEXT:/QUESTION: packets.
 * Returns 0 on success, negative on error. */
int llmm_compress_prompt_to_fit(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_cap,
                                size_t *out_len, llmm_compress_stats_t *stats);

#ifdef __cplusplus
}
#endif
