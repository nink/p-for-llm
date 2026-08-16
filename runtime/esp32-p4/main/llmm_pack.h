#pragma once

#include <stddef.h>
#include <stdio.h>

enum { LLMM_PACK_Q_MAX = 512, LLMM_PACK_A_MAX = 768 };

int llmm_pack_answer(FILE *fp, const char *question, char *out, size_t out_bytes);
/* Load a PSRAM hash index of canada.kpack. 0 = RAM lookups, -1 = no memory (SD fallback). */
int llmm_pack_index_build(FILE *fp);
void llmm_pack_index_reset(void);
