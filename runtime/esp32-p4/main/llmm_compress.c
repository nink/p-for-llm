#include "llmm_compress.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"

#define LLMM_COMPRESS_MAX_SENTS 256U
#define LLMM_COMPRESS_TARGET_RATIO 8.0f

typedef struct {
    uint16_t start;
    uint16_t len;
    float score;
} llmm_sent_t;

static int llmm_is_word_char(unsigned char c)
{
    return (unsigned)((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'');
}

static size_t llmm_estimate_tokens(const char *text, size_t len)
{
    size_t words = 0;
    int in_word = 0;
    for (size_t i = 0; i < len; ++i) {
        const int w = llmm_is_word_char((unsigned char)text[i]);
        if (w && !in_word) {
            ++words;
            in_word = 1;
        } else if (!w) {
            in_word = 0;
        }
    }
    if (words == 0) return len > 0 ? 1U : 0U;
    return (words * 4U + 2U) / 3U; /* ~ words / 0.75 */
}

static int llmm_is_stop_term(const char *term, size_t len)
{
    static const char *const stops[] = {
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "do", "does", "did", "can", "could", "would", "should",
        "to", "of", "in", "on", "for", "and", "or", "with", "from", "about", "this", "that",
        "these", "those", "it", "its", "be", "as", "at", "by",
    };
    char buf[24];
    if (len == 0 || len >= sizeof(buf)) return 0;
    for (size_t i = 0; i < len; ++i) buf[i] = (char)tolower((unsigned char)term[i]);
    buf[len] = '\0';
    for (size_t i = 0; i < sizeof(stops) / sizeof(stops[0]); ++i) {
        if (strcmp(buf, stops[i]) == 0) return 1;
    }
    return 0;
}

static size_t llmm_collect_question_terms(const char *question, size_t qlen, char terms[][24], size_t max_terms)
{
    size_t count = 0;
    size_t i = 0;
    while (i < qlen && count < max_terms) {
        while (i < qlen && !llmm_is_word_char((unsigned char)question[i])) ++i;
        size_t start = i;
        while (i < qlen && llmm_is_word_char((unsigned char)question[i])) ++i;
        size_t len = i - start;
        if (len > 2 && !llmm_is_stop_term(question + start, len) && len < 24) {
            for (size_t k = 0; k < len; ++k) terms[count][k] = (char)tolower((unsigned char)question[start + k]);
            terms[count][len] = '\0';
            ++count;
        }
    }
    return count;
}

static int llmm_has_term(const char *hay, size_t hay_len, const char *term)
{
    const size_t tlen = strlen(term);
    if (tlen == 0 || tlen > hay_len) return 0;
    for (size_t i = 0; i + tlen <= hay_len; ++i) {
        size_t k = 0;
        for (; k < tlen; ++k) {
            if ((char)tolower((unsigned char)hay[i + k]) != term[k]) break;
        }
        if (k == tlen) return 1;
    }
    return 0;
}

static float llmm_score_sentence(const char *s, size_t len, char terms[][24], size_t term_count)
{
    float score = 1.0f;
    int has_digit = 0;
    int nameish = 0;
    for (size_t i = 0; i < len; ++i) {
        if (isdigit((unsigned char)s[i])) has_digit = 1;
        if (i + 3 < len && s[i] >= 'A' && s[i] <= 'Z' && s[i + 1] >= 'a' && s[i + 1] <= 'z') nameish = 1;
    }
    if (len >= 3 && s[0] == '#' && s[1] == '#') score -= 4.0f; /* "## Section N" padding */
    else if ((len > 0 && s[0] == '#') || (len > 0 && len < 80 && s[len - 1] == ':')) score += 3.0f;
    if (has_digit) score += 2.0f;
    if (nameish) score += 1.0f;
    size_t hits = 0;
    for (size_t t = 0; t < term_count; ++t) {
        if (llmm_has_term(s, len, terms[t])) {
            score += 4.0f;
            ++hits;
        }
    }
    if (hits >= 2) score += 4.0f;
    if (len >= 40 && len <= 280) score += 1.0f;
    else if (len < 20) score -= 1.0f;
    else if (len > 500) score -= 2.0f;
    if (llmm_has_term(s, len, "as mentioned") || llmm_has_term(s, len, "in conclusion") ||
        llmm_has_term(s, len, "click here")) {
        score -= 2.0f;
    }
    return score;
}

static size_t llmm_split_sentences(const char *text, size_t len, llmm_sent_t *out, size_t max_out)
{
    size_t count = 0;
    size_t start = 0;
    while (start < len && count < max_out) {
        while (start < len && isspace((unsigned char)text[start])) ++start;
        if (start >= len) break;
        size_t end = start;
        while (end < len) {
            const char c = text[end];
            if (c == '\n') break;
            if ((c == '.' || c == '!' || c == '?') && (end + 1 >= len || isspace((unsigned char)text[end + 1]))) {
                ++end;
                break;
            }
            ++end;
        }
        size_t s = start;
        size_t e = end;
        while (s < e && isspace((unsigned char)text[s])) ++s;
        while (e > s && isspace((unsigned char)text[e - 1])) --e;
        if (e > s && (e - s) >= 8) {
            int dup = 0;
            for (size_t k = 0; k < count; ++k) {
                if (out[k].len == (uint16_t)(e - s) &&
                    memcmp(text + out[k].start, text + s, e - s) == 0) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                out[count].start = (uint16_t)s;
                out[count].len = (uint16_t)(e - s);
                out[count].score = 0.0f;
                ++count;
            }
        }
        if (end < len && text[end] == '\n') {
            start = end + 1;
        } else {
            start = end > start ? end : start + 1;
        }
    }
    return count;
}

static int llmm_find_marker(const char *hay, size_t hay_len, const char *needle, size_t *pos)
{
    const size_t nlen = strlen(needle);
    if (nlen == 0 || nlen > hay_len) return 0;
    for (size_t i = 0; i + nlen <= hay_len; ++i) {
        if (memcmp(hay + i, needle, nlen) == 0) {
            *pos = i;
            return 1;
        }
    }
    return 0;
}

static int llmm_extract_user_body(const char *in, size_t in_len, size_t *body_off, size_t *body_len,
                                  size_t *prefix_len, size_t *suffix_off)
{
    size_t user_mark = 0;
    if (llmm_find_marker(in, in_len, "<|im_start|>user\n", &user_mark)) {
        const size_t content_off = user_mark + strlen("<|im_start|>user\n");
        size_t end_mark = 0;
        if (!llmm_find_marker(in + content_off, in_len - content_off, "<|im_end|>", &end_mark)) return -1;
        *prefix_len = content_off;
        *body_off = content_off;
        *body_len = end_mark;
        *suffix_off = content_off + end_mark;
        return 1;
    }
    *prefix_len = 0;
    *body_off = 0;
    *body_len = in_len;
    *suffix_off = in_len;
    return 0;
}

static int llmm_parse_context_question(const char *body, size_t body_len, size_t *ctx_off, size_t *ctx_len,
                                       size_t *q_off, size_t *q_len)
{
    size_t qmark = 0;
    if (!llmm_find_marker(body, body_len, "\nQUESTION:", &qmark) &&
        !llmm_find_marker(body, body_len, "QUESTION:", &qmark)) {
        return -1;
    }
    size_t q_start = qmark;
    if (body[q_start] == '\n') ++q_start;
    q_start += strlen("QUESTION:");
    while (q_start < body_len && (body[q_start] == ' ' || body[q_start] == '\t')) ++q_start;
    size_t q_end = q_start;
    while (q_end < body_len && body[q_end] != '\n') ++q_end;

    size_t ctx_start = 0;
    if (body_len >= 8 && memcmp(body, "CONTEXT:", 8) == 0) {
        ctx_start = 8;
        while (ctx_start < qmark && (body[ctx_start] == '\n' || body[ctx_start] == '\r')) ++ctx_start;
    }
    *ctx_off = ctx_start;
    *ctx_len = (qmark > ctx_start) ? (qmark - ctx_start) : 0;
    *q_off = q_start;
    *q_len = q_end - q_start;
    return (*q_len > 0) ? 0 : -1;
}

static size_t llmm_append(char *dst, size_t cap, size_t used, const char *src, size_t len)
{
    if (used >= cap) return used;
    size_t room = cap - used;
    size_t n = len < room ? len : room;
    memcpy(dst + used, src, n);
    return used + n;
}

int llmm_compress_prompt_to_fit(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_cap, size_t *out_len,
                                llmm_compress_stats_t *stats)
{
    if (in == NULL || out == NULL || out_len == NULL || out_cap < 64 || in_len == 0) return -1;
    if (stats != NULL) memset(stats, 0, sizeof(*stats));

    if (in_len <= out_cap && in_len <= LLMM_COMPRESS_BYPASS_BYTES) {
        memcpy(out, in, in_len);
        *out_len = in_len;
        if (stats != NULL) {
            stats->source_bytes = (uint32_t)in_len;
            stats->packet_bytes = (uint32_t)in_len;
            stats->compressed = 0;
        }
        return 0;
    }

    size_t cap = out_cap;
    if (cap > LLMM_COMPRESS_FITTED_MAX_BYTES) cap = LLMM_COMPRESS_FITTED_MAX_BYTES;
    if (cap < 64) return -1;

    const char *text = (const char *)in;
    size_t body_off = 0, body_len = 0, prefix_len = 0, suffix_off = 0;
    const int chatml = llmm_extract_user_body(text, in_len, &body_off, &body_len, &prefix_len, &suffix_off);
    if (chatml < 0) return -2;

    const char *body = text + body_off;
    size_t ctx_off = 0, ctx_len = 0, q_off = 0, q_len = 0;
    if (llmm_parse_context_question(body, body_len, &ctx_off, &ctx_len, &q_off, &q_len) != 0) {
        size_t suffix_len = in_len - suffix_off;
        size_t overhead = prefix_len + suffix_len;
        if (overhead >= cap) return -3;
        size_t room = cap - overhead;
        size_t used = 0;
        used = llmm_append((char *)out, cap, used, text, prefix_len);
        used = llmm_append((char *)out, cap, used, body, room < body_len ? room : body_len);
        used = llmm_append((char *)out, cap, used, text + suffix_off, suffix_len);
        *out_len = used;
        if (stats != NULL) {
            stats->source_bytes = (uint32_t)in_len;
            stats->packet_bytes = (uint32_t)used;
            stats->compressed = 1;
        }
        return 0;
    }

    const char *source = body + ctx_off;
    const char *question = body + q_off;
    char terms[32][24];
    const size_t term_count = llmm_collect_question_terms(question, q_len, terms, 32);

    /* Keep large scratch off the USB task stack (PSRAM). */
    llmm_sent_t *sents = heap_caps_malloc(sizeof(llmm_sent_t) * LLMM_COMPRESS_MAX_SENTS, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint16_t *order = heap_caps_malloc(sizeof(uint16_t) * LLMM_COMPRESS_MAX_SENTS, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *selected = heap_caps_malloc(LLMM_COMPRESS_MAX_SENTS, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    char *body_out = heap_caps_malloc(cap + 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (sents == NULL || order == NULL || selected == NULL || body_out == NULL) {
        heap_caps_free(sents);
        heap_caps_free(order);
        heap_caps_free(selected);
        heap_caps_free(body_out);
        return -5;
    }

    const size_t sent_count = llmm_split_sentences(source, ctx_len, sents, LLMM_COMPRESS_MAX_SENTS);
    for (size_t i = 0; i < sent_count; ++i) {
        sents[i].score = llmm_score_sentence(source + sents[i].start, sents[i].len, terms, term_count);
        order[i] = (uint16_t)i;
    }
    for (size_t i = 1; i < sent_count; ++i) {
        uint16_t key = order[i];
        size_t j = i;
        while (j > 0) {
            const llmm_sent_t *a = &sents[order[j - 1]];
            const llmm_sent_t *b = &sents[key];
            if (a->score > b->score || (a->score == b->score && order[j - 1] < key)) break;
            order[j] = order[j - 1];
            --j;
        }
        order[j] = key;
    }

    const size_t source_tokens = llmm_estimate_tokens(source, ctx_len);
    size_t ratio_budget = source_tokens / (size_t)LLMM_COMPRESS_TARGET_RATIO;
    if (ratio_budget < 32U) ratio_budget = 32U;
    if (ratio_budget > LLMM_COMPRESS_MAX_TOKENS) ratio_budget = LLMM_COMPRESS_MAX_TOKENS;

    size_t suffix_len = in_len - suffix_off;
    size_t wrapper = prefix_len + suffix_len;
    static const char k_prefix[] = "CONTEXT:\n";
    static const char k_mid[] = "\n\nQUESTION: ";
    static const char k_suffix[] = "\nAnswer using only CONTEXT.";
    const size_t packet_fixed = strlen(k_prefix) + strlen(k_mid) + q_len + strlen(k_suffix);
    if (wrapper + packet_fixed + 8 > cap) {
        heap_caps_free(sents);
        heap_caps_free(order);
        heap_caps_free(selected);
        heap_caps_free(body_out);
        return -4;
    }
    const size_t body_budget = cap - wrapper;
    const size_t content_budget = body_budget > packet_fixed ? body_budget - packet_fixed : 0;

    memset(selected, 0, LLMM_COMPRESS_MAX_SENTS);
    size_t kept = 0;

    for (size_t r = 0; r < sent_count; ++r) {
        const uint16_t idx = order[r];
        if (kept >= 8U) break;
        if (kept >= 3U && sents[idx].score < 5.0f) break;
        selected[idx] = 1;
        size_t body_bytes = packet_fixed;
        size_t est_words = 0;
        for (size_t i = 0; i < sent_count; ++i) {
            if (!selected[i]) continue;
            if (body_bytes > packet_fixed) body_bytes += 1;
            body_bytes += sents[i].len;
            est_words += (size_t)(sents[i].len / 5) + 1U;
        }
        const size_t est_tokens = (est_words * 4U + 2U) / 3U + llmm_estimate_tokens(question, q_len) + 12U;
        if (body_bytes > body_budget || (kept > 0 && est_tokens > (size_t)(ratio_budget * 1.15f + 0.5f)) ||
            (body_bytes - packet_fixed) > content_budget) {
            selected[idx] = 0;
            continue;
        }
        ++kept;
    }

    size_t bu = 0;
    bu = llmm_append(body_out, cap, bu, k_prefix, strlen(k_prefix));
    size_t dropped = sent_count > kept ? sent_count - kept : 0;
    if (kept == 0) {
        size_t room = content_budget;
        if (room > ctx_len) room = ctx_len;
        bu = llmm_append(body_out, cap, bu, source, room);
        dropped = sent_count;
    } else {
        int first = 1;
        for (size_t i = 0; i < sent_count; ++i) {
            if (!selected[i]) continue;
            if (!first) bu = llmm_append(body_out, cap, bu, "\n", 1);
            bu = llmm_append(body_out, cap, bu, source + sents[i].start, sents[i].len);
            first = 0;
        }
    }
    bu = llmm_append(body_out, cap, bu, k_mid, strlen(k_mid));
    bu = llmm_append(body_out, cap, bu, question, q_len);
    bu = llmm_append(body_out, cap, bu, k_suffix, strlen(k_suffix));

    size_t used = 0;
    used = llmm_append((char *)out, cap, used, text, prefix_len);
    used = llmm_append((char *)out, cap, used, body_out, bu);
    used = llmm_append((char *)out, cap, used, text + suffix_off, suffix_len);
    *out_len = used;

    if (stats != NULL) {
        stats->source_bytes = (uint32_t)in_len;
        stats->packet_bytes = (uint32_t)used;
        stats->kept_sentences = (uint32_t)kept;
        stats->dropped_sentences = (uint32_t)dropped;
        stats->compressed = 1;
    }

    heap_caps_free(sents);
    heap_caps_free(order);
    heap_caps_free(selected);
    heap_caps_free(body_out);
    return 0;
}
