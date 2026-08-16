#include "llmm_pack.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"

#define KPACK_HEADER_BYTES 32
#define KPACK_DIR_ENT 12
#define KPACK_MAX_FOLD 64
#define KPACK_MAX_REC 16
#define KPACK_MAX_NAME 80
#define KPACK_MAX_PT 40
#define KPACK_MAX_TERM 40

static const char kC3Ascii[] = "aaaaaa.ceeeeiiii.nooooo..uuuuy..aaaaaa.ceeeeiiii.nooooo..uuuuy.y";

typedef struct {
    char name[KPACK_MAX_NAME + 1];
    char pt[KPACK_MAX_PT + 1];
    char term[KPACK_MAX_TERM + 1];
    char code[5];
} pack_row_t;

static int is_stop(const char *tok)
{
    static const char *const stop[] = {
        "a", "an", "the", "of", "or", "in", "is", "which", "what", "where",
        "province", "territory", "capital", NULL};
    for (int i = 0; stop[i] != NULL; ++i) {
        if (strcmp(tok, stop[i]) == 0) return 1;
    }
    return 0;
}

static size_t utf8_step(const uint8_t *raw, size_t n, size_t i)
{
    uint8_t b0 = raw[i];
    if ((b0 & 0xE0U) == 0xC0U) return (i + 2 <= n) ? 2 : 1;
    if ((b0 & 0xF0U) == 0xE0U) return (i + 3 <= n) ? 3 : 1;
    if ((b0 & 0xF8U) == 0xF0U) return (i + 4 <= n) ? 4 : 1;
    return 1;
}

void llmm_pack_fold(const char *text, char *out, size_t out_bytes)
{
    if (out_bytes == 0) return;
    const uint8_t *raw = (const uint8_t *)text;
    size_t n = strlen(text);
    size_t o = 0;
    int space = 1;
    for (size_t i = 0; i < n;) {
        uint8_t b0 = raw[i];
        char ch = 0;
        if (b0 < 0x80U) {
            if ((b0 >= '0' && b0 <= '9') || (b0 >= 'a' && b0 <= 'z')) ch = (char)b0;
            else if (b0 >= 'A' && b0 <= 'Z') ch = (char)(b0 + 32);
            else if (b0 == '\'' || b0 == '`') {
                i += 1;
                continue;
            } else {
                ch = ' ';
            }
            i += 1;
        } else if (b0 == 0xC3U && i + 1 < n && raw[i + 1] >= 0x80U && raw[i + 1] <= 0xBFU) {
            char mapped = kC3Ascii[raw[i + 1] - 0x80U];
            ch = (mapped != '.') ? mapped : ' ';
            i += 2;
        } else if (b0 == 0xE2U && i + 2 < n && raw[i + 1] == 0x80U &&
                   (raw[i + 2] == 0x98U || raw[i + 2] == 0x99U)) {
            i += 3;
            continue;
        } else {
            i += utf8_step(raw, n, i);
            ch = ' ';
        }
        if (ch == ' ') {
            if (!space && o + 1 < out_bytes) {
                out[o++] = ' ';
                space = 1;
            }
        } else if (o + 1 < out_bytes) {
            out[o++] = ch;
            space = 0;
        }
    }
    if (o > 0 && out[o - 1] == ' ') o -= 1;
    out[o] = 0;
}

static int starts_ieq(const char *s, const char *prefix)
{
    while (*prefix) {
        unsigned char a = (unsigned char)*s++;
        unsigned char b = (unsigned char)*prefix++;
        if (tolower(a) != tolower(b)) return 0;
        if (a == 0) return 0;
    }
    return 1;
}

static const char *extract_pt_name(const char *q, char *buf, size_t buf_n)
{
    static const char *lead = "which province or territory is ";
    if (!starts_ieq(q, lead)) return NULL;
    const char *name = q + strlen(lead);
    size_t n = strlen(name);
    while (n > 0 && (name[n - 1] == '?' || name[n - 1] == ' ')) n -= 1;
    if (n >= 3 && (name[n - 3] == ' ' || name[n - 3] == '\t') &&
        tolower((unsigned char)name[n - 2]) == 'i' &&
        tolower((unsigned char)name[n - 1]) == 'n') {
        n -= 3;
        while (n > 0 && name[n - 1] == ' ') n -= 1;
    }
    if (n == 0 || n >= buf_n) return NULL;
    memcpy(buf, name, n);
    buf[n] = 0;
    return buf;
}

static int capital_answer(const char *q, char *out, size_t out_bytes)
{
    static const struct {
        const char *key;
        const char *val;
    } fwd[] = {
        {"canada", "Ottawa, Ontario (federal capital)"},
        {"alberta", "Edmonton"},
        {"british columbia", "Victoria"},
        {"bc", "Victoria"},
        {"manitoba", "Winnipeg"},
        {"new brunswick", "Fredericton"},
        {"newfoundland and labrador", "St. John's"},
        {"newfoundland", "St. John's"},
        {"nova scotia", "Halifax"},
        {"northwest territories", "Yellowknife"},
        {"nwt", "Yellowknife"},
        {"nunavut", "Iqaluit"},
        {"ontario", "Toronto"},
        {"prince edward island", "Charlottetown"},
        {"pei", "Charlottetown"},
        {"quebec", "Quebec City"},
        {"saskatchewan", "Regina"},
        {"yukon", "Whitehorse"},
        {NULL, NULL},
    };
    static const struct {
        const char *key;
        const char *city;
        const char *pt;
    } rev[] = {
        {"ottawa", "Ottawa", "Canada (federal capital, in Ontario)"},
        {"edmonton", "Edmonton", "Alberta"},
        {"victoria", "Victoria", "British Columbia"},
        {"winnipeg", "Winnipeg", "Manitoba"},
        {"fredericton", "Fredericton", "New Brunswick"},
        {"st johns", "St. John's", "Newfoundland and Labrador"},
        {"halifax", "Halifax", "Nova Scotia"},
        {"yellowknife", "Yellowknife", "Northwest Territories"},
        {"iqaluit", "Iqaluit", "Nunavut"},
        {"toronto", "Toronto", "Ontario"},
        {"charlottetown", "Charlottetown", "Prince Edward Island"},
        {"quebec city", "Quebec City", "Quebec"},
        {"regina", "Regina", "Saskatchewan"},
        {"whitehorse", "Whitehorse", "Yukon"},
        {NULL, NULL, NULL},
    };

    const char *which_cap = NULL;
    for (const char *p = q; *p; ++p) {
        if (!starts_ieq(p, " is the capital of which")) continue;
        which_cap = p;
        break;
    }
    if (which_cap != NULL) {
        char city[96];
        size_t n = (size_t)(which_cap - q);
        if (n > 0 && n < sizeof(city)) {
            memcpy(city, q, n);
            city[n] = 0;
            char folded[96];
            llmm_pack_fold(city, folded, sizeof(folded));
            for (int i = 0; rev[i].key != NULL; ++i) {
                if (strcmp(folded, rev[i].key) == 0) {
                    snprintf(out, out_bytes, "%s is the capital of %s.", city, rev[i].pt);
                    return 1;
                }
            }
            snprintf(out, out_bytes, "Unknown — no capital packed for '%s'.", city);
            return 1;
        }
    }

    const char *cap = NULL;
    for (const char *p = q; *p; ++p) {
        if (starts_ieq(p, "capital of ")) {
            cap = p + strlen("capital of ");
            break;
        }
    }
    if (cap == NULL) return 0;
    char place[96];
    snprintf(place, sizeof(place), "%s", cap);
    size_t n = strlen(place);
    while (n > 0 && (place[n - 1] == '?' || place[n - 1] == ' ')) place[--n] = 0;
    char folded[96];
    llmm_pack_fold(place, folded, sizeof(folded));
    if (strcmp(folded, "which province or territory") == 0 || strcmp(folded, "which province") == 0) {
        return 0;
    }
    for (int i = 0; fwd[i].key != NULL; ++i) {
        if (strcmp(folded, fwd[i].key) == 0) {
            snprintf(out, out_bytes, "%s", fwd[i].val);
            return 1;
        }
    }
    snprintf(out, out_bytes, "Unknown — no capital packed for '%s'.", place);
    return 1;
}

static int read_u16(FILE *fp, uint16_t *v)
{
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return -1;
    *v = (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
    return 0;
}

static int read_u32(FILE *fp, uint32_t *v)
{
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int dir_at(FILE *fp, uint32_t dir_off, uint32_t i,
                  uint32_t *fold_off, uint32_t *rec_off, uint16_t *n_rec, uint16_t *fold_len)
{
    if (fseek(fp, (long)(dir_off + i * KPACK_DIR_ENT), SEEK_SET) != 0) return -1;
    if (read_u32(fp, fold_off) != 0 || read_u32(fp, rec_off) != 0) return -1;
    if (read_u16(fp, n_rec) != 0 || read_u16(fp, fold_len) != 0) return -1;
    return 0;
}

static int fold_at(FILE *fp, uint32_t fold_off, uint16_t fold_len, char *buf, size_t buf_n)
{
    if (fold_len >= buf_n || fold_len > KPACK_MAX_FOLD) return -1;
    if (fseek(fp, (long)fold_off, SEEK_SET) != 0) return -1;
    if (fread(buf, 1, fold_len, fp) != fold_len) return -1;
    buf[fold_len] = 0;
    return 0;
}

typedef struct {
    uint32_t hash;
    uint32_t rec_off_len; /* low 24 rec_off, high 8 fold_len */
    uint32_t fold_off;
} pack_ix_ent_t;

static pack_ix_ent_t *s_ix;
static uint32_t s_ix_n;

static uint32_t fold_hash(const char *s)
{
    uint32_t h = 2166136261u;
    for (; *s != 0; ++s) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static int ix_cmp(const void *a, const void *b)
{
    const pack_ix_ent_t *x = a;
    const pack_ix_ent_t *y = b;
    if (x->hash < y->hash) return -1;
    if (x->hash > y->hash) return 1;
    return 0;
}

void llmm_pack_index_reset(void)
{
    if (s_ix != NULL) {
        heap_caps_free(s_ix);
        s_ix = NULL;
    }
    s_ix_n = 0;
}

int llmm_pack_index_build(FILE *fp)
{
    if (s_ix != NULL) return 0;
    if (fp == NULL) return -1;
    uint8_t header[KPACK_HEADER_BYTES];
    if (fseek(fp, 0, SEEK_SET) != 0) return -2;
    if (fread(header, 1, sizeof(header), fp) != sizeof(header)) return -2;
    if (memcmp(header, "KPK1", 4) != 0) return -3;
    uint32_t n_keys = (uint32_t)header[8] | ((uint32_t)header[9] << 8) |
                      ((uint32_t)header[10] << 16) | ((uint32_t)header[11] << 24);
    uint32_t dir_off = (uint32_t)header[12] | ((uint32_t)header[13] << 8) |
                       ((uint32_t)header[14] << 16) | ((uint32_t)header[15] << 24);
    if (n_keys == 0 || n_keys > 400000U) return -3;

    s_ix = (pack_ix_ent_t *)heap_caps_malloc(
        (size_t)n_keys * sizeof(pack_ix_ent_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_ix == NULL) return -1;

    enum { BATCH = 2048 };
    static uint8_t dir[BATCH * KPACK_DIR_ENT];
    uint32_t done = 0;
    while (done < n_keys) {
        uint32_t n = n_keys - done;
        if (n > BATCH) n = BATCH;
        if (fseek(fp, (long)(dir_off + done * KPACK_DIR_ENT), SEEK_SET) != 0) goto fail;
        if (fread(dir, KPACK_DIR_ENT, n, fp) != n) goto fail;
        uint32_t first_fold = (uint32_t)dir[0] | ((uint32_t)dir[1] << 8) |
                              ((uint32_t)dir[2] << 16) | ((uint32_t)dir[3] << 24);
        if (fseek(fp, (long)first_fold, SEEK_SET) != 0) goto fail;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t *e = dir + j * KPACK_DIR_ENT;
            uint32_t foff = (uint32_t)e[0] | ((uint32_t)e[1] << 8) |
                            ((uint32_t)e[2] << 16) | ((uint32_t)e[3] << 24);
            uint32_t roff = (uint32_t)e[4] | ((uint32_t)e[5] << 8) |
                            ((uint32_t)e[6] << 16) | ((uint32_t)e[7] << 24);
            uint16_t flen = (uint16_t)(e[10] | (e[11] << 8));
            if (flen == 0 || flen > KPACK_MAX_FOLD || (roff & ~0xFFFFFFU) != 0) goto fail;
            char fold[KPACK_MAX_FOLD + 1];
            if (fread(fold, 1, flen, fp) != flen) goto fail;
            fold[flen] = 0;
            pack_ix_ent_t *ent = &s_ix[done + j];
            ent->hash = fold_hash(fold);
            ent->rec_off_len = (roff & 0xFFFFFFU) | ((uint32_t)flen << 24);
            ent->fold_off = foff;
        }
        done += n;
    }
    qsort(s_ix, n_keys, sizeof(s_ix[0]), ix_cmp);
    s_ix_n = n_keys;
    return 0;
fail:
    llmm_pack_index_reset();
    return -2;
}

static int find_key_sd(FILE *fp, uint32_t n_keys, uint32_t dir_off, const char *folded,
                       uint32_t *rec_off, uint16_t *n_rec)
{
    uint32_t lo = 0;
    uint32_t hi = n_keys;
    char got[KPACK_MAX_FOLD + 1];
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2U;
        uint32_t foff, roff;
        uint16_t nr, flen;
        if (dir_at(fp, dir_off, mid, &foff, &roff, &nr, &flen) != 0) return -1;
        if (fold_at(fp, foff, flen, got, sizeof(got)) != 0) return -1;
        int cmp = strcmp(got, folded);
        if (cmp < 0) lo = mid + 1U;
        else hi = mid;
    }
    if (lo >= n_keys) return 1;
    uint32_t foff, roff;
    uint16_t nr, flen;
    if (dir_at(fp, dir_off, lo, &foff, &roff, &nr, &flen) != 0) return -1;
    if (fold_at(fp, foff, flen, got, sizeof(got)) != 0) return -1;
    if (strcmp(got, folded) != 0) return 1;
    *rec_off = roff;
    *n_rec = nr;
    return 0;
}

static int find_key_ram(FILE *fp, const char *folded, uint32_t *rec_off, uint16_t *n_rec)
{
    uint32_t h = fold_hash(folded);
    uint32_t lo = 0;
    uint32_t hi = s_ix_n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2U;
        if (s_ix[mid].hash < h) lo = mid + 1U;
        else hi = mid;
    }
    char got[KPACK_MAX_FOLD + 1];
    for (; lo < s_ix_n && s_ix[lo].hash == h; ++lo) {
        uint16_t flen = (uint16_t)(s_ix[lo].rec_off_len >> 24);
        if (fold_at(fp, s_ix[lo].fold_off, flen, got, sizeof(got)) != 0) return -1;
        if (strcmp(got, folded) == 0) {
            *rec_off = s_ix[lo].rec_off_len & 0xFFFFFFU;
            *n_rec = KPACK_MAX_REC;
            return 0;
        }
    }
    return 1;
}

static int find_key(FILE *fp, uint32_t n_keys, uint32_t dir_off, const char *folded,
                    uint32_t *rec_off, uint16_t *n_rec)
{
    if (s_ix != NULL && s_ix_n > 0U) return find_key_ram(fp, folded, rec_off, n_rec);
    return find_key_sd(fp, n_keys, dir_off, folded, rec_off, n_rec);
}

static int load_rows(FILE *fp, uint32_t rec_off, uint16_t n_rec, pack_row_t *rows, int *out_n)
{
    if (fseek(fp, (long)rec_off, SEEK_SET) != 0) return -1;
    int stored = fgetc(fp);
    if (stored < 0) return -1;
    int count = stored;
    if (count > n_rec) count = n_rec;
    if (count > KPACK_MAX_REC) count = KPACK_MAX_REC;
    for (int i = 0; i < count; ++i) {
        int nlen = fgetc(fp);
        if (nlen < 0 || nlen > KPACK_MAX_NAME) return -1;
        if (fread(rows[i].name, 1, (size_t)nlen, fp) != (size_t)nlen) return -1;
        rows[i].name[nlen] = 0;
        int plen = fgetc(fp);
        if (plen < 0 || plen > KPACK_MAX_PT) return -1;
        if (fread(rows[i].pt, 1, (size_t)plen, fp) != (size_t)plen) return -1;
        rows[i].pt[plen] = 0;
        int tlen = fgetc(fp);
        if (tlen < 0 || tlen > KPACK_MAX_TERM) return -1;
        if (fread(rows[i].term, 1, (size_t)tlen, fp) != (size_t)tlen) return -1;
        rows[i].term[tlen] = 0;
        if (fread(rows[i].code, 1, 4, fp) != 4) return -1;
        rows[i].code[4] = 0;
        for (int c = 3; c >= 0 && rows[i].code[c] == ' '; --c) rows[i].code[c] = 0;
    }
    *out_n = count;
    return 0;
}

static void format_provinces(const pack_row_t *rows, int n, char *out, size_t out_bytes)
{
    const char *pts[KPACK_MAX_REC];
    int npt = 0;
    for (int i = 0; i < n; ++i) {
        if (rows[i].pt[0] == 0) continue;
        int seen = 0;
        for (int j = 0; j < npt; ++j) {
            if (strcmp(pts[j], rows[i].pt) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen && npt < KPACK_MAX_REC) pts[npt++] = rows[i].pt;
    }
    if (npt == 0) {
        snprintf(out, out_bytes, "Unknown — no province packed for '%s'.", rows[0].name);
        return;
    }
    if (npt == 1) {
        const pack_row_t *row = rows;
        for (int i = 0; i < n; ++i) {
            if (strcmp(rows[i].pt, pts[0]) == 0) {
                row = &rows[i];
                break;
            }
        }
        if (row->term[0]) {
            snprintf(out, out_bytes, "%s — %s, in %s.", row->name, row->term, row->pt);
        } else {
            snprintf(out, out_bytes, "%s — in %s.", row->name, row->pt);
        }
        return;
    }
    if (npt == 2) {
        snprintf(out, out_bytes, "%s is an official name in %s and %s.", rows[0].name, pts[0], pts[1]);
        return;
    }
    size_t used = (size_t)snprintf(out, out_bytes, "%s is an official name in ", rows[0].name);
    for (int i = 0; i < npt && used < out_bytes; ++i) {
        const char *sep = (i == 0) ? "" : (i == npt - 1) ? ", and " : ", ";
        used += (size_t)snprintf(out + used, out_bytes - used, "%s%s", sep, pts[i]);
    }
    if (used < out_bytes) snprintf(out + used, out_bytes - used, ".");
}

static int lookup_folded(FILE *fp, uint32_t n_keys, uint32_t dir_off, const char *folded,
                         pack_row_t *rows, int *n_rows)
{
    uint32_t rec_off = 0;
    uint16_t n_rec = 0;
    int st = find_key(fp, n_keys, dir_off, folded, &rec_off, &n_rec);
    if (st != 0) return st;
    return load_rows(fp, rec_off, n_rec, rows, n_rows);
}

static int pick_and_load(FILE *fp, uint32_t n_keys, uint32_t dir_off, const char *q,
                         pack_row_t *rows, int *n_rows)
{
    char namebuf[192];
    char folded[KPACK_MAX_FOLD + 8];
    if (extract_pt_name(q, namebuf, sizeof(namebuf))) {
        llmm_pack_fold(namebuf, folded, sizeof(folded));
        int st = lookup_folded(fp, n_keys, dir_off, folded, rows, n_rows);
        if (st <= 0) return st;
        return 1;
    }

    char qfold[256];
    llmm_pack_fold(q, qfold, sizeof(qfold));
    char *tok[24];
    int nt = 0;
    for (char *p = strtok(qfold, " "); p != NULL && nt < 24; p = strtok(NULL, " ")) {
        tok[nt++] = p;
    }
    char best[KPACK_MAX_FOLD + 8] = {0};
    int best_parts = 0;
    int max_len = nt < 12 ? nt : 12;
    for (int length = max_len; length >= 1; --length) {
        for (int start = 0; start + length <= nt; ++start) {
            if (length == 1 && is_stop(tok[start])) continue;
            char span[KPACK_MAX_FOLD + 8];
            size_t used = 0;
            span[0] = 0;
            for (int k = 0; k < length; ++k) {
                used += (size_t)snprintf(span + used, sizeof(span) - used, "%s%s", k ? " " : "", tok[start + k]);
                if (used >= sizeof(span)) break;
            }
            int ntmp = 0;
            pack_row_t tmp[KPACK_MAX_REC];
            if (lookup_folded(fp, n_keys, dir_off, span, tmp, &ntmp) == 0) {
                snprintf(best, sizeof(best), "%s", span);
                best_parts = length;
                memcpy(rows, tmp, sizeof(tmp));
                *n_rows = ntmp;
                /* longest-first: take first hit at this length */
                start = nt;
                length = 0;
            }
        }
        if (best_parts) break;
    }
    return best_parts ? 0 : 1;
}

int llmm_pack_answer(FILE *fp, const char *question, char *out, size_t out_bytes)
{
    if (fp == NULL || question == NULL || out == NULL || out_bytes < 8) return -1;
    if (capital_answer(question, out, out_bytes)) return 0;
    if (s_ix == NULL) (void)llmm_pack_index_build(fp);

    uint8_t header[KPACK_HEADER_BYTES];
    if (fseek(fp, 0, SEEK_SET) != 0) return -2;
    if (fread(header, 1, sizeof(header), fp) != sizeof(header)) return -2;
    if (memcmp(header, "KPK1", 4) != 0) return -3;
    uint16_t version = (uint16_t)(header[4] | (header[5] << 8));
    if (version != 1) return -3;
    uint32_t n_keys = (uint32_t)header[8] | ((uint32_t)header[9] << 8) |
                      ((uint32_t)header[10] << 16) | ((uint32_t)header[11] << 24);
    uint32_t dir_off = (uint32_t)header[12] | ((uint32_t)header[13] << 8) |
                       ((uint32_t)header[14] << 16) | ((uint32_t)header[15] << 24);
    if (n_keys == 0) return -3;

    pack_row_t rows[KPACK_MAX_REC];
    int n_rows = 0;
    int st = pick_and_load(fp, n_keys, dir_off, question, rows, &n_rows);
    if (st < 0) return st;
    if (st != 0 || n_rows <= 0) {
        snprintf(out, out_bytes,
                 "Unknown — no official Canadian name in that question. "
                 "I only answer from the gazetteer (places, lakes, rivers, parks, peaks).");
        return 0;
    }
    format_provinces(rows, n_rows, out, out_bytes);
    return 0;
}
