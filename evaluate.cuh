#pragma once

#include "game_logic.cuh"

// ─── 整数评估常量（×1024 定点数，确保 GPU/CPU 结果完全一致）─────────────
enum : int {
    EVAL_NEG_W8      = 10240,  // 10.0
    EVAL_NEG_W_OTHER  = 6144,   // 6.0
    EVAL_SMOOTH       = 819,    // 0.8
    EVAL_MONO         = 512,    // 0.5
    EVAL_MERGE        = 2048,   // 2.0
    EVAL_MONO_EXTRA   = 82,     // 0.08
    EVAL_EMPTY_BASE   = 24576,  // 24.0
    EVAL_EMPTY_COEF   = 358,    // 0.35
    EVAL_CORNER_MUL   = 2048,   // 2.0
    EVAL_DEAD_PENALTY = 30720,  // 30.0
};
// dist_w 定点数: {12.0, 7.44, 4.6128, 2.859936, 1.77316032, 1.09935936, 0.68160288} * 1024
// dist_w 定点数: {12.0, 7.44, 4.6128, 2.859936, 1.77316032, 1.09935936, 0.68160288} * 1024
__host__ __device__ static const int dist_w_int[7] = {12288, 7619, 4724, 2929, 1816, 1126, 698};

// ─── OI-2048 纯整数评估（GPU/CPU 结果完全一致）────────────────────────
__host__ __device__ static int evaluate(const int *grid, const int *pos_w, int cr, int cc, int = 0) {
    int log_grid[16];
    int phase_max_log = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int v = grid[i];
        int l = v != 0 ? (v > 0 ? ilog2(v) : ilog2(-v)) : 0;
        log_grid[i] = l;
        if (l > phase_max_log) phase_max_log = l;
    }

    // ── 阶段自适应权重（根据最大方块值分三阶段）────────────────
    int phase_smooth, phase_mono, phase_merge, phase_empty_coef;
    if (phase_max_log >= 12) {
        // 后期（≥4096）：优先存活，加强平滑/单调/合并
        phase_smooth = EVAL_SMOOTH * 3 / 2;       // 1.5x
        phase_mono   = EVAL_MONO * 5 / 4;         // 1.25x
        phase_merge  = EVAL_MERGE * 3 / 2;        // 1.5x
        phase_empty_coef = EVAL_EMPTY_COEF * 4 / 3; // 1.33x
    } else if (phase_max_log >= 8) {
        // 中期（≥256）：平衡
        phase_smooth = EVAL_SMOOTH;
        phase_mono   = EVAL_MONO;
        phase_merge  = EVAL_MERGE;
        phase_empty_coef = EVAL_EMPTY_COEF;
    } else {
        // 早期：重视建角，降低平滑/空格权重避免过早优化
        phase_smooth = EVAL_SMOOTH * 3 / 4;       // 0.75x
        phase_mono   = EVAL_MONO;                 // 1.0x
        phase_merge  = EVAL_MERGE * 2 / 3;        // 0.67x
        phase_empty_coef = EVAL_EMPTY_COEF * 2 / 3; // 0.67x
    }

    int score = 0;
    int empty_cnt = 0;
    int max_val = 0;
    int max_log = 0;
    int max_pos_r = -1;
    int max_pos_c = -1;
    int row_sign = cc == 3 ? -1 : 1;
    int col_sign = cr == 3 ? -1 : 1;

#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int idx = r * 4 + c;
            int v = grid[idx];
            int log_v = log_grid[idx];
            bool v_pos = false;

            if (v == 0) {
                empty_cnt++;
            } else {
                v_pos = v > 0;
                if (v_pos) {
                    score += log_v * pos_w[idx];
                    if (v > max_val) { max_val = v; max_log = log_v; max_pos_r = r; max_pos_c = c; }
                } else {
                    int best_nbr = 0;
                    if (r > 0 && grid[idx - 4] > 0) best_nbr = max(best_nbr, grid[idx - 4]);
                    if (r < 3 && grid[idx + 4] > 0) best_nbr = max(best_nbr, grid[idx + 4]);
                    if (c > 0 && grid[idx - 1] > 0) best_nbr = max(best_nbr, grid[idx - 1]);
                    if (c < 3 && grid[idx + 1] > 0) best_nbr = max(best_nbr, grid[idx + 1]);
                    if (v <= -8) {
                        for (int c2 = 0; c2 < 4; c2++) {
                            if (int nbr = grid[r * 4 + c2]; nbr > 0 && nbr > best_nbr) best_nbr = nbr;
                        }
                        for (int r2 = 0; r2 < 4; r2++) {
                            if (int nbr = grid[r2 * 4 + c]; nbr > 0 && nbr > best_nbr) best_nbr = nbr;
                        }
                    }
                    if (best_nbr > 0) {
                        if (v <= -8 && -v * best_nbr > 65536) { /* skip */ }
                        else {
                            int w = v <= -8 ? EVAL_NEG_W8 : EVAL_NEG_W_OTHER;
                            score += (log_v + ilog2(best_nbr)) * w;
                        }
                    }
                }
            }

            if (c < 3) {
                if (int b = grid[idx + 1]; v_pos && b > 0) {
                    int lb = log_grid[idx + 1];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= diff * phase_smooth;
                    score += (v >= b ? 1 : -1) * row_sign * log_v * phase_mono;
                    if (v == b) score += log_v * phase_merge;
                }
            }

            if (r < 3) {
                if (int b = grid[idx + 4]; v_pos && b > 0) {
                    int lb = log_grid[idx + 4];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= diff * phase_smooth;
                    score += (v >= b ? 1 : -1) * col_sign * log_v * phase_mono;
                    if (v == b) score += log_v * phase_merge;
                }
            }
        }
    }

    if (int mono_extra = (max_log - 8) * EVAL_MONO_EXTRA; mono_extra > 0) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) {
                int idx = r * 4 + c;
                int b = grid[idx + 1];
                if (int a = grid[idx]; a > 0 && b > 0)
                    score += (a >= b ? 1 : -1) * row_sign * log_grid[idx] * mono_extra;
            }
        }
        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 3; r++) {
                int idx = r * 4 + c;
                int a = grid[idx];
                if (int b = grid[idx + 4]; a > 0 && b > 0)
                    score += (a >= b ? 1 : -1) * col_sign * log_grid[idx] * mono_extra;
            }
        }
    }

    int empty_bonus = EVAL_EMPTY_BASE;
    if (max_val > 0) empty_bonus += max_log * max_log * phase_empty_coef;
    score += empty_cnt * empty_bonus;

    if (max_pos_r >= 0 && (max_pos_r != cr || max_pos_c != cc)) {
        int dist = abs(max_pos_r - cr) + abs(max_pos_c - cc);
        score -= max_log * dist * EVAL_CORNER_MUL;
    }

    // ── 死局 / 合并潜力评估（提前触发 + 分级惩罚）──────────────
    if (empty_cnt <= 6 && phase_max_log >= 8) {
        int merge_cnt = 0;
        // 水平相邻可合并对
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 3; c++) {
                int a = grid[r * 4 + c], b = grid[r * 4 + c + 1];
                if (a == 0 || b == 0) continue;
                if (a > 0 && a == b) merge_cnt++;
                if (a > 0 && b < 0 && -b * a <= 65536) merge_cnt++;
                if (a < 0 && b > 0 && -a * b <= 65536) merge_cnt++;
            }
        // 垂直相邻可合并对
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 3; r++) {
                int a = grid[r * 4 + c], b = grid[(r + 1) * 4 + c];
                if (a == 0 || b == 0) continue;
                if (a > 0 && a == b) merge_cnt++;
                if (a > 0 && b < 0 && -b * a <= 65536) merge_cnt++;
                if (a < 0 && b > 0 && -a * b <= 65536) merge_cnt++;
            }
        // ≤-8 非相邻合并潜力（行/列方向各计 1）
        for (int i = 0; i < 16; i++) {
            if (grid[i] > -8) continue;
            int r = i >> 2, c = i & 3;
            for (int cc1 = 0; cc1 < 4; cc1++) {
                if (int b = grid[r * 4 + cc1]; b > 0 && cc1 != c && -grid[i] * b <= 65536) { merge_cnt++; break; }
            }
            for (int rr = 0; rr < 4; rr++) {
                if (int b = grid[rr * 4 + c]; b > 0 && rr != r && -grid[i] * b <= 65536) { merge_cnt++; break; }
            }
        }

        if (merge_cnt == 0) {
            // 完全死局：重罚
            score -= phase_max_log * EVAL_DEAD_PENALTY;
        } else if (empty_cnt <= 3 && merge_cnt <= 1) {
            // 近乎死局（空格极少 + 合并可能极少）
            score -= phase_max_log * EVAL_DEAD_PENALTY * 2 / 3;
        } else if (empty_cnt <= 4 && merge_cnt <= 2) {
            // 接近死局，轻度惩罚
            score -= phase_max_log * EVAL_DEAD_PENALTY / 3;
        }
    }

    return score;
}
