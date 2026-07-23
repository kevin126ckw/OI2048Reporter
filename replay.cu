#include "replay.cuh"
#include "evaluate.cuh"

// ─── 贪心回放（与 GPU 逻辑一致）────────────────────────────────────────
HostSimResult replay_game(uint32_t rng_seed, int strategy, int target_score) {
    HostSimResult res;
    res.score = 0;
    res.steps = 0;
    res.history.clear();

    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    int pos_w[16];
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    uint32_t rng = rng_seed;
    int grid[16] = {0};

    // 初始状态: before=全空, after=初始两方块
    HistoryEntry init{};
    for (int & i : init.before) i = 0;
    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);
    copy_grid(grid, init.after);
    res.history.push_back(init);

    while (can_move(grid) && res.steps < MAX_STEPS) {
        int best_eval = -2000000000;
        int best_dir = -1;

        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            apply_move(temp, dir, &moved);
            if (!moved) continue;
            int e = evaluate(temp, pos_w, cr, cc);
            if (e > best_eval) {
                best_eval = e;
                best_dir = dir;
            }
        }

        if (best_dir < 0) break;

        HistoryEntry entry{};
        res.score += apply_move(grid, best_dir);
        copy_grid(grid, entry.before);  // 移动后、新方块前
        add_random_tile(grid, &rng);
        copy_grid(grid, entry.after);   // 移动后 + 新方块
        res.history.push_back(entry);
        res.steps++;

        // GPU 模拟用 target_score 找到达标的种子，
        // 但 Host 回放需要走到自然结束（游戏规则要求通关或无合法移动才能提交）
    }

    copy_grid(grid, res.final_grid);
    return res;
}

// ─── CPU expectimax 深搜（1-ply 前向搜索，加权采样）─────────────
// 根据实际生成概率加权：2(78.3%), 4(8.7%), -1(11.2%), -2(1.8%)
// 为保持速度，每空格只采样 2 和 -1 两种方块（覆盖 ~89.5% 概率），
// 剩余 10.5% 用 tile=2 的结果近似
static int choose_move_expectimax(const int *grid, int cr, int cc, const int *pos_w) {
    // 加权系数（总权重 1000）
    // 2: 783, 4: 87(近似用2), -1: 112, -2: 18(近似用2)
    // → tile2 综合权重 = 783 + 87 + 18 = 888, tile-1 权重 = 112
    constexpr int W2 = 888;
    constexpr int WN1 = 112;
    constexpr int WSUM = W2 + WN1; // 1000

    int best_expected = -2000000000;
    int best_dir = -1;

    for (int dir = 0; dir < 4; dir++) {
        int temp[16];
        copy_grid(grid, temp);
        bool moved;
        apply_move(temp, dir, &moved);
        if (!moved) continue;

        int empty[16];
        int empty_cnt = 0;
        for (int i = 0; i < 16; i++) if (temp[i] == 0) empty[empty_cnt++] = i;

        if (empty_cnt == 0) {
            int e = evaluate(temp, pos_w, cr, cc);
            if (e > best_expected) { best_expected = e; best_dir = dir; }
            continue;
        }

        // 跳跃采样（最多 4 个样本，每个样本尝试 2 和 -1 两种方块）
        int step = (empty_cnt <= 4) ? 1 : (empty_cnt / 4);
        int samples = 0;
        int weighted_sum = 0;

        for (int si = 0; si < empty_cnt && samples < 4; si += step) {
            int pos = empty[si];

            // ── 放置 tile=2 并做 1-ply 最佳应对 ──
            int best_r2 = -2000000000;
            {
                int after_tile[16];
                copy_grid(temp, after_tile);
                after_tile[pos] = 2;
                for (int d2 = 0; d2 < 4; d2++) {
                    int temp2[16];
                    copy_grid(after_tile, temp2);
                    bool moved2;
                    apply_move(temp2, d2, &moved2);
                    if (!moved2) continue;
                    int e = evaluate(temp2, pos_w, cr, cc);
                    if (e > best_r2) best_r2 = e;
                }
                if (best_r2 < -1900000000)
                    best_r2 = evaluate(after_tile, pos_w, cr, cc);
            }

            // ── 放置 tile=-1 并做 1-ply 最佳应对 ──
            int best_rn1 = -2000000000;
            {
                int after_tile[16];
                copy_grid(temp, after_tile);
                after_tile[pos] = -1;
                for (int d2 = 0; d2 < 4; d2++) {
                    int temp2[16];
                    copy_grid(after_tile, temp2);
                    bool moved2;
                    apply_move(temp2, d2, &moved2);
                    if (!moved2) continue;
                    int e = evaluate(temp2, pos_w, cr, cc);
                    if (e > best_rn1) best_rn1 = e;
                }
                if (best_rn1 < -1900000000)
                    best_rn1 = evaluate(after_tile, pos_w, cr, cc);
            }

            weighted_sum += best_r2 * W2 + best_rn1 * WN1;
            samples++;
        }
        int expected = weighted_sum / (samples * WSUM);

        if (expected > best_expected) {
            best_expected = expected;
            best_dir = dir;
        }
    }

    return best_dir;
}

HostSimResult replay_game_expectimax(uint32_t rng_seed, int strategy, int target_score) {
    HostSimResult res;
    res.score = 0;
    res.steps = 0;
    res.history.clear();

    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    int pos_w[16];
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    uint32_t rng = rng_seed;
    int grid[16] = {0};

    HistoryEntry init{};
    for (int &i : init.before) i = 0;
    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);
    copy_grid(grid, init.after);
    res.history.push_back(init);

    while (can_move(grid) && res.steps < MAX_STEPS) {
        int best_dir = choose_move_expectimax(grid, cr, cc, pos_w);

        if (best_dir < 0) break;

        HistoryEntry entry{};
        res.score += apply_move(grid, best_dir);
        copy_grid(grid, entry.before);
        add_random_tile(grid, &rng);
        copy_grid(grid, entry.after);
        res.history.push_back(entry);
        res.steps++;
    }

    copy_grid(grid, res.final_grid);
    return res;
}
