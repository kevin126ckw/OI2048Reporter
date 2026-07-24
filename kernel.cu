#include "kernel.cuh"
#include "evaluate.cuh"

__global__ __launch_bounds__(256, 2) void simulate_games(const uint64_t base_seed,
        const int target_score, SimResult *results) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= NUM_THREADS) return;

    const int strategy = tid & 3;
    const int cr = strategy >= 2 ? 3 : 0;
    const int cc = strategy == 1 || strategy == 3 ? 3 : 0;
    int pos_w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int r = i >> 2;
        const int c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    auto rng = static_cast<uint32_t>(base_seed + tid * 2654435761ULL);
    int grid[16] = {};

    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);

    int score = 0;
    int steps = 0;

    while (true) {
        int best_eval = -2000000000;
        int best_dir = -1;
        int best_temp[16];
        int best_move_score = 0;

#pragma unroll
        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            const int move_score = apply_move(temp, dir, &moved);
            if (!moved) continue;
            if (const int e = evaluate(temp, pos_w, cr, cc); e > best_eval) {
                best_eval = e;
                best_dir = dir;
                copy_grid(temp, best_temp);
                best_move_score = move_score;
            }
        }

        if (best_dir < 0) break;
        if (steps >= MAX_STEPS) break;

        copy_grid(best_temp, grid);
        score += best_move_score;
        add_random_tile(grid, &rng);
        steps++;

        if (score >= target_score) break;
    }

    results[tid].score = score;
}
