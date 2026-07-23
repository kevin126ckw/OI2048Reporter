#include "submit_format.cuh"

#include <random>

using std::string;

// ─── 随机字符串生成 ────────────────────────────────────────────────────
static string generate_gameid() {
    static constexpr char charset[] = "0123456789qwertyuiopasdfghjklzxcvbnm";
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, 35);
    string id;
    for (int i = 0; i < 16; i++) id += charset[dist(rng)];
    return id;
}

// ─── JSON 格式化 ──────────────────────────────────────────────────────
static string grid_to_json(const int *grid) {
    string s = "[";
    for (int r = 0; r < 4; r++) {
        s += '[';
        for (int c = 0; c < 4; c++) {
            int v = grid[r * 4 + c];
            if (v == 0) s += "null";
            else s += std::to_string(v);
            if (c < 3) s += ',';
        }
        s += ']';
        if (r < 3) s += ',';
    }
    s += ']';
    return s;
}

static string history_to_json(const std::vector<HistoryEntry> &history) {
    string s = "[";
    for (size_t i = 0; i < history.size(); i++) {
        s += '[';
        s += grid_to_json(history[i].before);
        s += ',';
        s += grid_to_json(history[i].after);
        s += ']';
        if (i + 1 < history.size()) s += ',';
    }
    s += ']';
    return s;
}

string generate_submit_data(const HostSimResult &result, bool cheated) {
    string gameid = generate_gameid();
    int mv = max_value_log2(result.final_grid);
    string fg_json = grid_to_json(result.final_grid);

    string data = "{";
    data += R"("gameid":")" + gameid + "\",";
    data += R"("finalGrid":")" + fg_json + "\",";
    data += "\"maxValueLog\":" + std::to_string(mv) + ",";
    data += "\"steps\":" + std::to_string(result.steps) + ",";
    data += R"("history":")" + history_to_json(result.history) + "\",";
    data += "\"score\":" + std::to_string(result.score) + ",";
    data += R"("scoreStringForReference":")" + std::to_string(result.score) + "\",";
    data += "\"cheated\":" + string(cheated ? "true" : "false");
    data += '}';
    return data;
}
