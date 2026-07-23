#pragma once

#include <string>

#include "replay.cuh"

// ─── 生成提交 JSON ────────────────────────────────────────────────────
std::string generate_submit_data(const HostSimResult &result, bool cheated = false);
