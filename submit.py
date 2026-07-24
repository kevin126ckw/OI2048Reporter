#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""2048 成绩提交脚本 —— 从 submit_output.json 读取数据、验证并 POST"""

import json
import sys
import re
import copy
import urllib.request
import urllib.parse

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "submit_output.json")
JSON_FILE_ALT = os.path.join(SCRIPT_DIR, "cmake-build-debug", "submit_output.json")
SUBMIT_URL = "https://apps.ak-ioi.com/oi-2048/ranklist/submit_score.php"

# ─── 2048 游戏逻辑（与 CUDA 内核完全一致）─────────────────────────────

VALID_TILES = {2, 4, -1, -2}
INITIAL_TILES = {2, 4}


def slide_line(arr: list) -> int:
    """处理一行 4 个元素，沿正方向滑动合并，返回得分"""
    score = 0
    merged = [False] * 4

    for i in range(4):
        if arr[i] == 0 or arr[i] is None:
            continue

        farthest = i
        while farthest > 0 and (arr[farthest - 1] == 0 or arr[farthest - 1] is None):
            farthest -= 1

        prev = farthest - 1
        did_merge = False

        if prev >= 0 and arr[prev] not in (0, None) and not merged[prev]:
            a = arr[prev]
            b = arr[i]

            # 同值合并（不检查邻接）
            if a == b and 32768 >= a >= -2:
                arr[prev] = a * 2
                score += a * 2
                arr[i] = 0
                merged[prev] = True
                did_merge = True
            # 倍增方块合并：服务端 is_adj 豁免 <= -8 的方块
            elif i == prev + 1 or a <= -8 or b <= -8:
                if a > 0 and b <= -1 and -b * a <= 65536:
                    arr[prev] = -b * a
                    score += arr[prev]
                    arr[i] = 0
                    merged[prev] = True
                    did_merge = True
                elif a <= -1 and b > 0 and -a * b <= 65536:
                    arr[prev] = -a * b
                    score += arr[prev]
                    arr[i] = 0
                    merged[prev] = True
                    did_merge = True

        if not did_merge and farthest != i:
            arr[farthest] = arr[i]
            arr[i] = 0

    return score


def apply_move(grid, direction: int):
    """对 4x4 棋盘执行移动，返回 (new_grid, score_gain, moved)"""
    g = copy.deepcopy(grid)
    score = 0

    if direction == 0:  # 上
        for col in range(4):
            col_arr = [g[row][col] for row in range(4)]
            score += slide_line(col_arr)
            for row in range(4):
                g[row][col] = col_arr[row]
    elif direction == 2:  # 下
        for col in range(4):
            col_arr = [g[3 - row][col] for row in range(4)]
            score += slide_line(col_arr)
            for row in range(4):
                g[row][col] = col_arr[3 - row]
    elif direction == 1:  # 右
        for row in range(4):
            row_arr = [g[row][3 - col] for col in range(4)]
            score += slide_line(row_arr)
            for col in range(4):
                g[row][col] = row_arr[3 - col]
    else:  # 左
        for row in range(4):
            row_arr = [g[row][col] for col in range(4)]
            score += slide_line(row_arr)
            for col in range(4):
                g[row][col] = row_arr[col]

    moved = g != grid
    return g, score, moved


# ─── 验证逻辑 ─────────────────────────────────────────────────────

def grid_eq(a, b):
    """比较两个棋盘，0 和 None 视为等价"""
    for r in range(4):
        for c in range(4):
            va = a[r][c] if a[r][c] is not None else 0
            vb = b[r][c] if b[r][c] is not None else 0
            if va != vb:
                return False
    return True


def count_tiles(g):
    return sum(1 for r in range(4) for c in range(4)
               if g[r][c] not in (0, None))


def null_to_zero(g):
    return [[0 if v is None else v for v in row] for row in g]


def validate_history(data):
    """完整复现并验证游戏记录，返回问题列表，列表为空表示验证通过"""
    history = json.loads(data["history"])
    final_grid = json.loads(data["finalGrid"])
    issues = []

    if len(history) != data["steps"] + 1:
        issues.append(
            f"history 长度 {len(history)} != steps+1 ({data['steps'] + 1})"
        )

    # 1) 初始状态
    h0 = history[0]
    before_null = all(
        v is None for row in h0[0] for v in row
    )
    if not before_null:
        issues.append("初始 before 不全为 null")

    initial_tiles = [(r, c, v) for r in range(4) for c in range(4)
                     if (v := h0[1][r][c]) is not None]
    if len(initial_tiles) != 2:
        issues.append(f"初始 after 有 {len(initial_tiles)} 个方块（应为 2）")
    for _, _, v in initial_tiles:
        if v not in INITIAL_TILES:
            issues.append(f"初始方块值 {v} 非法（应为 2 或 4）")

    # 2) 最终棋盘
    if not grid_eq(history[-1][1], final_grid):
        issues.append("最终 after 与 finalGrid 不匹配")

    # 3) 逐步复现
    total_score = 0
    for i in range(1, len(history)):
        prev_after = history[i - 1][1]   # 上一步结束时的棋盘
        mid_want   = history[i][0]        # 玩家移动后、电脑新方块前的棋盘
        after_want = history[i][1]        # 玩家移动+电脑新方块后的棋盘

        found_dir = False
        found_gain = 0

        for d in range(4):
            mid, gain, moved = apply_move(prev_after, d)
            if not moved:
                continue
            if not grid_eq(mid, mid_want):
                continue
            found_dir = True
            found_gain = gain
            break

        if not found_dir:
            issues.append(
                f"步骤 {i}: 无法找到合法方向使 before → mid 匹配"
            )
            continue

        total_score += found_gain

        empty_pos = [(r, c) for r in range(4) for c in range(4)
                     if mid_want[r][c] in (0, None)]
        found_tile = False
        for r, c in empty_pos:
            for tv in VALID_TILES:
                test = copy.deepcopy(mid_want)
                test[r][c] = tv
                if grid_eq(test, after_want):
                    found_tile = True
                    break
            if found_tile:
                break

        if not found_tile:
            issues.append(
                f"步骤 {i}: 无法通过添加一个合法方块使 mid → after 匹配"
            )

    if total_score != data["score"]:
        issues.append(
            f"验证得分 {total_score} ≠ 声明得分 {data['score']}"
        )

    # 4) maxValueLog 一致性
    max_v = 0
    for r in range(4):
        for c in range(4):
            v = final_grid[r][c]
            if v is not None and v > max_v:
                max_v = v
    real_log2 = 0
    if max_v > 0:
        t = max_v
        while t > 1:
            t >>= 1
            real_log2 += 1
    if real_log2 != data["maxValueLog"]:
        issues.append(
            f"maxValueLog 应为 {real_log2} 而非 {data['maxValueLog']}"
        )

    return issues


# ─── 主流程 ───────────────────────────────────────────────────────

def load_data():
    path = None
    for p in (JSON_FILE, JSON_FILE_ALT):
        if os.path.exists(p):
            path = p
            break
    if not path:
        print(f"错误: 找不到 submit_output.json")
        print(f"  已搜索: {JSON_FILE}")
        print(f"  已搜索: {JSON_FILE_ALT}")
        print("请先运行 OI2048Reporter 生成数据，例如:")
        print("  ./cmake-build-debug/OI2048Reporter 10000")
        sys.exit(1)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: {path} 格式无效: {e}")
        sys.exit(1)

    print(f"已加载 {path}")
    print(f"  分数: {data['score']}")
    print(f"  步数: {data['steps']}")
    print(f"  最大方块 log2: {data['maxValueLog']}")
    print()
    return data


def main():
    data = load_data()

    # ── 验证 ──
    print("正在验证游戏记录...")
    issues = validate_history(data)
    if issues:
        print(f"\n❌ 验证失败，共 {len(issues)} 个问题:")
        for iss in issues:
            print(f"  - {iss}")
        print("\n拒绝提交，请修复后重试。")
        sys.exit(1)
    print("✅ 验证通过\n")

    # ── 用户名 ──
    while True:
        username = input("请输入用户名: ").strip()
        if not username:
            print("用户名不能为空")
            continue
        if not re.match(r"^\w+$", username):
            print("用户名只能包含字母、数字和下划线")
            continue
        if len(username) < 2:
            print("用户名至少需要 2 个字符")
            continue
        if len(username) > 14:
            print("用户名长度不能超过 14")
            continue
        break

    # ── 提交 ──
    payload = urllib.parse.urlencode({
        "username": username,
        "data": json.dumps(data)
    }).encode("utf-8")

    req = urllib.request.Request(
        SUBMIT_URL,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://apps.ak-ioi.com",
            "Referer": "https://apps.ak-ioi.com/oi-2048/",
        }
    )

    print(f"\n正在提交到 {SUBMIT_URL} ...")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"HTTP 错误 {e.code}: {body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"网络错误: {e.reason}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"服务器返回非 JSON: {body}")
        sys.exit(1)

    print()
    if result.get("success"):
        print(f"✅ 提交成功！排名: {result['data']}")
    else:
        print(f"❌ 提交失败: {result.get('data', '未知原因')}")


if __name__ == "__main__":
    main()
