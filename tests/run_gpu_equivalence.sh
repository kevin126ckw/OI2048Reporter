#!/usr/bin/env bash
# ─── 多 GPU 等价性测试 ────────────────────────────────────────────────
# 同一基准种子下，每局游戏的结果只取决于全局线程号 tid 和批次号，与「用几张卡、怎么划分」无关。
# 本脚本用 --dump-scores 导出每个批次全部线程的分数，然后逐字节比较：
#
#     1 个切片 (--gpus 0)                ← 基准（等同于旧版单卡行为）
#     N 个逻辑切片 (--logical-devices N) ← 多卡调度路径（在单卡上模拟）
#
# 覆盖两条路径：
#   1. 目标分数不可达：所有批次完整跑完，比较全部批次的逐线程分数
#   2. 目标分数可达：触发提前终止（found），比较实际处理过的那些批次
# 两者都必须完全一致，否则说明多 GPU 划分改变了搜索结果。
# 无 GPU 环境自动跳过（退出码 0）。
#
# 用法: bash tests/run_gpu_equivalence.sh [可执行文件路径] [批次数]

set -uo pipefail

BIN="${1:-./cmake-build-release/OI2048Reporter}"
BATCHES="${2:-3}"
SEED=20240926
TARGET=100000000   # 不可能达到，保证每个批次都完整跑完
EARLY_TARGET=1000  # 必定达到，用于覆盖提前终止路径

if [[ ! -x "$BIN" ]]; then
    echo "找不到可执行文件: $BIN" >&2
    exit 1
fi
BIN="$(readlink -f "$BIN")"

if ! "$BIN" --list-devices > /dev/null 2>&1; then
    echo "[跳过] 当前环境无法访问 CUDA 设备"
    exit 0
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR" || exit 1

run_case() { # run_case <名称> <目标分数> <批次数> [额外参数...]
    local name="$1" target="$2" batches="$3"; shift 3
    echo "  运行 $name (目标 $target, $batches 批): $*"
    if ! "$BIN" "$target" "$batches" --seed "$SEED" --dump-scores "$name.bin" "$@" \
            > "$name.log" 2>&1; then
        echo "[失败] $name 退出码非 0，日志:" >&2
        tail -20 "$name.log" >&2
        exit 1
    fi
}

echo "多 GPU 等价性测试 (批次数=$BATCHES, 种子=$SEED)"
run_case single "$TARGET" "$BATCHES" --gpus 0
for n in 2 3 4; do
    run_case "logical${n}" "$TARGET" "$BATCHES" --gpus 0 --logical-devices "$n"
done

echo "提前终止路径:"
run_case early_single "$EARLY_TARGET" "$BATCHES" --gpus 0
run_case early_logical4 "$EARLY_TARGET" "$BATCHES" --gpus 0 --logical-devices 4

echo "收尾路径 (只有 1~2 个批次，主循环收尾才处理结果):"
run_case single_b1 "$TARGET" 1 --gpus 0
run_case logical2_b1 "$TARGET" 1 --gpus 0 --logical-devices 2
run_case single_b2 "$TARGET" 2 --gpus 0
run_case logical2_b2 "$TARGET" 2 --gpus 0 --logical-devices 2

fail=0
for n in 2 3 4; do
    if cmp -s single.bin "logical${n}.bin"; then
        echo "  [通过] 1 切片 与 ${n} 个逻辑切片 的逐线程分数完全一致"
    else
        echo "  [失败] 1 切片 与 ${n} 个逻辑切片 的分数不一致" >&2
        fail=1
    fi
done
for b in 1 2; do
    if cmp -s "single_b${b}.bin" "logical2_b${b}.bin"; then
        echo "  [通过] ${b} 个批次时 1 切片 与 2 个逻辑切片 的逐线程分数完全一致"
    else
        echo "  [失败] ${b} 个批次时 1 切片 与 2 个逻辑切片 的分数不一致" >&2
        fail=1
    fi
done
if cmp -s early_single.bin early_logical4.bin; then
    echo "  [通过] 提前终止时 1 切片 与 4 个逻辑切片 的逐线程分数完全一致"
else
    echo "  [失败] 提前终止时 1 切片 与 4 个逻辑切片 的分数不一致" >&2
    fail=1
fi
# 提前终止必须真的提前了（否则这条路径没被覆盖）；批次数为 1 时无对比意义，跳过
if [[ "$BATCHES" -ge 2 ]] && [[ "$(wc -c < early_single.bin)" -ge "$(wc -c < single.bin)" ]]; then
    echo "  [失败] 提前终止没有生效，未覆盖 found 路径" >&2
    fail=1
fi

# 结构性校验：记录头、线程数、批次号（批次号必须是 0,1,2,... 的前缀）
if command -v python3 > /dev/null 2>&1; then
    if ! python3 - "$BATCHES" single.bin logical2.bin logical3.bin logical4.bin \
                        early_single.bin early_logical4.bin \
                        single_b1.bin logical2_b1.bin single_b2.bin logical2_b2.bin <<'PY'; then
import struct, sys

batches = int(sys.argv[1])
expected_threads = None
for path in sys.argv[2:]:
    data = open(path, "rb").read()
    offset = 0
    seen = []
    while offset < len(data):
        if offset + 8 > len(data):
            sys.exit(f"[失败] {path}: 记录头不完整")
        index, count = struct.unpack_from("<ii", data, offset)
        offset += 8
        if count <= 0 or offset + 4 * count > len(data):
            sys.exit(f"[失败] {path}: 记录长度非法 (count={count})")
        offset += 4 * count
        if expected_threads is None:
            expected_threads = count
        elif count != expected_threads:
            sys.exit(f"[失败] {path}: 线程数 {count} != {expected_threads}")
        seen.append(index)
    if not seen:
        sys.exit(f"[失败] {path}: 没有任何记录")
    if seen != list(range(len(seen))):
        sys.exit(f"[失败] {path}: 批次号 {seen} 不是 0,1,2,... 的前缀")
    if len(seen) > batches:
        sys.exit(f"[失败] {path}: 记录数 {len(seen)} 超过批次数 {batches}")
    print(f"  [通过] {path}: {len(seen)} 个批次 × {expected_threads} 线程, 结构完整")
PY
        fail=1
    fi
fi

if [[ "$fail" -ne 0 ]]; then
    echo "多 GPU 等价性测试失败" >&2
    exit 1
fi
echo "多 GPU 等价性测试通过"
exit 0
