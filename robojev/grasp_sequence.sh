#!/bin/bash
# 按顺序：抓 -> 放下 -> 抓下一个。每个任务统计抓取时间（相机就绪 -> 抬起结束，不含放下）。
#
#   conda activate laya
#   cd ~/yuhang_workspace/算法/Jev/RoboJev
#   bash robojev/grasp_sequence.sh                                   # 默认：乒乓球 -> 水瓶 -> 蓝色圆柱
#   bash robojev/grasp_sequence.sh "pick up the X" "pick up the Y"   # 自己给顺序
#
# 前提：臂已上电；PlayJev 服务开着（见 COMPOSITE_PIPELINE.md）。
# 结果：logs/seq_<时间>/ 下每个任务一份日志 + 录像，summary.txt 是汇总表。
# 设备被别的进程占着（比如 collect）就停下，不杀进程。

set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/laya/bin/python}
export PYTHONPATH=                       # ROS 往 PYTHONPATH 塞的包会让导入报错

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=("pick up the ping pong ball" "pick up the water bottle" "pick up the blue cylinder")

RUN=logs/seq_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
SUMMARY=$RUN/summary.txt
printf "%-4s %-30s %-6s %-10s %-10s %-8s %s\n" "序号" "指令" "结果" "抓取时间" "进程总时间" "伺服步数" "夹爪读数(实/应)" > "$SUMMARY"

busy() {
    for d in /dev/ttyUSB* /dev/ttyACM* /dev/video*; do
        [ -e "$d" ] || continue
        if fuser "$d" >/dev/null 2>&1; then
            echo "  $d 被占用：$(fuser -v "$d" 2>&1 | tail -n +2 | tr -s ' ' | head -2 | tr '\n' ' ')"
            return 0
        fi
    done
    return 1
}

echo "顺序：$(printf '%s -> ' "${TASKS[@]}")结束"
echo "输出：$RUN"
if busy; then echo "设备被占用，不开始"; exit 1; fi
echo "回到初始状态（0 位、张爪）"
timeout 200 "$PY" -m robojev.place_and_home --no-place --lift 0.0 2>&1 | grep "回 0 位"

for i in "${!TASKS[@]}"; do
    INS=${TASKS[$i]}
    K=$((i + 1))
    TAG=${K}_$(echo "$INS" | sed 's/^pick up the //; s/ /_/g')
    LOG=$RUN/$TAG.log
    echo
    echo "===== [$K/${#TASKS[@]}] $INS ====="
    if busy; then echo "设备被占用，停止（不杀进程）"; break; fi

    T0=$(date +%s.%N)
    timeout 590 "$PY" -m robojev.run_composite --instruction "$INS" --stage grasp --record "$RUN/$TAG" > "$LOG" 2>&1
    T1=$(date +%s.%N)

    grep -E "俯视里物体像素|交接|俯视里已对准|抓取高度冻结|到位|步数用完|看不到目标|够不着|跟不上|宽度 .*应读|Error|错误" "$LOG" | grep -v "^W\|WARN" | tail -8
    STEPS=$(grep -c '^\[[ 0-9]*\] 中心' "$LOG")
    RESULT=$(grep -q -- "-> 成功" "$LOG" && echo 成功 || echo 失败)
    GRIP=$(grep -o "实读 [0-9.]*" "$LOG" | tail -1 | awk '{print $2}')
    EXPECT=$(grep -o "应读 [0-9.]*" "$LOG" | tail -1 | awk '{print $2}')
    TIMES=$("$PY" - "$RUN/$TAG" "$T0" "$T1" <<'PYEOF'
import glob, os, sys
rec, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
fs = sorted(glob.glob(rec + "/*.jpg"))
lift = [f for f in fs if f.endswith("_lift.jpg")]
g = f"{os.path.getmtime(lift[-1]) - os.path.getmtime(fs[0]):.1f}s" if fs and lift else "—"
print(g, f"{t1 - t0:.1f}s")
PYEOF
)
    GT=${TIMES% *}; TT=${TIMES#* }
    # 程序中途报错分两种：硬件/环境问题（相机打不开、臂没上电、急停、夹爪没反馈）后面的任务也不会好，停；
    # 这个任务自己的过程出错（比如灯光闪导致测不出俯视换算）只算这一项失败，接着做下一个
    if ! grep -q "应读" "$LOG" && grep -q "Traceback\|Error" "$LOG"; then
        WHY=$(grep -E "^[A-Za-z]*Error" "$LOG" | tail -1)
        printf "%-4s %-30s %-6s %-10s %-10s %-8s %s\n" "$K" "$INS" "出错" "—" "$TT" "$STEPS" "${WHY:0:60}" >> "$SUMMARY"
        timeout 200 "$PY" -m robojev.place_and_home --no-place --lift 0.07 2>&1 | grep "回 0 位"
        if grep -q "打不开\|NOT powered\|急停\|夹爪反馈过期\|夹爪通信故障" "$LOG"; then
            echo "硬件/环境问题（$WHY），停止后续任务"
            break
        fi
        echo "这个任务出错（$WHY），记为失败，继续下一个"
        continue
    fi
    echo "结果 $RESULT  抓取时间 $GT（相机就绪 -> 抬起结束）  进程总时间 $TT  伺服 $STEPS 步"
    printf "%-4s %-30s %-6s %-10s %-10s %-8s %s\n" "$K" "$INS" "$RESULT" "$GT" "$TT" "$STEPS" "${GRIP:-—}/${EXPECT:-—}" >> "$SUMMARY"

    # 放下（不计时）：夹爪没合死 = 手里可能有东西，降回抓取高度再张爪；否则直接回 0 位
    if [ -n "$GRIP" ] && "$PY" -c "import sys; sys.exit(0 if float('$GRIP') < 0.97 else 1)"; then
        timeout 200 "$PY" -m robojev.place_and_home --down 0.06 2>&1 | grep "回 0 位"
    else
        timeout 200 "$PY" -m robojev.place_and_home --no-place --lift 0.07 2>&1 | grep "回 0 位"
    fi
done

echo
echo "===== 汇总（$SUMMARY）====="
cat "$SUMMARY"
