#!/usr/bin/env python3
"""
RobotJev 工作站代理 —— 网页「实时推理测试」页和 RobotJev 那台机器之间的桥。

和抓取的 atom_agent.py 是**两个代理**：端口不同（8766）、令牌不同，各管各的。
这一个只做两件事：
  * 点选抓取：拉起 robojev.click_grasp_web，把画面 / 状态 / 鼠标命令在网页和它之间转
  * 恢复机械臂：python -m atom_robot.ar5.unseize [--recover --home]

    conda activate laya
    python robojev_agent.py --token <网页上给的令牌> --site http://<网站地址>

只绑 127.0.0.1，请求要带令牌、Origin 要是网站自己 —— 网页要从这台机器自己的浏览器打开。
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "1.0"
PROFILE = "robojev"                 # 网页拿它核对：连上的是不是 RobotJev 代理
PROFILE_NAME = "RobotJev 代理"

CFG = {}
JOBS = {}
JOB_LOCK = threading.Lock()
SEQ_LOCK = threading.Lock()
MAX_LOG_LINES = 3000

# 和网页交换文件的目录。固定死，网页给不了别的路径 —— /infer 接口只碰这里
INFER_DIR = os.path.expanduser("~/.atom_agent/robojev_click")
INFER_CMDS = ("click", "drag", "exclude", "cycle", "reset", "resnap", "confirm", "cancel")

# 白名单。{robojev_dir} / {infer_dir} 由代理自己填，网页给了也不用。
# PYTHONPATH= 照 click_grasp.py 文档里的跑法：laya 环境里不能带别的包路径
TEMPLATES = {
    "robojev_click":     {"cwd": "robojev",
                          "cmd": "PYTHONPATH= python -m robojev.click_grasp_web --link {infer_dir}"},
    "robojev_click_dry": {"cwd": "robojev",
                          "cmd": "PYTHONPATH= python -m robojev.click_grasp_web --link {infer_dir} --dry"},
    # 恢复机械臂：和 collect 启动时那套一样（diagnose → 上电 → 回 0 位）。在 算法/DexGraspVLA-/SO101 下跑
    "unseize":           {"cwd": "workdir", "cmd": "PYTHONPATH= python -m atom_robot.ar5.unseize"},
    "unseize_recover":   {"cwd": "workdir", "cmd": "PYTHONPATH= python -m atom_robot.ar5.unseize --recover --home"},
}


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def find_conda_hook():
    """conda.sh 在哪。bash -lc 里 conda activate 默认用不了，得先 source 它。"""
    if CFG.get("conda_sh"):
        return CFG["conda_sh"] if os.path.isfile(CFG["conda_sh"]) else ""
    cands = []
    exe = os.environ.get("CONDA_EXE", "")
    if exe:
        cands.append(os.path.join(os.path.dirname(os.path.dirname(exe)), "etc", "profile.d", "conda.sh"))
    base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX", "")
    if base:
        cands.append(os.path.join(base, "etc", "profile.d", "conda.sh"))
    home = os.path.expanduser("~")
    for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge", "conda"):
        cands.append(os.path.join(home, d, "etc", "profile.d", "conda.sh"))
    cands += ["/opt/conda/etc/profile.d/conda.sh"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


def pick_workdir(asked):
    """网页带来的工作目录（atom_robot 所在，算法/DexGraspVLA-/SO101）优先；没给或不存在就用启动时的。"""
    fallback = os.path.expanduser(CFG["workdir"])
    asked = os.path.expanduser(str(asked or "").strip())
    if asked and os.path.isdir(asked):
        return asked, ""
    if asked:
        return fallback, "网页给的工作目录不存在：%s —— 用代理启动时的 %s" % (asked, fallback)
    return fallback, ""


def build_command(key, workdir=None):
    tpl = TEMPLATES.get(key)
    if not tpl:
        raise ValueError("RobotJev 代理不跑「%s」" % key)
    inner = tpl["cmd"].format(infer_dir=shlex.quote(INFER_DIR))
    wd, note = pick_workdir(workdir)
    cwd = os.path.expanduser(CFG["robojev_dir"]) if tpl["cwd"] == "robojev" else wd
    hook = find_conda_hook()
    prefix = (". %s && " % shlex.quote(hook)) if hook else "[ -f ~/.bashrc ] && . ~/.bashrc; "
    cmd = "%sconda activate %s && cd %s && %s" % (prefix, shlex.quote(CFG["env"]), shlex.quote(cwd), inner)
    return cmd, cwd, note


def start_job(key, workdir=None):
    cmd, cwd, note = build_command(key, workdir)
    jid = "%s_%s" % (key, datetime.now().strftime("%H%M%S"))
    job = {"id": jid, "key": key, "cmd": cmd, "gui": False,
           "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "status": "running", "exit_code": None, "lines": [], "proc": None}
    if note:
        job["lines"].append("[agent] " + note)
    with JOB_LOCK:
        JOBS[jid] = job

    def run():
        try:
            proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                                    cwd=cwd if os.path.isdir(cwd) else None)
            job["proc"] = proc
            for line in proc.stdout:
                with JOB_LOCK:
                    for piece in line.rstrip("\n").split("\r"):     # \r 刷新的状态行拆成一行行
                        if piece.rstrip():
                            job["lines"].append(piece.rstrip())
                    if len(job["lines"]) > MAX_LOG_LINES:
                        del job["lines"][:-MAX_LOG_LINES]
            proc.wait()
            job["exit_code"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "failed"
        except Exception as exc:                                    # noqa: BLE001
            job["status"] = "failed"
            job["lines"].append("[agent] 启动失败: %s" % exc)
        finally:
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    threading.Thread(target=run, daemon=True).start()
    return job


def job_view(job):
    return {"id": job["id"], "key": job["key"], "status": job["status"], "exit_code": job["exit_code"],
            "gui": False, "started_at": job["started_at"], "finished_at": job.get("finished_at"),
            "cmd": job["cmd"], "lines": job["lines"][-120:]}


def click_job():
    """正在跑的点选抓取（没有就 None）。同时只许一个：335 只能被一个进程打开。"""
    for j in reversed(list(JOBS.values())):
        if j["key"] in ("robojev_click", "robojev_click_dry") and j["status"] == "running":
            return j
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass                                                        # 轮询太密，不刷屏

    # ---- 安全 ----
    def _origin_ok(self, origin):
        if not origin:
            return True                                             # 同源 / 非浏览器
        if origin in CFG["allow_origins"]:
            return True
        return bool(re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin))

    def _guard(self, data=None):
        origin = self.headers.get("Origin", "")
        if not self._origin_ok(origin):
            self._send({"error": "来源不被允许: %s" % origin}, 403)
            return True
        token = (data or {}).get("token") or self.headers.get("X-Agent-Token", "") \
            or (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
        if token != CFG["token"]:
            self._send({"error": "令牌不正确"}, 401)
            return True
        return False

    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin and self._origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Agent-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Vary", "Origin")

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:                                           # noqa: BLE001
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    # ---- GET ----
    def do_GET(self):
        path = urlparse(self.path).path
        if self._guard():
            return
        if path == "/ping":
            self._send({"ok": True, "agent": VERSION, "profile": PROFILE, "profile_name": PROFILE_NAME,
                        "workdir": CFG["workdir"], "robojev_dir": CFG["robojev_dir"], "env": CFG["env"],
                        "commands": sorted(TEMPLATES), "conda_sh": find_conda_hook() or "(未找到，退回 ~/.bashrc)",
                        "home": os.path.expanduser("~"),
                        "jobs": [job_view(j) for j in list(JOBS.values())[-5:]]})
            return
        if path.startswith("/job/"):
            job = JOBS.get(path[5:])
            self._send(job_view(job) if job else {"error": "任务不存在"}, 200 if job else 404)
            return
        if path == "/infer/state":
            try:
                with open(os.path.join(INFER_DIR, "state.json"), "r", encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:                                       # noqa: BLE001
                st = {}
            job = click_job()
            self._send({"ok": True, "state": st, "running": job is not None,
                        "job": job_view(job) if job else None,
                        "age": (time.time() - float(st.get("ts", 0))) if st else None})
            return
        if path == "/infer/view":
            p = os.path.join(INFER_DIR, "view.jpg")
            if not os.path.isfile(p):
                self._send({"error": "还没有画面"}, 404)
                return
            with open(p, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send({"error": "RobotJev 代理没有 %s" % path}, 404)

    # ---- POST ----
    def do_POST(self):
        path = urlparse(self.path).path
        data = self._body()
        if self._guard(data):
            return

        if path == "/run":                                          # 恢复机械臂走这里
            key = str(data.get("step", ""))
            if key not in ("unseize", "unseize_recover"):
                self._send({"error": "RobotJev 代理不跑「%s」" % key}, 400)
                return
            job = start_job(key, data.get("workdir"))
            log("执行 %s" % job["cmd"])
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/stop":
            job = JOBS.get(str(data.get("id", "")))
            ok = False
            if job and job.get("proc"):
                try:
                    job["proc"].terminate()
                    job["status"] = "stopped"
                    ok = True
                except Exception:                                   # noqa: BLE001
                    pass
            self._send({"ok": ok})
            return

        if path == "/infer/start":
            if click_job():
                self._send({"error": "已经有一个点选抓取在跑了 —— 335 只能被一个进程打开。先在网页上退出它"}, 409)
                return
            os.makedirs(INFER_DIR, exist_ok=True)
            for name in ("state.json", "view.jpg"):                 # 别让网页先看到上一轮的「已结束」
                try:
                    os.remove(os.path.join(INFER_DIR, name))
                except OSError:
                    pass
            job = start_job("robojev_click_dry" if data.get("dry") else "robojev_click", data.get("workdir"))
            log("点选抓取%s → %s" % ("（只看判定）" if data.get("dry") else "", job["cmd"]))
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/infer/cmd":
            cmd = str(data.get("cmd", ""))
            if cmd not in INFER_CMDS:
                self._send({"error": "不认识的命令：%s" % cmd}, 400)
                return
            if not click_job():
                self._send({"error": "点选抓取没在跑"}, 409)
                return
            doc = {"cmd": cmd, "ts": time.time(), "shift": bool(data.get("shift"))}
            for k in ("x", "y", "x0", "y0", "x1", "y1", "d"):
                if k in data:
                    try:
                        doc[k] = float(data[k])
                    except (TypeError, ValueError):
                        self._send({"error": "%s 不是数字" % k}, 400)
                        return
            path_c = os.path.join(INFER_DIR, "control.json")
            with SEQ_LOCK:                                          # seq 递增，脚本按它去重
                try:
                    with open(path_c, "r", encoding="utf-8") as f:
                        seq = int(json.load(f).get("seq", 0))
                except Exception:                                   # noqa: BLE001
                    seq = 0
                doc["seq"] = seq + 1
                with open(path_c + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(doc, f)
                os.replace(path_c + ".tmp", path_c)
            self._send({"ok": True, "seq": doc["seq"]})
            return

        self._send({"error": "RobotJev 代理没有 %s" % path}, 404)


def _workspace():
    """工作区根：从本文件往上找同时有 算法/ 和 数据_硬件/ 的目录，不写死本机绝对路径；
    单独拷走时用环境变量 YUHANG_WS 指过去。"""
    env = os.environ.get("YUHANG_WS")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "算法")) and os.path.isdir(os.path.join(d, "数据_硬件")):
            return d
        d = os.path.dirname(d)
    # 找不到就按现在的层级推：算法/Jev/RobotJev/robojev_agent.py → 往上 4 级是工作区根
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    p = argparse.ArgumentParser(description="RobotJev 工作站代理")
    p.add_argument("--token", required=True, help="和网站上 RobotJev 代理的令牌一致")
    p.add_argument("--site", default="http://192.168.5.100:8090", help="网站地址（只用来放行它的请求）")
    p.add_argument("--port", type=int, default=8766, help="本机监听端口；抓取代理是 8765，别撞")
    p.add_argument("--workdir", default=os.path.join(_workspace(), "算法", "DexGraspVLA-", "SO101"),
                   help="恢复机械臂在这里跑（python -m atom_robot.ar5.unseize）；网页给了以网页为准")
    p.add_argument("--robojev-dir", default=os.path.dirname(os.path.abspath(__file__)),   # 本文件就在仓库根目录
                   help="RobotJev 仓库目录（python -m robojev.click_grasp_web 在这里跑）")
    p.add_argument("--env", default="laya", help="conda 环境")
    p.add_argument("--conda-sh", default="", help="conda.sh 路径，自动探测不到时手动指定")
    a = p.parse_args()
    CFG.update({"token": a.token, "site": a.site, "port": a.port, "workdir": a.workdir,
                "robojev_dir": a.robojev_dir, "env": a.env, "conda_sh": a.conda_sh,
                "allow_origins": {a.site.rstrip("/")}})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)      # 只绑本机
    log("RobotJev 工作站代理 v%s" % VERSION)
    log("监听 http://127.0.0.1:%d  （只接受本机连接）" % a.port)
    log("网站 %s" % a.site)
    log("RobotJev 仓库 %s   环境 %s" % (a.robojev_dir, a.env))
    rj = os.path.expanduser(a.robojev_dir)
    if not os.path.isfile(os.path.join(rj, "robojev", "click_grasp_web.py")):
        log("⚠ %s/robojev/ 下没有 click_grasp_web.py —— 从网页上下载放进去，点选抓取才能用" % rj)
    log("按 Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("退出")


if __name__ == "__main__":
    sys.exit(main())
