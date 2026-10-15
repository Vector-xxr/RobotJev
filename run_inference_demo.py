"""Laya 推理冒烟测试：Router 自动路由 + 单模型直调 + 延迟测量。

跑法：
    conda activate laya
    python run_inference_demo.py
"""

import json
import os
import time

import torch

import laya
from laya import Router

STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

# 三条不同语言的 state，用来看 Router 怎么选 checkpoint
MULTILINGUAL = [
    ("英文", {"body": "Please refund the duplicate charge on invoice 4411."}),
    ("中文", {"body": "你们三月份重复扣了我两次款，今天不退款我们就取消订阅。"}),
    ("印地语", {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}),
]


def summarise(res):
    a = res["answers"]
    return (
        f"department={a['department']['choice']} ({a['department']['confidence']:.2f})  "
        f"urgency={a['urgency']['score']:.2f}/2  "
        f"churn={a['churn_risk']['noul']:.3f}  "
        f"refund={a['refund_requested']['noul']:.3f}"
    )


def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("laya", laya.__version__)

    # ---- 1) Router：三个 checkpoint 全部常驻 ----
    t0 = time.time()
    # 三个 checkpoint 优先用项目里的 weights/laya（拷走工作区就能断网跑），没有才从 HuggingFace 取
    w = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "laya")
    local = os.path.isdir(w)
    models = ({"english": (w, None), "multilingual": (w, "multilingual"),
               "typed-decisions": (w, "typed-decisions")} if local else None)
    router = Router(preload=True, models=models)
    print(f"\n[Router] preload 三个 checkpoint 用时 {time.time() - t0:.1f}s")

    for name, state in MULTILINGUAL:
        decision = router.route(state, QUESTIONS)          # 只做路由，不跑前向
        t0 = time.time()
        res = router.predict(state, QUESTIONS)
        dt = (time.time() - t0) * 1000
        print(f"  {name:4s} -> {res['routing']['model']:12s} {dt:6.1f} ms  "
              f"| {summarise(res)}")
        print(f"        理由: {decision.reason}")

    # ---- 2) 单模型直调 ----
    print("\n[单模型] convaiinnovations/laya (英文 ModernBERT-large)")
    agent = laya.load(w if local else "convaiinnovations/laya")
    res = agent.predict(STATE, QUESTIONS)
    print("  ", summarise(res))
    print("  ", json.dumps(res["usage"]))

    # ---- 3) 热路径延迟 ----
    for _ in range(3):
        agent.predict(STATE, QUESTIONS)
    runs = []
    for _ in range(20):
        t0 = time.time()
        agent.predict(STATE, QUESTIONS)
        runs.append((time.time() - t0) * 1000)
    runs.sort()
    print(f"\n[延迟] 4 个问题单次前向 x20：中位 {runs[10]:.1f} ms  "
          f"min {runs[0]:.1f}  p95 {runs[18]:.1f}")


if __name__ == "__main__":
    main()
