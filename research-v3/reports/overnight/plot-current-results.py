"""Plot existing local result records only; never train, predict Q, or use SSH."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "research-v3"
FOLDER = Path(__file__).resolve().parent
OUT = FOLDER / "charts-v1"
OUT.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

snapshot_bytes = (ROOT / "training/overnight/cluster-progress.json").read_bytes()
snapshot = json.loads(snapshot_bytes)
plans = [read(ROOT / "training/overnight" / name)
         for name in ("run-plan.json", "extension-run-plan.json")]
design = read(ROOT / "matrix/design_manifest.json")
conditions = {row["condition_id"]: row for row in design["conditions"]}
selected = {row["entry"]["run_id"]: row for plan in plans for row in plan["runs"]}
assert len(selected) == sum(len(plan["runs"]) for plan in plans) == 201
observed = {row["run_id"]: row for row in snapshot["completed"]}
assert len(observed) == len(snapshot["completed"])
records, pending = [], []

for run_id, row in selected.items():
    entry = row["entry"]
    condition = conditions[entry["condition_id"]]
    old = row["completed_checkpoint_reuse"]
    if old:
        path = ROOT / "runs" / run_id / "status.json"
        assert digest(path) == row["status_sha256"]
        status = read(path)
        assert status["status"] == "complete" and status["reload_verified"]
        metrics = status.get("validation", {})
        seconds = status.get("total_seconds")
        epochs = status.get("epochs_completed")
    elif run_id in observed:
        metrics = observed[run_id]
        seconds, epochs = metrics.get("seconds"), metrics.get("epochs")
    else:
        pending.append(run_id)
        continue
    metric = "macro_ap" if row["dataset_id"] == "go_emotions" else "macro_f1"
    score = metrics.get(metric)
    strict = bool(entry["run"].get("strict_no_val", False))
    if score is None:
        assert strict, (run_id, "Unexplained missing metric")
    else:
        assert np.isfinite(score) and 0 <= score <= 1
    records.append(dict(
        run_id=run_id, dataset=row["dataset_id"], model=row["model_id"],
        n=row["training_rows"], data_seed=entry["data_seed"],
        training_seed=entry["training_seed"], priority=row["priority"],
        condition_id=entry["condition_id"],
        experiment="M2" if entry["run"]["hparams"].get("arm") else condition["experiment_id"],
        aliases=",".join(sorted({a["experiment_id"] for a in entry["semantic_aliases"]})),
        ratio=condition["metadata"].get("requested_ratio"),
        actual_ratio=condition["metadata"].get("actual_max_min_ratio"),
        arm=entry["run"]["hparams"].get("arm"), metric=metric, score=score,
        source="reused" if old else "new_cluster", strict_no_val=strict,
        seconds=seconds, epochs=epochs,
        evidence_role="exploratory_V_selected_checkpoint_not_independent_Q",
    ))

new = [r for r in records if r["source"] == "new_cluster"]
numeric = [r for r in records if r["score"] is not None]
assert len(new) == len(observed)
assert len(records) + len(pending) == len(selected)
assert all(r["strict_no_val"] for r in records if r["score"] is None)
with (FOLDER / "current-validation-results.csv").open(encoding="utf-8-sig", newline="") as f:
    previous = {r["run_id"]: r for r in csv.DictReader(f)}
for r in records:
    if r["run_id"] in previous:
        value = previous[r["run_id"]]["value"]
        assert (value == "" and r["score"] is None) or (
            value != "" and r["score"] is not None and abs(float(value) - r["score"]) < 1e-12)

(OUT / "source-cluster-progress.json").write_bytes(snapshot_bytes)
for name, rows in (("all-completed-records.csv", records), ("new-cluster-records.csv", new)):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(rows)

font_path = Path("C:/Windows/Fonts/msyh.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "figure.facecolor": "#f7f9fc",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#f7f9fc",
    "pdf.fonttype": 42, "axes.titleweight": "bold",
})
DATASETS = ("yelp", "go_emotions", "banking77", "yahoo_answers_topics")
DS = {"yelp": "Yelp · 情感二分类", "go_emotions": "GoEmotions · 28标签情绪",
      "banking77": "BANKING77 · 77类意图", "yahoo_answers_topics": "Yahoo · 10类主题"}
MODEL = {"tfidf_lr": "TF-IDF LR", "tfidf_svm": "TF-IDF SVM", "w2v_svm": "Word2Vec SVM",
         "bilstm": "BiLSTM", "distilbert": "DistilBERT", "modernbert": "ModernBERT",
         "m2_distilbert": "M2 DistilBERT"}
COLORS = {"tfidf_lr": "#7c91a7", "tfidf_svm": "#2774ae", "w2v_svm": "#bd7359",
          "bilstm": "#e5a12a", "distilbert": "#129983", "modernbert": "#8a5bbb"}
ARMS = ("pretrained_frozen", "pretrained_unfrozen", "random_frozen", "random_unfrozen")
ARM = {"pretrained_frozen": "预训练 / 冻结", "pretrained_unfrozen": "预训练 / 解冻",
       "random_frozen": "随机初始化 / 冻结", "random_unfrozen": "随机初始化 / 解冻"}
stamp = datetime.fromisoformat(snapshot["recorded_at_utc"]).astimezone(timezone(timedelta(hours=8)))
STAMP = stamp.strftime("%m月%d日 %H:%M SGT")
FOOT = "验证集 V 探索性结果；已用 V 选择检查点，不能作为独立正式测试集 Q 成绩。"
figures, coverage = [], {}
pdf = PdfPages(OUT / "当前全部训练结果图表.pdf")

def metric(ds):
    return "Macro-AP" if ds == "go_emotions" else "Macro-F1"

def heading(fig, title, subtitle):
    fig.suptitle(title + "\n" + subtitle, fontsize=17, linespacing=1.65)
    fig.supxlabel(FOOT + "  快照：" + STAMP, fontsize=9, color="#536275")

def save(fig, name, title, represented):
    fig.savefig(OUT / (name + ".png"), dpi=160, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    figures.append((name, title))
    coverage[name] = list(dict.fromkeys(represented))
    plt.close(fig)

def one(rows):
    assert len(rows) == 1, len(rows)
    return rows[0]

def primary(r):
    return r["data_seed"] == 2026 and r["training_seed"] in (None, 42)

def label(r):
    m = {"tfidf_svm": "SVM", "bilstm": "BiLSTM", "distilbert": "DistilBERT"}.get(r["model"], MODEL[r["model"]])
    n = f'{r["n"]:,}'
    condition = f'{r["experiment"]} n={n}'
    if r["ratio"] is not None:
        condition += f' r={r["ratio"]}'
    if r["experiment"] == "E4":
        condition += " c=" + r["condition_id"][:5]
    seed = f'd={r["data_seed"]}'
    if r["training_seed"] is not None:
        seed += f'/t={r["training_seed"]}'
    return f"{m} | {condition} | {seed}"

# 01: Every new fit, including all six intentional no-validation runs.
fig, axes = plt.subplots(2, 2, figsize=(21, 16), layout="constrained")
for ax, ds in zip(axes.flat, DATASETS):
    rows = sorted((r for r in new if r["dataset"] == ds),
                  key=lambda r: (r["experiment"], r["n"], r["ratio"] or 0, r["model"], r["data_seed"], r["run_id"]))
    for i, r in enumerate(rows):
        if r["score"] is None:
            ax.axhspan(i - .36, i + .36, color="#edf0f4")
            ax.text(.05, i, "按协议不使用 V · 无验证成绩", va="center", color="#687687", fontsize=9)
        else:
            ax.barh(i, r["score"], height=.65, color=COLORS[r["model"]], alpha=.9)
            ax.text(r["score"] + .012, i, f'{r["score"]:.4f}', va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.09)
    ax.set_xlabel(metric(ds))
    ax.set_title(DS[ds] + f"\n新增完成 {len(rows)} 项，其中 {sum(r['score'] is not None for r in rows)} 项有V成绩")
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
heading(fig, f"刚完成的全部 {len(new)} 项新增训练：逐条结果", "不同条件分行展示；d=数据种子，t=训练种子，r=请求类比例；灰行不代表0分")
save(fig, "01-新增训练逐条结果", "全部新增训练逐条结果", [r["run_id"] for r in new])

# 02: Include ModernBERT, which was outside the five-model main table.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
ids = []
models = tuple(COLORS)
for ax, ds in zip(axes.flat, DATASETS):
    for i, model in enumerate(models):
        hit = [r for r in numeric if r["dataset"] == ds and r["model"] == model
               and r["n"] == 6000 and ("E1" in r["aliases"].split(",") or "E5" in r["aliases"].split(",")) and primary(r)]
        # E5 ModernBERT uses the E1 condition but retains its E5 semantic alias.
        if model != "modernbert":
            hit = [r for r in hit if "E1" in r["aliases"].split(",")]
        if hit:
            r = one(hit)
            ax.bar(i, r["score"], color=COLORS[model], width=.7)
            ax.text(i, r["score"] + .018, f'{r["score"]:.3f}', ha="center", fontsize=10)
            ids.append(r["run_id"])
        else:
            ax.text(i, .04, "未完成/\n未选中", ha="center", color="#7d8995", fontsize=8)
    ax.set_xticks(range(len(models)), [MODEL[m].replace(" ", "\n", 1) for m in models], fontsize=9)
    ax.set_ylim(0, 1.05 if ds != "go_emotions" else .5)
    ax.set_title(DS[ds])
    ax.set_ylabel(metric(ds))
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
heading(fig, "相同 6,000 条训练样本：全部六类模型", "主数据种子2026；神经模型种子42；包含已有ModernBERT补充结果，不同任务指标分别展示")
save(fig, "02-六类模型同样本量比较", "六类模型比较（含ModernBERT）", ids)

# 03: Primary-seed curves, with diamonds identifying new fits.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
ids = []
for ax, ds in zip(axes.flat, DATASETS):
    for model in ("tfidf_svm", "bilstm", "distilbert"):
        rows = sorted((r for r in numeric if r["dataset"] == ds and r["model"] == model
                       and "E1" in r["aliases"].split(",") and primary(r)), key=lambda r: r["n"])
        assert len({r["n"] for r in rows}) == len(rows)
        ax.plot([r["n"] for r in rows], [r["score"] for r in rows], color=COLORS[model],
                linewidth=2, label=MODEL[model])
        for r in rows:
            ax.scatter(r["n"], r["score"], color=COLORS[model],
                       marker="D" if r["source"] == "new_cluster" else "o", s=55, zorder=3)
            ids.append(r["run_id"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_title(DS[ds])
    ax.set_ylabel(metric(ds))
    ax.set_xlabel("训练样本量（对数刻度）")
    ax.grid(alpha=.15)
    ax.legend(loc="lower right", fontsize=8)
heading(fig, "样本量与验证表现", "只连接实际完成点；主种子固定；◆ 新增训练，● 复用；缺失条件不插值")
save(fig, "03-已有学习曲线", "学习曲线", ids)

# 04: Matched E3 contrasts only, never mixed with E1 baselines.
fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
ids = []
for ax, ds in zip(axes, ("yelp", "banking77", "yahoo_answers_topics")):
    for x, model in enumerate(("tfidf_svm", "bilstm", "distilbert")):
        rows = [r for r in numeric if r["dataset"] == ds and r["model"] == model
                and r["experiment"] == "E3" and primary(r)]
        a, b = [one([r for r in rows if r["ratio"] == ratio]) for ratio in (1, 10)]
        ax.bar(x - .19, a["score"], width=.36, color=COLORS[model], alpha=.35)
        ax.bar(x + .19, b["score"], width=.36, color=COLORS[model])
        for offset, r in ((-.19, a), (.19, b)):
            ax.text(x + offset, r["score"] + .015, f'{r["score"]:.3f}', ha="center", fontsize=8)
            ids.append(r["run_id"])
        ax.text(x, 1.01, f'下降 {(a["score"] - b["score"]) * 100:.2f}点', ha="center", fontsize=9,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.04)
    ax.set_xticks(range(3), ["SVM", "BiLSTM", "DistilBERT"])
    ax.set_ylabel("Macro-F1")
    sample = one([r for r in numeric if r["dataset"] == ds and r["model"] == "tfidf_svm"
                  and r["experiment"] == "E3" and r["ratio"] == 10 and primary(r)])
    ax.set_title(DS[ds] + f'\nn={sample["n"]:,}，实际最大/最小类比例={sample["actual_ratio"]:.2f}', pad=37)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
axes[1].legend(handles=[Patch(facecolor="#677c95", alpha=.35, label="均衡训练"),
                       Patch(facecolor="#677c95", label="倾斜训练")], loc="upper right", frameon=False)
heading(fig, "训练分布倾斜后的变化：全部9组匹配对照", "每个数据集内训练总量相同；样本组成也会变化；较小降幅不等于较高绝对性能")
save(fig, "04-不平衡匹配对照", "不平衡对照", ids)

# 05: Every M2 replicate, rather than only means.
fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
ids = []
for i, arm in enumerate(ARMS):
    rows = sorted((r for r in numeric if r["arm"] == arm), key=lambda r: (r["data_seed"], r["training_seed"]))
    values = np.array([r["score"] for r in rows])
    ax.barh(i, values.mean(), color=["#9ac8bc", "#129983", "#c6ccd4", "#7c91a7"][i], alpha=.55, height=.65)
    ax.scatter(values, i + np.linspace(-.19, .19, len(rows)), color="#193e5e", s=42, zorder=3)
    ax.text(values.max() + .02, i, f'均值 {values.mean():.4f}\n范围 {values.min():.4f}–{values.max():.4f}', va="center")
    ids.extend(r["run_id"] for r in rows)
ax.set_yticks(range(4), [ARM[a] for a in ARMS])
ax.invert_yaxis()
ax.set_xlim(0, 1.04)
ax.set_xlabel("V Macro-F1；每个点为一个数据种子×训练种子组合")
ax.grid(axis="x", alpha=.15)
ax.set_axisbelow(True)
heading(fig, "BANKING77机制实验：四臂全部36次结果", "每臂9次，1,000条训练；M2训练配置不同于E1，不能将其成绩直接拼进E1学习曲线")
save(fig, "05-机制实验全部重复", "机制实验36次重复", ids)

# 06: All available numeric results, including E4/E5 and every seed.
EC = {"E1": "#2774ae", "E3": "#d58223", "E4": "#b45483", "E5": "#8a5bbb", "M2": "#129983"}
fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
ids = []
for ax, ds in zip(axes.flat, DATASETS):
    rows = [r for r in numeric if r["dataset"] == ds]
    groups = [m for m in MODEL if m != "m2_distilbert" and any(r["model"] == m for r in rows)]
    groups += ["M2:" + a for a in ARMS if any(r["arm"] == a for r in rows)]
    for i, group in enumerate(groups):
        hits = sorted((r for r in rows if ("M2:" + r["arm"] if r["arm"] else r["model"]) == group),
                      key=lambda r: (r["experiment"], r["n"], r["data_seed"], r["training_seed"] or 0, r["run_id"]))
        offsets = np.linspace(-.27, .27, len(hits)) if len(hits) > 1 else [0]
        for r, offset in zip(hits, offsets):
            color = EC[r["experiment"]]
            ax.scatter(r["score"], i + offset, marker="D" if r["source"] == "new_cluster" else "o",
                       facecolor=color if r["source"] == "new_cluster" else "white", edgecolor=color,
                       linewidth=1.3, s=44, zorder=3)
            ids.append(r["run_id"])
    ax.set_yticks(range(len(groups)), [ARM[g[3:]] if g.startswith("M2:") else MODEL[g] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(-.015, 1.02)
    ax.set_xlabel(metric(ds))
    ax.set_title(DS[ds] + f"\n{len(rows)}条数值成绩；各点可能来自不同样本量/实验条件")
    ax.grid(axis="x", alpha=.15)
handles = [Line2D([], [], marker="o", linestyle="", color=c, label=e) for e, c in EC.items()]
handles += [Line2D([], [], marker="D", linestyle="", color="#455468", label="新增集群训练"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color="#455468", label="已有复用")]
axes.flat[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
heading(fig, f"全部可用验证成绩：{len(numeric)}个点，一个不漏", "颜色为实验条件；垂直错开仅为防重叠，不是误差条；不能跨条件将点云当作模型均值排名")
save(fig, "06-全部验证成绩散点", "全部114条验证成绩", ids)
assert set(ids) == {r["run_id"] for r in numeric}

# 07: Recorded elapsed times for all new fits, not a hardware-controlled benchmark.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
ids = []
for ax, ds in zip(axes.flat, DATASETS):
    rows = [r for r in new if r["dataset"] == ds]
    models = [m for m in COLORS if any(r["model"] == m for r in rows)]
    for i, model in enumerate(models):
        hits = sorted((r for r in rows if r["model"] == model), key=lambda r: r["run_id"])
        for offset, r in zip(np.linspace(-.22, .22, len(hits)) if len(hits) > 1 else [0], hits):
            assert r["seconds"] is not None and r["seconds"] > 0
            ax.scatter(r["seconds"], i + offset, s=55, color=COLORS[model],
                       marker="x" if r["score"] is None else "o")
            ids.append(r["run_id"])
    ax.set_yticks(range(len(models)), [MODEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("记录的单次训练总耗时（秒，对数刻度）")
    ax.set_title(DS[ds] + f"\n{len(rows)}次新增训练；×表示按协议没有V成绩")
    ax.grid(axis="x", alpha=.15)
heading(fig, "刚完成的全部新增训练：记录耗时", "各点的样本量、CPU/GPU和运行条件不相同；本图不能作为统一硬件上的速度排名")
save(fig, "07-新增训练记录耗时", "全部新增训练耗时", ids)

# 08: Coverage makes missing conditions and intentional missing scores explicit.
fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
ax = axes[0]
for i, ds in enumerate(DATASETS):
    old_n = sum(r["dataset"] == ds and r["source"] == "reused" for r in records)
    new_n = sum(r["dataset"] == ds and r["source"] == "new_cluster" and r["score"] is not None for r in records)
    strict_n = sum(r["dataset"] == ds and r["score"] is None for r in records)
    left = 0
    for count, color in ((old_n, "#a7b6c8"), (new_n, "#129983"), (strict_n, "#e5a12a")):
        ax.barh(i, count, left=left, color=color, height=.6)
        if count:
            ax.text(left + count / 2, i, str(count), ha="center", va="center", color="#182b3c")
        left += count
ax.set_yticks(range(4), [DS[ds] for ds in DATASETS])
ax.invert_yaxis()
ax.set_xlabel("已完成的物理训练请求数")
ax.set_title("已完成记录的来源")
ax.legend(handles=[Patch(color="#a7b6c8", label="已有复用"), Patch(color="#129983", label="新增且有V"),
                   Patch(color="#e5a12a", label="新增、严格无V")], loc="lower right", fontsize=9)
ax = axes[1]
for i, (title, scope) in enumerate((("P0", [r for r in plans[0]["runs"] if r["priority"] == 0]),
                                  ("P1", [r for r in plans[0]["runs"] if r["priority"] == 1]),
                                  ("P2", [r for r in plans[0]["runs"] if r["priority"] == 2]),
                                  ("补充批次", plans[1]["runs"]))):
    done = sum(r["entry"]["run_id"] in {v["run_id"] for v in records} for r in scope)
    ax.barh(i, done, color="#129983", height=.6)
    ax.barh(i, len(scope) - done, left=done, color="#e5eaf0", height=.6)
    ax.text(len(scope) + 1, i, f"{done}/{len(scope)}", va="center")
ax.set_yticks(range(4), ["P0", "P1", "P2", "补充批次"])
ax.invert_yaxis()
ax.set_xlim(0, max(len(p["runs"]) for p in plans) + 15)
ax.set_xlabel("选中请求数（包含复用）")
ax.set_title("范围覆盖：已完成 / 选中")
heading(fig, f"数据完整性：选中201项，已完成{len(records)}项，待完成{len(pending)}项",
        f"可画验证成绩 {len(numeric)}项；按协议无V {len(records)-len(numeric)}项；正式Q未运行")
save(fig, "08-完成范围与缺失说明", "完成范围和缺失说明", [r["run_id"] for r in records])
pdf.close()

assert set(coverage["01-新增训练逐条结果"]) == {r["run_id"] for r in new}
assert set(coverage["07-新增训练记录耗时"]) == {r["run_id"] for r in new}
manifest = {
    "snapshot_utc": snapshot["recorded_at_utc"], "snapshot_singapore": stamp.isoformat(),
    "selected": len(selected), "completed": len(records), "new_completed": len(new),
    "reused": sum(r["source"] == "reused" for r in records), "numeric_V": len(numeric),
    "strict_no_V": sum(r["score"] is None for r in records), "pending": len(pending),
    "formal_Q_generated_by_this_script": False, "training_started_by_this_script": False,
    "figure_count": len(figures), "per_figure_run_ids": coverage,
    "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in (
        ROOT / "training/overnight/run-plan.json", ROOT / "training/overnight/extension-run-plan.json",
        ROOT / "matrix/design_manifest.json", OUT / "source-cluster-progress.json")},
    "artifacts": {},
}
for path in sorted(OUT.iterdir()):
    if path.suffix in {".png", ".pdf", ".csv"}:
        manifest["artifacts"][path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
(OUT / "plot-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
index = ["# 当前训练结果图表", "", f"数据快照：{stamp.isoformat()}。",
         f"已完成{len(records)}项：新增{len(new)}项、复用{manifest['reused']}项；其中{len(numeric)}项有V成绩、{manifest['strict_no_V']}项按协议不使用V。",
         "", "全部图均为验证集探索性结果，正式Q没有运行。", "",
         "[下载8页完整图册](当前全部训练结果图表.pdf)", "",
         "[全部已完成数据](all-completed-records.csv) · [新增训练数据](new-cluster-records.csv) · [来源与逐图覆盖记录](plot-manifest.json)", ""]
for name, title in figures:
    index += ["## " + title, "", f"![{title}]({name}.png)", ""]
(OUT / "README.md").write_text("\n".join(index), encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k not in {"per_figure_run_ids", "source_sha256", "artifacts"}}, ensure_ascii=False))
