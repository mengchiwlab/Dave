#!/usr/bin/env python3
"""
AH股IPO评分系统 V6.1 - 新增|Alpha|因子 + IC权重优化
"""

import numpy as np
import pandas as pd

# ========== Spearman ==========
def spearman_corr(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return 0
    xr, yr = df["x"].rank(), df["y"].rank()
    n = len(xr)
    mx, my = xr.mean(), yr.mean()
    cov = ((xr - mx) * (yr - my)).sum() / n
    sx = np.sqrt(((xr - mx) ** 2).sum() / n)
    sy = np.sqrt(((yr - my) ** 2).sum() / n)
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0


# ========== 行业映射 ==========
IND_PATTERNS = {
    "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基|芯碁|峰岹": ("半导体", 1.3),
    "新能源|光伏|储能|宁德|钧达|大金重工|天岳|先导": ("新能源", 1.2),
    "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": ("科技制造", 1.2),
    "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏|牧原": ("制造", 1.1),
    "医药|医疗|生物|恒瑞|可孚|迈威": ("医药", 1.0),
    "新材料|军工|国恩|沃尔|吉宏|赤峰": ("新材料军工", 1.0),
    "饮料|食品|消费|安井|东鹏|海天": ("消费", 0.8),
    "银行|保险|地产|期货": ("金融地产", 0.7),
    "化工|钢铁|纺织|滨化": ("周期", 0.7),
}

def classify_industry(name):
    for p, (label, weight) in IND_PATTERNS.items():
        if any(k in name for k in p.split("|")):
            return label, weight
    return "其他", 1.0


# ========== 平滑评分 ==========
def smooth_score(value, thresholds, scores):
    if value <= thresholds[0]:
        return scores[0]
    if value >= thresholds[-1]:
        return scores[-1]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= value <= thresholds[i + 1]:
            t = (value - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            return scores[i] + t * (scores[i + 1] - scores[i])
    return scores[-1]


# ========== V6.0 评分（基准）==========
def calc_score_v60(row):
    r = row["ret45"]
    b = row.get("beta_real", 1)
    cs = row["cs"]
    p = row.get("pos_real", 0.5)
    a = row.get("alpha_real", 0)
    disc = -row["ha_premium"]
    cap = row.get("cap_real", row.get("cap", None))
    sp = row.get("sp", 2)

    _, ind_beta_weight = classify_industry(row["correct_name"])
    ind_base = 4
    if "半导体" in row["correct_name"] or "芯片" in row["correct_name"] or "集成" in row["correct_name"]:
        ind_base = 5
    elif "新能源" in row["correct_name"] or "光伏" in row["correct_name"]:
        ind_base = 4
    elif "软件" in row["correct_name"] or "通信" in row["correct_name"] or "科技" in row["correct_name"]:
        ind_base = 4
    elif "机器人" in row["correct_name"] or "精密" in row["correct_name"]:
        ind_base = 4
    elif "医药" in row["correct_name"] or "医疗" in row["correct_name"] or "生物" in row["correct_name"]:
        ind_base = 4
    elif "新材料" in row["correct_name"] or "军工" in row["correct_name"]:
        ind_base = 3
    elif "饮料" in row["correct_name"] or "食品" in row["correct_name"] or "消费" in row["correct_name"]:
        ind_base = 2
    elif "银行" in row["correct_name"] or "保险" in row["correct_name"] or "地产" in row["correct_name"]:
        ind_base = 1
    elif "化工" in row["correct_name"] or "钢铁" in row["correct_name"] or "纺织" in row["correct_name"]:
        ind_base = 1
    else:
        ind_base = 2

    s_ret = smooth_score(r, [-30, -15, -5, 5, 15, 30], [1, 4, 8, 12, 15, 15])
    beta_base = 21 if b > 2 else 16 if b > 1.5 else 11 if b > 1 else 6 if b > 0.5 else 3
    direction = (r + 10) / 30
    direction = max(-0.5, min(1.0, direction))
    s_beta_raw = beta_base * direction if direction >= 0 else (21 - beta_base) * abs(direction)
    s_beta = min(21, s_beta_raw * ind_beta_weight)
    s_cs = smooth_score(cs, [0, 20, 30, 40, 50, 70], [1, 4, 7, 10, 13, 13])
    s_pos = smooth_score(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], [2, 4, 6, 8, 10, 8])
    s_vol = smooth_score(abs(a), [0, 0.5, 1.0, 2.0, 3.0], [0.5, 1, 2, 3, 3])
    s_disc = smooth_score(disc, [0, 20, 30, 40, 50, 70], [0.5, 2, 4, 6, 8, 10])
    s_interact = 8 if r > -5 and disc > 40 else 5 if r > -5 and disc > 30 else 3 if r > -15 and disc > 40 else 1 if disc > 30 else 0
    s_ind = ind_base * 3.0
    s_cap = 1 if cap and cap < 200 else 3 if cap and cap < 1000 else 4 if cap and cap < 5000 else 5
    s_sp = sp * 1.33

    return s_ret + s_beta + s_cs + s_pos + s_vol + s_disc + s_interact + s_ind + s_cap + s_sp


# ========== V6.1 评分（|Alpha| + IC权重）==========
def calc_score_v61(row):
    r = row["ret45"]
    b = row.get("beta_real", 1)
    cs = row["cs"]
    p = row.get("pos_real", 0.5)
    a = row.get("alpha_real", 0)
    disc = -row["ha_premium"]
    cap = row.get("cap_real", row.get("cap", None))
    sp = row.get("sp", 2)

    _, ind_beta_weight = classify_industry(row["correct_name"])
    ind_base = 4
    if "半导体" in row["correct_name"] or "芯片" in row["correct_name"] or "集成" in row["correct_name"]:
        ind_base = 5
    elif "新能源" in row["correct_name"] or "光伏" in row["correct_name"]:
        ind_base = 4
    elif "软件" in row["correct_name"] or "通信" in row["correct_name"] or "科技" in row["correct_name"]:
        ind_base = 4
    elif "机器人" in row["correct_name"] or "精密" in row["correct_name"]:
        ind_base = 4
    elif "医药" in row["correct_name"] or "医疗" in row["correct_name"] or "生物" in row["correct_name"]:
        ind_base = 4
    elif "新材料" in row["correct_name"] or "军工" in row["correct_name"]:
        ind_base = 3
    elif "饮料" in row["correct_name"] or "食品" in row["correct_name"] or "消费" in row["correct_name"]:
        ind_base = 2
    elif "银行" in row["correct_name"] or "保险" in row["correct_name"] or "地产" in row["correct_name"]:
        ind_base = 1
    elif "化工" in row["correct_name"] or "钢铁" in row["correct_name"] or "纺织" in row["correct_name"]:
        ind_base = 1
    else:
        ind_base = 2

    # === IC权重分配（基于V6.0回测IC）===
    # 市值: 0.595 | ret45: 0.581 | cs: 0.545 | beta: 0.497 | pos: 0.242 | disc: 0.099 | |alpha|: ~0.3(est)
    # 总分100，按IC比例分配
    
    # 1. A股涨跌幅 (20分) - IC=0.581
    s_ret = smooth_score(r, [-30, -15, -5, 5, 15, 30], [2, 5, 10, 15, 20, 20])

    # 2. 条件Beta (18分) - IC=0.497
    beta_base = 18 if b > 2 else 14 if b > 1.5 else 10 if b > 1 else 5 if b > 0.5 else 2
    direction = (r + 10) / 30
    direction = max(-0.5, min(1.0, direction))
    s_beta_raw = beta_base * direction if direction >= 0 else (18 - beta_base) * abs(direction)
    s_beta = min(18, s_beta_raw * ind_beta_weight)

    # 3. 基石占比 (16分) - IC=0.545
    s_cs = smooth_score(cs, [0, 20, 30, 40, 50, 70], [1, 5, 8, 12, 16, 16])

    # 4. 市值 (16分) - IC=0.595
    s_cap = 2 if cap and cap < 200 else 6 if cap and cap < 1000 else 10 if cap and cap < 5000 else 16

    # 5. 交互项 (12分) - 高IC交互
    s_interact = 12 if r > -5 and disc > 40 else 8 if r > -5 and disc > 30 else 4 if r > -15 and disc > 40 else 1 if disc > 30 else 0

    # 6. 价格位置 (8分) - IC=0.242
    s_pos = smooth_score(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], [2, 4, 5, 6, 8, 6])

    # 7. AH折价率 (5分) - IC=0.099
    s_disc = smooth_score(disc, [0, 20, 30, 40, 50, 70], [0.5, 1, 2, 3, 4, 5])

    # 8. |Alpha| (3分) - 新增因子
    s_abs_alpha = smooth_score(abs(a), [0, 0.5, 1.0, 2.0, 3.0], [0.5, 1, 2, 3, 3])

    # 9. Alpha方向 (1分) - 保留方向信息
    s_alpha = 1 if a > 1 else 0.5 if a > 0 else 0 if a > -0.5 else -0.5

    # 10. 行业 (15分)
    s_ind = ind_base * 3.0

    # 11. 保荐人 (4分)
    s_sp = sp * 1.33

    total = s_ret + s_beta + s_cs + s_pos + s_disc + s_abs_alpha + s_alpha + s_interact + s_ind + s_cap + s_sp
    return total


# ========== 回测引擎 ==========
def run_backtest(df, score_fn, label):
    df = df.copy()
    df["score"] = df.apply(score_fn, axis=1)
    ic = spearman_corr(df["score"], df["return_1d"])
    try:
        df["group"] = pd.qcut(df["score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    except ValueError:
        df["group"] = pd.qcut(df["score"], 3, labels=["Q1", "Q2", "Q3"], duplicates="drop")
    gs = df.groupby("group", observed=False)["return_1d"].agg(["mean", "median", "count"])
    groups = gs.index.tolist()
    ls = gs.loc[groups[-1], "mean"] - gs.loc[groups[0], "mean"] if len(groups) >= 2 else 0
    return {"label": label, "ic": ic, "groups": gs, "long_short": ls, "n": len(df),
            "top_mean": gs.loc[groups[-1], "mean"] if len(groups) >= 1 else 0,
            "bottom_mean": gs.loc[groups[0], "mean"] if len(groups) >= 1 else 0}


def print_results(result):
    print(f"\n{'='*70}")
    print(f"📊 {result['label']}")
    print(f"{'='*70}")
    print(f"  综合评分IC: {result['ic']:+.3f}")
    print(f"  样本数: {result['n']}")
    print(f"\n  分组回测:")
    print(f"  {'分组':<6} {'平均首日涨幅':>12} {'中位数':>10} {'数量':>6}")
    print("  " + "-" * 40)
    for gname, row in result["groups"].iterrows():
        bar = "█" * int(max(0, row["mean"]) / 3) if row["mean"] > 0 else "░" * int(abs(row["mean"]) / 3)
        print(f"  {gname:<4} {row['mean']:>+10.2f}% {row['median']:>+8.2f}% {int(row['count']):>5} {bar}")
    print(f"\n  🔄 多空收益 (Top-Bottom): {result['long_short']:+.2f}%")


def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 V6.1 - |Alpha| + IC权重优化")
    print("=" * 70)

    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    df = df.dropna(subset=["return_1d", "beta_real", "ret45", "ha_premium"])
    print(f"\n📋 有效样本: {len(df)} 只")

    results = []

    r1 = run_backtest(df, calc_score_v60, "V6.0 基准版")
    results.append(r1)
    print_results(r1)

    r2 = run_backtest(df, calc_score_v61, "V6.1 |Alpha|+IC权重")
    results.append(r2)
    print_results(r2)

    print(f"\n{'='*70}")
    print("📈 对比总结")
    print(f"{'='*70}")
    print(f"\n  {'版本':<22} {'IC':>8} {'多空收益':>10} {'Top组平均':>12}")
    print("  " + "-" * 55)
    for r in results:
        marker = "✅" if r["ic"] == max(x["ic"] for x in results) else ""
        print(f"  {r['label']:<20} {r['ic']:>+7.3f}  {r['long_short']:>+8.2f}%   {r['top_mean']:>+9.2f}% {marker}")

    # 具体对比
    df["score_v60"] = df.apply(calc_score_v60, axis=1)
    df["score_v61"] = df.apply(calc_score_v61, axis=1)

    print(f"\n{'='*70}")
    print("🔍 V6.1 Top5 股票")
    print(f"{'='*70}")
    samples = df.nlargest(5, "score_v61")[["correct_name", "ret45", "beta_real", "return_1d", "score_v60", "score_v61"]]
    print(f"\n  {'名称':<10} {'ret45':>8} {'Beta':>6} {'首日':>8} {'V6.0':>6} {'V6.1':>6}")
    for _, row in samples.iterrows():
        print(f"  {row['correct_name']:<8} {row['ret45']:>+7.1f}% {row['beta_real']:>5.2f} {row['return_1d']:>+7.1f}% {row['score_v60']:>6.0f} {row['score_v61']:>6.0f}")

    df.to_csv("output/ah_ipo_backtest_v61.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 回测数据已保存: output/ah_ipo_backtest_v61.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
