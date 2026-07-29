#!/usr/bin/env python3
"""
AH股IPO评分系统 - 条件Beta vs 原始Beta 对比回测
"""

import os
from datetime import datetime

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


# ========== 原始评分 V5.8 ==========
def calc_score_original(row):
    r = row["ret45"]
    b = row.get("beta_real", row.get("beta", 1))
    cs = row["cs"]
    p = row.get("pos_real", row.get("pos", 0.5))
    a = row.get("alpha_real", row.get("alpha", 0))
    disc = -row["ha_premium"]
    ind = row.get("ind", 2)
    cap = row.get("cap_real", row.get("cap", None))
    sp = row.get("sp", 2)

    s_ret = 20 if r > 15 else 16 if r > 5 else 12 if r > -5 else 8 if r > -15 else 4
    s_beta = 15 if b > 2 else 12 if b > 1.5 else 9 if b > 1 else 6 if b > 0.5 else 3
    s_cs = 15 if cs > 50 else 12 if cs > 40 else 8 if cs > 30 else 4
    s_pos = 12 if p > 0.8 else 10 if p > 0.6 else 7 if p > 0.4 else 4 if p > 0.2 else 2
    s_alpha = 12 if a > 1 else 9 if a > 0.5 else 6 if a > 0 else 3 if a > -0.5 else 0
    s_disc = 12 if disc > 50 else 9 if disc > 40 else 6 if disc > 30 else 3 if disc > 20 else 1
    s_ind = ind
    s_cap = 4 if cap and cap < 200 else 3 if cap and cap < 1000 else 2 if cap and cap < 5000 else 1
    s_sp = sp

    total = s_ret + s_beta + s_cs + s_pos + s_alpha + s_disc + s_ind + s_cap + s_sp
    return total


# ========== 条件Beta评分 ==========
def calc_beta_score_conditional(beta, ret45):
    """
    Beta 评分：方向依赖
    市场涨时高 Beta 是优点，市场跌时高 Beta 是缺点
    """
    if ret45 >= 5:  # A股大涨，高 Beta 放大涨幅 → 加分
        return 21 if beta > 2 else 16 if beta > 1.5 else 11 if beta > 1 else 6 if beta > 0.5 else 3
    elif ret45 >= -5:  # 震荡市，中性看待
        return 15 if beta > 2 else 12 if beta > 1.5 else 9 if beta > 1 else 5 if beta > 0.5 else 2
    else:  # A股大跌，高 Beta 放大跌幅 → 减分（反向）
        # beta 越高，得分越低
        return 3 if beta > 2 else 6 if beta > 1.5 else 9 if beta > 1 else 12 if beta > 0.5 else 15


def calc_score_conditional(row):
    r = row["ret45"]
    b = row.get("beta_real", row.get("beta", 1))
    cs = row["cs"]
    p = row.get("pos_real", row.get("pos", 0.5))
    a = row.get("alpha_real", row.get("alpha", 0))
    disc = -row["ha_premium"]
    ind = row.get("ind", 2)
    cap = row.get("cap_real", row.get("cap", None))
    sp = row.get("sp", 2)

    s_ret = 17 if r > 15 else 14 if r > 5 else 10 if r > -5 else 6 if r > -15 else 3
    s_beta = calc_beta_score_conditional(b, r)  # 条件Beta
    s_cs = 15 if cs > 50 else 12 if cs > 40 else 8 if cs > 30 else 4
    s_pos = 10 if p > 0.8 else 8 if p > 0.6 else 6 if p > 0.4 else 3 if p > 0.2 else 1
    s_alpha = 2 if a > 1 else 1.5 if a > 0.5 else 1 if a > 0 else 0.5 if a > -0.5 else 0
    s_disc = 9 if disc > 50 else 7 if disc > 40 else 5 if disc > 30 else 2 if disc > 20 else 1
    s_ind = ind * 3.4
    s_cap = 4 if cap and cap < 200 else 3 if cap and cap < 1000 else 2 if cap and cap < 5000 else 1
    s_sp = sp * 1.67

    total = s_ret + s_beta + s_cs + s_pos + s_alpha + s_disc + s_ind + s_cap + s_sp
    return total


# ========== 更平滑的条件Beta ==========
def calc_beta_score_smooth(beta, ret45):
    """
    平滑连续的条件Beta评分
    """
    # 基础 Beta 分（正向）
    base = 21 if beta > 2 else 16 if beta > 1.5 else 11 if beta > 1 else 6 if beta > 0.5 else 3
    
    # 方向修正系数
    # ret45 映射到 -0.5 ~ 1.0 范围
    direction = (ret45 + 10) / 30
    direction = max(-0.5, min(1.0, direction))
    
    if direction >= 0:
        return base * direction
    else:
        # 反向：ret45 跌，高 Beta 减分
        reverse = 21 - base  # beta 越高，reverse 越小（扣分越多）
        return reverse * abs(direction)


def calc_score_smooth(row):
    r = row["ret45"]
    b = row.get("beta_real", row.get("beta", 1))
    cs = row["cs"]
    p = row.get("pos_real", row.get("pos", 0.5))
    a = row.get("alpha_real", row.get("alpha", 0))
    disc = -row["ha_premium"]
    ind = row.get("ind", 2)
    cap = row.get("cap_real", row.get("cap", None))
    sp = row.get("sp", 2)

    s_ret = 17 if r > 15 else 14 if r > 5 else 10 if r > -5 else 6 if r > -15 else 3
    s_beta = calc_beta_score_smooth(b, r)  # 平滑条件Beta
    s_cs = 15 if cs > 50 else 12 if cs > 40 else 8 if cs > 30 else 4
    s_pos = 10 if p > 0.8 else 8 if p > 0.6 else 6 if p > 0.4 else 3 if p > 0.2 else 1
    s_alpha = 2 if a > 1 else 1.5 if a > 0.5 else 1 if a > 0 else 0.5 if a > -0.5 else 0
    s_disc = 9 if disc > 50 else 7 if disc > 40 else 5 if disc > 30 else 2 if disc > 20 else 1
    s_ind = ind * 3.4
    s_cap = 4 if cap and cap < 200 else 3 if cap and cap < 1000 else 2 if cap and cap < 5000 else 1
    s_sp = sp * 1.67

    total = s_ret + s_beta + s_cs + s_pos + s_alpha + s_disc + s_ind + s_cap + s_sp
    return total


# ========== 行业映射 ==========
def get_industry_score(name):
    patterns = {
        "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基|芯碁|峰岹": 5,
        "新能源|光伏|储能|宁德|钧达|大金重工|天岳|先导": 4,
        "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": 4,
        "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏|牧原": 4,
        "医药|医疗|生物|恒瑞|可孚|迈威": 4,
        "新材料|军工|国恩|沃尔|吉宏|赤峰": 3,
        "饮料|食品|消费|安井|东鹏|海天": 2,
        "银行|保险|地产|期货": 1,
        "化工|钢铁|纺织|滨化": 1,
    }
    for p, s in patterns.items():
        if any(k in name for k in p.split("|")):
            return s
    return 2


# ========== 回测引擎 ==========
def run_backtest(df, score_col, label):
    df = df.copy()
    df["score"] = df.apply(score_col, axis=1)

    # IC
    ic = spearman_corr(df["score"], df["return_1d"])

    # 分组回测
    try:
        df["group"] = pd.qcut(df["score"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"], duplicates="drop")
    except ValueError:
        # 如果分不了5组，尝试用更少组
        df["group"] = pd.qcut(df["score"], 3, labels=["Q1(最低)", "Q2", "Q3(最高)"], duplicates="drop")

    gs = df.groupby("group", observed=False)["return_1d"].agg(["mean", "median", "std", "count"])
    groups = gs.index.tolist()

    if len(groups) >= 2:
        ls = gs.loc[groups[-1], "mean"] - gs.loc[groups[0], "mean"]
    else:
        ls = 0

    return {
        "label": label,
        "ic": ic,
        "groups": gs,
        "long_short": ls,
        "n": len(df),
        "top_mean": gs.loc[groups[-1], "mean"] if len(groups) >= 1 else 0,
        "bottom_mean": gs.loc[groups[0], "mean"] if len(groups) >= 1 else 0,
    }


def print_results(result):
    print(f"\n{'='*70}")
    print(f"📊 {result['label']}")
    print(f"{'='*70}")
    print(f"  综合评分IC: {result['ic']:+.3f}")
    print(f"  样本数: {result['n']}")
    print(f"\n  分组回测:")
    print(f"  {'分组':<10} {'平均首日涨幅':>12} {'中位数':>10} {'标准差':>8} {'数量':>6}")
    print("  " + "-" * 60)
    for gname, row in result["groups"].iterrows():
        bar = "█" * int(max(0, row["mean"]) / 3) if row["mean"] > 0 else "░" * int(abs(row["mean"]) / 3)
        print(f"  {gname:<10} {row['mean']:>+10.2f}% {row['median']:>+8.2f}% {row['std']:>7.2f}% {int(row['count']):>5} {bar}")
    print(f"\n  🔄 多空收益 (Top-Bottom): {result['long_short']:+.2f}%")


def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 - 条件Beta vs 原始Beta 对比回测")
    print("=" * 70)

    # 读取数据
    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    # 过滤掉没有首日涨幅的
    df = df.dropna(subset=["return_1d"])
    # 过滤掉没有beta_real的（部分数据缺失）
    df = df.dropna(subset=["beta_real"])
    print(f"\n📋 加载 {len(df)} 只有效IPO数据（含Beta数据）")

    # 补充行业得分
    df["ind"] = df["correct_name"].apply(get_industry_score)
    # 补充保荐人默认分
    df["sp"] = df.get("sp", 2)

    # 运行三个版本
    results = []

    # 1. 原始版本
    r1 = run_backtest(df, calc_score_original, "原始评分 (V5.8)")
    results.append(r1)
    print_results(r1)

    # 2. 阶梯条件Beta
    r2 = run_backtest(df, calc_score_conditional, "条件Beta (阶梯)")
    results.append(r2)
    print_results(r2)

    # 3. 平滑条件Beta
    r3 = run_backtest(df, calc_score_smooth, "条件Beta (平滑)")
    results.append(r3)
    print_results(r3)

    # 对比总结
    print(f"\n{'='*70}")
    print("📈 对比总结")
    print(f"{'='*70}")
    print(f"\n  {'版本':<20} {'IC':>8} {'多空收益':>10} {'Top组平均':>12}")
    print("  " + "-" * 55)
    for r in results:
        marker = "✅" if r["ic"] == max(x["ic"] for x in results) else ""
        print(f"  {r['label']:<18} {r['ic']:>+7.3f}  {r['long_short']:>+8.2f}%   {r['top_mean']:>+9.2f}% {marker}")

    # 一些具体案例对比
    print(f"\n{'='*70}")
    print("🔍 典型股票评分对比")
    print(f"{'='*70}")

    df["score_orig"] = df.apply(calc_score_original, axis=1)
    df["score_cond"] = df.apply(calc_score_conditional, axis=1)
    df["score_smooth"] = df.apply(calc_score_smooth, axis=1)

    # 找几个有代表性的
    samples = df.nlargest(3, "beta_real")[["correct_name", "ret45", "beta_real", "return_1d",
                                              "score_orig", "score_cond", "score_smooth"]]
    print(f"\n  高Beta股票 (Top 3):")
    print(f"  {'名称':<10} {'ret45':>8} {'Beta':>6} {'首日':>8} {'原始':>6} {'阶梯':>6} {'平滑':>6}")
    for _, row in samples.iterrows():
        print(f"  {row['correct_name']:<8} {row['ret45']:>+7.1f}% {row['beta_real']:>5.2f} {row['return_1d']:>+7.1f}% {row['score_orig']:>6.0f} {row['score_cond']:>6.0f} {row['score_smooth']:>6.0f}")

    # A股大跌但Beta高的
    down_samples = df[(df["ret45"] < -10) & (df["beta_real"] > 1.5)][["correct_name", "ret45", "beta_real", "return_1d",
                                                                         "score_orig", "score_cond", "score_smooth"]]
    if len(down_samples) > 0:
        print(f"\n  A股大跌 + 高Beta 的股票:")
        print(f"  {'名称':<10} {'ret45':>8} {'Beta':>6} {'首日':>8} {'原始':>6} {'阶梯':>6} {'平滑':>6}")
        for _, row in down_samples.iterrows():
            print(f"  {row['correct_name']:<8} {row['ret45']:>+7.1f}% {row['beta_real']:>5.2f} {row['return_1d']:>+7.1f}% {row['score_orig']:>6.0f} {row['score_cond']:>6.0f} {row['score_smooth']:>6.0f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
