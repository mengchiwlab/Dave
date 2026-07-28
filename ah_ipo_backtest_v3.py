#!/usr/bin/env python3
"""
AH股IPO评分系统 - 真实历史数据回测 V3
修正: 市值用用户原始表格的发行市值, 而非富途A股总市值
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


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


def calc_score(row):
    r = row["ret45"]
    b = row.get("beta_real", 1)
    cs = row["cs"]
    p = row.get("pos_real", 0.5)
    a = row.get("alpha_real", 0)
    disc = -row["ha_premium"]
    ind = row.get("ind", 2)
    # 用用户的发行市值(港元)换算成人民币亿元
    cap_hkd = row["cap_hkd"]
    cap = cap_hkd * 0.92 if pd.notna(cap_hkd) else None
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


def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 - 真实数据回测 V3 (修正版)")
    print("=" * 70)

    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    df["ind"] = df["correct_name"].apply(get_industry_score)
    df["cap"] = df["cap_hkd"] * 0.92
    df["discount"] = -df["ha_premium"]

    # 计算评分
    df["score"] = df.apply(calc_score, axis=1)

    # ========== 1. IC分析 ==========
    print("\n📈 单因子IC分析 (与首日涨幅的相关性)")
    print("-" * 70)

    factors = {
        "ret45": ("A股45日涨幅", 1),
        "beta_real": ("Beta", -1),
        "cs": ("基石占比", 1),
        "pos_real": ("价格位置", 1),
        "alpha_real": ("Alpha", 1),
        "discount": ("AH折价率", 1),
        "ind": ("行业得分", 1),
        "cap": ("发行市值", -1),  # 修正: 用H股发行市值
    }

    ic_results = []
    for col, (name, direction) in factors.items():
        valid = df[[col, "return_1d"]].dropna()
        if len(valid) < 5:
            continue
        ic = spearman_corr(valid[col] * direction, valid["return_1d"])
        ic_results.append({"factor": col, "name": name, "ic": ic, "ic_abs": abs(ic)})

    ic_df = pd.DataFrame(ic_results).sort_values("ic_abs", ascending=False)
    for _, row in ic_df.iterrows():
        bar = "█" * int(abs(row["ic"]) * 30)
        print(f"  {row['name']:<14} IC={row['ic']:+.3f}  {bar}")

    score_ic = spearman_corr(df["score"], df["return_1d"])
    print(f"\n  📊 综合评分IC: {score_ic:+.3f}")
    level = "✅ 优秀" if abs(score_ic) > 0.4 else "✅ 良好" if abs(score_ic) > 0.3 else "⚠️ 一般" if abs(score_ic) > 0.15 else "❌ 较弱"
    print(f"  {level}")

    # ========== 2. 分组回测 ==========
    print("\n📊 按评分分组 - 首日涨幅表现")
    print("-" * 70)

    df["group"] = pd.qcut(df["score"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"], duplicates="drop")
    gs = df.groupby("group", observed=False)["return_1d"].agg(["mean", "median", "std", "count"])

    print(f"  {'分组':<10} {'平均首日':>10} {'中位数':>10} {'标准差':>8} {'数量':>5}")
    print("  " + "-" * 55)
    for gname, row in gs.iterrows():
        bar = "█" * int(max(0, row["mean"]) / 3) if row["mean"] > 0 else "░" * int(abs(row["mean"]) / 3)
        print(f"  {gname:<10} {row['mean']:>+8.2f}% {row['median']:>+8.2f}% {row['std']:>7.2f}% {int(row['count']):>4} {bar}")

    groups = gs.index.tolist()
    if len(groups) >= 2:
        ls = gs.loc[groups[-1], "mean"] - gs.loc[groups[0], "mean"]
        print(f"\n  🔄 多空收益 (Top-Bottom): {ls:+.2f}%")

    # ========== 3. 权重优化 ==========
    print("\n📝 权重优化建议 (基于IC)")
    print("-" * 70)

    total_ic = ic_df["ic_abs"].sum()
    current = {"ret45": 20, "beta_real": 15, "cs": 15, "pos_real": 12, "alpha_real": 12, "discount": 12, "ind": 8, "cap": 4, "sp": 3}

    print(f"  {'因子':<14} {'当前':>5} {'IC加权':>7} {'变化':>6} {'建议'}")
    print("  " + "-" * 55)
    for _, row in ic_df.iterrows():
        col = row["factor"]
        name = row["name"]
        cur = current.get(col, 0)
        icw = int(row["ic_abs"] / total_ic * 100)
        delta = icw - cur
        arrow = "⬆️" if delta > 3 else "⬇️" if delta < -3 else "✅"
        print(f"  {name:<12} {cur:>5} {icw:>7} {delta:>+5}   {arrow}")

    # ========== 4. 背离案例 ==========
    print("\n🔍 背离案例分析")
    print("-" * 70)

    print("\n  ⚠️ 高评分但首日差:")
    for _, row in df.nlargest(8, "score").nsmallest(4, "return_1d").iterrows():
        print(f"     • {row['correct_name']:<10} 评分={row['score']:.0f} 首日={row['return_1d']:+.1f}% 折价={row['discount']:.1f}%")

    print("\n  💎 低评分但首日好:")
    for _, row in df.nsmallest(8, "score").nlargest(4, "return_1d").iterrows():
        print(f"     • {row['correct_name']:<10} 评分={row['score']:.0f} 首日={row['return_1d']:+.1f}% 折价={row['discount']:.1f}%")

    # ========== 5. 导出 ==========
    output = f"output/ah_ipo_backtest_v3_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
