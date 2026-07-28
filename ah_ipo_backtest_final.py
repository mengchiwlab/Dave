#!/usr/bin/env python3
"""
AH股IPO评分系统 - 真实历史数据回测 V2
使用用户原始ret45 + 富途补充的Beta/Alpha/市值/价格位置
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


# ========== 评分 V5.8 ==========
def calc_score(row):
    r = row["ret45"]          # 用户原始数据
    b = row.get("beta_real", row.get("beta", 1))
    cs = row["cs"]
    p = row.get("pos_real", row.get("pos", 0.5))
    a = row.get("alpha_real", row.get("alpha", 0))
    disc = -row["ha_premium"]  # HA溢价转discount
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
    factors = [
        ("A股涨跌幅", s_ret, 20, f"{r:.1f}%"),
        ("Beta(波动)", s_beta, 15, f"β={b:.2f}"),
        ("基石占比", s_cs, 15, f"{cs:.1f}%"),
        ("价格位置", s_pos, 12, f"分位: {p*100:.0f}%"),
        ("Alpha", s_alpha, 12, f"α={a*100:.1f}%"),
        ("AH折价率", s_disc, 12, f"{disc:.1f}%"),
        ("行业", s_ind, 8, ""),
        ("市值规模", s_cap, 4, f"{cap:.0f}亿" if pd.notna(cap) else "N/A"),
        ("保荐人", s_sp, 3, ""),
    ]
    return total, factors


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


def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 - 真实数据回测 V2")
    print("=" * 70)

    # 读取增强数据
    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    print(f"\n📋 加载 {len(df)} 只IPO数据")

    # 补充行业得分
    df["ind"] = df["correct_name"].apply(get_industry_score)

    # 计算评分
    scores, factor_list = [], []
    for _, row in df.iterrows():
        total, factors = calc_score(row)
        scores.append(total)
        factor_list.append(factors)
    df["score"] = scores

    # ========== 1. IC分析 ==========
    print("\n" + "=" * 70)
    print("📈 单因子IC分析 (与首日涨幅的相关性)")
    print("=" * 70)

    factor_cols = {
        "ret45": ("A股45日涨幅", 1),
        "beta_real": ("Beta", -1),
        "cs": ("基石占比", 1),
        "pos_real": ("价格位置", 1),
        "alpha_real": ("Alpha", 1),
        "ha_premium": ("HA溢价(负=折价)", -1),  # 负向: 溢价越低(折价越高)越好
        "ind": ("行业得分", 1),
        "cap_real": ("市值", -1),
    }

    ic_results = []
    for col, (name, direction) in factor_cols.items():
        valid = df[[col, "return_1d"]].dropna()
        if len(valid) < 5:
            continue
        ic = spearman_corr(valid[col] * direction, valid["return_1d"])
        ic_results.append({"factor": col, "name": name, "ic": ic, "ic_abs": abs(ic)})

    ic_df = pd.DataFrame(ic_results).sort_values("ic_abs", ascending=False)
    for _, row in ic_df.iterrows():
        bar = "█" * int(abs(row["ic"]) * 30)
        print(f"  {row['name']:<12} IC={row['ic']:+.3f}  {bar}")

    # 综合评分IC
    score_ic = spearman_corr(df["score"], df["return_1d"])
    print(f"\n  📊 综合评分IC: {score_ic:+.3f}")
    print(f"  {'✅ 预测能力优秀' if abs(score_ic) > 0.4 else '✅ 预测能力良好' if abs(score_ic) > 0.3 else '⚠️ 预测能力一般'}")

    # ========== 2. 分组回测 ==========
    print("\n" + "=" * 70)
    print("📊 按评分分组 - 首日涨幅表现")
    print("=" * 70)

    df["group"] = pd.qcut(df["score"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"], duplicates="drop")
    gs = df.groupby("group", observed=False)["return_1d"].agg(["mean", "median", "std", "count"])

    print(f"\n  {'分组':<10} {'平均首日涨幅':>12} {'中位数':>10} {'标准差':>8} {'数量':>6}")
    print("  " + "-" * 60)
    for gname, row in gs.iterrows():
        bar = "█" * int(max(0, row["mean"]) / 3) if row["mean"] > 0 else "░" * int(abs(row["mean"]) / 3)
        print(f"  {gname:<10} {row['mean']:>+10.2f}% {row['median']:>+8.2f}% {row['std']:>7.2f}% {int(row['count']):>5} {bar}")

    groups = gs.index.tolist()
    if len(groups) >= 2:
        ls = gs.loc[groups[-1], "mean"] - gs.loc[groups[0], "mean"]
        print(f"\n  🔄 多空收益 (Top-Bottom): {ls:+.2f}%")

    # ========== 3. 阈值分析 ==========
    print("\n" + "=" * 70)
    print("📊 关键因子阈值分析")
    print("=" * 70)

    # AH折价率 (注意: ha_premium 是负值=折价)
    print("\n  HA溢价率 vs 首日涨幅:")
    df["disc"] = -df["ha_premium"]
    for label, low, high in [("<20%", 0, 20), ("20-30%", 20, 30), ("30-40%", 30, 40), ("40-50%", 40, 50), (">50%", 50, 100)]:
        subset = df[(df["disc"] >= low) & (df["disc"] < high)]
        if len(subset) > 0:
            print(f"     {label}: 平均={subset['return_1d'].mean():+.1f}% (n={len(subset)})")

    # ret45
    print("\n  A股发行期涨幅 vs 首日涨幅:")
    for label, low, high in [("<-10%", -100, -10), ("-10~0%", -10, 0), ("0~10%", 0, 10), (">10%", 10, 100)]:
        subset = df[(df["ret45"] >= low) & (df["ret45"] < high)]
        if len(subset) > 0:
            print(f"     {label}: 平均={subset['return_1d'].mean():+.1f}% (n={len(subset)})")

    # ========== 4. 权重优化建议 ==========
    print("\n" + "=" * 70)
    print("📝 基于IC的权重优化建议")
    print("=" * 70)

    total_ic = ic_df["ic_abs"].sum()
    current_w = {"ret45": 20, "beta_real": 15, "cs": 15, "pos_real": 12, "alpha_real": 12, "ha_premium": 12, "ind": 8, "cap_real": 4, "sp": 3}

    print(f"\n  {'因子':<14} {'当前':>6} {'IC加权':>8} {'建议'}")
    print("  " + "-" * 45)
    for _, row in ic_df.iterrows():
        col = row["factor"]
        name = row["name"]
        cur = current_w.get(col, 0)
        icw = int(row["ic_abs"] / total_ic * 100)
        delta = icw - cur
        arrow = "⬆️" if delta > 3 else "⬇️" if delta < -3 else "✅"
        print(f"  {name:<12} {cur:>5} {icw:>7}   {arrow}")

    # ========== 5. 背离案例 ==========
    print("\n" + "=" * 70)
    print("🔍 评分与实际表现背离的案例")
    print("=" * 70)

    df["pred_rank"] = df["score"].rank(ascending=False)
    df["actual_rank"] = df["return_1d"].rank(ascending=False)

    print("\n  ⚠️ 评分高但首日差 (Top5评分中表现最差的3只):")
    for _, row in df.nlargest(5, "score").nsmallest(3, "return_1d").iterrows():
        print(f"     • {row['correct_name']}: 评分={row['score']:.0f}, 首日={row['return_1d']:+.1f}%, 折价={-row['ha_premium']:.1f}%, ret45={row['ret45']:.1f}%")

    print("\n  💎 评分低但首日好 (Bottom5评分中表现最好的3只):")
    for _, row in df.nsmallest(5, "score").nlargest(3, "return_1d").iterrows():
        print(f"     • {row['correct_name']}: 评分={row['score']:.0f}, 首日={row['return_1d']:+.1f}%, 折价={-row['ha_premium']:.1f}%, ret45={row['ret45']:.1f}%")

    # ========== 6. 导出完整回测数据 ==========
    output = f"output/ah_ipo_backtest_final_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n💾 回测数据已保存: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
