#!/usr/bin/env python3
"""
AH股IPO评分系统 - 基于真实历史数据回测
数据来源: /Users/wangmengchi/Documents/远程仓库/锚定评分/ah 股.md
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ========== 从Markdown解析数据 ==========
def parse_md_data():
    """解析用户的markdown表格数据"""
    md_path = "/Users/wangmengchi/Documents/远程仓库/锚定评分/ah 股.md"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取表格行
    lines = content.strip().split("\n")
    data = []

    for line in lines:
        # 匹配表格行: | 1 | 滨化股份 | -18.68 | ... |
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        cols = [c for c in cols if c]  # 去掉空元素
        if len(cols) < 9:
            continue
        # 跳过表头行
        if cols[0] in ["序号", "---"] or "序号" in cols[0]:
            continue
        try:
            idx = int(cols[0])
        except ValueError:
            continue

        # 提取数值 (处理各种格式)
        def parse_num(s):
            s = str(s).strip().replace(",", "").replace("**", "").replace("==", "")
            if s == "" or s == "-":
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan

        row = {
            "name": cols[1],
            "return_1d": parse_num(cols[2]),  # 首日涨幅%
            "list_date": cols[3],
            "issue_date": cols[4],
            "ret45": parse_num(cols[5]),  # 发行期间A股涨跌幅
            "cs": parse_num(cols[6]),  # 基石占比%
            "cap_hkd": parse_num(cols[7]),  # 发行市值亿港元
            "ha_premium": parse_num(cols[8]),  # 发行溢价HA%
            "current_premium": parse_num(cols[9]) if len(cols) > 9 else np.nan,  # 现在溢价率
        }
        data.append(row)

    return pd.DataFrame(data)


# ========== 补充缺失因子 ==========
def enrich_factors(df):
    """用规则/近似值补充缺失因子"""

    # 1. AH折价率: 发行溢价HA% 转 discount
    # 用户的"发行溢价 HA%"含义: H股价格相对A股的溢价
    # 例如 -51.4 表示H股比A股便宜51.4%, 即AH折价率 = 51.4
    df["discount"] = -df["ha_premium"]

    # 2. 市值换算: 亿港元 → 人民币亿元 (假设汇率0.92)
    df["cap"] = df["cap_hkd"] * 0.92

    # 3. 行业得分: 根据名称关键词映射 (简化版)
    industry_map = {
        "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基": 5,
        "新能源|光伏|储能|宁德|均达|大金重工|天岳|先导": 4,
        "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": 4,
        "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏|牧原": 4,
        "医药|医疗|生物|恒瑞|可孚|迈威": 4,
        "新材料|军工|国恩|沃尔|吉宏|赤峰": 3,
        "饮料|食品|消费|安井|东鹏|海天": 2,
        "银行|保险|地产|期货": 1,
        "化工|钢铁|纺织|滨化": 1,
    }

    def get_ind_score(name):
        for pattern, score in industry_map.items():
            for keyword in pattern.split("|"):
                if keyword in name:
                    return score
        return 2  # 默认

    df["ind"] = df["name"].apply(get_ind_score)

    # 4. 保荐人得分: 目前无法从名称推断,用默认值
    df["sp"] = 2.0  # 默认中位数

    # 5. Beta: 从A股数据获取最佳,这里用行业近似
    # 半导体/科技类Beta偏高(1.3-1.8),消费类偏低(0.8-1.2)
    beta_map = {
        "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基": 1.5,
        "新能源|光伏|储能|宁德|均达|大金重工|天岳|先导": 1.4,
        "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": 1.3,
        "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏": 1.3,
        "医药|医疗|生物|恒瑞|可孚|迈威": 1.0,
        "新材料|军工|国恩|沃尔|吉宏|赤峰": 1.1,
        "饮料|食品|消费|安井|东鹏|海天|牧原": 0.9,
        "银行|保险|地产|期货": 0.7,
        "化工|钢铁|纺织|滨化": 1.1,
    }

    def get_beta(name):
        for pattern, b in beta_map.items():
            for keyword in pattern.split("|"):
                if keyword in name:
                    return b
        return 1.2

    df["beta"] = df["name"].apply(get_beta)

    # 6. Alpha: 简化处理, 用 ret45 - beta * 市场收益
    # 假设期间市场收益 ≈ ret45 的平均值
    market_ret = df["ret45"].mean()
    df["alpha"] = (df["ret45"] / 100 - df["beta"] * market_ret / 100) * 252 / 45  # 年化

    # 7. 价格位置: 用ret45反推近似 (简化假设)
    # ret45高 → 价格在高位; ret45低 → 在低位
    # 标准化到0-1
    ret_min, ret_max = df["ret45"].min(), df["ret45"].max()
    range_ret = ret_max - ret_min if ret_max != ret_min else 1
    df["pos"] = 0.3 + 0.4 * (df["ret45"] - ret_min) / range_ret  # 映射到0.3-0.7区间

    return df


# ========== Spearman相关 ==========
def spearman_corr(x, y):
    """手动计算Spearman秩相关系数"""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    # 对齐
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return 0
    x_rank = df["x"].rank()
    y_rank = df["y"].rank()
    n = len(x_rank)
    mean_xr = x_rank.mean()
    mean_yr = y_rank.mean()
    cov = ((x_rank - mean_xr) * (y_rank - mean_yr)).sum() / n
    std_x = np.sqrt(((x_rank - mean_xr) ** 2).sum() / n)
    std_y = np.sqrt(((y_rank - mean_yr) ** 2).sum() / n)
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)


# ========== 当前评分体系 V5.8 ==========
def calc_score_v58(row):
    """计算当前版本评分"""
    r = row.get("ret45", 0)
    b = row.get("beta", 1)
    cs = row.get("cs", 0)
    p = row.get("pos", 0.5)
    a = row.get("alpha", 0)
    disc = row.get("discount", 0)
    ind = row.get("ind", 2)
    cap = row.get("cap", None)
    sp = row.get("sp", 1)

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
        ("市值规模", s_cap, 4, f"{cap:.0f}亿" if cap else "N/A"),
        ("保荐人", s_sp, 3, ""),
    ]
    return total, factors


# ========== 回测主程序 ==========
def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 - 真实历史数据回测")
    print("=" * 70)

    # 1. 解析数据
    df = parse_md_data()
    print(f"\n✅ 解析到 {len(df)} 只历史IPO数据")

    # 2. 补充因子
    df = enrich_factors(df)

    # 3. 计算当前评分
    scores = []
    for _, row in df.iterrows():
        total, _ = calc_score_v58(row)
        scores.append(total)
    df["score"] = scores

    # 4. IC分析
    print("\n" + "=" * 70)
    print("📈 单因子IC分析 (与未来首日涨幅的相关性)")
    print("=" * 70)

    factor_cols = {
        "ret45": "A股45日涨幅",
        "beta": "Beta",
        "cs": "基石占比",
        "pos": "价格位置",
        "alpha": "Alpha",
        "discount": "AH折价率",
        "ind": "行业得分",
        "cap": "市值",
        "sp": "保荐人得分",
    }

    ic_results = []
    for col, name in factor_cols.items():
        valid = df[[col, "return_1d"]].dropna()
        if len(valid) < 5:
            continue
        # 考虑方向
        direction = -1 if col in ["beta", "cap"] else 1
        ic = spearman_corr(valid[col] * direction, valid["return_1d"])
        ic_results.append({"factor": col, "name": name, "ic": ic, "ic_abs": abs(ic)})

    ic_df = pd.DataFrame(ic_results).sort_values("ic_abs", ascending=False)

    for _, row in ic_df.iterrows():
        bar = "█" * int(abs(row["ic"]) * 30)
        print(f"  {row['name']:<12} IC={row['ic']:+.3f}  {bar}")

    # 5. 评分与收益的IC
    score_ic = spearman_corr(df["score"], df["return_1d"])
    print(f"\n  📊 综合评分IC: {score_ic:+.3f}")
    if abs(score_ic) > 0.3:
        print(f"  ✅ 评分体系有效性较好 (|IC|>0.3)")
    elif abs(score_ic) > 0.15:
        print(f"  ⚠️ 评分体系有一定效果 (|IC|>0.15)")
    else:
        print(f"  ❌ 评分体系效果较弱 (|IC|<0.15)")

    # 6. 分组回测
    print("\n" + "=" * 70)
    print("📊 按评分分组 - 首日涨幅表现")
    print("=" * 70)

    df["group"] = pd.qcut(df["score"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"], duplicates="drop")
    group_stats = df.groupby("group")["return_1d"].agg(["mean", "median", "std", "count"])

    print(f"\n  {'分组':<10} {'平均首日涨幅':>12} {'中位数':>10} {'标准差':>8} {'数量':>6}")
    print("  " + "-" * 60)
    for gname, row in group_stats.iterrows():
        bar = "█" * int(max(0, row["mean"]) / 2) if row["mean"] > 0 else "░" * int(abs(row["mean"]) / 2)
        print(f"  {gname:<10} {row['mean']:>+10.2f}% {row['median']:>+8.2f}% {row['std']:>7.2f}% {int(row['count']):>5}")

    # 多空收益
    groups = group_stats.index.tolist()
    if len(groups) >= 2:
        ls = group_stats.loc[groups[-1], "mean"] - group_stats.loc[groups[0], "mean"]
        print(f"\n  🔄 多空收益 (Top-Bottom): {ls:+.2f}%")

    # 7. 找出评分与实际表现背离的案例
    print("\n" + "=" * 70)
    print("🔍 评分与实际表现背离的案例 (学习价值)")
    print("=" * 70)

    df["pred_rank"] = df["score"].rank(ascending=False)
    df["actual_rank"] = df["return_1d"].rank(ascending=False)
    df["rank_diff"] = abs(df["pred_rank"] - df["actual_rank"])

    # 评分高但表现差 (伪阳性)
    false_pos = df.nlargest(5, "score").nsmallest(3, "return_1d")
    print("\n  ⚠️ 评分高但首日涨幅差 (可能因子权重问题):")
    for _, row in false_pos.iterrows():
        print(f"     • {row['name']}: 评分={row['score']:.0f}, 首日={row['return_1d']:+.1f}%, 折价={row['discount']:.1f}%, ret45={row['ret45']:.1f}%")

    # 评分低但表现好 (伪阴性)
    false_neg = df.nsmallest(5, "score").nlargest(3, "return_1d")
    print("\n  💎 评分低但首日涨幅好 (可能遗漏了重要因子):")
    for _, row in false_neg.iterrows():
        print(f"     • {row['name']}: 评分={row['score']:.0f}, 首日={row['return_1d']:+.1f}%, 折价={row['discount']:.1f}%, ret45={row['ret45']:.1f}%")

    # 8. 单因子的阈值分析
    print("\n" + "=" * 70)
    print("📊 关键因子阈值分析")
    print("=" * 70)

    # AH折价率
    print("\n  AH折价率 vs 首日涨幅:")
    disc_bins = [0, 20, 30, 40, 50, 100]
    disc_labels = ["<20%", "20-30%", "30-40%", "40-50%", ">50%"]
    df["disc_bin"] = pd.cut(df["discount"], bins=disc_bins, labels=disc_labels)
    disc_stats = df.groupby("disc_bin", observed=False)["return_1d"].agg(["mean", "count"])
    for bname, row in disc_stats.iterrows():
        print(f"     {bname}: 平均首日={row['mean']:+.1f}% (n={int(row['count'])})")

    # ret45
    print("\n  A股45日涨幅 vs 首日涨幅:")
    ret_bins = [-30, -15, -5, 5, 15, 30]
    ret_labels = ["<-15%", "-15~-5%", "-5~5%", "5~15%", ">15%"]
    df["ret_bin"] = pd.cut(df["ret45"], bins=ret_bins, labels=ret_labels)
    ret_stats = df.groupby("ret_bin", observed=False)["return_1d"].agg(["mean", "count"])
    for bname, row in ret_stats.iterrows():
        print(f"     {bname}: 平均首日={row['mean']:+.1f}% (n={int(row['count'])})")

    # 9. 导出数据
    os.makedirs("output", exist_ok=True)
    output_csv = f"output/ipo_backtest_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 数据已导出: {output_csv}")

    # 10. 权重优化建议
    print("\n" + "=" * 70)
    print("📝 基于IC的权重优化建议")
    print("=" * 70)

    total_ic = ic_df["ic_abs"].sum()
    print(f"\n  {'因子':<12} {'当前':>6} {'IC加权':>8} {'建议'}")
    print("  " + "-" * 45)
    for col, name in factor_cols.items():
        current = {"ret45": 20, "beta": 15, "cs": 15, "pos": 12, "alpha": 12, "discount": 12, "ind": 8, "cap": 4, "sp": 3}[col]
        ic_row = ic_df[ic_df["factor"] == col]
        if len(ic_row) > 0:
            ic_w = int(ic_row.iloc[0]["ic_abs"] / total_ic * 100)
            delta = ic_w - current
            arrow = "⬆️" if delta > 3 else "⬇️" if delta < -3 else "✅"
            print(f"  {name:<10} {current:>5} {ic_w:>7}   {arrow}")
        else:
            print(f"  {name:<10} {current:>5} {'N/A':>7}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
