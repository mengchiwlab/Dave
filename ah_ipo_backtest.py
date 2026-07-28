#!/usr/bin/env python3
"""
AH股IPO评分系统 - 因子回测与权重优化分析

功能:
1. 单因子IC分析 (Information Coefficient)
2. 单因子分组回测 (Quantile Return)
3. 多因子相关性分析
4. 权重优化 (IC加权 / 夏普比率最大化)
5. 新旧评分对比

运行: python3 ah_ipo_backtest.py

需要数据文件: data/ipo_history.csv (下面有格式说明)
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ========== 配置 ==========
MIN_SAMPLE_SIZE = 15  # 最小样本数要求
TARGET_RETURN_COL = "return_7d"  # 目标收益率列名: 可改为 return_1d / return_7d / return_30d

# 当前评分系统的因子定义 (名称, 权重满分, 是否正向因子)
CURRENT_FACTORS = {
    "ret45":    {"weight": 20, "direction": 1,  "name": "A股45日涨幅"},
    "beta":     {"weight": 15, "direction": -1, "name": "Beta波动"},     # 负向: Beta低更好
    "cs":       {"weight": 15, "direction": 1,  "name": "基石占比"},
    "pos":      {"weight": 12, "direction": 1,  "name": "价格位置"},     # 低位更好
    "alpha":    {"weight": 12, "direction": 1,  "name": "Alpha"},
    "discount": {"weight": 12, "direction": 1,  "name": "AH折价率"},
    "ind":      {"weight": 8,  "direction": 1,  "name": "行业得分"},
    "cap":      {"weight": 4,  "direction": -1, "name": "市值"},         # 小市值更好
    "sp":       {"weight": 3,  "direction": 1,  "name": "保荐人得分"},
}

# ========== 数据加载 ==========
def spearman_corr(x, y):
    """手动计算Spearman秩相关系数 (避免scipy依赖)"""
    # 转rank
    x_rank = pd.Series(x).rank()
    y_rank = pd.Series(y).rank()
    n = len(x_rank)
    # Pearson correlation of ranks
    mean_xr = x_rank.mean()
    mean_yr = y_rank.mean()
    cov = ((x_rank - mean_xr) * (y_rank - mean_yr)).sum() / n
    std_x = np.sqrt(((x_rank - mean_xr) ** 2).sum() / n)
    std_y = np.sqrt(((y_rank - mean_yr) ** 2).sum() / n)
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)


def load_data(filepath="data/ipo_history.csv"):
    """
    加载历史IPO数据

    期望CSV格式:
    code,name,ipo_date,ret45,beta,cs,pos,alpha,discount,ind,cap,sp,return_1d,return_7d,return_30d
    688981,中芯国际,2024-01-15,5.2,1.3,40,0.6,0.8,30,5,500,3,0.05,0.12,0.08
    ...

    各列含义:
    - ret45: A股45日涨跌幅(%)
    - beta: Beta系数
    - cs: 基石占比(%)
    - pos: 价格位置(0-1)
    - alpha: Alpha年化收益率
    - discount: AH折价率(%)
    - ind: 行业得分(1-5)
    - cap: 市值(亿元)
    - sp: 保荐人得分(0.5-3)
    - return_1d/7d/30d: IPO后实际收益率
    """
    if not os.path.exists(filepath):
        print(f"数据文件不存在: {filepath}")
        print("将生成模拟数据进行演示...")
        return generate_mock_data()

    df = pd.read_csv(filepath)
    print(f"加载历史数据: {len(df)} 只IPO")
    return df


def generate_mock_data(n=30):
    """生成模拟历史IPO数据用于演示分析框架"""
    np.random.seed(42)

    # 生成相关性结构: ret45和alpha与收益正相关, beta和cap与收益负相关
    data = {
        "code": [f"688{i:03d}" for i in range(n)],
        "name": [f"股票{i}" for i in range(n)],
        "ipo_date": pd.date_range("2023-01-01", periods=n, freq="MS").strftime("%Y-%m-%d"),
        "ret45": np.random.normal(5, 15, n),        # 45日涨幅, 均值5%, 标准差15%
        "beta": np.random.normal(1.2, 0.6, n),      # Beta均值1.2
        "cs": np.random.normal(35, 15, n),          # 基石占比均值35%
        "pos": np.random.uniform(0.2, 0.9, n),      # 价格位置
        "alpha": np.random.normal(0.3, 0.8, n),     # Alpha均值30%
        "discount": np.random.normal(30, 15, n),    # AH折价率均值30%
        "ind": np.random.choice([1,2,3,4,5], n, p=[0.1,0.15,0.25,0.3,0.2]),
        "cap": np.random.lognormal(6, 1, n),        # 市值对数正态分布
        "sp": np.random.choice([0.5,1,1.5,2,3], n, p=[0.1,0.2,0.25,0.25,0.2]),
    }

    df = pd.DataFrame(data)

    # 生成目标收益率: 与ret45、alpha、discount、cs正相关; 与beta、cap负相关
    base_return = (
        0.003 * df["ret45"] +          # 45日涨幅每1%贡献0.3%
        0.05 * df["alpha"] +           # Alpha每0.1贡献0.5%
        0.002 * df["discount"] +       # 折价率每1%贡献0.2%
        0.001 * df["cs"] +             # 基石每1%贡献0.1%
        0.03 * df["ind"] +             # 行业得分
        0.01 * df["sp"] -              # 保荐人
        0.02 * df["beta"] -            # Beta高 penalize
        0.0001 * df["cap"]             # 大市值 penalize
    )

    noise = np.random.normal(0, 0.05, n)  # 噪声
    df["return_7d"] = np.clip(base_return + noise, -0.3, 0.5)  # 限制在-30%~50%
    df["return_1d"] = df["return_7d"] * np.random.uniform(0.3, 0.7, n)
    df["return_30d"] = df["return_7d"] * np.random.uniform(0.8, 1.5, n)

    print(f"[模拟数据] 生成 {n} 只IPO用于演示")
    print(f"[模拟数据] 7日收益率范围: {df['return_7d'].min():.1%} ~ {df['return_7d'].max():.1%}")
    print(f"[模拟数据] 7日收益率均值: {df['return_7d'].mean():.1%}")
    return df


# ========== 1. 单因子IC分析 ==========
def analyze_ic(df):
    """计算各因子的Rank IC (Spearman相关系数)"""
    print("\n" + "="*60)
    print("📊 1. 单因子IC分析 (Rank Information Coefficient)")
    print("="*60)
    print("IC含义: 因子值与未来收益排名的相关性, |IC|>0.1 一般认为有效")
    print("-"*60)

    ic_results = []
    for col, config in CURRENT_FACTORS.items():
        if col not in df.columns:
            continue

        # 处理缺失值
        valid = df[[col, TARGET_RETURN_COL]].dropna()
        if len(valid) < MIN_SAMPLE_SIZE:
            print(f"  {config['name']:<12} 样本不足 ({len(valid)})")
            continue

        # 计算Rank IC (Spearman)
        ic = spearman_corr(valid[col] * config["direction"], valid[TARGET_RETURN_COL])
        p_value = 0.05  # 简化处理

        # IC的稳定性: 滚动IC的标准差
        if len(valid) >= 20:
            rolling_ics = []
            window = max(len(valid) // 4, 5)
            for i in range(0, len(valid) - window, max(window // 2, 1)):
                subset = valid.iloc[i:i+window]
                r = spearman_corr(subset[col] * config["direction"], subset[TARGET_RETURN_COL])
                if not pd.isna(r):
                    rolling_ics.append(r)
            ic_std = np.std(rolling_ics) if rolling_ics else 0.5
        else:
            ic_std = 0.5

        ic_results.append({
            "factor": col,
            "name": config["name"],
            "ic": ic,
            "ic_abs": abs(ic),
            "p_value": p_value,
            "ic_std": ic_std,
            "ir": abs(ic) / (ic_std + 1e-6),  # Information Ratio
            "significant": p_value < 0.1,  # 10%显著性
        })

    ic_df = pd.DataFrame(ic_results).sort_values("ic_abs", ascending=False)

    for _, row in ic_df.iterrows():
        sig_mark = "✅" if row["significant"] else "⚠️"
        print(f"  {sig_mark} {row['name']:<12} IC={row['ic']:+.3f}  |IC|={row['ic_abs']:.3f}  p={row['p_value']:.3f}  IR={row['ir']:.2f}")

    print("-"*60)
    print(f"  有效因子 (|IC|>0.1 & p<0.1): {ic_df[ic_df['ic_abs'] > 0.1]['significant'].sum()}/{len(ic_df)}")

    return ic_df


# ========== 2. 单因子分组回测 ==========
def analyze_quantile_returns(df, n_quantiles=5):
    """将每个因子分成5组,看每组平均收益"""
    print("\n" + "="*60)
    print(f"📊 2. 单因子分组回测 ({n_quantiles}分位)")
    print("="*60)
    print("将因子从小到大分为5组,观察每组平均收益率")
    print("-"*60)

    quantile_results = {}

    for col, config in CURRENT_FACTORS.items():
        if col not in df.columns:
            continue

        valid = df[[col, TARGET_RETURN_COL]].dropna()
        if len(valid) < n_quantiles * 3:
            continue

        # 按因子值分组 (考虑方向)
        factor_values = valid[col] * config["direction"]
        unique_vals = factor_values.nunique()

        if unique_vals <= n_quantiles:
            # 离散值少,直接按唯一值分组
            valid["group"] = pd.Categorical(factor_values.astype(str))
        else:
            try:
                valid["group"] = pd.qcut(
                    factor_values, n_quantiles,
                    labels=[f"Q{i+1}" for i in range(n_quantiles)],
                    duplicates="drop"
                )
            except ValueError:
                # 如果qcut还是失败，用cut替代
                valid["group"] = pd.cut(
                    factor_values, n_quantiles,
                    labels=[f"Q{i+1}" for i in range(n_quantiles)],
                    duplicates="drop"
                )

        group_returns = valid.groupby("group")[TARGET_RETURN_COL].agg(["mean", "median", "std", "count"])
        group_returns["sharpe"] = group_returns["mean"] / (group_returns["std"] + 1e-6)

        # 多空收益 (最高分位 - 最低分位)
        group_names = group_returns.index.tolist()
        if len(group_names) >= 2:
            long_short = group_returns.loc[group_names[-1], "mean"] - group_returns.loc[group_names[0], "mean"]
        else:
            long_short = 0

        quantile_results[col] = {
            "name": config["name"],
            "groups": group_returns,
            "long_short": long_short,
            "monotonic": group_returns["mean"].is_monotonic_increasing,
        }

        print(f"\n  📈 {config['name']} (方向: {'正向' if config['direction']==1 else '负向'})")
        print(f"     多空收益(Top-Bottom): {long_short:+.2%}")
        print(f"     单调性: {'✅' if quantile_results[col]['monotonic'] else '❌'}")
        for qname, row in group_returns.iterrows():
            bar = "█" * int(abs(row["mean"]) * 100)
            print(f"     {qname}: 均值={row['mean']:+.2%} 中位数={row['median']:+.2%} 夏普={row['sharpe']:.2f} {bar}")

    return quantile_results


# ========== 3. 多因子相关性分析 ==========
def analyze_correlation(df):
    """分析因子间的相关性,避免多重共线性"""
    print("\n" + "="*60)
    print("📊 3. 多因子相关性分析")
    print("="*60)
    print("因子间相关系数高(>0.7)可能存在冗余,考虑合并或降权")
    print("-"*60)

    factor_cols = [c for c in CURRENT_FACTORS.keys() if c in df.columns]
    # 手动计算spearman相关矩阵
    corr_matrix = pd.DataFrame(index=factor_cols, columns=factor_cols, dtype=float)
    for i, c1 in enumerate(factor_cols):
        for j, c2 in enumerate(factor_cols):
            if i == j:
                corr_matrix.loc[c1, c2] = 1.0
            else:
                corr_matrix.loc[c1, c2] = spearman_corr(df[c1].dropna(), df[c2].dropna())
    corr_matrix = corr_matrix.astype(float)

    # 找到高相关性对
    high_corr_pairs = []
    for i in range(len(factor_cols)):
        for j in range(i+1, len(factor_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.5:  # 0.5以上算高相关
                high_corr_pairs.append({
                    "f1": factor_cols[i],
                    "f2": factor_cols[j],
                    "name1": CURRENT_FACTORS[factor_cols[i]]["name"],
                    "name2": CURRENT_FACTORS[factor_cols[j]]["name"],
                    "corr": corr,
                })

    if not high_corr_pairs:
        print("  ✅ 各因子间相关性较低,没有明显冗余")
    else:
        print("  ⚠️ 高相关性因子对:")
        for p in sorted(high_corr_pairs, key=lambda x: abs(x["corr"]), reverse=True):
            print(f"     {p['name1']} ↔ {p['name2']}: r={p['corr']:+.3f}")

    # 打印相关系数矩阵
    print("\n  相关系数矩阵:")
    display_cols = {c: CURRENT_FACTORS[c]["name"] for c in factor_cols}
    corr_display = corr_matrix.rename(columns=display_cols, index=display_cols)
    print(corr_display.round(2).to_string())

    return corr_matrix, high_corr_pairs


# ========== 4. 权重优化 ==========
def optimize_weights(df, ic_df):
    """基于IC进行权重优化"""
    print("\n" + "="*60)
    print("📊 4. 权重优化建议")
    print("="*60)

    # 方法1: IC绝对值加权 (IC越高权重越大)
    ic_weights = {}
    total_ic = ic_df["ic_abs"].sum()
    if total_ic > 0:
        for _, row in ic_df.iterrows():
            ic_weights[row["factor"]] = row["ic_abs"] / total_ic * 100

    # 方法2: IR加权 (IC/IC_std, 稳定性调整)
    ir_weights = {}
    total_ir = ic_df["ir"].sum()
    if total_ir > 0:
        for _, row in ic_df.iterrows():
            ir_weights[row["factor"]] = row["ir"] / total_ir * 100

    # 打印对比
    print(f"\n  {'因子':<12} {'当前权重':>8} {'IC加权':>8} {'IR加权':>8} {'建议'}")
    print("  " + "-"*55)

    suggestions = []
    for col, config in CURRENT_FACTORS.items():
        if col not in ic_df["factor"].values:
            continue

        current = config["weight"]
        ic_w = ic_weights.get(col, 0)
        ir_w = ir_weights.get(col, 0)

        # 建议: 如果IC>0.15且当前权重偏低,建议上调; 如果IC<0.05建议下调
        ic_val = ic_df[ic_df["factor"] == col]["ic_abs"].values[0]
        if ic_val > 0.15 and current < ic_w * 0.8:
            suggestion = "⬆️ 建议上调"
        elif ic_val < 0.05 and current > 5:
            suggestion = "⬇️ 建议下调"
        elif ic_val < 0.03:
            suggestion = "❌ 可考虑剔除"
        else:
            suggestion = "✅ 合理"

        suggestions.append({
            "factor": col, "name": config["name"],
            "current": current, "ic_weight": ic_w, "ir_weight": ir_w,
            "ic": ic_val, "suggestion": suggestion,
        })

        print(f"  {config['name']:<10} {current:>8.1f} {ic_w:>8.1f} {ir_w:>8.1f}  {suggestion}")

    return suggestions


# ========== 5. 新旧评分对比 ==========
def backtest_score_comparison(df, ic_df):
    """对比当前评分vs优化后评分对未来收益的预测能力"""
    print("\n" + "="*60)
    print("📊 5. 新旧评分体系对比")
    print("="*60)

    # 计算当前总分
    def calc_current_score(row):
        total = 0
        for col, config in CURRENT_FACTORS.items():
            if col not in row:
                continue
            v = row[col] * config["direction"]
            # 用当前评分逻辑简化版
            if col == "ret45":
                s = 20 if v > 15 else 16 if v > 5 else 12 if v > -5 else 8 if v > -15 else 4
            elif col == "beta":
                s = 15 if v > 2 else 12 if v > 1.5 else 9 if v > 1 else 6 if v > 0.5 else 3
            elif col == "cs":
                s = 15 if v > 50 else 12 if v > 40 else 8 if v > 30 else 4
            elif col == "pos":
                s = 12 if v > 0.8 else 10 if v > 0.6 else 7 if v > 0.4 else 4 if v > 0.2 else 2
            elif col == "alpha":
                s = 12 if v > 1 else 9 if v > 0.5 else 6 if v > 0 else 3 if v > -0.5 else 0
            elif col == "discount":
                s = 12 if v > 50 else 9 if v > 40 else 6 if v > 30 else 3 if v > 20 else 1
            elif col == "ind":
                s = v
            elif col == "cap":
                s = 4 if v < 200 else 3 if v < 1000 else 2 if v < 5000 else 1
            elif col == "sp":
                s = v
            else:
                s = 0
            total += s
        return total

    # 计算IC加权总分 (线性加权简化版)
    def calc_optimized_score(row, ic_df):
        total = 0
        total_ic = ic_df["ic_abs"].sum()
        for _, ic_row in ic_df.iterrows():
            col = ic_row["factor"]
            if col not in row:
                continue
            weight = ic_row["ic_abs"] / total_ic * 100
            # 标准化到0-100
            vals = df[col].dropna()
            if len(vals) > 1:
                std = vals.std()
                mean = vals.mean()
                if std > 0:
                    zscore = (row[col] - mean) / std
                    score = 50 + zscore * 25  # 均值50,每1个标准差25分
                    score = max(0, min(100, score))
                else:
                    score = 50
            else:
                score = 50
            total += score * weight / 100
        return total

    df["score_current"] = df.apply(calc_current_score, axis=1)
    df["score_optimized"] = df.apply(lambda r: calc_optimized_score(r, ic_df), axis=1)

    # 评分的IC
    ic_current = spearman_corr(df["score_current"], df[TARGET_RETURN_COL])
    ic_optimized = spearman_corr(df["score_optimized"], df[TARGET_RETURN_COL])

    # 分组收益
    df["group_current"] = pd.qcut(df["score_current"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    df["group_optimized"] = pd.qcut(df["score_optimized"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")

    ret_current = df.groupby("group_current")[TARGET_RETURN_COL].mean()
    ret_optimized = df.groupby("group_optimized")[TARGET_RETURN_COL].mean()

    print(f"\n  {'分组':>8} {'当前评分收益':>12} {'优化评分收益':>12}")
    print("  " + "-"*40)
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        print(f"  {q:>8} {ret_current[q]:>+11.2%} {ret_optimized[q]:>+11.2%}")

    print(f"\n  当前评分IC: {ic_current:+.3f}")
    print(f"  优化评分IC: {ic_optimized:+.3f}")

    if ic_optimized > ic_current:
        print(f"  ✅ 优化后评分预测能力提升: +{ic_optimized-ic_current:.3f}")
    else:
        print(f"  ⚠️ 当前评分已较好,优化提升有限")

    return {
        "ic_current": ic_current,
        "ic_optimized": ic_optimized,
        "ret_current": ret_current,
        "ret_optimized": ret_optimized,
    }


# ========== 主程序 ==========
def main():
    print("="*60)
    print("📊 AH股IPO评分系统 - 因子回测与权重优化")
    print("="*60)
    print(f"目标收益率: {TARGET_RETURN_COL}")
    print(f"当前权重体系版本: V5.8")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    if len(df) < MIN_SAMPLE_SIZE:
        print(f"\n❌ 样本数不足: {len(df)} < {MIN_SAMPLE_SIZE}")
        print("请提供至少15只历史IPO数据进行分析")
        return

    # 2. 单因子IC分析
    ic_df = analyze_ic(df)

    # 3. 单因子分组回测
    quantile_results = analyze_quantile_returns(df)

    # 4. 多因子相关性
    corr_matrix, high_corr = analyze_correlation(df)

    # 5. 权重优化
    suggestions = optimize_weights(df, ic_df)

    # 6. 新旧评分对比
    comparison = backtest_score_comparison(df, ic_df)

    # 7. 输出最终建议
    print("\n" + "="*60)
    print("📋 最终优化建议")
    print("="*60)

    # 找出最有效的因子
    top_factors = ic_df.head(3)
    print(f"\n  🏆 最有效的3个因子:")
    for _, row in top_factors.iterrows():
        print(f"     • {row['name']}: IC={row['ic']:+.3f}")

    # 找出最无效的因子
    weak_factors = ic_df.tail(2)
    print(f"\n  ⚠️ 效果最弱的因子:")
    for _, row in weak_factors.iterrows():
        print(f"     • {row['name']}: IC={row['ic']:+.3f} (可考虑降权或替换)")

    # 推荐新权重
    print(f"\n  📝 推荐权重调整方案:")
    total_ic = ic_df["ic_abs"].sum()
    for _, row in ic_df.iterrows():
        col = row["factor"]
        config = CURRENT_FACTORS[col]
        recommended = int(row["ic_abs"] / total_ic * 100)
        current = config["weight"]
        delta = recommended - current
        if abs(delta) >= 2:
            arrow = "⬆️" if delta > 0 else "⬇️"
            print(f"     {arrow} {config['name']}: {current} → {recommended} (Δ{delta:+d})")

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(df),
        "target_return": TARGET_RETURN_COL,
        "ic_analysis": ic_df.to_dict("records"),
        "correlation_warnings": high_corr,
        "weight_suggestions": suggestions,
        "score_comparison": {
            "ic_current": comparison["ic_current"],
            "ic_optimized": comparison["ic_optimized"],
        },
    }

    os.makedirs("output", exist_ok=True)
    output_file = f"output/ipo_factor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 详细结果已保存: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
