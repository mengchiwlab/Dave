#!/usr/bin/env python3
"""
AH股IPO评分系统 - 模型精进探索
分析因子交互效应、非线性关系、行业差异
"""

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

# ========== 行业映射 ==========
IND_PATTERNS = {
    "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基|芯碁|峰岹": "半导体",
    "新能源|光伏|储能|宁德|钧达|大金重工|天岳|先导": "新能源",
    "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": "科技制造",
    "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏|牧原": "制造",
    "医药|医疗|生物|恒瑞|可孚|迈威": "医药",
    "新材料|军工|国恩|沃尔|吉宏|赤峰": "新材料军工",
    "饮料|食品|消费|安井|东鹏|海天": "消费",
    "银行|保险|地产|期货": "金融地产",
    "化工|钢铁|纺织|滨化": "周期",
}

def classify_industry(name):
    for p, label in IND_PATTERNS.items():
        if any(k in name for k in p.split("|")):
            return label
    return "其他"

# ========== 加载数据 ==========
df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
df = df.dropna(subset=["return_1d", "beta_real", "ret45", "ha_premium"])
df["industry"] = df["correct_name"].apply(classify_industry)
df["discount"] = -df["ha_premium"]  # 折价率
df["cap"] = df["cap_real"].fillna(df["cap_hkd"] * 0.92)

print("=" * 70)
print("📊 AH股IPO评分模型 - 精进方向探索")
print("=" * 70)
print(f"有效样本: {len(df)} 只")

# ========== 1. 当前因子IC分析 ==========
print("\n" + "=" * 70)
print("1️⃣ 单因子IC分析")
print("=" * 70)

factor_cols = {
    "ret45": "A股45日涨幅",
    "beta_real": "Beta",
    "cs": "基石占比",
    "pos_real": "价格位置",
    "alpha_real": "Alpha",
    "discount": "AH折价率",
    "cap": "市值",
}

for col, name in factor_cols.items():
    valid = df[[col, "return_1d"]].dropna()
    if len(valid) < 5:
        continue
    ic = spearman_corr(valid[col], valid["return_1d"])
    bar = "█" * int(abs(ic) * 40)
    print(f"  {name:<15} IC={ic:+.3f}  {bar}")

# ========== 2. 因子交互效应 ==========
print("\n" + "=" * 70)
print("2️⃣ 因子交互效应分析")
print("=" * 70)

# 2.1 ret45 × discount 交互
df["ret45_x_disc"] = df["ret45"] * df["discount"] / 100
ic = spearman_corr(df["ret45_x_disc"], df["return_1d"])
print(f"\n  ret45 × 折价率:    IC={ic:+.3f} {'⬆️ 强交互' if abs(ic) > 0.3 else ''}")

# 2.2 beta × ret45 交互 (我们已有条件Beta，验证交互效应)
df["beta_x_ret45"] = df["beta_real"] * df["ret45"]
ic = spearman_corr(df["beta_x_ret45"], df["return_1d"])
print(f"  Beta × ret45:      IC={ic:+.3f} {'⬆️ 强交互' if abs(ic) > 0.3 else ''}")

# 2.3 cs × discount 交互
df["cs_x_disc"] = df["cs"] * df["discount"] / 100
ic = spearman_corr(df["cs_x_disc"], df["return_1d"])
print(f"  基石 × 折价率:     IC={ic:+.3f} {'⬆️ 强交互' if abs(ic) > 0.3 else ''}")

# 2.4 pos × beta 交互
df["pos_x_beta"] = df["pos_real"] * df["beta_real"]
ic = spearman_corr(df["pos_x_beta"], df["return_1d"])
print(f"  位置 × Beta:       IC={ic:+.3f} {'⬆️ 强交互' if abs(ic) > 0.3 else ''}")

# 2.5 alpha × beta 交互
df["alpha_x_beta"] = df["alpha_real"] * df["beta_real"]
ic = spearman_corr(df["alpha_x_beta"], df["return_1d"])
print(f"  Alpha × Beta:      IC={ic:+.3f} {'⬆️ 强交互' if abs(ic) > 0.3 else ''}")

# ========== 3. 行业差异分析 ==========
print("\n" + "=" * 70)
print("3️⃣ 行业差异分析")
print("=" * 70)

g = df.groupby("industry").agg({
    "return_1d": ["mean", "median", "std", "count"],
    "ret45": "mean",
    "beta_real": "mean",
    "discount": "mean",
}).round(2)

for ind in g.index:
    row = g.loc[ind]
    mean_ret = row[('return_1d', 'mean')]
    median_ret = row[('return_1d', 'median')]
    count = int(row[('return_1d', 'count')])
    if count >= 3:
        print(f"  {ind:<12} 首日={mean_ret:>+6.1f}% (中位数={median_ret:>+5.1f}%) n={count:>2}  "
              f"ret45={row[('ret45','mean')]:>+5.1f}% β={row[('beta_real','mean')]:.2f}")

# ========== 4. 非线性探索 ==========
print("\n" + "=" * 70)
print("4️⃣ 非线性关系探索")
print("=" * 70)

# 4.1 ret45 的非线性：极端正值可能更好
for label, low, high in [("<-15%", -100, -15), ("-15~-5%", -15, -5), ("-5~5%", -5, 5), ("5~15%", 5, 15), (">15%", 15, 100)]:
    subset = df[(df["ret45"] >= low) & (df["ret45"] < high)]
    if len(subset) > 0:
        print(f"  ret45 {label:<12} 首日={subset['return_1d'].mean():>+6.1f}% (n={len(subset)})")

# 4.2 discount 的非线性：极高折价是否边际递减
print()
for label, low, high in [("<30%", 0, 30), ("30-40%", 30, 40), ("40-50%", 40, 50), (">50%", 50, 100)]:
    subset = df[(df["discount"] >= low) & (df["discount"] < high)]
    if len(subset) > 0:
        print(f"  折价 {label:<10} 首日={subset['return_1d'].mean():>+6.1f}% (n={len(subset)})")

# 4.3 双因子联合阈值
df["high_disc"] = df["discount"] > 40
df["good_ret"] = df["ret45"] > -5

print("\n  双因子联合:")
print(f"  高折价(>40%) + A股不差(ret45>-5%): 首日={df[df['high_disc'] & df['good_ret']]['return_1d'].mean():+.1f}% (n={len(df[df['high_disc'] & df['good_ret']])})")
print(f"  高折价(>40%) + A股差(ret45<=-5%):   首日={df[df['high_disc'] & ~df['good_ret']]['return_1d'].mean():+.1f}% (n={len(df[df['high_disc'] & ~df['good_ret']])})")
print(f"  低折价(<=40%) + A股不差:            首日={df[~df['high_disc'] & df['good_ret']]['return_1d'].mean():+.1f}% (n={len(df[~df['high_disc'] & df['good_ret']])})")
print(f"  低折价(<=40%) + A股差:              首日={df[~df['high_disc'] & ~df['good_ret']]['return_1d'].mean():+.1f}% (n={len(df[~df['high_disc'] & ~df['good_ret']])})")

# ========== 5. 市值效应 ==========
print("\n" + "=" * 70)
print("5️⃣ 市值效应")
print("=" * 70)

df_cap = df.dropna(subset=["cap"])
df_cap["cap_group"] = pd.qcut(df_cap["cap"], 3, labels=["小市值", "中市值", "大市值"], duplicates="drop")
for gname in ["小市值", "中市值", "大市值"]:
    subset = df_cap[df_cap["cap_group"] == gname]
    if len(subset) > 0:
        print(f"  {gname:<6} 首日={subset['return_1d'].mean():>+6.1f}% β={subset['beta_real'].mean():.2f} (n={len(subset)})")

# ========== 6. 改进建议 ==========
print("\n" + "=" * 70)
print("📝 模型精进建议")
print("=" * 70)

# 找到IC最高的交互项
interactions = {
    "ret45 × 折价率": spearman_corr(df["ret45_x_disc"], df["return_1d"]),
    "Beta × ret45": spearman_corr(df["beta_x_ret45"], df["return_1d"]),
    "基石 × 折价率": spearman_corr(df["cs_x_disc"], df["return_1d"]),
    "位置 × Beta": spearman_corr(df["pos_x_beta"], df["return_1d"]),
}

print("\n  📈 最有潜力的改进方向（按交互IC排序）:")
for name, ic in sorted(interactions.items(), key=lambda x: -abs(x[1])):
    if abs(ic) > 0.1:
        print(f"     {ic:+.3f}  {name}")

print("\n  🔧 具体改进方案:")
print("""
  1. 【交互项加分】ret45 × 折价率
     → A股涨 + 高折价 = 超级利好（情绪好+便宜）
     建议: 当 ret45 > 0 且 discount > 40% 时，额外加 5-10 分

  2. 【行业差异化权重】
     → 半导体/新能源 的 Beta 应该更高权重（弹性大=首日炒得狠）
     → 消费/金融 的 Beta 应该更低权重（弹性小=首日温和）

  3. 【市值小票效应】
     → 小市值股票首日涨幅更大（资金容易拉起来）
     建议: 市值 < 200亿 额外加 2-5 分

  4. 【阶梯改连续】
     → ret45 > 15 给 17 分，14.9 给 14 分，跳跃太大
     建议: 用 sigmoid/线性插值替代阶梯函数

  5. 【Alpha因子】
     → 当前 Alpha IC 很低 (~0.02)，要么提升权重，要么替换
     建议: 换成 "个股波动率/残差标准差" 或 "换手率"
  """)
