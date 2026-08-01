#!/usr/bin/env python3
"""
AH股IPO折价规律深度分析
基于用户50只历史数据 + 富途OpenD补充验证
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def parse_md_data():
    """解析用户markdown表格数据"""
    md_path = "/Users/wangmengchi/Documents/远程仓库/锚定评分/ah 股.md"
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    data = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 9:
            continue
        try:
            idx = int(cols[0])
        except ValueError:
            continue

        def num(s):
            s = str(s).replace(",", "").replace("**", "").replace("==", "")
            return float(s) if s not in ("", "-") else np.nan

        data.append({
            "idx": idx,
            "name": cols[1],
            "return_1d": num(cols[2]),
            "list_date": cols[3],
            "issue_date": cols[4],
            "ret45": num(cols[5]),
            "cs": num(cols[6]),
            "cap_hkd": num(cols[7]),
            "issue_premium": num(cols[8]),  # 发行时HA溢价率
            "current_premium": num(cols[9]) if len(cols) > 9 else np.nan,  # 当前HA溢价率
        })
    return pd.DataFrame(data)


def analyze_discount_patterns(df):
    """分析折价规律"""
    print("=" * 80)
    print("📊 AH股IPO折价规律深度分析")
    print("=" * 80)

    # 计算折价变化
    df["discount_issue"] = -df["issue_premium"]  # 发行时AH折价率
    df["discount_current"] = -df["current_premium"]  # 当前AH折价率
    df["discount_change"] = df["discount_current"] - df["discount_issue"]  # 折价变化
    # 正值 = 折价扩大，负值 = 折价收敛

    # 计算上市到当前的时长(月)
    df["list_dt"] = pd.to_datetime(df["list_date"], format="%Y/%m/%d")
    df["months_since_list"] = (datetime(2026, 7, 25) - df["list_dt"]).dt.days / 30

    print(f"\n📋 样本: {len(df)} 只AH股IPO")
    print(f"时间跨度: {df['list_date'].min()} ~ {df['list_date'].max()}")

    # ========== 1. 折价整体变化 ==========
    print("\n" + "=" * 80)
    print("1️⃣ 折价整体变化趋势")
    print("=" * 80)

    converged = df[df["discount_change"] < -5]  # 折价收敛 >5%
    expanded = df[df["discount_change"] > 5]  # 折价扩大 >5%
    stable = df[df["discount_change"].abs() <= 5]

    print(f"\n  折价收敛(>5%): {len(converged)}只 ({len(converged)/len(df)*100:.0f}%)")
    print(f"  折价扩大(>5%): {len(expanded)}只 ({len(expanded)/len(df)*100:.0f}%)")
    print(f"  基本稳定(±5%): {len(stable)}只 ({len(stable)/len(df)*100:.0f}%)")

    print(f"\n  平均折价变化: {df['discount_change'].mean():+.1f}%")
    print(f"  中位数折价变化: {df['discount_change'].median():+.1f}%")

    # ========== 2. 时间维度 ==========
    print("\n" + "=" * 80)
    print("2️⃣ 折价变化 vs 上市时间")
    print("=" * 80)

    df_recent = df[df["months_since_list"] <= 2]  # 近2个月
    df_mid = df[(df["months_since_list"] > 2) & (df["months_since_list"] <= 6)]  # 2-6月
    df_old = df[df["months_since_list"] > 6]  # 6月以上

    for label, subset in [("近2个月", df_recent), ("2-6个月", df_mid), ("6个月以上", df_old)]:
        if len(subset) > 0:
            avg_change = subset["discount_change"].mean()
            converged_pct = (subset["discount_change"] < -5).sum() / len(subset) * 100
            print(f"\n  {label} (n={len(subset)}):")
            print(f"    平均折价变化: {avg_change:+.1f}%")
            print(f"    折价收敛比例: {converged_pct:.0f}%")

    # ========== 3. 首日涨幅 vs 折价 ==========
    print("\n" + "=" * 80)
    print("3️⃣ 首日涨幅 vs 发行折价")
    print("=" * 80)

    # 分组
    df["disc_bin"] = pd.cut(df["discount_issue"], bins=[0, 20, 30, 40, 50, 100], labels=["<20%", "20-30%", "30-40%", "40-50%", ">50%"])
    disc_stats = df.groupby("disc_bin", observed=False)["return_1d"].agg(["mean", "median", "count"])

    print("\n  发行折价率 vs 首日平均涨幅:")
    for bname, row in disc_stats.iterrows():
        print(f"    {bname}: 平均首日={row['mean']:+.1f}% 中位数={row['median']:+.1f}% (n={int(row['count'])})")

    # ========== 4. 折价收敛机制：A股跌 vs H股涨？ ==========
    print("\n" + "=" * 80)
    print("4️⃣ 折价收敛机制分析")
    print("=" * 80)

    # 你的假设：折价收敛以A股下跌为代价
    # 验证：发行期间A股涨幅 vs 当前A股表现（我们没有当前A股数据，但可以用ret45推断）

    # 分组：发行时A股涨 vs A股跌
    df["a_rise_issue"] = df["ret45"] > 0

    print("\n  发行时A股45日表现 vs 后续折价变化:")
    for label, subset in [("A股上涨(发行期)", df[df["a_rise_issue"]]), ("A股下跌(发行期)", df[~df["a_rise_issue"]])]:
        if len(subset) > 0:
            print(f"\n    {label} (n={len(subset)}):")
            print(f"      平均发行折价: {subset['discount_issue'].mean():.1f}%")
            print(f"      平均当前折价: {subset['discount_current'].mean():.1f}%")
            print(f"      平均折价变化: {subset['discount_change'].mean():+.1f}%")
            print(f"      首日平均涨幅: {subset['return_1d'].mean():+.1f}%")

    # ========== 5. 基石占比 vs 折价 ==========
    print("\n" + "=" * 80)
    print("5️⃣ 基石占比 vs 折价/首日表现")
    print("=" * 80)

    df["cs_bin"] = pd.cut(df["cs"], bins=[0, 30, 40, 50, 100], labels=["<30%", "30-40%", "40-50%", ">50%"])
    cs_stats = df.groupby("cs_bin", observed=False).agg({
        "return_1d": "mean",
        "discount_issue": "mean",
        "discount_change": "mean",
    })

    print("\n  基石占比分组:")
    for bname, row in cs_stats.iterrows():
        print(f"    {bname}: 首日={row['return_1d']:+.1f}% 发行折价={row['discount_issue']:.1f}% 折价变化={row['discount_change']:+.1f}%")

    # ========== 6. 特例分析 ==========
    print("\n" + "=" * 80)
    print("6️⃣ 特例分析：折价反常收敛的股票")
    print("=" * 80)

    # 找出折价收敛最多的（即当前溢价率比发行时高 = 折价减少）
    converged_top = df.nsmallest(5, "discount_change")  # discount_change最小 = 收敛最多
    print("\n  折价收敛最多的5只:")
    for _, row in converged_top.iterrows():
        print(f"    {row['name']}: 发行折价={row['discount_issue']:.1f}% → 当前={row['discount_current']:.1f}% (变化{row['discount_change']:+.1f}%) 首日={row['return_1d']:+.1f}%")

    # 折价扩大最多的
    expanded_top = df.nlargest(5, "discount_change")
    print("\n  折价扩大最多的5只:")
    for _, row in expanded_top.iterrows():
        print(f"    {row['name']}: 发行折价={row['discount_issue']:.1f}% → 当前={row['discount_current']:.1f}% (变化{row['discount_change']:+.1f}%) 首日={row['return_1d']:+.1f}%")

    # ========== 7. 相关性分析 ==========
    print("\n" + "=" * 80)
    print("7️⃣ 相关性矩阵")
    print("=" * 80)

    corr_cols = ["return_1d", "discount_issue", "discount_current", "discount_change", "ret45", "cs", "cap_hkd"]
    corr_df = df[corr_cols].corr()
    print(f"\n  首日涨幅 vs 发行折价: {corr_df.loc['return_1d', 'discount_issue']:+.3f}")
    print(f"  首日涨幅 vs 当前折价: {corr_df.loc['return_1d', 'discount_current']:+.3f}")
    print(f"  发行折价 vs 折价变化: {corr_df.loc['discount_issue', 'discount_change']:+.3f}")
    print(f"  45日涨幅 vs 首日涨幅: {corr_df.loc['ret45', 'return_1d']:+.3f}")
    print(f"  基石占比 vs 首日涨幅: {corr_df.loc['cs', 'return_1d']:+.3f}")

    return df


def try_futu_data_enrichment(df):
    """尝试从富途OpenD获取补充数据"""
    print("\n" + "=" * 80)
    print("🔗 尝试从富途OpenD获取补充验证数据")
    print("=" * 80)

    try:
        from futu import OpenQuoteContext, KLType, RET_OK
    except ImportError:
        print("  ⚠️ 未安装futu-api，跳过OpenD数据补充")
        return df

    # 我们只分析最近上市的5只，看看A股上市后到现在的表现
    recent = df.nsmallest(5, "list_date")

    print(f"\n  分析最近5只上市后A股表现:")
    print(f"  {'名称':<12} {'上市日期':<12} {'发行折价':>8} {'当前折价':>8} {'A股上市→现在':>14}")
    print("  " + "-" * 60)

    with OpenQuoteContext(host="127.0.0.1", port=11111) as ctx:
        for _, row in recent.iterrows():
            code = str(row.get("a_code", ""))
            if not code:
                # 尝试匹配代码
                continue

            ci = int(code)
            futu_code = f"SH.{code}" if ci >= 600000 else f"SZ.{code}"

            list_dt = datetime.strptime(row["list_date"], "%Y/%m/%d")
            end = datetime.now()

            ret, kline, _ = ctx.request_history_kline(
                futu_code,
                start=list_dt.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                ktype=KLType.K_DAY,
            )

            if ret == RET_OK and len(kline) > 1:
                first_close = kline["close"].iloc[0]
                last_close = kline["close"].iloc[-1]
                total_ret = (last_close - first_close) / first_close * 100

                print(f"  {row['name']:<12} {row['list_date']:<12} {row['discount_issue']:>7.1f}% {row['discount_current']:>7.1f}% {total_ret:>+13.1f}%")
            else:
                print(f"  {row['name']:<12} {row['list_date']:<12} 数据获取失败")

    return df


def main():
    df = parse_md_data()
    df = analyze_discount_patterns(df)

    # 尝试OpenD补充
    df = try_futu_data_enrichment(df)

    # 保存分析结果
    os.makedirs("output", exist_ok=True)
    output = f"output/ah_ipo_discount_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n💾 分析结果已保存: {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
