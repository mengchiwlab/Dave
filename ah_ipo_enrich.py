#!/usr/bin/env python3
"""
AH股IPO数据增强工具 V2
1. 补全A股代码（修正名称匹配）
2. 从富途OpenD获取真实Beta/Alpha/市值/价格位置
3. 输出完整数据供回测使用
"""

import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 手动修正的代码映射（3只名称不匹配）
MANUAL_CODES = {
    "芯基微装": ("688630", "芯碁微装"),
    "国恩科技": ("002768", "国恩股份"),
    "Fortior 峰绍科技": ("688279", "峰岹科技"),
}


@contextmanager
def quote_ctx():
    from futu import OpenQuoteContext
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        yield ctx
    finally:
        ctx.close()


def parse_md_data():
    """解析用户markdown表格"""
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

        name = cols[1]
        # 修正名称
        if name in MANUAL_CODES:
            code, correct_name = MANUAL_CODES[name]
        else:
            code, correct_name = None, name

        data.append({
            "idx": idx,
            "name": name,
            "correct_name": correct_name,
            "a_code": code,
            "return_1d": num(cols[2]),
            "list_date": cols[3],
            "issue_date": cols[4],
            "ret45": num(cols[5]),
            "cs": num(cols[6]),
            "cap_hkd": num(cols[7]),
            "ha_premium": num(cols[8]),
            "current_premium": num(cols[9]) if len(cols) > 9 else np.nan,
        })
    return pd.DataFrame(data)


def fetch_real_data(code, issue_date_str):
    """从富途OpenD获取真实市场数据"""
    from futu import KLType, RET_OK

    if code is None:
        return {}

    try:
        ci = int(code)
        futu_code = f"SH.{code}" if ci >= 600000 else f"SZ.{code}"
    except ValueError:
        return {}

    # 解析发行日期
    try:
        issue_dt = datetime.strptime(issue_date_str, "%Y/%m/%d")
    except ValueError:
        return {}

    # 获取发行日前120天的K线
    end = issue_dt.strftime("%Y-%m-%d")
    start = (issue_dt - timedelta(days=180)).strftime("%Y-%m-%d")

    result = {}

    with quote_ctx() as ctx:
        # 1. K线数据
        ret, kline, _ = ctx.request_history_kline(futu_code, start=start, end=end, ktype=KLType.K_DAY)
        if ret == RET_OK and len(kline) >= 30:
            closes = kline["close"].values
            # 45日涨幅 (发行日前45日→前1日)
            if len(closes) >= 46:
                ret45_real = (closes[-1] - closes[-46]) / closes[-46] * 100
            else:
                ret45_real = (closes[-1] - closes[0]) / closes[0] * 100

            high, low = kline["high"].max(), kline["low"].min()
            pos = (closes[-1] - low) / (high - low) if high > low else 0.5

            result["ret45_real"] = ret45_real
            result["pos_real"] = pos

            # 2. Beta/Alpha (vs 沪深300)
            ret_b, bench, _ = ctx.request_history_kline("SH.000300", start=start, end=end, ktype=KLType.K_DAY)
            if ret_b == RET_OK and len(bench) >= 20:
                min_len = min(len(closes), len(bench["close"].values))
                sr = pd.Series(closes[-min_len:]).pct_change().dropna().values
                br = pd.Series(bench["close"].values[-min_len:]).pct_change().dropna().values
                n = min(len(sr), len(br))
                if n > 20:
                    sr, br = sr[-n:], br[-n:]
                    beta = np.cov(sr, br)[0, 1] / np.var(br) if np.var(br) > 0 else 1
                    alpha = np.mean(sr - beta * br) * 252
                    result["beta_real"] = beta
                    result["alpha_real"] = alpha

        # 3. 市值 (用总股本×最新收盘价)
        ret_v, val = ctx.get_market_snapshot([futu_code])
        if ret_v == RET_OK and len(val) > 0:
            row = val.iloc[0]
            # 优先用 issued_shares × last_price
            shares = None
            price = row.get("last_price") or row.get("prev_close_price")
            if "issued_shares" in row and pd.notna(row["issued_shares"]):
                shares = float(row["issued_shares"])
            elif "outstanding_shares" in row and pd.notna(row["outstanding_shares"]):
                shares = float(row["outstanding_shares"])

            if shares and price and price > 0:
                cap_yi = shares * price / 1e8  # 亿元
                result["cap_real"] = cap_yi
            else:
                # 回退到 total_market_val
                tmv = row.get("total_market_val")
                if tmv and pd.notna(tmv):
                    result["cap_real"] = float(tmv) / 1e8

    return result


def main():
    print("=" * 60)
    print("🔧 AH股IPO数据增强工具 V2")
    print("=" * 60)

    # 1. 解析基础数据
    df = parse_md_data()
    print(f"\n📋 解析到 {len(df)} 只股票")

    # 2. 对于没有手动代码的，尝试从之前的缓存匹配
    cache_file = os.path.expanduser("~/.ah_ipo_stock_cache.csv")
    if os.path.exists(cache_file):
        stock_db = pd.read_csv(cache_file)
        for idx, row in df.iterrows():
            if pd.isna(row["a_code"]):
                name = row["name"]
                # 精确匹配
                match = stock_db[stock_db["name"] == name]
                if len(match) == 0:
                    # 模糊匹配
                    for _, srow in stock_db.iterrows():
                        if name in srow["name"] or srow["name"] in name:
                            code = str(srow["code"]).split(".")[-1]
                            df.at[idx, "a_code"] = code
                            df.at[idx, "correct_name"] = srow["name"]
                            break
                else:
                    code = str(match.iloc[0]["code"]).split(".")[-1]
                    df.at[idx, "a_code"] = code
                    df.at[idx, "correct_name"] = match.iloc[0]["name"]

    # 3. 从富途获取真实数据
    print("\n🔄 从富途OpenD获取真实市场数据...")
    real_data_list = []

    for _, row in df.iterrows():
        code = row["a_code"]
        name = row["correct_name"]
        issue_date = row["issue_date"]

        if pd.isna(code):
            print(f"  ⚠️ {row['name']}: 无代码，跳过")
            real_data_list.append({})
            continue

        print(f"  📊 {name} ({code}) ...", end=" ")
        try:
            data = fetch_real_data(str(int(code)), issue_date)
            real_data_list.append(data)
            fields = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in data.items()])
            print(f"✅ {fields}")
        except Exception as e:
            print(f"❌ {e}")
            real_data_list.append({})

    # 合并数据
    for i, data in enumerate(real_data_list):
        for k, v in data.items():
            df.at[i, k] = v

    # 4. 对比原始ret45 vs 真实ret45
    print("\n" + "=" * 60)
    print("📊 ret45 数据质量检查")
    print("=" * 60)

    df["ret45_diff"] = df["ret45_real"] - df["ret45"]
    df["ret45_diff_abs"] = df["ret45_diff"].abs()

    for _, row in df.iterrows():
        if pd.notna(row.get("ret45_real")):
            diff = row["ret45_diff"]
            flag = "✅" if abs(diff) < 2 else "⚠️" if abs(diff) < 5 else "❌"
            print(f"  {flag} {row['correct_name']:<10} 原始={row['ret45']:>+7.2f}%  富途={row['ret45_real']:>+7.2f}%  Δ={diff:>+5.2f}%")

    # 5. 导出完整数据
    os.makedirs("output", exist_ok=True)
    output = f"output/ah_ipo_enriched_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n💾 完整数据已保存: {output}")

    # 6. 打印代码映射表
    print("\n" + "=" * 60)
    print("📋 A股代码映射表（可直接复制使用）")
    print("=" * 60)
    for _, row in df.iterrows():
        code = row["a_code"]
        name = row["correct_name"]
        orig = row["name"]
        if orig != name:
            print(f"  {orig:<15} → {name:<10} ({code})")
        else:
            print(f"  {name:<15} ({code})")


if __name__ == "__main__":
    main()
