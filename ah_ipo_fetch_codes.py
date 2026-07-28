#!/usr/bin/env python3
"""
AH股IPO数据补充工具
1. 用富途OpenD匹配A股6位代码
2. 尝试获取保荐人、基石等IPO详细信息
"""

import os
import re
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache

import pandas as pd


# ========== 富途连接 ==========
@contextmanager
def quote_context(host="127.0.0.1", port=11111):
    ctx = None
    try:
        from futu import OpenQuoteContext
        ctx = OpenQuoteContext(host=host, port=port)
        yield ctx
    except Exception as e:
        print(f"❌ 无法连接富途OpenD: {e}")
        yield None
    finally:
        if ctx:
            ctx.close()


@lru_cache(maxsize=1)
def load_stock_database():
    """加载全市场A股数据库"""
    cache_file = os.path.expanduser("~/.ah_ipo_stock_cache.csv")
    cache_ttl_hours = 24

    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mtime).total_seconds() < cache_ttl_hours * 3600:
            try:
                return pd.read_csv(cache_file)
            except Exception:
                pass

    print("🔄 从富途OpenD更新股票数据库...")
    stocks = []
    with quote_context() as ctx:
        if ctx is None:
            return pd.DataFrame()
        try:
            from futu import RET_OK
            for market in ["SH", "SZ"]:
                ret, data = ctx.get_stock_basicinfo(market=market, stock_type="STOCK")
                if ret == RET_OK and len(data) > 0:
                    stocks.append(data)
            if stocks:
                df = pd.concat(stocks, ignore_index=True)
                df.to_csv(cache_file, index=False)
                print(f"✅ 已缓存 {len(df)} 只股票")
                return df
        except Exception as e:
            print(f"⚠️ 获取失败: {e}")

    return pd.DataFrame()


def find_stock_code(name, stock_db):
    """根据名称查找A股代码"""
    if stock_db.empty:
        return None, None

    # 精确匹配
    for _, row in stock_db.iterrows():
        stock_name = row.get("name", "")
        if name == stock_name:
            code = row["code"].split(".")[-1] if "." in str(row["code"]) else str(row["code"])
            return code, stock_name

    # 模糊匹配（包含关系）
    for _, row in stock_db.iterrows():
        stock_name = row.get("name", "")
        if name in stock_name or stock_name in name:
            code = row["code"].split(".")[-1] if "." in str(row["code"]) else str(row["code"])
            return code, stock_name

    return None, None


def try_get_ipo_detail(ctx, code):
    """尝试从富途获取IPO详细信息"""
    if ctx is None or code is None:
        return {}

    result = {}
    try:
        # 尝试获取市场快照，看看有没有额外字段
        ci = int(code)
        futu_code = f"SH.{code}" if ci >= 600000 else f"SZ.{code}"

        ret, data = ctx.get_market_snapshot([futu_code])
        if ret == 0 and len(data) > 0:
            row = data.iloc[0]
            # 打印所有字段，看看有没有保荐人/基石相关
            result["fields"] = list(row.index)

        # 尝试获取股票基本信息
        market = "SH" if ci >= 600000 else "SZ"
        ret2, data2 = ctx.get_stock_basicinfo(market=market, stock_type="STOCK")
        if ret2 == 0 and len(data2) > 0:
            match = data2[data2["code"].str.contains(code, na=False)]
            if len(match) > 0:
                result["basic_fields"] = list(match.columns)

    except Exception as e:
        result["error"] = str(e)

    return result


def parse_names_from_md():
    """从用户markdown文件解析公司名称"""
    md_path = "/Users/wangmengchi/Documents/远程仓库/锚定评分/ah 股.md"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    names = []
    for line in content.strip().split("\n"):
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        cols = [c for c in cols if c]
        if len(cols) >= 3:
            try:
                idx = int(cols[0])
                name = cols[1]
                if name and name not in ["名称", "---"]:
                    names.append({"idx": idx, "name": name})
            except ValueError:
                continue
    return names


def main():
    print("=" * 60)
    print("🔍 AH股IPO数据补充工具")
    print("=" * 60)

    # 1. 解析公司名称
    name_list = parse_names_from_md()
    print(f"\n📋 从markdown解析到 {len(name_list)} 只股票")

    # 2. 加载股票数据库
    stock_db = load_stock_database()
    if stock_db.empty:
        print("\n❌ 无法加载股票数据库，请确保富途OpenD已启动")
        return

    print(f"\n📊 股票数据库: {len(stock_db)} 只")

    # 3. 匹配代码
    print("\n🔍 开始匹配A股代码...")
    results = []
    matched = 0

    with quote_context() as ctx:
        for item in name_list:
            idx = item["idx"]
            name = item["name"]
            code, full_name = find_stock_code(name, stock_db)

            # 尝试获取IPO详情
            ipo_detail = {}
            if code:
                ipo_detail = try_get_ipo_detail(ctx, code)
                matched += 1

            results.append({
                "idx": idx,
                "name": name,
                "a_code": code,
                "matched_name": full_name,
                "futu_fields": ipo_detail.get("fields", []),
                "basic_fields": ipo_detail.get("basic_fields", []),
            })

            status = "✅" if code else "❌"
            print(f"  {status} {idx:>2}. {name:<12} → {code if code else '未找到'}")

    print(f"\n📈 匹配结果: {matched}/{len(name_list)} 只成功")

    # 4. 分析富途有哪些字段可用
    print("\n" + "=" * 60)
    print("📦 富途OpenD可用字段分析")
    print("=" * 60)

    all_fields = set()
    all_basic = set()
    for r in results:
        all_fields.update(r.get("futu_fields", []))
        all_basic.update(r.get("basic_fields", []))

    print(f"\n  市场快照(get_market_snapshot)字段:")
    for f in sorted(all_fields):
        print(f"    • {f}")

    print(f"\n  基本信息(get_stock_basicinfo)字段:")
    for f in sorted(all_basic):
        print(f"    • {f}")

    # 5. 保存结果
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    output = "output/ah_ipo_code_mapping.csv"
    df[["idx", "name", "a_code", "matched_name"]].to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n💾 代码映射已保存: {output}")

    # 6. 输出未匹配的，需要手动处理
    unmatched = [r for r in results if r["a_code"] is None]
    if unmatched:
        print(f"\n⚠️ 以下 {len(unmatched)} 只未匹配，需要手动查找代码:")
        for r in unmatched:
            print(f"     • {r['name']}")


if __name__ == "__main__":
    main()
