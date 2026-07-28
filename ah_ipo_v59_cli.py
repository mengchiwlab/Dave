#!/usr/bin/env python3
"""
AH股IPO评分系统 V5.9 CLI版
改进: 修复折扣率bug | argparse | 连接池 | 输入验证 | 离线模式 | 结果导出

运行: python3 ah_ipo_v58_cli.py
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

# ========== 数据库 ==========
CS_DB = {
    "GIC": 10, "高瓴": 9, "摩根大通": 9, "淡马锡": 8, "贝莱德": 8,
    "橡树资本": 7, "红杉": 7, "腾讯": 6, "阿里": 6, "中国人寿": 5,
    "国企结构调整": 5, "知名机构": 4, "产业资本": 3, "无知名基石": 0
}
SP_DB = {
    "高盛": 3, "摩根士丹利": 3, "中金": 3, "中信证券": 3,
    "华泰": 2, "海通": 2, "中信建投": 1.5, "国泰君安": 1.5,
    "招银": 1, "农银": 1, "其他中资": 1, "小型券商": 0.5
}
IND_DB = {
    "半导体/芯片/AI": 5, "新能源/光伏/储能": 4, "云计算/软件/通信": 4,
    "机器人/精密制造": 4, "创新药/医疗器械": 4, "新材料/军工": 3,
    "食品饮料/消费": 2, "医药": 2, "银行/保险/地产": 1, "化工/钢铁/纺织": 1
}


# ========== 富途连接管理 ==========
@contextmanager
def quote_context(host="127.0.0.1", port=11111):
    """安全的富途OpenD连接上下文管理器"""
    ctx = None
    try:
        from futu import OpenQuoteContext
        ctx = OpenQuoteContext(host=host, port=port)
        yield ctx
    finally:
        if ctx:
            ctx.close()


# ========== 评分函数 V5.9 ==========
def calc_score_v58(data):
    """
    计算IPO评分
    data字段: alpha, beta, ret45, pos, discount(%), cs(%), sp, ind, cap(亿元)
    """
    a = data["alpha"]
    b = data["beta"]
    r = data["ret45"]
    p = data["pos"]
    disc = data["discount"]          # 现在统一用百分比，如 30 表示 30%
    cs = data["cs"]
    sp = data["sp"]
    ind = data["ind"]
    cap = data.get("cap")

    s_ret = 17 if r > 15 else 14 if r > 5 else 10 if r > -5 else 6 if r > -15 else 3
    s_beta = 21 if b > 2 else 16 if b > 1.5 else 11 if b > 1 else 6 if b > 0.5 else 3
    s_cs = 15 if cs > 50 else 12 if cs > 40 else 8 if cs > 30 else 4
    s_pos = 10 if p > 0.8 else 8 if p > 0.6 else 6 if p > 0.4 else 3 if p > 0.2 else 1
    s_alpha = 2 if a > 1 else 1.5 if a > 0.5 else 1 if a > 0 else 0.5 if a > -0.5 else 0
    # 修复: discount 直接使用百分比，不再 /100
    s_disc = 9 if disc > 50 else 7 if disc > 40 else 5 if disc > 30 else 2 if disc > 20 else 1
    s_ind = ind * 3.4  # 行业得分满分约17分 (5*3.4)
    s_cap = 4 if cap and cap < 200 else 3 if cap and cap < 1000 else 2 if cap and cap < 5000 else 1
    s_sp = sp * 1.67  # 保荐人满分约5分 (3*1.67)

    total = s_ret + s_beta + s_cs + s_pos + s_alpha + s_disc + s_ind + s_cap + s_sp

    factors = [
        ("A股涨跌幅", s_ret, 17, f"45日: {r:.1f}%"),
        ("Beta(波动)", s_beta, 21, f"β={b:.2f}"),
        ("基石占比", s_cs, 15, f"{cs:.1f}%"),
        ("价格位置", s_pos, 10, f"分位: {p * 100:.0f}%"),
        ("Alpha", s_alpha, 2, f"α={a * 100:.1f}%"),
        ("AH折价率", s_disc, 9, f"{disc:.1f}%"),
        ("行业", s_ind, 17, ""),
        ("市值规模", s_cap, 4, f"{cap:.0f}亿" if cap else "N/A"),
        ("保荐人", s_sp, 5, ""),
    ]
    return total, factors


def fetch_opend_data(stock_code, min_days=60):
    """从富途OpenD获取A股数据"""
    try:
        from futu import KLType, RET_OK
    except ImportError:
        print("错误: 未安装futu-api，请运行: pip install futu-api")
        return None

    with quote_context() as quote_ctx:
        ci = int(stock_code)
        futu_code = f"SH.{stock_code}" if ci >= 600000 else f"SZ.{stock_code}"

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        ret, kline, _ = quote_ctx.request_history_kline(
            futu_code, start=start, end=end, ktype=KLType.K_DAY
        )
        if ret != RET_OK or len(kline) < min_days:
            print(f"警告: K线数据不足(仅{len(kline)}天，需至少{min_days}天)")
            return None

        closes = kline["close"].values
        ret45 = (closes[-1] - closes[-46]) / closes[-46] * 100 if len(closes) >= 46 else (
            (closes[-1] - closes[0]) / closes[0] * 100
        )
        high, low = kline["high"].max(), kline["low"].min()
        pos = (closes[-1] - low) / (high - low) if high > low else 0.5

        # 计算Alpha/Beta
        ret_b, bench, _ = quote_ctx.request_history_kline(
            "SH.000300", start=start, end=end, ktype=KLType.K_DAY
        )
        alpha, beta = 0, 1
        if ret_b == RET_OK and len(bench) >= 20:
            min_len = min(len(closes), len(bench["close"].values))
            sr = pd.Series(closes[-min_len:]).pct_change().dropna().values
            br = pd.Series(bench["close"].values[-min_len:]).pct_change().dropna().values
            n = min(len(sr), len(br))
            if n > 20:
                sr, br = sr[-n:], br[-n:]
                beta = np.cov(sr, br)[0, 1] / np.var(br) if np.var(br) > 0 else 1
                alpha = np.mean(sr - beta * br) * 252

        # 获取市值
        ret_v, val = quote_ctx.get_market_snapshot(code_list=[futu_code])
        cap_yi = None
        if ret_v == RET_OK and len(val) > 0:
            row = val.iloc[0]
            # 优先用总股本×收盘价计算市值（最准确）
            total_shares = None
            for share_key in ["total_share", "issued_shares", "shares", "total_shares"]:
                if share_key in row:
                    v = pd.to_numeric(row[share_key], errors="coerce")
                    if pd.notna(v) and v > 0:
                        total_shares = float(v)
                        break
            if total_shares and closes[-1] > 0:
                cap_yi = total_shares * closes[-1] / 1e8
            else:
                for cap_key in ["total_market_val", "market_cap", "market_value", "capitalization"]:
                    if cap_key in row:
                        v = pd.to_numeric(row[cap_key], errors="coerce")
                        if pd.notna(v) and v > 0:
                            raw = float(v)
                            cap_yi = raw / 1e8 if raw > 1e8 else raw
                            break

        return {
            "alpha": alpha,
            "beta": beta,
            "ret45": ret45,
            "pos": pos,
            "cap": cap_yi,
        }


@lru_cache(maxsize=1)
def _load_all_stocks():
    """加载全市场股票列表（带本地缓存）"""
    cache_file = os.path.expanduser("~/.ah_ipo_stock_cache.csv")
    cache_ttl_hours = 24

    # 检查缓存
    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mtime).total_seconds() < cache_ttl_hours * 3600:
            try:
                df = pd.read_csv(cache_file)
                return df.to_dict("records")
            except Exception:
                pass

    # 从富途拉取
    try:
        from futu import RET_OK
    except ImportError:
        return []

    print("正在从富途更新股票列表缓存...")
    stocks = []
    with quote_context() as quote_ctx:
        for market in ["SH", "SZ"]:
            ret, data = quote_ctx.get_stock_basicinfo(market=market, stock_type="STOCK")
            if ret == RET_OK and len(data) > 0:
                stocks.extend(data.to_dict("records"))

    # 写入缓存
    try:
        pd.DataFrame(stocks).to_csv(cache_file, index=False)
    except Exception:
        pass

    return stocks


def get_stock_code(name):
    """根据名称查找股票代码"""
    stocks = _load_all_stocks()
    name = name.strip()

    # 精确匹配优先
    for row in stocks:
        code_6 = row["code"].split(".")[-1]
        stock_name = row.get("name", "")
        if name == stock_name:
            return code_6, stock_name

    # 模糊匹配
    for row in stocks:
        code_6 = row["code"].split(".")[-1]
        stock_name = row.get("name", "")
        if name in stock_name or stock_name in name:
            return code_6, stock_name

    return None, None


def resolve_sp(sp_name):
    """解析保荐人分数"""
    sp_name = sp_name.strip()
    for key, score in SP_DB.items():
        if key in sp_name or sp_name in key:
            return score
    return SP_DB.get("其他中资", 1)


def resolve_ind(ind_name):
    """解析行业分数"""
    ind_name = ind_name.strip()
    for key, score in IND_DB.items():
        if any(k in ind_name for k in key.split("/")):
            return score
    return 2  # 默认2分


def grade_and_advice(total):
    """根据总分返回等级和建议"""
    if total >= 80:
        return "A", "强烈推荐参与"
    elif total >= 65:
        return "B+", "推荐参与"
    elif total >= 50:
        return "B", "中性偏积极"
    elif total >= 40:
        return "C", "观望"
    else:
        return "D", "建议回避"


def print_score(name, code, total, factors, grade, advice):
    """打印评分结果"""
    print("=" * 60)
    print(f"  {name} ({code})  AH股IPO评分 V5.9")
    print("=" * 60)
    print(f"\n  总分: {total:.1f} / 100  |  等级: {grade}  |  建议: {advice}")
    print("\n  因子得分明细:")
    print("-" * 60)
    for fname, score, max_score, detail in factors:
        bar_len = int(score / max_score * 20) if max_score > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {fname:<12} {bar} {score:.0f}/{max_score}  {detail}")
    print("=" * 60)


def interactive_mode(args):
    """交互模式"""
    print("=" * 60)
    print("  AH股IPO评分系统 V5.9 CLI")
    print("  基于50只历史IPO回测优化权重 | Beta↑行业↑Alpha↓")
    print("=" * 60)

    # 输入公司名称
    name = input("\n输入A股公司名称: ").strip()
    if not name:
        print("公司名不能为空")
        return None

    print(f"查找 {name} ...")
    code, full_name = get_stock_code(name)
    if not code:
        # 尝试直接作为代码
        if name.isdigit() and len(name) == 6:
            code, full_name = name, name
        else:
            print(f"未找到 '{name}'")
            return None
    else:
        print(f"找到: {full_name} ({code})")

    # 收集参数
    def ask(prompt, default, cast=float):
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"输入无效，使用默认值 {default}")
            return default

    disc = ask("AH折价率(%) [默认30]: ", 30.0)
    cs = ask("基石占比(%) [默认40]: ", 40.0)
    sp = input("保荐人 [默认中金]: ").strip() or "中金"
    ind = input("行业 [默认半导体/芯片/AI]: ").strip() or "半导体/芯片/AI"

    return run_score(full_name, code, disc, cs, sp, ind, args.offline, args.output)


def run_score(name, code, disc, cs, sp, ind, offline=False, output=None):
    """执行评分"""
    data = {"discount": disc, "cs": cs, "sp": resolve_sp(sp), "ind": resolve_ind(ind)}

    if not offline:
        print(f"\n获取 {name} 的A股数据...")
        market_data = fetch_opend_data(code)
        if market_data:
            data.update(market_data)
            print(
                f"Alpha={data['alpha'] * 100:.1f}% Beta={data['beta']:.2f} "
                f"45日={data['ret45']:.1f}% 位置={data['pos'] * 100:.0f}% "
                f"市值={data.get('cap', 'N/A')}亿"
            )
        else:
            print("警告: OpenD数据获取失败，使用默认值。请检查富途牛牛是否启动")
            data.update({"alpha": 0, "beta": 1, "ret45": 0, "pos": 0.5, "cap": None})
    else:
        print("离线模式: 使用默认值")
        data.update({"alpha": 0, "beta": 1, "ret45": 0, "pos": 0.5, "cap": None})

    total, factors = calc_score_v58(data)
    grade, advice = grade_and_advice(total)

    print_score(name, code, total, factors, grade, advice)

    result = {
        "name": name,
        "code": code,
        "total": round(total, 1),
        "grade": grade,
        "advice": advice,
        "factors": [
            {"name": n, "score": s, "max": m, "detail": d}
            for n, s, m, d in factors
        ],
        "inputs": {
            "discount": disc,
            "cs": cs,
            "sp": sp,
            "sp_score": data["sp"],
            "ind": ind,
            "ind_score": data["ind"],
        },
        "market_data": {
            "alpha": round(data["alpha"], 4),
            "beta": round(data["beta"], 2),
            "ret45": round(data["ret45"], 2),
            "pos": round(data["pos"], 4),
            "cap": data.get("cap"),
        },
        "timestamp": datetime.now().isoformat(),
    }

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AH股IPO评分系统 V5.9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  交互模式:  python3 ah_ipo_v58_cli.py
  快速评分:  python3 ah_ipo_v58_cli.py --code 688981 --discount 35 --cs 45
  离线模式:  python3 ah_ipo_v58_cli.py --code 688981 --offline
  导出JSON:  python3 ah_ipo_v58_cli.py --code 688981 --output result.json
        """,
    )
    parser.add_argument("--code", "-c", help="A股6位代码或公司名称")
    parser.add_argument("--discount", "-d", type=float, default=30.0, help="AH折价率%% (默认30)")
    parser.add_argument("--cs", type=float, default=40.0, help="基石占比%% (默认40)")
    parser.add_argument("--sp", "-s", default="中金", help="保荐人 (默认:中金)")
    parser.add_argument("--ind", "-i", default="半导体/芯片/AI", help="行业 (默认:半导体/芯片/AI)")
    parser.add_argument("--offline", "-o", action="store_true", help="离线模式(不连接OpenD)")
    parser.add_argument("--output", "-O", help="导出结果到JSON文件")
    parser.add_argument("--interactive", "-I", action="store_true", help="强制交互模式")

    args = parser.parse_args()

    if args.interactive or not args.code:
        result = interactive_mode(args)
        sys.exit(0 if result else 1)

    # 命令行模式
    code = args.code.strip()
    if code.isdigit() and len(code) == 6:
        full_name = code
    else:
        print(f"查找 {code} ...")
        resolved_code, full_name = get_stock_code(code)
        if not resolved_code:
            print(f"未找到 '{code}'，请确认名称正确或直接使用6位代码")
            sys.exit(1)
        code = resolved_code
        print(f"找到: {full_name} ({code})")

    result = run_score(
        full_name, code,
        args.discount, args.cs, args.sp, args.ind,
        offline=args.offline, output=args.output
    )
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
