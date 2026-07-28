#!/usr/bin/env python3
"""
AH股IPO评分系统 V5.9 Streamlit版
基于50只历史IPO回测优化权重 | Beta↑行业↑Alpha↓

安装依赖: pip3 install streamlit pandas numpy
运行: streamlit run ah_ipo_v58_streamlit.py
会自动打开浏览器，或手动访问 http://localhost:8501
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

# ========== 页面配置 ==========
st.set_page_config(
    page_title="AH股IPO评分系统 V5.9",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 数据库 ==========
CS_DB = {
    "GIC": 10, "高瓴": 9, "摩根大通": 9, "淡马锡": 8, "贝莱德": 8,
    "橡树资本": 7, "红杉": 7, "腾讯": 6, "阿里": 6, "中国人寿": 5,
    "国企结构调整": 5, "知名机构": 4, "产业资本": 3, "无知名基石": 0
}
SP_OPTIONS = [
    ("高盛", 3), ("摩根士丹利", 3), ("中金", 3), ("中信证券", 3),
    ("华泰", 2), ("海通", 2), ("中信建投", 1.5), ("国泰君安", 1.5),
    ("招银", 1), ("农银", 1), ("其他中资", 1), ("小型券商", 0.5)
]
SP_DB = dict(SP_OPTIONS)

IND_OPTIONS = [
    ("半导体/芯片/AI", 5),
    ("新能源/光伏/储能", 4),
    ("云计算/软件/通信", 4),
    ("机器人/精密制造", 4),
    ("创新药/医疗器械", 4),
    ("新材料/军工", 3),
    ("食品饮料/消费", 2),
    ("医药", 2),
    ("银行/保险/地产", 1),
    ("化工/钢铁/纺织", 1),
]
IND_DB = dict(IND_OPTIONS)

# ========== 富途连接管理 ==========
@contextmanager
def quote_context(host="127.0.0.1", port=11111):
    ctx = None
    try:
        from futu import OpenQuoteContext
        ctx = OpenQuoteContext(host=host, port=port)
        yield ctx
    finally:
        if ctx:
            ctx.close()


# ========== 评分函数 V5.9 (回测优化版) ==========
def calc_score_v58(data):
    a = data["alpha"]
    b = data["beta"]
    r = data["ret45"]
    p = data["pos"]
    disc = data["discount"]
    cs = data["cs"]
    sp = data["sp"]
    ind = data["ind"]
    cap = data.get("cap")

    s_ret = 17 if r > 15 else 14 if r > 5 else 10 if r > -5 else 6 if r > -15 else 3
    s_beta = 21 if b > 2 else 16 if b > 1.5 else 11 if b > 1 else 6 if b > 0.5 else 3
    s_cs = 15 if cs > 50 else 12 if cs > 40 else 8 if cs > 30 else 4
    s_pos = 10 if p > 0.8 else 8 if p > 0.6 else 6 if p > 0.4 else 3 if p > 0.2 else 1
    s_alpha = 2 if a > 1 else 1.5 if a > 0.5 else 1 if a > 0 else 0.5 if a > -0.5 else 0
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
    try:
        from futu import KLType, RET_OK
    except ImportError:
        return None, "未安装futu-api: pip3 install futu-api"

    with quote_context() as quote_ctx:
        ci = int(stock_code)
        futu_code = f"SH.{stock_code}" if ci >= 600000 else f"SZ.{stock_code}"

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        ret, kline, _ = quote_ctx.request_history_kline(
            futu_code, start=start, end=end, ktype=KLType.K_DAY
        )
        if ret != RET_OK or len(kline) < min_days:
            return None, f"K线数据不足(仅{len(kline)}天)"

        closes = kline["close"].values
        ret45 = (closes[-1] - closes[-46]) / closes[-46] * 100 if len(closes) >= 46 else (
            (closes[-1] - closes[0]) / closes[0] * 100
        )
        high, low = kline["high"].max(), kline["low"].min()
        pos = (closes[-1] - low) / (high - low) if high > low else 0.5

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
                cap_yi = total_shares * closes[-1] / 1e8  # 股数 × 股价(元) ÷ 1亿 = 亿元
            else:
                # 回退：从市值字段解析
                for cap_key in ["total_market_val", "market_cap", "market_value", "capitalization"]:
                    if cap_key in row:
                        v = pd.to_numeric(row[cap_key], errors="coerce")
                        if pd.notna(v) and v > 0:
                            # 富途不同字段单位不同：有的已经是亿，有的是元
                            # 用股价反推校验：如果结果>10倍股价×股本，说明单位不对
                            raw = float(v)
                            # 尝试两种解释：直接当亿 / 当元再除1e8
                            if raw > 1e8:
                                cap_yi = raw / 1e8  # 当元处理
                            else:
                                cap_yi = raw  # 当亿处理
                            break

        return {
            "alpha": alpha,
            "beta": beta,
            "ret45": ret45,
            "pos": pos,
            "cap": cap_yi,
        }, None


@lru_cache(maxsize=1)
def _load_all_stocks():
    cache_file = os.path.expanduser("~/.ah_ipo_stock_cache.csv")
    cache_ttl_hours = 24

    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mtime).total_seconds() < cache_ttl_hours * 3600:
            try:
                df = pd.read_csv(cache_file)
                return df.to_dict("records")
            except Exception:
                pass

    try:
        from futu import RET_OK
    except ImportError:
        return []

    stocks = []
    with quote_context() as quote_ctx:
        for market in ["SH", "SZ"]:
            ret, data = quote_ctx.get_stock_basicinfo(market=market, stock_type="STOCK")
            if ret == RET_OK and len(data) > 0:
                stocks.extend(data.to_dict("records"))

    try:
        pd.DataFrame(stocks).to_csv(cache_file, index=False)
    except Exception:
        pass

    return stocks


def get_stock_code(name):
    stocks = _load_all_stocks()
    name = name.strip()

    for row in stocks:
        code_6 = row["code"].split(".")[-1]
        stock_name = row.get("name", "")
        if name == stock_name:
            return code_6, stock_name

    for row in stocks:
        code_6 = row["code"].split(".")[-1]
        stock_name = row.get("name", "")
        if name in stock_name or stock_name in name:
            return code_6, stock_name

    return None, None


def grade_and_advice(total):
    if total >= 80:
        return "A", "强烈推荐参与", "🟢"
    elif total >= 65:
        return "B+", "推荐参与", "🟢"
    elif total >= 50:
        return "B", "中性偏积极", "🟡"
    elif total >= 40:
        return "C", "观望", "🟠"
    else:
        return "D", "建议回避", "🔴"


# ========== Streamlit UI ==========
def main():
    st.title("📊 AH股IPO评分系统 V5.9")
    st.caption("基于50只历史IPO回测优化权重 | Beta↑行业↑Alpha↓")

    # 侧边栏 - 输入参数
    with st.sidebar:
        st.header("📝 输入参数")

        stock_input = st.text_input(
            "公司名称或A股6位代码",
            placeholder="例如: 中芯国际 或 688981",
            help="支持公司名称模糊搜索，或直接输入6位股票代码"
        )

        discount = st.slider("AH折价率 (%)", 0, 100, 30, 1)
        cs_ratio = st.slider("基石占比 (%)", 0, 100, 40, 1)

        sp_name = st.selectbox(
            "保荐人",
            options=[k for k, _ in SP_OPTIONS],
            index=2,  # 默认中金
        )

        ind_name = st.selectbox(
            "行业",
            options=[k for k, _ in IND_OPTIONS],
            index=0,  # 默认半导体
        )

        # 发行市值 - AH股IPO评分用H股发行市值，不是A股总市值
        cap_hkd = st.number_input(
            "H股发行市值 (亿港元)",
            min_value=1.0,
            max_value=50000.0,
            value=100.0,
            step=10.0,
            help="从招股书获取的H股发行市值，不是A股当前总市值"
        )

        offline_mode = st.checkbox("离线模式 (不连接富途OpenD)", value=False)

        st.divider()
        st.info("💡 正常使用请确保富途牛牛已启动\nOpenD服务在运行(默认端口11111)")

    # 主区域 - 评分按钮和结果
    col1, col2 = st.columns([1, 3])

    with col1:
        score_btn = st.button("🚀 开始评分", type="primary", use_container_width=True)

    if score_btn:
        if not stock_input or not stock_input.strip():
            st.error("请输入公司名称或6位代码")
            return

        stock_input = stock_input.strip()

        # 解析代码
        if stock_input.isdigit() and len(stock_input) == 6:
            code, name = stock_input, stock_input
        else:
            with st.spinner(f"查找 {stock_input} ..."):
                code, name = get_stock_code(stock_input)
            if not code:
                st.error(f"未找到 '{stock_input}'")
                return
            st.success(f"找到: {name} ({code})")

        sp_score = SP_DB.get(sp_name, 1)
        ind_score = IND_DB.get(ind_name, 2)

        data = {
            "discount": discount,
            "cs": cs_ratio,
            "sp": sp_score,
            "ind": ind_score,
        }

        # 获取市场数据
        if not offline_mode:
            with st.spinner(f"获取 {name} 的A股数据..."):
                md, err = fetch_opend_data(code)
            if md:
                data.update(md)
                st.info(
                    f"Alpha={data['alpha']*100:.1f}% | Beta={data['beta']:.2f} | "
                    f"45日={data['ret45']:.1f}% | 位置={data['pos']*100:.0f}% | "
                    f"市值={data.get('cap', 'N/A')}亿"
                )
            else:
                st.warning(f"⚠️ OpenD获取失败: {err}，使用默认值")
                data.update({"alpha": 0, "beta": 1, "ret45": 0, "pos": 0.5, "cap": None})
        else:
            st.info("离线模式: 使用默认值")
            data.update({"alpha": 0, "beta": 1, "ret45": 0, "pos": 0.5, "cap": None})

        # 使用用户输入的H股发行市值（优先），覆盖OpenD的A股总市值
        data["cap"] = cap_hkd * 0.92  # 亿港元 → 人民币亿元

        # 计算评分
        total, factors = calc_score_v58(data)
        grade, advice, emoji = grade_and_advice(total)

        # 显示总分
        st.divider()
        score_col, info_col = st.columns([1, 2])

        with score_col:
            st.metric(label="总分", value=f"{total:.1f}", delta=f"等级 {grade}")
            st.subheader(f"{emoji} {advice}")

        with info_col:
            st.write(f"**{name} ({code})**")
            st.write(f"保荐人: {sp_name} ({sp_score}分) | 行业: {ind_name} ({ind_score}分)")
            st.write(f"AH折价率: {discount}% | 基石占比: {cs_ratio}%")

        # 因子明细表格
        st.divider()
        st.subheader("📋 因子得分明细")

        factor_data = []
        for fname, score, max_score, detail in factors:
            pct = score / max_score * 100 if max_score > 0 else 0
            factor_data.append({
                "因子": fname,
                "得分": score,
                "满分": max_score,
                "得分率": f"{pct:.0f}%",
                "详情": detail,
            })

        df_factors = pd.DataFrame(factor_data)
        st.dataframe(
            df_factors,
            column_config={
                "得分率": st.column_config.ProgressColumn(
                    "得分率",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # 导出JSON
        st.divider()
        result_json = json.dumps({
            "name": name,
            "code": code,
            "total": round(total, 1),
            "grade": grade,
            "advice": advice,
            "factors": [{"name": n, "score": s, "max": m, "detail": d} for n, s, m, d in factors],
            "inputs": {"discount": discount, "cs": cs_ratio, "sp": sp_name, "ind": ind_name},
            "market_data": {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items() if k not in ["discount", "cs", "sp", "ind"]},
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2)

        filename = f"AH_IPO_{name}_{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            label="⬇️ 下载JSON结果",
            data=result_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
        )

    else:
        # 初始状态
        st.info("👈 在左侧输入参数，然后点击 **开始评分**")

        # 展示示例
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 如何使用
            1. **输入公司名称或6位代码** — 支持模糊搜索，如"中芯"会匹配"中芯国际"
            2. **设置参数** — AH折价率、基石占比、保荐人、行业
            3. **点击开始评分** — 系统自动从富途OpenD获取A股数据
            4. **查看结果** — 总分、等级、各因子得分明细
            5. **导出JSON** — 保存评分结果到本地

            ### 离线模式
            如果富途牛牛未启动，勾选"离线模式"，系统会使用默认值进行评分：
            - Alpha = 0, Beta = 1
            - 45日涨幅 = 0%
            - 价格位置 = 50%

            ### 评分等级
            | 总分 | 等级 | 建议 |
            |------|------|------|
            | ≥80 | A | 强烈推荐参与 |
            | ≥65 | B+ | 推荐参与 |
            | ≥50 | B | 中性偏积极 |
            | ≥40 | C | 观望 |
            | <40 | D | 建议回避 |
            """)


if __name__ == "__main__":
    main()
