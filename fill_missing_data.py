#!/usr/bin/env python3
"""
补全25只缺失的Beta/Alpha/市值/价格位置数据
使用腾讯财经API（与HTML版本一致）
"""

import json
import urllib.request
import numpy as np
import pandas as pd

def fetch_tencent_klines(code):
    """获取腾讯K线数据"""
    tcode = f"sh{code}" if code.startswith('6') else f"sz{code}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tcode},day,,,120,qfq"
    
    try:
        res = urllib.request.urlopen(url, timeout=10)
        data = json.loads(res.read().decode('utf-8'))
        
        stock_data = data.get('data', {}).get(tcode, {})
        klines = stock_data.get('qfqday') or stock_data.get('day')
        if not klines:
            return None
        
        return [{
            'date': d[0].replace('-', ''),
            'open': float(d[1]),
            'close': float(d[2]),
            'low': float(d[3]),
            'high': float(d[4]),
        } for d in klines]
    except Exception as e:
        print(f"  ⚠️ {code} K线获取失败: {e}")
        return None

def fetch_tencent_bench():
    """获取沪深300"""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,,,120,qfq"
    try:
        res = urllib.request.urlopen(url, timeout=10)
        data = json.loads(res.read().decode('utf-8'))
        klines = data.get('data', {}).get('sh000300', {}).get('qfqday') or data.get('data', {}).get('sh000300', {}).get('day')
        if not klines:
            return None
        return [{'date': d[0].replace('-', ''), 'close': float(d[2])} for d in klines]
    except:
        return None

def fetch_stock_info(code):
    """获取股票信息（市值）"""
    tcode = f"sh{code}" if code.startswith('6') else f"sz{code}"
    url = f"https://qt.gtimg.cn/q={tcode}"
    try:
        res = urllib.request.urlopen(url, timeout=10)
        text = res.read().decode('gbk', errors='ignore')
        m = text.split('~')
        if len(m) > 44:
            return {'name': m[1], 'cap': float(m[44])}  # 亿元
    except:
        pass
    return {'name': code, 'cap': None}

def calc_beta(sr, br):
    n = len(sr)
    if n < 2: return 1.0
    ms, mb = np.mean(sr), np.mean(br)
    cov = np.mean((sr - ms) * (br - mb))
    vr = np.mean((br - mb) ** 2)
    return cov / vr if vr > 0 else 1.0

def calc_alpha(sr, br, beta):
    n = len(sr)
    if n < 2: return 0.0
    return (np.mean(sr) - beta * np.mean(br)) * 252

def calc_idio_vol(sr, br, beta):
    n = len(sr)
    if n < 2: return 0.0
    residuals = sr - beta * br
    return np.std(residuals) * np.sqrt(252)

def process_stock(row, bench_klines):
    """处理单只股票"""
    code = str(int(row['a_code'])).zfill(6)
    name = row['correct_name']
    
    print(f"  {name} ({code})...", end=' ')
    
    klines = fetch_tencent_klines(code)
    if not klines or len(klines) < 45:
        print("K线不足")
        return None
    
    # 对齐
    stock_map = {d['date']: d for d in klines}
    bench_map = {d['date']: d for d in bench_klines}
    dates = sorted([d for d in stock_map if d in bench_map])
    
    if len(dates) < 20:
        print("对齐后数据不足")
        return None
    
    stock_aligned = [stock_map[d] for d in dates]
    bench_aligned = [bench_map[d] for d in dates]
    closes = np.array([d['close'] for d in stock_aligned])
    bench_closes = np.array([d['close'] for d in bench_aligned])
    
    # 45日涨幅
    ret45 = (closes[-1] - closes[-46]) / closes[-46] * 100 if len(closes) >= 46 else (closes[-1] - closes[0]) / closes[0] * 100
    
    # 价格位置
    high, low = closes.max(), closes.min()
    pos = (closes[-1] - low) / (high - low) if high > low else 0.5
    
    # Beta/Alpha
    stock_r = np.diff(closes) / closes[:-1]
    bench_r = np.diff(bench_closes) / bench_closes[:-1]
    beta = calc_beta(stock_r, bench_r)
    alpha = calc_alpha(stock_r, bench_r, beta)
    
    # 市值
    info = fetch_stock_info(code)
    cap = info.get('cap')
    
    print(f"✅ β={beta:.2f} α={alpha:.2f} ret45={ret45:.1f}% 市值={cap:.0f}亿" if cap else f"✅ β={beta:.2f} α={alpha:.2f} ret45={ret45:.1f}%")
    
    return {
        'beta_real': beta,
        'alpha_real': alpha,
        'ret45_real': ret45,
        'pos_real': pos,
        'cap_real': cap,
    }

def main():
    print("=" * 70)
    print("📊 补全25只缺失的Beta/Alpha/市值数据")
    print("=" * 70)
    
    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    missing = df[df['beta_real'].isna()].copy()
    
    print(f"\n需要补全: {len(missing)} 只")
    
    # 获取基准数据
    print("\n获取沪深300数据...")
    bench_klines = fetch_tencent_bench()
    if not bench_klines:
        print("❌ 无法获取沪深300数据")
        return
    print(f"✅ 沪深300: {len(bench_klines)} 条K线")
    
    # 逐个补全
    success = 0
    for idx, row in missing.iterrows():
        result = process_stock(row, bench_klines)
        if result:
            for col, val in result.items():
                df.at[idx, col] = val
            success += 1
    
    print(f"\n✅ 成功补全: {success}/{len(missing)} 只")
    
    # 保存
    df.to_csv("output/ah_ipo_enriched_20260728.csv", index=False, encoding="utf-8-sig")
    print(f"💾 数据已保存")
    
    # 验证
    print(f"\n📋 验证: 有beta_real的样本 = {df['beta_real'].notna().sum()}/50")
    
    # 更新模型
    print("\n" + "=" * 70)
    print("🔄 用50只样本重新训练模型...")
    print("=" * 70)

if __name__ == "__main__":
    main()
