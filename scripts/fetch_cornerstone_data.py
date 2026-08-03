#!/usr/bin/env python3
"""
基石投资者数据爬取脚本 - 从AiPO/港交所获取基石投资者明细
Usage: python3 fetch_cornerstone_data.py <ipo_code>
Output: cornerstones.json
"""

import json
import time
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

# AiPO 新股详情页模板
AIPO_URL_TEMPLATE = "https://www.aipo.com.cn/ipo/{code}"

def fetch_aipo_page(code):
    """获取AiPO新股详情页"""
    url = AIPO_URL_TEMPLATE.format(code=code)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode('utf-8')
    except URLError as e:
        print(f"[ERROR] 无法获取AiPO数据: {e}")
        return None

def generate_default_cornerstone_rules():
    """
    生成基石投资者质量评分规则
    用于前端计算基石质量加权分
    """
    rules = {
        "institution_tiers": {
            # 主权基金 / 国家队
            "GIC":              {"tier": 1, "weight": 1.0,  "name": "新加坡政府投资"},
            "Temasek":          {"tier": 1, "weight": 1.0,  "name": "淡马锡"},
            "CPPIB":            {"tier": 1, "weight": 1.0,  "name": "加拿大养老金"},
            "ADIA":             {"tier": 1, "weight": 1.0,  "name": "阿布扎比投资局"},
            "中国国有企业结构调整基金": {"tier": 1, "weight": 0.95, "name": "国调基金"},
            "中国保险投资基金":       {"tier": 1, "weight": 0.95, "name": "中保投"},
            "深圳国资":          {"tier": 1, "weight": 0.90, "name": "深圳国资系"},
            
            # 顶级对冲基金 / 资管
            "Fidelity":         {"tier": 2, "weight": 0.85, "name": "富达基金"},
            "Capital Group":    {"tier": 2, "weight": 0.85, "name": "资本集团"},
            "BlackRock":        {"tier": 2, "weight": 0.85, "name": "贝莱德"},
            "Vanguard":         {"tier": 2, "weight": 0.85, "name": "先锋领航"},
            "T. Rowe Price":    {"tier": 2, "weight": 0.80, "name": "普信集团"},
            "Oaktree":          {"tier": 2, "weight": 0.80, "name": "橡树资本"},
            "高瓴资本":          {"tier": 2, "weight": 0.80, "name": "Hillhouse"},
            "红杉资本":          {"tier": 2, "weight": 0.80, "name": "Sequoia"},
            
            # 知名产业资本 / 战略投资者
            "腾讯":             {"tier": 3, "weight": 0.75, "name": "腾讯控股"},
            "阿里":             {"tier": 3, "weight": 0.75, "name": "阿里巴巴"},
            "字节":             {"tier": 3, "weight": 0.75, "name": "字节跳动"},
            "小米":             {"tier": 3, "weight": 0.70, "name": "小米集团"},
            "比亚迪":           {"tier": 3, "weight": 0.70, "name": "比亚迪"},
            "宁德时代":         {"tier": 3, "weight": 0.70, "name": "CATL"},
            
            # 一般机构
            "基石":             {"tier": 4, "weight": 0.50, "name": "一般基石投资者"},
            "投资":             {"tier": 4, "weight": 0.45, "name": "一般投资机构"},
            "资管":             {"tier": 4, "weight": 0.45, "name": "资产管理公司"},
            "基金":             {"tier": 4, "weight": 0.40, "name": "一般基金"},
            
            # 低质量/个人
            "个人":             {"tier": 5, "weight": 0.15, "name": "个人投资者"},
            "家族":             {"tier": 5, "weight": 0.20, "name": "家族办公室"},
            "私人":             {"tier": 5, "weight": 0.15, "name": "私人投资者"},
        },
        "lockup_weights": {
            0: 0.6,    # 无锁定
            6: 0.9,    # 6个月
            12: 1.0,   # 12个月（标准）
            18: 1.05,  # 18个月
            24: 1.1,   # 24个月+
        },
        "calculation": {
            "description": "基石质量分 = Σ(每家机构金额占比 × 机构权重 × 锁定期权重)",
            "max_score": 100,
            "example": "GIC认购40% + 6个月锁定 → 0.4 × 1.0 × 0.9 = 0.36"
        }
    }
    return rules

def save_cornerstone_rules(rules, filepath="cornerstone_rules.json"):
    """保存基石评分规则到JSON"""
    output = {
        "update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "手动整理 + 市场惯例",
        "rules": rules
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存基石评分规则到 {filepath}")

def main():
    print("=" * 50)
    print("港股IPO基石投资者评分规则生成工具")
    print("=" * 50)
    
    print("[INFO] 生成基石投资者质量评分规则...")
    rules = generate_default_cornerstone_rules()
    save_cornerstone_rules(rules)
    
    print("\n规则摘要：")
    print(f"- 机构分级: {len(rules['institution_tiers'])} 类")
    print(f"- 锁定期权重: {len(rules['lockup_weights'])} 档")
    print(f"\n顶级机构示例:")
    for name, info in list(rules['institution_tiers'].items())[:5]:
        print(f"  {name}: 权重={info['weight']}, 等级=T{info['tier']}")
    
    print("\n提示：")
    print("1. 此规则文件用于前端计算基石质量加权分")
    print("2. 可根据实际回测结果调整权重")
    print("3. 新增机构可直接编辑 cornerstone_rules.json")

if __name__ == "__main__":
    main()
