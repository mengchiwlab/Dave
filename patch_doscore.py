#!/usr/bin/env python3
"""替换HTML中的doScore函数为成熟模型逻辑"""

with open('/Users/wangmengchi/Desktop/AH评分/ah_ipo_v60_web.html', 'r') as f:
    lines = f.readlines()

# 找到doScore和showError的位置
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'function doScore()' in line:
        start_idx = i
    if start_idx is not None and 'function showError(msg)' in line:
        end_idx = i
        break

print(f"doScore: 第{start_idx+1}行 到 第{end_idx}行")

new_doscore = '''function doScore() {
    if (!stockData.code) { showError('请先获取数据'); return; }

    const disc = parseFloat(document.getElementById('discount').value);
    const cs = parseFloat(document.getElementById('cs').value);
    const sp = parseFloat(document.getElementById('sp').value);
    if (isNaN(disc) || disc < 0 || disc > 100) { showError('AH折价率必须在 0-100% 之间'); return; }
    if (isNaN(cs) || cs < 0 || cs > 100) { showError('基石占比必须在 0-100% 之间'); return; }

    const r = stockData.ret45, b = stockData.beta, p = stockData.pos;
    const a = stockData.alpha;
    const vol = stockData.idioVol;
    const cap = stockData.cap || 0;

    const indSelect = document.getElementById('ind');
    const opt = indSelect.options[indSelect.selectedIndex];
    const autoInfo = stockData.indInfo || { label: "其他", weight: 1.0, base: 2 };
    const indChanged = indSelect.selectedIndex !== 0;
    const effectiveLabel = indChanged ? opt.dataset.label : autoInfo.label;
    const effectiveBase = indChanged ? parseInt(opt.value) : autoInfo.base;

    // === V6.4 成熟系数模型 (参考专业产品设计) ===
    // logit = -0.18 + Σ(coefficient × transform(value))
    // 负系数 = 保护因子(值高→破发概率低), 正系数 = 风险因子(值高→破发概率高)

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z)))); }

    const definitions = [
        { key: 'discount', label: 'AH折价率', coef: -0.42, trans: v => (v - 20) / 15, fmt: v => `${v.toFixed(1)}%` },
        { key: 'ret45', label: 'A股45日走势', coef: -0.31, trans: v => v / 15, fmt: v => `${v.toFixed(1)}%` },
        { key: 'volatility', label: '残差年化波动率', coef: 0.34, trans: v => (v - 30) / 15, fmt: v => `${v.toFixed(1)}%` },
        { key: 'beta', label: '市场Beta', coef: 0.18, trans: v => v - 1, fmt: v => v.toFixed(2) },
        { key: 'position', label: 'A股价格位置', coef: 0.17, trans: v => (v*100 - 50) / 30, fmt: v => `${(v*100).toFixed(1)}%` },
        { key: 'cornerstone', label: '基石占比', coef: -0.24, trans: v => (v - 25) / 20, fmt: v => `${v.toFixed(1)}%` },
        { key: 'marketCap', label: '预计上市市值', coef: -0.20, trans: v => Math.log(Math.max(v, 10) / 100), fmt: v => `${v.toFixed(0)}亿港元` },
        { key: 'alpha', label: '年化Alpha', coef: -0.10, trans: v => v*100 / 12, fmt: v => `${(v*100).toFixed(1)}%` },
    ];

    const values = {
        ret45: r, position: p, beta: b, volatility: vol*100,
        alpha: a, discount: disc, cornerstone: cs, marketCap: cap,
    };

    let logit = -0.18;
    const features = [];

    for (const def of definitions) {
        const v = values[def.key];
        if (v === null || !Number.isFinite(v)) continue;
        const t = clamp(def.trans(v), -2.5, 2.5);
        const contrib = def.coef * t;
        logit += contrib;
        features.push({
            label: def.label, contribution: contrib, value: v,
            detail: def.fmt(v), coef: def.coef
        });
    }

    // 行业和保荐人（平滑类别编码）
    const industryMap = { 5: -0.30, 4: -0.15, 3: 0.10, 2: 0.25, 1: 0.40 };
    const sponsorMap = { 3: -0.15, 2: 0.05, 1.5: 0.10, 1: 0.15, 0.5: 0.25 };
    const indContrib = industryMap[effectiveBase] || 0.20;
    const spContrib = sponsorMap[sp] || 0.15;
    logit += indContrib + spContrib;

    features.push({ label: '行业历史风险', contribution: indContrib, value: null, detail: () => effectiveLabel });
    features.push({ label: '保荐人历史风险', contribution: spContrib, value: null, detail: () => '' });

    const rawProb = sigmoid(logit);
    const probability = Math.max(0.03, Math.min(0.92, rawProb));
    const score = 100 * (1 - probability);

    let grade, advice, boxClass, riskNote;
    if (probability < 0.10) {
        grade = 'A'; advice = '模型预测破发风险很低'; boxClass = 'grade-a'; riskNote = '多数保护性因子占优，但仍需独立判断';
    } else if (probability < 0.20) {
        grade = 'B+'; advice = '模型预测破发风险较低'; boxClass = 'grade-bp'; riskNote = '保护性因子较好，注意市场波动';
    } else if (probability < 0.35) {
        grade = 'B'; advice = '模型预测存在一定破发风险'; boxClass = 'grade-b'; riskNote = '多空因子交织，需结合更多信息';
    } else if (probability < 0.50) {
        grade = 'C'; advice = '模型预测破发风险偏高'; boxClass = 'grade-c'; riskNote = '风险因子占优，建议谨慎';
    } else {
        grade = 'D'; advice = '模型预测破发风险较高'; boxClass = 'grade-d'; riskNote = '多项风险因子显著，建议回避或极小仓位';
    }

    // 分类正面/负面因子
    const protective = features.filter(f => f.contribution < 0).sort((a, b) => a.contribution - b.contribution).slice(0, 3);
    const risky = features.filter(f => f.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 3);

    document.getElementById('resName').textContent = `${stockData.name} (${stockData.code})`;
    document.getElementById('resScore').textContent = score.toFixed(0);
    document.getElementById('resGrade').textContent = `等级 ${grade}`;
    document.getElementById('resAdvice').textContent = advice;
    document.getElementById('scoreBox').className = 'score-box ' + boxClass;

    // 顶部概率展示
    let html = `
        <tr style="background:#f0f7ff"><td colspan="4" style="padding:16px;text-align:center">
            <div style="font-size:14px;color:#555;margin-bottom:8px">📊 估算首日破发概率</div>
            <div style="font-size:48px;font-weight:800;color:${probability < 0.20 ? '#00c853' : probability < 0.35 ? '#ffab00' : '#dd2c00'}">${(probability*100).toFixed(1)}%</div>
            <div style="font-size:13px;color:#888;margin-top:4px">评分 ${score.toFixed(0)} / 100</div>
            <div style="font-size:11px;color:#aaa;margin-top:4px">成熟系数模型 · 数据截至 ${new Date().toLocaleDateString('zh-CN')}</div>
        </td></tr>
    `;

    // 正面/负面因子摘要
    if (protective.length > 0) {
        html += `<tr style="background:#e8f5e9"><td colspan="4" style="padding:10px 12px;font-size:13px;color:#2e7d32">
            <b>主要降低风险：</b>${protective.map(f => `${f.label} (${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(2)})`).join('、')}
        </td></tr>`;
    }
    if (risky.length > 0) {
        html += `<tr style="background:#ffebee"><td colspan="4" style="padding:10px 12px;font-size:13px;color:#c62828">
            <b>主要提高风险：</b>${risky.map(f => `${f.label} (${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(2)})`).join('、')}
        </td></tr>`;
    }

    // 因子明细表（按贡献绝对值排序）
    const ordered = [...features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    const maxContrib = Math.max(...ordered.map(f => Math.abs(f.contribution)), 0.01);

    for (const f of ordered) {
        const isRisk = f.contribution > 0;
        const width = Math.max(3, Math.abs(f.contribution) / maxContrib * 100);
        const detail = f.value === null ? (typeof f.detail === 'function' ? f.detail() : f.detail) : f.detail(f.value);
        html += `<tr>
            <td>${f.label}</td>
            <td><div class="progress-bar"><div class="progress-fill" style="width:${width.toFixed(0)}%;background:${isRisk ? '#dd2c00' : '#00c853'}"></div></div></td>
            <td class="score-cell" style="color:${isRisk ? '#c62828' : '#2e7d32'}">${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(2)}</td>
            <td style="color:#888;font-size:12px">${detail}</td>
        </tr>`;
    }

    html += `<tr style="background:#f8f9fa"><td colspan="4" style="padding:12px;font-size:12px;color:#888">
        <div style="color:#c62828;font-size:11px;margin-bottom:4px">⚠️ ${riskNote}</div>
        <div>该概率由成熟系数模型估算，基于市场Beta/折价率/波动率/A股走势等因子。结果仅用于研究，不构成投资建议。</div>
    </td></tr>`;

    document.getElementById('factorsBody').innerHTML = html;
    document.getElementById('result').style.display = 'block';
}
'''

# 构建新文件
new_lines = lines[:start_idx] + [new_doscore] + lines[end_idx:]

with open('/Users/wangmengchi/Desktop/AH评分/ah_ipo_v60_web.html', 'w') as f:
    f.writelines(new_lines)

print("✅ doScore函数已替换")
