// Cloudflare Worker - AH股IPO数据代理
// 部署地址: https://你的用户名.workers.dev

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS 允许所有来源
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    const code = url.searchParams.get('code');
    if (!code || !/^\d{6}$/.test(code)) {
      return jsonResponse({ error: '请输入6位A股代码' }, 400, corsHeaders);
    }

    try {
      const result = await getStockData(code);
      return jsonResponse(result, 200, corsHeaders);
    } catch (e) {
      return jsonResponse({ error: e.message }, 500, corsHeaders);
    }
  }
};

async function getStockData(code) {
  const market = code.startsWith('6') ? '1' : '0';
  const secid = `${market}.${code}`;

  // 1. 获取股票K线 (120天)
  const klineUrl = `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=${secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&end=20500101&lmt=120&_=${Date.now()}`;
  const klineRes = await fetch(klineUrl);
  const klineJson = await klineRes.json();

  if (!klineJson.data || !klineJson.data.klines || klineJson.data.klines.length < 30) {
    throw new Error('K线数据不足30天');
  }

  const klines = klineJson.data.klines.map(line => {
    const p = line.split(',');
    return { date: p[0], open: +p[1], close: +p[2], high: +p[3], low: +p[4] };
  });

  // 2. 获取沪深300K线 (同期)
  const benchUrl = `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&end=20500101&lmt=120&_=${Date.now()}`;
  const benchRes = await fetch(benchUrl);
  const benchJson = await benchRes.json();

  if (!benchJson.data || !benchJson.data.klines) {
    throw new Error('沪深300数据获取失败');
  }

  const benchKlines = benchJson.data.klines.map(line => {
    const p = line.split(',');
    return { date: p[0], close: +p[2] };
  });

  // 3. 按日期对齐
  const stockMap = new Map(klines.map(d => [d.date, d]));
  const benchMap = new Map(benchKlines.map(d => [d.date, d]));
  const dates = klines.map(d => d.date).filter(d => benchMap.has(d));
  dates.sort();

  const stockAligned = dates.map(d => stockMap.get(d));
  const benchAligned = dates.map(d => benchMap.get(d));

  // 4. 计算指标
  const closes = stockAligned.map(d => d.close);
  const benchCloses = benchAligned.map(d => d.close);

  // 45日涨幅
  const ret45 = closes.length >= 46
    ? (closes.at(-1) - closes.at(-46)) / closes.at(-46) * 100
    : (closes.at(-1) - closes[0]) / closes[0] * 100;

  // 价格位置 (120天区间)
  const high120 = Math.max(...closes);
  const low120 = Math.min(...closes);
  const pos = (closes.at(-1) - low120) / (high120 - low120);

  // Beta & Alpha
  const stockR = []; const benchR = [];
  for (let i = 1; i < stockAligned.length; i++) {
    stockR.push((closes[i] - closes[i-1]) / closes[i-1]);
    benchR.push((benchCloses[i] - benchCloses[i-1]) / benchCloses[i-1]);
  }

  const beta = calcBeta(stockR, benchR);
  const alpha = calcAlpha(stockR, benchR, beta);

  // 5. 获取名称和市值 (腾讯API)
  const tencentCode = code.startsWith('6') ? `sh${code}` : `sz${code}`;
  const tencentRes = await fetch(`https://qt.gtimg.cn/q=${tencentCode}`);
  const tencentText = await tencentRes.text();

  let name = code;
  let marketCap = null;

  // 解析: v_sz300308="51~中际旭创~300308~908.00~..."
  const m = tencentText.match(/v_[^=]+="([^"]+)"/);
  if (m) {
    const parts = m[1].split('~');
    name = parts[1] || code;
    // 总市值(万元) → 亿元
    if (parts[44]) marketCap = parseFloat(parts[44]) / 1e4;
  }

  return {
    code,
    name,
    ret45: Math.round(ret45 * 100) / 100,
    pos: Math.round(pos * 10000) / 10000,
    beta: Math.round(beta * 100) / 100,
    alpha: Math.round(alpha * 100) / 100,
    cap: marketCap ? Math.round(marketCap) : null,
  };
}

function calcBeta(sr, br) {
  const n = sr.length;
  if (n < 2) return 1;
  const ms = sr.reduce((a, b) => a + b, 0) / n;
  const mb = br.reduce((a, b) => a + b, 0) / n;
  let cov = 0, vr = 0;
  for (let i = 0; i < n; i++) {
    cov += (sr[i] - ms) * (br[i] - mb);
    vr += (br[i] - mb) ** 2;
  }
  return vr > 0 ? (cov / n) / (vr / n) : 1;
}

function calcAlpha(sr, br, beta) {
  const n = sr.length;
  if (n < 2) return 0;
  const ms = sr.reduce((a, b) => a + b, 0) / n;
  const mb = br.reduce((a, b) => a + b, 0) / n;
  return (ms - beta * mb) * 252;
}

function jsonResponse(data, status, cors) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  });
}
