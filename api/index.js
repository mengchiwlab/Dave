// Vercel Serverless Function - AH股IPO数据代理
// 部署: 1) 推送到GitHub  2) Vercel导入仓库

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const code = req.query.code;
  if (!code || !/^\d{6}$/.test(code)) {
    res.status(400).json({ error: '请输入6位A股代码' });
    return;
  }

  try {
    const result = await getStockData(code);
    res.status(200).json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};

async function getStockData(code) {
  const market = code.startsWith('6') ? '1' : '0';
  const secid = `${market}.${code}`;

  // 1. K线
  const klineRes = await fetch(
    `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=${secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&end=20500101&lmt=120&_=${Date.now()}`
  );
  const klineJson = await klineRes.json();

  if (!klineJson.data || !klineJson.data.klines || klineJson.data.klines.length < 30) {
    throw new Error('K线数据不足30天');
  }

  const klines = klineJson.data.klines.map(line => {
    const p = line.split(',');
    return { date: p[0], close: +p[2] };
  });

  // 2. 沪深300
  const benchRes = await fetch(
    `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&end=20500101&lmt=120&_=${Date.now()}`
  );
  const benchJson = await benchRes.json();

  if (!benchJson.data || !benchJson.data.klines) {
    throw new Error('沪深300数据获取失败');
  }

  const benchKlines = benchJson.data.klines.map(line => {
    const p = line.split(',');
    return { date: p[0], close: +p[2] };
  });

  // 3. 对齐
  const stockMap = new Map(klines.map(d => [d.date, d]));
  const benchMap = new Map(benchKlines.map(d => [d.date, d]));
  const dates = klines.map(d => d.date).filter(d => benchMap.has(d));
  dates.sort();

  const stockAligned = dates.map(d => stockMap.get(d));
  const benchAligned = dates.map(d => benchMap.get(d));
  const closes = stockAligned.map(d => d.close);
  const benchCloses = benchAligned.map(d => d.close);

  // 4. 计算指标
  const ret45 = closes.length >= 46
    ? (closes[closes.length - 1] - closes[closes.length - 46]) / closes[closes.length - 46] * 100
    : (closes[closes.length - 1] - closes[0]) / closes[0] * 100;

  const high120 = Math.max(...closes);
  const low120 = Math.min(...closes);
  const pos = (closes[closes.length - 1] - low120) / (high120 - low120);

  const stockR = [], benchR = [];
  for (let i = 1; i < stockAligned.length; i++) {
    stockR.push((closes[i] - closes[i-1]) / closes[i-1]);
    benchR.push((benchCloses[i] - benchCloses[i-1]) / benchCloses[i-1]);
  }

  const beta = calcBeta(stockR, benchR);
  const alpha = calcAlpha(stockR, benchR, beta);

  // 5. 名称和市值
  const tencentCode = code.startsWith('6') ? `sh${code}` : `sz${code}`;
  const tencentRes = await fetch(`https://qt.gtimg.cn/q=${tencentCode}`);
  const tencentText = await tencentRes.text();

  let name = code;
  let marketCap = null;
  const m = tencentText.match(/v_[^=]+="([^"]+)"/);
  if (m) {
    const parts = m[1].split('~');
    name = parts[1] || code;
    if (parts[44]) marketCap = parseFloat(parts[44]) / 1e4;
  }

  return {
    code, name,
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
