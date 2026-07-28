// Vercel Serverless Function - 简化版：只代理原始K线数据
// 前端JS负责计算Beta/Alpha

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
    const result = await getRawData(code);
    res.status(200).json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};

async function getRawData(code) {
  const market = code.startsWith('6') ? '1' : '0';
  const secid = `${market}.${code}`;

  // 1. 获取股票K线 (120天) - 只取close
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

  // 2. 获取沪深300K线
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

  // 3. 获取名称和市值 (腾讯API - 超快)
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
    code,
    name,
    cap: marketCap ? Math.round(marketCap) : null,
    klines,           // 前端计算Beta/Alpha
    benchKlines,      // 前端计算Beta/Alpha
  };
}
