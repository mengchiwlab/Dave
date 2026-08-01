#!/bin/bash
# 一键部署AH评分系统到Netlify
# 运行: bash deploy_netlify.sh

echo "📦 AH股IPO评分系统 V6.1 - 一键部署"
echo "========================================"

# 检查是否安装了netlify-cli
if ! command -v netlify &> /dev/null; then
    echo "⚠️  正在安装 Netlify CLI..."
    npm install -g netlify-cli
fi

# 创建工作目录
DEPLOY_DIR="/tmp/ah_ipo_deploy"
mkdir -p "$DEPLOY_DIR"

# 复制HTML文件
cp "/Users/wangmengchi/Desktop/AH评分/ah_ipo_v60_web.html" "$DEPLOY_DIR/index.html"

cd "$DEPLOY_DIR"

# 部署
echo "🚀 正在部署..."
netlify deploy --prod --dir . --site AH-IPO-Scorer 2>/dev/null || netlify deploy --prod --dir .

echo ""
echo "✅ 部署完成！把上面的链接发给朋友即可。"
