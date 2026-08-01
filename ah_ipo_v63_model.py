#!/usr/bin/env python3
"""
AH股IPO评分系统 V6.3 - 纯NumPy逻辑回归预测破发概率
不依赖sklearn，参数可直接用于前端
"""

import numpy as np
import pandas as pd

# ========== 行业编码 ==========
IND_PATTERNS = {
    "半导体|芯片|集成|微装|澜起|兆易|纳芯|豪威|国民|芯基|芯碁|峰岹": "半导体",
    "新能源|光伏|储能|宁德|钧达|大金重工|天岳|先导": "新能源",
    "软件|通信|科技|剑桥|广和通|美格智能|龙旗|华勤|立讯|蓝思|三一|大族": "科技制造",
    "机器人|精密|制造|埃斯顿|兆威|三环|鼎泰|广合|胜宏|牧原": "制造",
    "医药|医疗|生物|恒瑞|可孚|迈威": "医药",
    "新材料|军工|国恩|沃尔|吉宏|赤峰": "新材料军工",
    "饮料|食品|消费|安井|东鹏|海天": "消费",
    "银行|保险|地产|期货": "金融地产",
    "化工|钢铁|纺织|滨化": "周期",
}

def classify_industry(name):
    for p, label in IND_PATTERNS.items():
        if any(k in name for k in p.split("|")):
            return label
    return "其他"

# ========== 纯NumPy逻辑回归 ==========
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def standardize(X):
    """标准化: (X - mean) / std"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # 避免除零
    return (X - mean) / std, mean, std

def logistic_regression_fit(X, y, lr=0.1, epochs=10000, tol=1e-6):
    """梯度下降训练逻辑回归"""
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    
    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        
        # 梯度
        dw = (1/n) * X.T @ (p - y)
        db = (1/n) * np.sum(p - y)
        
        # 更新
        w -= lr * dw
        b -= lr * db
        
        # 早停
        loss = -np.mean(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8))
        if epoch % 1000 == 0:
            pass  # 不打印
        if epoch > 100 and np.linalg.norm(dw) < tol:
            break
    
    return w, b

def predict_proba(X, w, b):
    z = X @ w + b
    return sigmoid(z)

# ========== Platt Scaling (概率校准) ==========
def platt_scaling(raw_probs, y_true):
    """用sigmoid拟合原始概率到真实概率"""
    # 简单版：用逻辑回归拟合 raw_prob -> y
    raw_probs = np.array(raw_probs).reshape(-1, 1)
    y_true = np.array(y_true)
    
    # 训练一个小逻辑回归
    X_s, mean, std = standardize(raw_probs)
    w, b = logistic_regression_fit(X_s, y_true, lr=0.5, epochs=5000)
    
    def calibrate(probs):
        p = (np.array(probs).reshape(-1, 1) - mean) / std
        return predict_proba(p, w, b)
    
    return calibrate, w, b, mean, std


# ========== 特征工程 ==========
def prepare_features(df):
    df = df.copy()
    df['discount'] = -df['ha_premium']
    df['industry'] = df['correct_name'].apply(classify_industry)
    
    # 行业编码（目标编码）
    industry_break_rate = df.groupby('industry')['is_broken'].mean()
    global_rate = df['is_broken'].mean()
    industry_smooth = {}
    for ind in industry_break_rate.index:
        n = (df['industry'] == ind).sum()
        industry_smooth[ind] = industry_break_rate[ind] if n >= 3 else global_rate
    df['industry_encoded'] = df['industry'].map(industry_smooth)
    
    # 保荐人编码
    sp_map = {3: 0.6, 2: 0.5, 1.5: 0.45, 1: 0.4, 0.5: 0.35}
    if 'sp' in df.columns:
        df['sp_encoded'] = df['sp'].map(lambda x: sp_map.get(x, 0.4))
    else:
        df['sp_encoded'] = 0.5  # 默认
    
    # 交互项
    df['ret45_x_discount'] = df['ret45'] * df['discount'] / 100
    df['beta_x_ret45'] = df['beta_real'] * df['ret45']
    
    features = [
        'ret45', 'beta_real', 'cs', 'pos_real', 'alpha_real',
        'discount', 'cap_real', 'industry_encoded', 'sp_encoded',
        'ret45_x_discount', 'beta_x_ret45'
    ]
    
    return df, features


# ========== 时间滚动回测 ==========
def walk_forward_validation(df, features, min_train=10):
    df = df.sort_values('list_date').reset_index(drop=True)
    
    predictions = []
    calibrators = []  # 存储每个时间点的校准参数
    
    for i in range(min_train, len(df)):
        train_df = df.iloc[:i]
        test_row = df.iloc[i:i+1]
        
        X_train_raw = train_df[features].fillna(train_df[features].median()).values
        y_train = train_df['is_broken'].values
        X_test_raw = test_row[features].fillna(train_df[features].median()).values
        
        # 标准化
        X_train, mean, std = standardize(X_train_raw)
        X_test = (X_test_raw - mean) / std
        
        # 训练逻辑回归
        w, b = logistic_regression_fit(X_train, y_train, lr=0.1, epochs=5000)
        raw_prob = predict_proba(X_test, w, b)[0]
        
        # Platt校准（用训练集校准）
        train_raw_probs = predict_proba(X_train, w, b)
        calibrate_fn, pw, pb, pm, ps = platt_scaling(train_raw_probs, y_train)
        prob = calibrate_fn(np.array([raw_prob]))[0]
        
        pred = 1 if prob >= 0.5 else 0
        
        predictions.append({
            'name': test_row['correct_name'].values[0],
            'list_date': test_row['list_date'].values[0],
            'actual': test_row['is_broken'].values[0],
            'return_1d': test_row['return_1d'].values[0],
            'prob_broken': prob,
            'pred': pred,
            'n_train': len(train_df),
        })
        
        calibrators.append({'w': w.tolist(), 'b': float(b), 'mean': mean.tolist(), 
                           'std': std.tolist(), 'pw': float(pw), 'pb': float(pb),
                           'pm': float(pm), 'ps': float(ps)})
    
    return pd.DataFrame(predictions), calibrators


# ========== 评估 ==========
def evaluate(results):
    print("\n" + "=" * 70)
    print("📊 模型评估")
    print("=" * 70)
    
    y_true = results['actual'].values
    y_prob = results['prob_broken'].values
    y_pred = results['pred'].values
    
    n = len(results)
    accuracy = (y_true == y_pred).mean()
    
    print(f"\n  测试样本: {n}")
    print(f"  实际破发: {y_true.sum()}")
    print(f"  预测破发: {y_pred.sum()}")
    print(f"  准确率: {accuracy:.3f}")
    
    # ROC-AUC (手工计算)
    def roc_auc(y, prob):
        pos = prob[y == 1]
        neg = prob[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        pairs = 0
        correct = 0
        for p in pos:
            for n in neg:
                pairs += 1
                if p > n:
                    correct += 1
                elif p == n:
                    correct += 0.5
        return correct / pairs if pairs > 0 else 0.5
    
    print(f"  ROC-AUC: {roc_auc(y_true, y_prob):.3f}")
    
    # Brier Score
    brier = np.mean((y_prob - y_true) ** 2)
    print(f"  Brier Score: {brier:.3f}")
    
    # 校准检查
    print("\n  📈 概率校准检查:")
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        subset = results[results['prob_broken'] >= threshold]
        if len(subset) > 0:
            actual_rate = subset['actual'].mean()
            print(f"     预测破发概率>={threshold:.0%}: {len(subset)}只, 实际破发率={actual_rate:.1%}")
    
    # 按概率分档
    results['score'] = 100 * (1 - results['prob_broken'])
    results['grade'] = results['score'].apply(
        lambda s: 'A' if s >= 80 else 'B+' if s >= 60 else 'B' if s >= 45 else 'C' if s >= 35 else 'D'
    )
    
    print("\n  📊 按评分等级:")
    for g in ['A', 'B+', 'B', 'C', 'D']:
        subset = results[results['grade'] == g]
        if len(subset) > 0:
            print(f"     {g}: {len(subset)}只, 平均预测破发={subset['prob_broken'].mean():.1%}, 实际破发={subset['actual'].mean():.1%}")
    
    return results


# ========== 最终模型参数 ==========
def extract_final_model(df, features):
    """用全量数据训练，提取参数供前端使用"""
    X_raw = df[features].fillna(df[features].median()).values
    y = df['is_broken'].values
    
    X, mean, std = standardize(X_raw)
    w, b = logistic_regression_fit(X, y, lr=0.1, epochs=5000)
    
    # 校准
    raw_probs = predict_proba(X, w, b)
    calibrate_fn, pw, pb, pm, ps = platt_scaling(raw_probs, y)
    
    params = {
        'feature_names': features,
        'means': mean.tolist(),
        'stds': std.tolist(),
        'coef': w.tolist(),
        'intercept': float(b),
        'platt_w': float(pw),
        'platt_b': float(pb),
        'platt_mean': float(pm),
        'platt_std': float(ps),
        'n_samples': len(df),
        'n_broken': int(y.sum()),
        'n_features': len(features),
    }
    
    import json
    with open('output/model_v63_params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 模型参数已保存: output/model_v63_params.json")
    
    print("\n  📋 因子影响方向 (逻辑回归系数):")
    for name, coef in zip(features, w):
        direction = "↑ 增加破发概率" if coef > 0 else "↓ 降低破发概率"
        print(f"     {name:<20} {coef:>+.3f}  {direction}")
    
    return params


def main():
    print("=" * 70)
    print("📊 AH股IPO评分系统 V6.3 - 逻辑回归预测破发概率")
    print("=" * 70)
    
    df = pd.read_csv("output/ah_ipo_enriched_20260728.csv")
    df = df.dropna(subset=['return_1d', 'beta_real', 'ret45', 'ha_premium'])
    df['is_broken'] = (df['return_1d'] < 0).astype(int)
    
    print(f"\n📋 总样本: {len(df)} 只")
    print(f"   破发: {df['is_broken'].sum()} ({df['is_broken'].mean()*100:.1f}%)")
    print(f"   未破发: {len(df) - df['is_broken'].sum()}")
    
    df, features = prepare_features(df)
    print(f"\n🔧 特征数: {len(features)}")
    
    # 时间滚动回测
    print("\n" + "=" * 70)
    print("🔄 时间滚动回测 (Walk-forward)")
    print("=" * 70)
    
    results, _ = walk_forward_validation(df, features, min_train=10)
    results = evaluate(results)
    
    # 逐只展示
    print("\n" + "=" * 70)
    print("📋 逐只预测结果")
    print("=" * 70)
    print(f"\n  {'名称':<10} {'日期':<12} {'实际':>6} {'预测破发率':>12} {'评分':>6} {'首日涨幅':>10} {'等级':>4}")
    print("  " + "-" * 70)
    for _, row in results.iterrows():
        actual = "💥" if row['actual'] else "✅"
        print(f"  {row['name']:<8} {str(row['list_date']):<10} {actual:>4} {row['prob_broken']:>10.1%} {row['score']:>6.0f} {row['return_1d']:>+8.1f}% {row['grade']:>4}")
    
    # 最终模型
    print("\n" + "=" * 70)
    print("🎯 训练最终模型 (全量数据)")
    print("=" * 70)
    extract_final_model(df, features)
    
    results.to_csv("output/ah_ipo_v63_walkforward.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 回测结果: output/ah_ipo_v63_walkforward.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
