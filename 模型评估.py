# -*- coding: utf-8 -*-
"""评分卡模型评估：AUC、KS 与分数段违约率。

用法：
    python 模型评估.py --data 信贷数据.csv --target 违约标签 --score 预测分
    python 模型评估.py --demo          # 用随机数据演示输出格式

说明：
    只用 numpy 实现，不依赖 sklearn，方便在没装机器学习库的机器上跑。
"""
import argparse

import numpy as np


def auc_score(y_true, y_score):
    """用秩和公式计算 AUC（等价于 ROC 曲线下面积）。

    对每个正样本统计“有多少负样本的分数比它低”，再除以正负样本对数。
    分数相同的情况用平均秩处理。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("标签必须同时包含正负样本")

    # 平均秩：分数从低到高排，相同分数取平均名次
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sorted_scores = y_score[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            avg_rank = (start + 1 + i) / 2.0
            ranks[order[start:i]] = avg_rank
            start = i

    sum_pos_rank = ranks[y_true == 1].sum()
    return (sum_pos_rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ks_statistic(y_true, y_score):
    """KS = 正样本累计分布与负样本累计分布的最大差值。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.sort(np.unique(y_score))
    pos = y_true == 1
    neg = ~pos

    cum_pos = np.array([(y_score[pos] <= t).mean() if pos.any() else 0 for t in thresholds])
    cum_neg = np.array([(y_score[neg] <= t).mean() if neg.any() else 0 for t in thresholds])
    return float(np.max(np.abs(cum_pos - cum_neg)))


def score_band_table(y_true, y_score, bins=10):
    """按分数等频切段，输出每段的样本数、违约数和违约率。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.quantile(y_score, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9          # 保证最小值落在第一段
    rows = []
    for i in range(bins):
        mask = (y_score > edges[i]) & (y_score <= edges[i + 1])
        n = int(mask.sum())
        bad = int(y_true[mask].sum())
        rows.append({
            "区间": f"({edges[i]:.4f}, {edges[i + 1]:.4f}]",
            "样本数": n,
            "违约数": bad,
            "违约率": round(bad / n, 4) if n else 0.0,
        })
    return rows


def load_data(path, target_col, score_col):
    import pandas as pd

    df = pd.read_csv(path)
    for col in (target_col, score_col):
        if col not in df.columns:
            raise SystemExit(f"数据里没有列 {col}，现有列：{list(df.columns)}")
    return df[target_col].values, df[score_col].values


def main():
    parser = argparse.ArgumentParser(description="评分卡模型评估：AUC / KS / 分数段违约率")
    parser.add_argument("--data", help="CSV 路径")
    parser.add_argument("--target", default="target", help="违约标签列名，默认 target")
    parser.add_argument("--score", default="score", help="预测分数列名，默认 score")
    parser.add_argument("--bins", type=int, default=10, help="分数段数，默认 10")
    parser.add_argument("--demo", action="store_true", help="用随机数据演示")
    args = parser.parse_args()

    if args.demo:
        rng = np.random.default_rng(2026)
        y = rng.integers(0, 2, 2000)
        # 让分数与标签相关，AUC 才有意义
        score = y * 0.6 + rng.normal(0, 0.5, 2000)
    elif args.data:
        y, score = load_data(args.data, args.target, args.score)
    else:
        parser.error("请提供 --data 或使用 --demo")

    print(f"样本数：{len(y)}，违约样本：{int(np.sum(y))}，违约率：{np.mean(y):.2%}")
    print(f"AUC：{auc_score(y, score):.4f}")
    print(f"KS ：{ks_statistic(y, score):.4f}")
    print("\n分数段明细：")
    print(f"{'区间':>28}{'样本数':>8}{'违约数':>8}{'违约率':>10}")
    for row in score_band_table(y, score, args.bins):
        print(f"{row['区间']:>28}{row['样本数']:>8}{row['违约数']:>8}{row['违约率']:>10.2%}")


if __name__ == "__main__":
    main()
