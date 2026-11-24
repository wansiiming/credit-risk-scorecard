import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体显示
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass


def get_desktop_path():
    """获取桌面路径（兼容中文系统）"""
    # 方法1：使用系统变量
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    # 方法2：如果方法1不行，尝试使用中文"桌面"
    if not os.path.exists(desktop):
        desktop_cn = os.path.join(os.path.expanduser("~"), "桌面")
        if os.path.exists(desktop_cn):
            desktop = desktop_cn
        else:
            # 方法3：尝试其他可能的路径
            possible_paths = [
                os.path.join(os.environ['USERPROFILE'], 'Desktop'),
                os.path.join(os.environ['USERPROFILE'], '桌面'),
                os.path.join('C:\\', 'Users', os.getlogin(), 'Desktop'),
                os.path.join('C:\\', 'Users', os.getlogin(), '桌面'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    desktop = path
                    break

    return desktop


def calculate_woe_iv(df, feature, target, bins=5, method='quantile'):
    """
    计算特征的WOE和IV值
    """
    temp_df = df[[feature, target]].copy()

    # 处理缺失值
    if temp_df[feature].isnull().any():
        temp_df = temp_df.dropna(subset=[feature])

    # 如果数据太少，直接返回
    if len(temp_df) < 10:
        return None, 0, None

    # 创建分箱
    try:
        if method == 'quantile':
            temp_df['bin'] = pd.qcut(temp_df[feature], q=bins, duplicates='drop', precision=3)
        else:
            temp_df['bin'] = pd.cut(temp_df[feature], bins=bins, precision=3)
    except Exception as e:
        try:
            # 尝试等宽分箱
            temp_df['bin'] = pd.cut(temp_df[feature], bins=bins, precision=3)
        except:
            return None, 0, None

    # 统计每个分箱的好坏样本数
    grouped = temp_df.groupby('bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    # 避免除零错误
    total_bad = max(grouped['bad'].sum(), 0.5)
    total_good = max(grouped['good'].sum(), 0.5)

    grouped['bad_dist'] = grouped['bad'] / total_bad
    grouped['good_dist'] = grouped['good'] / total_good

    # 计算WOE（添加平滑）
    grouped['good_dist_adj'] = grouped['good_dist'].clip(lower=0.0001)
    grouped['bad_dist_adj'] = grouped['bad_dist'].clip(lower=0.0001)
    grouped['woe'] = np.log(grouped['bad_dist_adj'] / grouped['good_dist_adj'])

    # 计算IV成分
    grouped['iv_comp'] = (grouped['bad_dist'] - grouped['good_dist']) * grouped['woe']

    # 总IV值
    iv_value = grouped['iv_comp'].sum()

    return grouped, iv_value, temp_df['bin']


def monotonic_binning(df, feature, target, max_bins=8):
    """
    尝试找到使WOE单调的分箱方案
    """
    best_iv = -1
    best_n_bins = 0
    best_result = None
    best_bins = None

    # 尝试不同的分箱数量
    for n_bins in range(4, max_bins + 1):
        try:
            result, iv, bins_series = calculate_woe_iv(df, feature, target, bins=n_bins)

            if result is None or len(result) < 3:  # 至少需要3个分箱
                continue

            # 检查单调性
            woe_values = result['woe'].values
            if len(woe_values) < 2:
                continue

            # 检查是否单调
            diffs = np.diff(woe_values)
            is_monotonic_increasing = np.all(diffs >= -0.001)  # 允许微小波动
            is_monotonic_decreasing = np.all(diffs <= 0.001)
            is_monotonic = is_monotonic_increasing or is_monotonic_decreasing

            # 如果单调且IV值更好，则更新最佳分箱
            if is_monotonic and iv > best_iv:
                best_iv = iv
                best_n_bins = n_bins
                best_result = result
                best_bins = bins_series

        except Exception as e:
            continue

    # 如果没有找到单调分箱，使用最佳IV值的分箱
    if best_result is None:
        best_iv = -1
        for n_bins in range(3, max_bins + 1):
            try:
                result, iv, bins_series = calculate_woe_iv(df, feature, target, bins=n_bins)
                if result is not None and iv > best_iv:
                    best_iv = iv
                    best_n_bins = n_bins
                    best_result = result
                    best_bins = bins_series
            except:
                continue

    return best_result, best_iv, best_n_bins, best_bins


def analyze_feature_binning(df, feature, target):
    """
    分析单个特征的分箱
    """
    print(f"分析特征: {feature}")

    # 获取分箱结果
    result, iv, n_bins, bins_series = monotonic_binning(df, feature, target)

    if result is None:
        print(f"  ✗ 特征 {feature} 无法分箱")
        return None

    # 创建详细的结果字典
    feature_result = {
        '特征名称': feature,
        'IV值': iv,
        '分箱数': n_bins,
        '分箱详情': result,
        '分箱序列': bins_series
    }

    print(f"  ✓ IV值: {iv:.6f}, 分箱数: {n_bins}")

    return feature_result


# 主程序
def main():
    print("=" * 80)
    print("开始分箱分析")
    print("=" * 80)

    # 加载数据
    file_path = r"C:\Users\王子鸣\OneDrive\桌面\数据集分割结果\测试集.csv"
    print(f"加载数据文件: {file_path}")

    # 尝试多种编码方式加载数据
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"使用编码 {encoding} 成功加载数据")
            break
        except Exception as e:
            continue

    if df is None:
        print("错误: 无法加载数据文件，请检查文件路径和编码")
        return None, None

    print(f"数据集形状: {df.shape}")
    print(f"数据列: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")

    # 识别目标变量
    target_column = None
    target_candidates = ['y', 'target', 'label', 'class', 'is_default', 'default', 'bad', 'good', 'result', 'flag']

    for col in df.columns:
        col_lower = col.lower()
        if any(candidate in col_lower for candidate in target_candidates):
            target_column = col
            print(f"找到目标变量: {col}")
            break

    if target_column is None:
        # 尝试找到二元分类的目标变量
        unique_counts = {}
        for col in df.columns:
            unique_vals = df[col].nunique()
            unique_counts[col] = unique_vals

        # 找到唯一值最少的列（假设是目标变量）
        sorted_cols = sorted(unique_counts.items(), key=lambda x: x[1])
        for col, n_unique in sorted_cols:
            if n_unique <= 10:  # 假设目标变量唯一值不超过10
                target_column = col
                print(f"自动选择目标变量: '{target_column}' (唯一值数: {n_unique})")
                break

        if target_column is None:
            target_column = sorted_cols[0][0]
            print(f"警告: 未找到合适的目标变量，使用第一列 '{target_column}' 作为目标变量")

    print(f"\n目标变量: {target_column}")
    print(f"目标变量分布:")
    print(df[target_column].value_counts().sort_index())

    # 分离特征
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 选择数值型特征
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    print(f"\n找到 {len(numeric_features)} 个数值型特征")

    # 分析每个特征
    print("\n" + "=" * 80)
    print("开始分箱分析各个特征")
    print("=" * 80)

    all_results = []
    iv_values = []

    for feature in numeric_features[:50]:  # 限制分析前50个特征，防止太多
        result = analyze_feature_binning(df, feature, target_column)
        if result is not None and result['IV值'] >= 0:
            all_results.append(result)
            iv_values.append((feature, result['IV值']))

    # 按IV值排序
    iv_values.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("IV值排名:")
    print("=" * 80)

    iv_ranking = []
    for i, (feature, iv) in enumerate(iv_values[:20], 1):
        print(f"{i:2d}. {feature:30s} IV = {iv:.6f}")
        iv_ranking.append({'排名': i, '特征': feature, 'IV值': iv})

    # 找出IV值大于0.1的特征
    significant_features = [(feature, iv) for feature, iv in iv_values if iv > 0.1]

    print("\n" + "=" * 80)
    print(f"IV值大于0.1的特征 (共{len(significant_features)}个):")
    print("=" * 80)

    if significant_features:
        for i, (feature, iv) in enumerate(significant_features[:10], 1):
            print(f"{i:2d}. {feature:30s} IV = {iv:.6f}")
    else:
        print("未找到IV值大于0.1的特征")

    # 保存结果
    print("\n" + "=" * 80)
    print("保存分析结果到桌面...")

    # 获取桌面路径
    desktop_path = get_desktop_path()
    print(f"桌面路径: {desktop_path}")

    # 创建保存结果的文件夹（使用中文名称）
    results_folder = os.path.join(desktop_path, "分箱分析结果")
    os.makedirs(results_folder, exist_ok=True)
    print(f"结果保存到: {results_folder}")

    # 1. 保存IV值排名
    if iv_ranking:
        iv_df = pd.DataFrame(iv_ranking)
        iv_file = os.path.join(results_folder, "IV值排名.xlsx")
        iv_df.to_excel(iv_file, index=False)
        print(f"✓ 1. IV值排名已保存: {os.path.basename(iv_file)}")

    # 2. 保存显著特征的分箱详情
    if significant_features:
        sig_file = os.path.join(results_folder, "显著特征分箱详情.xlsx")
        with pd.ExcelWriter(sig_file, engine='openpyxl') as writer:
            for feature, iv in significant_features[:15]:  # 保存前15个显著特征
                # 找到该特征的详细结果
                for result in all_results:
                    if result['特征名称'] == feature:
                        result_df = result['分箱详情'].reset_index()
                        result_df.insert(0, '特征', feature)
                        result_df.insert(1, 'IV值', iv)

                        # 重命名列
                        result_df = result_df.rename(columns={
                            'index': '分箱区间',
                            'total': '样本数',
                            'bad': '坏样本数',
                            'good': '好样本数',
                            'bad_dist': '坏样本占比',
                            'good_dist': '好样本占比',
                            'woe': 'WOE值',
                            'iv_comp': 'IV贡献'
                        })

                        # 格式化和舍入
                        result_df['坏样本占比'] = result_df['坏样本占比'].round(4)
                        result_df['好样本占比'] = result_df['好样本占比'].round(4)
                        result_df['WOE值'] = result_df['WOE值'].round(4)
                        result_df['IV贡献'] = result_df['IV贡献'].round(6)

                        # 保存到Excel
                        sheet_name = feature[:20]  # Excel工作表名最多31个字符
                        sheet_name = sheet_name.replace(':', '_').replace('\\', '_').replace('/', '_').replace('?',
                                                                                                               '_').replace(
                            '*', '_').replace('[', '_').replace(']', '_')
                        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        break

        print(f"✓ 2. 显著特征分箱详情已保存: {os.path.basename(sig_file)}")

    # 3. 保存所有特征的分箱摘要
    if all_results:
        summary_data = []
        for result in all_results:
            feature_name = result['特征名称']
            iv = result['IV值']
            n_bins = result['分箱数']
            woe_values = []
            if result['分箱详情'] is not None:
                woe_values = result['分箱详情']['woe'].round(4).tolist()

            # 检查单调性
            is_monotonic = "否"
            if len(woe_values) > 1:
                increasing = all(woe_values[i] <= woe_values[i + 1] for i in range(len(woe_values) - 1))
                decreasing = all(woe_values[i] >= woe_values[i + 1] for i in range(len(woe_values) - 1))
                is_monotonic = "是" if increasing or decreasing else "否"

            # 预测能力评级
            if iv < 0.02:
                predictive_power = "无预测能力"
            elif iv < 0.1:
                predictive_power = "弱预测能力"
            elif iv < 0.3:
                predictive_power = "中等预测能力"
            elif iv < 0.5:
                predictive_power = "强预测能力"
            else:
                predictive_power = "极强预测能力"

            summary_data.append({
                '特征名称': feature_name,
                'IV值': round(iv, 6),
                '预测能力': predictive_power,
                '分箱数': n_bins,
                '是否单调': is_monotonic,
                'WOE值': str(woe_values)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(results_folder, "所有特征分箱摘要.xlsx")
        summary_df.to_excel(summary_file, index=False)
        print(f"✓ 3. 所有特征分箱摘要已保存: {os.path.basename(summary_file)}")

    # 4. 保存分箱数据结果
    if significant_features and all_results:
        sig_features = [feature for feature, iv in significant_features[:10]]

        # 创建一个新的DataFrame包含原始数据和分箱结果
        result_df = df[[target_column] + sig_features].copy()

        for feature in sig_features:
            for result in all_results:
                if result['特征名称'] == feature:
                    result_df[f'{feature}_分箱'] = result['分箱序列']
                    break

        data_file = os.path.join(results_folder, "分箱数据结果.xlsx")
        result_df.to_excel(data_file, index=False)
        print(f"✓ 4. 分箱数据结果已保存: {os.path.basename(data_file)}")

    # 5. 生成详细的分析报告
    report_file = os.path.join(results_folder, "分箱分析报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                   分箱分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件: {os.path.basename(file_path)}\n")
        f.write(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n")
        f.write(f"目标变量: {target_column}\n\n")

        # 目标变量信息
        target_stats = df[target_column].value_counts().sort_index()
        f.write("目标变量分布:\n")
        for value, count in target_stats.items():
            percentage = count / len(df) * 100
            f.write(f"  值 {value}: {count} 个 ({percentage:.1f}%)\n")
        f.write("\n")

        f.write(f"数值型特征数量: {len(numeric_features)}\n")
        f.write(f"成功分析的特征数量: {len(all_results)}\n")
        f.write(f"IV值大于0.1的特征数量: {len(significant_features)}\n\n")

        # 特征分析结果
        f.write("=" * 80 + "\n")
        f.write("重要特征分析结果:\n")
        f.write("=" * 80 + "\n\n")

        if significant_features:
            f.write("前10个IV值最高的特征:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'排名':<4} {'特征名称':<30} {'IV值':<10} {'预测能力':<15} {'分箱数':<8} {'是否单调'}\n")
            f.write("-" * 100 + "\n")

            for i, (feature, iv) in enumerate(significant_features[:10], 1):
                # 获取其他信息
                n_bins = 0
                is_monotonic = "否"
                for result in all_results:
                    if result['特征名称'] == feature:
                        n_bins = result['分箱数']
                        if result['分箱详情'] is not None:
                            woe_values = result['分箱详情']['woe'].round(4).tolist()
                            if len(woe_values) > 1:
                                increasing = all(woe_values[i] <= woe_values[i + 1] for i in range(len(woe_values) - 1))
                                decreasing = all(woe_values[i] >= woe_values[i + 1] for i in range(len(woe_values) - 1))
                                is_monotonic = "是" if increasing or decreasing else "否"
                        break

                # 预测能力评级
                if iv < 0.02:
                    predictive_power = "无预测能力"
                elif iv < 0.1:
                    predictive_power = "弱预测能力"
                elif iv < 0.3:
                    predictive_power = "中等预测能力"
                elif iv < 0.5:
                    predictive_power = "强预测能力"
                else:
                    predictive_power = "极强预测能力"

                f.write(f"{i:<4} {feature:<30} {iv:<10.6f} {predictive_power:<15} {n_bins:<8} {is_monotonic}\n")
        else:
            f.write("未找到IV值大于0.1的特征。\n")
            f.write("建议检查:\n")
            f.write("1. 目标变量是否正确\n")
            f.write("2. 数据质量是否有问题\n")
            f.write("3. 是否需要调整分箱策略\n")

        # IV值解读
        f.write("\n" + "=" * 80 + "\n")
        f.write("IV值解读指南:\n")
        f.write("=" * 80 + "\n\n")

        f.write("IV值范围       预测能力评估       推荐使用\n")
        f.write("-" * 80 + "\n")
        f.write("小于0.02       无预测能力         不建议使用\n")
        f.write("0.02 - 0.1     弱预测能力         谨慎使用\n")
        f.write("0.1 - 0.3      中等预测能力       推荐使用\n")
        f.write("0.3 - 0.5      强预测能力         强烈推荐\n")
        f.write("大于0.5        极强预测能力       需要检查是否存在数据泄露\n\n")

        # 分箱建议
        f.write("=" * 80 + "\n")
        f.write("分箱质量建议:\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 优秀分箱特征应满足:\n")
        f.write("   - IV值 > 0.1\n")
        f.write("   - WOE值呈现单调趋势\n")
        f.write("   - 每个分箱有足够的样本量\n")
        f.write("   - 坏样本率在不同分箱间有明显差异\n\n")

        f.write("2. 对于IV值较低的特征:\n")
        f.write("   - 考虑与其他特征组合\n")
        f.write("   - 尝试不同的分箱方法\n")
        f.write("   - 考虑特征工程转换\n\n")

        f.write("3. 后续建模建议:\n")
        f.write("   - 优先使用IV值大于0.1的特征\n")
        f.write("   - 检查特征间的相关性\n")
        f.write("   - 注意WOE的单调性对于逻辑回归模型非常重要\n")

    print(f"✓ 5. 分箱分析报告已保存: {os.path.basename(report_file)}")

    # 6. 保存Python代码（可选）
    code_file = os.path.join(results_folder, "分箱分析代码.py")
    with open(__file__, 'r', encoding='utf-8') as src:
        with open(code_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    print(f"✓ 6. 分析代码已保存: {os.path.basename(code_file)}")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    # 显示最重要的特征详情
    if significant_features:
        print("\n前5个最重要的特征详情:")
        print("-" * 100)
        for i, (feature, iv) in enumerate(significant_features[:5], 1):
            print(f"\n{i}. {feature}")
            print(f"   IV值: {iv:.6f}")

            # 获取详细分箱信息
            for result in all_results:
                if result['特征名称'] == feature:
                    n_bins = result['分箱数']
                    print(f"   分箱数: {n_bins}")

                    if result['分箱详情'] is not None:
                        print(f"   分箱详情:")
                        detail_df = result['分箱详情'].reset_index()
                        print(detail_df[['bin', 'total', 'bad', 'woe']].to_string(index=False))

                    break
    else:
        print("\n未找到IV值大于0.1的特征")
        print("可能是以下原因:")
        print("1. 目标变量选择不正确")
        print("2. 数据中缺乏强预测特征")
        print("3. 需要调整分箱参数")

    print(f"\n所有结果已保存到: {results_folder}")
    print("包含以下文件:")
    print("  1. IV值排名.xlsx - 所有特征的IV值排序")
    print("  2. 显著特征分箱详情.xlsx - 重要特征的详细分箱信息")
    print("  3. 所有特征分箱摘要.xlsx - 所有特征的汇总信息")
    print("  4. 分箱数据结果.xlsx - 包含分箱结果的数据文件")
    print("  5. 分箱分析报告.txt - 完整的分析报告")
    print("  6. 分箱分析代码.py - 本分析使用的代码")

    return all_results, significant_features


# 运行主程序
if __name__ == "__main__":
    print("分箱分析工具 v1.0")
    print("作者: 数据分析助手")
    print("功能: 对数据集进行分箱分析，计算WOE和IV值")
    print("=" * 80)

    try:
        all_results, significant_features = main()

        # 等待用户查看结果
        input("\n按Enter键退出程序...")

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        print("错误详情:")
        import traceback

        traceback.print_exc()

        print("\n常见问题解决方法:")
        print("1. 检查文件路径是否正确")
        print("2. 确保数据文件存在且可访问")
        print("3. 检查数据文件编码")
        print("4. 确保有足够的磁盘空间")

        input("\n按Enter键退出程序...")