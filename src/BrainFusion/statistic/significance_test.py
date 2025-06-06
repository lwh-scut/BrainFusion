import scipy.stats as stats
import numpy as np

from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests


def calculate_significance(data_dict, method="t-test", paired=False):
    """
    计算字典中数据之间的显著性。

    Args:
        data_dict (dict): 包含多组数据的字典，键为组名，值为数据列表。
        method (str): 选择统计方法，支持 "t-test", "paired-t-test", "anova",
                      "mann-whitney", "wilcoxon", "kruskal-wallis"。
        paired (bool): 是否进行配对检验（仅对 t-test 和非参数 Wilcoxon 生效）。

    Returns:
        list: 包含每对组的统计量和 p 值的显著性测试结果。
    """
    groups = list(data_dict.keys())
    data = list(data_dict.values())
    results = []

    if len(groups) < 2:
        raise ValueError("至少需要两组数据进行比较。")

    for group1, group2 in combinations(groups, 2):
        result = {}
        data1 = data_dict[group1]
        data2 = data_dict[group2]

        if method == "t-test":
            if paired:
                stat, p_value = stats.ttest_rel(data1, data2)
                result["method"] = "paired t-test"
            else:
                stat, p_value = stats.ttest_ind(data1, data2)
                result["method"] = "independent t-test"

        elif method == "anova":
            stat, p_value = stats.f_oneway(data1, data2)
            result["method"] = "ANOVA"

        elif method == "mann-whitney":
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            result["method"] = "Mann-Whitney U test"

        elif method == "wilcoxon":
            if not paired:
                raise ValueError("Wilcoxon 检验仅支持配对数据的比较。")
            if len(data1) != len(data2):
                raise ValueError("Wilcoxon 检验要求两组数据的长度相等。")
            differences = np.array(data1) - np.array(data2)
            try:
                stat, p_value = stats.wilcoxon(differences)
                result["method"] = "Wilcoxon signed-rank test"
            except ValueError as e:
                raise ValueError(f"Wilcoxon 检验中出现错误: {str(e)}")

        elif method == "kruskal-wallis":
            stat, p_value = stats.kruskal(data1, data2)
            result["method"] = "Kruskal-Wallis test"

        else:
            raise ValueError(f"未知的统计方法: {method}")

        result["group_comparison"] = f"{group1} vs {group2}"
        result["stat"] = stat
        result["p_value"] = p_value
        results.append(result)

    return results


def multiple_comparison_correction(results, correction_method="bonferroni"):
    """
        对显著性检验的 p 值进行多重比较校正。

        Args:
            results (list): 包含每对组的统计量和 p 值的显著性测试结果。
            correction_method (str): 校正方法，支持 "bonferroni" 和 "fdr_bh"。

        Returns:
            list: 更新后的显著性测试结果，包含校正后的 p 值和显著性标记。
        """
    p_values = [result["p_value"] for result in results]

    if len(p_values) <= 2:
        for result in results:
            result["corrected_p_value"] = result["p_value"]
            result["significant_after_correction"] = result["p_value"] < 0.05
        return results

    corrected_results = multipletests(p_values, method=correction_method)
    corrected_p_values = corrected_results[1]  # 校正后的 p 值
    significant_flags = corrected_results[0]  # 显著性标记

    for i, result in enumerate(results):
        result["corrected_p_value"] = corrected_p_values[i]
        result["significant_after_correction"] = significant_flags[i]

    return results


if __name__ == '__main__':
    np.random.seed(42)  # 固定随机种子以便重现
    n = 50  # 样本大小
    data1 = np.random.normal(loc=50, scale=10, size=n)  # 第一组样本
    data2 = np.random.normal(loc=52, scale=10, size=n)  # 第二组样本
    data3 = np.random.normal(loc=52, scale=10, size=n)  # 第二组样本

    data = {
        "Group A": data1,
        "Group B": data2,
        "Group C": data3
    }
    results = calculate_significance(data, method="wilcoxon", paired=True)
    corrected_results = multiple_comparison_correction(results, correction_method="fdr_bh")

    print(corrected_results)

