import scipy.stats as stats
import numpy as np

from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests


def calculate_significance(data_dict, method="t-test", paired=False):
    """
    Perform statistical tests between group means or distributions.

    :param data_dict: Dictionary containing data groups with group names as keys
    :type data_dict: dict
    :param method: Statistical method to use (default: "t-test")
    :type method: str
    :param paired: Flag for paired/dependent samples (default: False)
    :type paired: bool
    :return: List of results containing test statistics and p-values
    :rtype: list
    :raises ValueError: For insufficient groups or invalid method names
    """
    groups = list(data_dict.keys())
    data = list(data_dict.values())
    results = []

    if len(groups) < 2:
        raise ValueError("At least two groups required for comparison")

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
                raise ValueError("Wilcoxon test requires paired data")
            if len(data1) != len(data2):
                raise ValueError("Equal sample size required for Wilcoxon test")
            differences = np.array(data1) - np.array(data2)
            try:
                stat, p_value = stats.wilcoxon(differences)
                result["method"] = "Wilcoxon signed-rank test"
            except ValueError as e:
                raise ValueError(f"Wilcoxon test error: {str(e)}")

        elif method == "kruskal-wallis":
            stat, p_value = stats.kruskal(data1, data2)
            result["method"] = "Kruskal-Wallis test"

        else:
            raise ValueError(f"Unsupported statistical method: {method}")

        result["group_comparison"] = f"{group1} vs {group2}"
        result["stat"] = stat
        result["p_value"] = p_value
        results.append(result)

    return results


def multiple_comparison_correction(results, correction_method="bonferroni"):
    """
    Apply multiple comparison correction to raw p-values.

    :param results: Output from calculate_significance function
    :type results: list
    :param correction_method: Correction method (default: "bonferroni")
    :type correction_method: str
    :return: Results with corrected p-values and significance flags
    :rtype: list
    """
    p_values = [result["p_value"] for result in results]

    # Skip correction if ≤2 comparisons
    if len(p_values) <= 1:
        for result in results:
            result["corrected_p_value"] = result["p_value"]
            result["significant_after_correction"] = result["p_value"] < 0.05
        return results

    # Apply chosen correction method
    corrected = multipletests(p_values, method=correction_method)
    corrected_p_vals = corrected[1]  # Adjusted p-values
    significance_flags = corrected[0]  # Significance indicators

    # Update results with corrected values
    for i, result in enumerate(results):
        result["corrected_p_value"] = corrected_p_vals[i]
        result["significant_after_correction"] = significance_flags[i]

    return results


if __name__ == '__main__':
    # Demonstration of statistical functions
    np.random.seed(42)  # Set random seed for reproducibility
    n = 50  # Sample size per group

    # Generate synthetic group data
    group_data = {
        "Group A": np.random.normal(loc=50, scale=10, size=n),
        "Group B": np.random.normal(loc=52, scale=10, size=n),
        "Group C": np.random.normal(loc=52, scale=10, size=n)
    }

    # Compute significance tests
    test_results = calculate_significance(group_data, method="mann-whitney")

    # Apply multiple comparison correction
    corrected_results = multiple_comparison_correction(
        test_results,
        correction_method="fdr_bh"
    )

    print(corrected_results)