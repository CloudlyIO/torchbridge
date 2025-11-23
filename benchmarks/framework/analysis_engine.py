"""
Statistical Analysis Engine for Benchmark Results

Advanced statistical analysis including significance testing,
effect size calculation, and performance regression analysis.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy import stats

@dataclass
class StatisticalAnalysis:
    """Statistical analysis results"""
    mean_difference: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistically_significant: bool
    practical_significance: bool

class AnalysisEngine:
    """Advanced statistical analysis for benchmark results"""

    def __init__(self, significance_threshold: float = 0.05, effect_size_threshold: float = 0.5):
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold

    def compare_implementations(self, baseline_metrics: List[float],
                               optimized_metrics: List[float]) -> StatisticalAnalysis:
        """Compare two implementations with statistical rigor"""

        # Convert to numpy arrays
        baseline = np.array(baseline_metrics)
        optimized = np.array(optimized_metrics)

        # Calculate basic statistics
        mean_diff = np.mean(baseline) - np.mean(optimized)

        # Welch's t-test (assumes unequal variances)
        t_stat, p_value = stats.ttest_ind(baseline, optimized, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) +
                             (len(optimized) - 1) * np.var(optimized, ddof=1)) /
                            (len(baseline) + len(optimized) - 2))

        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval for the difference
        se_diff = pooled_std * np.sqrt(1/len(baseline) + 1/len(optimized))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff

        return StatisticalAnalysis(
            mean_difference=mean_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            statistically_significant=p_value < self.significance_threshold,
            practical_significance=abs(effect_size) > self.effect_size_threshold
        )