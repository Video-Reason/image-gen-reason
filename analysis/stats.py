"""
Statistical Analysis for Comparing GPT-4o and Human Evaluations
================================================================

This module performs comprehensive statistical analysis to compare GPT-4o
evaluations with human evaluations, testing for statistical significance
and finding the convergence threshold where both methods become equivalent.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EvaluationComparator:
    """Compare GPT-4o and human evaluation methods statistically."""
    
    def __init__(self, data_dir: str = "data/evaluations"):
        """Initialize the comparator with data directory."""
        self.data_dir = Path(data_dir)
        self.gpt4o_dir = self.data_dir / "gpt4o-eval"
        self.human_dir = self.data_dir / "human-eval"
        
        # Store loaded evaluations
        self.gpt4o_scores = defaultdict(list)
        self.human_scores = defaultdict(list)
        self.paired_scores = []
        
    def load_evaluations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all evaluation data from both GPT-4o and human evaluators."""
        gpt4o_data = []
        human_data = []
        
        # Load GPT-4o evaluations
        for json_file in self.gpt4o_dir.rglob("*.json"):
            if "GPT4OEvaluator_all_models" in str(json_file):
                continue  # Skip aggregated file
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if 'result' in data and 'solution_correctness_score' in data['result']:
                    gpt4o_data.append({
                        'model': data['metadata'].get('model_name', 'unknown'),
                        'task_type': data['metadata'].get('task_type', 'unknown'),
                        'task_id': data['metadata'].get('task_id', 'unknown'),
                        'score': data['result']['solution_correctness_score'],
                        'evaluator': 'gpt4o'
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Load human evaluations
        for json_file in self.human_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if 'result' in data and 'solution_correctness_score' in data['result']:
                    human_data.append({
                        'model': data['metadata'].get('model_name', 'unknown'),
                        'task_type': data['metadata'].get('task_type', 'unknown'),
                        'task_id': data['metadata'].get('task_id', 'unknown'),
                        'score': data['result']['solution_correctness_score'],
                        'evaluator': 'human'
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        gpt4o_df = pd.DataFrame(gpt4o_data)
        human_df = pd.DataFrame(human_data)
        
        print(f"Loaded {len(gpt4o_df)} GPT-4o evaluations")
        print(f"Loaded {len(human_df)} human evaluations")
        
        return gpt4o_df, human_df
    
    def prepare_paired_data(self, gpt4o_df: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare paired data for statistical comparison."""
        # Merge on model, task_type, and task_id to get paired scores
        paired = pd.merge(
            gpt4o_df,
            human_df,
            on=['model', 'task_type', 'task_id'],
            suffixes=('_gpt4o', '_human')
        )
        
        paired = paired[['model', 'task_type', 'task_id', 'score_gpt4o', 'score_human']]
        print(f"\nFound {len(paired)} paired evaluations")
        
        return paired
    
    def basic_statistics(self, paired_df: pd.DataFrame) -> Dict:
        """Calculate basic descriptive statistics."""
        stats_dict = {
            'gpt4o': {
                'mean': paired_df['score_gpt4o'].mean(),
                'std': paired_df['score_gpt4o'].std(),
                'median': paired_df['score_gpt4o'].median(),
                'min': paired_df['score_gpt4o'].min(),
                'max': paired_df['score_gpt4o'].max()
            },
            'human': {
                'mean': paired_df['score_human'].mean(),
                'std': paired_df['score_human'].std(),
                'median': paired_df['score_human'].median(),
                'min': paired_df['score_human'].min(),
                'max': paired_df['score_human'].max()
            }
        }
        
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        for evaluator in ['gpt4o', 'human']:
            print(f"\n{evaluator.upper()}:")
            for stat, value in stats_dict[evaluator].items():
                print(f"  {stat:10s}: {value:.3f}")
        
        return stats_dict
    
    def paired_t_test(self, paired_df: pd.DataFrame) -> Tuple[float, float]:
        """Perform paired t-test to check if means are different."""
        t_stat, p_value = stats.ttest_rel(
            paired_df['score_gpt4o'], 
            paired_df['score_human']
        )
        
        print("\n" + "="*60)
        print("PAIRED T-TEST")
        print("="*60)
        print(f"H0: There is no difference between GPT-4o and human evaluations")
        print(f"H1: There is a significant difference")
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: REJECT H0 - Methods are significantly different (p < 0.05)")
        else:
            print("Result: FAIL TO REJECT H0 - Methods are not significantly different (p >= 0.05)")
        
        return t_stat, p_value
    
    def wilcoxon_test(self, paired_df: pd.DataFrame) -> Tuple[float, float]:
        """Perform Wilcoxon signed-rank test (non-parametric alternative)."""
        statistic, p_value = stats.wilcoxon(
            paired_df['score_gpt4o'], 
            paired_df['score_human']
        )
        
        print("\n" + "="*60)
        print("WILCOXON SIGNED-RANK TEST (Non-parametric)")
        print("="*60)
        print(f"Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Methods are significantly different (p < 0.05)")
        else:
            print("Result: Methods are not significantly different (p >= 0.05)")
        
        return statistic, p_value
    
    def correlation_analysis(self, paired_df: pd.DataFrame) -> Dict:
        """Calculate various correlation coefficients."""
        pearson_r, pearson_p = pearsonr(paired_df['score_gpt4o'], paired_df['score_human'])
        spearman_r, spearman_p = spearmanr(paired_df['score_gpt4o'], paired_df['score_human'])
        kendall_tau, kendall_p = kendalltau(paired_df['score_gpt4o'], paired_df['score_human'])
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        print(f"Pearson correlation:   r = {pearson_r:.4f}, p = {pearson_p:.4f}")
        print(f"Spearman correlation:  ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
        print(f"Kendall's tau:         τ = {kendall_tau:.4f}, p = {kendall_p:.4f}")
        
        return {
            'pearson': (pearson_r, pearson_p),
            'spearman': (spearman_r, spearman_p),
            'kendall': (kendall_tau, kendall_p)
        }
    
    def inter_rater_reliability(self, paired_df: pd.DataFrame) -> float:
        """Calculate Cohen's kappa for inter-rater reliability."""
        kappa = cohen_kappa_score(
            paired_df['score_gpt4o'].astype(int), 
            paired_df['score_human'].astype(int)
        )
        
        print("\n" + "="*60)
        print("INTER-RATER RELIABILITY")
        print("="*60)
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Interpretation
        if kappa < 0:
            interpretation = "Poor agreement"
        elif kappa < 0.2:
            interpretation = "Slight agreement"
        elif kappa < 0.4:
            interpretation = "Fair agreement"
        elif kappa < 0.6:
            interpretation = "Moderate agreement"
        elif kappa < 0.8:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        
        print(f"Interpretation: {interpretation}")
        
        return kappa
    
    def convergence_analysis(self, paired_df: pd.DataFrame, min_n: int = 10, step: int = 5) -> Dict:
        """
        Analyze how p-values change with sample size to find convergence threshold.
        """
        n_samples = len(paired_df)
        sample_sizes = list(range(min_n, n_samples + 1, step))
        
        p_values = []
        correlations = []
        kappas = []
        mean_differences = []
        
        print("\n" + "="*60)
        print("CONVERGENCE ANALYSIS")
        print("="*60)
        print("Testing statistical equivalence at different sample sizes...")
        
        for n in sample_sizes:
            # Random sample of size n
            sample = paired_df.sample(n=n, replace=False)
            
            # T-test p-value
            _, p_val = stats.ttest_rel(sample['score_gpt4o'], sample['score_human'])
            p_values.append(p_val)
            
            # Correlation
            r, _ = pearsonr(sample['score_gpt4o'], sample['score_human'])
            correlations.append(r)
            
            # Cohen's kappa
            kappa = cohen_kappa_score(
                sample['score_gpt4o'].astype(int), 
                sample['score_human'].astype(int)
            )
            kappas.append(kappa)
            
            # Mean difference
            mean_diff = abs(sample['score_gpt4o'].mean() - sample['score_human'].mean())
            mean_differences.append(mean_diff)
        
        # Find convergence threshold (where p-value consistently > 0.05)
        convergence_threshold = None
        for i, p_val in enumerate(p_values):
            if p_val > 0.05:
                # Check if it stays above 0.05 for the next few samples
                if i + 3 < len(p_values) and all(p > 0.05 for p in p_values[i:i+3]):
                    convergence_threshold = sample_sizes[i]
                    break
        
        print(f"\nConvergence threshold: n = {convergence_threshold}")
        if convergence_threshold:
            print(f"Methods become statistically equivalent at n >= {convergence_threshold}")
        else:
            print("No clear convergence threshold found in the current data")
        
        return {
            'sample_sizes': sample_sizes,
            'p_values': p_values,
            'correlations': correlations,
            'kappas': kappas,
            'mean_differences': mean_differences,
            'convergence_threshold': convergence_threshold
        }
    
    def bootstrap_confidence_intervals(self, paired_df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict:
        """
        Calculate bootstrap confidence intervals for the difference in means.
        """
        differences = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = paired_df.sample(n=len(paired_df), replace=True)
            diff = sample['score_gpt4o'].mean() - sample['score_human'].mean()
            differences.append(diff)
        
        differences = np.array(differences)
        ci_lower = np.percentile(differences, 2.5)
        ci_upper = np.percentile(differences, 97.5)
        
        print("\n" + "="*60)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*60)
        print(f"Mean difference: {np.mean(differences):.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        if ci_lower <= 0 <= ci_upper:
            print("Result: CI contains zero - no significant difference")
        else:
            print("Result: CI does not contain zero - significant difference exists")
        
        return {
            'mean_diff': np.mean(differences),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'differences': differences
        }
    
    def analyze_by_task_type(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze agreement by task type."""
        print("\n" + "="*60)
        print("ANALYSIS BY TASK TYPE")
        print("="*60)
        
        results = []
        
        for task_type in paired_df['task_type'].unique():
            task_data = paired_df[paired_df['task_type'] == task_type]
            
            if len(task_data) > 1:
                t_stat, p_value = stats.ttest_rel(
                    task_data['score_gpt4o'], 
                    task_data['score_human']
                )
                
                correlation, _ = pearsonr(
                    task_data['score_gpt4o'], 
                    task_data['score_human']
                )
                
                mean_diff = abs(task_data['score_gpt4o'].mean() - task_data['score_human'].mean())
                
                results.append({
                    'task_type': task_type,
                    'n_samples': len(task_data),
                    'mean_difference': mean_diff,
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        results_df = pd.DataFrame(results)
        print("\n", results_df.to_string())
        
        return results_df
    
    def plot_comparisons(self, paired_df: pd.DataFrame, convergence_data: Dict, save_path: str = "analysis/statistics"):
        """Create comprehensive visualization of the comparison."""
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GPT-4o vs Human Evaluation Statistical Comparison', fontsize=16)
        
        # 1. Scatter plot with regression line
        ax = axes[0, 0]
        ax.scatter(paired_df['score_gpt4o'], paired_df['score_human'], alpha=0.5)
        
        # Add regression line
        z = np.polyfit(paired_df['score_gpt4o'], paired_df['score_human'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(paired_df['score_gpt4o'].min(), paired_df['score_gpt4o'].max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Add diagonal line (perfect agreement)
        ax.plot([0, 5], [0, 5], 'k--', alpha=0.3, label='Perfect agreement')
        
        ax.set_xlabel('GPT-4o Score')
        ax.set_ylabel('Human Score')
        ax.set_title('Score Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution comparison
        ax = axes[0, 1]
        ax.hist(paired_df['score_gpt4o'], bins=10, alpha=0.5, label='GPT-4o', color='blue')
        ax.hist(paired_df['score_human'], bins=10, alpha=0.5, label='Human', color='red')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Bland-Altman plot
        ax = axes[0, 2]
        mean_scores = (paired_df['score_gpt4o'] + paired_df['score_human']) / 2
        diff_scores = paired_df['score_gpt4o'] - paired_df['score_human']
        
        ax.scatter(mean_scores, diff_scores, alpha=0.5)
        ax.axhline(y=diff_scores.mean(), color='r', linestyle='-', label=f'Mean: {diff_scores.mean():.2f}')
        ax.axhline(y=diff_scores.mean() + 1.96*diff_scores.std(), color='r', linestyle='--', alpha=0.5, label='±1.96 SD')
        ax.axhline(y=diff_scores.mean() - 1.96*diff_scores.std(), color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Mean Score')
        ax.set_ylabel('Difference (GPT-4o - Human)')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. P-value convergence
        ax = axes[1, 0]
        ax.plot(convergence_data['sample_sizes'], convergence_data['p_values'], 'b-', marker='o', markersize=4)
        ax.axhline(y=0.05, color='r', linestyle='--', label='p=0.05 threshold')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('P-value')
        ax.set_title('P-value Convergence with Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Correlation convergence
        ax = axes[1, 1]
        ax.plot(convergence_data['sample_sizes'], convergence_data['correlations'], 'g-', marker='o', markersize=4)
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Correlation Convergence with Sample Size')
        ax.grid(True, alpha=0.3)
        
        # 6. Mean difference convergence
        ax = axes[1, 2]
        ax.plot(convergence_data['sample_sizes'], convergence_data['mean_differences'], 'r-', marker='o', markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='No difference')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Mean Absolute Difference')
        ax.set_title('Mean Difference Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/gpt4o_vs_human_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlot saved to {save_path}/gpt4o_vs_human_comparison.png")
    
    def run_full_analysis(self):
        """Run the complete statistical analysis."""
        print("\n" + "="*80)
        print(" GPT-4o vs HUMAN EVALUATION STATISTICAL COMPARISON ".center(80))
        print("="*80)
        
        # Load data
        gpt4o_df, human_df = self.load_evaluations()
        
        # Prepare paired data
        paired_df = self.prepare_paired_data(gpt4o_df, human_df)
        
        if len(paired_df) == 0:
            print("No paired data found! Cannot perform comparison.")
            return
        
        # Basic statistics
        basic_stats = self.basic_statistics(paired_df)
        
        # Statistical tests
        t_stat, t_p = self.paired_t_test(paired_df)
        w_stat, w_p = self.wilcoxon_test(paired_df)
        
        # Correlation analysis
        correlations = self.correlation_analysis(paired_df)
        
        # Inter-rater reliability
        kappa = self.inter_rater_reliability(paired_df)
        
        # Bootstrap confidence intervals
        bootstrap_results = self.bootstrap_confidence_intervals(paired_df)
        
        # Convergence analysis
        convergence_data = self.convergence_analysis(paired_df)
        
        # Analysis by task type
        task_analysis = self.analyze_by_task_type(paired_df)
        
        # Create visualizations
        self.plot_comparisons(paired_df, convergence_data)
        
        # Summary
        print("\n" + "="*80)
        print(" SUMMARY ".center(80))
        print("="*80)
        
        print(f"\n1. Sample Size: {len(paired_df)} paired evaluations")
        
        print(f"\n2. Statistical Significance:")
        print(f"   - Paired t-test p-value: {t_p:.4f}")
        print(f"   - Wilcoxon test p-value: {w_p:.4f}")
        
        conclusion = "NOT significantly different" if t_p >= 0.05 else "significantly different"
        print(f"   - Conclusion: Methods are {conclusion} at α=0.05")
        
        print(f"\n3. Agreement Metrics:")
        print(f"   - Pearson correlation: {correlations['pearson'][0]:.4f}")
        print(f"   - Cohen's kappa: {kappa:.4f}")
        
        print(f"\n4. Convergence:")
        if convergence_data['convergence_threshold']:
            print(f"   - Methods converge at n >= {convergence_data['convergence_threshold']}")
        else:
            print(f"   - No clear convergence threshold found")
        
        print(f"\n5. Mean Difference: {bootstrap_results['mean_diff']:.4f}")
        print(f"   - 95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
        
        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE ".center(80))
        print("="*80)
        
        return {
            'paired_df': paired_df,
            'basic_stats': basic_stats,
            't_test': (t_stat, t_p),
            'wilcoxon': (w_stat, w_p),
            'correlations': correlations,
            'kappa': kappa,
            'bootstrap': bootstrap_results,
            'convergence': convergence_data,
            'task_analysis': task_analysis
        }


def main():
    """Main function to run the statistical comparison."""
    comparator = EvaluationComparator()
    results = comparator.run_full_analysis()
    
    # Save results to file
    output_file = "analysis/statistics/statistical_comparison_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    save_results = {
        'summary': {
            'n_samples': len(results['paired_df']),
            't_test_p_value': float(results['t_test'][1]),
            'wilcoxon_p_value': float(results['wilcoxon'][1]),
            'pearson_correlation': float(results['correlations']['pearson'][0]),
            'cohens_kappa': float(results['kappa']),
            'mean_difference': float(results['bootstrap']['mean_diff']),
            'ci_lower': float(results['bootstrap']['ci_lower']),
            'ci_upper': float(results['bootstrap']['ci_upper']),
            'convergence_threshold': results['convergence']['convergence_threshold']
        },
        'task_analysis': results['task_analysis'].to_dict('records')
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
