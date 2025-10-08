#!/usr/bin/env python3
"""
Statistical Analysis and Modeling for FOLFOX vs FOLFIRI Sequencing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_analysis_data():
    """Load the preprocessed analysis cohort"""
    print("Loading analysis cohort...")
    
    # Load the saved cohort
    analysis_df = pd.read_csv('analysis_cohort.csv')
    
    # Load sample data for additional features
    sample_data = pd.read_csv('msk_chord_2024/data_clinical_sample.txt', 
                              sep='\t', comment='#', low_memory=False)
    
    # Filter for colorectal samples
    colorectal_samples = sample_data[
        sample_data['CANCER_TYPE'].str.contains('Colorectal', case=False, na=False)
    ]
    
    # Merge MSI status from samples
    msi_data = colorectal_samples[['PATIENT_ID', 'MSI_TYPE', 'MSI_SCORE']].drop_duplicates('PATIENT_ID')
    analysis_df = analysis_df.merge(msi_data, on='PATIENT_ID', how='left')
    
    # Clean data
    analysis_df['OS_STATUS'] = analysis_df['OS_STATUS'].str.extract(r'(\d+)').astype(float)
    
    print(f"Loaded {len(analysis_df)} patients for analysis")
    
    return analysis_df

def survival_analysis_by_firstline(df):
    """Perform survival analysis comparing first-line treatments"""
    print("\n" + "="*80)
    print("SURVIVAL ANALYSIS BY FIRST-LINE TREATMENT")
    print("="*80)
    
    # Filter for patients with OS data
    survival_df = df[df['OS_MONTHS'].notna() & df['OS_STATUS'].notna()].copy()
    
    # Create Kaplan-Meier curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: FOLFOX vs FOLFIRI first-line
    kmf = KaplanMeierFitter()
    
    for treatment, color in [('FOLFOX', 'blue'), ('FOLFIRI', 'red')]:
        mask = survival_df['FIRST_LINE'] == treatment
        kmf.fit(
            survival_df[mask]['OS_MONTHS'],
            survival_df[mask]['OS_STATUS'],
            label=f'{treatment} (n={mask.sum()})'
        )
        kmf.plot_survival_function(ax=ax1, color=color)
    
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Overall Survival Probability')
    ax1.set_title('Overall Survival by First-Line Treatment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-rank test
    folfox_mask = survival_df['FIRST_LINE'] == 'FOLFOX'
    folfiri_mask = survival_df['FIRST_LINE'] == 'FOLFIRI'
    
    results = logrank_test(
        survival_df[folfox_mask]['OS_MONTHS'],
        survival_df[folfiri_mask]['OS_MONTHS'],
        survival_df[folfox_mask]['OS_STATUS'],
        survival_df[folfiri_mask]['OS_STATUS']
    )
    
    print(f"\nLog-rank test (FOLFOX vs FOLFIRI first-line):")
    print(f"  Test statistic: {results.test_statistic:.3f}")
    print(f"  p-value: {results.p_value:.4f}")
    
    # Plot 2: Treatment sequences
    sequences_to_plot = ['FOLFOX_only', 'FOLFIRI_only', 'FOLFOX_to_FOLFIRI', 'FOLFIRI_to_FOLFOX']
    colors = ['blue', 'red', 'green', 'orange']
    
    for seq, color in zip(sequences_to_plot, colors):
        mask = survival_df['TREATMENT_SEQUENCE'] == seq
        if mask.sum() >= 10:  # Only plot if sufficient sample size
            kmf.fit(
                survival_df[mask]['OS_MONTHS'],
                survival_df[mask]['OS_STATUS'],
                label=f'{seq} (n={mask.sum()})'
            )
            kmf.plot_survival_function(ax=ax2, color=color)
    
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Overall Survival Probability')
    ax2.set_title('Overall Survival by Treatment Sequence')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('survival_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate median survival for each sequence
    print("\nMedian Overall Survival by Treatment Sequence:")
    for seq in survival_df['TREATMENT_SEQUENCE'].value_counts().index[:6]:
        subset = survival_df[survival_df['TREATMENT_SEQUENCE'] == seq]
        if len(subset) >= 10:
            kmf.fit(subset['OS_MONTHS'], subset['OS_STATUS'])
            median_survival = kmf.median_survival_time_
            print(f"  {seq}: {median_survival:.1f} months (n={len(subset)})")

def cox_regression_analysis(df):
    """Perform Cox proportional hazards regression"""
    print("\n" + "="*80)
    print("COX PROPORTIONAL HAZARDS REGRESSION")
    print("="*80)
    
    # Prepare data for Cox regression
    cox_df = df[df['OS_MONTHS'].notna() & df['OS_STATUS'].notna()].copy()
    
    # Create binary variables
    cox_df['FIRST_LINE_FOLFOX'] = (cox_df['FIRST_LINE'] == 'FOLFOX').astype(int)
    cox_df['AGE_OVER_65'] = (cox_df['CURRENT_AGE_DEID'] >= 65).astype(int)
    cox_df['MALE'] = (cox_df['GENDER'] == 'Male').astype(int)
    cox_df['MSI_HIGH'] = (cox_df['MSI_TYPE'] == 'Instable').astype(int)
    cox_df['RECEIVED_SECOND_LINE'] = (cox_df['NUM_LINES'] >= 2).astype(int)
    
    # Select features for Cox model
    features = ['FIRST_LINE_FOLFOX', 'AGE_OVER_65', 'MALE', 'MSI_HIGH', 
                'RECEIVED_SECOND_LINE', 'OS_MONTHS', 'OS_STATUS']
    
    cox_data = cox_df[features].dropna()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='OS_MONTHS', event_col='OS_STATUS')
    
    print("\nCox Regression Results:")
    print(cph.summary)
    
    # Plot hazard ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax)
    ax.set_title('Hazard Ratios with 95% Confidence Intervals')
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('hazard_ratios.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return cph

def subgroup_analysis(df):
    """Analyze treatment effectiveness in patient subgroups"""
    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS")
    print("="*80)
    
    subgroups = {
        'Age < 65': df['CURRENT_AGE_DEID'] < 65,
        'Age ≥ 65': df['CURRENT_AGE_DEID'] >= 65,
        'Male': df['GENDER'] == 'Male',
        'Female': df['GENDER'] == 'Female',
        'MSI-Stable': df['MSI_TYPE'] == 'Stable',
        'MSI-High': df['MSI_TYPE'] == 'Instable'
    }
    
    results = []
    
    for subgroup_name, mask in subgroups.items():
        subgroup_df = df[mask & df['OS_MONTHS'].notna() & df['OS_STATUS'].notna()]
        
        if len(subgroup_df) < 20:
            continue
        
        # Calculate median OS for FOLFOX vs FOLFIRI
        folfox_os = subgroup_df[subgroup_df['FIRST_LINE'] == 'FOLFOX']['OS_MONTHS'].median()
        folfiri_os = subgroup_df[subgroup_df['FIRST_LINE'] == 'FOLFIRI']['OS_MONTHS'].median()
        
        # Count patients
        n_folfox = (subgroup_df['FIRST_LINE'] == 'FOLFOX').sum()
        n_folfiri = (subgroup_df['FIRST_LINE'] == 'FOLFIRI').sum()
        
        # Perform log-rank test if sufficient sample size
        if n_folfox >= 10 and n_folfiri >= 10:
            folfox_mask = subgroup_df['FIRST_LINE'] == 'FOLFOX'
            folfiri_mask = subgroup_df['FIRST_LINE'] == 'FOLFIRI'
            
            log_rank = logrank_test(
                subgroup_df[folfox_mask]['OS_MONTHS'],
                subgroup_df[folfiri_mask]['OS_MONTHS'],
                subgroup_df[folfox_mask]['OS_STATUS'],
                subgroup_df[folfiri_mask]['OS_STATUS']
            )
            p_value = log_rank.p_value
        else:
            p_value = np.nan
        
        results.append({
            'Subgroup': subgroup_name,
            'N_Total': len(subgroup_df),
            'N_FOLFOX': n_folfox,
            'N_FOLFIRI': n_folfiri,
            'Median_OS_FOLFOX': folfox_os,
            'Median_OS_FOLFIRI': folfiri_os,
            'Difference': folfox_os - folfiri_os if not pd.isna(folfox_os) and not pd.isna(folfiri_os) else np.nan,
            'P_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\nMedian Overall Survival by Subgroup:")
    for _, row in results_df.iterrows():
        print(f"\n{row['Subgroup']} (n={row['N_Total']}):")
        print(f"  FOLFOX: {row['Median_OS_FOLFOX']:.1f} months (n={row['N_FOLFOX']})")
        print(f"  FOLFIRI: {row['Median_OS_FOLFIRI']:.1f} months (n={row['N_FOLFIRI']})")
        if not pd.isna(row['Difference']):
            print(f"  Difference: {row['Difference']:.1f} months")
        if not pd.isna(row['P_value']):
            print(f"  P-value: {row['P_value']:.4f}")
    
    # Create forest plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_results = results_df.dropna(subset=['Difference'])
    y_pos = np.arange(len(valid_results))
    
    # Plot differences with error bars (simplified - would need CIs ideally)
    ax.barh(y_pos, valid_results['Difference'], xerr=2, align='center', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_results['Subgroup'])
    ax.set_xlabel('Difference in Median OS (FOLFOX - FOLFIRI) in months')
    ax.set_title('Subgroup Analysis: Treatment Effect on Overall Survival')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add sample sizes to labels
    for i, (idx, row) in enumerate(valid_results.iterrows()):
        ax.text(row['Difference'] + 0.5, i, 
                f"n={row['N_FOLFOX']}/{row['N_FOLFIRI']}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('subgroup_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results_df

def sequence_effectiveness_analysis(df):
    """Analyze effectiveness of treatment sequences"""
    print("\n" + "="*80)
    print("TREATMENT SEQUENCE EFFECTIVENESS")
    print("="*80)
    
    # Focus on main sequences
    main_sequences = ['FOLFOX_only', 'FOLFIRI_only', 'FOLFOX_to_FOLFIRI', 'FOLFIRI_to_FOLFOX']
    
    sequence_stats = []
    for seq in main_sequences:
        seq_df = df[df['TREATMENT_SEQUENCE'] == seq]
        
        if len(seq_df) >= 10:
            stats_dict = {
                'Sequence': seq,
                'N': len(seq_df),
                'Median_OS': seq_df['OS_MONTHS'].median(),
                'Mean_OS': seq_df['OS_MONTHS'].mean(),
                'Std_OS': seq_df['OS_MONTHS'].std(),
                'Deaths': seq_df['OS_STATUS'].sum(),
                'Death_Rate': seq_df['OS_STATUS'].mean()
            }
            sequence_stats.append(stats_dict)
    
    sequence_stats_df = pd.DataFrame(sequence_stats)
    
    print("\nTreatment Sequence Statistics:")
    print(sequence_stats_df.to_string(index=False))
    
    # Statistical comparison
    print("\n\nPairwise comparisons (Mann-Whitney U test):")
    for i in range(len(main_sequences)):
        for j in range(i+1, len(main_sequences)):
            seq1, seq2 = main_sequences[i], main_sequences[j]
            data1 = df[df['TREATMENT_SEQUENCE'] == seq1]['OS_MONTHS'].dropna()
            data2 = df[df['TREATMENT_SEQUENCE'] == seq2]['OS_MONTHS'].dropna()
            
            if len(data1) >= 10 and len(data2) >= 10:
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"  {seq1} vs {seq2}: p = {p_value:.4f}")
    
    return sequence_stats_df

def main():
    """Main statistical analysis pipeline"""
    print("="*80)
    print("STATISTICAL ANALYSIS: FOLFOX vs FOLFIRI SEQUENCING")
    print("="*80)
    
    # Load data
    df = load_analysis_data()
    
    # Basic descriptive statistics
    print("\nPatient Characteristics:")
    print(f"  Total patients: {len(df)}")
    print(f"  Median age: {df['CURRENT_AGE_DEID'].median():.1f} years")
    print(f"  Male: {(df['GENDER']=='Male').mean():.1%}")
    print(f"  Deaths: {df['OS_STATUS'].sum()} ({df['OS_STATUS'].mean():.1%})")
    
    # Survival analysis
    survival_analysis_by_firstline(df)
    
    # Cox regression
    cox_model = cox_regression_analysis(df)
    
    # Subgroup analysis
    subgroup_results = subgroup_analysis(df)
    
    # Sequence effectiveness
    sequence_stats = sequence_effectiveness_analysis(df)
    
    # Save results
    print("\nSaving analysis results...")
    subgroup_results.to_csv('subgroup_analysis_results.csv', index=False)
    sequence_stats.to_csv('sequence_statistics.csv', index=False)
    
    print("\nAnalysis complete! Results saved to CSV files and plots generated.")
    
    return df, cox_model, subgroup_results, sequence_stats

if __name__ == "__main__":
    df, cox_model, subgroup_results, sequence_stats = main()
