"""
Genomic Analysis for FOLFOX vs FOLFIRI Study
Author: Abhinav Agarwal, Stanford University
Co-Author: Casey Nguyen, KOS AI, Stanford Research Park
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

# Set publication quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

class GenomicAnalysis:
    """
    Comprehensive genomic analysis for colorectal cancer cohort
    """
    
    def __init__(self):
        """Initialize genomic analysis"""
        print("="*70)
        print("GENOMIC ANALYSIS FOR COLORECTAL CANCER")
        print("Abhinav Agarwal, Stanford University")
        print("Casey Nguyen, KOS AI, Stanford Research Park")
        print("="*70)
        
        # Load cohort data
        self.cohort = pd.read_csv('analysis_cohort.csv')
        self.mutations = None
        self.cna = None
        self.sv = None
        
    def load_genomic_data(self):
        """Load and process genomic data"""
        print("\n" + "="*70)
        print("1. LOADING GENOMIC DATA")
        print("="*70)
        
        # Load mutations
        print("\nLoading mutation data...")
        mutations_df = pd.read_csv('msk_chord_2024/data_mutations.txt', sep='\t', low_memory=False)
        
        # Filter for our cohort patients
        cohort_patients = set(self.cohort['PATIENT_ID'].unique())
        mutations_df['PATIENT_ID'] = mutations_df['Tumor_Sample_Barcode'].str.extract(r'(P-\d+)')
        self.mutations = mutations_df[mutations_df['PATIENT_ID'].isin(cohort_patients)]
        
        print(f"  Loaded {len(self.mutations)} mutations for {len(self.mutations['PATIENT_ID'].unique())} patients")
        print(f"  Cohort coverage: {len(self.mutations['PATIENT_ID'].unique())}/{len(cohort_patients)} " +
              f"({100*len(self.mutations['PATIENT_ID'].unique())/len(cohort_patients):.1f}%)")
        
        # Load copy number alterations
        try:
            print("\nLoading copy number data...")
            cna_df = pd.read_csv('msk_chord_2024/data_cna.txt', sep='\t', low_memory=False)
            # Process CNA data (simplified for now)
            self.cna = cna_df
            print(f"  Loaded CNA data for {len(cna_df.columns)-1} samples")
        except:
            print("  CNA data loading skipped")
        
        return self.mutations
    
    def analyze_key_mutations(self):
        """Analyze key colorectal cancer driver mutations"""
        print("\n" + "="*70)
        print("2. KEY DRIVER MUTATIONS ANALYSIS")
        print("="*70)
        
        # Key CRC genes
        key_genes = {
            'RAS_pathway': ['KRAS', 'NRAS', 'HRAS'],
            'BRAF': ['BRAF'],
            'TP53': ['TP53'],
            'APC': ['APC'],
            'PIK3CA': ['PIK3CA'],
            'SMAD4': ['SMAD4'],
            'FBXW7': ['FBXW7'],
            'TCF7L2': ['TCF7L2'],
            'MSI_genes': ['MLH1', 'MSH2', 'MSH6', 'PMS2'],
            'EGFR_pathway': ['EGFR', 'ERBB2', 'ERBB3']
        }
        
        mutation_summary = []
        
        for category, genes in key_genes.items():
            # Count mutations
            mutated_patients = self.mutations[
                self.mutations['Hugo_Symbol'].isin(genes)
            ]['PATIENT_ID'].unique()
            
            n_mutated = len(mutated_patients)
            pct_mutated = 100 * n_mutated / len(self.cohort)
            
            mutation_summary.append({
                'Gene_Category': category,
                'Genes': ', '.join(genes),
                'N_Mutated': n_mutated,
                'Percent_Mutated': pct_mutated
            })
            
            print(f"\n{category}:")
            print(f"  Genes: {', '.join(genes)}")
            print(f"  Mutated: {n_mutated}/{len(self.cohort)} ({pct_mutated:.1f}%)")
            
            # Analyze specific mutations for key genes
            if category == 'RAS_pathway':
                self._analyze_ras_mutations()
            elif category == 'BRAF':
                self._analyze_braf_mutations()
        
        self.mutation_summary_df = pd.DataFrame(mutation_summary)
        self.mutation_summary_df.to_csv('research_paper/genomic_mutation_summary.csv', index=False)
        
        return self.mutation_summary_df
    
    def _analyze_ras_mutations(self):
        """Detailed RAS mutation analysis"""
        ras_mutations = self.mutations[self.mutations['Hugo_Symbol'].isin(['KRAS', 'NRAS', 'HRAS'])]
        
        if len(ras_mutations) > 0:
            # Common KRAS mutations
            kras_muts = ras_mutations[ras_mutations['Hugo_Symbol'] == 'KRAS']
            if len(kras_muts) > 0:
                common_kras = kras_muts['HGVSp_Short'].value_counts().head(5)
                print("\n  Top KRAS mutations:")
                for mut, count in common_kras.items():
                    print(f"    {mut}: {count} patients")
    
    def _analyze_braf_mutations(self):
        """Detailed BRAF mutation analysis"""
        braf_mutations = self.mutations[self.mutations['Hugo_Symbol'] == 'BRAF']
        
        if len(braf_mutations) > 0:
            # Check for V600E
            v600e = braf_mutations[braf_mutations['HGVSp_Short'].str.contains('V600E', na=False)]
            print(f"\n  BRAF V600E: {len(v600e['PATIENT_ID'].unique())} patients")
    
    def correlate_with_treatment(self):
        """Correlate mutations with treatment response"""
        print("\n" + "="*70)
        print("3. MUTATION-TREATMENT CORRELATIONS")
        print("="*70)
        
        # Add mutation status to cohort
        for gene_set_name, genes in [
            ('RAS_mut', ['KRAS', 'NRAS', 'HRAS']),
            ('BRAF_mut', ['BRAF']),
            ('TP53_mut', ['TP53']),
            ('APC_mut', ['APC']),
            ('PIK3CA_mut', ['PIK3CA'])
        ]:
            mutated_patients = self.mutations[
                self.mutations['Hugo_Symbol'].isin(genes)
            ]['PATIENT_ID'].unique()
            
            self.cohort[gene_set_name] = self.cohort['PATIENT_ID'].isin(mutated_patients).astype(int)
        
        # Analyze treatment response by mutation status
        results = []
        
        for mutation in ['RAS_mut', 'BRAF_mut', 'TP53_mut', 'APC_mut', 'PIK3CA_mut']:
            print(f"\n{mutation}:")
            
            # Overall distribution
            mut_pos = self.cohort[mutation].sum()
            mut_neg = len(self.cohort) - mut_pos
            print(f"  Positive: {mut_pos}, Negative: {mut_neg}")
            
            if mut_pos < 10:  # Skip if too few mutated patients
                print("  Too few mutated patients for analysis")
                continue
            
            # By treatment
            for treatment in ['FOLFOX', 'FOLFIRI']:
                treated = self.cohort[self.cohort['FIRST_LINE'] == treatment]
                mut_treated = treated[mutation].sum()
                pct = 100 * mut_treated / len(treated)
                print(f"  {treatment}: {mut_treated}/{len(treated)} ({pct:.1f}%)")
            
            # Survival analysis by mutation and treatment
            for treatment in ['FOLFOX', 'FOLFIRI']:
                treated = self.cohort[self.cohort['FIRST_LINE'] == treatment]
                
                if treated[mutation].sum() >= 5:  # Need sufficient numbers
                    mut_pos_survival = treated[treated[mutation] == 1]['OS_MONTHS'].median()
                    mut_neg_survival = treated[treated[mutation] == 0]['OS_MONTHS'].median()
                    
                    # Log-rank test
                    try:
                        lr_result = logrank_test(
                            treated[treated[mutation] == 1]['OS_MONTHS'],
                            treated[treated[mutation] == 0]['OS_MONTHS'],
                            treated[treated[mutation] == 1]['OS_STATUS'].str.contains('DECEASED'),
                            treated[treated[mutation] == 0]['OS_STATUS'].str.contains('DECEASED')
                        )
                        p_value = lr_result.p_value
                    except:
                        p_value = np.nan
                    
                    results.append({
                        'Mutation': mutation,
                        'Treatment': treatment,
                        'Mut_Pos_Median_OS': mut_pos_survival,
                        'Mut_Neg_Median_OS': mut_neg_survival,
                        'Difference': mut_pos_survival - mut_neg_survival,
                        'P_value': p_value
                    })
                    
                    print(f"    {treatment} OS: Mut+ {mut_pos_survival:.1f} vs Mut- {mut_neg_survival:.1f} months")
        
        self.mutation_treatment_results = pd.DataFrame(results)
        self.mutation_treatment_results.to_csv('research_paper/mutation_treatment_correlations.csv', index=False)
        
        return self.mutation_treatment_results
    
    def analyze_mutation_burden(self):
        """Analyze tumor mutation burden (TMB)"""
        print("\n" + "="*70)
        print("4. TUMOR MUTATION BURDEN ANALYSIS")
        print("="*70)
        
        # Calculate TMB per patient
        tmb = self.mutations.groupby('PATIENT_ID').size().reset_index(name='TMB')
        
        # Merge with cohort
        self.cohort = self.cohort.merge(tmb, on='PATIENT_ID', how='left')
        self.cohort['TMB'] = self.cohort['TMB'].fillna(0)
        
        # TMB statistics
        print(f"\nTMB Statistics:")
        print(f"  Median: {self.cohort['TMB'].median():.1f}")
        print(f"  Mean: {self.cohort['TMB'].mean():.1f}")
        print(f"  Range: {self.cohort['TMB'].min():.0f}-{self.cohort['TMB'].max():.0f}")
        
        # TMB by treatment
        print(f"\nTMB by Treatment:")
        for treatment in ['FOLFOX', 'FOLFIRI']:
            treated = self.cohort[self.cohort['FIRST_LINE'] == treatment]
            print(f"  {treatment}: Median {treated['TMB'].median():.1f}, Mean {treated['TMB'].mean():.1f}")
        
        # TMB categories
        tmb_median = self.cohort['TMB'].median()
        self.cohort['TMB_high'] = (self.cohort['TMB'] > tmb_median).astype(int)
        
        # Survival by TMB
        print(f"\nSurvival by TMB:")
        tmb_high_os = self.cohort[self.cohort['TMB_high'] == 1]['OS_MONTHS'].median()
        tmb_low_os = self.cohort[self.cohort['TMB_high'] == 0]['OS_MONTHS'].median()
        print(f"  TMB-high: {tmb_high_os:.1f} months")
        print(f"  TMB-low: {tmb_low_os:.1f} months")
        
        # TMB interaction with treatment
        print(f"\nTMB-Treatment Interaction:")
        for treatment in ['FOLFOX', 'FOLFIRI']:
            treated = self.cohort[self.cohort['FIRST_LINE'] == treatment]
            tmb_high_treated = treated[treated['TMB_high'] == 1]
            tmb_low_treated = treated[treated['TMB_high'] == 0]
            
            if len(tmb_high_treated) >= 5:
                print(f"  {treatment}:")
                print(f"    TMB-high: {tmb_high_treated['OS_MONTHS'].median():.1f} months (n={len(tmb_high_treated)})")
                print(f"    TMB-low: {tmb_low_treated['OS_MONTHS'].median():.1f} months (n={len(tmb_low_treated)})")
    
    def create_genomic_visualizations(self):
        """Create publication-quality genomic visualizations"""
        print("\n" + "="*70)
        print("5. GENERATING GENOMIC VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Mutation frequency waterfall plot
        if hasattr(self, 'mutation_summary_df'):
            mut_df = self.mutation_summary_df.sort_values('Percent_Mutated', ascending=True)
            axes[0, 0].barh(range(len(mut_df)), mut_df['Percent_Mutated'], 
                          color=plt.cm.viridis(np.linspace(0.3, 0.9, len(mut_df))))
            axes[0, 0].set_yticks(range(len(mut_df)))
            axes[0, 0].set_yticklabels(mut_df['Gene_Category'])
            axes[0, 0].set_xlabel('Mutation Frequency (%)')
            axes[0, 0].set_title('Driver Mutation Frequencies', fontweight='bold')
            axes[0, 0].axvline(x=50, color='red', linestyle='--', alpha=0.3)
        
        # 2. TMB distribution
        if 'TMB' in self.cohort.columns:
            axes[0, 1].hist(self.cohort['TMB'], bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(self.cohort['TMB'].median(), color='red', 
                             linestyle='--', label=f'Median: {self.cohort["TMB"].median():.1f}')
            axes[0, 1].set_xlabel('Tumor Mutation Burden')
            axes[0, 1].set_ylabel('Number of Patients')
            axes[0, 1].set_title('TMB Distribution', fontweight='bold')
            axes[0, 1].legend()
        
        # 3. Mutation co-occurrence heatmap
        if 'RAS_mut' in self.cohort.columns:
            mutation_cols = [col for col in self.cohort.columns if col.endswith('_mut')]
            if len(mutation_cols) >= 2:
                mut_matrix = self.cohort[mutation_cols]
                correlation = mut_matrix.corr()
                sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                          center=0, ax=axes[0, 2], cbar_kws={'label': 'Correlation'})
                axes[0, 2].set_title('Mutation Co-occurrence', fontweight='bold')
        
        # 4. Treatment response by RAS status
        if 'RAS_mut' in self.cohort.columns:
            ras_data = []
            for treatment in ['FOLFOX', 'FOLFIRI']:
                for ras_status in [0, 1]:
                    subset = self.cohort[(self.cohort['FIRST_LINE'] == treatment) & 
                                       (self.cohort['RAS_mut'] == ras_status)]
                    if len(subset) > 0:
                        ras_data.append({
                            'Treatment': treatment,
                            'RAS': 'Mutant' if ras_status else 'Wild-type',
                            'Median_OS': subset['OS_MONTHS'].median()
                        })
            
            if ras_data:
                ras_df = pd.DataFrame(ras_data)
                ras_pivot = ras_df.pivot(index='RAS', columns='Treatment', values='Median_OS')
                ras_pivot.plot(kind='bar', ax=axes[1, 0], color=['#2E86AB', '#F18F01'])
                axes[1, 0].set_ylabel('Median OS (months)')
                axes[1, 0].set_title('Survival by RAS Status', fontweight='bold')
                axes[1, 0].legend(title='Treatment')
                axes[1, 0].set_xlabel('RAS Status')
        
        # 5. BRAF V600E analysis
        if 'BRAF_mut' in self.cohort.columns:
            braf_data = []
            for treatment in ['FOLFOX', 'FOLFIRI']:
                for braf_status in [0, 1]:
                    subset = self.cohort[(self.cohort['FIRST_LINE'] == treatment) & 
                                       (self.cohort['BRAF_mut'] == braf_status)]
                    if len(subset) > 0:
                        braf_data.append({
                            'Treatment': treatment,
                            'BRAF': 'V600E' if braf_status else 'Wild-type',
                            'N': len(subset),
                            'Median_OS': subset['OS_MONTHS'].median()
                        })
            
            if braf_data:
                braf_df = pd.DataFrame(braf_data)
                # Show sample sizes
                x = np.arange(len(braf_df))
                axes[1, 1].bar(x, braf_df['Median_OS'], color=['#2E86AB', '#2E86AB', '#F18F01', '#F18F01'])
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([f"{row['Treatment']}\n{row['BRAF']}\n(n={row['N']})" 
                                           for _, row in braf_df.iterrows()], fontsize=8)
                axes[1, 1].set_ylabel('Median OS (months)')
                axes[1, 1].set_title('BRAF V600E Impact', fontweight='bold')
        
        # 6. Mutation landscape by treatment
        if hasattr(self, 'mutation_treatment_results') and len(self.mutation_treatment_results) > 0:
            # Create forest plot of mutation effects
            mut_effects = self.mutation_treatment_results[
                self.mutation_treatment_results['Treatment'] == 'FOLFOX'
            ].copy()
            
            if len(mut_effects) > 0:
                y_pos = np.arange(len(mut_effects))
                axes[1, 2].barh(y_pos, mut_effects['Difference'], 
                              color=['red' if x < 0 else 'green' for x in mut_effects['Difference']])
                axes[1, 2].set_yticks(y_pos)
                axes[1, 2].set_yticklabels(mut_effects['Mutation'].str.replace('_mut', ''))
                axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=1)
                axes[1, 2].set_xlabel('OS Difference (Mut+ vs Mut-) in months')
                axes[1, 2].set_title('Mutation Impact on FOLFOX', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('research_paper/figures/genomic_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('research_paper/figures/genomic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Figures saved to research_paper/figures/genomic_analysis.*")
    
    def integrate_with_ml_models(self):
        """Add genomic features to machine learning models"""
        print("\n" + "="*70)
        print("6. INTEGRATING GENOMIC FEATURES WITH ML MODELS")
        print("="*70)
        
        # Prepare genomic features
        genomic_features = []
        
        # Add mutation status features
        mutation_cols = [col for col in self.cohort.columns if col.endswith('_mut')]
        if mutation_cols:
            genomic_features.extend(mutation_cols)
        
        # Add TMB if available
        if 'TMB' in self.cohort.columns:
            genomic_features.append('TMB')
            genomic_features.append('TMB_high')
        
        print(f"\nGenomic features for ML: {genomic_features}")
        
        if genomic_features:
            # Prepare data for modeling
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            
            # Basic features
            basic_features = ['CURRENT_AGE_DEID', 'GENDER_MALE', 'NUM_LINES']
            self.cohort['GENDER_MALE'] = (self.cohort['GENDER'] == 'Male').astype(int)
            self.cohort['TREATMENT'] = (self.cohort['FIRST_LINE'] == 'FOLFOX').astype(int)
            
            # Combined features
            all_features = basic_features + genomic_features + ['TREATMENT']
            
            # Prepare outcome (e.g., 2-year survival)
            self.cohort['SURVIVAL_2YR'] = (self.cohort['OS_MONTHS'] >= 24).astype(int)
            
            # Remove rows with missing values
            model_data = self.cohort[all_features + ['SURVIVAL_2YR']].dropna()
            
            X = model_data[all_features]
            y = model_data['SURVIVAL_2YR']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            print("\nModel Performance:")
            
            # Model without genomic features
            rf_basic = RandomForestClassifier(n_estimators=100, random_state=42)
            basic_indices = [i for i, f in enumerate(all_features) if f not in genomic_features]
            rf_basic.fit(X_train_scaled[:, basic_indices], y_train)
            auc_basic = roc_auc_score(y_test, rf_basic.predict_proba(X_test_scaled[:, basic_indices])[:, 1])
            print(f"  Without genomic features: AUC = {auc_basic:.3f}")
            
            # Model with genomic features
            rf_genomic = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_genomic.fit(X_train_scaled, y_train)
            auc_genomic = roc_auc_score(y_test, rf_genomic.predict_proba(X_test_scaled)[:, 1])
            print(f"  With genomic features: AUC = {auc_genomic:.3f}")
            print(f"  Improvement: {auc_genomic - auc_basic:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': rf_genomic.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop Genomic Feature Importance:")
            for _, row in feature_importance[feature_importance['Feature'].isin(genomic_features)].head().iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.3f}")
            
            feature_importance.to_csv('research_paper/genomic_feature_importance.csv', index=False)

def main():
    """Run complete genomic analysis"""
    analyzer = GenomicAnalysis()
    
    # Load genomic data
    analyzer.load_genomic_data()
    
    # Analyze key mutations
    analyzer.analyze_key_mutations()
    
    # Correlate with treatment
    analyzer.correlate_with_treatment()
    
    # Analyze mutation burden
    analyzer.analyze_mutation_burden()
    
    # Create visualizations
    analyzer.create_genomic_visualizations()
    
    # Integrate with ML models
    analyzer.integrate_with_ml_models()
    
    print("\n" + "="*70)
    print("GENOMIC ANALYSIS COMPLETE")
    print("Results saved to research_paper/")
    print("="*70)

if __name__ == "__main__":
    main()
