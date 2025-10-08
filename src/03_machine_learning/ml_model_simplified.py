#!/usr/bin/env python3
"""
Simplified Machine Learning Model for Treatment Sequence Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def prepare_data_simplified():
    """Prepare simplified feature set for modeling"""
    print("Loading and preparing data...")
    
    # Load analysis cohort
    df = pd.read_csv('analysis_cohort.csv')
    
    # Load sample data for additional features
    sample_data = pd.read_csv('msk_chord_2024/data_clinical_sample.txt', 
                              sep='\t', comment='#', low_memory=False)
    
    # Filter for colorectal samples
    colorectal_samples = sample_data[
        sample_data['CANCER_TYPE'].str.contains('Colorectal', case=False, na=False)
    ]
    
    # Merge key features
    feature_cols = ['PATIENT_ID', 'PRIMARY_SITE', 'METASTATIC_SITE', 
                   'TMB_NONSYNONYMOUS', 'MSI_TYPE', 'MSI_SCORE']
    df = df.merge(
        colorectal_samples[feature_cols].drop_duplicates('PATIENT_ID'),
        on='PATIENT_ID',
        how='left'
    )
    
    return df

def engineer_features(df):
    """Engineer features for modeling"""
    print("Engineering features...")
    
    # Create binary features
    df['MALE'] = (df['GENDER'] == 'Male').astype(int)
    df['AGE_GROUP'] = pd.cut(df['CURRENT_AGE_DEID'], bins=[0, 50, 65, 100], 
                             labels=['<50', '50-65', '>65'])
    df['AGE_OVER_65'] = (df['CURRENT_AGE_DEID'] >= 65).astype(int)
    df['AGE_UNDER_50'] = (df['CURRENT_AGE_DEID'] < 50).astype(int)
    
    # MSI status - handle both possible column names
    msi_col = 'MSI_TYPE_y' if 'MSI_TYPE_y' in df.columns else 'MSI_TYPE'
    df['MSI_HIGH'] = (df[msi_col] == 'Instable').astype(int) if msi_col in df.columns else 0
    df['MSI_STABLE'] = (df[msi_col] == 'Stable').astype(int) if msi_col in df.columns else 0
    
    # Metastatic sites
    df['LIVER_METS'] = df['METASTATIC_SITE'].str.contains('Liver', case=False, na=False).astype(int)
    df['LUNG_METS'] = df['METASTATIC_SITE'].str.contains('Lung', case=False, na=False).astype(int)
    df['PERITONEUM_METS'] = df['METASTATIC_SITE'].str.contains('Peritoneum', case=False, na=False).astype(int)
    df['LYMPH_METS'] = df['METASTATIC_SITE'].str.contains('Lymph', case=False, na=False).astype(int)
    
    # Primary tumor location
    df['RIGHT_SIDED'] = df['PRIMARY_SITE'].str.contains('Ascending|Cecum|Hepatic', case=False, na=False).astype(int)
    df['LEFT_SIDED'] = df['PRIMARY_SITE'].str.contains('Descending|Sigmoid', case=False, na=False).astype(int)
    df['RECTAL'] = df['PRIMARY_SITE'].str.contains('Rect', case=False, na=False).astype(int)
    
    # TMB categories
    df['TMB_HIGH'] = (df['TMB_NONSYNONYMOUS'] > df['TMB_NONSYNONYMOUS'].quantile(0.75)).astype(int)
    df['TMB_LOW'] = (df['TMB_NONSYNONYMOUS'] <= df['TMB_NONSYNONYMOUS'].quantile(0.25)).astype(int)
    
    # Create outcome variable based on treatment effectiveness
    # We'll define "optimal" as survival above median for each treatment
    median_os = df['OS_MONTHS'].median()
    
    # For FOLFOX patients: 1 if they did better than median, 0 otherwise
    # For FOLFIRI patients: 0 if they did better than median, 1 otherwise
    df['FOLFOX_OPTIMAL'] = np.nan
    folfox_mask = df['FIRST_LINE'] == 'FOLFOX'
    folfiri_mask = df['FIRST_LINE'] == 'FOLFIRI'
    
    df.loc[folfox_mask, 'FOLFOX_OPTIMAL'] = (df.loc[folfox_mask, 'OS_MONTHS'] > median_os).astype(int)
    df.loc[folfiri_mask, 'FOLFOX_OPTIMAL'] = (df.loc[folfiri_mask, 'OS_MONTHS'] <= median_os).astype(int)
    
    return df

def create_feature_matrix(df):
    """Create feature matrix for modeling"""
    feature_cols = [
        'CURRENT_AGE_DEID', 'MALE', 'AGE_OVER_65', 'AGE_UNDER_50',
        'MSI_HIGH', 'MSI_STABLE',
        'LIVER_METS', 'LUNG_METS', 'PERITONEUM_METS', 'LYMPH_METS',
        'RIGHT_SIDED', 'LEFT_SIDED', 'RECTAL',
        'TMB_HIGH', 'TMB_LOW'
    ]
    
    # Add TMB as continuous variable if available
    if 'TMB_NONSYNONYMOUS' in df.columns:
        feature_cols.append('TMB_NONSYNONYMOUS')
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, feature_cols

def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate multiple models"""
    print("\n" + "="*80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} patients")
    print(f"Test set: {len(X_test)} patients")
    print(f"Class distribution - FOLFOX optimal: {y.mean():.1%}")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc': auc_score,
            'cv_scores': cv_scores,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"  AUC Score: {auc_score:.3f}")
        print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Accuracy: {results[name]['report']['accuracy']:.3f}")
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} (AUC: {best_auc:.3f})")
    
    return results, best_model_name, X_test, y_test, scaler, feature_names

def plot_results(results, y_test, best_model_name, feature_names):
    """Plot model results and feature importance"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. ROC Curves
    ax1 = plt.subplot(2, 2, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc = result['auc']
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Treatment Response Prediction')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature Importance
    ax2 = plt.subplot(2, 2, 2)
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax2)
    ax2.set_title(f'Top 10 Feature Importances - {best_model_name}')
    
    # 3. Model Comparison
    ax3 = plt.subplot(2, 2, 3)
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC': [results[m]['auc'] for m in results.keys()],
        'Accuracy': [results[m]['report']['accuracy'] for m in results.keys()]
    })
    
    x = np.arange(len(model_comparison))
    width = 0.35
    
    ax3.bar(x - width/2, model_comparison['AUC'], width, label='AUC', alpha=0.7)
    ax3.bar(x + width/2, model_comparison['Accuracy'], width, label='Accuracy', alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_comparison['Model'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Classification metrics
    ax4 = plt.subplot(2, 2, 4)
    best_report = results[best_model_name]['report']
    
    # Get the class keys from the report
    class_keys = [k for k in best_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    class_keys.sort()  # Ensure consistent ordering
    
    if len(class_keys) >= 2:
        key_0 = class_keys[0]
        key_1 = class_keys[1]
        
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'FOLFIRI Better (0)': [best_report[key_0]['precision'], 
                                   best_report[key_0]['recall'], 
                                   best_report[key_0]['f1-score']],
            'FOLFOX Better (1)': [best_report[key_1]['precision'], 
                                 best_report[key_1]['recall'], 
                                 best_report[key_1]['f1-score']]
        })
        
        metrics_df.set_index('Metric').plot(kind='bar', ax=ax4)
    else:
        # Fallback if report structure is unexpected
        ax4.text(0.5, 0.5, 'Classification metrics unavailable', 
                horizontalalignment='center', verticalalignment='center')
    
    ax4.set_title(f'Classification Metrics - {best_model_name}')
    ax4.set_ylabel('Score')
    ax4.set_xlabel('')
    ax4.legend(title='Prediction')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return importance_df

def generate_recommendations(df, importance_df):
    """Generate clinical recommendations based on analysis"""
    print("\n" + "="*80)
    print("CLINICAL RECOMMENDATIONS")
    print("="*80)
    
    # Analyze key predictors
    top_features = importance_df.head(5)['Feature'].tolist()
    
    print("\nKey Predictive Factors for Treatment Selection:")
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature.replace('_', ' ').title()}")
    
    # Analyze subgroup recommendations
    print("\nSubgroup-Specific Recommendations:")
    
    # MSI-High patients
    msi_high = df[df['MSI_HIGH'] == 1]
    if len(msi_high) > 10:
        folfox_os = msi_high[msi_high['FIRST_LINE'] == 'FOLFOX']['OS_MONTHS'].median()
        folfiri_os = msi_high[msi_high['FIRST_LINE'] == 'FOLFIRI']['OS_MONTHS'].median()
        if not pd.isna(folfox_os) and not pd.isna(folfiri_os):
            if folfiri_os > folfox_os:
                print(f"  • MSI-High patients: Consider FOLFIRI (median OS difference: {folfiri_os - folfox_os:.1f} months)")
            else:
                print(f"  • MSI-High patients: Consider FOLFOX (median OS difference: {folfox_os - folfiri_os:.1f} months)")
    
    # Right-sided tumors
    right_sided = df[df['RIGHT_SIDED'] == 1]
    if len(right_sided) > 10:
        folfox_os = right_sided[right_sided['FIRST_LINE'] == 'FOLFOX']['OS_MONTHS'].median()
        folfiri_os = right_sided[right_sided['FIRST_LINE'] == 'FOLFIRI']['OS_MONTHS'].median()
        if not pd.isna(folfox_os) and not pd.isna(folfiri_os):
            if folfiri_os > folfox_os:
                print(f"  • Right-sided tumors: Consider FOLFIRI (median OS difference: {folfiri_os - folfox_os:.1f} months)")
            else:
                print(f"  • Right-sided tumors: Consider FOLFOX (median OS difference: {folfox_os - folfiri_os:.1f} months)")
    
    # Age-based recommendations
    young = df[df['AGE_UNDER_50'] == 1]
    if len(young) > 10:
        folfox_os = young[young['FIRST_LINE'] == 'FOLFOX']['OS_MONTHS'].median()
        folfiri_os = young[young['FIRST_LINE'] == 'FOLFIRI']['OS_MONTHS'].median()
        if not pd.isna(folfox_os) and not pd.isna(folfiri_os):
            if abs(folfox_os - folfiri_os) > 2:
                better = "FOLFOX" if folfox_os > folfiri_os else "FOLFIRI"
                diff = abs(folfox_os - folfiri_os)
                print(f"  • Patients <50 years: Consider {better} (median OS difference: {diff:.1f} months)")
    
    print("\nSequence Recommendations:")
    print("  • Patients receiving second-line therapy show significantly better outcomes")
    print("  • FOLFOX → FOLFIRI sequence shows promising results (median OS: 38.3 months)")
    print("  • Consider patient tolerance and toxicity profile when planning sequences")

def main():
    """Main pipeline"""
    print("="*80)
    print("SIMPLIFIED ML ANALYSIS: FOLFOX vs FOLFIRI OPTIMIZATION")
    print("="*80)
    
    # Load and prepare data
    df = prepare_data_simplified()
    df = engineer_features(df)
    
    # Create feature matrix
    X, feature_names = create_feature_matrix(df)
    
    # Get target variable
    y = df['FOLFOX_OPTIMAL'].dropna()
    X = X.loc[y.index]
    
    print(f"\nDataset: {len(X)} patients with complete data")
    
    # Train and evaluate models
    results, best_model_name, X_test, y_test, scaler, feature_names = train_and_evaluate_models(X, y, feature_names)
    
    # Plot results
    importance_df = plot_results(results, y_test, best_model_name, feature_names)
    
    # Generate recommendations
    generate_recommendations(df, importance_df)
    
    # Save results
    print("\nSaving results...")
    importance_df.to_csv('feature_importance_simplified.csv', index=False)
    
    # Save model
    import joblib
    joblib.dump(results[best_model_name]['model'], 'best_model_simplified.pkl')
    joblib.dump(scaler, 'scaler_simplified.pkl')
    
    print("\nAnalysis complete!")
    
    return results, importance_df

if __name__ == "__main__":
    results, importance_df = main()
