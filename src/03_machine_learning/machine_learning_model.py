#!/usr/bin/env python3
"""
Machine Learning Models for Treatment Sequence Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Prepare features for machine learning"""
    print("Preparing features for machine learning...")
    
    # Load additional data for more features
    sample_data = pd.read_csv('msk_chord_2024/data_clinical_sample.txt', 
                              sep='\t', comment='#', low_memory=False)
    mutation_data = pd.read_csv('msk_chord_2024/data_mutations.txt', 
                               sep='\t', low_memory=False)
    
    # Get colorectal samples
    colorectal_samples = sample_data[
        sample_data['CANCER_TYPE'].str.contains('Colorectal', case=False, na=False)
    ]
    
    # Merge additional features
    feature_cols = ['PATIENT_ID', 'PRIMARY_SITE', 'METASTATIC_SITE', 'TMB_NONSYNONYMOUS']
    df = df.merge(
        colorectal_samples[feature_cols].drop_duplicates('PATIENT_ID'),
        on='PATIENT_ID',
        how='left'
    )
    
    # Extract key mutations for colorectal cancer
    key_genes = ['KRAS', 'BRAF', 'APC', 'TP53', 'PIK3CA', 'SMAD4', 'NRAS']
    patient_mutations = {}
    
    for patient_id in df['PATIENT_ID'].unique():
        patient_mut = mutation_data[mutation_data['Tumor_Sample_Barcode'].str.contains(patient_id, na=False)]
        mutations = {}
        for gene in key_genes:
            mutations[f'{gene}_mutated'] = int(gene in patient_mut['Hugo_Symbol'].values)
        patient_mutations[patient_id] = mutations
    
    mutation_df = pd.DataFrame.from_dict(patient_mutations, orient='index').reset_index()
    mutation_df.columns = ['PATIENT_ID'] + list(mutation_df.columns[1:])
    
    df = df.merge(mutation_df, on='PATIENT_ID', how='left')
    
    # Create binary outcome: FOLFOX better (1) vs FOLFIRI better (0)
    df['FOLFOX_BETTER'] = np.nan
    
    # Define based on survival outcomes
    df.loc[df['FIRST_LINE'] == 'FOLFOX', 'FOLFOX_BETTER'] = (df['OS_MONTHS'] > df['OS_MONTHS'].median()).astype(int)
    df.loc[df['FIRST_LINE'] == 'FOLFIRI', 'FOLFOX_BETTER'] = (df['OS_MONTHS'] <= df['OS_MONTHS'].median()).astype(int)
    
    return df

def create_feature_matrix(df):
    """Create feature matrix for modeling"""
    print("Creating feature matrix...")
    
    # Select features
    feature_cols = [
        'CURRENT_AGE_DEID', 'TMB_NONSYNONYMOUS',
        'KRAS_mutated', 'BRAF_mutated', 'APC_mutated', 
        'TP53_mutated', 'PIK3CA_mutated', 'SMAD4_mutated', 'NRAS_mutated'
    ]
    
    # Add binary features
    df['MALE'] = (df['GENDER'] == 'Male').astype(int)
    df['MSI_HIGH'] = (df['MSI_TYPE'] == 'Instable').astype(int)
    df['LIVER_METS'] = df['METASTATIC_SITE'].str.contains('Liver', case=False, na=False).astype(int)
    df['LUNG_METS'] = df['METASTATIC_SITE'].str.contains('Lung', case=False, na=False).astype(int)
    df['PERITONEUM_METS'] = df['METASTATIC_SITE'].str.contains('Peritoneum', case=False, na=False).astype(int)
    df['RIGHT_SIDED'] = df['PRIMARY_SITE'].str.contains('Ascending|Cecum', case=False, na=False).astype(int)
    
    feature_cols.extend(['MALE', 'MSI_HIGH', 'LIVER_METS', 'LUNG_METS', 
                        'PERITONEUM_METS', 'RIGHT_SIDED'])
    
    # Handle missing values
    X = df[feature_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, feature_cols

def train_models(X, y, feature_names):
    """Train multiple machine learning models"""
    print("\n" + "="*80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc': auc_score,
            'cv_scores': cv_scores,
            'report': classification_report(y_test, y_pred)
        }
        
        print(f"  AUC Score: {auc_score:.3f}")
        print(f"  CV AUC (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = name
    
    print(f"\nBest Model: {best_model} (AUC: {best_auc:.3f})")
    
    return results, best_model, X_train, X_test, y_train, y_test, scaler, feature_names

def plot_feature_importance(results, best_model, feature_names):
    """Plot feature importance for the best model"""
    print("\nAnalyzing feature importance...")
    
    model = results[best_model]['model']
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), y='Feature', x='Importance', ax=ax)
    ax.set_title(f'Top 15 Feature Importances - {best_model}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return importance_df

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc = result['auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Treatment Response Prediction')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def predict_optimal_treatment(model, scaler, patient_features, feature_names):
    """Predict optimal treatment for a patient"""
    
    # Ensure features are in correct order
    patient_df = pd.DataFrame([patient_features], columns=feature_names)
    
    # Scale if necessary
    if isinstance(model, LogisticRegression):
        patient_scaled = scaler.transform(patient_df)
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
    else:
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0]
    
    if prediction == 1:
        return "FOLFOX", probability[1]
    else:
        return "FOLFIRI", probability[0]

def main():
    """Main machine learning pipeline"""
    print("="*80)
    print("MACHINE LEARNING ANALYSIS: TREATMENT OPTIMIZATION")
    print("="*80)
    
    # Load preprocessed data
    df = pd.read_csv('analysis_cohort.csv')
    
    # Prepare features
    df = prepare_features(df)
    
    # Create feature matrix
    X, feature_names = create_feature_matrix(df)
    
    # Create target variable
    y = df['FOLFOX_BETTER'].dropna()
    X = X.loc[y.index]
    
    print(f"\nDataset size: {len(X)} patients")
    print(f"FOLFOX better: {y.sum()} ({y.mean():.1%})")
    print(f"FOLFIRI better: {(1-y).sum()} ({(1-y).mean():.1%})")
    
    # Train models
    results, best_model, X_train, X_test, y_train, y_test, scaler, feature_names = train_models(X, y, feature_names)
    
    # Plot feature importance
    importance_df = plot_feature_importance(results, best_model, feature_names)
    
    # Plot ROC curves
    plot_roc_curves(results, y_test)
    
    # Save the best model
    print("\nSaving best model and results...")
    import joblib
    joblib.dump(results[best_model]['model'], 'best_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # Save feature importance
    if importance_df is not None:
        importance_df.to_csv('feature_importance.csv', index=False)
    
    # Example prediction
    print("\n" + "="*80)
    print("EXAMPLE PATIENT PREDICTION")
    print("="*80)
    
    example_patient = {
        'CURRENT_AGE_DEID': 65,
        'TMB_NONSYNONYMOUS': 5.5,
        'KRAS_mutated': 1,
        'BRAF_mutated': 0,
        'APC_mutated': 1,
        'TP53_mutated': 1,
        'PIK3CA_mutated': 0,
        'SMAD4_mutated': 0,
        'NRAS_mutated': 0,
        'MALE': 1,
        'MSI_HIGH': 0,
        'LIVER_METS': 1,
        'LUNG_METS': 0,
        'PERITONEUM_METS': 0,
        'RIGHT_SIDED': 0
    }
    
    optimal_treatment, confidence = predict_optimal_treatment(
        results[best_model]['model'], scaler, example_patient, feature_names
    )
    
    print("Patient characteristics:")
    print("  Age: 65, Male")
    print("  KRAS mutated, APC mutated, TP53 mutated")
    print("  MSI-stable, Liver metastases")
    print(f"\nRecommended first-line treatment: {optimal_treatment}")
    print(f"Confidence: {confidence:.1%}")
    
    return results, best_model, importance_df

if __name__ == "__main__":
    results, best_model, importance_df = main()
