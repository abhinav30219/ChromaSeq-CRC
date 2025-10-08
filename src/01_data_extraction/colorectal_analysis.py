#!/usr/bin/env python3
"""
Colorectal Cancer Treatment Sequence Analysis
Investigating optimal FOLFOX vs FOLFIRI sequencing for Stage 4 patients
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants for treatment regimen identification
FOLFOX_DRUGS = {'FLUOROURACIL', 'LEUCOVORIN', 'OXALIPLATIN'}
FOLFIRI_DRUGS = {'FLUOROURACIL', 'LEUCOVORIN', 'IRINOTECAN'}

def load_data():
    """Load all necessary data files"""
    print("Loading MSK CHORD dataset...")
    
    # Load clinical data
    patient_data = pd.read_csv('msk_chord_2024/data_clinical_patient.txt', 
                               sep='\t', comment='#', low_memory=False)
    sample_data = pd.read_csv('msk_chord_2024/data_clinical_sample.txt', 
                              sep='\t', comment='#', low_memory=False)
    
    # Load treatment timeline
    treatment_data = pd.read_csv('msk_chord_2024/data_timeline_treatment.txt', 
                                 sep='\t', low_memory=False)
    
    # Load additional timeline data
    progression_data = pd.read_csv('msk_chord_2024/data_timeline_progression.txt', 
                                  sep='\t', low_memory=False)
    diagnosis_data = pd.read_csv('msk_chord_2024/data_timeline_diagnosis.txt', 
                                sep='\t', low_memory=False)
    
    print(f"Loaded patient data: {len(patient_data)} patients")
    print(f"Loaded sample data: {len(sample_data)} samples")
    print(f"Loaded treatment data: {len(treatment_data)} treatment records")
    
    return {
        'patient': patient_data,
        'sample': sample_data,
        'treatment': treatment_data,
        'progression': progression_data,
        'diagnosis': diagnosis_data
    }

def identify_colorectal_patients(data):
    """Extract colorectal cancer patients from the dataset"""
    print("\nIdentifying colorectal cancer patients...")
    
    # Get colorectal samples
    colorectal_samples = data['sample'][
        data['sample']['CANCER_TYPE'].str.contains('Colorectal', case=False, na=False)
    ]
    
    # Get unique patient IDs
    colorectal_patient_ids = colorectal_samples['PATIENT_ID'].unique()
    
    # Filter patient data
    colorectal_patients = data['patient'][
        data['patient']['PATIENT_ID'].isin(colorectal_patient_ids)
    ]
    
    print(f"Found {len(colorectal_patient_ids)} colorectal cancer patients")
    
    return colorectal_patients, colorectal_samples

def filter_stage4_patients(colorectal_patients):
    """Filter for Stage 4 patients only"""
    print("\nFiltering for Stage 4 patients...")
    
    # Stage information is in STAGE_HIGHEST_RECORDED column
    stage4_patients = colorectal_patients[
        colorectal_patients['STAGE_HIGHEST_RECORDED'] == 'Stage 4'
    ]
    
    print(f"Found {len(stage4_patients)} Stage 4 colorectal cancer patients")
    
    return stage4_patients

def identify_treatment_regimens(patient_ids, treatment_data):
    """Identify FOLFOX and FOLFIRI regimens from individual drug records"""
    print("\nIdentifying treatment regimens...")
    
    # Filter treatments for our patient cohort
    cohort_treatments = treatment_data[
        treatment_data['PATIENT_ID'].isin(patient_ids)
    ]
    
    # Convert drug names to uppercase for matching
    cohort_treatments['AGENT_UPPER'] = cohort_treatments['AGENT'].str.upper()
    
    regimen_data = []
    
    for patient_id in patient_ids:
        patient_treatments = cohort_treatments[
            cohort_treatments['PATIENT_ID'] == patient_id
        ].copy()
        
        if len(patient_treatments) == 0:
            continue
        
        # Sort by start date
        patient_treatments = patient_treatments.sort_values('START_DATE')
        
        # Group treatments by overlapping time windows (within 30 days)
        treatment_groups = []
        current_group = []
        last_date = None
        
        for _, treatment in patient_treatments.iterrows():
            if last_date is None or treatment['START_DATE'] - last_date <= 30:
                current_group.append(treatment)
                last_date = treatment['START_DATE']
            else:
                if current_group:
                    treatment_groups.append(pd.DataFrame(current_group))
                current_group = [treatment]
                last_date = treatment['START_DATE']
        
        if current_group:
            treatment_groups.append(pd.DataFrame(current_group))
        
        # Identify regimens in each group
        for group_idx, group in enumerate(treatment_groups):
            drugs_in_group = set(group['AGENT_UPPER'].str.strip())
            
            # Check for FOLFOX
            if len(drugs_in_group.intersection(FOLFOX_DRUGS)) >= 2 and 'OXALIPLATIN' in drugs_in_group:
                regimen_data.append({
                    'PATIENT_ID': patient_id,
                    'REGIMEN': 'FOLFOX',
                    'START_DATE': group['START_DATE'].min(),
                    'STOP_DATE': group['STOP_DATE'].max(),
                    'SEQUENCE_NUMBER': group_idx + 1,
                    'DRUGS': ', '.join(drugs_in_group.intersection(FOLFOX_DRUGS))
                })
            
            # Check for FOLFIRI
            elif len(drugs_in_group.intersection(FOLFIRI_DRUGS)) >= 2 and 'IRINOTECAN' in drugs_in_group:
                regimen_data.append({
                    'PATIENT_ID': patient_id,
                    'REGIMEN': 'FOLFIRI',
                    'START_DATE': group['START_DATE'].min(),
                    'STOP_DATE': group['STOP_DATE'].max(),
                    'SEQUENCE_NUMBER': group_idx + 1,
                    'DRUGS': ', '.join(drugs_in_group.intersection(FOLFIRI_DRUGS))
                })
    
    regimen_df = pd.DataFrame(regimen_data)
    
    if len(regimen_df) > 0:
        print(f"Identified {len(regimen_df)} FOLFOX/FOLFIRI treatment periods")
        print(f"FOLFOX treatments: {(regimen_df['REGIMEN'] == 'FOLFOX').sum()}")
        print(f"FOLFIRI treatments: {(regimen_df['REGIMEN'] == 'FOLFIRI').sum()}")
    else:
        print("No FOLFOX/FOLFIRI regimens identified")
    
    return regimen_df

def determine_treatment_sequences(regimen_df):
    """Determine treatment sequences for each patient"""
    print("\nDetermining treatment sequences...")
    
    if len(regimen_df) == 0:
        return pd.DataFrame()
    
    sequences = []
    
    for patient_id in regimen_df['PATIENT_ID'].unique():
        patient_regimens = regimen_df[
            regimen_df['PATIENT_ID'] == patient_id
        ].sort_values('START_DATE')
        
        # Get first-line and second-line treatments
        treatments = patient_regimens['REGIMEN'].tolist()
        
        if len(treatments) == 1:
            sequence = f"{treatments[0]}_only"
        elif len(treatments) >= 2:
            sequence = f"{treatments[0]}_to_{treatments[1]}"
        else:
            continue
        
        sequences.append({
            'PATIENT_ID': patient_id,
            'TREATMENT_SEQUENCE': sequence,
            'FIRST_LINE': treatments[0] if treatments else None,
            'SECOND_LINE': treatments[1] if len(treatments) > 1 else None,
            'NUM_LINES': len(treatments),
            'FIRST_LINE_START': patient_regimens.iloc[0]['START_DATE'],
            'FIRST_LINE_STOP': patient_regimens.iloc[0]['STOP_DATE']
        })
    
    sequence_df = pd.DataFrame(sequences)
    
    if len(sequence_df) > 0:
        print(f"Treatment sequences identified for {len(sequence_df)} patients")
        print("\nSequence distribution:")
        print(sequence_df['TREATMENT_SEQUENCE'].value_counts())
    
    return sequence_df

def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("COLORECTAL CANCER TREATMENT SEQUENCE ANALYSIS")
    print("Investigating optimal FOLFOX vs FOLFIRI sequencing")
    print("=" * 80)
    
    # Load data
    data = load_data()
    
    # Identify colorectal cancer patients
    colorectal_patients, colorectal_samples = identify_colorectal_patients(data)
    
    # Filter for Stage 4 patients
    stage4_patients = filter_stage4_patients(colorectal_patients)
    
    # Get patient IDs
    stage4_patient_ids = stage4_patients['PATIENT_ID'].tolist()
    
    # Identify treatment regimens
    regimen_df = identify_treatment_regimens(stage4_patient_ids, data['treatment'])
    
    # Determine treatment sequences
    sequence_df = determine_treatment_sequences(regimen_df)
    
    # Merge with patient data for analysis
    if len(sequence_df) > 0:
        # Check available columns and merge appropriately
        available_cols = ['PATIENT_ID', 'OS_MONTHS', 'OS_STATUS', 'GENDER', 'CURRENT_AGE_DEID']
        
        # Check for MSI-related columns
        for col in stage4_patients.columns:
            if 'MSI' in col.upper():
                available_cols.append(col)
                break
        
        analysis_df = sequence_df.merge(
            stage4_patients[available_cols],
            on='PATIENT_ID',
            how='left'
        )
        
        # Save intermediate results
        print("\nSaving intermediate results...")
        regimen_df.to_csv('identified_regimens.csv', index=False)
        sequence_df.to_csv('treatment_sequences.csv', index=False)
        analysis_df.to_csv('analysis_cohort.csv', index=False)
        
        print(f"Analysis cohort saved: {len(analysis_df)} patients")
        
        # Basic statistics
        print("\n" + "=" * 80)
        print("PRELIMINARY STATISTICS")
        print("=" * 80)
        
        print("\nTreatment sequence distribution:")
        print(analysis_df['TREATMENT_SEQUENCE'].value_counts())
        
        print("\nMedian overall survival by first-line treatment:")
        for treatment in ['FOLFOX', 'FOLFIRI']:
            subset = analysis_df[analysis_df['FIRST_LINE'] == treatment]
            if len(subset) > 0:
                median_os = subset['OS_MONTHS'].median()
                print(f"  {treatment}: {median_os:.1f} months (n={len(subset)})")
    else:
        print("\nNo patients with identifiable FOLFOX/FOLFIRI treatments found")
        print("This may be due to:")
        print("  1. Different drug naming conventions in the dataset")
        print("  2. Limited Stage 4 patients receiving these regimens")
        print("  3. Need to adjust the treatment identification algorithm")
    
    return data, stage4_patients, regimen_df, sequence_df

if __name__ == "__main__":
    data, patients, regimens, sequences = main()
