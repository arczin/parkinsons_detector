
"""
Uses Oxford and Istanbul University datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "clean_features"

INPUT_FILES = [
    DATASETS_DIR / "features" / "istanul-university.csv",
    DATASETS_DIR / "features" / "parkinsons.data",
]

CORE_FEATURES_CANONICAL = [
    "mdvp_fo_hz", "mdvp_fhi_hz", "mdvp_flo_hz",
    "mdvp_jitter_percent", "mdvp_jitter_abs", "mdvp_rap", "mdvp_ppq", "jitter_ddp",
    "mdvp_shimmer", "mdvp_shimmer_db", "shimmer_apq3", "shimmer_apq5", "mdvp_apq", "shimmer_dda",
    "nhr", "hnr", "rpde", "dfa", "spread1", "spread2", "d2", "ppe"
]


ISTANBUL_TO_OXFORD_MAPPING = {

    'ppe': 'ppe',
    'dfa': 'dfa',
    'rpde': 'rpde',
    'spread1': 'spread1',
    'spread2': 'spread2',
    'd2': 'd2',
    

    'locpctjitter': 'mdvp_jitter_percent',
    'locabsjitter': 'mdvp_jitter_abs',
    'rapjitter': 'mdvp_rap',
    'ppq5jitter': 'mdvp_ppq',
    'ddpjitter': 'jitter_ddp',
    

    'locshimmer': 'mdvp_shimmer',
    'locdbshimmer': 'mdvp_shimmer_db',
    'apq3shimmer': 'shimmer_apq3',
    'apq5shimmer': 'shimmer_apq5',
    'apq11shimmer': 'mdvp_apq',
    'ddashimmer': 'shimmer_dda',
    

    'nhr': 'nhr',
    'hnr': 'hnr',
    

    'meanfo': 'mdvp_fo_hz',
    'maxfo': 'mdvp_fhi_hz',
    'minfo': 'mdvp_flo_hz',
}



def _detect_sep(first_line: str) -> str:
    """Detect CSV separator (tab or comma)"""
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores"""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(":", "_", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Find the label/target column"""
    for cand in ["label", "class", "status", "target", "y"]:
        if cand in df.columns:
            return cand
    return None


def standardize_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Convert label column to binary 0/1 format"""
    df = df.copy()
    vals = df[label_col]
    
    if vals.dtype == object:
        mapped = (
            vals.astype(str).str.strip().str.lower()
            .replace({
                "pd": 1, "parkinson": 1, "parkinsons": 1, "p": 1,
                "1": 1, "1.0": 1, "true": 1, "t": 1,
                "hc": 0, "healthy": 0, "h": 0, "control": 0,
                "0": 0, "0.0": 0, "false": 0, "f": 0
            })
        )
        df[label_col] = mapped
    
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    
    if label_col != 'label':
        df.rename(columns={label_col: "label"}, inplace=True)
    
    return df



def load_istanbul_dataset(filepath: str) -> pd.DataFrame:
    """Load Istanbul University Dataset with multi-level headers"""
    print(f"[LOAD] {filepath}")
    print(f"       Detected Istanbul University dataset")
    

    df = pd.read_csv(filepath, header=[0, 1, 2])
    print(f"       Initial shape: {df.shape}")
    

    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):

            feature_name = col[1] if len(col) > 1 else col[0]
            new_columns.append(feature_name)
        else:
            new_columns.append(col)
    
    df.columns = new_columns
    

    df = clean_column_names(df)
    

    df['subject_id'] = 'istanbul_' + df['id'].astype(str)
    df['dataset'] = 'istanbul'
    

    label_col = find_label_column(df)
    if label_col:
        df = standardize_label(df, label_col)
    

    exclude_cols = ['id', 'gender', 'subject_id', 'dataset', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"       Final shape: {df.shape}")
    print(f"       Features: {len(feature_cols)}")
    print(f"       Subjects: {df['subject_id'].nunique()}")
    print(f"       Samples: PD={df['label'].sum()}, HC={len(df)-df['label'].sum()}")
    
    return df


def load_oxford_dataset(filepath: str) -> pd.DataFrame:
    """Load Oxford Parkinson's Dataset"""
    print(f"[LOAD] {filepath}")
    print(f"       Detected Oxford Parkinson's dataset")
    

    with open(filepath, "r", errors="ignore") as f:
        first_line = f.readline()
    sep = _detect_sep(first_line)
    

    df = pd.read_csv(filepath, sep=sep, engine="python")
    sep_name = "TAB" if sep == "\t" else "COMMA"
    print(f"       Initial shape: {df.shape}, separator: {sep_name}")
    

    df = clean_column_names(df)
    

    if 'name' in df.columns:
        df['subject_id'] = df['name'].str.extract(r'(phon_\w+_\w+)', expand=False)
        if df['subject_id'].isna().any():
            df['subject_id'] = 'oxford_' + df.index.astype(str)
    else:
        df['subject_id'] = 'oxford_' + df.index.astype(str)
    
    df['dataset'] = 'oxford'
    

    label_col = find_label_column(df)
    if label_col:
        df = standardize_label(df, label_col)
    

    exclude_cols = ['name', 'subject_id', 'dataset', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"       Final shape: {df.shape}")
    print(f"       Features: {len(feature_cols)}")
    print(f"       Subjects: {df['subject_id'].nunique()}")
    print(f"       Samples: PD={df['label'].sum()}, HC={len(df)-df['label'].sum()}")
    
    return df


def load_dataset(filepath: str) -> pd.DataFrame:
    """Auto-detect and load appropriate dataset"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    filename = filepath.name.lower()
    

    if 'istanbul' in filename or 'istanul' in filename or 'university' in filename:
        return load_istanbul_dataset(filepath)
    else:
        return load_oxford_dataset(filepath)



def align_features_to_canonical(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Align dataset features to canonical Oxford naming scheme
    Returns: (aligned_df, list_of_canonical_feature_names)
    """
    print(f"\n[ALIGN] Aligning features for {dataset_name}")
    

    exclude_cols = ['subject_id', 'dataset', 'label']
    current_features = [col for col in df.columns if col not in exclude_cols]
    

    aligned_df = df[exclude_cols].copy()
    canonical_features = []
    
    if 'istanbul' in dataset_name.lower():

        mapped_count = 0
        direct_count = 0
        
        for istanbul_feat, oxford_feat in ISTANBUL_TO_OXFORD_MAPPING.items():
            if istanbul_feat in current_features:
                aligned_df[oxford_feat] = df[istanbul_feat]
                canonical_features.append(oxford_feat)
                if istanbul_feat == oxford_feat:
                    direct_count += 1
                else:
                    mapped_count += 1
        
        print(f"       ✓ Direct matches: {direct_count}")
        print(f"       ✓ Mapped features: {mapped_count}")
        print(f"       ✓ Total aligned: {len(canonical_features)}")
        
    else:

        for feat in current_features:
            if feat in CORE_FEATURES_CANONICAL:
                aligned_df[feat] = df[feat]
                canonical_features.append(feat)
        
        print(f"       ✓ Canonical features: {len(canonical_features)}")
    
    return aligned_df, canonical_features


def merge_datasets(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """Merge multiple datasets on common features"""
    print("\n" + "="*80)
    print("DATASET MERGING")
    print("="*80)
    
    if len(dfs) == 1:
        print(f"\n[INFO] Single dataset: {dfs[0]['dataset'].iloc[0]}")
        exclude_cols = ['subject_id', 'dataset', 'label']
        features = [col for col in dfs[0].columns if col not in exclude_cols]
        return dfs[0], features
    

    feature_sets = []
    dataset_names = []
    
    for df in dfs:
        dataset_name = df['dataset'].iloc[0]
        dataset_names.append(dataset_name)
        exclude_cols = ['subject_id', 'dataset', 'label']
        features = [col for col in df.columns if col not in exclude_cols]
        feature_sets.append(set(features))
        print(f"  {dataset_name}: {len(features)} features")
    

    common_features = set.intersection(*feature_sets)
    print(f"\n✓ Common features across all datasets: {len(common_features)}")
    
    if len(common_features) == 0:
        print("\n[ERROR] No common features found!")
        print("This shouldn't happen with proper feature alignment.")
        raise ValueError("Feature alignment failed - no common features")
    

    merged_dfs = []
    for df in dfs:
        keep_cols = ['subject_id', 'dataset', 'label'] + sorted(list(common_features))
        merged_dfs.append(df[keep_cols])
    
    merged = pd.concat(merged_dfs, axis=0, ignore_index=True)
    
    print(f"\n✓ Merged dataset:")
    print(f"  Total samples: {len(merged)}")
    print(f"  Total features: {len(common_features)}")
    print(f"  PD samples: {merged['label'].sum()}")
    print(f"  HC samples: {len(merged) - merged['label'].sum()}")
    
    for dataset_name in dataset_names:
        count = (merged['dataset'] == dataset_name).sum()
        print(f"  {dataset_name}: {count} samples")
    

    feature_cols = sorted(list(common_features))
    missing_before = merged[feature_cols].isnull().sum().sum()
    
    if missing_before > 0:
        print(f"\n⚠ Missing values: {missing_before}")
        print(f"  Filling with class-wise median...")
        
        for feat in feature_cols:
            merged[feat] = merged.groupby('label')[feat].transform(
                lambda x: x.fillna(x.median())
            )
        
        missing_after = merged[feature_cols].isnull().sum().sum()
        print(f"  ✓ After imputation: {missing_after}")
    
    return merged, feature_cols



def subject_level_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by subjects to prevent data leakage"""
    print("\n" + "="*80)
    print("SUBJECT-LEVEL TRAIN-TEST SPLIT")
    print("="*80)
    

    subject_labels = df.groupby('subject_id')['label'].first()
    unique_subjects = subject_labels.index.tolist()
    
    pd_subjects = (subject_labels == 1).sum()
    hc_subjects = (subject_labels == 0).sum()
    
    print(f"\nSubject statistics:")
    print(f"  Total subjects: {len(unique_subjects)}")
    print(f"  PD subjects: {pd_subjects} ({pd_subjects/len(unique_subjects)*100:.1f}%)")
    print(f"  HC subjects: {hc_subjects} ({hc_subjects/len(unique_subjects)*100:.1f}%)")
    

    min_class_size = min(pd_subjects, hc_subjects)
    
    if min_class_size < 5:
        print(f"\n[WARN] Small minority class ({min_class_size}). Using random split.")
        train_subjects, test_subjects = train_test_split(
            unique_subjects, test_size=test_size, random_state=random_state
        )
    else:

        train_subjects, test_subjects = train_test_split(
            unique_subjects, test_size=test_size, random_state=random_state,
            stratify=subject_labels
        )
    

    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    test_df = df[df['subject_id'].isin(test_subjects)].copy()
    

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    

    train_subjects_set = set(train_df['subject_id'])
    test_subjects_set = set(test_df['subject_id'])
    overlap = train_subjects_set.intersection(test_subjects_set)
    
    if len(overlap) > 0:
        raise ValueError(f"ERROR: {len(overlap)} subjects appear in both train and test!")
    
    print(f"\n✓ Split completed (80/20):")
    print(f"  Train: {len(train_subjects)} subjects, {len(train_df)} samples")
    print(f"    PD: {(train_df['label']==1).sum()}, HC: {(train_df['label']==0).sum()}")
    print(f"  Test:  {len(test_subjects)} subjects, {len(test_df)} samples")
    print(f"    PD: {(test_df['label']==1).sum()}, HC: {(test_df['label']==0).sum()}")
    print(f"  ✓ No subject overlap!")
    
    return train_df, test_df



def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize features using StandardScaler fitted on training data"""
    print("\n" + "="*80)
    print("FEATURE NORMALIZATION")
    print("="*80)
    
    scaler = StandardScaler()
    

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    print(f"✓ Fitted StandardScaler on training data")
    

    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print(f"✓ Transformed test data")
    
    print(f"\nTraining set statistics:")
    print(f"  Mean: {train_df[feature_cols].mean().mean():.6f}")
    print(f"  Std:  {train_df[feature_cols].std().mean():.6f}")
    
    return train_df, test_df



def main():
    """Main preprocessing pipeline"""
    print("="*80)
    print("PARKINSON'S DISEASE DATASET PREPROCESSING")
    print("="*80)
    

    print("\n[STEP 1] Loading datasets...")
    dfs = []
    for filepath in INPUT_FILES:
        try:
            df = load_dataset(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
            continue
    
    if not dfs:
        raise SystemExit("ERROR: No datasets loaded successfully!")
    

    print("\n[STEP 2] Aligning features to canonical naming...")
    aligned_dfs = []
    for df in dfs:
        dataset_name = df['dataset'].iloc[0]
        aligned_df, canonical_features = align_features_to_canonical(df, dataset_name)
        aligned_dfs.append(aligned_df)
        print(f"       {dataset_name}: {len(canonical_features)} features aligned")
    

    print("\n[STEP 3] Merging datasets...")
    merged_df, feature_cols = merge_datasets(aligned_dfs)
    

    print("\n[STEP 4] Splitting data...")
    train_df, test_df = subject_level_split(merged_df)
    

    print("\n[STEP 5] Normalizing features...")
    train_df, test_df = normalize_features(train_df, test_df, feature_cols)
    

    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    

    output_cols = feature_cols + ['label']
    
    train_path = OUTPUT_DIR / "train_features.csv"
    test_path = OUTPUT_DIR / "test_features.csv"
    
    train_df[output_cols].to_csv(train_path, index=False)
    test_df[output_cols].to_csv(test_path, index=False)
    
    print(f"✓ {train_path}")
    print(f"  Shape: {train_df[output_cols].shape}")
    print(f"  PD: {(train_df['label']==1).sum()}, HC: {(train_df['label']==0).sum()}")
    
    print(f"\n✓ {test_path}")
    print(f"  Shape: {test_df[output_cols].shape}")
    print(f"  PD: {(test_df['label']==1).sum()}, HC: {(test_df['label']==0).sum()}")
    

    feature_names_path = OUTPUT_DIR / "feature_names.txt"
    with feature_names_path.open('w') as f:
        f.write('\n'.join(feature_cols))
    print(f"\n✓ {feature_names_path}")
    print(f"  Features: {len(feature_cols)}")
    

    summary_path = OUTPUT_DIR / "preprocessing_summary.txt"
    with summary_path.open('w') as f:
        f.write("PARKINSON'S DISEASE PREPROCESSING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Features: {len(feature_cols)}\n\n")
        f.write("Training Set:\n")
        f.write(f"  Samples: {len(train_df)}\n")
        f.write(f"  PD: {(train_df['label']==1).sum()}\n")
        f.write(f"  HC: {(train_df['label']==0).sum()}\n")
        f.write(f"  Subjects: {train_df['subject_id'].nunique()}\n\n")
        f.write("Test Set:\n")
        f.write(f"  Samples: {len(test_df)}\n")
        f.write(f"  PD: {(test_df['label']==1).sum()}\n")
        f.write(f"  HC: {(test_df['label']==0).sum()}\n")
        f.write(f"  Subjects: {test_df['subject_id'].nunique()}\n\n")
        f.write("Features:\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"  {i:2d}. {feat}\n")
    
    print(f"✓ {summary_path}")
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  • train_features.csv ({len(train_df)} samples)")
    print(f"  • test_features.csv ({len(test_df)} samples)")
    print(f"  • feature_names.txt ({len(feature_cols)} features)")
    print(f"  • preprocessing_summary.txt")
    print("\n🚀 Ready for machine learning!")


if __name__ == "__main__":
    main()
