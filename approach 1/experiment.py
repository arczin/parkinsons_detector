import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = BASE_DIR / "models"
CLEAN_FEATURES_DIR = BASE_DIR / "clean_features"

for directory in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

train_path = CLEAN_FEATURES_DIR / "train_features.csv"
test_path = CLEAN_FEATURES_DIR / "test_features.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("-"*80)
print("PARKINSON'S DISEASE DETECTION - COMPREHENSIVE EXPERIMENTAL STUDY")
print("-"*80)
print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train)}")
print(f"  Test samples: {len(test)}")
print(f"  Total samples: {len(train) + len(test)}")
print(f"\nClass Distribution (Training):")
print(train['label'].value_counts())
print(f"  Class Balance: {train['label'].value_counts()[1] / len(train):.1%} PD patients")

FEATURES = [c for c in train.columns if c not in ["label", "dataset"]]
print(f"\nNumber of features: {len(FEATURES)}")
print(f"Features: {', '.join(FEATURES)}")

X_train, y_train = train[FEATURES], train["label"]
X_test, y_test = test[FEATURES], test["label"]

print("\n" + "-"*80)
print("EXPERIMENT 1: FEATURE SCALING METHODS")
print("-"*80)

scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'NoScaling': None
}

scaling_results = []

for scaler_name, scaler in scalers.items():
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', 
              probability=True, random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(svm, X_train_scaled, y_train, cv=cv,
                               scoring=['accuracy', 'roc_auc'], 
                               return_train_score=False)
    
    svm.fit(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    test_auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])
    
    scaling_results.append({
        'Scaler': scaler_name,
        'CV_Accuracy': cv_scores['test_accuracy'].mean(),
        'CV_AUC': cv_scores['test_roc_auc'].mean(),
        'Test_Accuracy': test_acc,
        'Test_AUC': test_auc
    })
    
    print(f"\n{scaler_name}:")
    print(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}")
    print(f"  CV AUC: {cv_scores['test_roc_auc'].mean():.3f} ± {cv_scores['test_roc_auc'].std():.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Test AUC: {test_auc:.3f}")

scaling_df = pd.DataFrame(scaling_results)
scaling_table_path = TABLES_DIR / "01_scaling_comparison.csv"
scaling_df.to_csv(scaling_table_path, index=False)
print(f"\n✓ Table saved: {scaling_table_path.relative_to(BASE_DIR)}")


best_scaler_name = scaling_df.loc[scaling_df['CV_AUC'].idxmax(), 'Scaler']
best_scaler = scalers[best_scaler_name]
print(f"\n→ Best Scaler: {best_scaler_name}")

if best_scaler is not None:
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
else:
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values


print("\n" + "-"*80)
print("EXPERIMENT 2: FEATURE SELECTION METHODS")
print("-"*80)

feature_selection_results = []


for n_features in [5, 10, 15, len(FEATURES)]:
    if n_features > len(FEATURES):
        continue
    
    print(f"\nTesting with {n_features} features:")
    

    if n_features < len(FEATURES):
        selector_anova = SelectKBest(f_classif, k=n_features)
        X_train_fs = selector_anova.fit_transform(X_train_scaled, y_train)
        X_test_fs = selector_anova.transform(X_test_scaled)
        selected_features = [FEATURES[i] for i in selector_anova.get_support(indices=True)]
    else:
        X_train_fs = X_train_scaled
        X_test_fs = X_test_scaled
        selected_features = FEATURES
    

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10,
                                min_samples_leaf=5, class_weight='balanced', 
                                random_state=42, n_jobs=-1)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(rf, X_train_fs, y_train, cv=cv,
                               scoring=['accuracy', 'roc_auc'], 
                               return_train_score=False)
    
    rf.fit(X_train_fs, y_train)
    test_acc = accuracy_score(y_test, rf.predict(X_test_fs))
    test_auc = roc_auc_score(y_test, rf.predict_proba(X_test_fs)[:, 1])
    
    feature_selection_results.append({
        'N_Features': n_features,
        'Method': 'ANOVA F-test' if n_features < len(FEATURES) else 'All Features',
        'CV_Accuracy': cv_scores['test_accuracy'].mean(),
        'CV_AUC': cv_scores['test_roc_auc'].mean(),
        'Test_Accuracy': test_acc,
        'Test_AUC': test_auc,
        'Selected_Features': ', '.join(selected_features[:5]) + ('...' if n_features > 5 else '')
    })
    
    print(f"  Method: {'ANOVA F-test' if n_features < len(FEATURES) else 'All Features'}")
    print(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.3f}")
    print(f"  CV AUC: {cv_scores['test_roc_auc'].mean():.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Test AUC: {test_auc:.3f}")
    if n_features <= 10:
        print(f"  Selected: {', '.join(selected_features)}")

fs_df = pd.DataFrame(feature_selection_results)
feature_selection_table_path = TABLES_DIR / "02_feature_selection.csv"
fs_df.to_csv(feature_selection_table_path, index=False)
print(f"\n✓ Table saved: {feature_selection_table_path.relative_to(BASE_DIR)}")


X_train_final = X_train_scaled
X_test_final = X_test_scaled


print("\n" + "-"*80)
print("EXPERIMENT 3: MACHINE LEARNING ALGORITHMS COMPARISON")
print("-"*80)

models = {
    'Logistic Regression': LogisticRegression(
        penalty='l2', C=1.0, class_weight='balanced', 
        max_iter=1000, random_state=42
    ),
    'Support Vector Machine (RBF)': SVC(
        kernel='rbf', C=1.0, gamma='scale', 
        class_weight='balanced', probability=True, random_state=42
    ),
    'Support Vector Machine (Linear)': SVC(
        kernel='linear', C=1.0, 
        class_weight='balanced', probability=True, random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1
    ),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        min_samples_split=10, min_samples_leaf=5,
        subsample=0.8, random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42
    ),
    'Multi-Layer Perceptron': MLPClassifier(
        hidden_layer_sizes=(50, 25), activation='relu',
        solver='adam', alpha=0.01, max_iter=1000,
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_results = []

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    

    needs_scaling = model_name not in ['Decision Tree', 'Random Forest', 
                                       'Gradient Boosting', 'AdaBoost']
    X_train_use = X_train_scaled if needs_scaling else X_train.values
    X_test_use = X_test_scaled if needs_scaling else X_test.values
    

    cv_results = cross_validate(
        model, X_train_use, y_train, cv=cv,
        scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'],
        return_train_score=True,
        n_jobs=-1
    )
    

    model.fit(X_train_use, y_train)
    

    y_pred = model.predict(X_test_use)
    y_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    model_results.append({
        'Model': model_name,
        'CV_Accuracy_Mean': cv_results['test_accuracy'].mean(),
        'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
        'CV_AUC_Mean': cv_results['test_roc_auc'].mean(),
        'CV_AUC_Std': cv_results['test_roc_auc'].std(),
        'CV_Precision': cv_results['test_precision'].mean(),
        'CV_Recall': cv_results['test_recall'].mean(),
        'CV_F1': cv_results['test_f1'].mean(),
        'Train_Accuracy': cv_results['train_accuracy'].mean(),
        'Overfit_Gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean(),
        'Test_Accuracy': test_acc,
        'Test_Precision': test_prec,
        'Test_Recall': test_rec,
        'Test_F1': test_f1,

        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    })
    

    auc_str = f"{test_auc:.3f}" if test_auc is not None else "N/A"
    print(f"  CV Acc: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"  CV AUC: {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
    print(f"  Test Acc: {test_acc:.3f}, AUC: {auc_str}")
    print(f"  Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
    print(f"  Overfit Gap: {cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean():.3f}")

model_df = pd.DataFrame(model_results)
model_df = model_df.sort_values('CV_AUC_Mean', ascending=False)
model_comparison_table_path = TABLES_DIR / "03_model_comparison.csv"
model_df.to_csv(model_comparison_table_path, index=False)
print(f"\n✓ Table saved: {model_comparison_table_path.relative_to(BASE_DIR)}")


print("\n" + "-"*80)
print("EXPERIMENT 4: HYPERPARAMETER TUNING")
print("-"*80)

top_3_models = model_df.head(3)['Model'].tolist()
tuning_results = []


if 'Random Forest' in top_3_models:
    print("\nTuning Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10]
    }
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(rf_base, param_grid_rf, cv=5, scoring='roc_auc', 
                          n_jobs=-1, verbose=0)
    grid_rf.fit(X_train, y_train)
    
    tuning_results.append({
        'Model': 'Random Forest',
        'Best_Params': str(grid_rf.best_params_),
        'Best_CV_Score': grid_rf.best_score_,
        'Test_Accuracy': accuracy_score(y_test, grid_rf.predict(X_test)),
        'Test_AUC': roc_auc_score(y_test, grid_rf.predict_proba(X_test)[:, 1])
    })
    print(f"  Best params: {grid_rf.best_params_}")
    print(f"  Best CV AUC: {grid_rf.best_score_:.3f}")


if any('SVM' in m for m in top_3_models):
    print("\nTuning SVM (RBF)...")
    param_grid_svm = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    grid_svm = GridSearchCV(svm_base, param_grid_svm, cv=5, scoring='roc_auc', 
                           n_jobs=-1, verbose=0)
    grid_svm.fit(X_train_scaled, y_train)
    
    tuning_results.append({
        'Model': 'SVM (RBF)',
        'Best_Params': str(grid_svm.best_params_),
        'Best_CV_Score': grid_svm.best_score_,
        'Test_Accuracy': accuracy_score(y_test, grid_svm.predict(X_test_scaled)),
        'Test_AUC': roc_auc_score(y_test, grid_svm.predict_proba(X_test_scaled)[:, 1])
    })
    print(f"  Best params: {grid_svm.best_params_}")
    print(f"  Best CV AUC: {grid_svm.best_score_:.3f}")


if 'Logistic Regression' in top_3_models:
    print("\nTuning Logistic Regression...")
    param_grid_lr = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    grid_lr = GridSearchCV(lr_base, param_grid_lr, cv=5, scoring='roc_auc', 
                          n_jobs=-1, verbose=0)
    grid_lr.fit(X_train_scaled, y_train)
    
    tuning_results.append({
        'Model': 'Logistic Regression',
        'Best_Params': str(grid_lr.best_params_),
        'Best_CV_Score': grid_lr.best_score_,
        'Test_Accuracy': accuracy_score(y_test, grid_lr.predict(X_test_scaled)),
        'Test_AUC': roc_auc_score(y_test, grid_lr.predict_proba(X_test_scaled)[:, 1])
    })
    print(f"  Best params: {grid_lr.best_params_}")
    print(f"  Best CV AUC: {grid_lr.best_score_:.3f}")

tuning_df = pd.DataFrame(tuning_results)
tuning_table_path = TABLES_DIR / "04_hyperparameter_tuning.csv"
tuning_df.to_csv(tuning_table_path, index=False)
print(f"\n✓ Table saved: {tuning_table_path.relative_to(BASE_DIR)}")


print("\n" + "-"*80)
print("GENERATING VISUALIZATIONS")
print("-"*80)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

model_df_sorted = model_df.sort_values('Test_AUC', ascending=True)
y_pos = np.arange(len(model_df_sorted))

ax1.barh(y_pos, model_df_sorted['Test_Accuracy'], alpha=0.8, label='Accuracy')
ax1.barh(y_pos, model_df_sorted['Test_AUC'], alpha=0.6, label='AUC')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(model_df_sorted['Model'], fontsize=9)
ax1.set_xlabel('Score', fontsize=10)
ax1.set_title('Test Set Performance Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)


ax2.scatter(model_df['CV_AUC_Mean'], model_df['Test_AUC'], s=100, alpha=0.6)
ax2.plot([0.5, 1.0], [0.5, 1.0], 'r--', alpha=0.5, label='Perfect Agreement')
ax2.set_xlabel('Cross-Validation AUC', fontsize=10)
ax2.set_ylabel('Test Set AUC', fontsize=10)
ax2.set_title('CV vs Test Performance', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

for idx, row in model_df.iterrows():
    ax2.annotate(row['Model'].split()[0], 
                (row['CV_AUC_Mean'], row['Test_AUC']),
                fontsize=7, alpha=0.7)

plt.tight_layout()
model_comparison_fig_path = FIGURES_DIR / "01_model_comparison.png"
plt.savefig(model_comparison_fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {model_comparison_fig_path.relative_to(BASE_DIR)}")
plt.close()


best_model_name = model_df.iloc[0]['Model']
best_model_data = model_df.iloc[0]

fig, ax = plt.subplots(figsize=(6, 5))
cm_array = np.array([[best_model_data['TN'], best_model_data['FP']],
                     [best_model_data['FN'], best_model_data['TP']]])

sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['HC', 'PD'], yticklabels=['HC', 'PD'],
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title(f'Confusion Matrix - {best_model_name}\n' +
             f'Accuracy: {best_model_data["Test_Accuracy"]:.3f}, ' +
             f'AUC: {best_model_data["Test_AUC"]:.3f}',
             fontsize=12, fontweight='bold')

plt.tight_layout()
confusion_matrix_fig_path = FIGURES_DIR / "02_confusion_matrix.png"
plt.savefig(confusion_matrix_fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {confusion_matrix_fig_path.relative_to(BASE_DIR)}")
plt.close()


if 'Random Forest' in models:
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     min_samples_split=10, min_samples_leaf=5,
                                     class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    feat_imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(feat_imp))
    ax.barh(y_pos, feat_imp['importance'], alpha=0.8, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_imp['feature'], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title('Feature Importance - Random Forest', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    feature_importance_fig_path = FIGURES_DIR / "03_feature_importance.png"
    plt.savefig(feature_importance_fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {feature_importance_fig_path.relative_to(BASE_DIR)}")
    plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(model_df))
width = 0.35

bars1 = ax.bar(x - width/2, model_df['Train_Accuracy'], width, 
               label='Training', alpha=0.8, color='lightcoral')
bars2 = ax.bar(x + width/2, model_df['CV_Accuracy_Mean'], width,
               label='Cross-Validation', alpha=0.8, color='skyblue')

ax.set_xlabel('Model', fontsize=10)
ax.set_ylabel('Accuracy', fontsize=10)
ax.set_title('Training vs Cross-Validation Performance\n(Overfitting Analysis)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.split()[0] for m in model_df['Model']], 
                   rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')

plt.tight_layout()
overfitting_fig_path = FIGURES_DIR / "04_overfitting_analysis.png"
plt.savefig(overfitting_fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {overfitting_fig_path.relative_to(BASE_DIR)}")
plt.close()


print("\n" + "-"*80)
print("GENERATING LATEX TABLES")
print("-"*80)


latex_table1 = model_df[['Model', 'CV_Accuracy_Mean', 'CV_AUC_Mean', 
                         'Test_Accuracy', 'Test_AUC', 'Sensitivity', 'Specificity']].copy()
latex_table1.columns = ['Model', 'CV Acc', 'CV AUC', 'Test Acc', 'Test AUC', 'Sens', 'Spec']
latex_table1 = latex_table1.round(3)

latex_table_path = TABLES_DIR / "05_latex_main_results.tex"
with latex_table_path.open('w') as f:
    f.write(latex_table1.to_latex(index=False, escape=False,
                                  caption='Performance comparison of machine learning algorithms',
                                  label='tab:main_results'))
print(f"✓ LaTeX table saved: {latex_table_path.relative_to(BASE_DIR)}")


print("\n" + "-"*80)
print("EXPERIMENTAL SUMMARY REPORT")
print("-"*80)

summary_report = f"""
PARKINSON'S DISEASE DETECTION - EXPERIMENTAL RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
- Total Samples: {len(train) + len(test)}
- Training Samples: {len(train)} (PD: {sum(y_train)}, HC: {len(y_train) - sum(y_train)})
- Test Samples: {len(test)} (PD: {sum(y_test)}, HC: {len(y_test) - sum(y_test)})
- Number of Features: {len(FEATURES)}
- Class Balance: {sum(y_train) / len(y_train):.1%} PD patients

EXPERIMENT 1: FEATURE SCALING
Best Method: {best_scaler_name}
- Cross-Validation AUC: {scaling_df.loc[scaling_df['Scaler'] == best_scaler_name, 'CV_AUC'].values[0]:.3f}

EXPERIMENT 2: FEATURE SELECTION
Best Configuration: {fs_df.loc[fs_df['CV_AUC'].idxmax(), 'N_Features']} features
- Cross-Validation AUC: {fs_df['CV_AUC'].max():.3f}

EXPERIMENT 3: MODEL COMPARISON
Top 3 Models:
1. {model_df.iloc[0]['Model']}
   - CV AUC: {model_df.iloc[0]['CV_AUC_Mean']:.3f} ± {model_df.iloc[0]['CV_AUC_Std']:.3f}
   - Test AUC: {model_df.iloc[0]['Test_AUC']:.3f}
   - Sensitivity: {model_df.iloc[0]['Sensitivity']:.3f}
   - Specificity: {model_df.iloc[0]['Specificity']:.3f}

2. {model_df.iloc[1]['Model']}
   - CV AUC: {model_df.iloc[1]['CV_AUC_Mean']:.3f} ± {model_df.iloc[1]['CV_AUC_Std']:.3f}
   - Test AUC: {model_df.iloc[1]['Test_AUC']:.3f}
   - Sensitivity: {model_df.iloc[1]['Sensitivity']:.3f}
   - Specificity: {model_df.iloc[1]['Specificity']:.3f}

3. {model_df.iloc[2]['Model']}
   - CV AUC: {model_df.iloc[2]['CV_AUC_Mean']:.3f} ± {model_df.iloc[2]['CV_AUC_Std']:.3f}
   - Test AUC: {model_df.iloc[2]['Test_AUC']:.3f}
   - Sensitivity: {model_df.iloc[2]['Sensitivity']:.3f}
   - Specificity: {model_df.iloc[2]['Specificity']:.3f}

BEST OVERALL MODEL: {model_df.iloc[0]['Model']}
Final Test Performance:
- Accuracy: {model_df.iloc[0]['Test_Accuracy']:.3f}
- AUC-ROC: {model_df.iloc[0]['Test_AUC']:.3f}
- Precision: {model_df.iloc[0]['Test_Precision']:.3f}
- Recall (Sensitivity): {model_df.iloc[0]['Sensitivity']:.3f}
- Specificity: {model_df.iloc[0]['Specificity']:.3f}
- F1-Score: {model_df.iloc[0]['Test_F1']:.3f}

CONFUSION MATRIX:
                Predicted
                HC    PD
Actual  HC     {int(model_df.iloc[0]['TN']):3d}   {int(model_df.iloc[0]['FP']):3d}
        PD     {int(model_df.iloc[0]['FN']):3d}   {int(model_df.iloc[0]['TP']):3d}

OVERFITTING ANALYSIS:
Average Train-Val Gap: {model_df['Overfit_Gap'].mean():.3f}
Models with gap < 0.05: {sum(model_df['Overfit_Gap'] < 0.05)}/{len(model_df)}

FILES GENERATED:
- Tables: {TABLES_DIR.relative_to(BASE_DIR)}/
- Figures: {FIGURES_DIR.relative_to(BASE_DIR)}/
- Models: {MODELS_DIR.relative_to(BASE_DIR)}/
"""

report_path = RESULTS_DIR / "EXPERIMENTAL_REPORT.txt"
with report_path.open('w') as f:
    f.write(summary_report)

print(summary_report)
print("-"*80)
print(f"✓ Full report saved: {report_path.relative_to(BASE_DIR)}")
print("-"*80)
