import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import os

# Step 1: 데이터 로드 및 분할
print("Step 1: 데이터 로드 및 분할 시작")
data = pd.read_csv("train.csv")
print("  [데이터 로드 완료] train.csv 파일 성공적으로 로드됨")

X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 범주형 변수 리스트 및 NaN 처리
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print("🔍 [CATBoost] cat_features 설정:", cat_features)
X[cat_features] = X[cat_features].fillna("missing")

# 데이터 분할 (클래스 불균형 대비)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

# 클래스 불균형 확인
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 scale_pos_weight: {default_scale_pos_weight:.2f}\n")

# Optuna 목적 함수
def objective(trial):
    print(f">> [CATBoost] Trial {trial.number} 시작")
    param = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "iterations": trial.suggest_int("iterations", 200, 800, step=50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "auto_class_weights": "Balanced",  # 자동 클래스 불균형 처리
        "verbose": False
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}")
    
    model = CatBoostClassifier(cat_features=cat_features, **param, eval_metric="AUC")
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

# Optuna 최적화
print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)  # Pruner 유지
)
study.optimize(objective, n_trials=100)

# 최적 결과 출력
print(">> [CATBoost] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

# 최적 파라미터로 모델 재학습
best_model = CatBoostClassifier(
    cat_features=cat_features,
    eval_metric="AUC",
    **study.best_params
)
best_model.fit(X_train, y_train)

from sklearn.model_selection import StratifiedKFold

def evaluate_ensemble_roc_kfold(X, y, model_params, cat_features, n_splits=5):
    """K-Fold 기반 앙상블 모델 평가"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))  # Out-of-Fold 예측값 저장용
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"▶ Fold {fold + 1}/{n_splits} 훈련 시작")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # 모델 생성 및 학습
        model = CatBoostClassifier(
            cat_features=cat_features,
            eval_metric="AUC",
            **model_params
        )
        model.fit(X_train_fold, y_train_fold, verbose=False)

        # 예측 수행
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = y_pred

        # Fold별 ROC-AUC 계산
        fold_auc = roc_auc_score(y_val_fold, y_pred)
        auc_scores.append(fold_auc)
        print(f"  ▶ Fold {fold + 1} ROC-AUC: {fold_auc:.4f}")

    # 전체 데이터셋에 대한 최종 ROC-AUC 계산
    final_auc = roc_auc_score(y, oof_preds)
    print(f"\n🔥 K-Fold 기반 앙상블 모델 최종 ROC-AUC: {final_auc:.4f}")
    print(f"🔥 개별 Fold 평균 ROC-AUC: {np.mean(auc_scores):.4f}")
    print(f"🔥 Fold ROC-AUC 표준편차: {np.std(auc_scores):.4f}")
    
    return final_auc, oof_preds, auc_scores

# 앙상블 ROC-AUC 계산 및 출력
print("\n>> [CATBoost] K-Fold 기반 앙상블 성능 평가")
kfold_auc, oof_predictions, fold_scores = evaluate_ensemble_roc_kfold(
    X, y, study.best_params, cat_features, n_splits=5
)

print("\n=== 최종 성능 비교 ===")
print(f"단일 모델 ROC-AUC: {study.best_value:.4f}")
print(f"K-Fold 앙상블 ROC-AUC: {kfold_auc:.4f}")
print(f"성능 차이: {(kfold_auc - study.best_value):.4f}")

# 최적 파라미터 저장
if not os.path.exists('open'):
    os.makedirs('open')
best_params_path = "open/best_catboost_params.pkl"
joblib.dump(study.best_params, best_params_path)
print(f"\n>> [CATBoost] 최적 파라미터 저장 완료: {os.path.abspath(best_params_path)}")

# OOF 예측값 저장
oof_predictions_df = pd.DataFrame({
    'true_values': y,
    'oof_predictions': oof_predictions
})
oof_predictions_df.to_csv('open/catboost_oof_predictions.csv', index=False)
print(">> [CATBoost] OOF 예측값 저장 완료")
