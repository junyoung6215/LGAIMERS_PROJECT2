print(">> [CATBoost_Optimization] 파일 실행 시작")  # 프로그램 시작 로그

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import os

# 진행 상황을 기록하기 위한 업데이트 함수
def update_best_parameters(file_path, new_score, new_params):
    # 파일 존재 여부 확인 후 최적 파라미터 업데이트
    if os.path.exists(file_path):
        old = joblib.load(file_path)
        old_score = old.get("score", 0)
        if new_score > old_score:
            best = {"score": new_score, "params": new_params}
            joblib.dump(best, file_path)
            print("새로운 최적 파라미터 업데이트 완료: score = {:.4f}".format(new_score))
        else:
            print("기존 최적 파라미터 유지: score = {:.4f}".format(old_score))
            best = old
    else:
        best = {"score": new_score, "params": new_params}
        joblib.dump(best, file_path)
        print("최적 파라미터 저장 완료: score = {:.4f}".format(new_score))
    return best

# Step 1: 데이터 로드 및 전처리
print(">> [CATBoost_Optimization] Step 1: 데이터 로드 및 분할 시작")
data = pd.read_csv("train.csv")
if "ID" in data.columns:
    data = data.drop(columns=["ID"])
print("  [데이터 로드 완료] train.csv 파일 로드 성공")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 범주형 변수 처리
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print("  [전처리] 범주형 변수 설정:", cat_features)
X[cat_features] = X[cat_features].fillna("missing")

# 데이터 분할 (학습/테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [분할 완료] 학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 클래스 불균형 확인
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성: {neg_count}, 양성: {pos_count}, 기본 scale_pos_weight: {default_scale_pos_weight:.2f}\n")

# Optuna 목적 함수 (하이퍼파라미터 탐색)
def objective(trial):
    print(f">> [Optuna] Trial {trial.number} 시작")
    # 하이퍼파라미터 샘플링
    param = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "iterations": trial.suggest_int("iterations", 200, 800, step=50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "auto_class_weights": "Balanced",
        "verbose": False
    }
    print(f"  [Optuna] Trial {trial.number} 설정 파라미터: {param}")
    
    # 모델 객체 생성 및 학습
    model = CatBoostClassifier(cat_features=cat_features, **param, eval_metric="AUC")
    model.fit(X_train, y_train)
    print(f"  [Optuna] Trial {trial.number} 모델 학습 완료")
    
    # 성능 평가 (ROC-AUC 계산)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f">> [Optuna] Trial {trial.number} 완료: ROC-AUC = {auc:.4f}\n")
    return auc

# Step 2: Optuna 최적화 수행
print(">> [CATBoost_Optimization] Step 2: Optuna 하이퍼파라미터 최적화 시작")
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
)
study.optimize(objective, n_trials=100)

print(">> [CATBoost_Optimization] 최적화 완료")
print("  최적의 ROC-AUC:", study.best_value)
print("  최적의 파라미터:", study.best_params)

# 최적 파라미터 저장 (업데이트 단계)
if not os.path.exists('open'):
    os.makedirs('open')
best_params_path = "open/best_catboost_params.pkl"
update_best_parameters(best_params_path, study.best_value, study.best_params)

# Step 3: 최적 파라미터로 모델 재학습
print(">> [CATBoost_Optimization] Step 3: 최적 모델 재학습 시작")
best_model = CatBoostClassifier(
    cat_features=cat_features,
    eval_metric="AUC",
    **study.best_params
)
best_model.fit(X_train, y_train)
print("  [모델 재학습 완료]")

# Step 4: K-Fold 기반 앙상블 평가 함수
def evaluate_ensemble_roc_kfold(X, y, model_params, cat_features, n_splits=5):
    print(">> [앙상블 평가] K-Fold 평가 시작")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  [Fold {fold + 1}/{n_splits}] 훈련 시작")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # 각 Fold마다 모델 학습
        model = CatBoostClassifier(
            cat_features=cat_features,
            eval_metric="AUC",
            **model_params
        )
        model.fit(X_train_fold, y_train_fold, verbose=False)
        print(f"  [Fold {fold + 1}] 학습 완료")
        
        # 예측 및 ROC-AUC 계산
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = y_pred
        fold_auc = roc_auc_score(y_val_fold, y_pred)
        auc_scores.append(fold_auc)
        print(f"  [Fold {fold + 1}] ROC-AUC: {fold_auc:.4f}")
        
    final_auc = roc_auc_score(y, oof_preds)
    print(f">> [앙상블 평가] 전체 ROC-AUC: {final_auc:.4f}")
    print(f">> [앙상블 평가] 평균 Fold ROC-AUC: {np.mean(auc_scores):.4f}, 표준편차: {np.std(auc_scores):.4f}")
    return final_auc, oof_preds, auc_scores

# Step 5: 앙상블 평가 수행 및 결과 비교
print(">> [CATBoost_Optimization] Step 5: 앙상블 평가 진행")
kfold_auc, oof_predictions, fold_scores = evaluate_ensemble_roc_kfold(
    X, y, study.best_params, cat_features, n_splits=5
)
print("=== 최종 성능 비교 ===")
print(f"  단일 모델 ROC-AUC: {study.best_value:.4f}")
print(f"  K-Fold 앙상블 ROC-AUC: {kfold_auc:.4f}")
print(f"  성능 차이: {(kfold_auc - study.best_value):.4f}")

# Step 6: OOF 예측값 저장
print(">> [CATBoost_Optimization] Step 6: OOF 예측값 저장 시작")
oof_predictions_df = pd.DataFrame({'true_values': y, 'oof_predictions': oof_predictions})
oof_predictions_df.to_csv('open/catboost_oof_predictions.csv', index=False)
print(">> [CATBoost_Optimization] 모든 작업 완료")
