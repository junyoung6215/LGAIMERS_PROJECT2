print(">> [LightGBM_Optimization] 파일 실행 시작")  # 파일 실행 로그

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import os

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
# train.csv 파일을 읽어오고, 타겟 '임신 성공 여부'와 특징 데이터를 분리합니다.
data = pd.read_csv("train.csv")
if "ID" in data.columns:
    data = data.drop(columns=["ID"])
print("  [데이터 로드 완료] train.csv 파일 읽기 성공")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 범주형 컬럼 처리: category 타입으로 변환
string_columns = X.select_dtypes(include=['object']).columns.tolist()
print("🔍 [LightGBM] 범주형 변수:", string_columns)
for col in string_columns:
    X[col] = X[col].fillna("missing")  # 결측치 처리
    X[col] = X[col].astype("category")  # category 타입으로 변환
print("  [범주형 변수 변환 완료] category 타입으로 변환됨")

# Stratified 방식으로 데이터를 80:20 비율로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

# 클래스 불균형 확인 및 scale_pos_weight 계산
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 scale_pos_weight: {default_scale_pos_weight:.2f}\n")

def objective(trial):
    print(f">> [LightGBM] Trial {trial.number} 시작")
    # LightGBM의 하이퍼파라미터 탐색 범위 설정
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 20.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0, step=0.1),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.2)
    }
    num_boost_round = trial.suggest_int("num_boost_round", 500, 2000, step=100)
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}, num_boost_round = {num_boost_round}")

    print(f">> [LightGBM] Trial {trial.number} - 모델 학습 시작")
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=string_columns)
    lgb_eval = lgb.Dataset(X_test, label=y_test, categorical_feature=string_columns, reference=lgb_train)
    
    # Callback 함수로 early stopping 사용, verbose_eval 제거
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # 로그 출력 억제
    ]
    
    gbm = lgb.train(
        param,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_eval],
        callbacks=callbacks
    )
    
    print(f">> [LightGBM] Trial {trial.number} - 예측 및 평가 진행")
    y_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(">> [LightGBM] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_ensemble_roc_kfold(X, y, model_params, n_splits=5):
    print(">> [LightGBM] K-Fold 앙상블 평가 함수 실행")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  [Fold {fold+1}/{n_splits}] 훈련 시작")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        # 모델 학습 (verbose_eval 제거)
        model = lgb.train(model_params,
                          train_data,
                          valid_sets=[val_data])
        print(f"  [Fold {fold+1}] 모델 학습 완료")
        
        # 예측 및 ROC-AUC 계산
        y_pred = model.predict(X_val_fold)
        oof_preds[val_idx] = y_pred
        fold_auc = roc_auc_score(y_val_fold, y_pred)
        auc_scores.append(fold_auc)
        print(f"  [Fold {fold+1}] ROC-AUC: {fold_auc:.4f}")
        
    final_auc = roc_auc_score(y, oof_preds)
    print(f">> [LightGBM] 최종 ROC-AUC: {final_auc:.4f}")
    print(f">> [LightGBM] 평균 Fold ROC-AUC: {np.mean(auc_scores):.4f}, 표준편차: {np.std(auc_scores):.4f}")
    return final_auc, oof_preds, auc_scores

# 앙상블 평가 코드 추가
print("\n>> [LightGBM] K-Fold 기반 앙상블 성능 평가")
ensemble_auc, oof_predictions, fold_scores = evaluate_ensemble_roc_kfold(X, y, study.best_params, n_splits=5)

print("\n=== 최종 성능 비교 ===")
print(f"단일 모델 ROC-AUC: {study.best_value:.4f}")
print(f"K-Fold 앙상블 ROC-AUC: {ensemble_auc:.4f}")
print(f"성능 차이: {(ensemble_auc - study.best_value):.4f}")

# 'open' 디렉토리 존재 확인 및 생성
if not os.path.exists('open'):
    os.makedirs('open')
    print(">> [LightGBM] 'open' 디렉토리 생성됨")

# 최적 파라미터 저장 (LightGBM 전용 파일명으로 수정)
best_params_path = "open/best_lightgbm_params.pkl"
joblib.dump(study.best_params, best_params_path)
print(f">> [LightGBM] 최적 파라미터 저장 완료: {os.path.abspath(best_params_path)}")

# OOF 예측값 저장
oof_df = pd.DataFrame({'true_values': y, 'oof_predictions': oof_predictions})
oof_df.to_csv('open/lightgbm_oof_predictions.csv', index=False)
print(">> [LightGBM] OOF 예측값 저장 완료")
print(">> [LightGBM_Optimization] 파일 실행 종료")
