import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
# train.csv 파일을 읽어오고, 타겟 '임신 성공 여부'와 특징 데이터를 분리합니다.
data = pd.read_csv("train.csv")
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
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0, step=0.1)
    }
    num_boost_round = trial.suggest_int("num_boost_round", 100, 1000, step=50)
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}, num_boost_round = {num_boost_round}")

    print(f">> [LightGBM] Trial {trial.number} - 모델 학습 시작")
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=string_columns)
    lgb_eval = lgb.Dataset(X_test, label=y_test, categorical_feature=string_columns, reference=lgb_train)
    
    # Use callback functions for early stopping and logging control
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # Suppress training log output
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
study.optimize(objective, n_trials=5)
print(">> [LightGBM] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

# 'open' 디렉토리 존재 확인 및 생성
import os
if not os.path.exists('open'):
    os.makedirs('open')
    print(">> [LightGBM] 'open' 디렉토리 생성됨")

# 최적 파라미터 저장 (LightGBM 전용 파일명으로 수정)
best_params_path = "open/best_lgbm_params.pkl"
joblib.dump(study.best_params, best_params_path)
print(f">> [LightGBM] 최적 파라미터 저장 완료: {os.path.abspath(best_params_path)}")
