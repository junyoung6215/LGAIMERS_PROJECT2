import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
# train.csv 파일을 불러오고, 타겟 변수 '임신 성공 여부'를 기준으로 데이터를 분리합니다.
data = pd.read_csv("train.csv")
print("  [데이터 로드 완료] train.csv 파일 성공적으로 로드됨")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 범주형 변수 리스트 생성 (원본 데이터 유지) 및 NaN 값을 'missing'으로 대체
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print("🔍 [CATBoost] cat_features 설정:", cat_features)
for col in cat_features:
    X[col] = X[col].fillna("missing")
    print(f"  [채우기] {col} NaN 값을 'missing'으로 대체")

# stratify 옵션을 사용해 데이터를 80:20 비율로 분할 (클래스 불균형 대비)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

# 클래스 불균형 확인을 위한 비율 계산
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 scale_pos_weight: {default_scale_pos_weight:.2f}\n")

def objective(trial):
    print(f">> [CATBoost] Trial {trial.number} 시작")
    # CatBoost 하이퍼파라미터 범위 설정
    param = {
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        # CatBoost는 class_weights를 리스트로 받으므로 음성은 1, 양성 가중치는 탐색
        "class_weights": [1.0, trial.suggest_float("scale_pos_weight", 0.5, 2.0, step=0.1)]
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}")
    
    # CatBoost 모델 학습 시작
    print(f">> [CATBoost] Trial {trial.number} - 모델 학습 시작")
    model = CatBoostClassifier(cat_features=cat_features, **param, eval_metric="Logloss", verbose=False)
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측 후 ROC-AUC 평가
    print(f">> [CATBoost] Trial {trial.number} - 예측 및 평가 진행")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
print(">> [CATBoost] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

# 기존 저장 및 추가 저장을 하나로 합침: open 폴더 내에 단일 파일로 저장
best_params_path = "open/best_catboost_params.pkl"
joblib.dump(study.best_params, best_params_path)
import os
print(f">> [CATBoost] 최적 파라미터 저장 완료: {os.path.abspath(best_params_path)}")
