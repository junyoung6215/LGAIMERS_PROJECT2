print(">> [RandomForest_Optimization] 파일 실행 시작")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import os

def update_best_parameters(file_path, new_score, new_params):
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

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
data = pd.read_csv("train.csv")
if "ID" in data.columns:
    data = data.drop(columns=["ID"])
print("  [데이터 로드 완료] train.csv 파일을 성공적으로 읽어왔습니다.")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

if "id" in X.columns:
    X.drop(columns=["id"], inplace=True)
    print("  [전처리] ID 컬럼 제거됨")

missing_percent = X.isnull().mean()
cols_to_drop = missing_percent[missing_percent > 0.9].index.tolist()
if cols_to_drop:
    X.drop(columns=cols_to_drop, inplace=True)
    print("  [전처리] 결측치 90% 이상 컬럼 제거:", cols_to_drop)

from sklearn.preprocessing import LabelEncoder
string_cols = X.select_dtypes(include=['object']).columns.tolist()
print("  [전처리] Label Encoding 적용할 문자열 컬럼:", string_cols)
for col in string_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 class_weight: {default_weight:.2f}\n")

def objective(trial):
    print(f">> [RandomForest] Trial {trial.number} 시작")
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {params}")
    
    print(f">> [RandomForest] Trial {trial.number} - 모델 학습 시작")
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(f">> [RandomForest] Trial {trial.number} - 예측 및 평가 진행")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(">> [RandomForest] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

if not os.path.exists('open'):
    os.makedirs('open')
    print(">> [RandomForest] 'open' 디렉토리 생성됨")

best_params_path = "open/best_rf_params.pkl"
update_best_parameters(best_params_path, study.best_value, study.best_params)

print(">> [RandomForest_Optimization] 파일 실행 종료")
