import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
# train.csv 파일을 읽어들이고 타겟 변수와 특징 데이터를 분리합니다.
data = pd.read_csv("train.csv")
print("  [데이터 로드 완료] train.csv 파일을 성공적으로 읽어왔습니다.")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 전처리: 불필요한 ID 컬럼 제거 (존재 시)
if "id" in X.columns:
    X.drop(columns=["id"], inplace=True)
    print("  [전처리] ID 컬럼 제거됨")

# 전처리: 결측치 비율 90% 이상인 컬럼 제거
missing_percent = X.isnull().mean()
cols_to_drop = missing_percent[missing_percent > 0.9].index.tolist()
if cols_to_drop:
    X.drop(columns=cols_to_drop, inplace=True)
    print("  [전처리] 결측치 90% 이상 컬럼 제거:", cols_to_drop)

# 전처리: 문자열 컬럼 Label Encoding 적용
from sklearn.preprocessing import LabelEncoder
string_cols = X.select_dtypes(include=['object']).columns.tolist()
print("  [전처리] Label Encoding 적용할 문자열 컬럼:", string_cols)
for col in string_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Stratified split으로 80:20 비율로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

# 클래스 불균형 확인
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 class_weight: {default_weight:.2f}\n")

def objective(trial):
    print(f">> [RandomForest] Trial {trial.number} 시작")
    # RandomForest 모형의 하이퍼파라미터 탐색 범위 설정
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        # 클래스 불균형 처리를 위한 옵션도 탐색합니다.
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {params}")
    
    # RandomForest 모델 학습
    print(f">> [RandomForest] Trial {trial.number} - 모델 학습 시작")
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측 후 ROC-AUC 평가 진행
    print(f">> [RandomForest] Trial {trial.number} - 예측 및 평가 진행")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
print(">> [RandomForest] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

# 'open' 디렉토리 존재 확인 및 생성
import os
if not os.path.exists('open'):
    os.makedirs('open')
