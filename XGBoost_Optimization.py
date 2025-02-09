import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
import os

# Step 1: 데이터 로드 및 분할 (타겟: "임신 성공 여부")
print("Step 1: 데이터 로드 및 분할 시작")
data = pd.read_csv("train.csv")
print("  [데이터 로드 완료] train.csv 파일을 성공적으로 읽어왔습니다.")
X = data.drop(columns=["임신 성공 여부"])
y = data["임신 성공 여부"]

# 범주형 변수 처리
cat_features = X.select_dtypes(include=['object']).columns
print("🔍 [XGBoost] 범주형 변수:", cat_features.tolist())

# 결측치를 'missing'으로 채우고 Label Encoding 적용
label_encoders = {}
for col in cat_features:
    X[col] = X[col].fillna("missing")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
print("  [범주형 변수 변환 완료] Label encoding 적용됨")
print(f"  [변환 후 특성 수] {X.shape[1]}개")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  [데이터 분할 완료] 학습 데이터 shape: {X_train.shape}, 테스트 데이터 shape: {X_test.shape}")

# 클래스 불균형 확인
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
default_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"  [클래스 비율] 음성:{neg_count}, 양성:{pos_count}, 기본 scale_pos_weight: {default_scale_pos_weight:.2f}\n")

def objective(trial):
    print(f">> [XGBoost] Trial {trial.number} 시작")
    # XGBoost 하이퍼파라미터 탐색 범위 설정
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0, step=0.1),
        "random_state": 42
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}")
    
    # XGBoost 모델 학습 - early_stopping_rounds 제거
    print(f">> [XGBoost] Trial {trial.number} - 모델 학습 시작")
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 테스트 데이터 예측 및 평가
    print(f">> [XGBoost] Trial {trial.number} - 예측 및 평가 진행")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[Optuna] Trial {trial.number} 완료: ROC-AUC = {auc}\n")
    return auc

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
print(">> [XGBoost] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

# 'open' 디렉토리 존재 확인 및 생성
if not os.path.exists('open'):
    os.makedirs('open')
    print(">> [XGBoost] 'open' 디렉토리 생성됨")

# 최적 파라미터 저장
best_params_path = "open/best_xgb_params.pkl"
joblib.dump(study.best_params, best_params_path)
print(f">> [XGBoost] 최적 파라미터 저장 완료: {os.path.abspath(best_params_path)}")
