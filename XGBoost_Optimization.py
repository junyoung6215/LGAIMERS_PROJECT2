import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
import os

print(">> [XGBoost_Optimization] 파일 실행 시작")

# XGBoost용 파라미터 업데이트 함수: 기존 pkl 파일 로드, 비교, 업데이트 과정을 상세 로그로 출력
def update_best_parameters(params_path, new_roc, new_params):
    if os.path.exists(params_path):
        old_record = joblib.load(params_path)
        old_roc = old_record.get("roc_auc", 0)
        print(f"[XGBoost LOG] 기존 파라미터 로드 성공: {old_record}")
    else:
        old_roc = 0
        print("[XGBoost LOG] 기존 파라미터 파일이 없음. 새 파일 생성 예정.")
    print(f"[XGBoost LOG] 새 ROC-AUC: {new_roc:.4f} vs 기존 ROC-AUC: {old_roc:.4f}")
    if new_roc > old_roc:
        best_record = {"params": new_params, "roc_auc": new_roc}
        joblib.dump(best_record, params_path)
        print(f"[XGBoost LOG] 🏆 파라미터 업데이트 완료: 새 ROC-AUC = {new_roc:.4f}")
    else:
        print(f"[XGBoost LOG] ℹ 업데이트 없이 기존 파라미터 유지: 기존 ROC-AUC = {old_roc:.4f}")

# Step 1: 데이터 로드 및 분할 시작
print("Step 1: 데이터 로드 및 분할 시작")
data = pd.read_csv("train.csv")
if "ID" in data.columns:
    data = data.drop(columns=["ID"])
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

# Optuna 목적 함수 - XGBoost 모델을 위한 하이퍼파라미터 탐색
def objective(trial):
    print(f">> [XGBoost] Trial {trial.number} 시작")
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "random_state": 42
    }
    print(f"[Optuna] Trial {trial.number} 설정된 파라미터: {param}")
    
    from sklearn.model_selection import StratifiedKFold
    auc_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 수정 후: 튜닝된 n_estimators 값을 그대로 사용해 한 번에 학습
        model = xgb.XGBClassifier(**param, use_label_encoder=False)
        model.fit(X_tr, y_tr, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores)
    print(f"[Optuna] Trial {trial.number} 완료: 평균 ROC-AUC = {mean_auc:.4f}\n")
    return mean_auc

def optimize_xgb():
    best_param_file = "open/best_xgb_params.pkl"
    old_score = 0
    if os.path.exists(best_param_file):
        prev = joblib.load(best_param_file)
        old_score = prev.get("roc_auc", 0)
        print(f"[XGBoost LOG] 기존 XGBoost 파라미터 파일 로드됨: {prev}")
    else:
        print("[XGBoost LOG] 기존 XGBoost 파라미터 파일이 없습니다.")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    new_score = study.best_value
    print(f"[XGBoost LOG] 최적화 완료: 새 ROC-AUC = {new_score:.4f}")
    if new_score > old_score:
        best_params = study.best_params
        best_params["roc_auc"] = new_score
        joblib.dump(best_params, best_param_file)
        print(f"[XGBoost LOG] 새 파라미터 저장 완료: {best_params}")
    else:
        print(f"[XGBoost LOG] 기존 파라미터 유지: 기존 ROC-AUC = {old_score:.4f}")
    
    # 최종적으로 update_best_parameters 함수를 호출하여 업데이트 진행
    update_best_parameters(best_param_file, new_score, study.best_params)

if __name__ == "__main__":
    optimize_xgb()
print("Step 3: 최적 파라미터 저장 완료")