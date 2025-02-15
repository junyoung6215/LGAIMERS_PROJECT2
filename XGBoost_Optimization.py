import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
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

print("Step 2: Optuna를 통한 하이퍼파라미터 최적화 시작")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(">> [XGBoost] 최적화 완료")
print("최적의 ROC-AUC:", study.best_value)
print("최적의 파라미터:", study.best_params)

if not os.path.exists('open'):
    os.makedirs('open')
    print(">> [XGBoost] 'open' 디렉토리 생성됨")

best_params_path = "open/best_xgb_params.pkl"
update_best_parameters(best_params_path, study.best_value, study.best_params)
print("Step 3: 최적 파라미터 저장 완료")