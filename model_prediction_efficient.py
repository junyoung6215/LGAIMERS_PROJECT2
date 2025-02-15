import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# 모델 파일 및 파라미터 파일을 저장할 폴더 생성
MODEL_PATH = "open/models"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_and_preprocess_data():
    """
    train.csv와 test.csv를 로드하고 전처리 수행.
    - train.csv: 학습 데이터와 타겟 분리 (만약 "ID" 컬럼이 있다면 제거)
    - test.csv: 테스트 데이터 및 ID 분리 (ID 컬럼이 없으면 인덱스를 사용)
    - 범주형 변수의 결측치는 "missing"으로 채우고 Label Encoding 적용
    """
    print("\n=== 데이터 로드 및 전처리 시작 ===")
    try:
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        print("✓ 데이터 파일 로드 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {str(e)}")
        return None, None, None, None

    # train 데이터에 'ID' 컬럼이 있으면 제거합니다.
    if "ID" in train.columns:
        train = train.drop(columns=["ID"])
    
    # 학습 데이터와 타겟 분리
    X = train.drop(columns=["임신 성공 여부"])
    y = train["임신 성공 여부"]
    
    # test 데이터에서 'ID' 컬럼은 추후 제출용으로 분리
    if "ID" in test.columns:
        test_ids = test["ID"].values
        X_test = test.drop(columns=["ID"])
    else:
        print("ℹ️ test.csv에 'ID' 컬럼이 없으므로 인덱스를 사용합니다.")
        test_ids = np.arange(len(test))
        X_test = test.copy()

    # 범주형 변수 처리: 결측치 채우기 및 Label Encoding 적용
    from sklearn.preprocessing import LabelEncoder
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].fillna("missing")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    cat_features_test = X_test.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features_test:
        X_test[col] = X_test[col].fillna("missing")
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col])
    
    print(f"✓ 학습 데이터 shape: {X.shape}")
    print(f"✓ 테스트 데이터 shape: {X_test.shape}")
    
    return X, X_test, y, test_ids

def load_best_params():
    """
    저장된 각 모델의 최적 파라미터 파일에서 파라미터를 불러옴.
    만약 파일에 "params" 키가 있다면 해당 값을, 없다면 로드한 dict 자체를 사용합니다.
    """
    print("\n=== 최적 파라미터 로드 ===")
    try:
        best_params = {}

        # XGBoost
        xgb_params = joblib.load('open/best_xgb_params.pkl')
        best_params["xgb"] = xgb_params["params"] if isinstance(xgb_params, dict) and "params" in xgb_params else xgb_params

        # LightGBM
        try:
            lgb_params = joblib.load('open/best_lgbm_params.pkl')
        except Exception:
            lgb_params = joblib.load('open/best_lightgbm_params.pkl')
        best_params["lgb"] = lgb_params["params"] if isinstance(lgb_params, dict) and "params" in lgb_params else lgb_params

        # CatBoost
        cat_params = joblib.load('open/best_catboost_params.pkl')
        best_params["cat"] = cat_params["params"] if isinstance(cat_params, dict) and "params" in cat_params else cat_params

        # RandomForest
        rf_params = joblib.load('open/best_rf_params.pkl')
        best_params["rf"] = rf_params["params"] if isinstance(rf_params, dict) and "params" in rf_params else rf_params

        print("✓ 모든 모델의 최적 파라미터를 성공적으로 로드했습니다.")
        return best_params
    except Exception as e:
        print(f"❌ 파라미터 로드 실패: {str(e)}")
        return None

def load_or_train_models(X_train, y_train, params, force_retrain=False):
    """
    저장된 최적 파라미터를 이용해 모델 객체를 생성한 후,
    open/models 폴더에 저장된 모델 파일이 있으면 로드하고,
    없으면 학습시켜 저장한다.
    """
    print("\n=== 모델 로드 또는 학습 시작 ===")
    models = {
        "xgboost": xgb.XGBClassifier(**params["xgb"], use_label_encoder=False, eval_metric="auc"),
        "lightgbm": lgb.LGBMClassifier(**params["lgb"]),
        "catboost": CatBoostClassifier(**params["cat"], verbose=False, eval_metric="AUC"),
        "randomforest": RandomForestClassifier(**params["rf"], random_state=42, n_jobs=-1)
    }
    
    for name, model in models.items():
        model_path = f"{MODEL_PATH}/{name}_model.pkl"
        if os.path.exists(model_path) and not force_retrain:
            print(f"✓ {name} 모델 로드 중...")
            models[name] = joblib.load(model_path)
        else:
            print(f"⚙️ {name} 모델 학습 중...")
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            print(f"✓ {name} 모델 학습 및 저장 완료")
    
    return models

def evaluate_model(model, X_val, y_val):
    """
    모델을 사용하여 검증 데이터(X_val)로 예측을 수행하고 ROC-AUC 점수를 계산하여 반환.
    """
    y_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    return score, y_pred

def main():
    print("\n🚀 개별 모델 예측 실행")
    # 데이터 로드 및 전처리
    X, X_test, y, test_ids = load_and_preprocess_data()
    if X is None:
        return
    # 학습 데이터를 80:20 비율로 나누어 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 저장된 최적 파라미터 불러오기
    params = load_best_params()
    if params is None:
        return
    
    # 최적 파라미터를 사용하여 모델 로드 또는 학습
    models = load_or_train_models(X_train, y_train, params, force_retrain=False)
    
    # 각 모델의 검증 ROC-AUC 평가 및 출력
    scores = {}
    for name, model in models.items():
        score, _ = evaluate_model(model, X_val, y_val)
        scores[name] = score
        print(f"{name} 모델의 검증 ROC-AUC: {score:.4f}")
    
    # 각 모델을 사용하여 test.csv에 대해 예측 수행 및 개별 CSV 파일 저장
    for name, model in models.items():
        print(f"{name} 모델 예측 수행 중...")
        pred = model.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            "ID": test_ids,
            "probability": pred
        })
        output_path = f"submission_{name}.csv"
        submission.to_csv(output_path, index=False)
        print(f"✨ {name} 예측 완료! {output_path} 파일 생성됨")
    
    # 검증 ROC-AUC 요약 출력
    print("\n=== 개별 모델 검증 ROC-AUC 요약 ===")
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()

print(">> [model_prediction_efficient] 파일 실행 종료")
