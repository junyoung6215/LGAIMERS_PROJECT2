import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc  # Garbage Collector
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 모델 및 파라미터 저장 경로 설정
MODEL_PATH = "open/models"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    print("\n=== 데이터 로드 및 전처리 시작 ===")
    
    try:
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        print("✓ 데이터 파일 로드 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {str(e)}")
        return None, None, None, None
    
    # 특성과 타겟 분리
    X = train.drop(columns=["임신 성공 여부"])
    y = train["임신 성공 여부"]
    test_ids = test["ID"].values
    X_test = test.drop(columns=["ID"])
    
    # 범주형 변수 처리
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].fillna("missing")
        X_test[col] = X_test[col].fillna("missing")
    
    print(f"✓ 학습 데이터 shape: {X.shape}")
    print(f"✓ 테스트 데이터 shape: {X_test.shape}")
    
    return X, X_test, y, test_ids

def load_best_params():
    """최적 파라미터 로드"""
    print("\n=== 최적 파라미터 로드 ===")
    try:
        params = {
            "xgb": joblib.load('open/best_xgb_params.pkl'),
            "lgb": joblib.load('open/best_lgbm_params.pkl'),
            "cat": joblib.load('open/best_catboost_params.pkl')
        }
        print("✓ 모든 모델의 최적 파라미터 로드 완료")
        return params
    except Exception as e:
        print(f"❌ 파라미터 로드 실패: {str(e)}")
        return None

def load_or_train_models(X_train, y_train, params, force_retrain=False):
    """모델 로드 또는 학습"""
    print("\n=== 모델 로드 또는 학습 시작 ===")
    
    models = {
        "xgboost": xgb.XGBClassifier(**params["xgb"], use_label_encoder=False),
        "lightgbm": lgb.LGBMClassifier(**params["lgb"]),
        "catboost": CatBoostClassifier(**params["cat"], verbose=False)
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

def ensemble_predict(models, X_test, weights=None):
    """앙상블 예측 수행"""
    print("\n=== 앙상블 예측 시작 ===")
    
    if weights is None:
        weights = {name: 1/len(models) for name in models.keys()}
        print("ℹ️ 가중치가 지정되지 않아 균등 가중치 사용")
    
    predictions = {}
    for name, model in models.items():
        print(f"⚙️ {name} 모델 예측 중...")
        pred = model.predict_proba(X_test)[:, 1]
        predictions[name] = pred
    
    # 가중 평균으로 최종 예측
    final_pred = sum(predictions[name] * weight 
                    for name, weight in weights.items())
    
    print("✓ 앙상블 예측 완료")
    return final_pred

def main():
    print("\n🚀 임신 성공 예측 모델 실행")
    print("사용 모델: XGBoost, LightGBM, CatBoost")
    
    # 데이터 로드 및 전처리
    X, X_test, y, test_ids = load_and_preprocess_data()
    if X is None:
        return
    
    # 최적 파라미터 로드
    params = load_best_params()
    if params is None:
        return
    
    # 모델 로드 또는 학습
    models = load_or_train_models(X, y, params, force_retrain=False)
    
    # 앙상블 예측 수행
    predictions = ensemble_predict(models, X_test)
    
    # 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    print("\n✨ 예측 완료! submission.csv 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
