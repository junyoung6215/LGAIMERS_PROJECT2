import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import gc
from sklearn.metrics import roc_auc_score
import optuna
import os

# 모델 및 가중치 저장 경로 설정
MODEL_PATH = "open/models"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_best_params():
    """Optuna로 찾은 최적 파라미터 로드"""
    try:
        best_params = {
            "xgb": joblib.load('open/best_xgb_params.pkl'),
            "lgb": joblib.load('open/best_lgbm_params.pkl'),
            "rf": joblib.load('open/best_rf_params.pkl'),
            "cat": joblib.load('open/best_catboost_params.pkl')
        }
        print("모든 모델의 최적 파라미터를 성공적으로 로드했습니다.")
        return best_params
    except Exception as e:
        print(f"파라미터 로드 중 에러 발생: {str(e)}")
        return None

def load_and_merge_params():
    """각 모델의 최적 파라미터를 로드하고 병합"""
    try:
        # 각 모델의 파라미터 파일 로드
        params = {
            "xgb": joblib.load('open/best_xgb_params.pkl'),
            "lgb": joblib.load('open/best_lgbm_params.pkl'),
            "cat": joblib.load('open/best_catboost_params.pkl')
        }
        
        # 통합 파라미터 파일 저장
        joblib.dump(params, 'open/best_params.pkl')
        print("모든 모델의 최적 파라미터를 성공적으로 로드하고 병합했습니다.")
        return params
    except Exception as e:
        print(f"파라미터 로드 중 에러 발생: {str(e)}")
        return None

def load_or_train_models(x_train, y_train, best_params, force_retrain=False):
    """모델 로드 또는 학습"""
    models = {
        "xgboost": xgb.XGBClassifier(**best_params["xgb"], use_label_encoder=False),
        "lightgbm": lgb.LGBMClassifier(**best_params["lgb"]),
        "catboost": CatBoostClassifier(**best_params["cat"], verbose=False)
    }
    
    if not force_retrain:
        try:
            for name in models.keys():
                model_path = f"{MODEL_PATH}/{name}_model.pkl"
                if os.path.exists(model_path):
                    models[name] = joblib.load(model_path)
                    print(f"{name} 모델을 로드했습니다.")
                else:
                    raise FileNotFoundError
            return models
        except:
            print("저장된 모델을 찾을 수 없어 새로 학습합니다.")
    
    print("\n=== 개별 모델 학습 시작 ===")
    for name, model in models.items():
        print(f"\n{name} 모델 학습 중...")
        model.fit(x_train, y_train)
        joblib.dump(model, f"{MODEL_PATH}/{name}_model.pkl")
        gc.collect()
        print(f"{name} 모델 학습 및 저장 완료")
    
    return models

def optimize_blend_ratios(models, x_val, y_val):
    """Optuna를 사용해 블렌딩 비율 최적화"""
    def objective(trial):
        # 키 이름을 models과 일치하도록 수정
        weights = {
            'xgboost': trial.suggest_float('xgboost', 0.0, 1.0),
            'lightgbm': trial.suggest_float('lightgbm', 0.0, 1.0),
            'catboost': trial.suggest_float('catboost', 0.0, 1.0)
        }
        
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        preds = {}
        for name, model in models.items():
            preds[name] = model.predict_proba(x_val)[:,1]
            gc.collect()
        
        final_pred = sum(weights[name] * pred for name, pred in preds.items())
        return roc_auc_score(y_val, final_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    # 최적 가중치 저장 시에도 동일한 키 이름 사용
    best_weights = {
        'xgboost': study.best_params['xgboost'],
        'lightgbm': study.best_params['lightgbm'],
        'catboost': study.best_params['catboost']
    }
    
    total = sum(best_weights.values())
    best_weights = {k: v/total for k, v in best_weights.items()}
    
    print("\n=== 최적 블렌딩 가중치 ===")
    for model, weight in best_weights.items():
        print(f"{model}: {weight:.4f} ({weight*100:.1f}%)")
    
    return best_weights

def weighted_blend_predict(models, x_test, weights):
    """가중치 기반 예측"""
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(x_test)[:,1]
        gc.collect()
    
    # weights 키가 "w_xgboost" 또는 "xgboost" 형식 둘 다 처리할 수 있도록 변경
    final_pred = sum(weights.get(f"w_{name}", weights.get(name, 0)) * pred for name, pred in preds.items())
    print(f"✅ 최종 예측값 샘플: {final_pred[:10]}")  # 예측값 일부 출력
    return final_pred

def run_blending_pipeline(X_train, y_train, X_test, test_ids, force_retrain=False):
    """블렌딩 전용 파이프라인"""
    print("\n=== 블렌딩 파이프라인 시작 ===")
    
    # 검증 세트 분리
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 모델 파라미터 로드 및 병합
    best_params = load_and_merge_params()
    if best_params is None:
        return None, None, None
    
    # 모델 로드 또는 학습
    models = load_or_train_models(X_train_main, y_train_main, best_params, force_retrain)
    
    # 개별 모델 성능 평가
    print("\n=== 개별 모델 성능 ===")
    scores = {}
    for name, model in models.items():
        pred = model.predict_proba(X_val)[:,1]
        score = roc_auc_score(y_val, pred)
        scores[name] = score
        print(f"{name} ROC-AUC: {score:.4f}")
    
    # 블렌딩 가중치 최적화
    weights_path = f"{MODEL_PATH}/blend_weights.pkl"
    if os.path.exists(weights_path) and not force_retrain:
        best_weights = joblib.load(weights_path)
        print("\n기존 블렌딩 가중치를 로드했습니다.")
    else:
        best_weights = optimize_blend_ratios(models, X_val, y_val)
        joblib.dump(best_weights, weights_path)
    
    # 최종 예측 및 CSV 저장
    final_pred = weighted_blend_predict(models, X_test, best_weights)
    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": final_pred
    })
    
    submission.to_csv("blend_prediction.csv", index=False)
    if os.path.exists("blend_prediction.csv"):
        print("✅ blend_prediction.csv 파일이 정상적으로 생성되었습니다.")
    else:
        print("❌ blend_prediction.csv 파일이 생성되지 않았습니다.")
    return submission, scores, best_weights

def main():
    from model_prediction_efficient import load_and_preprocess_data
    
    print("블렌딩 모델 실행")
    print("사용 모델: XGBoost(xgb), LightGBM(lgb), CatBoost(cat)")
    
    # 데이터 로드
    X_train, X_test, y_train, test_ids = load_and_preprocess_data()
    
    # 블렌딩 실행
    submission, scores, weights = run_blending_pipeline(
        X_train, y_train, X_test, test_ids, force_retrain=False
    )
    
    if submission is not None:
        # 최종 성능 요약 출력
        print("\n=== 최종 성능 요약 ===")
        print("-" * 40)
        print("모델명".ljust(15), "ROC-AUC".rjust(10))
        print("-" * 40)
        for model, score in scores.items():
            print(f"{model.ljust(15)} {score:.4f}".rjust(15))
        print("-" * 40)

if __name__ == "__main__":
    main()
