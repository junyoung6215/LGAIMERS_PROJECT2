import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    models = {
        "xgboost": xgb.XGBClassifier(**best_params["xgb"], use_label_encoder=False),
        "lightgbm": lgb.LGBMClassifier(**best_params["lgb"]),
        "catboost": CatBoostClassifier(**best_params["cat"], verbose=False)
    }
    
    for name, model in models.items():
        model_path = f"{MODEL_PATH}/{name}_model.pkl"
        
        if os.path.exists(model_path) and not force_retrain:
            print(f"{name} 모델 불러오는 중...")
            models[name] = joblib.load(model_path)
        else:
            print(f"{name} 모델 학습 중...")
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
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
            preds[name] = model.predict_proba(x_val)[:, 1]
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

def optimize_blend_ratios_kfold(models, X, y, n_splits=5):
    """K-Fold 기반 블렌딩 가중치 최적화"""
    print("\n🔄 K-Fold 기반 블렌딩 가중치 최적화 시작")
    print(f"  • 데이터 크기: {len(X)} 샘플")
    print(f"  • Fold 수: {n_splits}")
    print(f"  • 모델 개수: {len(models)} (XGBoost, LightGBM, CatBoost)")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def objective(trial):
        trial_num = trial.number + 1
        print(f"\n📊 Trial {trial_num} 시작")
        
        weights = {
            'xgboost': trial.suggest_float('xgboost', 0.0, 1.0),
            'lightgbm': trial.suggest_float('lightgbm', 0.0, 1.0),
            'catboost': trial.suggest_float('catboost', 0.0, 1.0)
        }
        
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        print("  📈 현재 시도 중인 가중치:")
        for model_name, weight in weights.items():
            print(f"    • {model_name}: {weight:.4f} ({weight*100:.1f}%)")
        
        # 각 trial마다 oof 예측값 배열 초기화
        trial_oof_preds = np.zeros(len(y))
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n  🔄 Fold {fold}/{n_splits} 진행 중")
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"    • 학습 데이터: {len(X_train_fold)} 샘플")
            print(f"    • 검증 데이터: {len(X_val_fold)} 샘플")

            preds = {}
            for name, model in models.items():
                print(f"    🔧 {name} 모델 학습 중...")
                
                # 모델별로 적절한 fit() 파라미터 사용
                if name == "xgboost":
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                elif name == "lightgbm":
                    # LightGBM은 verbose 인자를 생략하여 기본 로그 설정 사용
                    model.fit(X_train_fold, y_train_fold)
                elif name == "catboost":
                    model.fit(X_train_fold, y_train_fold, silent=True)
                
                preds[name] = model.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, preds[name])
                print(f"      ↳ {name} Fold {fold} ROC-AUC: {fold_score:.4f}")
            
            fold_blend = sum(weights[name] * pred for name, pred in preds.items())
            trial_oof_preds[val_idx] = fold_blend
            
            fold_blend_score = roc_auc_score(y_val_fold, fold_blend)
            fold_scores.append(fold_blend_score)
            print(f"    ✨ Fold {fold} 블렌딩 ROC-AUC: {fold_blend_score:.4f}")
        
        final_auc = roc_auc_score(y, trial_oof_preds)
        print(f"\n  📊 Trial {trial_num} 결과:")
        print(f"    • 평균 Fold ROC-AUC: {np.mean(fold_scores):.4f}")
        print(f"    • 전체 ROC-AUC: {final_auc:.4f}")
        print(f"    • Fold 점수 편차: {np.std(fold_scores):.4f}")
        
        return final_auc

    # Optuna로 최적 가중치 찾기
    print("\n🔍 Optuna 최적화 시작")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    best_weights = {k: v/sum(study.best_params.values()) for k, v in study.best_params.items()}
    
    print("\n🏆 최종 최적 블렌딩 가중치:")
    for model, weight in best_weights.items():
        print(f"  • {model}: {weight:.4f} ({weight*100:.1f}%)")
    print(f"  • 최종 ROC-AUC: {study.best_value:.4f}")
    
    return best_weights

def weighted_blend_predict(models, x_test, weights):
    """가중치 기반 예측"""
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(x_test)[:, 1]
        gc.collect()
    
    # weights 키가 "w_xgboost" 또는 "xgboost" 형식 둘 다 처리할 수 있도록 변경
    final_pred = sum(weights.get(f"w_{name}", weights.get(name, 0)) * pred for name, pred in preds.items())
    print(f"✅ 최종 예측값 샘플: {final_pred[:10]}")  # 예측값 일부 출력
    return final_pred

def run_blending_pipeline(X_train, y_train, X_test, test_ids, force_retrain=False):
    """K-Fold 기반 블렌딩 파이프라인"""
    print("\n=== 📋 블렌딩 파이프라인 시작 ===")
    
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
        pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        scores[name] = score
        print(f"{name} ROC-AUC: {score:.4f}")
    
    # 블렌딩 가중치 최적화: 항상 최적화를 실행하도록 수정
    weights_path = f"{MODEL_PATH}/blend_weights.pkl"
    best_weights = optimize_blend_ratios_kfold(models, X_train, y_train, n_splits=5)
    joblib.dump(best_weights, weights_path)
    
    # 블렌딩 예측 및 ROC-AUC 계산
    final_pred = weighted_blend_predict(models, X_val, best_weights)
    blend_roc_auc = roc_auc_score(y_val, final_pred)
    scores["ensemble_blend"] = blend_roc_auc
    
    print(f"\n=== 앙상블 블렌딩 모델 ROC-AUC: {blend_roc_auc:.4f} ===")
    
    # 최종 예측 및 CSV 저장
    final_test_pred = weighted_blend_predict(models, X_test, best_weights)
    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": final_test_pred
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
