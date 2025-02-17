import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import gc
import os

MODEL_PATH = "open/models"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_best_params():
    """저장된 최적 파라미터 파일에서 각 모델의 'params' 부분만 불러옴"""
    try:
        best_params = {}
        # XGBoost
        xgb_params = joblib.load('open/best_xgb_params.pkl')
        xgb_dict = xgb_params["params"] if isinstance(xgb_params, dict) and "params" in xgb_params else xgb_params
        xgb_dict.pop("roc_auc", None)  # "roc_auc" 키 제거
        best_params["xgb"] = xgb_dict

        # LightGBM
        try:
            lgb_params = joblib.load('open/best_lgbm_params.pkl')
        except Exception:
            lgb_params = joblib.load('open/best_lightgbm_params.pkl')
        lgb_dict = lgb_params["params"] if isinstance(lgb_params, dict) and "params" in lgb_params else lgb_params
        lgb_dict.pop("roc_auc", None)
        best_params["lgb"] = lgb_dict

        # CatBoost
        cat_params = joblib.load('open/best_catboost_params.pkl')
        cat_dict = cat_params["params"] if isinstance(cat_params, dict) and "params" in cat_params else cat_params
        cat_dict.pop("roc_auc", None)
        best_params["cat"] = cat_dict

        # RandomForest
        rf_params = joblib.load('open/best_rf_params.pkl')
        rf_dict = rf_params["params"] if isinstance(rf_params, dict) and "params" in rf_params else rf_params
        rf_dict.pop("roc_auc", None)
        best_params["rf"] = rf_dict

        print("✓ 모든 모델의 최적 파라미터를 성공적으로 로드했습니다.")
        return best_params
    except Exception as e:
        print(f"❌ 파라미터 로드 중 에러 발생: {str(e)}")
        return None

def load_and_merge_params():
    """여러 모델의 최적 파라미터를 하나의 dict로 병합 후 저장"""
    try:
        params = {
            "xgb": joblib.load('open/best_xgb_params.pkl'),
            "lgb": None,
            "cat": joblib.load('open/best_catboost_params.pkl')
        }
        try:
            lgb_params = joblib.load('open/best_lgbm_params.pkl')
        except Exception:
            lgb_params = joblib.load('open/best_lightgbm_params.pkl')
        params["lgb"] = lgb_params

        # 각 모델 파라미터에서 메타정보(roc_auc) 제거
        for key in params:
            model_params = params[key]["params"] if isinstance(params[key], dict) and "params" in params[key] else params[key]
            model_params.pop("roc_auc", None)
            params[key] = model_params

        joblib.dump(params, 'open/best_params.pkl')
        print("✓ 모든 모델의 최적 파라미터를 성공적으로 로드하고 병합했습니다.")
        return params
    except Exception as e:
        print(f"❌ 파라미터 로드 중 에러 발생: {str(e)}")
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

def optimize_blend_ratios_kfold(models, X, y, n_splits=5):
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
        
        trial_oof_preds = np.zeros(len(y))
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n  🔄 Fold {fold}/{n_splits} 진행 중")
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            preds = {}
            for name, model in models.items():
                print(f"    🔧 {name} 모델 학습 중...")
                if name == "xgboost":
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                elif name == "lightgbm":
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

    print("\n🔍 Optuna 최적화 시작")
    import optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    best_weights = {k: v/sum(study.best_params.values()) for k, v in study.best_params.items()}
    print("\n🏆 최종 최적 블렌딩 가중치:")
    for model, weight in best_weights.items():
        print(f"  • {model}: {weight:.4f} ({weight*100:.1f}%)")
    print(f"  • 최종 ROC-AUC: {study.best_value:.4f}")
    
    return best_weights

def weighted_blend_predict(models, x_test, weights):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(x_test)[:, 1]
        gc.collect()
    final_pred = sum(weights.get(name, 0) * preds[name] for name in models.keys())
    print(f"✅ 최종 예측값 샘플: {final_pred[:10]}")
    return final_pred

def update_best_parameters(file_path, new_score, new_params, param_type="model"):
    """
    파라미터(모델 파라미터 또는 블렌딩 가중치)를 업데이트하는 통합 함수
    Args:
        file_path: 저장할 파일 경로
        new_score: 새로운 ROC-AUC 점수
        new_params: 새로운 파라미터 또는 가중치
        param_type: "model" 또는 "blend"
    """
    print(f"\n>> [{param_type.upper()}] 파라미터 업데이트 검토")
    if os.path.exists(file_path):
        old_data = joblib.load(file_path)
        old_score = old_data.get("roc_auc", 0)
        print(f"  • 기존 ROC-AUC: {old_score:.4f}")
        print(f"  • 새로운 ROC-AUC: {new_score:.4f}")
        
        if new_score > old_score:
            best_data = {
                "params" if param_type == "model" else "weights": new_params,
                "roc_auc": new_score
            }
            joblib.dump(best_data, file_path)
            print(f"✅ 성능 향상으로 인한 업데이트 완료 ({old_score:.4f} → {new_score:.4f})")
            return best_data
        else:
            print(f"ℹ️ 기존 파라미터 유지 (새로운 성능이 더 낮거나 같음)")
            return old_data
    else:
        best_data = {
            "params" if param_type == "model" else "weights": new_params,
            "roc_auc": new_score
        }
        joblib.dump(best_data, file_path)
        print(f"✨ 새로운 파라미터 저장 완료 (ROC-AUC: {new_score:.4f})")
        return best_data

def load_blend_weights(weights_path):
    """
    블렌딩 가중치를 안전하게 로드하는 함수
    실패하면 기본 가중치를 반환
    """
    try:
        if os.path.exists(weights_path):
            data = joblib.load(weights_path)
            if isinstance(data, dict):
                # "weights" 키가 있는 경우
                if "weights" in data:
                    return data["weights"], data.get("roc_auc", 0)
                # 직접 가중치가 저장된 경우
                if all(k in data for k in ["xgboost", "lightgbm", "catboost"]):
                    return data, 0
            print("⚠️ 저장된 가중치 파일의 형식이 올바르지 않음")
        else:
            print("⚠️ 저장된 가중치 파일이 없음")
    except Exception as e:
        print(f"⚠️ 가중치 로드 중 에러 발생: {str(e)}")
    
    # 기본 가중치 반환
    default_weights = {
        "xgboost": 0.34,
        "lightgbm": 0.33,
        "catboost": 0.33
    }
    print("ℹ️ 기본 가중치를 사용합니다:", default_weights)
    return default_weights, 0

def save_blend_weights(weights_path, weights, score):
    """
    블렌딩 가중치를 지정된 형식으로 저장
    """
    data = {
        "weights": weights,
        "roc_auc": score
    }
    joblib.dump(data, weights_path)
    print(f"✅ 가중치 저장 완료 (ROC-AUC: {score:.4f})")

def run_blending_pipeline(X_train, y_train, X_test, test_ids, force_retrain=False):
    print("\n=== 📋 블렌딩 파이프라인 시작 ===")
    print(f"[CONFIG] force_retrain: {force_retrain}")
    
    # force_retrain 값 설명: True인 경우 기존 저장된 모델을 무시하고 재학습, False인 경우 저장된 모델이 있으면 로드합니다.
    print(f"[DEBUG] force_retrain 값: {force_retrain} (True: 모델 재학습, False: 기존 저장 모델 사용)")
    
    # 데이터 분할: 전체 학습 데이터를 메인 학습셋과 검증셋으로 분할합니다.
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print("[INFO] 데이터 분할 완료: 학습 메인셋과 검증셋 구성")
    
    best_params = load_and_merge_params()
    if best_params is None:
        print("[ERROR] 최적 파라미터 로드 실패. 블렌딩 파이프라인 종료")
        return None, None, None
    
    print("[INFO] 개별 모델 학습/로딩 시작")
    models = load_or_train_models(X_train_main, y_train_main, best_params, force_retrain)
    print("[INFO] 개별 모델 학습/로딩 완료")
    
    print("\n=== 개별 모델 성능 평가 시작 ===")
    scores = {}
    for name, model in models.items():
        pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        scores[name] = score
        print(f"[평가] {name} 모델 ROC-AUC: {score:.4f}")
    
    weights_path = f"{MODEL_PATH}/blend_weights.pkl"
    if not force_retrain:
        best_weights, old_score = load_blend_weights(weights_path)
        if old_score > 0:
            print(f"[INFO] 기존 가중치 로드됨 (ROC-AUC: {old_score:.4f})")
    
    if force_retrain or old_score == 0:
        print("[INFO] 새로운 블렌딩 가중치 최적화 시작")
        best_weights = optimize_blend_ratios_kfold(models, X_train, y_train, n_splits=5)
        # 검증 세트에서의 성능 평가
        val_pred = weighted_blend_predict(models, X_val, best_weights)
        new_score = roc_auc_score(y_val, val_pred)
        
        # 기존 점수보다 더 좋은 경우에만 업데이트
        if new_score > old_score:
            save_blend_weights(weights_path, best_weights, new_score)
            print(f"✨ 성능 향상: {old_score:.4f} → {new_score:.4f}")
        else:
            print(f"ℹ️ 기존 가중치 유지 (기존: {old_score:.4f} >= 새로운: {new_score:.4f})")
    
    print("\n=== 검증 데이터에 대한 앙상블 예측 수행 ===")
    final_pred = weighted_blend_predict(models, X_val, best_weights)
    blend_roc_auc = roc_auc_score(y_val, final_pred)
    scores["ensemble_blend"] = blend_roc_auc
    print(f"[평가] 앙상블 블렌딩 모델 ROC-AUC: {blend_roc_auc:.4f}")
    
    print("\n=== 테스트 데이터에 대한 최종 예측 및 제출 파일 변환 ===")
    final_test_pred = weighted_blend_predict(models, X_test, best_weights)
    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": final_test_pred
    })
    
    submission.to_csv("blend_prediction.csv", index=False)
    if os.path.exists("blend_prediction.csv"):
        print(f"[SUCCESS] 'blend_prediction.csv' 파일이 정상적으로 생성되었습니다: {os.path.abspath('blend_prediction.csv')}")
    else:
        print("[ERROR] 'blend_prediction.csv' 파일 생성 실패")
        
    print("=== 📋 블렌딩 파이프라인 종료 ===")
    return submission, scores, best_weights

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

    # train 데이터에서 'ID' 컬럼이 있으면 제거
    if "ID" in train.columns:
        train = train.drop(columns=["ID"])
    
    # 학습 데이터와 타겟, 테스트 데이터 및 ID 분리
    X = train.drop(columns=["임신 성공 여부"])
    y = train["임신 성공 여부"]
    
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

def main():
    from model_prediction_efficient import load_and_preprocess_data
    print("블렌딩 모델 실행")
    print("사용 모델: XGBoost(xgb), LightGBM(lgb), CatBoost(cat)")
    X_train, X_test, y_train, test_ids = load_and_preprocess_data()
    
    # force_retrain 옵션 처리 예시
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force retrain models and reoptimize weights')
    args = parser.parse_args()
    
    submission, scores, weights = run_blending_pipeline(
        X_train, y_train, X_test, test_ids, 
        force_retrain=args.force_retrain
    )
    if submission is not None:
        print("\n=== 최종 성능 요약 ===")
        print("-" * 40)
        print("모델명".ljust(15), "ROC-AUC".rjust(10))
        print("-" * 40)
        for model, score in scores.items():
            print(f"{model.ljust(15)} {score:.4f}".rjust(15))
        print("-" * 40)

if __name__ == "__main__":
    main()

print(">> [model_ensemble] 파일 실행 시작")

import pandas as pd
import numpy as np
import os
import joblib

# Step 1: 개별 모델 예측 결과 파일 불러오기 (예시)
print(">> [앙상블] 개별 모델 예측값 로드 시작")
lightgbm_preds = pd.read_csv("open/lightgbm_oof_predictions.csv")
catboost_preds = pd.read_csv("open/catboost_oof_predictions.csv")
# 새로 추가: xgboost OOF 예측값 로드 (미리 생성하여 'open/xgboost_oof_predictions.csv'에 저장)
xgboost_preds = pd.read_csv("open/xgboost_oof_predictions.csv")
print("  [앙상블] 예측값 파일 로드 완료")

# 새로 추가: 저장된 블렌딩 가중치 로드 (없으면 단순 평균 처리)
weights_path = "open/models/blend_weights.pkl"
if os.path.exists(weights_path):
    blend_weights = joblib.load(weights_path)
    print("✓ 블렌딩 가중치 로드됨:", blend_weights)
else:
    print("❌ 블렌딩 가중치 파일 없음, 단순 평균 사용")
    blend_weights = {"lightgbm": 0.33, "catboost": 0.33, "xgboost": 0.34}

# Step 2: 가중치를 적용한 앙상블 예측 수행
print(">> [앙상블] 가중치 적용 앙상블 수행")
ensemble_preds = blend_weights.get("lightgbm", 0) * lightgbm_preds['oof_predictions'] \
                + blend_weights.get("catboost", 0) * catboost_preds['oof_predictions'] \
                + blend_weights.get("xgboost", 0) * xgboost_preds['oof_predictions']
ensemble_df = pd.DataFrame({
    'true_values': lightgbm_preds['true_values'],
    'ensemble_predictions': ensemble_preds
})
print("  [앙상블] 최종 앙상블 완료")

# Step 3: 앙상블 결과 저장
submission = pd.DataFrame({
    "ID": test_ids,              # 테스트 ID (예: "TEST_00000", "TEST_00001", ...)
    "probability": ensemble_preds
})
submission.to_csv("open/ensemble_predictions.csv", index=False)
print(f">> [앙상블] 앙상블 결과 저장 완료: {os.path.abspath('open/ensemble_predictions.csv')}")
print(">> [model_ensemble] 파일 실행 종료")

