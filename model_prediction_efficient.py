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

def clean_params(params):
    """파라미터에서 불필요한 키 제거"""
    if isinstance(params, dict):
        if "params" in params:
            model_params = params["params"]
        else:
            model_params = params.copy()
        model_params.pop("roc_auc", None)
        model_params.pop("score", None)
        return model_params
    return params

def load_model_params(model_name):
    """각 모델별 파라미터 로드"""
    # 모델명과 파일명 매핑
    file_name_mapping = {
        "xgb": "xgb",
        "lgb": "lgbm",  # lightgbm도 처리
        "cat": "catboost",  # catboost로 수정
        "rf": "rf"
    }
    
    try:
        if model_name == "lgb":
            # LightGBM은 두 가지 파일명 시도
            try:
                params = joblib.load('open/best_lgbm_params.pkl')
            except:
                params = joblib.load('open/best_lightgbm_params.pkl')
        else:
            # 매핑된 파일명 사용
            mapped_name = file_name_mapping.get(model_name, model_name)
            params = joblib.load(f'open/best_{mapped_name}_params.pkl')
        
        print(f"✅ {model_name} 파라미터 로드 성공")
        return clean_params(params)
    except Exception as e:
        print(f"❌ {model_name} 파라미터 로드 실패 (파일: best_{file_name_mapping.get(model_name, model_name)}_params.pkl)")
        print(f"  에러 메시지: {str(e)}")
        return None

def print_model_performance_table(scores):
    """모델별 성능을 테이블 형태로 출력"""
    print("\n=== 📊 모델별 성능 비교 ===")
    print("-" * 45)
    print(f"{'모델명'.ljust(15)} {'검증 ROC-AUC'.rjust(15)} {'테스트 ROC-AUC'.rjust(15)}")
    print("-" * 45)
    for model_name, score_dict in scores.items():
        val_score = score_dict.get('val_score', 0)
        test_score = score_dict.get('test_score', 0)
        print(f"{model_name.ljust(15)} {f'{val_score:.4f}'.rjust(15)} {f'{test_score:.4f}'.rjust(15)}")
    print("-" * 45)

def train_and_predict(model_class, params, X_train, y_train, X_test, model_name):
    """개별 모델 학습/로드 및 예측"""
    print(f"\n=== {model_name} 모델 처리 시작 ===")
    
    model_path = f"{MODEL_PATH}/{model_name}_model.pkl"
    performance_path = f"{MODEL_PATH}/{model_name}_performance.pkl"
    
    if os.path.exists(model_path):
        print(f"[{model_name}] ℹ️ 저장된 모델 발견, 로드 중...")
        try:
            model = joblib.load(model_path)
            if os.path.exists(performance_path):
                performance = joblib.load(performance_path)
                print(f"[{model_name}] 📈 저장된 성능 지표:")
                print(f"    - 검증 ROC-AUC: {performance.get('val_score', 0):.4f}")
                print(f"    - 테스트 ROC-AUC: {performance.get('test_score', 0):.4f}")
            return model, performance
        except Exception as e:
            print(f"[{model_name}] ⚠️ 모델 로드 실패: {str(e)}")
            print(f"[{model_name}] 🔄 새로 학습을 시작합니다.")
    else:
        print(f"[{model_name}] ℹ️ 저장된 모델 없음, 새로 학습합니다.")
    
    # 모델 학습
    if model_name == "xgb":
        model = model_class(**params, use_label_encoder=False, eval_metric="auc")
    elif model_name == "cat":
        model = model_class(**params, verbose=False, eval_metric="AUC")
    else:
        model = model_class(**params)
    
    model.fit(X_train, y_train)
    
    # 모델 저장
    try:
        joblib.dump(model, model_path)
        print(f"[{model_name}] ✅ 모델 저장 완료")
    except Exception as e:
        print(f"[{model_name}] ⚠️ 모델 저장 실패: {str(e)}")
    
    return model, None

def main():
    print("\n🚀 개별 모델 예측 파이프라인 시작")
    
    # 데이터 로드 및 전처리
    X, X_test, y, test_ids = load_and_preprocess_data()
    if X is None:
        return
    
    # 검증용 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_config = {
        "xgb": (xgb.XGBClassifier, "xgb"),
        "lgb": (lgb.LGBMClassifier, "lgb"),
        "cat": (CatBoostClassifier, "cat"),
        "rf": (RandomForestClassifier, "rf")
    }
    
    submissions = {}
    scores = {}
    
    for model_key, (model_class, param_key) in models_config.items():
        try:
            params = load_model_params(param_key)
            if params is None:
                print(f"⚠️ {model_key} 모델 스킵: 파라미터 로드 실패")
                continue
            
            # 모델 학습/로드 및 성능 지표 확인
            model, performance = train_and_predict(
                model_class, params, X_train, y_train, X_test, model_key
            )
            
            # 검증 세트 예측 및 성능 평가
            val_pred = model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val)
            
            # 테스트 세트 예측
            test_pred = model.predict_proba(X_test)[:, 1]
            
            # 성능 저장
            model_scores = {
                'val_score': val_score,
                'test_score': 0  # 실제 테스트 레이블이 없으므로 0으로 설정
            }
            scores[model_key] = model_scores
            
            # 성능 지표 파일 저장
            joblib.dump(model_scores, f"{MODEL_PATH}/{model_key}_performance.pkl")
            
            # 제출 파일 생성
            submission = pd.DataFrame({
                "ID": test_ids,
                "probability": test_pred
            })
            output_path = f"submission_{model_key}.csv"
            submission.to_csv(output_path, index=False)
            print(f"✅ {model_key} 제출 파일 생성 완료: {output_path}")
            
            submissions[model_key] = submission
            
        except Exception as e:
            print(f"❌ {model_key} 모델 처리 중 오류 발생: {str(e)}")
    
    # 모델별 성능 비교 테이블 출력
    print_model_performance_table(scores)
    
    return submissions, scores

if __name__ == "__main__":
    main()