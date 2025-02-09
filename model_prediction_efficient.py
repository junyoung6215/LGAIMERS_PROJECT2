import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc  # Garbage Collector
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """데이터 로드 및 전처리 함수"""
    print("1. 데이터 로드 시작")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # ID 컬럼 분리
    test_ids = test['id'] if 'id' in test.columns else pd.Series([f"TEST_{i:05d}" for i in range(len(test))])
    
    # 타겟 분리
    y_train = train['임신 성공 여부']
    X_train = train.drop(['임신 성공 여부', 'id'] if 'id' in train.columns else ['임신 성공 여부'], axis=1)
    X_test = test.drop(['id'] if 'id' in test.columns else [], axis=1)
    
    # 메모리 정리
    del train, test
    gc.collect()
    
    print("2. 전처리 시작")
    # 범주형 변수 처리 - Label Encoding 사용
    le_dict = {}
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        # 결측치는 'missing'으로 대체
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
        
        # train과 test의 unique 값을 합쳐서 인코딩
        le.fit(pd.concat([X_train[col], X_test[col]]).unique())
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        le_dict[col] = le
    
    # 수치형 변수 결측치 처리
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    print("3. 전처리 완료")
    return X_train, X_test, y_train, test_ids

def predict_with_model(model_name, model_params, X_train, X_test, y_train, test_ids):
    """단일 모델 학습 및 예측"""
    print(f"\n4. {model_name} 모델 학습 시작")
    
    try:
        # 검증용 데이터 분할
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        if model_name == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBClassifier(**model_params, use_label_encoder=False)
        elif model_name == "LightGBM":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**model_params)
        elif model_name == "CatBoost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**model_params, verbose=False)
        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params)
        
        # 학습
        model.fit(X_train_fit, y_train_fit)
        
        # 검증 데이터로 ROC-AUC 계산
        val_predictions = model.predict_proba(X_val)[:, 1]
        roc_score = roc_auc_score(y_val, val_predictions)
        print(f"검증 데이터 ROC-AUC 점수 ({model_name}): {roc_score:.4f}")
        
        # 전체 학습 데이터로 재학습
        model.fit(X_train, y_train)
        
        # 테스트 데이터 예측
        print(f"5. {model_name} 예측 시작")
        predictions = model.predict_proba(X_test)[:, 1]
        
        # 결과 저장
        submission = pd.DataFrame({
            "ID": test_ids,
            "probability": predictions
        })
        
        output_file = f"{model_name}_prediction.csv"
        submission.to_csv(output_file, index=False)
        print(f"6. 결과 저장 완료: {output_file}")
        
        # 메모리 정리
        del model, predictions, submission
        gc.collect()
        
        return True, roc_score
        
    except Exception as e:
        print(f"Error with {model_name}: {str(e)}")
        return False, 0.0

def main():
    # 데이터 로드 및 전처리
    X_train, X_test, y_train, test_ids = load_and_preprocess_data()
    
    # 모델별 실행 및 점수 저장
    models = {
        "XGBoost": "open/best_xgb_params.pkl",
        "LightGBM": "open/best_lgbm_params.pkl",
        "CatBoost": "open/best_catboost_params.pkl",
        "RandomForest": "open/best_rf_params.pkl"
    }
    
    model_scores = {}
    
    for model_name, param_file in models.items():
        print(f"\n===== {model_name} 모델 실행 =====")
        try:
            params = joblib.load(param_file)
            success, roc_score = predict_with_model(model_name, params, X_train, X_test, y_train, test_ids)
            if success:
                print(f"{model_name} 완료!")
                model_scores[model_name] = roc_score
            gc.collect()
        except Exception as e:
            print(f"{model_name} 실패: {str(e)}")
    
    # 최고 성능 모델 출력
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1])
        print("\n===== 모델 성능 비교 =====")
        for model, score in model_scores.items():
            print(f"{model}: ROC-AUC = {score:.4f}")
        print(f"\n🏆 최고 성능 모델: {best_model[0]} (ROC-AUC: {best_model[1]:.4f})")
        
        # 점수 저장
        scores_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'ROC_AUC'])
        scores_df.to_csv('model_scores.csv', index=False)
        print(f"모델 성능 비교 결과가 model_scores.csv 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
