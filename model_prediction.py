import os
import pickle
import pandas as pd

# 모델 라이브러리 임포트
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_best_model_params():
    if not os.path.exists('open/best_model_params.pkl'):
        raise FileNotFoundError("open/best_model_params.pkl 파일을 찾을 수 없습니다.")
    with open('open/best_model_params.pkl', 'rb') as f:
        params_dict = pickle.load(f)
    return params_dict

def select_best_model(params_dict):
    # ROC-AUC 점수가 가장 높은 모델 선택
    best_model = None
    best_score = -1
    for model_name, info in params_dict.items():
        if info['score'] > best_score:
            best_score = info['score']
            best_model = (model_name, info['params'])
    print(f"선택된 모델: {best_model[0]}, 파라미터: {best_model[1]}")
    return best_model

def train_and_predict():
    try:
        # 최적화 결과 로드 및 모델 선택
        params_dict = load_best_model_params()
        model_name, best_params = select_best_model(params_dict)
        
        # 학습 데이터 로드 (preprocessed_data.csv)
        df = pd.read_csv('preprocessed_data.csv')
        # 'ID' 컬럼이 있다면 분리 (전처리 시 제거되었을 수 있음)
        if 'ID' in df.columns:
            df = df.set_index('ID')
        if 'target' not in df.columns:
            raise ValueError("target 컬럼이 존재하지 않습니다.")
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 모델 인스턴스 생성 (각 라이브러리의 클래스 사용, random_state=42 고정)
        if model_name == 'CatBoost':
            model = CatBoostClassifier(**best_params, verbose=0, random_state=42)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**best_params, random_state=42)
        elif model_name == 'XGBoost':
            model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_name == 'LightGBM':
            model = LGBMClassifier(**best_params, random_state=42)
        else:
            raise ValueError("선택된 모델이 올바르지 않습니다.")
        
        print("모델 학습을 시작합니다...")
        model.fit(X, y)
        print("모델 학습이 완료되었습니다.")

        # 테스트 데이터 로드: open/test.csv (반드시 'ID' 컬럼 포함)
        if not os.path.exists('open/test.csv'):
            raise FileNotFoundError("open/test.csv 파일을 찾을 수 없습니다.")
        test_df = pd.read_csv('open/test.csv')
        if 'ID' not in test_df.columns:
            raise ValueError("테스트 파일에 'ID' 컬럼이 포함되어 있지 않습니다.")
        test_ids = test_df['ID']
        X_test = test_df.drop('ID', axis=1)
        
        # predict_proba() 로 확률 예측 (클래스 1의 확률)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # 예측 결과 저장: open/{model이름}_prediction.csv
        result_df = pd.DataFrame({'ID': test_ids, 'probability': probabilities})
        output_path = f"open/{model_name}_prediction.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"예측 결과가 {output_path} 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    train_and_predict()
