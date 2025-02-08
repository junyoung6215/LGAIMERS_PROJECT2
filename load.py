import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    try:
        # 파일 존재 여부 확인
        if not os.path.exists('preprocessed_data.csv'):
            raise FileNotFoundError("preprocessed_data.csv 파일을 찾을 수 없습니다.")
        
        # 파일 로딩
        df = pd.read_csv('preprocessed_data.csv')
        print("preprocessed_data.csv 파일을 성공적으로 로드했습니다.")
        
        # 'target' 컬럼 존재 여부 확인 및 타입 변환
        if 'target' not in df.columns:
            raise ValueError("target 컬럼이 존재하지 않습니다.")
        df['target'] = df['target'].astype(int)
        
        # X(특성)와 y(타겟) 분리
        X = df.drop('target', axis=1)
        y = df['target']
        print("데이터와 타겟이 성공적으로 분리되었습니다.")
        
        # train/test 데이터 분할 (80%/20%, random_state=42 고정)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("train/test 데이터 분할이 완료되었습니다.")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"에러 발생: {e}")
        return None, None, None, None

if __name__ == "__main__":
    load_and_preprocess_data()
