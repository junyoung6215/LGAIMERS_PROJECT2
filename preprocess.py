import os
import pandas as pd
import numpy as np

def load_and_preprocess_csv():
    try:
        # 파일 존재 여부 확인
        if not os.path.exists('train.csv'):
            raise FileNotFoundError("train.csv 파일을 찾을 수 없습니다.")
        
        # 파일 로딩: utf-8 시도, 실패 시 cp949 재시도
        try:
            df = pd.read_csv('train.csv', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv('train.csv', encoding='cp949')
        print("train.csv 파일을 성공적으로 로드했습니다.")
        
        # 'ID' 컬럼 제거 (존재하면)
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
            print("ID 컬럼이 제거되었습니다.")
        
        # 수치형 결측치 처리: 각 컬럼의 중앙값으로 대체
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        print("수치형 컬럼의 결측치가 중앙값으로 대체되었습니다.")
        
        # (옵션) 범주형 결측치 처리: 최빈값 또는 'unknown' (필요 시)
        # categorical_cols = df.select_dtypes(include=['object']).columns
        # for col in categorical_cols:
        #     if df[col].isnull().any():
        #         df[col] = df[col].fillna('unknown')
        
        # 'target' 컬럼 존재 확인 및 int형 변환
        if 'target' not in df.columns:
            raise ValueError("target 컬럼이 존재하지 않습니다.")
        df['target'] = df['target'].astype(int)
        print("target 컬럼이 int형으로 변환되었습니다.")
        
        # 전처리 결과 저장 (인덱스 없이, utf-8-sig 인코딩)
        df.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')
        print("preprocessed_data.csv 파일이 성공적으로 저장되었습니다.")
        
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    load_and_preprocess_csv()
