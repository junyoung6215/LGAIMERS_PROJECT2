import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    print("Step 1: 데이터 로드")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"파일 로드 성공: {input_file}")
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='cp949')
        print(f"파일 로드 성공 (cp949 인코딩): {input_file}")
    print(f"초기 데이터 shape: {df.shape}\n")

    print("Step 2: 데이터 타입 정제")
    # object 형식인 컬럼 중, 숫자로 변환 가능한 것은 변환 시도
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"  [변환] {col} -> numeric")
            except Exception:
                print(f"  [유지] {col} remains as object")
    print("데이터 타입 정제 완료.\n")
    
    print("Step 3: 결측치 분석 및 처리")
    missing_percent = df.isnull().mean() * 100
    low_missing_cols = missing_percent[missing_percent < 3].index.tolist()
    high_missing_cols = missing_percent[missing_percent > 90].index.tolist()
    print("  결측치 비율 <3% 컬럼:", low_missing_cols)
    print("  결측치 비율 >90% 컬럼:", high_missing_cols)

    # 매우 높은 결측치 비율 (>90%) 컬럼 제거
    if high_missing_cols:
        print("  [제거] 결측치 비율이 매우 높은 컬럼 제거:", high_missing_cols)
        df.drop(columns=high_missing_cols, inplace=True)

    # 나머지 결측치 처리: indicator 컬럼 추가 후, numeric은 중앙값, 그 외는 최빈값 대체
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            indicator_col = col + "_missing"
            df[indicator_col] = df[col].isnull().astype(int)
            print(f"  [indicator 추가] {col} -> {indicator_col}")
            if np.issubdtype(df[col].dtype, np.number):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    [대체] {col}: 중앙값 {median_val}으로 대체")
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col].fillna(mode_val, inplace=True)
                print(f"    [대체] {col}: 최빈값 '{mode_val}'으로 대체")
    print("결측치 처리 완료.\n")

    print("Step 4: 이상치 처리")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # 이진 변수는 outlier 처리를 건너뜀
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            print(f"  [건너뜀] 이진 변수 {col}은 이상치 처리 생략")
        else:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            df[col] = df[col].clip(lower, upper)
            print(f"  [이상치 처리] {col}: {outliers}건 클리핑 [{lower}, {upper}] 적용")
    print("이상치 처리 완료.\n")

    print("Step 5: 범주형/텍스트 변수 처리")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        # 불필요한 공백 제거, 소문자 변환 등 기본 정제
        df[col] = df[col].astype(str).str.strip().str.lower()
        print(f"  [정제] {col} 문자열 정제 (공백 제거, 소문자 변환)")
    print("범주형/텍스트 변수 처리 완료.\n")
    
    print("Step 6: 피처 파이프라인 통합 및 최종 데이터 확인")
    print(f"최종 데이터 shape: {df.shape}\n")

    print(f"Step 7: 전처리된 데이터 저장 -> {output_file}")
    # 모든 NaN 값을 빈 문자열로 대체해 비정상적 문자가 포함되는 문제 해결
    df.fillna("", inplace=True)
    # 필요에 따라 인코딩 옵션 변경: 기본 utf-8-sig, 또는 한글 포함 시 cp949, 기본 utf-8 등 선택 가능
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("데이터 전처리 완료.")

if __name__ == "__main__":
    preprocess_data('train.csv', 'preprocessed_data.csv')
