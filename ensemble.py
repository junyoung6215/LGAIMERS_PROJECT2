import os
import glob
import pandas as pd

def ensemble_predictions():
    try:
        prediction_files = glob.glob("open/*_prediction.csv")
        if not prediction_files:
            print("예측 결과 파일이 하나도 발견되지 않았습니다.")
            return
        
        dfs = []
        for file in prediction_files:
            try:
                df = pd.read_csv(file)
                if not {'ID', 'probability'}.issubset(df.columns):
                    print(f"경고: {file} 파일에 필요한 컬럼이 없습니다.")
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"{file} 파일 로드 중 에러: {e}")
        
        if not dfs:
            print("사용 가능한 예측 결과 파일이 없습니다.")
            return
        
        # 모든 파일을 ID를 기준으로 병합 후 확률 단순 평균
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = merged_df.merge(df, on='ID', suffixes=('', '_dup'))
            # 예측 확률 병합 시 중복 컬럼 처리: 두 컬럼의 평균을 구하는 예제
            merged_df['probability'] = merged_df[['probability', 'probability_dup']].mean(axis=1)
            merged_df.drop('probability_dup', axis=1, inplace=True)
        
        # (옵션) 가중 평균이나 다수결 투표 방식에 대한 옵션을 추가할 수 있음
        
        # 최종 결과 저장
        output_path = "open/ensemble_prediction.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"앙상블 예측 결과가 {output_path} 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"앙상블 예측 중 에러 발생: {e}")

if __name__ == "__main__":
    ensemble_predictions()
