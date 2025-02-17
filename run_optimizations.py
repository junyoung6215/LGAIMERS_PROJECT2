import subprocess
import time

def run_optimization(script_name):
    try:
        print(f">>> {script_name} 실행 시작")
        print(f"   현재 실행 중인 optimization 파일: {script_name}")  # 추가된 출력 메시지
        subprocess.run(["python3", script_name], check=True)
        print(f">>> {script_name} 실행 완료\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 실행 중 오류 발생: {e}")

def main():
    # 실행할 스크립트 목록 (필요 시 파일 경로 수정)
    scripts = [
        "CATBoost_Optimization.py",
        "LightGBM_Optimization.py",
        "XGBoost_Optimization.py"
    ]
    
    # 전체 반복 횟수 (원하는 만큼 조정)
    iterations = 3
    
    for i in range(iterations):
        print(f"=================== Iteration {i+1} 시작 ===================")
        for script in scripts:
            run_optimization(script)
            # 각 스크립트 사이 5초 대기 (필요 시 조정)
            time.sleep(5)
        print(f"=================== Iteration {i+1} 완료 ===================\n")
        # 반복 사이 10초 대기 (필요 시 조정)
        time.sleep(10)

if __name__ == "__main__":
    main()
