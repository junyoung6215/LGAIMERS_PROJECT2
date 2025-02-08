import os
import pickle
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 모델 라이브러리 임포트
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from load import load_and_preprocess_data

def optimize_model(model_name, X, y):
    best_score = -np.inf
    best_params = None
    
    def objective(trial):
        try:
            if model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'depth': trial.suggest_int('depth', 4, 6),
                    'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 5),
                    'verbose': 0,
                    'random_state': 42
                }
                model = CatBoostClassifier(**params)
            elif model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                model = XGBClassifier(**params)
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 62),
                    'random_state': 42
                }
                model = LGBMClassifier(**params)
            else:
                raise ValueError("정의되지 않은 모델입니다.")
            
            # 3-fold CV로 ROC-AUC 평가
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            return scores.mean()
        
        except Exception as e:
            print(f"{model_name} 최적화 중 에러 발생: {e}")
            return -np.inf

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best_score = study.best_value
    best_params = study.best_trial.params
    print(f"{model_name} 최적화 완료: best_score={best_score}, best_params={best_params}")
    return best_params, best_score

def run_optimization():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    if X_train is None:
        print("데이터 로드 실패")
        return
    
    models = ['CatBoost', 'RandomForest', 'XGBoost', 'LightGBM']
    results = {}
    
    for model_name in models:
        try:
            print(f"\n{model_name} 모델 최적화 시작...")
            params, score = optimize_model(model_name, X_train, y_train)
            results[model_name] = {'params': params, 'score': score}
        except Exception as e:
            print(f"{model_name} 모델 최적화 중 에러: {e}")
    
    # 결과를 open 폴더에 저장 (폴더 없으면 생성)
    os.makedirs('open', exist_ok=True)
    with open('open/best_model_params.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("최적화 결과가 open/best_model_params.pkl 파일에 저장되었습니다.")

if __name__ == "__main__":
    run_optimization()
