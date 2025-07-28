import logging
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

class ModelTraining:
    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        logging.info("Starting data splitting.")
        X = df.drop(columns=self.config['data']['target'], errors='ignore')
        y = df[self.config['data']['target']]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=self.config['splitting']['train_size'], random_state=self.config['splitting']['random_seed']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config['splitting']['test_size'], random_state=self.config['splitting']['random_seed']
        )
        logging.info("Data split into training, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        logging.info("Training and evaluating baseline models.")
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=self.config['models']['Ridge']['alpha']),
            'Lasso': Lasso(alpha=self.config['models']['Lasso']['alpha'], max_iter=self.config['models']['Lasso']['max_iter']),
            'ElasticNet': ElasticNet(alpha=self.config['models']['ElasticNet']['alpha'], l1_ratio=self.config['models']['ElasticNet']['l1_ratio'])
        }
        pipelines = {}
        metrics = {}
        for model_name, model in models.items():
            if self.config['models'][model_name]['enabled']:
                pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('regressor', model)
                ])
                pipeline.fit(X_train, y_train)
                pipelines[model_name] = pipeline
                metrics[model_name] = self._evaluate_model(pipeline, X_val, y_val, model_name)
        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {
            'RidgeTuned': Ridge(),
            'ElasticNetTuned': ElasticNet()
        }
        tuned_pipelines = {}
        tuned_metrics = {}
        param_grid = {
            'regressor__alpha': self.config['tuning']['GridSearchCV']['parameters']['alpha'],
            'regressor__fit_intercept': self.config['tuning']['GridSearchCV']['parameters']['fit_intercept']
        }
        cv = self.config['tuning']['GridSearchCV']['folds']
        scoring = self.config['tuning']['GridSearchCV']['scoring']
        for model_name, model in tuned_models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            if self.config['tuning']['GridSearchCV']['enabled']:
                grid_search = GridSearchCV(
                    pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                tuned_pipelines[model_name] = grid_search.best_estimator_
                tuned_metrics[model_name] = self._evaluate_model(
                    grid_search.best_estimator_, X_val, y_val, model_name
                )
                logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
            if self.config['tuning']['RandomizedSearchCV']['enabled']:
                random_search = RandomizedSearchCV(
                    pipeline, param_grid, n_iter=self.config['tuning']['RandomizedSearchCV']['iterations'],
                    cv=self.config['tuning']['RandomizedSearchCV']['folds'],
                    scoring=self.config['tuning']['RandomizedSearchCV']['scoring'],
                    random_state=self.config['tuning']['RandomizedSearchCV']['random_seed'],
                    n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                tuned_pipelines[model_name + '_Random'] = random_search.best_estimator_
                tuned_metrics[model_name + '_Random'] = self._evaluate_model(
                    random_search.best_estimator_, X_val, y_val, model_name + '_Random'
                )
                logging.info(f"Best params for {model_name}_Random: {random_search.best_params_}")
        logging.info("Hyperparameter tuning completed.")
        return tuned_pipelines, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        y_test_pred = model.predict(X_test)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R2': r2_score(y_test, y_test_pred)
        }
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.2f}")
        return metrics

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        y_val_pred = model.predict(X_val)
        metrics = {
            'MAE': mean_absolute_error(y_val, y_val_pred),
            'MSE': mean_squared_error(y_val, y_val_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'R2': r2_score(y_val, y_val_pred)
        }
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.2f}")
        return metrics