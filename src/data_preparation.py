import logging
import re
from typing import Any, Dict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

class DataPreparation:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning.")
        df = df.drop_duplicates()
        columns_to_drop = self.config['data'].get('drop', [])
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        if 'flat_type' in df.columns:
            df['flat_type'] = df['flat_type'].replace('FOUR ROOM', '4 ROOM')
        if 'storey_range' in df.columns:
            df['storey_range'] = df['storey_range'].apply(self._convert_storey_range)
        if 'town_id' in df.columns and 'town_name' in df.columns:
            df = self._fill_missing_names(df, 'town_id', 'town_name')
        if 'flatm_id' in df.columns and 'flatm_name' in df.columns:
            df = self._fill_missing_names(df, 'flatm_id', 'flatm_name')
        if 'remaining_lease' in df.columns:
            df['remaining_lease_months'] = df['remaining_lease'].apply(self._extract_lease_info)
            df = df.drop(columns=['remaining_lease'], errors='ignore')
        logging.info("Data cleaning completed.")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data preprocessing.")
        X = df.drop(columns=[self.config['data']['target']], errors='ignore')
        X_transformed = self.preprocessor.fit_transform(X)
        feature_names = self._get_feature_names()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        if self.config['data']['target'] in df.columns:
            X_transformed_df[self.config['data']['target']] = df[self.config['data']['target']]
        logging.info("Data preprocessing completed.")
        return X_transformed_df

    def _create_preprocessor(self) -> ColumnTransformer:
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(
                categories=[self.config['processing']['ordinal']['flat_type']['order']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        numerical_features = self.config['processing']['numerical']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('nom', nominal_transformer, self.config['processing']['nominal']),
                ('ord', ordinal_transformer, ['flat_type']),
                ('pass', 'passthrough', self.config['processing']['passthrough'])
            ],
            remainder='drop',
            n_jobs=-1
        )
        return preprocessor

    def _get_feature_names(self) -> list:
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'nom':
                feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features))
            elif name == 'ord':
                feature_names.extend(features)
            elif name == 'pass':
                feature_names.extend(features)
        return feature_names

    @staticmethod
    def _convert_storey_range(storey_range: str) -> float:
        if pd.isna(storey_range):
            return 0.0
        match = re.search(r'(\d+)\s*TO\s*(\d+)', storey_range)
        if match:
            low, high = map(int, match.groups())
            return (low + high) / 2
        return 0.0

    @staticmethod
    def _fill_missing_names(df: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
        missing_names = df[name_column].isna()
        name_mapping = (
            df[[id_column, name_column]]
            .dropna()
            .drop_duplicates()
            .set_index(id_column)[name_column]
            .to_dict()
        )
        df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(name_mapping)
        return df

    @staticmethod
    def _extract_lease_info(lease_str: str) -> int:
        if pd.isna(lease_str):
            return None
        years_match = re.search(r'(\d+)\s*years?', lease_str)
        months_match = re.search(r'(\d+)\s*months?', lease_str)
        numbers_match = re.match(r'^\d+$', lease_str.strip())
        if years_match:
            years = int(years_match.group(1))
        elif numbers_match:
            years = int(numbers_match.group(0))
        else:
            years = 0
        months = int(months_match.group(1)) if months_match else 0
        total_months = years * 12 + months
        return total_months