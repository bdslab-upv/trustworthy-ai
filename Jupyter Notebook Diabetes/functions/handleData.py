import pandas as pd
import numpy as np
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class handleData(BaseEstimator, TransformerMixin):
    def __init__(self, feat_num, feat_cat):
        self.feat_num = feat_num
        self.feat_cat = feat_cat
        
    
    def encode(self, X):

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.feat_num),  # 'passthrough' significa que las características numéricas no serán transformadas
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.feat_cat)
            ])

        X_encoded = preprocessor.fit_transform(X)
        feat_cat_encoded = list(preprocessor.named_transformers_['cat'].get_feature_names_out(self.feat_cat))
        X_names_encoded = self.feat_num + feat_cat_encoded

        X_encoded = pd.DataFrame(X_encoded, columns=X_names_encoded, index=X.index)

        return X_encoded, X_names_encoded, feat_cat_encoded
    

    def decode(self, X_encoded):

        encoded_cat_columns = [col for col in X_encoded.columns if col not in self.feat_num]
        X = pd.DataFrame(index=X_encoded.index)

        for original_column in self.feat_cat:
            encoded_cols = [col for col in encoded_cat_columns if col.startswith(original_column)]
            X[original_column] = X_encoded[encoded_cols].idxmax(axis=1).str.replace(original_column + '_', '')

        X = pd.concat([X_encoded[self.feat_num], X], axis=1)

        return X
    

    def get_random_samples(self, X, Y):
        unique_tags = pd.unique(Y)
        random_indexes = {tags: [] for tags in unique_tags}

        for tag in unique_tags:
            rdm_idx = random.choice(Y[Y == tag].index)
            rdm_feat = X.loc[rdm_idx].values.reshape(1, -1) # Make sure it is a 2D array
            rdm_feat = pd.DataFrame(rdm_feat, columns=X.columns)
            random_indexes[tag]=rdm_feat

        return random_indexes
    

    def perturb_sample(self, sample, X, times):
      
        num_std_devs = X[self.feat_num].std()  # Standard deviation for numerical features
        categories = {feat: X[feat].unique() for feat in self.feat_cat}  # Unique categories for categorical features
        perturbed_samples = pd.DataFrame(index=range(times), columns= sample.index.tolist())

        for i in range(times):
            perturbed_sample = sample.copy()
            for feat in self.feat_num:
                perturbed_sample[feat] += np.random.normal(0, num_std_devs[feat])  # Perturb numerical data based on std deviation
            for feat in self.feat_cat:
                perturbed_sample[feat] = np.random.choice(categories[feat])  # Perturb categorical data
            perturbed_samples.loc[i] = perturbed_sample
            
        return perturbed_samples
    
    def group_feature_importanceT(self, feature_names, feature_importance, feat_cat_encoded):
        grouped_feature_importance = {}

        for feature_name, importance in zip(feature_names, feature_importance):
            main_feature, class_label = feature_name.rsplit('_', 1) if '_' in feature_name and feature_name in feat_cat_encoded else (feature_name, None)
            grouped_feature_importance.setdefault(main_feature, {}).setdefault(class_label, 0)
            grouped_feature_importance[main_feature][class_label] += importance

        summed_grouped_features = {main_feature: np.sum(list(class_importances.values())) for main_feature, class_importances in grouped_feature_importance.items()}

        sorted_grouped_features = sorted(summed_grouped_features.items(), key=lambda x: x[1], reverse=True)

        return sorted_grouped_features

    def group_feature_importanceD(self, feature_names, feature_importance, feat_cat_encoded):
        grouped_feature_importance = {}

        for feature_name, importance in zip(feature_names, feature_importance):
            main_feature, class_label = feature_name.rsplit('_', 1) if '_' in feature_name and feature_name in feat_cat_encoded else (feature_name, None)
            grouped_feature_importance.setdefault(main_feature, {}).setdefault(class_label, 0)
            grouped_feature_importance[main_feature][class_label] += importance


        return grouped_feature_importance