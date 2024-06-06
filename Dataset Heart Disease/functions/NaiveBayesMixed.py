from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB

class NaiveBayesMixed(BaseEstimator, ClassifierMixin):
    """
    Mixed Naive Bayes classifier.

    This classifier combines Gaussian Naive Bayes for numerical features
    and Categorical Naive Bayes for categorical features. It preprocesses
    the categorical features using one-hot encoding.

    Parameters:
    ----------
    feat_num : list of column names corresponding to numerical features.
    feat_cat : list of column names corresponding to categorical features.

    Attributes:
    ----------
    gaussian_nb : Gaussian Naive Bayes model for numerical features.
    multinomial_nb : Categorical Naive Bayes model for categorical features.
    preprocessor : Preprocessor for categorical features, performing one-hot encoding.
    classes_ : Array of class labels.
    """

    def __init__(self, feat_num, feat_cat):
        self.feat_num = feat_num
        self.feat_cat = feat_cat

        self.gaussian_nb = GaussianNB()
        self.categorical_nb = CategoricalNB()
        self.classes_ = None  # Unknown classes until 'fit' method is called

    def fit(self, features, tags, sample_weight=None):
        numerical_features = features[self.feat_num]
        categorical_features = features[self.feat_cat]

        self.gaussian_nb.fit(numerical_features, tags, sample_weight=sample_weight)
        self.categorical_nb.fit(categorical_features, tags, sample_weight=sample_weight)
        self.classes_ = np.unique(tags) # Store the classes

        return self

    def predict(self, features):
        return self.categorical_nb.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def predict_proba(self, features):

        proba_continuous = self.gaussian_nb.predict_proba(features[self.feat_num])
        proba_categorical = self.categorical_nb.predict_proba(features[self.feat_cat])
        return proba_continuous * proba_categorical