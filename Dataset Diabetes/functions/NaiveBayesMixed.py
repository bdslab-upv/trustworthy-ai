from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB

class NaiveBayesMixed(BaseEstimator, ClassifierMixin):
    """
    Mixed Naive Bayes classifier.

    This classifier combines Gaussian Naive Bayes for numerical features
    and Categorical Naive Bayes for categorical features.

    Parameters:
    ----------
    feat_num : list of column names corresponding to numerical features.
    feat_cat : list of column names corresponding to categorical features.
    smoothing : float, optional (default=1e-09)
        Portion of the largest variance of all features added to variances for calculation stability.
    alpha_value : float, optional (default=1)
        Additive (Laplace/Lidstone) smoothing parameter for categorical features.

    Attributes:
    ----------
    gaussian_nb : GaussianNB
        Gaussian Naive Bayes model for numerical features.
    categorical_nb : CategoricalNB
        Categorical Naive Bayes model for categorical features.
    classes_ : array, shape (n_classes,)
        Array of class labels.
    """

    def __init__(self, feat_num, feat_cat, var_smoothing=1e-09, alpha=1):
        self.feat_num = feat_num
        self.feat_cat = feat_cat
        self.var_smoothing = var_smoothing
        self.alpha = alpha

        self.gaussian_nb = GaussianNB(var_smoothing=self.var_smoothing)
        self.categorical_nb = CategoricalNB(alpha= self.alpha)
        self.classes_ = None  # Unknown classes until 'fit' method is called

    def fit(self, features, tags, sample_weight=None):
        """
        Fit the model according to the given training data.

        Parameters:
        ----------
        features : pandas DataFrame, shape (n_samples, n_features)
            Training vectors.
        tags : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.

        Returns:
        -------
        self : object
            Returns self.
        """
        numerical_features = features[self.feat_num]
        categorical_features = features[self.feat_cat]

        self.gaussian_nb.fit(numerical_features, tags, sample_weight=sample_weight)
        self.categorical_nb.fit(categorical_features, tags, sample_weight=sample_weight)
        self.classes_ = np.unique(tags)  # Store the classes

        return self

    def predict(self, features):
        """
        Perform classification on an array of test vectors.

        Parameters:
        ----------
        features : pandas DataFrame, shape (n_samples, n_features)
            Test vectors.

        Returns:
        -------
        tags_pred : array, shape (n_samples,)
            Predicted target values for the provided data.
        """
        return self.categorical_nb.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def predict_proba(self, features):
        """
        Return probability estimates for the test vector.

        Parameters:
        ----------
        features : pandas DataFrame, shape (n_samples, n_features)
            Test vectors.

        Returns:
        -------
        proba : array, shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model.
        """
        proba_continuous = self.gaussian_nb.predict_proba(features[self.feat_num])
        proba_categorical = self.categorical_nb.predict_proba(features[self.feat_cat])
        return proba_continuous * proba_categorical
