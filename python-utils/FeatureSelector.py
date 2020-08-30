from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A pipeline-composable transformer for selecting features based on the datatype
    """
    def __init__(self, dtype='numeric'):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.dtype == 'numeric':
            num_cols = X.columns[X.dtypes != object].tolist()
            return X[num_cols]
        elif self.dtype == 'category':
            cat_cols = X.columns[X.dtypes == object].tolist()
            return X[cat_cols]
