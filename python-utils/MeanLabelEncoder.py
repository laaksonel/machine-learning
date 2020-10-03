from sklearn.base import BaseEstimator, TransformerMixin

"""
Mean label encoder.

Replaces categorical columns with the mean label value

"""
class MeanLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y):
        if self.cols is None:
            self.cols = X.select_dtypes(include='object').columns

        for col in self.cols:
            if col not in X:
                raise ValueError('Uknown column: \'' + col + '\')

        self.means = dict()

        for col in self.cols:
            tmp = dict()
            uniques = X[col].unique()
            for u in uniques:
                tmp[u] = y[X[col] == u].mean()
            self.means[col] = tmp

        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for col, means in self.means.items():
            vals = np.full(X.shape[0], np.nan)
            for val, label_mean in means.items():
                vals[X[col] == val] = label_mean
            X_[col] = vals
        return X_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
