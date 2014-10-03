import numpy as np
from sklearn.cross_validation import KFold

class StackModel(object):
    '''
    An ensemble method which uses the predicted probabilities from a list of
      individual models to train a super model for classifying.
    '''
    def __init__(self, supermodel, submodels):
        self.supermodel = supermodel
        self.submodels = submodels
        self.m = len(submodels)

    def fit(self, X, y):
        '''
        1) Train m sub-models, loading predict_proba from each into sub_preds
        2) Train supermodel using sub_preds as features
        '''
        n = len(y)
        sub_preds = np.zeros((n, self.m))

        for i, sm in enumerate(self.submodels):
            # train, predict proba, add to sub_preds column
            sm.fit(X, y)
            sub_preds[:, i] = sm.predict_proba(X)[:, 1]

        self.supermodel.fit(sub_preds, y)

    def predict(self, X):
        n = len(X)
        sub_preds = np.zeros((n, self.m))
        for i, sm in enumerate(self.submodels):
            sub_preds[:, i] = sm.predict_proba(X)[:, 1]

        final_preds = self.supermodel.predict(sub_preds)
        return final_preds

    def predict_proba(self, X):
        n = len(X)
        sub_preds = np.zeros((n, self.m))
        for i, sm in enumerate(self.submodels):
            sub_preds[:, i] = sm.predict_proba(X)[:, 1]

        final_preds = self.supermodel.predict_proba(sub_preds)[:,1]
        return final_preds

    def score(self, X, y):
        final_preds = self.predict(X)
        return np.mean(final_preds == y)

    def cv(self, X, y, n_folds = 5):
        scores = []
        for train, test in KFold(len(y), n_folds, shuffle=True):
            self.fit(X[train], y[train])
            scores.append(self.score(X[test], y[test]))
        return np.mean(scores)
