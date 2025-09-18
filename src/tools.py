import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, balanced_accuracy_score

def compute_costs(LoanAmount):
    return({'Risk_No Risk': 5.0 + .6 * LoanAmount,
        'No Risk_No Risk': 1.0 - .05 * LoanAmount,
        'Risk_Risk': 1.0,
        'No Risk_Risk': 1.0})

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    A custom metric for the German credit dataset
    '''
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
    costs = compute_costs(solution['LoanAmount'])
    y_true = solution['Risk']
    y_pred = submission['Risk']
    loss = (
        (y_true == 'Risk') * custom_weight['Risk'] * (
            (y_pred == 'Risk') * costs['Risk_Risk'] +
            (y_pred == 'No Risk') * costs['Risk_No Risk']
        ) +
        (y_true == 'No Risk') * custom_weight['No Risk'] * (
            (y_pred == 'Risk') * costs['No Risk_Risk'] +
            (y_pred == 'No Risk') * costs['No Risk_No Risk']
        )
    )
    return float(loss.mean())  # Convert to regular float

def leaderboad(y_pred, ddata, y_data=None):
    '''Calculate the custom metric for the given predictions and data.
    Args:
        y_pred: np.ndarray
            The predicted values.
        ddata: xgb.DMatrix or pd.DataFrame
            The data used to train the model.
        y_data: array-like, optional
            The true labels (required if ddata is DataFrame)
    Returns:
        float
            The leaderboard score.
    '''
    if type(ddata) == xgb.DMatrix:
        X_data = ddata.get_data()
        y_data = ddata.get_label()

        feature_names = ddata.feature_names
        df = pd.DataFrame(X_data.toarray(), columns=feature_names)
        df['Risk'] = y_data
        ddata = df
    else:
        ddata = ddata.copy()
        ddata['Risk'] = y_data
    y_true = ddata['Risk'].map({1: 'Risk', 0: 'No Risk'})
    y_pred_class = pd.Series((y_pred > 0.5).astype(int)).map({1: 'Risk', 0: 'No Risk'})
    solution_df = pd.DataFrame({'Risk': y_true, 'LoanAmount': (ddata['LoanAmount']) ** 2})
    submission_df = pd.DataFrame({'Risk': y_pred_class})
    loss = score(solution_df, submission_df, row_id_column_name='id')

    return float(loss)  # Convert np.float64 to regular float

def cv(model, X, y, n_splits=5, weights=None):
    '''Perform cross-validation.
    Args:
        model: sklearn model
            The model to train.
        X: pd.DataFrame
            The features.
        y: pd.Series
            The target.
        n_splits: int
            The number of splits.
        weights: np.ndarray
            The sample weights.
    Returns:
        list
            List of scores for each fold.
    '''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []

    for train_index, val_index in kf.split(X):
        X_t, X_val = X.iloc[train_index], X.iloc[val_index]
        y_t, y_val = y.iloc[train_index], y.iloc[val_index]

        if weights is not None:
            weight = pd.Series(weights.flatten()).iloc[train_index]
            try:
                model.fit(X_t, y_t, sample_weight=np.array(weight).flatten())
            except:
                model.fit(X_t, y_t, xgbclassifier__sample_weight=np.array(weight).flatten())
        else:
            model.fit(X_t, y_t)
        
        y_pred = model.predict(X_val)
        score_cv = leaderboad(y_pred, X_val, y_val)
        scores.append(float(score_cv))  # Convert np.float64 to regular float

    return scores

def cv_ac(model, X, y, n_splits=5, alpha=0.5):
    '''Perform cross-validation with a custom metric.
    Args:
        model: sklearn model
            The model to train.
        X: pd.DataFrame
            The features.
        y: pd.Series
            The target.
        n_splits: int
            The number of splits.
        alpha: float
            Weight for combining accuracy and leaderboard score.
    Returns:
        list
            List of combined custom scores for each split.
    '''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_t, X_val = X.iloc[train_index], X.iloc[val_index]
        y_t, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_t, y_t)
        
        y_pred = model.predict(X_val)
        
        # Calculate the custom leaderboard score
        score_cv = leaderboad(y_pred, X_val, y_val)
        
        # Calculate accuracy
        acc = accuracy_score(y_val, y_pred)
        
        # Combine the scores
        custom_score = alpha * acc + (1 - alpha) * (1 - (score_cv + 150) / 150)
        
        scores.append(float(custom_score))  # Convert np.float64 to regular float

    return scores

def specificity_score(y_true, y_pred):
    '''Calculate specificity score (True Negative Rate).'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def npv_score(y_true, y_pred):
    '''Calculate Negative Predictive Value.'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

# Create scorers for sklearn
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
specificity_scorer = make_scorer(specificity_score)
npv_scorer = make_scorer(npv_score)

def evaluate(model, X, y):
    '''Comprehensive evaluation of a model using multiple metrics.
    
    Args:
        model: sklearn pipeline
            The model pipeline to evaluate.
        X: pd.DataFrame
            The features.
        y: pd.Series
            The target.
    '''
    print(f'Performance of {model.steps[-1][0]}')
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': balanced_accuracy_scorer,
        'specificity': specificity_scorer,
        'npv': npv_scorer
    }
    scores = cross_validate(model, X, y, scoring=scoring, cv=5)
    print("accuracy: ", scores['test_accuracy'])
    print("mean_accuracy: ", np.mean(scores['test_accuracy']))
    print("balanced_accuracy: ", scores['test_balanced_accuracy'])
    print("mean_balanced_accuracy: ", np.mean(scores['test_balanced_accuracy']))
    print("specificity: ", scores['test_specificity'])
    print("mean_specificity: ", np.mean(scores['test_specificity']))
    print("npv: ", scores['test_npv'])
    print("mean_npv: ", np.mean(scores['test_npv']))
    leaderboad_score = cv(model, X, y)
    print("leaderboad score: ", leaderboad_score)
    print("mean leaderboad score: ", np.mean(leaderboad_score))
