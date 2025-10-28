from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from xgboost import XGBRegressor

SKLEARN_MODELS = [
    LinearRegression(n_jobs=-1),
    Lasso(),
    Ridge(),
    ElasticNet(),
    ExtraTreesRegressor(n_jobs=-1, random_state=42),
    GradientBoostingRegressor(random_state=42),
    HistGradientBoostingRegressor(random_state=42),
    RandomForestRegressor(n_jobs=-1, random_state=42),
    XGBRegressor(n_jobs=-1, random_state=42),
    StackingRegressor(
        estimators=[
            ("et", ExtraTreesRegressor(n_jobs=-1, random_state=42)),
            ("lr", Ridge(alpha=1.0)),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
    ),
]


CROSS_VALIDATION_FOLDS = 5
