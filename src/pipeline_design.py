# =============================================================================
# 1. SETUP: IMPORTS AND PREPARATION
# =============================================================================
# --- Import necessary libraries ---

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, GridSearchCV,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, make_scorer,mean_squared_log_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone
import time
from scipy.special import inv_boxcox
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# --- Define a custom scorer for Root Mean Squared Error (RMSE) ---
# This scorer will be used in cross-validation.
# We set greater_is_better=False because lower RMSE is better.
# squared=False directly returns the RMSE value.
rmse_scorer = make_scorer(
    mean_squared_error,
    greater_is_better=False,
    squared=False
)

# Define the K-Fold cross-validation strategy.
# This ensures all models are evaluated on the same data splits,
# making their performance comparable.
kfold = KFold(n_splits=5, shuffle=True, random_state=476677)


# =============================================================================
# 2. REUSABLE EVALUATION FUNCTION
# =============================================================================
def evaluate_model(estimator, X, y, cv, scoring):
    """
    Evaluate the model on the ORIGINAL y scale using RMSE only.
    Note: `is_log_transformed` is ignored for backward compatibility.
    """

    # ---------- 1) Build RMSE scorer on the original scale ----------
    # Expect `scoring` to be a valid RMSE scorer (greater_is_better=False)
    chosen_rmse = scoring

    # ---------- 2) Cross-validation ----------
    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring={"rmse": chosen_rmse},  # Only RMSE is evaluated
        return_train_score=False,
        n_jobs=-1
    )

    # ---------- 3) Reporting ----------
    # Negate to convert from "loss" (negative RMSE) to a positive metric
    rmse_scores = -cv_results["test_rmse"]

    # Derive a readable model name from the last pipeline step
    model_name = estimator.steps[-1][0].replace('_', ' ').title()
    print(f"--- {model_name} Cross-Validation Results ---")
    for i, s in enumerate(rmse_scores, 1):
        print(f"Fold {i} RMSE : {s:.4f}")

    print("-" * 30)
    print(f"[{model_name}] Mean {len(rmse_scores)}-fold CV RMSE : "
          f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    # Inference latency summary from cross_validate
    lat = cv_results["score_time"]
    print(f"Mean predict latency: {lat.mean():.6f}s ± {lat.std():.6f}s")

    return rmse_scores


def evaluate_model_y_log(estimator, X, y, cv, scoring):
    """
    Evaluate the model on the ORIGINAL y scale using RMSE only.
    Note: `is_log_transformed` is ignored for backward compatibility.
    """

    # ---------- 1) Build RMSE scorer on the original scale ----------
    # Expect `scoring` to be a valid RMSE scorer (greater_is_better=False)
    chosen_rmse = scoring

    # ---------- 2) Cross-validation ----------
    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring={"rmse": chosen_rmse},  # Only RMSE is evaluated
        return_train_score=False,
        n_jobs=-1
    )

    # ---------- 3) Reporting ----------
    # Negate to convert from "loss" (negative RMSE) to a positive metric
    rmse_scores = -cv_results["test_rmse"]

    # Derive a readable model name from the last pipeline step
    model_name = estimator.steps[-1][0].replace('_', ' ').title()
    print(f"--- {model_name} Cross-Validation Results ---")
    for i, s in enumerate(rmse_scores, 1):
        print(f"Fold {i} RMSE : {s:.4f}")

    print("-" * 30)
    print(f"[{model_name}] Mean {len(rmse_scores)}-fold CV RMSE : "
          f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    # Inference latency summary from cross_validate
    lat = cv_results["score_time"]
    print(f"Mean predict latency: {lat.mean():.6f}s ± {lat.std():.6f}s")

    return rmse_scores



def evaluate_model_y_log(estimator, X, y, cv, scoring, is_log_transformed=False):
    # ---------- 1. 构造 RMSE / RMSLE scorer ----------
    if is_log_transformed:
        # ① 逆变换 RMSE
        def inv_rmse(y_true_log, y_pred_log):
            y_pred = np.exp(y_pred_log)
            y_true = np.exp(y_true_log)
            return mean_squared_error(y_true, y_pred, squared=False)

        chosen_rmse = make_scorer(inv_rmse, greater_is_better=False)

        # ② 逆变换 RMSLE
        def inv_rmsle(y_true_log, y_pred_log):
            y_pred = np.exp(y_pred_log)
            y_true = np.exp(y_true_log)
            return np.sqrt(mean_squared_log_error(y_true, y_pred))

        # （保留未被直接调用的 scorers 变量，确保原接口不变）
        scorers = {
            "rmse": make_scorer(inv_rmse, greater_is_better=False),
            "rmsle": make_scorer(inv_rmsle, greater_is_better=False)
        }

        chosen_rmsle = make_scorer(inv_rmsle, greater_is_better=False)

        print("Note: evaluating on the ORIGINAL scale (inverse-transformed).!!!")
    else:
        # 直接在原始尺度上算
        chosen_rmse  = scoring
        chosen_rmsle = make_scorer(
            lambda yt, yp: np.sqrt(mean_squared_log_error(yt, yp)),
            greater_is_better=False
        )

    # ---------- 2. 交叉验证 ----------
    cv_results = cross_validate(
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        scoring={"rmse": chosen_rmse, "rmsle": chosen_rmsle},
        return_train_score=False,
        n_jobs=-1
    )

    # ---------- 3. 输出 ----------
    rmse_scores  = -cv_results["test_rmse"]     # 取反号 → 正向指标
    rmsle_scores = -cv_results["test_rmsle"]

    model_name = estimator.steps[-1][0].replace('_', ' ').title()
    print(f"--- {model_name} Cross-Validation Results ---")
    for i, s in enumerate(rmse_scores, 1):
        print(f"Fold {i} RMSE : {s:.4f}")

    print("-" * 30)
    print(f"[{model_name}] Mean {len(rmse_scores)}-fold CV RMSE : "
          f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}!!!")
    print(f"[{model_name}] Mean {len(rmsle_scores)}-fold CV RMSLE: "
          f"{rmsle_scores.mean():.4f}")

    lat = cv_results["score_time"]
    print(f"Mean predict latency: {lat.mean():.6f}s ± {lat.std():.6f}s")

    return rmse_scores, rmsle_scores



def evaluate_model_Box_Cox(estimator, X, y, cv, scoring, is_log_transformed=False, bc_lambda=None):
    """
    Evaluate a regression estimator with cross-validation.

    - If is_log_transformed=True (here used for Box-Cox transformed target),
      we inverse-transform predictions and ground truth back to the ORIGINAL scale
      and then compute RMSE / RMSLE on that original scale.
    - bc_lambda must be provided when using Box-Cox so that inverse transform works
      inside parallel scorers.
    """
    # ---------- 1) Build RMSE / RMSLE scorers ----------
    if is_log_transformed:
        if bc_lambda is None:
            raise ValueError("bc_lambda must be provided when is_log_transformed=True (Box-Cox mode).")

        # ① Inverse-transform RMSE (Box-Cox → original scale)
        def inv_rmse(y_true_bc, y_pred_bc, lam=bc_lambda):
            y_pred = inv_boxcox(y_pred_bc, lam)
            y_true = inv_boxcox(y_true_bc, lam)
            return mean_squared_error(y_true, y_pred, squared=False)

        chosen_rmse = make_scorer(inv_rmse, greater_is_better=False)

        # ② Inverse-transform RMSLE (compute MSLE on the original scale)
        def inv_rmsle(y_true_bc, y_pred_bc, lam=bc_lambda):
            y_pred = inv_boxcox(y_pred_bc, lam)
            y_true = inv_boxcox(y_true_bc, lam)
            return np.sqrt(mean_squared_log_error(y_true, y_pred))

        # Keep an extra dict (not used below) for interface compatibility/readability
        scorers = {
            "rmse": make_scorer(inv_rmse, greater_is_better=False),
            "rmsle": make_scorer(inv_rmsle, greater_is_better=False)
        }

        print("Note: evaluating on the ORIGINAL scale via Box-Cox inverse transform.")
        chosen_rmsle = make_scorer(inv_rmsle, greater_is_better=False)
    else:
        # Directly compute metrics on the original scale
        chosen_rmse  = scoring
        chosen_rmsle = make_scorer(
            lambda yt, yp: np.sqrt(mean_squared_log_error(yt, yp)),
            greater_is_better=False
        )

    # ---------- 2) Cross-validation ----------
    cv_results = cross_validate(
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        scoring={"rmse": chosen_rmse, "rmsle": chosen_rmsle},
        return_train_score=False,
        n_jobs=-1
    )

    # ---------- 3) Reporting ----------
    rmse_scores  = -cv_results["test_rmse"]     # flip sign → larger is better -> positive RMSE
    rmsle_scores = -cv_results["test_rmsle"]

    model_name = estimator.steps[-1][0].replace('_', ' ').title()
    print(f"--- {model_name} Cross-Validation Results ---")
    for i, s in enumerate(rmse_scores, 1):
        print(f"Fold {i} RMSE : {s:.4f}")

    print("-" * 30)
    print(f"[{model_name}] Mean {len(rmse_scores)}-fold CV RMSE : "
          f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}..")
    print(f"[{model_name}] Mean {len(rmsle_scores)}-fold CV RMSLE: "
          f"{rmsle_scores.mean():.4f}")

    lat = cv_results["score_time"]
    print(f"Mean predict latency: {lat.mean():.6f}s ± {lat.std():.6f}s")

    return rmse_scores, rmsle_scores



# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================

# --- Model 1: Ridge Regression ---
# RidgeCV performs cross-validation internally to find the best alpha.
# print("--- Evaluating Linear Models ---")
ridge_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('ridge_cv', RidgeCV(alphas=np.logspace(-2, 5, 200)))
])
# evaluate_model(ridge_pipe, df_train_x, df_train_y, kfold, rmse_scorer)


# --- Model 2: Elastic Net ---
# ElasticNetCV finds the best combination of alpha and l1_ratio.
elasticnet_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('elasticnet_cv', ElasticNetCV(
        alphas=np.logspace(-2, 5, 200),
        l1_ratio=np.linspace(0.1, 0.9, 9),
        cv=5,
        n_jobs=-1,
        max_iter=10000,
        random_state=42
    ))
])
# evaluate_model(elasticnet_pipe, df_train_x, df_train_y, kfold, rmse_scorer)


# --- Model 3: Lasso Regression ---
# LassoCV performs cross-validation to find the best alpha for L1 regularization.
lasso_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('lasso_cv', LassoCV(
        alphas=np.logspace(-2, 5, 200),
        cv=5,
        n_jobs=-1,
        max_iter=10000,
        random_state=42
    ))
])
# evaluate_model(lasso_pipe, df_train_x, df_train_y, kfold, rmse_scorer)
# print("-" * 30)

# --- Model 4: Support Vector Regressor (RBF Kernel) ---
# For SVR, we first use GridSearchCV to find the best hyperparameters.
# Then, we evaluate the best resulting model using our consistent evaluation function.
# print("\n--- Evaluating SVR Models (with Consistent CV) ---")
svr_rbf_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('svr', SVR(kernel='rbf'))
])

# Define the hyperparameter grid to search
rbf_param_grid = {
    'svr__C':       np.logspace(-2, 3, 10),
    'svr__epsilon': np.logspace(-3, 0, 8),
    'svr__gamma':   ['scale', 'auto'] + list(np.logspace(-4, 1, 6))
}

# Setup and fit GridSearchCV to find the best model
svr_rbf_gscv = GridSearchCV(
    estimator=svr_rbf_pipe,
    param_grid=rbf_param_grid,
    scoring='neg_mean_squared_error',
    cv=kfold, # Use the same kfold for internal CV
    n_jobs=-1,
    refit=True
)
# svr_rbf_gscv.fit(df_train_x, df_train_y)

# Extract the best estimator found by the grid search
# best_svr_rbf_model = svr_rbf_gscv.best_estimator_

# Now, evaluate this best model using the same external cross-validation as other models
# print(f"[RBF SVR] Best parameters found: {svr_rbf_gscv.best_params_}")
# evaluate_model(best_svr_rbf_model, df_train_x, df_train_y, kfold, rmse_scorer)


# --- Model 5: Linear Support Vector Regressor ---
# We follow the same two-step process: find the best model, then evaluate it.
linear_svr_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('svr', LinearSVR(random_state=42, dual='auto', max_iter=5000))
])

# Define the hyperparameter grid
linear_param_grid = {
    'svr__C':       np.logspace(-2, 3, 10),
    'svr__epsilon': np.logspace(-3, 0, 8)
}

# Setup and fit GridSearchCV
linear_svr_gscv = GridSearchCV(
    estimator=linear_svr_pipe,
    param_grid=linear_param_grid,
    scoring='neg_mean_squared_error',
    cv=kfold,
    n_jobs=-1,
    refit=True
)
# linear_svr_gscv.fit(df_train_x, df_train_y)

# Extract the best estimator
# best_linear_svr_model = linear_svr_gscv.best_estimator_

# Evaluate the best linear SVR model consistently
# print(f"[Linear SVR] Best parameters found: {linear_svr_gscv.best_params_}")
# evaluate_model(best_linear_svr_model, df_train_x, df_train_y, kfold, rmse_scorer)
# print("-" * 30)


# --- Model 6: XGBoost Regressor ---

# XGBoost 回归 Pipeline（树模型一般不需要特征缩放）
xgb_pipe = Pipeline([
    ('xgb_reg', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method='hist',     # 若有 GPU 可改 'gpu_hist'
        random_state=476677,
        n_jobs=-1,
        eval_metric='rmse'      # 训练内部度量，不影响外部CV评分
    ))
])

# CustomPreprocessor
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor that replicates the described training preprocessing.
    Key change from previous version: OneHotEncoder is applied as the *last*
    preprocessing step (after numeric imputation, categorical imputation,
    boolean encoding and year_built KNN imputation). Other behavior is
    preserved.

    Notes:
      - transform(...) returns a pandas.DataFrame (index preserved).
      - year_built in the final DataFrame is converted to pandas nullable Int64 dtype.
      - ensures OHE columns are ints and that missing output columns are added.
      - column order is aligned to the order recorded in fit (self.out_columns_).
    """

    def __init__(self, numeric, categorical, boolean, date_col='src_month', year_col='year_built'):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.boolean = list(boolean)
        self.date_col = date_col
        self.year_col = year_col

        # Will be initialized/fitted in fit
        self.num_imputer_ = KNNImputer(n_neighbors=5, weights='distance')
        self.cat_imputer_ = SimpleImputer(strategy='constant', fill_value='missing')
        self.bool_imputer_ = SimpleImputer(strategy='most_frequent')
        # note: sparse_output requires sklearn >= 1.2; if older sklearn use sparse=False
        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.bool_encoder_ = OrdinalEncoder(categories=[['no', 'yes']] * len(self.boolean))
        self.year_imputer_ = KNNImputer(n_neighbors=5, weights='distance')

        # Metadata saved after fit
        self.ohe_out_cols_ = None
        self.out_columns_ = None  # Post-transformation column order

    # ---------- Utility function: Parse src_month according to your rules ----------
    def _parse_src_month(self, s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, format='%b-%y', errors='coerce')
        yyyymm = (dt.dt.year * 100 + dt.dt.month).astype('float64')
        yyyymm[dt.isna()] = np.nan
        return yyyymm

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # 1) Parse src_month -> YYYYMM (preserve missing values)
        if self.date_col in df.columns:
            df[self.date_col] = self._parse_src_month(df[self.date_col])

        # 2) Convert year_built type and round (use float internally in sklearn to represent missing values)
        if self.year_col in df.columns:
            df[self.year_col] = pd.to_numeric(df[self.year_col], errors='coerce').round().astype('float64')

        # 3) numeric: KNN imputation (fit + transform on training data)
        if self.numeric:
            df[self.numeric] = self.num_imputer_.fit_transform(df[self.numeric])

        # 4) categorical: missing -> "missing" (fit the imputer)
        if self.categorical:
            df.loc[:, self.categorical] = self.cat_imputer_.fit_transform(df[self.categorical])

        # 5) boolean: 众数插补
        if self.boolean:
            imp = self.bool_imputer_.fit_transform(df[self.boolean])
            # 先用 DataFrame 承接，保持索引/列名
            df_imp = pd.DataFrame(imp, index=df.index, columns=self.boolean)
            df[self.boolean] = df_imp  # 这一步仍是 object，但没关系，下一步会逐列覆盖为 int

        # 6) boolean: 'no'/'yes' -> 0/1
        if self.boolean:
            enc = self.bool_encoder_.fit_transform(df[self.boolean])
            enc_df = pd.DataFrame(enc, index=df.index, columns=self.boolean).astype("int8")
            # 关键：逐列覆盖，避免多列切片导致的 object block 保留
            for c in self.boolean:
                df[c] = enc_df[c].to_numpy(copy=False)



        # 7) year_built: Use predictors = unique(numeric + boolean) + ['year_built'] for KNN imputation
        predictors = list(dict.fromkeys(list(self.numeric) + list(self.boolean)))
        year_predict_cols = [c for c in predictors if c in df.columns] + ([self.year_col] if self.year_col in df.columns else [])
        if year_predict_cols:
            sub = df[year_predict_cols].copy()
            sub_imputed = self.year_imputer_.fit_transform(sub)
            # Update only the year_built column
            if self.year_col in year_predict_cols:
                y_idx = year_predict_cols.index(self.year_col)
                df[self.year_col] = sub_imputed[:, y_idx]

        # 8) categorical: OHE and replace original columns (do this last)
        if self.categorical:
            ohe_arr = self.ohe_.fit_transform(df[self.categorical])
            self.ohe_out_cols_ = list(self.ohe_.get_feature_names_out(self.categorical))
            df_ohe = pd.DataFrame(ohe_arr, index=df.index, columns=self.ohe_out_cols_).astype(int)
            df = pd.concat([df.drop(columns=self.categorical), df_ohe], axis=1)

        # Record output column order (categorical columns dropped, OHE columns appended)
        self.out_columns_ = list(df.columns)
        return self

    def transform(self, X: pd.DataFrame):
        if self.out_columns_ is None:
            raise RuntimeError("CustomPreprocessor has not been fitted. Call fit(...) before transform(...).")

        df = X.copy()

        # 1) Parse src_month
        if self.date_col in df.columns:
            df[self.date_col] = self._parse_src_month(df[self.date_col])

        # 2) year_built -> numeric float (sklearn internals)
        if self.year_col in df.columns:
            df[self.year_col] = pd.to_numeric(df[self.year_col], errors='coerce').round().astype('float64')

        # 3) numeric: KNN imputer
        if self.numeric:
            # If some numeric columns are missing from incoming X, create them as NaN so imputer shape matches
            missing_nums = [c for c in self.numeric if c not in df.columns]
            if missing_nums:
                for c in missing_nums:
                    df[c] = np.nan
            df[self.numeric] = self.num_imputer_.transform(df[self.numeric])

        # 4) categorical: missing -> "missing" (ensure columns exist)
        if self.categorical:
            missing_cats = [c for c in self.categorical if c not in df.columns]
            if missing_cats:
                for c in missing_cats:
                    df[c] = np.nan
            df.loc[:, self.categorical] = self.cat_imputer_.transform(df[self.categorical])


        # 5) boolean: 众数插补（确保缺失列先补出来）
        if self.boolean:
            missing_bools = [c for c in self.boolean if c not in df.columns]
            for c in missing_bools:
                df[c] = np.nan
            imp = self.bool_imputer_.transform(df[self.boolean])
            df_imp = pd.DataFrame(imp, index=df.index, columns=self.boolean)
            df[self.boolean] = df_imp

        # 6) boolean: 编码为 0/1
        if self.boolean:
            enc = self.bool_encoder_.transform(df[self.boolean])
            enc_df = pd.DataFrame(enc, index=df.index, columns=self.boolean).astype("int8")
            for c in self.boolean:
                df[c] = enc_df[c].to_numpy(copy=False)


        # 7) year_built: KNN using predictors (numeric + boolean)
        predictors = list(dict.fromkeys(list(self.numeric) + list(self.boolean)))
        year_predict_cols = [c for c in predictors if c in df.columns] + ([self.year_col] if self.year_col in df.columns else [])
        if year_predict_cols:
            sub = df[year_predict_cols].copy()
            sub_imputed = self.year_imputer_.transform(sub)
            if self.year_col in year_predict_cols:
                y_idx = year_predict_cols.index(self.year_col)
                df[self.year_col] = sub_imputed[:, y_idx]

        # 8) categorical: OHE -> replace original categorical columns (do this last)
        if self.categorical:
            ohe_arr = self.ohe_.transform(df[self.categorical])
            # use stored ohe_out_cols_ if available, otherwise derive
            cols = self.ohe_out_cols_ if self.ohe_out_cols_ is not None else list(self.ohe_.get_feature_names_out(self.categorical))
            df_ohe = pd.DataFrame(ohe_arr, index=df.index, columns=cols).astype(int)
            df = pd.concat([df.drop(columns=self.categorical), df_ohe], axis=1)

        # Align with column order from fit (add missing columns, discard extras)
        for c in self.out_columns_:
            if c not in df.columns:
                df[c] = np.nan

        # ensure same column order
        df = df[self.out_columns_].copy()

        # final dtype tidy-ups:
        # - make OHE columns int if present
        if self.ohe_out_cols_:
            for c in self.ohe_out_cols_:
                if c in df.columns:
                    # astype Int64 would convert NaN -> <NA>, keep as int if no NaN else Int64
                    if df[c].isna().any():
                        df[c] = df[c].astype('Int64')
                    else:
                        df[c] = df[c].astype(int)

        # - convert year_col back to pandas nullable integer (Int64) if present
        if self.year_col in df.columns:
            # round to integer and set nullable Int64
            df[self.year_col] = pd.to_numeric(df[self.year_col], errors='coerce').round()
            df[self.year_col] = df[self.year_col].astype('Int64')

        # keep index from input and return DataFrame
        return df


