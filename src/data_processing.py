import pandas as pd
#  ──────────────────────────── 1.Define Columns Group  ────────────────────────────

def cols_by_type(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {'categorical': categorical, 'datetime': datetime, 'numeric': numeric}

# groups = cols_by_type(df_train)
# print(groups['categorical'])
# print(groups['datetime'])
# print(groups['numeric'])

#  ──────────────────────────── 1.Check Missing Value  ────────────────────────────
def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dtype · count · missing · missing_%
    Only includes columns with missing values.
    """
    n_rows = len(df)

    summary = (
        pd.DataFrame({
            "dtype": df.dtypes,
            "count": df.count(),                      # Non-missing values
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
        })
        .query("missing > 0")                         # Keep only columns with missing values
        .sort_values("missing", ascending=False)      # Sort by missing count (descending)
    )

    return summary

#  ──────────────────────────── 2.Define a function checking near-zero variance  ────────────────────────────
def near_zero_var(df, freq_cut=95/5, unique_cut=10):
    """
    Identifies columns with near-zero variance in a DataFrame and calculates indicators.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - freq_cut (float): Threshold for the frequency ratio (default = 95/5).
    - unique_cut (int): Threshold for the unique value ratio (default = 10).

    Returns:
    - pd.DataFrame: A sorted DataFrame containing:
        - variable: Column name
        - freq_ratio: Ratio of the most common value to the second most common value
        - unique_ratio: Ratio of unique values to total observations
        - high_freq_ratio: Binary indicator (1 if freq_ratio > freq_cut)
        - low_unique_ratio: Binary indicator (1 if unique_ratio < unique_cut)
    """
    results = []

    for col in df.columns:
        # Get the value counts
        counts = df[col].value_counts()

        # Calculate freq_ratio
        if len(counts) > 1:
            freq_ratio = counts.iloc[0] / counts.iloc[1]
        else:
            freq_ratio = float('inf')  # Only one unique value

        # Calculate unique_ratio
        unique_ratio = len(counts) / len(df)

        # Determine binary indicators
        high_freq_ratio = int(freq_ratio > freq_cut)
        low_unique_ratio = int(unique_ratio < unique_cut)

        # Append results
        results.append({
            'variable': col,
            'dtype':df[col].dtype,
            'freq_ratio': freq_ratio,
            'unique_ratio': unique_ratio,
            'high_freq_ratio': high_freq_ratio,
            'low_unique_ratio': low_unique_ratio
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Sort by 'high_freq_ratio' (descending) and 'low_unique_ratio' (ascending)
    results_df = results_df.sort_values(by=['freq_ratio', 'unique_ratio'], ascending=[False, True])

    return results_df


#  ──────────────────────────── 3.VIF  ──────────────────────────── 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif_table(df, features=None, dropna=True, verbose=True):
    """
    计算 features 列的 VIF，返回 DataFrame（按 VIF 降序）。
    - df: pandas DataFrame
    - features: list of column names to include; None -> 自动选数值列
    - dropna: True 则先按这些列 dropna（常用），False 则保留所有行（可能报错）
    返回: vif_df，包含 columns ['VIF', 'n_obs', 'const_cols_removed']
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = list(features)

    X = df[features].copy()
    if dropna:
        X = X.dropna(how='any')
    if X.shape[0] == 0:
        raise ValueError("No rows left after dropna; change dropna=False or provide other features.")

    # 移除方差为0的列（常数列）
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        features = [f for f in features if f not in const_cols]
        if verbose:
            print("Removed constant columns before VIF:", const_cols)

    # 如果只剩常数列或 0 列，直接返回空表
    if X.shape[1] == 0:
        return pd.DataFrame(columns=['VIF','n_obs']).astype(object)

    # 加常数项（VIF 的计算习惯）
    X_with_const = sm.add_constant(X, has_constant='add')

    vifs = []
    for i, col in enumerate(X.columns):
        try:
            # 因为第0列是常数项，所以变量列索引是 i+1
            vif_val = variance_inflation_factor(X_with_const.values, i+1)
        except Exception as e:
            # 若矩阵奇异或其它问题，设为 inf
            vif_val = np.inf
        vifs.append((col, vif_val))

    vif_df = pd.DataFrame(vifs, columns=['feature', 'VIF']).set_index('feature')
    vif_df['n_obs'] = X.shape[0]
    vif_df['VIF_gt_5'] = vif_df['VIF'] > 5
    vif_df['VIF_gt_10'] = vif_df['VIF'] > 10
    vif_df = vif_df.sort_values('VIF', ascending=False)
    # 将被移除的常数列也记录在返回的属性里（方便追踪）
    vif_df.attrs['const_cols_removed'] = const_cols
    return vif_df

def iterative_vif_filter(df, features, dropna=True, thresh=5.0, verbose=True, max_iter=100):
    """
    贪婪地删除 VIF 最大的变量直到所有变量 VIF <= thresh（或达到 max_iter）。
    返回保留下来的特征列表与最终 VIF 表。
    注意：这是一种启发式方法，应结合业务判断来删变量。
    """
    features = list(features)
    for it in range(max_iter):
        vif_table = compute_vif_table(df, features=features, dropna=dropna, verbose=False)
        if vif_table.shape[0] == 0:
            if verbose:
                print("No features left.")
            return [], vif_table
        max_vif = vif_table['VIF'].replace([np.inf], np.finfo(float).max).max()
        if max_vif <= thresh:
            if verbose:
                print(f"Stopped at iteration {it}: all VIF <= {thresh:.3f}")
            return features, vif_table
        # 找到 VIF 最大的特征（若多个取第一个）
        worst = vif_table['VIF'].idxmax()
        if verbose:
            print(f"Iter {it}: dropping '{worst}' (VIF={vif_table.loc[worst,'VIF']:.3g})")
        features.remove(worst)
    if verbose:
        print("Reached max_iter.")
    return features, compute_vif_table(df, features=features, dropna=dropna, verbose=False)


# # ---------------- 使用示例 ----------------
# # 假设你已有 `apartments_train` DataFrame，numeric 是你要检测的列名 list
# # numeric = ['area', 'bedrooms', 'bathrooms', 'age_2024', ...]
# # 1) 直接计算并查看 VIF 表
# vif_table = compute_vif_table(apartments_train, features=numeric, dropna=True)
# print(vif_table)

# # 2) 若你想标注并保存到 csv:
# vif_table.to_csv("vif_table.csv")

# # 3) 若想自动逐步剔除直到 VIF <= 5（谨慎使用）
# kept_features, final_vif = iterative_vif_filter(apartments_train, features=numeric, dropna=True, thresh=5.0)
# print("Kept features:", kept_features)
# print(final_vif)
