import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from itertools import combinations
from typing import Iterable, Tuple, Dict, List, Optional
import math
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------- Histplots ----------

# ===== Function Implementation =====
def plot_histplots(
    df: pd.DataFrame,
    cols: Iterable[str] = None,
    n_cols: int = 5,
    figsize_per_chart: Tuple[float, float] = (4, 3),
    bins: int = 30,                 # Fixed binning for faster performance
    kde: bool = False,              # Enable for smoothed curve; auto-subsamples large datasets
    save_png: bool = False
):
    """
    Lightweight histogram plotting for quick assessment of normality:
    - Independent binning per column; no global binning
    - Overlays normal curve based on sample μ/σ (not KDE)
    - Optional KDE on subsamples ≤20k
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in cols if c in df.columns]

    n_feats = len(cols)
    if n_feats == 0:
        raise ValueError("No numerical columns found suitable for histogram plotting.")

    n_rows = math.ceil(n_feats / n_cols)
    fw, fh = figsize_per_chart
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw * n_cols, fh * n_rows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    rows = []
    for i, col in enumerate(cols):
        ax = axes[i]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            ax.set_visible(False)
            continue

        # Histogram (line-only is faster; use density for easier curve overlay)
        sns.histplot(
            x=s, bins=bins, stat="density",
            element="step", fill=False, linewidth=1, ax=ax
        )

        # Overlay "normal curve" (based on sample μ/σ)
        mu, sigma = s.mean(), s.std(ddof=1)
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
            y = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, y, linewidth=1)  # No color specified, keep default

        # (Optional) KDE only on subsamples for large datasets to avoid extreme slowness
        if kde and len(s) > 0:
            s_sub = s.sample(min(len(s), 20_000), random_state=0)
            sns.kdeplot(x=s_sub, ax=ax, linewidth=1)

        ax.set_title(f"{col}  n={len(s)}  μ={mu:.3g}  σ={sigma:.3g}", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        # Brief statistics (for return)
        q1, q3 = s.quantile([0.25, 0.75])
        rows.append({
            "column": col, "count": int(len(s)),
            "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
            "mean": mu, "std": sigma
        })

    # Hide excess subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    corr_df = pd.DataFrame(rows, columns=[
        "column","count","min","q1","median","q3","max","mean","std"
    ])

    if save_png:
        fig.savefig("numeric_histplots_fast.png", dpi=160, bbox_inches="tight")

    return fig, axes, corr_df



# ---------- Box Plots ----------
def plot_boxplots(
    df: pd.DataFrame,
    value_col: str = None,                 # 为了兼容示例保留，但本函数不使用
    cols: Iterable[str] = None,
    n_cols: int = 5,
    figsize_per_chart: Tuple[float, float] = (4, 3),
    annotate: bool = True,
    p_adjust_method: str = "holm",         # 为了兼容示例保留，但本函数不使用
    save_png: bool = False,
    add_regression: bool = False,          # 为了兼容示例保留，但本函数不使用
    jitter: float = 0.0,                   # >0 时叠加散点以辅助观察离群点
    showfliers: bool = True,               # 是否显示箱线图自带的离群点
    random_state: int = 42                 # 控制抖动可复现
):
    """
    循环绘制数值列箱线图（IQR=1.5 规则），用于快速检测异常值。

    Parameters
    ----------
    df : DataFrame
        源数据。
    value_col : str, optional
        仅为兼容你的调用格式，本函数不使用。
    cols : Iterable[str], optional
        需要绘制的数值列名；若为 None，则自动选择 df 的数值列。
    n_cols : int
        子图每行列数。
    figsize_per_chart : (w, h)
        每个子图的尺寸（英寸）。
    annotate : bool
        是否在图上标注上下边界与异常值数量。
    p_adjust_method : str
        仅为兼容你的调用格式，本函数不使用。
    save_png : bool
        是否将整张图保存为 'numeric_boxplots.png'。
    add_regression : bool
        仅为兼容你的调用格式，本函数不使用。
    jitter : float
        若 > 0，则在箱线图上叠加散点（在 x 方向 ±jitter 的均匀抖动）。
    showfliers : bool
        是否显示箱线图的默认离群点标记。
    random_state : int
        抖动散点随机种子。

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray[matplotlib.axes.Axes]
    corr_df : DataFrame
        每列的异常值统计（列名保持为 corr_df 以兼容你的示例）：
        ['column','count','q1','q3','iqr','lower_bound','upper_bound',
         'n_outliers','pct_outliers','min','max','median','mean','std']
    """
    rng = np.random.default_rng(random_state)

    # 选择列
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = list(cols)

    # 过滤出真实存在的列，并尽量转为数值
    valid_cols = []
    for c in cols:
        if c in df.columns:
            # 强制转数值，无法转换的记为 NaN
            _s = pd.to_numeric(df[c], errors="coerce")
            if _s.notna().any():
                valid_cols.append(c)

    n_feats = len(valid_cols)
    if n_feats == 0:
        raise ValueError("未找到可用于箱线图的数值列。")

    n_rows = math.ceil(n_feats / n_cols)
    fw, fh = figsize_per_chart
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw * n_cols, fh * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    stats = []

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            ax.set_visible(False)
            continue

        # IQR 统计
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_mask = (s < lower) | (s > upper)
        n_out = int(out_mask.sum())
        pct_out = float(n_out / len(s) * 100.0)

        # 箱线图
        ax.boxplot([s.values], vert=True, whis=1.5, showfliers=showfliers, labels=[col])

        # 叠加散点（可选）
        if jitter and jitter > 0:
            x = 1 + rng.uniform(-jitter, jitter, size=len(s))
            ax.scatter(x, s.values, s=6, alpha=0.6, linewidths=0)

        # 标注与辅助线
        title = f"{col}  n={len(s)} | outliers={n_out} ({pct_out:.1f}%)"
        ax.set_title(title, fontsize=10)
        ax.axhline(lower, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(upper, linestyle="--", linewidth=1, alpha=0.6)

        if annotate:
            txt = f"LB={_fmt_num(lower)}  UB={_fmt_num(upper)}"
            ax.text(
                0.98, 0.02, txt,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, edgecolor="none")
            )

        stats.append({
            "column": col,
            "count": int(len(s)),
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "n_outliers": n_out, "pct_outliers": pct_out,
            "min": s.min(), "max": s.max(),
            "median": s.median(), "mean": s.mean(), "std": s.std(ddof=1)
        })

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    # 结果表
    corr_df = pd.DataFrame(stats, columns=[
        "column","count","q1","q3","iqr","lower_bound","upper_bound",
        "n_outliers","pct_outliers","min","max","median","mean","std"
    ])

    if save_png:
        fig.savefig("numeric_boxplots.png", dpi=200, bbox_inches="tight")

    return fig, axes, corr_df


def _fmt_num(x):
    """紧凑数值格式化，便于轴内标注。"""
    if x == 0 or abs(x) >= 0.01:
        return f"{x:.3g}"
    return f"{x:.2e}"




# ---------- 分类变量 ----------
#       plot_categorical       #
# ---------- 工具函数 ----------

def _auto_categorical_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    cats = []
    for c in df.columns:
        if c in exclude:
            continue
        if (
            pd.api.types.is_categorical_dtype(df[c].dtype)
            or pd.api.types.is_object_dtype(df[c].dtype)
            or pd.api.types.is_string_dtype(df[c].dtype)
        ):
            cats.append(c)
    return cats


def _pairwise_median_test(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[Tuple, float]:
    """两两 Mann-Whitney U 检验的 p 值 {(a,b): p}（a<b）。"""
    from scipy import stats
    groups = {k: v.dropna().to_numpy() for k, v in df.groupby(group_col)[value_col]}
    pvals = {}
    for a, b in combinations(groups, 2):
        x, y = groups[a], groups[b]
        if len(x) and len(y):
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        else:
            p = np.nan
        pvals[(a, b)] = p
    return pvals



def _adjust_pvals(pmap: Dict[Tuple, float], method: str = "none") -> Dict[Tuple, float]:
    """多重比较校正：holm / bonferroni / fdr_bh / none。"""
    method = (method or "none").lower()
    pairs = list(pmap)
    p = np.array([1.0 if np.isnan(v) else float(v) for v in pmap.values()])
    m = len(p)
    if m == 0 or method == "none":
        adj = p
    elif method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        adj = np.empty_like(p)
        running = 0.0
        for rank, idx in enumerate(order):
            cur = (m - rank) * p[idx]
            running = max(running, cur)
            adj[idx] = min(running, 1.0)
    elif method in ("fdr_bh", "bh", "benjamini-hochberg"):
        order = np.argsort(p)
        q = np.empty_like(p)
        for rank, idx in enumerate(order, 1):
            q[idx] = p[idx] * m / rank
        adj = np.empty_like(p)
        mn = 1.0
        for idx in order[::-1]:
            mn = min(mn, q[idx])
            adj[idx] = min(mn, 1.0)
    else:
        raise ValueError(f"未知 p 值校正方法：{method}")
    return {pairs[i]: float(adj[i]) for i in range(m)}


def _letter_names(n: int) -> List[str]:
    """a, b, …, z, aa, ab, …"""
    out, i = [], 0
    while len(out) < n:
        s, k = "", i
        while True:
            s = chr(ord("a") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s); i += 1
    return out


def _format_k(v: float) -> str:
    """以 k 单位格式化（千=1k）。"""
    vk = v / 1_000
    avk = abs(vk)
    if avk >= 100:
        txt = f"{vk:.0f}k"
    elif avk >= 10:
        txt = f"{vk:.1f}k"
    else:
        txt = f"{vk:.2f}k"
    if txt.startswith("-0"):
        txt = txt.replace("-0", "0", 1)
    return txt


# ---------- 主函数 ----------

def plot_categorical(
    df: pd.DataFrame,
    value_col: str = "price_z",
    cols: Optional[Iterable[str]] = None,      # None -> 自动选择类别列
    n_cols: int = 4,
    figsize_per_chart: Tuple[float, float] = (3.0, 3.0),  # (每柱最小宽度, 每子图高度)
    sort_by_count: bool = False,               # False -> 按均值降序
    annotate: bool = True,
    alpha: float = 0.05,
    p_adjust_method: str = "holm",             # 'holm' | 'bonferroni' | 'fdr_bh' | 'none'
    save_png: bool = False,
    # 字体
    title_fontsize: int = 12,
    label_fontsize: int = 12,
    xtick_fontsize: int = 12,
    annotation_fontsize: int = 10,
    main_letter_fontsize: int = 12,
):
    """
    多个类别变量的条形图（x=组别，y=price_z 均值，单位 k）。
    - 柱子中央：顺序字母 a/b/c…
    - 柱子顶部：均值(以 k) + 与其显著不同的柱子字母列表，如 100k(b,c)
    - 显著性：Welch t-test + 可选多重校正（Holm/Bonferroni/FDR-BH）
    - 全部子图共用同一 y 轴范围与刻度（单位 k）
    """
    if value_col not in df.columns:
        raise KeyError(f"value_col '{value_col}' 不在 DataFrame 列中。")

    # 选择类别列
    if cols is None:
        cols = _auto_categorical_cols(df, exclude=[value_col])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("未找到可用的类别列。")

    # 统一 y 轴范围
    col_info, all_medians, n_bars = {}, [], []
    for col in cols:
        grouped = df.groupby(col)[value_col]
        medians = grouped.median()
        counts = grouped.size().reindex(medians.index)
        if medians.empty:
            n_bars.append(0); continue
        order = counts.sort_values(ascending=False).index if sort_by_count else medians.sort_values(ascending=False).index
        medians_ordered = medians.reindex(order)
        col_info[col] = {"order": list(order), "medians": medians_ordered}
        all_medians.append(medians_ordered)
        n_bars.append(len(order))

    all_medians_concat = pd.concat(all_medians)
    y_min, y_max = float(all_medians_concat.min()), float(all_medians_concat.max())

    rng = y_max - y_min
    pad = 0.08 * (rng if rng else 1.0)
    y_bottom = y_min - 0.06 * (rng if rng else 1.0)
    y_top = y_max + pad * 2.0
    yticks = np.linspace(y_bottom, max(y_bottom + 1e-9, y_top - pad), num=5)

    # 画布尺寸
    per_bar_w, per_sub_h = figsize_per_chart
    widths_each = [max(1, n) * max(per_bar_w, 0.5) for n in n_bars]
    n_plots = len(cols)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_plots / n_cols))

    col_widths = [0.0] * n_cols
    for i, w in enumerate(widths_each):
        col_idx = i % n_cols
        col_widths[col_idx] = max(col_widths[col_idx], w)
    col_widths = [max(cw, 2.5) for cw in col_widths]
    fig_w, fig_h = float(sum(col_widths)), float(n_rows * per_sub_h)

    # ---- 使用 constrained_layout（解决 tight_layout 警告）----
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=col_widths, height_ratios=[1]*n_rows)
    # 可按需调整子图间距
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.25, hspace=0.35)

    axes = []
    for i in range(n_plots):
        r, c = divmod(i, n_cols)
        axes.append(fig.add_subplot(gs[r, c]))

    k_formatter = FuncFormatter(lambda v, pos: _format_k(v))

    # 绘制各子图
    for ax, col in zip(axes, cols):
        order = col_info[col]["order"]
        medians = col_info[col]["medians"]
        letters = _letter_names(len(order))
        letter_map = {cat: letters[i] for i, cat in enumerate(order)}


        # 中位数的两两检验 + 校正
        p_raw = _pairwise_median_test(df, col, value_col)
        p_adj = _adjust_pvals(p_raw, method=p_adjust_method)

        x = np.arange(len(order))
        h = medians.values
        bars = ax.bar(x, h)

        ax.set_title(col, fontsize=title_fontsize)
        ax.set_xticks(x, [str(k) for k in order], fontsize=xtick_fontsize)
        ax.set_ylabel(f"{value_col} (median, k)", fontsize=label_fontsize)
        ax.set_ylim(y_bottom, y_top)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(k_formatter)
        ax.tick_params(axis='y', labelsize=label_fontsize)

        if annotate:
            # 柱心字母（可见柱高的中点，确保不会被 y 轴下界裁掉）
            for bar, cat in zip(bars, order):
                yi = bar.get_height()
                y_mid_visible = (yi + y_bottom) / 2.0
                ax.text(bar.get_x() + bar.get_width()/2, y_mid_visible, letter_map[cat],
                        ha='center', va='center', fontsize=main_letter_fontsize,
                        color='white', fontweight='bold')

            # 顶部：均值(k) + 显著差异字母，如 100k(b,c)
            for bar, cat in zip(bars, order):
                yi = bar.get_height()
                sig_letters = []
                for other in order:
                    if other == cat:
                        continue
                    key = tuple(sorted((cat, other)))
                    p = p_adj.get(key, np.nan)
                    if not np.isnan(p) and p < alpha:
                        sig_letters.append(letter_map[other])
                tag = f"({','.join(sig_letters)})" if sig_letters else ""
                ax.text(bar.get_x() + bar.get_width()/2,
                        yi + (y_top - y_bottom) * 0.03,
                        _format_k(yi) + tag,
                        ha='center', va='bottom', fontsize=annotation_fontsize)

        ax.margins(x=0.05)

    # 关闭多余（当网格大于子图数）
    for j in range(len(axes), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        fig.add_subplot(gs[r, c]).axis('off')

    if save_png:
        fig.savefig("categorical_bars_welch_k.png", dpi=200, bbox_inches="tight")
    return fig


# ---------------- 使用示例（按需取消注释） ----------------
# categorical = ['obj_type', 'own_type', 'build_mat', 'cond_class', 'loc_code']
# fig = plot_categorical(
#     df=apartments_train, value_col='price_z',
#     cols=categorical[:-1],  # 或 cols=None 自动选择类别列
#     n_cols=4, figsize_per_chart=(3, 3),
#     sort_by_count=False, annotate=True,
#     alpha=0.05, p_adjust_method='holm',  # 也可 'bonferroni' / 'fdr_bh' / 'none'
#     save_png=False,
#     title_fontsize=12, label_fontsize=12,
#     xtick_fontsize=12, annotation_fontsize=10, main_letter_fontsize=12,
# )
# plt.show()


# ---------- 日期/时间变量 ----------
import math
from itertools import combinations                    # 预留，如需成对比较可使用
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter           # ← 新增
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def plot_time_series(
    df: pd.DataFrame,
    value_col: str,
    cols: list[str] | None,
    *,
    n_cols: int = 5,
    figsize_per_chart: tuple[float, float] = (0.5, 3),
    sort_by_count: bool = False,
    annotate: bool = True,
    alpha: float = 0.05,
    p_adjust_method: str = "holm",         # 'bonferroni' / 'fdr_bh' / 'none'
    save_png: bool | str = False,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    xtick_fontsize: int = 8,
    annotation_fontsize: int = 8,
    main_letter_fontsize: int = 8,
    max_xticks: int = 12                   # ← 新增：最大显示的 x 轴标签数量
):
    """
    绘制折线图：x=时间标签，y=每个时间标签下 value_col 的均值。

    重要说明：
    1. 本版本不会自动识别或转换时间变量；`cols` 必填且应已是期望的时间标签。
    2. 若标签是 datetime/period，也直接当作文本展示（绘图时映射为 0,1,2,...）。
    3. 所有子图共享 y 轴范围和刻度，单位用 k。
    4. 当 x 标签数 > 15，仅均匀显示不超过 max_xticks (默认 12) 个。
    """
    # -------- 参数校验 -----------------------------------------------------
    if not cols:
        raise ValueError("必须显式提供时间列列表 cols=[...]；本函数已禁用自动检测。")

    # -------- 预计算整体显著性（单因素 ANOVA） ------------------------------
    pvals_raw: list[float] = []
    for col in cols:
        groups = [
            g[value_col].dropna().values
            for _, g in df[[col, value_col]].dropna().groupby(col)
            if g[value_col].notna().any()
        ]
        if len(groups) >= 2:
            try:
                pvals_raw.append(stats.f_oneway(*groups).pvalue)
            except Exception:
                pvals_raw.append(np.nan)
        else:
            pvals_raw.append(np.nan)

    if p_adjust_method.lower() != "none":
        _, pvals_adj, _, _ = multipletests(
            pvals_raw, alpha=alpha, method=p_adjust_method
        )
    else:
        pvals_adj = pvals_raw

    # -------- 布局设置 -----------------------------------------------------
    n_charts = len(cols)
    n_rows = math.ceil(n_charts / n_cols)
    fig_w = max(3, figsize_per_chart[0] * n_cols)
    fig_h = max(2, figsize_per_chart[1] * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharey=False
    )
    axes = axes.flatten()

    # -------- 绘图 & 统计整体 y 范围 ---------------------------------------
    letters = "abcdefghijklmnopqrstuvwxyz"
    global_ymin, global_ymax = np.inf, -np.inf
    plotted_axes = []

    for idx, (col, ax) in enumerate(zip(cols, axes)):
        tmp = df[[col, value_col]].dropna()

        # 分组并选择排序方式
        grouped = tmp.groupby(col)[value_col].mean()
        if sort_by_count:
            order = tmp[col].value_counts().index
            grouped = grouped.loc[order]
        else:
            grouped = grouped.sort_index()

        # 更新全局 y 范围
        if not grouped.empty:
            global_ymin = min(global_ymin, grouped.min())
            global_ymax = max(global_ymax, grouped.max())

        # 将 x 轴标签映射到位置 0..n-1，避免日期/字符串混排问题
        x_pos = np.arange(len(grouped))
        ax.plot(x_pos, grouped.values, marker="o", linestyle="-")

        # -------- x 轴刻度与标签 ------------------------------------------
        labels_all = grouped.index.astype(str)
        if len(labels_all) > 15:
            # 均匀抽样不超过 max_xticks 个标签
            step = math.ceil(len(labels_all) / max_xticks)
            tick_idx = np.arange(0, len(labels_all), step)
            if tick_idx[-1] != len(labels_all) - 1:        # 确保最后一个被标注
                tick_idx = np.append(tick_idx, len(labels_all) - 1)
        else:
            tick_idx = np.arange(len(labels_all))

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels_all[tick_idx], rotation=90, fontsize=xtick_fontsize)

        # 轴标题 & 刻度
        ax.set_xlabel(col, fontsize=label_fontsize)
        ax.set_ylabel(f"mean({value_col})", fontsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=xtick_fontsize)

        # 数值标注
        if annotate:
            for x, y in zip(x_pos, grouped.values):
                ax.text(
                    x, y, f"{y/1000:,.0f}k",  # 标注同样使用 k 单位
                    ha="center", va="bottom",
                    fontsize=annotation_fontsize
                )

        # 显著性标记
        p_adj = pvals_adj[idx]
        star = "ns" if np.isnan(p_adj) else ("*" if p_adj < alpha else "ns")
        ax.set_title(
            f"{letters[idx]}) {col}  ({star})",
            loc="left", fontsize=title_fontsize, fontweight="bold",
        )

        plotted_axes.append(ax)

    # -------- 每子图独立 y 轴显示（不再强制共享范围） -------------------------
    # 只对 y 轴刻度标签做 k 单位格式化，但不设置统一 ylim，让每个子图根据自身数据自动缩放
    k_formatter = FuncFormatter(lambda y, _: f"{y/1000:,.0f}k")
    for ax in plotted_axes:
        ax.yaxis.set_major_formatter(k_formatter)
        # 不调用 ax.set_ylim(...) —— 保持 Matplotlib 的自动缩放

    # 删除多余子图（若 n_rows * n_cols > n_charts）
    for j in range(n_charts, n_rows * n_cols):
        fig.delaxes(axes[j])

    # -------- 整体布局 & 保存 ---------------------------------------------
    fig.suptitle(f"Time-Series Plots for {value_col}",
                 fontsize=main_letter_fontsize + 2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_png:
        fname = save_png if isinstance(save_png, str) else "time_series_plots.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, axes


# ---------- 数值型变量 ----------
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 尝试导入多重检验校正函数；若不可用，回退到简单的 Bonferroni
try:
    from statsmodels.stats.multitest import multipletests
    _HAS_MULTITEST = True
except Exception:
    _HAS_MULTITEST = False

def plot_numeric(
    df,
    value_col='price_z',
    cols=None,                     # list of numeric columns to plot; None 表示自动选取
    n_cols=3,
    figsize_per_chart=(4, 3),      # (width, height) for each subplot
    sort_by_count=False,           # 如果 True 按每列非空数量排序
    annotate=True,                 # 在子图上注释 r / p / n
    alpha=0.6,                     # 散点透明度
    p_adjust_method='holm',        # 'holm' / 'bonferroni' / 'fdr_bh' / 'none'
    save_png=False,
    filename='numeric_scatter_grid.png',
    title_fontsize=10,
    label_fontsize=10,
    xtick_fontsize=8,
    annotation_fontsize=8,
    main_letter_fontsize=10,
    add_regression=True,           # 是否绘制拟合直线
    jitter=0.0,                    # x 方向抖动（当某列接近离散时可用）
    marker='o',
    s=20                           # marker size
):
    """
    在 df 中以 cols（或自动选择 numeric 列）为 x，value_col 为 y，
    批量绘制散点图并注释 Pearson r, p-value, n。
    返回：(fig, axes, corr_df) 三元组，其中 corr_df 包含 r, p, p_adj, n。
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in dataframe")

    # 自动选择 numeric 列（排除目标列）
    if cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != value_col]
    else:
        numeric_cols = list(cols)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to plot (after excluding value_col).")

    # 计算每列与 value_col 的 r, p, n（按对丢弃 NaN）
    results = []
    for col in numeric_cols:
        tmp = df[[col, value_col]].dropna()
        n = tmp.shape[0]
        if n >= 2:
            try:
                r, p = pearsonr(tmp[col].values, tmp[value_col].values)
            except Exception:
                r, p = np.nan, np.nan
        else:
            r, p = np.nan, np.nan
        results.append({'feature': col, 'r': r, 'p': p, 'n': n})

    corr_df = pd.DataFrame(results).set_index('feature')

    # optional: sort_by_count
    if sort_by_count:
        corr_df = corr_df.sort_values(by='n', ascending=False)
        numeric_cols = corr_df.index.tolist()

    # p-value 校正
    if p_adjust_method is None:
        p_adjust_method = 'none'
    pvals = corr_df['p'].values
    valid_mask = ~np.isnan(pvals)
    p_adj = np.array([np.nan] * len(pvals), dtype=float)

    if p_adjust_method != 'none' and np.any(valid_mask):
        if _HAS_MULTITEST:
            # statsmodels multipletests 支持 many methods
            try:
                reject, pvals_adj, _, _ = multipletests(pvals[valid_mask], method=p_adjust_method)
                p_adj[valid_mask] = pvals_adj
            except Exception:
                # 回退到 Bonferroni
                p_adj[valid_mask] = np.minimum(1.0, pvals[valid_mask] * np.sum(valid_mask))
        else:
            # fallback: Bonferroni
            p_adj[valid_mask] = np.minimum(1.0, pvals[valid_mask] * np.sum(valid_mask))
    else:
        p_adj = pvals.copy()  # none: 不做校正

    corr_df['p_adj'] = p_adj

    # 绘图布局
    n_plots = len(numeric_cols)
    ncols = max(1, int(n_cols))
    nrows = math.ceil(n_plots / ncols)
    fig_w = figsize_per_chart[0] * ncols
    fig_h = figsize_per_chart[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).reshape(-1)

    # 绘每个子图
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        series_x = df[col]
        series_y = df[value_col]

        # 对应位置逐对 delete NaN
        mask = series_x.notna() & series_y.notna()
        x = series_x.loc[mask].values
        y = series_y.loc[mask].values

        if jitter and len(x) > 0:
            x = x + np.random.normal(scale=jitter, size=x.shape)

        ax.scatter(x, y, alpha=alpha, marker=marker, s=s)

        # 回归线（最小二乘），当 x 有 >=2 点且 add_regression True
        if add_regression and len(x) >= 2:
            try:
                coeffs = np.polyfit(x, y, deg=1)
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xx, np.polyval(coeffs, xx), linestyle='--')
            except Exception:
                pass

        # 标签字体
        ax.set_xlabel(col, fontsize=label_fontsize)
        ax.set_ylabel(value_col if i % ncols == 0 else '', fontsize=label_fontsize)  # 只在每行第1列画 y 标签
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=xtick_fontsize)

        # 注释 r, p, n
        if annotate:
            row = corr_df.loc[col]
            r = row['r']
            p = row['p']
            p_adj_val = row['p_adj']
            n_val = int(row['n'])
            # 格式化字符串
            r_s = f"r={r:.3f}" if not np.isnan(r) else "r=NaN"
            p_s = f"p={p:.3g}" if not np.isnan(p) else "p=NaN"
            padj_s = f"p_adj={p_adj_val:.3g}" if not np.isnan(p_adj_val) else "p_adj=NaN"
            ann = f"{r_s}\n{p_s}\n{padj_s}\nn={n_val}"
            # 置于子图右上角
            ax.annotate(ann, xy=(0.97, 0.97), xycoords='axes fraction',
                        fontsize=annotation_fontsize, horizontalalignment='right',
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        # 子图标题（可选加上主字母）
        ax.set_title(f"{chr(65 + i)}) {col}", fontsize=title_fontsize, loc='left')

    # 清理空子图
    for j in range(n_plots, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()

    if save_png:
        try:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {filename}")
        except Exception as e:
            print("Failed to save figure:", e)

    return fig, axes, corr_df

