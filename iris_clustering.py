"""
IRIS CLUSTERING ANALYSIS
========================
K-means (3 distance variants) vs hierarchical clustering (3 metrics x 3 linkages)

Validation
  Internal (used to CHOOSE method and k): elbow method + silhouette score
  External (used to CHECK the result, after choices are made):
      matched accuracy + Adjusted Rand Index against the species labels
The species labels are never used to fit a model or to pick the winner.

Install once:
    pip install kagglehub[pandas-datasets] pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

# ==============================================================
# 0. CONFIGURATION & REPRODUCIBILITY
# ==============================================================
RANDOM_STATE = 42
N_REPEATS = 10                 # random restarts for k-means / k-medoids
K_RANGE = list(range(2, 11))   # k = 2..10
K_CHOSEN = 3                   # final k (justified in section 4E)
TIE_TOLERANCE = 0.02           # silhouette gap treated as a tie in section 4D
DROP_DUPLICATES = False        # True removes duplicate rows before clustering
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# False (default): figures are only saved to the outputs/ folder and the script never pauses.
# True: each figure opens in a window and the script WAITS until you close it.
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def save_fig(name, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    print(f"  [figure saved: outputs/{name}]")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ==============================================================
# 3.0  LOAD DATA
# ==============================================================
def standardise_columns(df):
    """Map any common Iris column spelling onto clean snake_case names."""
    rename, drop = {}, []
    for c in df.columns:
        key = "".join(ch for ch in str(c).lower() if ch.isalnum())
        if "sepal" in key and "length" in key:
            rename[c] = "sepal_length"
        elif "sepal" in key and "width" in key:
            rename[c] = "sepal_width"
        elif "petal" in key and "length" in key:
            rename[c] = "petal_length"
        elif "petal" in key and "width" in key:
            rename[c] = "petal_width"
        elif key in ("species", "class", "variety", "target"):
            rename[c] = "species"
        elif key == "id":
            drop.append(c)
    df = df.drop(columns=drop).rename(columns=rename)
    return df[FEATURES + ["species"]]


def load_data():
    banner("3.0  LOADING DATA")
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, "arshid/iris-flower-dataset", "IRIS.csv"
        )
        print("Loaded from Kaggle (arshid/iris-flower-dataset)")
    except Exception as exc:  # no internet / no kagglehub / wrong file name
        print(f"Kaggle load failed ({type(exc).__name__}); using sklearn's copy of iris")
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=FEATURES)
        df["species"] = [iris.target_names[i] for i in iris.target]
    df = standardise_columns(df)
    df["species"] = df["species"].astype(str).str.replace("Iris-", "", regex=False).str.strip()
    print("First 5 records:")
    print(df.head())
    return df


# ==============================================================
# 3.1  DATA QUALITY
# ==============================================================
def data_quality(df):
    banner("3.1  DATA QUALITY")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    if df.isnull().any().any():
        df = df.dropna().reset_index(drop=True)
        print(f"Rows with missing values removed; {len(df)} rows remain")
    else:
        print("No missing values: nothing to remove")
    n_dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dup}")
    if n_dup:
        print(df[df.duplicated(keep=False)].sort_values(FEATURES))
        if DROP_DUPLICATES:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"Duplicate rows removed; {len(df)} rows remain")
        else:
            print("(kept: DROP_DUPLICATES is False; duplicates are real measurements)")
    print("\nClass balance:\n", df["species"].value_counts())
    return df


# ==============================================================
# 3.2  EXPLORATORY DATA ANALYSIS
# ==============================================================
def eda(df):
    banner("3.2  EXPLORATORY DATA ANALYSIS")
    species = list(df["species"].unique())

    # a. descriptive statistics -------------------------------------------------
    print("\n[a] Descriptive statistics (overall)")
    print(df[FEATURES].describe().round(2))
    print("\nDescriptive statistics by species")
    print(df.groupby("species")[FEATURES].agg(["mean", "std", "min", "max"]).round(2).T)
    cv = (df[FEATURES].std() / df[FEATURES].mean() * 100).round(1)
    print("\nCoefficient of variation (%), std/mean:\n", cv)

    # b. distributions ----------------------------------------------------------
    print("\n[b] Distributions")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, f in enumerate(FEATURES):
        sns.histplot(data=df, x=f, hue="species", kde=True, ax=axes[0, i])
        sns.boxplot(data=df, x="species", y=f, ax=axes[1, i])
        axes[0, i].set_title(f"{f}: distribution")
        axes[1, i].set_title(f"{f}: by species")
    save_fig("eda_distributions.png", fig)

    sk = pd.DataFrame({"skew": df[FEATURES].skew(), "excess_kurtosis": df[FEATURES].kurt()}).round(3)
    print("\nSkewness / kurtosis:\n", sk)

    rows = []
    for f in FEATURES:
        for s in species:
            w, p = stats.shapiro(df.loc[df.species == s, f])
            rows.append({"feature": f, "species": s, "W": round(w, 3), "p_value": round(p, 4),
                         "normal_at_5%": p > 0.05})
    print("\nShapiro-Wilk per species:\n", pd.DataFrame(rows).to_string(index=False))

    print("\nOutliers (IQR rule and |z| > 3)")
    any_out = False
    for f in FEATURES:
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_out = df[(df[f] < q1 - 1.5 * iqr) | (df[f] > q3 + 1.5 * iqr)]
        z_out = df[np.abs(stats.zscore(df[f])) > 3]
        if len(iqr_out) or len(z_out):
            any_out = True
            print(f"  {f}: IQR outliers at rows {list(iqr_out.index)}, z>3 at rows {list(z_out.index)}")
    if not any_out:
        print("  none found")
    print("  (outliers are only listed, not removed: they are genuine flowers)")

    # c. relationships ----------------------------------------------------------
    print("\n[c] Relationships between features")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, method in zip(axes, ["pearson", "spearman"], strict=True):
        corr = df[FEATURES].corr(method=method)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{method.capitalize()} correlation")
    save_fig("eda_correlation.png", fig)
    corr = df[FEATURES].corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
             if abs(corr.loc[a, b]) > 0.8]
    print("Highly correlated pairs (|r| > 0.8):", [(a, b, round(r, 2)) for a, b, r in pairs])

    g = sns.pairplot(df, hue="species", corner=False, diag_kind="kde")
    g.figure.suptitle("Pairplot by species", y=1.02)
    save_fig("eda_pairplot.png", g.figure)

    # d. ANOVA ------------------------------------------------------------------
    print("\n[d] Which features separate the species? (one-way ANOVA)")
    rows = []
    for f in FEATURES:
        groups = [df.loc[df.species == s, f].values for s in species]
        F, p = stats.f_oneway(*groups)
        grand = df[f].mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        sst = ((df[f] - grand) ** 2).sum()
        rows.append({
            "feature": f, "F": round(F, 1), "p_value": p, "eta_squared": round(ssb / sst, 3),
            "levene_p": round(stats.levene(*groups)[1], 4),
            "kruskal_H": round(stats.kruskal(*groups)[0], 1),
            "kruskal_p": stats.kruskal(*groups)[1],
        })
    anova = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    print(anova.to_string(index=False))
    print("\nRanking by effect size (eta squared):", list(anova["feature"]))
    print("Levene p < 0.05 means unequal variances: the Kruskal-Wallis columns are the robust check.")

    try:
        print("\nTukey HSD pairwise p-values (which species pairs differ?)")
        for f in FEATURES:
            res = stats.tukey_hsd(*[df.loc[df.species == s, f].values for s in species])
            pw = {f"{species[i]} vs {species[j]}": float(res.pvalue[i, j])
                  for i in range(len(species)) for j in range(i + 1, len(species))}
            print(f"  {f}: " + ", ".join(f"{k}: p={v:.4f}" for k, v in pw.items()))
    except Exception as exc:
        print("  Tukey HSD skipped:", exc)

    # e. PCA --------------------------------------------------------------------
    print("\n[e] PCA on standardised features")
    Xs = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)
    evr = pca.explained_variance_ratio_
    print("Explained variance ratio:", np.round(evr, 3), "| cumulative:", np.round(np.cumsum(evr), 3))
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[f"PC{i+1}" for i in range(4)]).round(3)
    print("\nLoadings (which features drive each component):\n", loadings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, 5), evr * 100, color="steelblue")
    axes[0].plot(range(1, 5), np.cumsum(evr) * 100, "o-", color="darkorange", label="cumulative")
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Scree plot"); axes[0].legend()
    X2 = pca.transform(Xs)[:, :2]
    for s in species:
        m = (df["species"] == s).values
        axes[1].scatter(X2[m, 0], X2[m, 1], label=s, alpha=0.8, edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axes[1].set_title("PCA 2D by species"); axes[1].legend()
    save_fig("eda_pca.png", fig)

    # f. scaling decision -------------------------------------------------------
    print("\n[f] Scaling decision")
    print(pd.DataFrame({"range": df[FEATURES].max() - df[FEATURES].min(),
                        "variance": df[FEATURES].var()}).round(3))
    print("Features are all in cm but their variances differ; standardising stops the")
    print("highest-variance feature from dominating every distance calculation.")

    # g. summary: deliberately NOT auto-written ---------------------------------
    print("\n[g] EDA SUMMARY: CHECKPOINT. Conclusions to be written together after reviewing the output above.")


# ==============================================================
# HELPERS FOR SECTION 4
# ==============================================================
def sil(X, labels, metric):
    """Silhouette that does not crash when a cut returns fewer than 2 clusters."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(X, labels, metric=metric)


def knee_point(ks, costs):
    """Elbow location: the k whose point lies furthest below the straight line
    joining the first and last point of the (normalised) cost curve."""
    x = np.array(ks, float)
    y = np.array(costs, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return int(x[np.argmax((1 - xn) - yn)])


def second_diff_k(ks, costs):
    """Elbow location by the largest second difference of the cost curve
    (the k where the curve bends most sharply)."""
    c = np.array(costs, float)
    sd = c[:-2] - 2 * c[1:-1] + c[2:]
    return int(np.array(ks)[1:-1][np.argmax(sd)])


def merge_gap_k(Z, kmax=10):
    """Hierarchical 'elbow': the k with the biggest jump in merge height.
    h[0] is the height of the final merge (2 -> 1 clusters)."""
    h = Z[::-1, 2]
    gaps = h[:kmax - 1] - h[1:kmax]
    return int(np.argmax(gaps) + 2)


def merge_heights_by_k(Z, kmax=10):
    """Height needed to merge k clusters into k-1, for k = 2..kmax."""
    h = Z[::-1, 2]
    return {k: h[k - 2] for k in range(2, kmax + 1)}


def kmedoids(D, k, n_init=N_REPEATS, max_iter=300, seed=RANDOM_STATE):
    """Simple k-medoids (alternating assign / update) on a precomputed distance matrix."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    best_cost, best_labels = np.inf, None
    for _ in range(n_init):
        medoids = [int(rng.integers(n))]
        while len(medoids) < k:                       # k-medoids++ style start
            d = D[:, medoids].min(axis=1) ** 2
            medoids.append(int(rng.choice(n, p=d / d.sum())) if d.sum() > 0 else int(rng.integers(n)))
        medoids = np.array(medoids)
        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new = medoids.copy()
            for j in range(k):
                members = np.where(labels == j)[0]
                if len(members):
                    new[j] = members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]
            if np.array_equal(new, medoids):
                break
            medoids = new
        labels = np.argmin(D[:, medoids], axis=1)
        cost = D[np.arange(n), medoids[labels]].sum()
        if cost < best_cost:
            best_cost, best_labels = cost, labels
    return best_labels, best_cost


# ==============================================================
# 4A  PARTITIONING METHODS (three variants, one per distance metric)
# ==============================================================
def build_partitioning(X_scaled):
    X_sph = normalize(X_scaled)                        # unit-length rows -> spherical k-means
    D_man = squareform(pdist(X_scaled, "cityblock"))

    def km_euclid(k, seed=RANDOM_STATE, n_init=N_REPEATS):
        km = KMeans(k, init="k-means++", max_iter=300, n_init=n_init, random_state=seed).fit(X_scaled)
        return km.labels_, km.inertia_

    def km_cosine(k, seed=RANDOM_STATE, n_init=N_REPEATS):
        km = KMeans(k, init="k-means++", max_iter=300, n_init=n_init, random_state=seed).fit(X_sph)
        return km.labels_, km.inertia_

    def km_manhattan(k, seed=RANDOM_STATE, n_init=N_REPEATS):
        return kmedoids(D_man, k, n_init=n_init, seed=seed)

    return {
        "KM-Euclid": dict(fit=km_euclid, metric="euclidean", metric_name="Euclidean"),
        "KM-Cosine": dict(fit=km_cosine, metric="cosine", metric_name="Cosine"),
        "KM-Manhat": dict(fit=km_manhattan, metric="cityblock", metric_name="Manhattan"),
    }


def run_partitioning(X_scaled, variants):
    banner("4A  PARTITIONING METHODS: elbow + silhouette, k = 2..10")
    rows, labels_at_k = [], {}
    for name, v in variants.items():
        for k in K_RANGE:
            labels, cost = v["fit"](k)
            rows.append({"variant": name, "k": k, "cost": cost,
                         "sil_own": sil(X_scaled, labels, v["metric"]),
                         "sil_euclid": sil(X_scaled, labels, "euclidean")})
            labels_at_k[(name, k)] = labels
    part = pd.DataFrame(rows)

    knees = {n: knee_point(K_RANGE, part[part.variant == n]["cost"]) for n in variants}
    knees2 = {n: second_diff_k(K_RANGE, part[part.variant == n]["cost"]) for n in variants}
    print(part.round(4).to_string(index=False))
    print("\nElbow per variant, kneedle (distance from chord):", knees)
    print("Elbow per variant, largest second difference    :", knees2)
    for n in variants:
        sub = part[part.variant == n]
        print(f"  {n}: best k by own-metric silhouette = {int(sub.loc[sub.sil_own.idxmax(), 'k'])}, "
              f"by Euclidean-ruler silhouette = {int(sub.loc[sub.sil_euclid.idxmax(), 'k'])}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, n in enumerate(variants):
        sub = part[part.variant == n]
        axes[0, j].plot(sub.k, sub.cost, "o-", color="steelblue")
        axes[0, j].axvline(knees[n], color="red", ls="--", label=f"kneedle k={knees[n]}")
        axes[0, j].axvline(knees2[n], color="green", ls="-.", label=f"2nd difference k={knees2[n]}")
        axes[0, j].axvline(K_CHOSEN, color="gray", ls=":", label=f"chosen k={K_CHOSEN}")
        axes[0, j].set_title(f"{n}: elbow"); axes[0, j].set_xlabel("k")
        axes[0, j].set_ylabel("cost (inertia / sum of distances)"); axes[0, j].legend()
        axes[1, j].plot(sub.k, sub.sil_own, "o-", color="darkorange", label="own metric")
        axes[1, j].plot(sub.k, sub.sil_euclid, "s--", color="green", label="Euclidean ruler")
        axes[1, j].axvline(K_CHOSEN, color="gray", ls=":")
        axes[1, j].set_title(f"{n}: silhouette"); axes[1, j].set_xlabel("k"); axes[1, j].legend()
    save_fig("4A_elbow_silhouette_partitioning.png", fig)
    return part, labels_at_k, knees, knees2


# ==============================================================
# 4B  HIERARCHICAL CLUSTERING (metric x linkage grid)
# ==============================================================
HIER_METRICS = {"Euclidean": "euclidean", "Manhattan": "cityblock", "Cosine": "cosine"}
LINKAGES = ["ward", "average", "complete"]


def run_hierarchical(X_scaled):
    banner("4B  HIERARCHICAL CLUSTERING: metric x linkage grid, k = 2..10")
    rows, Zs, gap_k = [], {}, {}
    for mname, metric in HIER_METRICS.items():
        dist = pdist(X_scaled, metric)
        for link in LINKAGES:
            if link == "ward" and metric != "euclidean":
                continue                                  # Ward is only valid with Euclidean
            Z = linkage(dist, link)
            Zs[(mname, link)] = Z
            coph = cophenet(Z, dist)[0]
            gap_k[(mname, link)] = merge_gap_k(Z)
            for k in K_RANGE:
                labels = fcluster(Z, k, "maxclust")
                rows.append({"metric": mname, "linkage": link, "config": f"{mname}-{link}", "k": k,
                             "n_clusters": len(np.unique(labels)),
                             "sil_own": sil(X_scaled, labels, metric),
                             "sil_euclid": sil(X_scaled, labels, "euclidean"),
                             "cophenetic": coph})
    hier = pd.DataFrame(rows)
    print(hier.groupby("config")[["cophenetic"]].first().round(4).rename(
        columns={"cophenetic": "cophenetic correlation"}))
    print("\nMerge-height 'elbow' (largest jump) suggests k =", {f"{a}-{b}": v for (a, b), v in gap_k.items()})

    # heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, col, title in zip(axes, ["sil_euclid", "sil_own"],
                              ["Silhouette (Euclidean ruler)", "Silhouette (own metric)"], strict=True):
        pv = hier.pivot(index="config", columns="k", values=col)
        sns.heatmap(pv, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Hierarchical: {title}")
    save_fig("4B_hierarchical_silhouette_heatmaps.png", fig)

    # best linkage per metric at the chosen k (common ruler)
    best_cfg = {}
    for mname in HIER_METRICS:
        sub = hier[(hier.metric == mname) & (hier.k == K_CHOSEN)]
        best_cfg[mname] = sub.loc[sub.sil_euclid.idxmax(), "linkage"]
    print(f"\nBest linkage per metric at k={K_CHOSEN} (Euclidean ruler):", best_cfg)

    # merge-height elbow analogue
    fig, ax = plt.subplots(figsize=(9, 5))
    for mname, link in best_cfg.items():
        mh = merge_heights_by_k(Zs[(mname, link)])
        h = np.array(list(mh.values()))
        ax.plot(list(mh.keys()), h / h.max(), "o-", label=f"{mname}-{link}")
    ax.axvline(K_CHOSEN, color="gray", ls=":", label=f"chosen k={K_CHOSEN}")
    ax.set_xlabel("k (clusters before the merge)"); ax.set_ylabel("merge height (scaled to max = 1)")
    ax.set_title("Hierarchical 'elbow': height needed to merge k clusters into k-1"); ax.legend()
    save_fig("4B_merge_heights.png", fig)

    # dendrograms
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, (mname, link) in zip(axes, best_cfg.items(), strict=False):
        Z = Zs[(mname, link)]
        cut = (Z[-(K_CHOSEN - 1), 2] + Z[-K_CHOSEN, 2]) / 2
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9,
                   show_contracted=True, color_threshold=cut, ax=ax)
        ax.axhline(cut, color="red", ls="--", label=f"cut -> {K_CHOSEN} clusters")
        ax.set_title(f"Dendrogram: {mname} + {link}"); ax.set_ylabel(f"distance ({mname})")
        ax.legend()
    save_fig("4B_dendrograms.png", fig)
    return hier, Zs, gap_k, best_cfg


# ==============================================================
# 4C-4D  METRIC COMPARISON & METHOD DECISION (internal measures only)
# ==============================================================
def build_candidates(X_scaled, variants, part, labels_at_k, hier, Zs, best_cfg):
    candidates = {}
    for name, v in variants.items():
        row = part[(part.variant == name) & (part.k == K_CHOSEN)].iloc[0]
        candidates[name] = dict(labels=labels_at_k[(name, K_CHOSEN)], family="Partitioning",
                                metric=v["metric_name"], config=name,
                                sil_own=row.sil_own, sil_euclid=row.sil_euclid)
    for mname, link in best_cfg.items():
        row = hier[(hier.metric == mname) & (hier.linkage == link) & (hier.k == K_CHOSEN)].iloc[0]
        name = f"HC-{mname}-{link}"
        candidates[name] = dict(labels=fcluster(Zs[(mname, link)], K_CHOSEN, "maxclust"),
                                family="Hierarchical", metric=mname, config=name,
                                sil_own=row.sil_own, sil_euclid=row.sil_euclid)
    return candidates


def decide_method(candidates, variants, part, labels_fit_fns):
    banner(f"4C-4D  METRIC COMPARISON & METHOD DECISION (k = {K_CHOSEN}, internal measures only)")
    table = pd.DataFrame([{k: v for k, v in c.items() if k != "labels"} | {"model": k}
                          for k, c in candidates.items()])
    table["complexity"] = (np.where(table.metric == "Euclidean", 0, 2)
                           + np.where(table.family == "Partitioning", 0, 1))
    table = table[["model", "family", "metric", "sil_own", "sil_euclid", "complexity"]]
    print(table.round(4).to_string(index=False))

    # Metric x family comparison
    cmp = table.pivot(index="metric", columns="family", values="sil_euclid").round(4)
    print("\nEuclidean-ruler silhouette by metric and family:\n", cmp)
    own = table.pivot(index="metric", columns="family", values="sil_own").round(4)
    print("\nOwn-metric silhouette by metric and family (NOT comparable across rows):\n", own)

    # Decision rule, fixed in advance
    print(f"\nDECISION RULE: highest Euclidean-ruler silhouette at k={K_CHOSEN}; models within "
          f"{TIE_TOLERANCE} of the top are treated as tied and the simpler one "
          "(Euclidean before other metrics, partitioning before hierarchical) wins.")
    top = table.sil_euclid.max()
    tied = table[table.sil_euclid >= top - TIE_TOLERANCE].sort_values(["complexity", "sil_euclid"],
                                                                      ascending=[True, False])
    winner = tied.iloc[0]["model"]
    print("Models in the tie band:", list(tied["model"]))
    print(f">>> WINNER by internal measures: {winner}")

    # Stability across seeds (partitioning variants)
    print(f"\nStability: {N_REPEATS} single-start runs per partitioning variant at k={K_CHOSEN}")
    stab = []
    for name, v in variants.items():
        scores = [sil(X_SCALED_GLOBAL, v["fit"](K_CHOSEN, seed=RANDOM_STATE + r, n_init=1)[0], "euclidean")
                  for r in range(N_REPEATS)]
        stab.append({"variant": name, "mean": np.mean(scores), "std": np.std(scores),
                     "min": np.min(scores), "max": np.max(scores)})
    print(pd.DataFrame(stab).round(4).to_string(index=False))

    # bar chart on the common ruler
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table.sil_euclid, 0.4, label="Euclidean ruler", color="steelblue")
    ax.bar(x + 0.2, table.sil_own, 0.4, label="own metric", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(table.model, rotation=30, ha="right")
    ax.set_ylabel("silhouette"); ax.legend()
    ax.set_title(f"Candidate comparison at k={K_CHOSEN}")
    save_fig("4C_candidate_comparison.png", fig)
    return table, winner


# ==============================================================
# 4E  CHOOSE k
# ==============================================================
def choose_k(part, hier, knees, knees2, gap_k, best_cfg):
    banner("4E  CHOOSING k")
    print("Evidence table (elbow_k: kneedle for partitioning, merge-height jump for hierarchical)")
    rows = []
    for n in part.variant.unique():
        sub = part[part.variant == n]
        rows.append({"model": n, "elbow_k": knees[n], "elbow_k_2nd_diff": knees2[n],
                     "best_k_silhouette_own": int(sub.loc[sub.sil_own.idxmax(), "k"]),
                     "best_k_silhouette_euclid": int(sub.loc[sub.sil_euclid.idxmax(), "k"])})
    for mname, link in best_cfg.items():
        sub = hier[(hier.metric == mname) & (hier.linkage == link)]
        rows.append({"model": f"HC-{mname}-{link}", "elbow_k": gap_k[(mname, link)],
                     "best_k_silhouette_own": int(sub.loc[sub.sil_own.idxmax(), "k"]),
                     "best_k_silhouette_euclid": int(sub.loc[sub.sil_euclid.idxmax(), "k"])})
    ev = pd.DataFrame(rows)
    print(ev.to_string(index=False))
    print(f"\nChosen k = {K_CHOSEN}.")
    print("CHECKPOINT: the written justification (k=2 vs k=3) is to be agreed together,")
    print("using this table, the dendrograms, the merge-height plot and the EDA/PCA view.")
    return ev


# ==============================================================
# 4F  EXTERNAL CHECK (after all choices are made)
# ==============================================================
def external_check(df, candidates):
    banner(f"4F  EXTERNAL CHECK vs species (k = {K_CHOSEN})")
    species = sorted(df["species"].unique())
    truth = pd.Categorical(df["species"], categories=species).codes
    rows, crosstabs = [], {}
    for name, c in candidates.items():
        labels = np.asarray(c["labels"])
        ct = pd.crosstab(pd.Series(labels, name="cluster"), df["species"])[species]
        crosstabs[name] = ct
        # one-to-one cluster -> species mapping that maximises correct flowers (Hungarian method)
        r, col = linear_sum_assignment(-ct.values)
        mapping = {ct.index[i]: species[j] for i, j in zip(r, col, strict=True)}
        correct = sum(ct.values[i, j] for i, j in zip(r, col, strict=True))
        row = {"model": name, "matched_accuracy": correct / len(df),
               "ARI": adjusted_rand_score(truth, labels)}
        for j, s in enumerate(species):
            hit = [ct.values[i, jj] for i, jj in zip(r, col, strict=True) if jj == j]
            row[f"recall_{s}"] = (hit[0] / ct[s].sum()) if hit else 0.0
        row["mapping"] = str(mapping)
        rows.append(row)
        print(f"\n--- {name} ---")
        print(ct)
        print("Cluster -> species mapping:", mapping)
        print(f"Matched accuracy: {row['matched_accuracy']:.3f} | ARI: {row['ARI']:.3f}")
    ext = pd.DataFrame(rows)
    print("\nSummary:\n", ext.drop(columns="mapping").round(3).to_string(index=False))
    return ext, crosstabs


# ==============================================================
# 5  OUTPUTS
# ==============================================================
def cluster_plot(df, X_scaled, candidates, winner):
    pca = PCA(2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    panels = [(n, c["labels"]) for n, c in candidates.items()]
    panels.append(("Ground truth (species)", pd.factorize(df["species"])[0]))
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (title, labels) in zip(axes.ravel(), panels, strict=False):
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="Set1", s=35, edgecolor="k", linewidth=0.3)
        ax.set_title(title + ("  [WINNER]" if title == winner else ""),
                     fontweight="bold" if title == winner else "normal")
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    save_fig("5_cluster_visualisation.png", fig)


def save_outputs(df, candidates, part, hier, cand_table, evidence, ext, crosstabs):
    banner("5  SAVING OUTPUTS")
    out = df.copy()
    for name, c in candidates.items():
        out[f"{name}_k{K_CHOSEN}"] = np.asarray(c["labels"]) + (1 if np.min(c["labels"]) == 0 else 0)
    csv_path = os.path.join(OUT_DIR, "iris_with_clusters.csv")
    out.to_csv(csv_path, index=False)
    print("Saved", csv_path)

    xlsx_path = os.path.join(OUT_DIR, "iris_clustering_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Data + Clusters", index=False)
        r = 0
        for title, tab in [("Partitioning: elbow & silhouette by k", part.round(4)),
                           ("Hierarchical grid: silhouette by config & k", hier.round(4)),
                           (f"Candidates at k={K_CHOSEN}", cand_table.round(4)),
                           ("Evidence for choosing k", evidence)]:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name="Internal validation", startrow=r, index=False, header=False)
            tab.to_excel(xw, sheet_name="Internal validation", startrow=r + 1, index=False)
            r += len(tab) + 4
        r = 0
        ext.round(3).to_excel(xw, sheet_name="External check", startrow=r, index=False)
        r += len(ext) + 3
        for name, ct in crosstabs.items():
            pd.DataFrame([[name]]).to_excel(xw, sheet_name="External check", startrow=r, index=False, header=False)
            ct.to_excel(xw, sheet_name="External check", startrow=r + 1)
            r += len(ct) + 4
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    print("Saved", xlsx_path)
    print("Figures saved in", OUT_DIR)


# ==============================================================
# MAIN
# ==============================================================
def main():
    global X_SCALED_GLOBAL
    df = load_data()
    df = data_quality(df)
    eda(df)

    banner("3.3  SCALING")
    X_scaled = StandardScaler().fit_transform(df[FEATURES])
    X_SCALED_GLOBAL = X_scaled
    print(f"After StandardScaler: mean ~ {X_scaled.mean():.6f}, std ~ {X_scaled.std():.4f}")

    variants = build_partitioning(X_scaled)
    part, labels_at_k, knees, knees2 = run_partitioning(X_scaled, variants)
    hier, Zs, gap_k, best_cfg = run_hierarchical(X_scaled)

    candidates = build_candidates(X_scaled, variants, part, labels_at_k, hier, Zs, best_cfg)
    cand_table, winner = decide_method(candidates, variants, part, None)
    evidence = choose_k(part, hier, knees, knees2, gap_k, best_cfg)

    ext, crosstabs = external_check(df, candidates)
    internal_winner_row = ext[ext.model == winner].iloc[0]
    best_acc = ext.loc[ext.matched_accuracy.idxmax()]
    print(f"\nInternal winner {winner}: accuracy {internal_winner_row.matched_accuracy:.3f}, "
          f"ARI {internal_winner_row.ARI:.3f}")
    print(f"Highest external accuracy: {best_acc.model} ({best_acc.matched_accuracy:.3f})")
    print("If these differ, report it honestly; the winner is NOT switched after seeing labels.")

    cluster_plot(df, X_scaled, candidates, winner)
    save_outputs(df, candidates, part, hier, cand_table, evidence, ext, crosstabs)


if __name__ == "__main__":
    main()