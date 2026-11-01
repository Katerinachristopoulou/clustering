# Iris Clustering Analysis

## 1. Project overview

The goal of this project was to group 150 iris flowers without using their species labels, in order to find their natural groupings. I compared k-means and hierarchical clustering. The question answered was which method, which distance metric and which number of clusters is optimal for this dataset.

The number of clusters was chosen mainly with the elbow method, and the methods and distance metrics were compared with the silhouette score. The species labels were used only afterwards to check how well the clusters match the real species.

**Main results.**

* The internal winner at $k = 3$ is `HC-Euclidean-average` (hierarchical clustering, Euclidean distance, average linkage), with a Euclidean silhouette of 0.480.
* $k = 3$ was chosen with the elbow method (the second-difference elbow gives $k = 3$ for all three partitioning variants) and because the dataset has three species. The silhouette alone would prefer $k = 2$ (see 7.6).
* Against the real species, the winner reaches a matched accuracy of 0.687 and an ARI of 0.562, because one of its three clusters holds only 3 flowers. The best external match is `HC-Manhattan-average`, with an accuracy of 0.887 and an ARI of 0.718. The winner was not changed after seeing the labels (see 7.7).

## 2. Files and outputs

* `iris_clustering.py` is the code.
* `requirements.txt` lists the libraries to install.
* `outputs/` holds the results produced by the script. It is created automatically next to the script.

### Figures (10 `.png` files)

* `eda_distributions.png`: histograms and boxplots of each feature by species.
* `eda_correlation.png`: Pearson and Spearman correlation heatmaps.
* `eda_pairplot.png`: scatter plot of every pair of features with the distribution of each feature on the diagonal. It shows the relationships between features and how well the species separate.
* `eda_pca.png`: scree plot and 2D PCA scatter by species.
* `4A_elbow_silhouette_partitioning.png`: the elbow (a practical, visual way to choose k) and the silhouette score (a calculated measure of how well separated the clusters are) for the three k-means variants.
* `4B_hierarchical_silhouette_heatmaps.png`: silhouette heatmaps for every metric and linkage combination.
* `4B_merge_heights.png`: the distance at which clusters are merged in the hierarchical trees. A big jump suggests the number of clusters, so it is the elbow equivalent for hierarchical clustering.
* `4B_dendrograms.png`: dendrograms showing how the flowers are linked into clusters, with the cut line at the chosen k.
* `4C_candidate_comparison.png`: the silhouette statistics of the six finalists at k=3.
* `5_cluster_visualisation.png`: PCA view of each finalist's clusters next to the real species.

### Data files

* `iris_with_clusters.csv`: the dataset with the cluster of each flower added as new columns, one for each finalist (for example `KM-Euclid_k3`).
* `iris_clustering_results.xlsx`: three sheets.
  * `Data + Clusters`: the same data as the csv file, the dataset with its final clustering.
  * `Internal validation`: the elbow and silhouette results used to choose the number of clusters and the method.
  * `External check`: the comparison of the clusters with the real species using accuracy and ARI (formulas below).

### Formulas used in the external check

Matched accuracy. Each cluster is matched to one species (one-to-one) so that the number of correctly placed flowers is as large as possible (Hungarian method).

$$\text{Accuracy} = \frac{\sum_j n_{j,\sigma(j)}}{n}$$

where $n_{j,\sigma(j)}$ is the number of flowers in cluster $j$ that belong to the species matched to it, and $n = 150$. In words: correct flowers divided by all flowers.

Adjusted Rand Index (ARI), the agreement between the clusters and the species, corrected for chance.

$$\text{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \dfrac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}}{\dfrac{1}{2}\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - \dfrac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}}$$

where $n_{ij}$ is the number of flowers in cluster $i$ and species $j$, $a_i$ is the size of cluster $i$, and $b_j$ is the size of species $j$. ARI is 1 for a perfect match and about 0 for a random grouping.

## 3. Installation and how to run

The script needs Python 3.10 or newer.

1. Install the libraries: `pip install -r requirements.txt`
2. Run the script: `python iris_clustering.py` (or use the Run button in VS Code).

**Data.** The script downloads `IRIS.csv` from the Kaggle dataset `arshid/iris-flower-dataset` with `kagglehub`. The first time, Kaggle may ask you to log in. If the download fails, the script uses the copy of iris that comes with scikit-learn (the same 150 flowers) and prints which source it used.

**Figures.** By default the figures are only saved in `outputs/` and the script never stops. To see them on screen, set `SHOW_PLOTS = True` near the top of the script. The script then waits until you close each figure window.

## 4. Configuration

The settings are at the top of `iris_clustering.py`. Changing them changes the whole analysis.

| Setting | Value | What it does |
| --- | --- | --- |
| `RANDOM_STATE` | 42 | Keeps the random trials stable. K-means and k-medoids start from random points, and the fixed seed makes every run give the same results. |
| `N_REPEATS` | 10 | The number of times k-means and k-medoids are restarted from different random starting points. The best run is kept. It is also the number of runs in the stability check. |
| `K_RANGE` | 2 to 10 | The range of numbers of clusters (k) that is tested for every method. |
| `K_CHOSEN` | 3 | The number of clusters used to compare the six finalists and to check them against the real species. |
| `TIE_TOLERANCE` | 0.02 | Two models whose silhouette scores differ by less than this are treated as tied, and the simpler one is chosen (Euclidean before other metrics, k-means before hierarchical). |
| `DROP_DUPLICATES` | `False` | `False` keeps duplicate rows. `True` removes them before the analysis (see section 5). |
| `FEATURES` | 4 columns | The measurements used for clustering: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`. The species column is never used. |
| `OUT_DIR` | `outputs/` | The folder where all results are saved. |
| `SHOW_PLOTS` | `False` | `False` only saves the figures. `True` also shows them on screen (see section 3). |

All the numbers in this README come from a run with these settings and the Kaggle copy of the data.

## 5. Dataset

**Source.** The classic iris dataset, downloaded from the Kaggle dataset `arshid/iris-flower-dataset`. It has 150 rows: 3 species (setosa, versicolor, virginica) with 50 flowers each. Each flower has 4 measurements in cm (sepal length, sepal width, petal length, petal width) and a species label.

**Loading (`load_data`).** The data is downloaded with `kagglehub`. If the download fails, the script uses the copy of iris that comes with scikit-learn and prints a message saying so. The species names are cleaned by removing the `Iris-` prefix, so they appear as `setosa`, `versicolor` and `virginica`.

**Column standardisation (`standardise_columns`).** Different copies of the dataset name their columns differently (for example `SepalLengthCm` or `sepal length (cm)`). This function renames them all to `sepal_length`, `sepal_width`, `petal_length`, `petal_width` and `species`, removes the `Id` column if there is one, and keeps only these five columns. The rest of the code can then work with any copy of the data.

**Quality checks (`data_quality`).** Before any analysis the script prints:

* The shape and data types of the dataset.
* Missing values in each column. Rows with missing values are removed, because clustering cannot use them.
* Duplicate rows. They are always listed. By default they are kept, because they are real measurements of different flowers. If `DROP_DUPLICATES = True` they are removed.
* The class balance, the number of flowers in each species.

The function returns the cleaned dataset, and every later step uses it. When rows are removed, the script prints how many rows remain.

**Result.** The dataset has shape (150, 5) and no missing values, so no rows are removed. There are 3 duplicate rows: the setosa flower (4.9, 3.1, 1.5, 0.1) appears three times (rows 9, 34 and 37) and the virginica flower (5.8, 2.7, 5.1, 1.9) appears twice (rows 101 and 142). They are kept. The classes are balanced, with 50 flowers of each species.

## 6. Exploratory data analysis (`eda`)

The function `eda` looks at the data before any clustering, to see what to expect: how the features are distributed, how they relate to each other, which ones separate the species, and whether scaling is needed. It has seven steps, a to g. The species labels are used here only to describe the data and are never used to fit a clustering model.

The tables are printed in the terminal. The four figures are saved in `outputs/`: `eda_distributions.png`, `eda_correlation.png`, `eda_pairplot.png` and `eda_pca.png`.

In the formulas, $N$ is the number of flowers, $x_i$ is one measurement, $\bar{x}$ is its mean and $s$ is its standard deviation.

### 6.1 Descriptive statistics (step a)

The script prints the count, mean, standard deviation, minimum, quartiles and maximum of each feature, first for all flowers and then for each species (mean, standard deviation, minimum and maximum). It also prints the coefficient of variation.

$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad s = \sqrt{\frac{\sum_{i=1}^{N}(x_i-\bar{x})^2}{N-1}} \qquad CV = \frac{s}{\bar{x}} \times 100\%$$

The mean is the centre of a feature, the standard deviation is how spread out it is, and the coefficient of variation compares the spread of features that have different average sizes. A large difference between the species means for a feature is the first sign that the feature can separate the species.

**Result.**

| Feature | Mean (all) | SD (all) | Min | Max | CV | Mean setosa | Mean versicolor | Mean virginica |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sepal_length | 5.84 | 0.83 | 4.3 | 7.9 | 14.2% | 5.01 | 5.94 | 6.59 |
| sepal_width | 3.05 | 0.43 | 2.0 | 4.4 | 14.2% | 3.42 | 2.77 | 2.97 |
| petal_length | 3.76 | 1.76 | 1.0 | 6.9 | 46.9% | 1.46 | 4.26 | 5.55 |
| petal_width | 1.20 | 0.76 | 0.1 | 2.5 | 63.7% | 0.24 | 1.33 | 2.03 |

The petal features have the largest coefficients of variation and the largest differences between the species means. Setosa has much smaller petals than the other two species (maximum petal length 1.9 cm, while the minimum for versicolor is 3.0 cm).

### 6.2 Distributions, normality and outliers (step b)

**Figure.** `eda_distributions.png` has two rows. The top row has a histogram with a smooth density curve (KDE) of each feature, coloured by species. The bottom row has a boxplot of each feature for each species.

**Skewness and kurtosis.** They describe the shape of a distribution. The script uses the pandas versions, which correct for sample size. The basic definitions are:

$$g_1 = \frac{\frac{1}{N}\sum (x_i-\bar{x})^3}{\left(\frac{1}{N}\sum (x_i-\bar{x})^2\right)^{3/2}} \qquad g_2 = \frac{\frac{1}{N}\sum (x_i-\bar{x})^4}{\left(\frac{1}{N}\sum (x_i-\bar{x})^2\right)^{2}} - 3$$

Skewness $g_1$ is 0 for a symmetric distribution, positive when there is a long tail to the right and negative when the tail is to the left. Excess kurtosis $g_2$ is 0 for a normal distribution, positive for heavy tails and negative for light tails.

**Shapiro-Wilk test of normality.** The test is run for every feature within each species, because the species are the natural groups. The statistic is

$$W = \frac{\left(\sum_{i=1}^{n} a_i\,x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i-\bar{x})^2}$$

where $x_{(i)}$ are the ordered values and $a_i$ are constants from the normal distribution. The null hypothesis is that the data are normal. If $p > 0.05$ the script marks `normal_at_5%` as `True`, meaning normality is not rejected. This matters because ANOVA (6.4) assumes roughly normal groups.

**Outliers.** Two rules are applied to each feature over all flowers:

$$\text{IQR rule: } x < Q_1 - 1.5\,\text{IQR} \ \text{ or } \ x > Q_3 + 1.5\,\text{IQR},\ \ \text{IQR} = Q_3 - Q_1 \qquad\qquad \text{z-score rule: } |z| > 3,\ \ z = \frac{x-\bar{x}}{s}$$

The script prints the row numbers of the flowers found. Outliers are only listed and are never removed, because they are genuine flowers.

**Result.**

| Feature | Skewness | Excess kurtosis |
| --- | --- | --- |
| sepal_length | 0.315 | -0.552 |
| sepal_width | 0.334 | 0.291 |
| petal_length | -0.274 | -1.402 |
| petal_width | -0.105 | -1.340 |

All skewness values are small. The petal features have strongly negative kurtosis (about -1.4), because over all flowers they have two peaks: setosa on its own, and versicolor and virginica together.

Shapiro-Wilk does not reject normality for 10 of the 12 feature and species combinations. The two exceptions are petal width for setosa ($W = 0.814$, $p < 0.001$) and petal width for versicolor ($W = 0.948$, $p = 0.027$). Petal length for setosa is just above the limit ($p = 0.055$).

Only sepal width has outliers: rows 15, 32, 33 and 60 by the IQR rule, and row 15 also by the z-score rule. They are kept.

### 6.3 Correlations and pairplot (step c)

**Correlation.** The correlation between two features $x$ and $y$ is measured in two ways.

$$r_{\text{Pearson}} = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2\,\sum (y_i-\bar{y})^2}}$$

Spearman's correlation is the same formula applied to the ranks of the values instead of the values. Both range from $-1$ to $1$: close to $1$ means the two features grow together, close to $-1$ means one grows when the other falls, and close to $0$ means no relationship. Pearson measures straight-line relationships and Spearman measures any steadily rising or falling relationship. `eda_correlation.png` shows both as heatmaps. The script also lists every pair of features with $|r| > 0.8$ (Pearson). Strongly correlated features carry similar information, which is a reason to expect that PCA (6.5) needs fewer than four components.

**Pairplot.** `eda_pairplot.png` is a grid of scatter plots of every pair of features, with the distribution of each feature (a KDE curve) on the diagonal, all coloured by species. In statistical terms it shows the pairwise relationships between the features, how the species form groups in each pair of dimensions, and where the groups overlap. It is the closest picture of the structure the clustering will try to find.

**Result.** Three pairs have $|r| > 0.8$ (Pearson):

| Pair | $r$ |
| --- | --- |
| petal_length and petal_width | 0.96 |
| sepal_length and petal_length | 0.87 |
| sepal_length and petal_width | 0.82 |

Three of the four features are strongly correlated with each other, while sepal width is not part of any strong pair.

### 6.4 Which features separate the species? (step d)

For each feature, the script tests whether the species have different average values. The three species are the groups, so $k = 3$.

**One-way ANOVA.** The variation of a feature is split into a part between the species and a part within the species.

$$SSB = \sum_{g=1}^{k} n_g(\bar{y}_g-\bar{y})^2 \qquad SSW = \sum_{g=1}^{k}\sum_{i=1}^{n_g}(y_{gi}-\bar{y}_g)^2 \qquad SST = SSB + SSW$$

$$F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(N-k)}$$

$\bar{y}_g$ is the mean of species $g$, $\bar{y}$ is the overall mean and $n_g$ is the size of species $g$. A large $F$ and a small $p$-value mean that the species averages differ by much more than the natural variation inside a species.

**Effect size.** A $p$-value says whether a difference exists, but not how large it is, so the script also reports eta squared:

$$\eta^2 = \frac{SSB}{SST}$$

It is the share of the total variation of a feature that is explained by the species, from 0 to 1. The table is sorted by $\eta^2$, so the features at the top are the ones that separate the species best.

**Levene's test.** ANOVA assumes that the species have similar variances. Levene's test checks this, using the absolute deviations $Z_{gi} = |y_{gi} - \tilde{y}_g|$ from each group median $\tilde{y}_g$ (the default in `scipy`):

$$W = \frac{N-k}{k-1}\cdot\frac{\sum_g n_g(\bar{Z}_g-\bar{Z})^2}{\sum_g\sum_i (Z_{gi}-\bar{Z}_g)^2}$$

If $p < 0.05$ the variances are unequal and ANOVA is less reliable for that feature.

**Kruskal-Wallis test.** This is the robust check that does not assume normality or equal variances. All $N$ flowers are ranked together, and the rank sums $R_g$ of the species are compared:

$$H = \frac{12}{N(N+1)}\sum_{g=1}^{k}\frac{R_g^2}{n_g} - 3(N+1)$$

(`scipy` also corrects $H$ for tied values.) If ANOVA and Kruskal-Wallis agree, the conclusion does not depend on the assumptions.

**Tukey HSD.** ANOVA only says that at least one species differs. Tukey's test compares every pair of species (setosa vs versicolor, setosa vs virginica and versicolor vs virginica) and prints a $p$-value for each pair of each feature:

$$q = \frac{|\bar{y}_i-\bar{y}_j|}{\sqrt{\dfrac{MSW}{2}\left(\dfrac{1}{n_i}+\dfrac{1}{n_j}\right)}}$$

The $p$-value comes from the studentized range distribution. This shows which pairs a feature separates. The pair versicolor and virginica is the one to watch, because it is the hardest to separate.

**Result.**

| Feature | $F$ | ANOVA $p$ | $\eta^2$ | Levene $p$ | Kruskal-Wallis $H$ | Kruskal-Wallis $p$ |
| --- | --- | --- | --- | --- | --- | --- |
| petal_length | 1179.0 | $3.1 \times 10^{-91}$ | 0.941 | < 0.001 | 130.4 | $4.8 \times 10^{-29}$ |
| petal_width | 959.3 | $4.4 \times 10^{-85}$ | 0.929 | < 0.001 | 131.1 | $3.4 \times 10^{-29}$ |
| sepal_length | 119.3 | $1.7 \times 10^{-31}$ | 0.619 | 0.002 | 96.9 | $8.9 \times 10^{-22}$ |
| sepal_width | 47.4 | $1.3 \times 10^{-16}$ | 0.392 | 0.525 | 62.5 | $2.7 \times 10^{-14}$ |

All four features differ between the species. The species explain 94% of the variation of petal length and 93% of petal width, but only 39% of sepal width. Levene's test finds unequal variances for three features, so the Kruskal-Wallis test is the more reliable one there, and it agrees with ANOVA for every feature.

Tukey HSD gives $p < 0.001$ for every pair of species for every feature, except versicolor vs virginica for sepal width ($p = 0.009$). So every pair is separated on average, but versicolor and virginica are separated by the smallest margin.

### 6.5 PCA (step e)

PCA rewrites the four features as four new uncorrelated components, ordered by how much of the variation they carry. It is used here only to look at the data in two dimensions. The clustering models use all four scaled features and not the PCA scores.

First the features are standardised, as in 6.6. Then the eigenvalues $\lambda_i$ and eigenvectors $v_i$ of the correlation matrix $C$ are found:

$$C\,v_i = \lambda_i\,v_i \qquad \text{explained variance ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j} \qquad Z = X_s V$$

$X_s$ is the standardised data and $V$ is the matrix of eigenvectors, so the columns of $Z$ are the component scores. The script prints:

* The explained variance ratio of each component and the cumulative sum.
* The loadings, which are the elements of the eigenvectors. Each loading is the weight of a feature in a component, so large values show which features drive PC1 and PC2. The sign of a component is arbitrary, and only the pattern matters.

`eda_pca.png` has the scree plot (bars for the percentage of variance of each component and a line for the cumulative total) and a scatter plot of PC1 against PC2, coloured by species.

**Result.**

| | PC1 | PC2 | PC3 | PC4 |
| --- | --- | --- | --- | --- |
| Explained variance | 72.8% | 23.0% | 3.7% | 0.5% |
| Cumulative | 72.8% | 95.8% | 99.5% | 100% |
| Loading sepal_length | 0.522 | 0.372 | 0.721 | -0.262 |
| Loading sepal_width | -0.263 | 0.926 | -0.242 | 0.124 |
| Loading petal_length | 0.581 | 0.021 | -0.141 | 0.801 |
| Loading petal_width | 0.566 | 0.065 | -0.634 | -0.524 |

The first two components keep 95.8% of the variation, so the 2D plot is a faithful picture of the data. PC1 is driven by petal length, petal width and sepal length (the three correlated features), and PC2 almost only by sepal width. In the scatter plot setosa forms a separate group on the left, while versicolor and virginica form one cloud on the right with a clear overlap.

### 6.6 Scaling decision (step f)

The script prints the range and the variance of each raw feature. The clustering methods are based on distances, for example the Euclidean distance between two flowers $a$ and $b$:

$$d(a,b) = \sqrt{\sum_{j=1}^{4}(a_j-b_j)^2}$$

A feature with a large variance would dominate this sum. So all features are standardised with

$$z = \frac{x-\bar{x}}{\sigma}$$

after which each feature has mean 0 and standard deviation 1 and counts equally in every distance. This is done in the main part of the script, which prints the mean and standard deviation of the scaled data as a check. All later steps use the scaled data.

**Result.**

| Feature | Range (cm) | Variance |
| --- | --- | --- |
| sepal_length | 3.6 | 0.686 |
| sepal_width | 2.4 | 0.188 |
| petal_length | 5.9 | 3.113 |
| petal_width | 2.4 | 0.582 |

The variance of petal length is about 16 times that of sepal width, so without scaling petal length would dominate the distances. After `StandardScaler` the check prints a mean of 0.000 and a standard deviation of 1.000.

### 6.7 EDA summary (step g)

* **Features that separate the species.** Petal length ($\eta^2 = 0.941$) and petal width ($\eta^2 = 0.929$) separate the species best, followed by sepal length (0.619). Sepal width is the weakest (0.392). ANOVA and Kruskal-Wallis agree, so this does not depend on the normality and equal variance assumptions.
* **Correlation.** Petal length, petal width and sepal length are strongly correlated ($r$ from 0.82 to 0.96), so they carry largely the same information. Sepal width adds a different direction.
* **Number of groups.** PCA keeps 95.8% of the variation in two components. The plot shows two clear groups: setosa, and versicolor and virginica together. Inside the second group the two species overlap. So the data suggest two well separated groups, and three groups only with overlap. This is the same tension that appears when choosing $k$ (7.6).
* **Scaling.** The variances of the features range from 0.188 to 3.113, so the data are standardised before clustering.

## 7. Methodology

The analysis compares two families of methods: partitioning methods (k-means and its variants) and hierarchical clustering. Each family is run with three distance metrics: Euclidean, Manhattan and cosine. Every method is tried for $k = 2$ to $10$ clusters. Two internal measures are used, and neither uses the species labels. Each has its own role:

* **The elbow method chooses the number of clusters.** It reads the cost curve of each model on its own, so it can say which $k$ suits a model, but it cannot compare models with each other (see 7.4). Together with what is known about the data, it is the main basis for $k$ (see 7.6).
* **The silhouette score chooses the method and the metric.** With the Euclidean ruler it puts every model on the same scale, so the six finalists can be ranked (see 7.5). It is also reported for every $k$ as a second opinion on the number of clusters. The species labels are used only at the end, to check the result.

### 7.1 Scaling

All methods use the standardised data. Each feature is transformed with

$$z = \frac{x-\mu}{\sigma}$$

(`StandardScaler`, where $\sigma$ is the standard deviation of the feature), so that every feature has mean 0 and standard deviation 1. The reason is that scaling lets all features be equally important in the analysis, so that no feature dominates the distances just because it has larger values (see 6.6). The scaled data is used for fitting every model and for computing every silhouette score.

### 7.2 Partitioning methods (`build_partitioning`, `run_partitioning`, `kmedoids`)

A partitioning method splits the flowers into exactly $k$ groups. It looks for the split with the lowest cost, where the cost measures how far the flowers are from the centre of their group. Three variants are used, one for each distance metric. Standard k-means can only use Euclidean distance, so spherical k-means (cosine) and k-medoids (Manhattan) are added. Each variant uses the right measure for its metric, which makes the comparison between the metrics valid.

| Variant | Distance | How it is done |
| --- | --- | --- |
| `KM-Euclid` | Euclidean | Standard k-means on the scaled data. |
| `KM-Cosine` | Cosine | K-means on the scaled data after each flower is rescaled to length 1 (spherical k-means). |
| `KM-Manhat` | Manhattan | K-medoids on a Manhattan distance matrix, written by hand in the function `kmedoids`. |

**K-means with Euclidean distance.** K-means minimises the within-cluster sum of squares (the inertia):

$$J = \sum_{c=1}^{k}\sum_{x_i \in C_c}\lVert x_i-\mu_c\rVert^2 \qquad \mu_c = \frac{1}{|C_c|}\sum_{x_i\in C_c}x_i$$

It starts from $k$ centres chosen with the k-means++ method, then repeats two steps until nothing changes (at most 300 times): assign each flower to its nearest centre, then move each centre to the mean of its flowers. The result depends on the starting centres, so the whole procedure is run `N_REPEATS` = 10 times from different starts and the run with the lowest $J$ is kept. The seed is `RANDOM_STATE`.

**Spherical k-means with cosine distance.** The cosine distance compares the direction of two flowers and ignores their length:

$$d_{\cos}(a,b) = 1-\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

Each scaled flower is divided by its length (`normalize`), and then k-means is run on these unit-length rows. For unit-length vectors $\lVert a-b\rVert^2 = 2\,d_{\cos}(a,b)$, so minimising the Euclidean cost on them minimises the cosine distance. This is a close approximation of spherical k-means: the centres are not rescaled to length 1 after each step, as the strict algorithm would do. The cost used for the elbow is the inertia on the normalised data.

**K-medoids with Manhattan distance.** The Manhattan distance adds the absolute differences of the features:

$$d_1(a,b) = \sum_{j=1}^{4}|a_j-b_j|$$

K-medoids is like k-means, but the centre of each group (the medoid) must be one of the real flowers, and the cost is the sum of distances to it:

$$J = \sum_{i=1}^{N} d_1\!\left(x_i,\ m_{c(i)}\right)$$

The function `kmedoids` works on a precomputed distance matrix. The first medoid is random and the others are chosen with a probability that grows with the squared distance to the medoids already chosen (the same idea as k-means++). Then it repeats two steps: assign each flower to its nearest medoid, then make the new medoid of each group the member with the smallest total distance to the other members. It stops when the medoids no longer change or after 300 rounds, is restarted `N_REPEATS` times, and keeps the run with the lowest cost. It was written by hand to avoid installing an extra package. It is a simple alternating method and can end in a local optimum, which is why the restarts are used.

**Output.** For every variant and every $k$ from 2 to 10, `run_partitioning` stores the cost, the silhouette with the model's own metric and the silhouette with the Euclidean distance (see 7.4). It also finds the elbow of each cost curve and saves the figure `4A_elbow_silhouette_partitioning.png`.

**Result.** Cost and silhouette for $k = 2$ to $5$ (the full table up to $k = 10$ is in the terminal output):

| Variant | $k$ | Cost | `sil_own` | `sil_euclid` |
| --- | --- | --- | --- | --- |
| `KM-Euclid` | 2 | 223.73 | 0.580 | 0.580 |
| | 3 | 140.97 | 0.459 | 0.459 |
| | 4 | 114.62 | 0.385 | 0.385 |
| | 5 | 91.30 | 0.347 | 0.347 |
| `KM-Cosine` | 2 | 58.52 | 0.734 | 0.570 |
| | 3 | 29.86 | 0.741 | 0.446 |
| | 4 | 23.64 | 0.650 | 0.358 |
| | 5 | 19.85 | 0.579 | 0.298 |
| `KM-Manhat` | 2 | 269.40 | 0.619 | 0.580 |
| | 3 | 207.82 | 0.495 | 0.447 |
| | 4 | 184.38 | 0.437 | 0.389 |
| | 5 | 169.20 | 0.345 | 0.329 |

The Euclidean silhouette is highest at $k = 2$ for all three variants. The own-metric silhouette is highest at $k = 2$ for `KM-Euclid` and `KM-Manhat` and at $k = 3$ for `KM-Cosine`. The elbows are in 7.4.

### 7.3 Hierarchical clustering (`run_hierarchical`)

Hierarchical clustering starts with every flower as its own cluster and repeatedly merges the two closest clusters until one cluster is left. The merges form a tree (a dendrogram). Cutting the tree at a given height gives $k$ clusters, and the script cuts it for every $k$ from 2 to 10.

**Grid of metric and linkage.** The script tries three distance metrics (Euclidean, Manhattan, cosine) with three linkage rules, which define the distance between two clusters $A$ and $B$:

$$\text{average: } D(A,B) = \frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B}d(a,b) \qquad \text{complete: } D(A,B) = \max_{a\in A,\,b\in B}d(a,b)$$

$$\text{Ward: } \Delta(A,B) = \frac{|A||B|}{|A|+|B|}\,\lVert\mu_A-\mu_B\rVert^2$$

Average linkage uses the mean distance between the two clusters, and complete linkage uses the largest distance. Ward merges the pair of clusters that increases the within-cluster sum of squares the least.

**Why Ward is Euclidean-only.** Ward's rule is defined with squared Euclidean distances between cluster means, the same quantity that k-means minimises. It has no meaning for Manhattan or cosine distances, so the script skips Ward for those metrics. The grid therefore has 7 combinations: Euclidean with Ward, average and complete, and Manhattan and cosine with average and complete.

**Cophenetic correlation.** This measures how well the tree preserves the original distances. For every pair of flowers $d_{ij}$ is their original distance and $t_{ij}$ is the height at which the tree first joins them:

$$c = \frac{\sum_{i<j}(d_{ij}-\bar{d})(t_{ij}-\bar{t})}{\sqrt{\sum_{i<j}(d_{ij}-\bar{d})^2\ \sum_{i<j}(t_{ij}-\bar{t})^2}}$$

A value close to 1 means the tree is a faithful summary of the distances.

**Output.** For each combination and each $k$ the script stores the silhouette with the model's own metric and with the Euclidean distance. If a cut returns fewer than $k$ clusters, the real number is recorded in `n_clusters`. The script then picks the best linkage for each metric at $k$ = `K_CHOSEN`, using the Euclidean silhouette. For these three it draws the dendrograms (`4B_dendrograms.png`), which show only the last 30 merges and have a red line at the height that gives `K_CHOSEN` clusters. It also saves the silhouette heatmaps (`4B_hierarchical_silhouette_heatmaps.png`) and the merge-height plot (`4B_merge_heights.png`).

**Result.**

| Combination | Cophenetic correlation | Merge-height jump suggests $k$ |
| --- | --- | --- |
| Cosine-average | 0.898 | 2 |
| Euclidean-average | 0.853 | 2 |
| Manhattan-average | 0.852 | 2 |
| Euclidean-ward | 0.823 | 2 |
| Euclidean-complete | 0.751 | 3 |
| Manhattan-complete | 0.733 | 2 |
| Cosine-complete | 0.721 | 6 |

Average linkage preserves the original distances best for every metric (cophenetic correlation 0.85 to 0.90), and it is also the best linkage at $k = 3$ by the Euclidean silhouette for all three metrics. So the three hierarchical finalists are `HC-Euclidean-average`, `HC-Manhattan-average` and `HC-Cosine-average`. Five of the seven trees have their biggest merge-height jump at $k = 2$.

### 7.4 Internal validation

Internal validation judges a clustering using only the data, without the species labels.

**Silhouette score.** For every flower $i$, let $a(i)$ be its mean distance to the other flowers of its own cluster, and $b(i)$ its mean distance to the flowers of the nearest other cluster:

$$s(i) = \frac{b(i)-a(i)}{\max\lbrace a(i),\ b(i)\rbrace} \qquad S = \frac{1}{N}\sum_{i=1}^{N}s(i)$$

$S$ lies between $-1$ and $1$. A high value means the flowers are close to their own cluster and far from the others. A value near 0 means the clusters overlap, and a negative value means flowers are probably in the wrong cluster. If a cut gives fewer than two clusters, the score is not defined and is stored as empty.

**Own metric and Euclidean ruler.** Every model gets two silhouette scores. The first (`sil_own`) is computed with the model's own distance, so a cosine model is judged with cosine distance. The second (`sil_euclid`) is computed for every model with the Euclidean distance on the scaled data. Scores based on different distances are on different scales, so only the Euclidean ruler can be used to compare models with each other. The own-metric score is reported as extra information. Because the Euclidean ruler is not neutral for models that use another distance, both scores are kept.

**Elbow method.** The cost of a partitioning method always falls when $k$ grows, because more clusters fit the data more closely. The elbow is the value of $k$ after which adding clusters brings little improvement. The costs of the three variants are in different units, so each curve is read on its own. This is why the elbow is used to choose $k$ for each model, but not to compare models: a cost of 140 for k-means and 208 for k-medoids cannot be compared. The script finds the elbow in three ways:

* **Kneedle (`knee_point`).** The values of $k$ and the cost are rescaled to the range 0 to 1, giving $k_n$ and $c_n$. The elbow is the $k$ with the largest value of

  $$(1-k_n) - c_n$$

  which is the point that lies furthest below the straight line joining the first and the last point of the curve.
* **Second difference (`second_diff_k`).** The elbow is the $k$ where the curve bends most sharply, that is, with the largest value of

  $$\Delta^2 c_k = c_{k-1} - 2c_k + c_{k+1}$$
* **Merge-height jump (`merge_gap_k`, `merge_heights_by_k`).** This is the equivalent for hierarchical clustering, which has no cost curve. Let $H(k)$ be the height at which $k$ clusters are merged into $k-1$. A large height means two clusters that are far apart are being forced together. The suggested number of clusters is the $k$ with the biggest jump

  $$H(k) - H(k+1)$$

  because up to $k$ clusters the merges are cheap, and the next merge is expensive. The function `merge_heights_by_k` gives the values of $H(k)$ for the plot.

The two partitioning elbow methods can disagree, so the script reports both.

**Result.** The two elbow methods disagree in this run. Kneedle gives $k = 5$ for `KM-Euclid` and $k = 4$ for `KM-Cosine` and `KM-Manhat`. The second difference gives $k = 3$ for all three. The merge-height jumps of the hierarchical models are in the table of 7.3.

### 7.5 Method decision (`build_candidates`, `decide_method`)

**The six candidates.** `build_candidates` collects six finalists at $k$ = `K_CHOSEN`: the three partitioning variants (`KM-Euclid`, `KM-Cosine`, `KM-Manhat`) and, for each metric, the best linkage from the hierarchical grid (named `HC-<metric>-<linkage>`, for example `HC-Cosine-complete`).

**Decision rule.** The rule is fixed in the code before the results are seen:

1. Rank the six models by the Euclidean-ruler silhouette (`sil_euclid`) and take the highest score.
2. Every model whose score is within `TIE_TOLERANCE` (0.02) of the highest is treated as tied.
3. Among the tied models, the simplest one wins. Each model gets a complexity score, 0 or 2 for the metric (0 for Euclidean, 2 for the others) plus 0 or 1 for the family (0 for partitioning, 1 for hierarchical), and the lowest score wins. If scores are equal, the higher silhouette wins.

The script prints the table of the six models, the models in the tie band and the winner.

**Stability check.** For each partitioning variant the script runs `N_REPEATS` single-start fits at $k$ = `K_CHOSEN` with different seeds and prints the mean, standard deviation, minimum and maximum of the Euclidean silhouette. A small standard deviation means the result does not depend on the random starting point.

**Note.** The rule uses only the silhouette, which does not look at the size of the clusters. Always check the cluster sizes of the winner in the cross-tables of 7.7. The comparison is drawn in `4C_candidate_comparison.png`.

The table of the six finalists and the winner are in 8.1 and 8.3.

**Result.** Stability over 10 single-start runs at $k = 3$ (Euclidean silhouette):

| Variant | Mean | SD | Min | Max |
| --- | --- | --- | --- | --- |
| `KM-Euclid` | 0.461 | 0.007 | 0.455 | 0.479 |
| `KM-Cosine` | 0.446 | 0.001 | 0.446 | 0.448 |
| `KM-Manhat` | 0.450 | 0.012 | 0.439 | 0.479 |

All three variants are stable (standard deviation 0.012 or less).

### 7.6 Choosing k (`choose_k`)

The number of clusters is set by `K_CHOSEN` (3). The function `choose_k` builds an evidence table with one row for each of the six finalists:

* `elbow_k`: the elbow by kneedle for the partitioning variants, and the merge-height jump for the hierarchical models.
* `elbow_k_2nd_diff`: the elbow by the second difference (partitioning variants only).
* `best_k_silhouette_own` and `best_k_silhouette_euclid`: the $k$ with the highest silhouette, with the model's own metric and with the Euclidean ruler.

The table is printed and saved in the sheet `Internal validation` of the Excel file. The script then points to this section, where the decision is written below.

The evidence table is in 8.2.

**Justification of the number of clusters.** The silhouette pointed to $k = 2$, but $k = 3$ was chosen, based on the elbow method and the fact that the dataset has three species. The silhouette and the chosen $k$ therefore do not point to the same value, and this is stated openly.

* **$k = 2$ is the best choice by the silhouette.** The Euclidean silhouette is highest at $k = 2$ for all six models, and the own-metric silhouette for five of the six (see the evidence table). For `KM-Euclid` the silhouette falls from 0.580 at $k = 2$ to 0.459 at $k = 3$. Setosa is clearly separated from the other two species, while versicolor and virginica overlap. The merge-height jump, the elbow equivalent for hierarchical clustering, also points to $k = 2$ for the three hierarchical finalists. If the choice were based only on the silhouette, $k = 2$ would be selected.
* **$k = 3$ was chosen with the elbow method and to match the natural grouping of the data**, which is the three species. The second-difference elbow points to $k = 3$ for all three partitioning variants. Kneedle points higher ($k = 4$ or $5$), so no partitioning elbow supports $k = 2$. In a real situation the best statistical value is not the only thing that matters. A good choice that describes the groups in a more useful way can be preferred, and $k = 3$ also tries to separate versicolor from virginica, which $k = 2$ cannot do.
* **This choice uses prior knowledge.** The number of species (3) is known domain knowledge, and together with the partitioning elbows it is the reason $k = 3$ was preferred over the $k = 2$ favoured by the silhouette. The species of the individual flowers are not used. The consequence is that the results at $k = 3$ must be read with the overlap of versicolor and virginica in mind, which is visible in the accuracy and in the cross-tables of 7.7.

### 7.7 External check (`external_check`)

After the method and $k$ have been chosen, the clusters of each finalist are compared with the real species.

**Cross-table.** For each model the script counts how many flowers of each species fall in each cluster. The numbers of the clusters are arbitrary, so cluster 1 of one model has nothing to do with cluster 1 of another model.

**Matching clusters to species (Hungarian method).** To count correct flowers, each cluster must be matched to one species. The script finds the one-to-one matching $\sigma$ that places the largest number of flowers correctly (`linear_sum_assignment`). It prints the matching it found.

**Matched accuracy and recall.** With $n_{j,\sigma(j)}$ the number of flowers in cluster $j$ that belong to its matched species, and $n_s$ the number of flowers of species $s$:

$$\text{Accuracy} = \frac{\sum_{j} n_{j,\sigma(j)}}{N} \qquad \text{Recall}_s = \frac{n_{\sigma^{-1}(s),\,s}}{n_s}$$

The recall of a species is the share of its flowers that are in the cluster matched to it, and it is 0 if no cluster is matched to it. The matching is one-to-one, so the accuracy is only meaningful when the number of clusters equals the number of species. That is why the check is run at $k = 3$.

**Adjusted Rand Index.** The ARI measures how well the clusters and the species agree, corrected for chance. Its formula is in section 2. It does not need a matching of clusters to species.

The results are printed and saved in the sheet `External check` of the Excel file, together with the cross-tables. The script also prints the accuracy of the internal winner and the highest accuracy of all models. If they differ, it says so, and the winner is not changed.

**Result.**

| Model | Matched accuracy | ARI | Recall setosa | Recall versicolor | Recall virginica |
| --- | --- | --- | --- | --- | --- |
| `HC-Manhattan-average` | **0.887** | **0.718** | 1.00 | 0.98 | 0.68 |
| `KM-Manhat` | 0.880 | 0.702 | 1.00 | 0.90 | 0.74 |
| `KM-Euclid` | 0.833 | 0.620 | 1.00 | 0.78 | 0.72 |
| `KM-Cosine` | 0.820 | 0.604 | 1.00 | 0.62 | 0.84 |
| `HC-Cosine-average` | 0.820 | 0.610 | 0.98 | 0.50 | 0.98 |
| `HC-Euclidean-average` (winner) | 0.687 | 0.562 | 1.00 | 1.00 | 0.06 |

Cross-table of the internal winner, `HC-Euclidean-average`:

| Cluster | setosa | versicolor | virginica | Matched to |
| --- | --- | --- | --- | --- |
| 1 | 50 | 0 | 0 | setosa |
| 2 | 0 | 0 | 3 | virginica |
| 3 | 0 | 50 | 47 | versicolor |

Cross-table of the best external match, `HC-Manhattan-average`:

| Cluster | setosa | versicolor | virginica | Matched to |
| --- | --- | --- | --- | --- |
| 1 | 50 | 0 | 0 | setosa |
| 2 | 0 | 49 | 16 | versicolor |
| 3 | 0 | 1 | 34 | virginica |

### Why the external measures are used only after the choices are made

The species labels are never used to fit a model or to choose the method. The number of species (3) is background knowledge that was one reason for choosing $k = 3$ (see 7.6), but the species of the individual flowers are not used for any choice. There are two reasons for keeping the comparison with the species until the end.

* **A fair test.** If the labels helped to choose the winner, the accuracy would be inflated, because the best model would have been selected to match the labels. As a final check on a choice made without the labels, it is an independent test.
* **The real situation.** In a real clustering problem there are no labels, so the internal measures are all that is available. The external check shows whether choosing with the elbow and the silhouette led to a good result.

If the internal winner is not the model with the best accuracy, this is reported and explained, and the winner is not switched afterwards.

## 8. Results

This section brings together the main results, with the numbers from the run described in section 4.

### 8.1 Candidate comparison at $k = 3$

| Model | Family | Metric | `sil_own` | `sil_euclid` | Complexity | In tie band |
| --- | --- | --- | --- | --- | --- | --- |
| `HC-Euclidean-average` | Hierarchical | Euclidean | 0.480 | 0.480 | 1 | Yes (winner) |
| `KM-Euclid` | Partitioning | Euclidean | 0.459 | 0.459 | 0 | No |
| `HC-Manhattan-average` | Hierarchical | Manhattan | 0.502 | 0.452 | 3 | No |
| `KM-Manhat` | Partitioning | Manhattan | 0.495 | 0.447 | 2 | No |
| `KM-Cosine` | Partitioning | Cosine | 0.741 | 0.446 | 2 | No |
| `HC-Cosine-average` | Hierarchical | Cosine | 0.721 | 0.429 | 3 | No |

Tie band: 0.460 to 0.480 (`TIE_TOLERANCE` = 0.02).

By the Euclidean ruler, `HC-Euclidean-average` scores highest (0.480), and it is the only model in the tie band. `KM-Euclid` misses the band by 0.0005. The Euclidean models score best in both families, which is partly expected, because the ruler itself is Euclidean (see section 9). The differences are small: all six models lie between 0.429 and 0.480, so no method or metric is clearly better than the others. The own-metric scores of the cosine models (0.72 and 0.74) are high, but they use a different scale and cannot be compared with the rest.

### 8.2 Evidence table for $k$

| Model | `elbow_k` | `elbow_k_2nd_diff` | `best_k_silhouette_own` | `best_k_silhouette_euclid` |
| --- | --- | --- | --- | --- |
| `KM-Euclid` | 5 | 3 | 2 | 2 |
| `KM-Cosine` | 4 | 3 | 3 | 2 |
| `KM-Manhat` | 4 | 3 | 2 | 2 |
| `HC-Euclidean-average` | 2 | | 2 | 2 |
| `HC-Manhattan-average` | 2 | | 2 | 2 |
| `HC-Cosine-average` | 2 | | 2 | 2 |

The silhouette points to $k = 2$: it is the best $k$ for all six models with the Euclidean ruler, and for five of six with their own metric. The elbows are mixed. For the partitioning variants the second difference gives $k = 3$ for all three, and kneedle gives 4 or 5, so no partitioning elbow points to $k = 2$. For the hierarchical models the merge-height jump gives $k = 2$. The measures agree that setosa is a clearly separate group, and disagree on whether versicolor and virginica should be split.

### 8.3 Final method and $k$

| | Choice |
| --- | --- |
| Method | `HC-Euclidean-average` (hierarchical clustering, Euclidean distance, average linkage) |
| Number of clusters | $k = 3$ |
| Euclidean silhouette at $k = 3$ | 0.480 |
| Cluster sizes | 50, 97, 3 |

The method was chosen by the decision rule fixed in advance (7.5). `HC-Euclidean-average` has the highest Euclidean silhouette and no other model is within 0.02 of it, so the simplicity rule did not need to be applied. The number of clusters, $k = 3$, was chosen with the elbow method (the second-difference elbow of all three partitioning variants) and the fact that the dataset has three species, over the $k = 2$ favoured by the silhouette (7.6). The weak point of the choice is the cluster sizes: one of the three clusters holds only 3 flowers, so the chosen model is close to the $k = 2$ solution.

### 8.4 Accuracy and ARI of the chosen model

| Model | Matched accuracy | ARI | Recall setosa | Recall versicolor | Recall virginica |
| --- | --- | --- | --- | --- | --- |
| `HC-Euclidean-average` (chosen) | 0.687 | 0.562 | 1.00 | 1.00 | 0.06 |
| `HC-Manhattan-average` (highest accuracy, for reference) | 0.887 | 0.718 | 1.00 | 0.98 | 0.68 |

The chosen model matches the species only partly: accuracy 0.687 and ARI 0.562, the lowest of the six models. It finds setosa perfectly and puts all versicolor in one cluster, but it places 47 of the 50 virginica flowers in the same cluster as versicolor, so the recall of virginica is only 0.06. Its third cluster holds only 3 virginica flowers. `HC-Manhattan-average` matches the species best (accuracy 0.887, ARI 0.718), followed by `KM-Manhat` (0.880, 0.702). Following the rule fixed in advance, the winner is not switched.

This shows the limit of choosing a model with the silhouette alone. The silhouette rewards the clean separation of setosa and does not penalise a very small cluster, so the model with the best silhouette is not the one that best recovers the three species. On data where two of the groups overlap, the silhouette should be read together with the cluster sizes.

## 9. Limitations and notes

* **The silhouette favours $k = 2$ on iris.** Setosa is far from the other two species, while versicolor and virginica overlap. The silhouette therefore scores the split into setosa and the rest highest, and with the Euclidean ruler every model has its best score at $k = 2$. A method chosen by the silhouette at $k = 3$ can still be weak at separating versicolor from virginica, as the chosen model shows (see 8.4).
* **The Euclidean ruler is not neutral for other metrics.** All models are compared with the Euclidean silhouette, so that the scores are on the same scale. But a model that was built with Manhattan or cosine distance is then judged with a distance it did not optimise, which gives an advantage to the Euclidean models. The own-metric scores are reported for this reason, but they cannot be compared across metrics.
* **Matched accuracy is only meaningful at $k = 3$.** The matching of clusters to species is one-to-one. With fewer clusters than species, some species get no cluster and a recall of 0. With more clusters, some clusters are left out of the count. So the external check is run only at $k = 3$, the number of species. The ARI does not have this problem.
* **The hand-written k-medoids is simple, not optimal.** It alternates between assigning flowers and updating medoids, and it can end in a local optimum. Better algorithms exist, such as PAM in the `scikit-learn-extra` package. The simple version was used to avoid an extra package, and the 10 restarts reduce the risk of a poor result.

## 10. Reproducibility

The analysis is made reproducible so that every run gives the same numbers, and the results in this README can be checked by running the script again.

* **Seeds.** `RANDOM_STATE = 42` sets the seed of NumPy, k-means, the hand-written k-medoids and PCA. The stability check uses the seeds 42 to 51 (`RANDOM_STATE + r`). Hierarchical clustering has no random step. With fixed seeds, every run on the same data and package versions gives the same numbers.
* **Data.** The numbers in this README come from the Kaggle copy of the data. The scikit-learn copy, used only if the download fails, may differ in a few values, so the results could differ slightly. The script prints which source it used.
* **Package versions.** The results were produced with:

| Package | Version |
| --- | --- |
| Python | 3.13.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |
| kagglehub | 1.0.3 |

`requirements.txt` pins these versions, so `pip install -r requirements.txt` installs the same environment.

## 11. References

* Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
* Iris flower dataset on Kaggle (`arshid/iris-flower-dataset`): https://www.kaggle.com/datasets/arshid/iris-flower-dataset
* Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
* Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
