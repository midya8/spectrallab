# Method, conventions and limits

## Data and fixed graph family

Two original seeded synthetic fixtures each have 96 rows in two dimensions: 48 samples per arc/ring, with small Gaussian coordinate noise from NumPy's generator seed 20261108. `spectrallab/data.py` is the complete generator. No outside dataset, source image or private data is used. Two generator labels are retained only for post-fit evaluation. There is no train/test split: graph construction and fitting see all coordinates, and no graph is selected by labels. This is a designed illustration, not a blinded benchmark.

All nine rules are fixed: unweighted union-kNN and mutual-kNN at k = 3, 6, 12; full RBF at sigma = 0.08, 0.25, 1.0. Here kNN k counts neighbors, not clusters. There are always two requested clusters. Directed kNN excludes self and sorts squared distances, breaking exact ties by original row index. Union uses OR; mutual uses AND. RBF uses exp(-squared distance / (2 sigma squared)). Every diagonal is zero. Duplicate points can have positive off-diagonal weight. Numerical underflow may delete tiny RBF edges; components use strictly positive computed weights. No display threshold enters fitting.

## Normalized spectral algorithm

For nonnegative symmetric W with zero diagonal, d_i = sum_j W_ij and L = I - D^(-1/2) W D^(-1/2). We reject zero-degree vertices rather than invent an inverse or a singleton convention. Components are found before eigenvalue solving. With more than two components, the zero eigenspace exceeds the two retained dimensions, so the fit is refused. With exactly two nontrivial components, the full zero eigenspace is retained and separates components up to an orthogonal basis change. Its row-normalized image can contain only two distinct locations; overplotting is expected, not missing rows.

NumPy `numpy.linalg.eigh` supplies the symmetric eigensolve using LAPACK. The code computes the full ascending spectrum, retains the first three vectors for witnesses, and clusters the first two, **including the trivial mode**. This is the normalized row-embedding algorithm of Ng, Jordan and Weiss, written with the smallest eigenvalues of I minus normalized affinity instead of the largest eigenvalues of normalized affinity. It is not the nontrivial-only graph drawing algorithm or an unnormalized Laplacian method.

For display, each vector's largest-absolute-value coordinate is made positive. That convention does not make eigenvectors unique. A fully retained repeated eigenspace can rotate; its projector and pairwise row-normalized distances remain unchanged. If the second/third eigenvalue boundary is within 1e-10, we refuse a unique fit claim. The tolerance is numerical, not a statistical significance threshold. Row norms at most 1e-14 are also refused. Eigensystem residuals and orthogonality are numerical checks, not formal certificates.

## K-means and graph cuts

The two-column eigenvector matrix is normalized row by row to unit length. Manual Lloyd k-means uses 12 uniform distinct-row initializations from seed 20261108, stable lower-center argmin ties, nearest-center assignment and cluster means, up to 200 updates. Empty-cluster runs are marked and excluded; exhausted iteration limits raise. Only the least embedding inertia selects a restart, with run index as an exact-inertia tie-breaker. All restart starts, histories, statuses, inertias and labels are retained. Cluster IDs are renamed in first-row occurrence order. IDs have no semantic meaning.

Raw-coordinate k-means uses the same seed, count and algorithm on the original coordinates. This is a separate geometry comparison, not a causal isolation of graph normalization. All fit API functions accept coordinates or weights only, not generator labels. No graph or requested cluster count is chosen using ARI.

For binary partition C, cut = sum_(i in C,j outside C) W_ij counts each crossing edge once. Volume(C) = sum_(i in C) d_i. Normalized cut = cut / volume(C) + cut / volume(complement). Zero-volume cuts are undefined and rejected. A disconnected graph can have cut zero for its component partition; that says nothing about a real class. Comparing normalized-cut magnitudes across different graphs does not compare a common objective.

The spectral relaxation and k-means rounding do not solve discrete normalized cut exactly. `examples/rounding.json` contains an original fixed seven-vertex integer graph, fitted eigenvectors/restarts and a rational exhaustive optimum. Fixing vertex 0 to cluster 0 removes complement duplicates; all 63 nonempty cuts are enumerated. The minimum 3450/4181 is below the fitted 0.860832137733142. Python Fraction and a separate Node BigInt implementation check that finite graph independently.

ARI is the standard adjusted pair agreement: subtract expected same-cluster pair count under fixed margins and divide by the corresponding maximum-minus-expectation. It is invariant to cluster names and can be negative. Generator labels are an audit reference chosen by fixture design, not proof of natural classes.

## Archive and browser

`examples/report.json` binds source coordinates/configurations through SHA256 of canonical JSON and stores all degrees, components, positive-edge counts, full eigenspectra, three eigenvectors, normalized embeddings, all k-means restarts and cut/ARI summaries. The hash called `graph_sha256` hashes the **graph inputs** (points and configuration), not floating affinity bytes; this avoids pretending that libm floating exp bytes are portable. `points_sha256` binds the archived coordinates exactly. Audits also regenerate original coordinates within 2e-14 and require original labels/configurations exactly. Floating witness equations/scores use finite bounded tolerances. The all-version agreement is numerical, not bitwise eigenvector reproducibility.

The Python audit rebuilds all graph weights, checks source/digest/configuration/degree/components, reproduces all spectra and statuses, verifies `L U = U Lambda` and `U' U = I`, checks unit-row normalization, replays every retained restart from archived embedding coordinates, and recomputes cut and ARI. Fresh-fit tests compare selected partitions and subspace projectors, not eigenvector signs. Mutation tests alter provenance, graph settings, degree, eigenvector, labels, metrics, refusal and graph-input hash.

Scalar JavaScript rebuilds W directly from archived points and parameters, independently computes degrees/components/cuts/ARI and every selected row's three eigenpair equations. It does not refit an eigensystem or classify a new point. Browser controls select actual archived fits; color overlays and edge threshold change drawings only. Refused fits clear all embedded points, spectral metrics and cluster membership. A point select and graph click inspect original row identity. JSON export includes the complete selected point neighborhood and all three residual rows, not only the 12 neighbors shown in the table.

## What is not claimed

This is dense O(n squared) graph storage and a dense NumPy eigensolve. No sparse/scalable solver, timing superiority, out-of-sample extension, chosen optimal graph, learned cluster count, ground-truth recovery guarantee, uncertainty calibration, causal explanation or independent predictive validation is claimed. An eigengap measures separation in the chosen matrix, not agreement with an intended class. A dataset with clean curved shapes was deliberately chosen to illustrate both success and failure. More real data would require a separate task and validation design.

## Primary and independent references

- Ng, Jordan and Weiss, *On Spectral Clustering: Analysis and an algorithm* (2001). Original normalized affinity, orthogonal retained vectors, unit-row normalization and clustering algorithm. https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf
- von Luxburg, *A Tutorial on Spectral Clustering*, updated March 2007. Graph construction, normalized Laplacians, connected components/nullspace multiplicity, relaxation and limits. https://pure.mpg.de/rest/items/item_1790853_2/component/file_3009830/content
- NumPy, `numpy.linalg.eigh`. Ascending eigenpairs and LAPACK primitive attribution. https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
- scikit-learn, `spectral_clustering`. Independent library documentation on affinities, initialization and disconnected-graph caveats; no sklearn code/import is used here. Its general connected-graph warning is not taken to mean exactly two nontrivial components cannot be separated by a fully retained zero eigenspace. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html
- Anselin, *An Introduction to Spatial Data Science with GeoDa*, spectral clustering chapter. Independent pedagogical account of algorithm variants. https://lanselin.github.io/introbook_vol2/the-spectral-clustering-algorithm.html
- Spielman, *Graph Clustering: Spectral Methods and Normalized Cuts*, Yale lecture, November 9, 2010. Objective versus heuristic and graph geometry. Its nontrivial-vector drawing convention differs from our retained trivial-mode row embedding. https://www.cs.yale.edu/homes/spielman/462/2010/lect19-10.pdf
- Cross Validated, disconnected-graph discussion (2018). Community evidence that isolated vertices and graph splitting require explicit handling, not authority for our algorithm. https://stats.stackexchange.com/questions/347217/how-to-handle-disconnected-graphs-in-spectral-clustering

Code and original fixtures are MIT licensed. No paper figures, external datasets or third-party implementation code are copied. Attribution is not endorsement.
