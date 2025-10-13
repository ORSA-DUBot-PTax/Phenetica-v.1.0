# **Phenetica v.1.0**
Phenetica is a powerful, cross-platform, standalone desktop application designed for comprehensive morphometric analysis of biological data using binary character traits. Built with modern Python technologies, Phenetica provides taxonomists with an intuitive graphical interface for performing advanced statistical analyses without the need for programming knowledge.

## **Why Phenetica?**
Phenetica bridges the gap between complex statistical methods and practical biological research, making sophisticated morphometric analyses accessible to everyone. The key features include:
1. **Cost-Free & Open Source** — No licensing fees, freely available for academic and research use.
2. **Cross-Platform** — Runs seamlessly on Windows (10, 11) and Linux systems.
3. **User-Friendly** — Modern GUI with no coding required. Cross-Platform — Runs seamlessly on Windows (10, 11) and Linux systems.
4. **Comprehensive** — Multiple analysis methods in one integrated tool.
5. **Publication-Ready** — High-quality visualizations suitable for scientific publications.
6. **Efficient** — Automated workflow from data input to results generation.

## **✨ Features**

📊**Similarity Analysis**
1. Simple Matching Coefficient (SMC) — Quantifies overall similarity between taxa.
2. Jaccard Similarity Index — Measures similarity based on shared presence of characters.
3. Exports similarity matrices in CSV format for further analysis.

🌳**Hierarchical Clustering**
1. UPGMA Dendrogram — Unweighted Pair Group Method with Arithmetic Mean (average linkage).
2. Alternative Clustering Methods — Single linkage, complete linkage, and Ward's method.
3. Publication-quality dendrograms.
4. Automatic label sizing based on dataset complexity.

📈**Ordination Methods**
1. Principal Component Analysis (PCA).
2. Non-metric Multidimensional Scaling (NMDS).
3. UMAP (Uniform Manifold Approximation and Projection).
4. t-SNE (t-Distributed Stochastic Neighbor Embedding).

🔥**Data Visualization**
1. Similarity Heatmap — Color-coded matrix showing relationships between all taxa.
2. Scatter Plots — Labeled data points with distinct colors per analysis.
3. Dendrograms — Multiple clustering visualizations with adjustable aesthetics.

💾**Output Management**
1. All results saved in organized outputs/ folder.
2. CSV files for numerical results (matrices, eigenvalues, loadings, clusters).
3. PNG files for all visualizations.

## Input Structure
Phenetica accepts a CSV file containing morphological or binary trait data. Each row represents a character, and each column represents a taxon.
Example format:
```
Fco	Fsi	Hfo	Hli	Hma	Hpa
1	1	1	1	1	1
0	0	0	0	0	0
1	1	0	0	0	0
```
**Analysis Options**
1. After uploading your CSV, please select one or multiple analyses using the provided checkboxes.
2. Click on the **“Run selected analyses”** button to execute all chosen analyses in a single step.

## License
This project is licensed under the MIT license.

## Developers
Sheikh Sunzid Ahmed & M. Oliur Rahman

Plant Taxonomy and Ethnobotany Laboratory, University of Dhaka

## Contact
If you have any questions, feedback, or issues, please don't hesitate to contact us at-
oliur.bot@du.ac.bd || sunzid79@gmail.com
