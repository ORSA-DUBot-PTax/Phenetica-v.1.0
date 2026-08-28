"""
Phenetica 1.0 - Single-file Streamlit application

This master file combines the Streamlit user interface and the complete
scientific/statistical analysis engine so the application can be developed,
run, and maintained from one Python file.

Run locally:
    streamlit run Phenetica_final.py

Original application authors:
    Sheikh Sunzid Ahmed and M. Oliur Rahman, 2025
    Department of Botany, University of Dhaka
"""

from __future__ import annotations

import base64
import io
import hashlib
import inspect
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]


@dataclass
class PreparedData:
    data: pd.DataFrame
    taxa: list[str]
    characters: list[str]
    orientation_message: str
    delimiter_used: str


@dataclass
class AnalysisResult:
    prepared: PreparedData
    selected: list[str]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    interactive_plots: dict[str, go.Figure] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_plot_style_params(n_taxa: int) -> dict[str, Any]:
    base_fig_width, base_fig_height = 8, 6
    base_font_size = 9
    font_scale = 1.0
    if n_taxa > 50:
        font_scale = 50 / n_taxa
    elif n_taxa < 15:
        font_scale = 1.15

    scatter_font_size = max(5, int(base_font_size * font_scale))
    dendro_height = max(6, n_taxa * 0.15)
    dendro_width = 12
    dendro_leaf_font_size = max(5, int(10 * font_scale * 1.1))
    heatmap_width = max(8, n_taxa * 0.28)
    heatmap_height = max(8, n_taxa * 0.18 + 4)

    return {
        "scatter_fig_size": (base_fig_width, base_fig_height),
        "scatter_font_size": scatter_font_size,
        "dendro_fig_size": (dendro_width, dendro_height),
        "dendro_leaf_font_size": dendro_leaf_font_size,
        "heatmap_fig_size": (heatmap_width, heatmap_height),
    }




def _make_legacy_mds(n_components: int, random_state: int) -> MDS:
    """Create the MDS estimator in a way that matches the original Phenetica call.

    The Tkinter version calls sklearn.manifold.MDS without setting metric=False,
    so its historical behavior is metric MDS over a precomputed dissimilarity
    matrix even though the UI labels the section "NMDS". This helper preserves
    that behavior while supporting both old and new scikit-learn parameter names.
    """
    params = inspect.signature(MDS).parameters
    kwargs: dict[str, Any] = {
        "n_components": n_components,
        "random_state": random_state,
        "n_init": 10,
    }

    if "metric_mds" in params:
        # scikit-learn >= 1.8-style API
        kwargs.update({"metric_mds": True, "metric": "precomputed", "init": "random"})
    else:
        # Older API used `metric` as the metric/non-metric boolean switch.
        kwargs.update({"metric": True, "dissimilarity": "precomputed"})

    return MDS(**kwargs)


def _decode_delimiter_choice(choice: str) -> str:
    normalized = choice.strip().lower()
    if normalized in {"tab", "\\t", "tsv"}:
        return "\t"
    if normalized in {"comma", ",", "csv"}:
        return ","
    return "auto"


def _read_raw_dataframe(file_bytes: bytes, delimiter: str) -> tuple[pd.DataFrame, str]:
    delimiter = _decode_delimiter_choice(delimiter)

    if delimiter == "auto":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t", index_col=0)
            if df.shape[1] <= 1:
                raise ValueError("Tab parsing produced one or fewer data columns")
            return df, "tab"
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=",", index_col=0)
            return df, "comma"

    sep = "\t" if delimiter == "\t" else ","
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, index_col=0)
    return df, "tab" if sep == "\t" else "comma"


def prepare_data(file_bytes: bytes, delimiter: str = "auto") -> PreparedData:
    """Read, orient, sanitize, and validate a Phenetica input matrix."""
    if not file_bytes:
        raise ValueError("The selected file is empty.")

    df, delimiter_used = _read_raw_dataframe(file_bytes, delimiter)
    if df.empty:
        raise ValueError("No usable data were found in the selected file.")

    # Match the original application's behavior.
    df = df.fillna(0)

    index_names = [str(i) for i in df.index]
    col_names = [str(c) for c in df.columns]

    def is_character_label(value: str) -> bool:
        return bool(
            re.match(r"^(Ch|ch|Char|char|Character|character)\b", value)
            or re.match(r"^Ch\d+", value)
        )

    index_has_char_labels = any(is_character_label(name) for name in index_names)
    cols_have_char_labels = any(is_character_label(name) for name in col_names)

    if index_has_char_labels and not cols_have_char_labels:
        data = df.T
        orientation_message = (
            "Characters × Taxa detected from labels; transposed to Taxa × Characters."
        )
    elif cols_have_char_labels and not index_has_char_labels:
        data = df
        orientation_message = "Taxa × Characters detected from labels; used directly."
    else:
        if df.shape[0] < df.shape[1]:
            data = df.T
            orientation_message = (
                "Orientation inferred from shape: fewer rows than columns, so the matrix "
                "was transposed to Taxa × Characters."
            )
        else:
            data = df
            orientation_message = (
                "Orientation inferred from shape: matrix treated directly as Taxa × Characters."
            )

    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    data.index = data.index.map(str)
    data.columns = data.columns.map(str)

    if data.shape[0] < 2:
        raise ValueError("At least two taxa are required for phenetic analyses.")
    if data.shape[1] < 1:
        raise ValueError("At least one character column is required.")

    taxa = list(data.index)
    characters = list(data.columns)

    if len(set(taxa)) != len(taxa):
        raise ValueError("Taxon names must be unique after data orientation is determined.")

    return PreparedData(
        data=data,
        taxa=taxa,
        characters=characters,
        orientation_message=orientation_message,
        delimiter_used=delimiter_used,
    )


def _df_csv_bytes(df: pd.DataFrame, *, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def _figure_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _add_scatter_labels_2d(fig: go.Figure, x: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 10, "line": {"width": 1}},
            hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )


def _add_scatter_labels_3d(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: list[str],
) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 5, "line": {"width": 0.5}},
            hovertemplate=(
                "%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )


def _mime_for_filename(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".zip"):
        return "application/zip"
    return "application/octet-stream"


def build_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, payload in sorted(files.items()):
            zf.writestr(filename, payload)
    return buffer.getvalue()


def save_files_to_directory(files: dict[str, bytes], output_dir: str | os.PathLike[str]) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in files.items():
        file_path = destination / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(payload)
    return destination.resolve()


def run_analysis(
    prepared: PreparedData,
    selected: Iterable[str],
    *,
    cluster_threshold: float = 0.2,
    random_state: int = 42,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    dendrogram_style: str = "Color",
    heatmap_theme: str = "Green",
) -> AnalysisResult:
    """Run the selected Phenetica analyses and return all outputs in memory."""

    selected_list = list(dict.fromkeys(selected))
    valid_keys = {"similarity", "clustering", "pca", "nmds", "heatmap", "umap", "tsne"}
    selected_list = [key for key in selected_list if key in valid_keys]

    result = AnalysisResult(prepared=prepared, selected=selected_list)

    def log(message: str) -> None:
        if message and message[0] in {"═", "─"}:
            line = message
        else:
            line = f"[{timestamp()}] {message}"
        result.logs.append(line)
        if log_callback:
            log_callback(line)

    def warn(message: str) -> None:
        result.warnings.append(message)
        log(f"⚠ {message}")

    def error(message: str) -> None:
        result.errors.append(message)
        log(f"✗ {message}")

    total_steps = max(1, len(selected_list))
    completed = 0

    def progress(label: str) -> None:
        nonlocal completed
        completed += 1
        value = min(1.0, completed / total_steps)
        if progress_callback:
            progress_callback(value, label)

    data = prepared.data
    taxa = prepared.taxa
    characters = prepared.characters
    binary = data.values
    n_taxa = len(taxa)

    params = get_plot_style_params(n_taxa)
    scatter_size = params["scatter_fig_size"]
    scatter_font = params["scatter_font_size"]
    dendro_size = params["dendro_fig_size"]
    dendro_font = params["dendro_leaf_font_size"]
    heatmap_size = params["heatmap_fig_size"]

    log("═" * 80)
    log("Starting Phenetica analyses pipeline...")
    log(f"Input shape: {data.shape[0]} taxa × {data.shape[1]} characters")
    log(prepared.orientation_message)
    log(f"Delimiter used: {prepared.delimiter_used}")
    log("═" * 80)

    # SMC is a dependency for several downstream methods, so calculate it once.
    dist_smc = pdist(binary, metric="hamming")
    sim_smc = 1 - squareform(dist_smc)
    sim_smc_df = pd.DataFrame(sim_smc, index=taxa, columns=taxa)
    dist_matrix_condensed = squareform(1 - sim_smc, checks=False)
    linkage_matrix_upgma = linkage(dist_matrix_condensed, method="average")

    if "similarity" in selected_list:
        log("─" * 80)
        log("Computing Simple Matching Coefficient (SMC) similarity...")
        result.tables["similarity_smc"] = sim_smc_df
        result.files["similarity_matrix_SMC.csv"] = _df_csv_bytes(sim_smc_df)
        log("✓ Prepared similarity_matrix_SMC.csv")

        try:
            dist_jac = pdist(binary, metric="jaccard")
            sim_jac = 1 - squareform(dist_jac)
            sim_jac_df = pd.DataFrame(sim_jac, index=taxa, columns=taxa)
            result.tables["similarity_jaccard"] = sim_jac_df
            result.files["similarity_matrix_Jaccard.csv"] = _df_csv_bytes(sim_jac_df)
            log("✓ Prepared similarity_matrix_Jaccard.csv")
        except Exception as exc:
            error(f"Jaccard similarity failed: {exc}")
        progress("Similarity matrices complete")

    if "clustering" in selected_list:
        log("─" * 80)
        log("Generating hierarchical clustering dendrograms...")
        methods = [
            ("single", "Single Linkage (Nearest Neighbor)"),
            ("complete", "Complete Linkage (Furthest Neighbor)"),
            ("average", "Average Linkage (UPGMA)"),
            ("ward", "Ward Linkage"),
        ]

        for method, method_name in methods:
            try:
                if method == "ward":
                    dist_input = pdist(binary, metric="euclidean")
                    ylabel = "Euclidean Distance"
                    convert_to_similarity = False
                else:
                    dist_input = dist_matrix_condensed
                    ylabel = "SMC Similarity"
                    convert_to_similarity = True

                link_mat = linkage(dist_input, method=method)
                fig, ax = plt.subplots(figsize=dendro_size)
                if dendrogram_style == "Black & White":
                    dendrogram(
                        link_mat,
                        labels=taxa,
                        leaf_rotation=90,
                        leaf_font_size=dendro_font,
                        ax=ax,
                        color_threshold=0,
                        link_color_func=lambda k: "black",
                    )
                    ax.tick_params(colors="black")
                    for spine in ax.spines.values():
                        spine.set_color("black")
                else:
                    dendrogram(
                        link_mat,
                        labels=taxa,
                        leaf_rotation=90,
                        leaf_font_size=dendro_font,
                        ax=ax,
                    )

                if convert_to_similarity:
                    y_ticks = ax.get_yticks()
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(
                        [f"{1-y:.3f}" if 0 <= y <= 1 else f"{1-y:.2f}" for y in y_ticks]
                    )

                ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
                ax.set_xlabel("Taxa", fontsize=11, fontweight="bold")
                ax.set_title(f"{method_name} Dendrogram", fontsize=12, fontweight="bold")
                fig.tight_layout()

                filename = f"dendrogram_{method}.png"
                result.files[filename] = _figure_png_bytes(fig)
                result.metadata[f"linkage_{method}"] = link_mat
                if method == "average":
                    linkage_matrix_upgma = link_mat
                log(f"✓ Prepared {filename}")
            except Exception as exc:
                error(f"{method_name} dendrogram failed: {exc}")

        progress("Hierarchical clustering complete")

    if "pca" in selected_list:
        log("─" * 80)
        log("Performing Principal Component Analysis (PCA)...")
        try:
            pca = PCA()
            pca_result = pca.fit_transform(binary)
            explained_var = pca.explained_variance_ratio_ * 100

            eigen_table = pd.DataFrame(
                {
                    "Component": np.arange(1, len(explained_var) + 1),
                    "Eigenvalue": pca.explained_variance_,
                    "Eigenvalue (%)": np.round(explained_var, 4),
                    "Cumulative (%)": np.round(np.cumsum(explained_var), 4),
                }
            )
            result.tables["pca_eigenvalues"] = eigen_table
            result.files["pca_eigenvalues.csv"] = _df_csv_bytes(eigen_table, index=False)

            loadings = pd.DataFrame(
                pca.components_.T,
                index=characters,
                columns=[f"PC{i + 1}" for i in range(len(explained_var))],
            )
            result.tables["pca_loadings"] = loadings
            result.files["pca_character_loadings.csv"] = _df_csv_bytes(loadings)

            # Static scree plot.
            fig, ax = plt.subplots(figsize=(7, 5))
            components = np.arange(1, len(explained_var) + 1)
            ax.bar(components, explained_var)
            ax.plot(components, explained_var, "o-", color="navy")
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Variance Explained (%)")
            ax.set_title("PCA Scree Plot")
            fig.tight_layout()
            result.files["pca_scree_plot.png"] = _figure_png_bytes(fig)

            # Interactive scree plot.
            pfig = go.Figure()
            pfig.add_bar(x=components, y=explained_var, name="Variance explained")
            pfig.add_scatter(x=components, y=explained_var, mode="lines+markers", name="Trend")
            pfig.update_layout(
                title="PCA Scree Plot",
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained (%)",
            )
            result.interactive_plots["pca_scree"] = pfig

            if pca_result.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=scatter_size)
                ax.scatter(
                    pca_result[:, 0],
                    pca_result[:, 1],
                    s=60,
                    c="#e74c3c",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(pca_result[i, 0], pca_result[i, 1], label, fontsize=scatter_font)
                ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}%)")
                ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}%)")
                ax.set_title("PCA 2D Scatter Plot")
                fig.tight_layout()
                result.files["pca_2d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_2d(pfig, pca_result[:, 0], pca_result[:, 1], taxa)
                pfig.update_layout(
                    title="PCA 2D Scatter Plot",
                    xaxis_title=f"PC1 ({explained_var[0]:.2f}%)",
                    yaxis_title=f"PC2 ({explained_var[1]:.2f}%)",
                )
                result.interactive_plots["pca_2d"] = pfig
            else:
                warn("PCA 2D plot skipped because fewer than two components are available.")

            if pca_result.shape[1] >= 3 and n_taxa >= 3:
                fig = plt.figure(figsize=scatter_size)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    pca_result[:, 0],
                    pca_result[:, 1],
                    pca_result[:, 2],
                    s=60,
                    c="#e74c3c",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(
                        pca_result[i, 0],
                        pca_result[i, 1],
                        pca_result[i, 2],
                        label,
                        fontsize=scatter_font,
                    )
                ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}%)")
                ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}%)")
                ax.set_zlabel(f"PC3 ({explained_var[2]:.2f}%)")
                ax.set_title("PCA 3D Scatter Plot")
                fig.tight_layout()
                result.files["pca_3d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_3d(
                    pfig,
                    pca_result[:, 0],
                    pca_result[:, 1],
                    pca_result[:, 2],
                    taxa,
                )
                pfig.update_layout(
                    title="PCA 3D Scatter Plot",
                    scene={
                        "xaxis_title": f"PC1 ({explained_var[0]:.2f}%)",
                        "yaxis_title": f"PC2 ({explained_var[1]:.2f}%)",
                        "zaxis_title": f"PC3 ({explained_var[2]:.2f}%)",
                    },
                )
                result.interactive_plots["pca_3d"] = pfig
            else:
                warn("PCA 3D plot skipped because at least three usable components are required.")

            log("✓ PCA outputs prepared")
        except Exception as exc:
            error(f"PCA failed: {exc}")
        progress("PCA complete")

    if "nmds" in selected_list:
        log("─" * 80)
        log("Running Non-metric Multidimensional Scaling (NMDS)...")
        try:
            dissim_matrix = 1 - sim_smc
            n_dims_max = min(6, n_taxa)
            stress_vals: list[float] = []

            for dims in range(1, n_dims_max + 1):
                nmds = _make_legacy_mds(dims, random_state)
                nmds.fit(dissim_matrix)
                stress_vals.append(float(nmds.stress_))

            stress_table = pd.DataFrame(
                {"Dimensions": range(1, n_dims_max + 1), "Stress": stress_vals}
            )
            result.tables["nmds_stress"] = stress_table
            result.files["nmds_stress_values.csv"] = _df_csv_bytes(stress_table, index=False)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(range(1, n_dims_max + 1), stress_vals, marker="o")
            ax.set_xlabel("Number of Dimensions")
            ax.set_ylabel("Stress")
            ax.set_title("NMDS Stress Plot")
            fig.tight_layout()
            result.files["nmds_stress_plot.png"] = _figure_png_bytes(fig)

            pfig = go.Figure()
            pfig.add_scatter(
                x=list(range(1, n_dims_max + 1)),
                y=stress_vals,
                mode="lines+markers",
            )
            pfig.update_layout(
                title="NMDS Stress Plot",
                xaxis_title="Number of Dimensions",
                yaxis_title="Stress",
            )
            result.interactive_plots["nmds_stress"] = pfig

            nmds2 = _make_legacy_mds(2, random_state)
            nmds_2d = nmds2.fit_transform(dissim_matrix)
            stress2 = float(nmds2.stress_)

            fig, ax = plt.subplots(figsize=scatter_size)
            ax.scatter(
                nmds_2d[:, 0],
                nmds_2d[:, 1],
                s=60,
                c="#3498db",
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            for i, label in enumerate(taxa):
                ax.text(nmds_2d[i, 0], nmds_2d[i, 1], label, fontsize=scatter_font)
            ax.set_xlabel("NMDS1")
            ax.set_ylabel("NMDS2")
            ax.set_title(f"NMDS 2D Scatter Plot (Stress={stress2:.4f})")
            fig.tight_layout()
            result.files["nmds_2d_plot.png"] = _figure_png_bytes(fig)

            pfig = go.Figure()
            _add_scatter_labels_2d(pfig, nmds_2d[:, 0], nmds_2d[:, 1], taxa)
            pfig.update_layout(
                title=f"NMDS 2D Scatter Plot (Stress={stress2:.4f})",
                xaxis_title="NMDS1",
                yaxis_title="NMDS2",
            )
            result.interactive_plots["nmds_2d"] = pfig
            result.metadata["nmds_stress_2d"] = stress2

            if n_taxa >= 3:
                nmds3 = _make_legacy_mds(3, random_state)
                nmds_3d = nmds3.fit_transform(dissim_matrix)
                stress3 = float(nmds3.stress_)

                fig = plt.figure(figsize=scatter_size)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    nmds_3d[:, 0],
                    nmds_3d[:, 1],
                    nmds_3d[:, 2],
                    s=60,
                    c="#3498db",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(
                        nmds_3d[i, 0],
                        nmds_3d[i, 1],
                        nmds_3d[i, 2],
                        label,
                        fontsize=scatter_font,
                    )
                ax.set_xlabel("NMDS1")
                ax.set_ylabel("NMDS2")
                ax.set_zlabel("NMDS3")
                ax.set_title(f"NMDS 3D Scatter Plot (Stress={stress3:.4f})")
                fig.tight_layout()
                result.files["nmds_3d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_3d(
                    pfig,
                    nmds_3d[:, 0],
                    nmds_3d[:, 1],
                    nmds_3d[:, 2],
                    taxa,
                )
                pfig.update_layout(
                    title=f"NMDS 3D Scatter Plot (Stress={stress3:.4f})",
                    scene={
                        "xaxis_title": "NMDS1",
                        "yaxis_title": "NMDS2",
                        "zaxis_title": "NMDS3",
                    },
                )
                result.interactive_plots["nmds_3d"] = pfig
                result.metadata["nmds_stress_3d"] = stress3
            else:
                warn("NMDS 3D plot skipped because at least three taxa are required.")

            log("✓ NMDS outputs prepared")
        except Exception as exc:
            error(f"NMDS failed: {exc}")
        progress("NMDS complete")

    if "heatmap" in selected_list:
        log("─" * 80)
        log("Creating SMC similarity heatmap...")
        try:
            fig = plt.figure(figsize=heatmap_size)
            annot_bool = scatter_font > 5 and n_taxa <= 30
            annot_kws = {"fontsize": scatter_font * 1.2} if annot_bool else {}
            heatmap_palettes = {
                "Green": "YlGnBu",
                "Blue": "Blues",
                "Purple": "Purples",
                "Red": "Reds",
                "Orange": "Oranges",
                "Viridis": "viridis",
                "Plasma": "plasma",
                "Inferno": "inferno",
                "Magma": "magma",
            }

            selected_cmap = heatmap_palettes.get(heatmap_theme, "YlGnBu")

            sns.heatmap(
                sim_smc_df,
                cmap=selected_cmap,
                annot=annot_bool,
                square=False,
                cbar_kws={"label": "Similarity", "shrink": 0.65},
                annot_kws=annot_kws,
                xticklabels=True,
                yticklabels=True,
                linewidths=0.3,
                linecolor="gray",
            )
            plt.tick_params(axis="x", labelsize=dendro_font)
            plt.tick_params(axis="y", labelsize=dendro_font)
            plt.title("Simple Matching Coefficient Similarity Heatmap")
            plt.tight_layout()
            result.files["similarity_heatmap.png"] = _figure_png_bytes(fig)

            pfig = go.Figure(
                data=go.Heatmap(
                    z=sim_smc_df.values,
                    x=taxa,
                    y=taxa,
                    colorscale="YlGnBu",
                    zmin=0,
                    zmax=1,
                    colorbar={"title": "Similarity"},
                    hovertemplate="Taxon X: %{x}<br>Taxon Y: %{y}<br>Similarity: %{z:.4f}<extra></extra>",
                )
            )
            pfig.update_layout(
                title="SMC Similarity Heatmap",
                xaxis_title="Taxa",
                yaxis_title="Taxa",
            )
            result.interactive_plots["heatmap"] = pfig
            log("✓ Similarity heatmap prepared")
        except Exception as exc:
            error(f"Heatmap generation failed: {exc}")
        progress("Heatmap complete")

    if "umap" in selected_list:
        log("─" * 80)
        log("Running UMAP (2D + 3D)...")
        if not UMAP_AVAILABLE:
            warn("UMAP was requested but the 'umap-learn' package is not installed.")
        elif n_taxa < 3:
            warn("UMAP skipped because at least three taxa are recommended/required for this configuration.")
        else:
            try:
                n_neighbors = max(2, min(5, n_taxa - 1))
                reducer2 = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=random_state,
                    n_components=2,
                )
                umap_2d = reducer2.fit_transform(binary)

                fig, ax = plt.subplots(figsize=scatter_size)
                ax.scatter(
                    umap_2d[:, 0],
                    umap_2d[:, 1],
                    s=60,
                    c="#9b59b6",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(umap_2d[i, 0], umap_2d[i, 1], label, fontsize=scatter_font)
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.set_title("UMAP 2D Scatter Plot")
                fig.tight_layout()
                result.files["umap_2d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_2d(pfig, umap_2d[:, 0], umap_2d[:, 1], taxa)
                pfig.update_layout(
                    title="UMAP 2D Scatter Plot",
                    xaxis_title="UMAP1",
                    yaxis_title="UMAP2",
                )
                result.interactive_plots["umap_2d"] = pfig

                reducer3 = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=random_state,
                    n_components=3,
                )
                umap_3d = reducer3.fit_transform(binary)

                fig = plt.figure(figsize=scatter_size)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    umap_3d[:, 0],
                    umap_3d[:, 1],
                    umap_3d[:, 2],
                    s=60,
                    c="#9b59b6",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(
                        umap_3d[i, 0],
                        umap_3d[i, 1],
                        umap_3d[i, 2],
                        label,
                        fontsize=scatter_font,
                    )
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.set_zlabel("UMAP3")
                ax.set_title("UMAP 3D Scatter Plot")
                fig.tight_layout()
                result.files["umap_3d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_3d(
                    pfig,
                    umap_3d[:, 0],
                    umap_3d[:, 1],
                    umap_3d[:, 2],
                    taxa,
                )
                pfig.update_layout(
                    title="UMAP 3D Scatter Plot",
                    scene={
                        "xaxis_title": "UMAP1",
                        "yaxis_title": "UMAP2",
                        "zaxis_title": "UMAP3",
                    },
                )
                result.interactive_plots["umap_3d"] = pfig
                log("✓ UMAP outputs prepared")
            except Exception as exc:
                error(f"UMAP failed: {exc}")
        progress("UMAP complete")

    if "tsne" in selected_list:
        log("─" * 80)
        log("Running t-SNE (2D + 3D)...")
        try:
            # Scikit-learn requires perplexity < n_samples. Keep the original
            # upper bound of 30 while making very small datasets safe.
            perplexity_val = max(1.0, min(30.0, float(n_taxa - 1)))

            tsne2 = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=perplexity_val,
                init="pca",
                learning_rate="auto",
            )
            tsne_2d = tsne2.fit_transform(binary)

            fig, ax = plt.subplots(figsize=scatter_size)
            ax.scatter(
                tsne_2d[:, 0],
                tsne_2d[:, 1],
                s=60,
                c="#e67e22",
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            for i, label in enumerate(taxa):
                ax.text(tsne_2d[i, 0], tsne_2d[i, 1], label, fontsize=scatter_font)
            ax.set_xlabel("t-SNE1")
            ax.set_ylabel("t-SNE2")
            ax.set_title("t-SNE 2D Scatter Plot")
            fig.tight_layout()
            result.files["tsne_2d_plot.png"] = _figure_png_bytes(fig)

            pfig = go.Figure()
            _add_scatter_labels_2d(pfig, tsne_2d[:, 0], tsne_2d[:, 1], taxa)
            pfig.update_layout(
                title="t-SNE 2D Scatter Plot",
                xaxis_title="t-SNE1",
                yaxis_title="t-SNE2",
            )
            result.interactive_plots["tsne_2d"] = pfig
            result.metadata["tsne_perplexity"] = perplexity_val

            if n_taxa >= 3:
                tsne3 = TSNE(
                    n_components=3,
                    random_state=random_state,
                    perplexity=perplexity_val,
                    init="pca",
                    learning_rate="auto",
                )
                tsne_3d = tsne3.fit_transform(binary)

                fig = plt.figure(figsize=scatter_size)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    tsne_3d[:, 0],
                    tsne_3d[:, 1],
                    tsne_3d[:, 2],
                    s=60,
                    c="#e67e22",
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                for i, label in enumerate(taxa):
                    ax.text(
                        tsne_3d[i, 0],
                        tsne_3d[i, 1],
                        tsne_3d[i, 2],
                        label,
                        fontsize=scatter_font,
                    )
                ax.set_xlabel("t-SNE1")
                ax.set_ylabel("t-SNE2")
                ax.set_zlabel("t-SNE3")
                ax.set_title("t-SNE 3D Scatter Plot")
                fig.tight_layout()
                result.files["tsne_3d_plot.png"] = _figure_png_bytes(fig)

                pfig = go.Figure()
                _add_scatter_labels_3d(
                    pfig,
                    tsne_3d[:, 0],
                    tsne_3d[:, 1],
                    tsne_3d[:, 2],
                    taxa,
                )
                pfig.update_layout(
                    title="t-SNE 3D Scatter Plot",
                    scene={
                        "xaxis_title": "t-SNE1",
                        "yaxis_title": "t-SNE2",
                        "zaxis_title": "t-SNE3",
                    },
                )
                result.interactive_plots["tsne_3d"] = pfig
            else:
                warn("t-SNE 3D plot skipped because at least three taxa are required.")

            log("✓ t-SNE outputs prepared")
        except Exception as exc:
            error(f"t-SNE failed: {exc}")
        progress("t-SNE complete")

    # Preserve the original app's cluster summary behavior, even if the user did
    # not explicitly request the clustering plot section.
    try:
        clusters = fcluster(linkage_matrix_upgma, t=cluster_threshold, criterion="distance")
        cluster_table = pd.DataFrame({"Taxa": taxa, "Cluster": clusters})
        result.tables["cluster_summary"] = cluster_table
        result.files["taxa_cluster_summary.csv"] = _df_csv_bytes(cluster_table, index=False)
        result.metadata["cluster_threshold"] = cluster_threshold
        log("✓ Prepared taxa_cluster_summary.csv")
    except Exception as exc:
        warn(f"Cluster summary could not be produced: {exc}")

    # Save the interpreted matrix too; this is useful for reproducibility and
    # makes orientation/cleaning transparent to the user.
    result.files["interpreted_data_matrix.csv"] = _df_csv_bytes(data)

    result.metadata.update(
        {
            "taxa_count": len(taxa),
            "character_count": len(characters),
            "delimiter_used": prepared.delimiter_used,
            "umap_available": UMAP_AVAILABLE,
            "generated_at": timestamp(),
        }
    )

    log("═" * 80)
    if result.errors:
        log("Pipeline completed with one or more analysis errors. Other successful outputs are available.")
    else:
        log("✓ All selected analyses completed successfully!")
    log("═" * 80)

    # Include a plain-text run log in the ZIP/output bundle.
    result.files["phenetica_run_log.txt"] = ("\n".join(result.logs) + "\n").encode("utf-8")
    return result


__all__ = [
    "AnalysisResult",
    "PreparedData",
    "UMAP_AVAILABLE",
    "build_zip_bytes",
    "get_plot_style_params",
    "prepare_data",
    "run_analysis",
    "save_files_to_directory",
]


# ============================================================================
# STREAMLIT USER INTERFACE
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
EXAMPLE_DATA_PATH = APP_DIR / "Example_data_matrix.csv"
HERO_IMAGE_PATH = APP_DIR / "banner_image.png"  # <--- FIXED: use actual file name
DEFAULT_LOCAL_OUTPUT_DIR = APP_DIR / "outputs"


st.set_page_config(
    page_title="Phenetica 1.0",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_base64_image(image_path: Path) -> str | None:
    """Return a base64 data URI for the image if the file exists."""
    try:
        if image_path.exists():
            img_bytes = image_path.read_bytes()
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
    except Exception:
        pass
    return None


def apply_custom_theme(theme_mode: str) -> None:
    if theme_mode == "Dark":
        app_bg = "#0f172a"
        text = "#e5e7eb"
        card_bg = "#111827"
        border = "#334155"
        muted = "#94a3b8"
        hero_start = "#14532d"
        hero_end = "#166534"
    else:
        app_bg = "#f8fafc"
        text = "#0f172a"
        card_bg = "#ffffff"
        border = "#e2e8f0"
        muted = "#64748b"
        hero_start = "#15803d"
        hero_end = "#22c55e"

    # Prepare hero background image as base64 data URI if available
    hero_bg_image = _get_base64_image(HERO_IMAGE_PATH)
    if hero_bg_image:
        hero_bg_css = f"""
            background:
                linear-gradient(90deg, rgba(5,46,22,0.82), rgba(20,83,45,0.42)),
                url("{hero_bg_image}");
        """
    else:
        # Fallback gradient if image missing
        hero_bg_css = f"""
            background: linear-gradient(90deg, {hero_start}, {hero_end});
        """

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 10% 0%, rgba(34,197,94,0.18), transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(22,101,52,0.12), transparent 35%),
                linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 45%, #f8fafc 100%);
            color: {text};
        }}

        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }}

        .phenetica-hero {{
            padding: 1.45rem 1.8rem;
            min-height: 150px;
            border-radius: 22px;
            {hero_bg_css}
            background-size: cover;
            background-position: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 14px 35px rgba(5,46,22,0.22);
        }}

        .phenetica-hero h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            text-shadow: 0 3px 12px rgba(0,0,0,0.3);
        }}

        .phenetica-hero p {{
            margin-top: 0.35rem;
            font-size: 1rem;
            opacity: 0.95;
        }}

        .phenetica-card {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(34,197,94,0.18);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 25px rgba(5,46,22,0.08);
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, #ffffff, #f0fdf4);
            border-radius: 16px;
            border: 1px solid rgba(22,101,52,0.18);
            padding: 0.85rem;
            box-shadow: 0 6px 18px rgba(5,46,22,0.08);
        }}

        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #052e16 0%, #166534 55%, #15803d 100%);
        }}

        
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }}

div[data-testid="stSidebar"] * {{
            color: white;
        }}

        .stButton > button {{

        .section-title {{
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }}

        .green-title {{
            color: #166534;
        }}

        .blue-title {{
            color: #047857;
        }}

        .orange-title {{
            color: #b45309;
        }}


            border-radius: 14px;
            font-weight: 700;
            border: 1px solid rgba(22,101,52,0.25);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def file_signature(payload: bytes, delimiter: str) -> str:
    return hashlib.sha256(payload + delimiter.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def cached_prepare(payload: bytes, delimiter: str):
    return prepare_data(payload, delimiter)


def get_input_bytes(source_choice: str, uploaded_file) -> tuple[bytes | None, str | None]:
    if source_choice == "Example dataset":
        if not EXAMPLE_DATA_PATH.exists():
            return None, None
        return EXAMPLE_DATA_PATH.read_bytes(), EXAMPLE_DATA_PATH.name

    if uploaded_file is None:
        return None, None
    return uploaded_file.getvalue(), uploaded_file.name


def display_static_image(result: AnalysisResult, filename: str, caption: str | None = None) -> None:
    payload = result.files.get(filename)
    if payload:
        st.image(payload, caption=caption or filename, use_container_width=True)


def display_plot(result: AnalysisResult, key: str) -> None:
    fig = result.interactive_plots.get(key)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_downloads(result: AnalysisResult) -> None:
    zip_payload = build_zip_bytes(result.files)
    st.download_button(
        "⬇️ Download all outputs (.zip)",
        data=zip_payload,
        file_name="Phenetica_outputs.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
        on_click="ignore",
    )

    st.caption(f"{len(result.files)} generated files are included in the ZIP archive.")
    with st.expander("Download individual files", expanded=False):
        for filename in sorted(result.files):
            payload = result.files[filename]
            if filename.lower().endswith(".csv"):
                mime = "text/csv"
            elif filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".txt"):
                mime = "text/plain"
            else:
                mime = "application/octet-stream"

            st.download_button(
                f"Download {filename}",
                data=payload,
                file_name=filename,
                mime=mime,
                key=f"download_{filename}",
                use_container_width=True,
                on_click="ignore",
            )



def build_live_heatmap_figure(result: AnalysisResult, heatmap_theme: str):
    """Create a live Plotly heatmap without rerunning the analysis pipeline."""
    palette_map = {
        "Green": "YlGnBu",
        "Blue": "Blues",
        "Purple": "Purples",
        "Red": "Reds",
        "Orange": "Oranges",
        "Viridis": "Viridis",
        "Plasma": "Plasma",
        "Inferno": "Inferno",
        "Magma": "Magma",
    }

    cmap = palette_map.get(heatmap_theme, "YlGnBu")

    if "similarity_smc" not in result.tables:
        return None

    sim_df = result.tables["similarity_smc"]

    fig = go.Figure(
        data=go.Heatmap(
            z=sim_df.values,
            x=sim_df.columns,
            y=sim_df.index,
            colorscale=cmap,
            zmin=0,
            zmax=1,
            colorbar={"title": "Similarity"},
            hovertemplate="Taxon X: %{x}<br>Taxon Y: %{y}<br>Similarity: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"SMC Similarity Heatmap ({heatmap_theme})",
        xaxis_title="Taxa",
        yaxis_title="Taxa",
    )

    return fig


def render_results(result: AnalysisResult) -> None:
    st.divider()
    st.header("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Taxa", result.metadata.get("taxa_count", len(result.prepared.taxa)))
    m2.metric("Characters", result.metadata.get("character_count", len(result.prepared.characters)))
    m3.metric("Generated files", len(result.files))
    m4.metric("Warnings / errors", len(result.warnings) + len(result.errors))

    tab_names = ["Overview"]
    if "similarity" in result.selected:
        tab_names.append("Similarity")
    if "clustering" in result.selected:
        tab_names.append("Clustering")
    if "pca" in result.selected:
        tab_names.append("PCA")
    if "nmds" in result.selected:
        tab_names.append("NMDS")
    if "heatmap" in result.selected:
        tab_names.append("Heatmap")
    if "umap" in result.selected:
        tab_names.append("UMAP")
    if "tsne" in result.selected:
        tab_names.append("t-SNE")
    tab_names.extend(["Downloads", "Run log"])

    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    with tab_map["Overview"]:
        st.subheader("Interpreted input matrix")
        st.info(result.prepared.orientation_message)
        st.dataframe(result.prepared.data, use_container_width=True)

        if "cluster_summary" in result.tables:
            st.subheader("Taxa cluster summary")
            st.dataframe(result.tables["cluster_summary"], use_container_width=True, hide_index=True)

        if result.warnings:
            st.warning("\n".join(f"• {message}" for message in result.warnings))
        if result.errors:
            st.error("\n".join(f"• {message}" for message in result.errors))

    if "Similarity" in tab_map:
        with tab_map["Similarity"]:
            if "similarity_smc" in result.tables:
                st.subheader("Simple Matching Coefficient (SMC)")
                st.dataframe(result.tables["similarity_smc"], use_container_width=True)
            if "similarity_jaccard" in result.tables:
                st.subheader("Jaccard similarity")
                st.dataframe(result.tables["similarity_jaccard"], use_container_width=True)

    if "Clustering" in tab_map:
        with tab_map["Clustering"]:
            cols = st.columns(2)
            dendrograms = [
                ("dendrogram_single.png", "Single Linkage"),
                ("dendrogram_complete.png", "Complete Linkage"),
                ("dendrogram_average.png", "Average Linkage (UPGMA)"),
                ("dendrogram_ward.png", "Ward Linkage"),
            ]
            for idx, (filename, title) in enumerate(dendrograms):
                with cols[idx % 2]:
                    if filename in result.files:
                        st.markdown(f"**{title}**")
                        display_static_image(result, filename, title)

    if "PCA" in tab_map:
        with tab_map["PCA"]:
            if "pca_eigenvalues" in result.tables:
                st.subheader("Eigenvalues and variance explained")
                st.dataframe(result.tables["pca_eigenvalues"], use_container_width=True, hide_index=True)
            display_plot(result, "pca_scree")
            display_plot(result, "pca_2d")
            display_plot(result, "pca_3d")
            if "pca_loadings" in result.tables:
                with st.expander("Character loadings"):
                    st.dataframe(result.tables["pca_loadings"], use_container_width=True)

    if "NMDS" in tab_map:
        with tab_map["NMDS"]:
            if "nmds_stress" in result.tables:
                st.dataframe(result.tables["nmds_stress"], use_container_width=True, hide_index=True)
            display_plot(result, "nmds_stress")
            display_plot(result, "nmds_2d")
            display_plot(result, "nmds_3d")

    if "Heatmap" in tab_map:
        with tab_map["Heatmap"]:
            live_heatmap = build_live_heatmap_figure(result, heatmap_theme)
            if live_heatmap is not None:
                st.plotly_chart(live_heatmap, use_container_width=True, config={"displaylogo": False})
            with st.expander("High-resolution static PNG"):
                display_static_image(result, "similarity_heatmap.png")

    if "UMAP" in tab_map:
        with tab_map["UMAP"]:
            if not UMAP_AVAILABLE:
                st.warning("UMAP is unavailable in the current environment. Install `umap-learn`.")
            display_plot(result, "umap_2d")
            display_plot(result, "umap_3d")

    if "t-SNE" in tab_map:
        with tab_map["t-SNE"]:
            perplexity = result.metadata.get("tsne_perplexity")
            if perplexity is not None:
                st.caption(f"Perplexity used: {perplexity:g}")
            display_plot(result, "tsne_2d")
            display_plot(result, "tsne_3d")

    with tab_map["Downloads"]:
        render_downloads(result)

    with tab_map["Run log"]:
        st.code("\n".join(result.logs), language="text")


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.title("Phenetica 1.0")
    st.caption("Advanced Morphometric and Phenetic Analyses Suite")

    st.divider()
    dendrogram_style = st.radio(
        "Dendrogram style",
        ["Color", "Black & White"],
        index=0,
        help="Choose between colored or publication-style monochrome dendrograms."
    )

    heatmap_theme = st.selectbox(
        "Similarity Heatmap Color Theme",
        [
            "Green",
            "Blue",
            "Purple",
            "Red",
            "Orange",
            "Viridis",
            "Plasma",
            "Inferno",
            "Magma",
        ],
        index=0,
        help="Choose the color palette for similarity heatmap visualization. The interactive heatmap updates without rerunning analysis."
    )

    st.divider()
    st.markdown("**Analysis Options**")
    analysis_similarity = st.checkbox("📊 Similarity Matrices (SMC + Jaccard)", value=True)
    analysis_clustering = st.checkbox("🌲 Hierarchical Clustering (4 methods)", value=True)
    analysis_pca = st.checkbox("📈 PCA (2D, 3D, eigenvalues, loadings)", value=True)
    analysis_nmds = st.checkbox("🎯 NMDS (2D + 3D)", value=False)
    analysis_heatmap = st.checkbox("🔥 Similarity Heatmap (SMC)", value=False)
    analysis_umap = st.checkbox("🗺️ UMAP (2D + 3D)", value=False)
    analysis_tsne = st.checkbox("🎨 t-SNE (2D + 3D)", value=False)

    st.markdown("**Developed by**")
    st.write("Sheikh Sunzid Ahmed and M. Oliur Rahman")
    st.caption("Department of Botany, University of Dhaka")
    st.caption("© 2025 | All Rights Reserved")

apply_custom_theme("Light")


# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="phenetica-hero">
        <h1>Phenetica 1.0</h1>
        <p>Advanced Morphometrics • Numerical Taxonomy • Phenetic Analysis Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Input section
# ---------------------------
st.markdown('<h2 class="section-title green-title">Select Data</h2>', unsafe_allow_html=True)
input_col, options_col = st.columns([1.35, 1])

with input_col:
    source_choice = st.radio(
        "Data source",
        ["Upload a dataset", "Example dataset"],
        horizontal=True,
    )

    uploaded_file = None
    if source_choice == "Upload a dataset":
        uploaded_file = st.file_uploader(
            "Upload CSV, TSV, or TXT data matrix",
            type=["csv", "tsv", "txt"],
            help="The first column should contain row labels (taxa or characters).",
        )
    else:
        if EXAMPLE_DATA_PATH.exists():
            st.success(f"Using `{EXAMPLE_DATA_PATH.name}` from the app folder.")
            st.download_button(
                "Download example dataset",
                data=EXAMPLE_DATA_PATH.read_bytes(),
                file_name=EXAMPLE_DATA_PATH.name,
                mime="text/csv",
                on_click="ignore",
            )
        else:
            st.error(
                "Example_data_matrix.csv was not found. Put it in the same folder as app.py."
            )

with options_col:
    delimiter_label = st.selectbox(
        "Delimiter",
        ["Auto-detect", "Tab", "Comma"],
        index=0,
        help="Auto-detect first tries tab-separated data, then comma-separated data.",
    )
    delimiter_map = {"Auto-detect": "auto", "Tab": "tab", "Comma": "comma"}
    delimiter = delimiter_map[delimiter_label]

    with st.expander("Advanced settings", expanded=False):
        cluster_threshold = st.number_input(
            "UPGMA cluster distance threshold",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.05,
            format="%.2f",
            help="The original Phenetica default is 0.20.",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            help="Used by NMDS, UMAP, and t-SNE for reproducibility.",
        )
        save_local = st.checkbox(
            "Also save files to ./outputs",
            value=False,
            help=(
                "Useful when running Streamlit locally. On cloud hosting, server-side files "
                "may be temporary, so use the ZIP download for permanent copies."
            ),
        )

input_bytes, input_name = get_input_bytes(source_choice, uploaded_file)
prepared = None

if input_bytes:
    try:
        prepared = cached_prepare(input_bytes, delimiter)
        signature = file_signature(input_bytes, delimiter)

        st.markdown("### Data preview")
        p1, p2, p3 = st.columns(3)
        p1.metric("Taxa", len(prepared.taxa))
        p2.metric("Characters", len(prepared.characters))
        p3.metric("Delimiter", prepared.delimiter_used.title())
        st.info(prepared.orientation_message)
        st.dataframe(prepared.data.head(20), use_container_width=True)
        if len(prepared.data) > 20:
            st.caption(f"Showing the first 20 of {len(prepared.data)} taxa.")
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        prepared = None
        signature = None
else:
    signature = None
    st.info("Select the example dataset or upload your own data matrix to continue.")


# ---------------------------
# Analysis selection
# ---------------------------

selected = []
if analysis_similarity:
    selected.append("similarity")
if analysis_clustering:
    selected.append("clustering")
if analysis_pca:
    selected.append("pca")
if analysis_nmds:
    selected.append("nmds")
if analysis_heatmap:
    selected.append("heatmap")
if analysis_umap:
    selected.append("umap")
if analysis_tsne:
    selected.append("tsne")


# ---------------------------
# Run pipeline
# ---------------------------
st.markdown('<h2 class="section-title orange-title">Run Analysis</h2>', unsafe_allow_html=True)
run_disabled = prepared is None or len(selected) == 0
run_button = st.button(
    "▶ Run Selected Analyses",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
)

if not selected:
    st.warning("Select at least one analysis.")

if run_button and prepared is not None:
    live_logs: list[str] = []
    progress_bar = st.progress(0.0, text="Starting analysis...")

    with st.status("Running Phenetica analyses...", expanded=True) as status:
        log_placeholder = st.empty()

        def log_callback(line: str) -> None:
            live_logs.append(line)
            log_placeholder.code("\n".join(live_logs[-20:]), language="text")

        def progress_callback(value: float, label: str) -> None:
            progress_bar.progress(value, text=label)
            status.update(label=label, state="running")

        try:
            result = run_analysis(
                prepared,
                selected,
                cluster_threshold=float(cluster_threshold),
                random_state=int(random_state),
                log_callback=log_callback,
                progress_callback=progress_callback,
                dendrogram_style=dendrogram_style,
                heatmap_theme=heatmap_theme,
            )

            local_output_path = None
            if save_local:
                run_folder = DEFAULT_LOCAL_OUTPUT_DIR / (
                    datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + (signature or "run")[:8]
                )
                local_output_path = save_files_to_directory(result.files, run_folder)
                result.metadata["local_output_path"] = str(local_output_path)
                live_logs.append(f"Saved server-side outputs to: {local_output_path}")
                log_placeholder.code("\n".join(live_logs[-20:]), language="text")

            result.metadata["input_name"] = input_name
            result.metadata["input_signature"] = signature
            st.session_state["analysis_result"] = result

            progress_bar.progress(1.0, text="Analysis complete")
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Pipeline error: {exc}")


# ---------------------------
# Persist/display last result
# ---------------------------
result = st.session_state.get("analysis_result")
if isinstance(result, AnalysisResult):
    current_signature = signature
    last_signature = result.metadata.get("input_signature")
    if current_signature and last_signature and current_signature != last_signature:
        st.info("The dataset or delimiter has changed. The results below are from the previous run.")

    local_path = result.metadata.get("local_output_path")
    if local_path:
        st.success(f"A local/server-side copy was saved to: `{local_path}`")

    render_results(result)