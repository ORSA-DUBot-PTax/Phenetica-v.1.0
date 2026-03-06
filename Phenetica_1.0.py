"""
Phenetica GUI (CustomTkinter Modern Version)
Author: Sheikh Sunzid Ahmed and M. Oliur Rahman, 2025
Enhanced UI: Modern aesthetics using customtkinter
"""

import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Import customtkinter for modern widgets
import customtkinter as ctk

# Set appearance and theme
ctk.set_appearance_mode("light")  # "light", "dark", or "system"
ctk.set_default_color_theme("green")  # themes: "blue", "green", "dark-blue"

# Check for UMAP availability
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ---------------------------
# Helper functions (unchanged)
# ---------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_plot_style_params(n_taxa):
    base_fig_width, base_fig_height = 8, 6
    base_font_size = 9
    font_scale = 1
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
        'scatter_fig_size': (base_fig_width, base_fig_height),
        'scatter_font_size': scatter_font_size,
        'dendro_fig_size': (dendro_width, dendro_height),
        'dendro_leaf_font_size': dendro_leaf_font_size,
        'heatmap_fig_size': (heatmap_width, heatmap_height)
    }

def show_splash_screen(root, duration=3500):
    """Modern splash screen using customtkinter"""
    splash = ctk.CTkToplevel(root)
    splash.title("")
    splash.overrideredirect(True)
    splash_width = 540
    splash_height = 300
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
    splash.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
    splash.configure(fg_color="#2b2b2b")  # Dark background for contrast
    splash.attributes('-topmost', True)

    # Main container with rounded corners
    main_frame = ctk.CTkFrame(splash, fg_color="#2b2b2b", corner_radius=20)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Logo / icon
    logo_label = ctk.CTkLabel(main_frame, text="🌿", font=("Segoe UI", 64), text_color="#2ecc71")
    logo_label.pack(pady=(40, 5))

    # Title
    title_label = ctk.CTkLabel(main_frame, text="Phenetica 1.0",
                               font=("Segoe UI", 32, "bold"), text_color="white")
    title_label.pack(pady=(0, 5))

    # Subtitle
    subtitle_label = ctk.CTkLabel(main_frame, text="Advanced Morphometric and Phenetic Analyses Suite",
                                   font=("Segoe UI", 14), text_color="#b0b0b0")
    subtitle_label.pack(pady=(0, 15))

    # Developers
    dev_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    dev_frame.pack(pady=(5, 0))

    dev_label = ctk.CTkLabel(dev_frame,
                              text="Sheikh Sunzid Ahmed & M. Oliur Rahman",
                              font=("Segoe UI", 12, "bold"), text_color="#e0e0e0")
    dev_label.pack()

    dept_label = ctk.CTkLabel(dev_frame,
                               text="Department of Botany, University of Dhaka",
                               font=("Segoe UI", 10), text_color="#a0a0a0")
    dept_label.pack(pady=(2, 0))

    # Loading indicator
    loading_label = ctk.CTkLabel(main_frame, text="Loading", font=("Segoe UI", 10), text_color="#3498db")
    loading_label.pack(pady=(25, 0))

    # Animated dots
    dots = [".", "..", "...", ""]
    dot_index = [0]

    def animate_dots():
        if splash.winfo_exists():
            loading_label.configure(text=f"Loading{dots[dot_index[0]]}")
            dot_index[0] = (dot_index[0] + 1) % len(dots)
            splash.after(400, animate_dots)

    animate_dots()
    splash.lift()
    splash.focus_force()

    def close_splash():
        if splash.winfo_exists():
            splash.destroy()

    splash.after(duration, close_splash)
    return splash


class PheneticaApp:
    def __init__(self):
        self.root = ctk.CTk()  # Main window
        self.root.withdraw()   # Hide until splash closes

        # Show splash screen
        splash = show_splash_screen(self.root, duration=3500)

        # Setup main window after splash
        self.root.after(3600, self.setup_main_window)

    def setup_main_window(self):
        """Build the main application window with customtkinter widgets"""
        root = self.root
        root.title("Phenetica 1.0 — Advanced Morphometric and Phenetic Analyses Suite")
        root.geometry("1000x850")
        root.minsize(900, 750)

        # Configure grid weights for responsive layout
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)  # Log area expands

        # ----- Header -----
        header_frame = ctk.CTkFrame(root, fg_color="#2ecc71", corner_radius=0, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_propagate(False)

        # Logo icon
        icon_label = ctk.CTkLabel(header_frame, text="🌿", font=("Segoe UI", 40), text_color="white")
        icon_label.grid(row=0, column=0, padx=(20, 10), pady=10)

        # Title
        title_label = ctk.CTkLabel(header_frame, text="Phenetica 1.0",
                                    font=("Segoe UI", 22, "bold"), text_color="white")
        title_label.grid(row=0, column=1, sticky="w", pady=10)

        # ----- Main Content Frame -----
        content_frame = ctk.CTkFrame(root, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.grid_columnconfigure(0, weight=1)

        # ----- Input File Section -----
        input_frame = ctk.CTkFrame(content_frame, corner_radius=15)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        input_frame.grid_columnconfigure(1, weight=1)

        # File selection
        file_label = ctk.CTkLabel(input_frame, text="Data Matrix File:", font=("Segoe UI", 12, "bold"))
        file_label.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), sticky="w")

        self.file_entry = ctk.CTkEntry(input_frame, placeholder_text="Select input file...", width=400)
        self.file_entry.grid(row=0, column=1, padx=(0, 10), pady=(20, 10), sticky="ew")

        browse_btn = ctk.CTkButton(input_frame, text="Browse", command=self.browse_file,
                                    width=100, fg_color="#3498db", hover_color="#2980b9")
        browse_btn.grid(row=0, column=2, padx=(0, 20), pady=(20, 10))

        # Delimiter selection
        delim_label = ctk.CTkLabel(input_frame, text="Delimiter:", font=("Segoe UI", 12, "bold"))
        delim_label.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="w")

        self.delim_var = tk.StringVar(value="auto")
        delim_combo = ctk.CTkComboBox(input_frame, variable=self.delim_var,
                                       values=["auto", "\\t", ","],
                                       state="readonly", width=150)
        delim_combo.grid(row=1, column=1, padx=(0, 10), pady=(0, 20), sticky="w")

        delim_hint = ctk.CTkLabel(input_frame, text="(auto-detect, tab, or comma)",
                                   font=("Segoe UI", 10), text_color="gray")
        delim_hint.grid(row=1, column=2, padx=(0, 20), pady=(0, 20), sticky="w")

        # ----- Analyses Selection Section -----
        analysis_frame = ctk.CTkFrame(content_frame, corner_radius=15)
        analysis_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_columnconfigure(1, weight=1)

        self.chk_vars = {}
        options = [
            ("📊 Similarity Matrices (SMC + Jaccard)", "similarity", True),
            ("🌲 Hierarchical Clustering (4 methods)", "clustering", True),
            ("📈 PCA (2D, 3D, eigenvalues, loadings)", "pca", True),
            ("🎯 NMDS (2D + 3D)", "nmds", False),
            ("🔥 Similarity Heatmap (SMC)", "heatmap", False),
            ("🗺️ UMAP (2D + 3D) — optional", "umap", False),
            ("🎨 t-SNE (2D + 3D) — optional", "tsne", False)
        ]

        # Place checkboxes in two columns
        for i, (label, key, default) in enumerate(options):
            var = tk.BooleanVar(value=default)
            self.chk_vars[key] = var
            chk = ctk.CTkCheckBox(analysis_frame, text=label, variable=var,
                                   font=("Segoe UI", 11))
            chk.grid(row=i//2, column=i%2, padx=20, pady=8, sticky="w")

        # ----- Control Buttons & Progress -----
        control_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        control_frame.grid_columnconfigure(1, weight=1)

        self.run_btn = ctk.CTkButton(control_frame, text="▶ Run Selected Analyses",
                                      command=self.run_clicked,
                                      fg_color="#27ae60", hover_color="#2ecc71",
                                      font=("Segoe UI", 14, "bold"), height=40,
                                      corner_radius=10)
        self.run_btn.grid(row=0, column=0, padx=(0, 20), sticky="w")

        self.progress = ctk.CTkProgressBar(control_frame, width=400, height=15,
                                           corner_radius=7, border_width=0,
                                           progress_color="#27ae60")
        self.progress.grid(row=0, column=1, padx=(0, 20), sticky="ew")
        self.progress.set(0)  # initial value

        # ----- Log Area -----
        log_frame = ctk.CTkFrame(content_frame, corner_radius=15)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = ctk.CTkTextbox(log_frame, font=("Consolas", 10),
                                       fg_color="#2b2b2b", text_color="#f0f0f0",
                                       corner_radius=10, border_width=1,
                                       border_color="#444")
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # ----- Footer -----
        footer_frame = ctk.CTkFrame(root, fg_color="#34495e", corner_radius=0, height=80)
        footer_frame.grid(row=3, column=0, sticky="ew")
        footer_frame.grid_columnconfigure(0, weight=1)
        footer_frame.grid_propagate(False)

        # Top thin border
        top_border = ctk.CTkFrame(footer_frame, fg_color="#3498db", height=3, corner_radius=0)
        top_border.pack(fill="x")

        # Footer content
        footer_content = ctk.CTkFrame(footer_frame, fg_color="transparent")
        footer_content.pack(expand=True, pady=8)

        developed_label = ctk.CTkLabel(footer_content,
                                        text="Developed by",
                                        font=("Times New Roman", 9),
                                        text_color="#95a5a6")
        developed_label.pack()

        authors_label = ctk.CTkLabel(footer_content,
                                      text="Sheikh Sunzid Ahmed and M. Oliur Rahman",
                                      font=("Times New Roman", 11, "bold"),
                                      text_color="#ecf0f1")
        authors_label.pack(pady=(2, 4))

        affiliation_label = ctk.CTkLabel(footer_content,
                                          text="Department of Botany, University of Dhaka",
                                          font=("Times New Roman", 10),
                                          text_color="#bdc3c7")
        affiliation_label.pack()

        copyright_label = ctk.CTkLabel(footer_content,
                                        text="© 2025 | All Rights Reserved",
                                        font=("Times New Roman", 9),
                                        text_color="#7f8c8d")
        copyright_label.pack(pady=(3, 0))

        # Initialize variables
        self.input_path = None
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")

        # Show main window
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        # Initial log message
        self.log("═" * 80)
        self.log("Welcome to Phenetica 1.0")
        self.log("Advanced Morphometric and Phenetic Analyses Suite")
        self.log("═" * 80)
        self.log("Ready to perform morphometric analyses.")
        self.log("Please select your input data file to begin.")
        self.log("")

    def log(self, message):
        """Append message to log textbox with timestamp"""
        t = timestamp()
        if message.startswith("═") or message.startswith("Advanced"):
            full = f"{message}\n"
        else:
            full = f"[{t}] {message}\n"
        self.log_text.insert("end", full)
        self.log_text.see("end")
        # Also print to console for debugging
        print(full, end='')

    def browse_file(self):
        """Open file dialog and set entry"""
        path = filedialog.askopenfilename(
            title="Select Data Matrix File",
            filetypes=[("CSV/TSV files", "*.csv *.tsv *.txt"), ("All files", "*.*")]
        )
        if path:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, path)
            self.input_path = path
            self.log(f"File selected: {os.path.basename(path)}")

    def run_clicked(self):
        """Start analysis in a separate thread"""
        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Input Required",
                                 "Please select a valid input file first.",
                                 parent=self.root)
            return

        os.makedirs(self.outputs_dir, exist_ok=True)
        self.run_btn.configure(state="disabled", fg_color="#95a5a6")
        self.progress.set(0)
        self.log("═" * 80)
        thread = threading.Thread(target=self.execute_pipeline, args=(path,))
        thread.start()

    def execute_pipeline(self, path):
        """Core analysis pipeline (unchanged logic)"""
        try:
            self.log("Starting analyses pipeline...")
            delim_choice = self.delim_var.get()
            if delim_choice == "auto":
                try:
                    df = pd.read_csv(path, sep='\t', index_col=0)
                    if df.shape[1] <= 1:
                        raise Exception("Tab read gave 1 column")
                except Exception:
                    df = pd.read_csv(path, sep=',', index_col=0)
            else:
                sep = '\t' if delim_choice == "\\t" else ','
                df = pd.read_csv(path, sep=sep, index_col=0)

            df = df.fillna(0)

            # Robust data orientation detection
            index_names = [str(i) for i in df.index]
            col_names = [str(c) for c in df.columns]

            def looks_like_chars(names):
                count = sum(1 for n in names if re.match(r'^(Ch|ch|Char|char|Character|character)\b|\bCh\d+', n))
                return count >= max(1, int(len(names) * 0.3))

            index_has_char_labels = any(re.match(r'^(Ch|ch|Char|char|Character|character)\b', n) or re.match(r'^Ch\d+', n) for n in index_names)
            cols_have_char_labels = any(re.match(r'^(Ch|ch|Char|char|Character|character)\b', n) or re.match(r'^Ch\d+', n) for n in col_names)

            if index_has_char_labels and not cols_have_char_labels:
                data = df.T
                taxa = list(data.index)
                characters = list(data.columns)
                self.log("✓ Data interpreted by label-detection: Characters × Taxa (transposed to Taxa × Characters)")
            elif cols_have_char_labels and not index_has_char_labels:
                data = df
                taxa = list(data.index)
                characters = list(data.columns)
                self.log("✓ Data interpreted by label-detection: Taxa × Characters (direct use)")
            else:
                if df.shape[0] < df.shape[1]:
                    data = df.T
                    taxa = list(data.index)
                    characters = list(data.columns)
                    self.log("✓ Data interpreted by shape-fallback: fewer rows than columns -> transposed to Taxa × Characters")
                else:
                    data = df
                    taxa = list(data.index)
                    characters = list(data.columns)
                    self.log("✓ Data interpreted by shape-fallback: Taxa × Characters (direct use)")

            data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
            self.log(f"✓ Input read: {os.path.basename(path)}")
            self.log(f"  Final data shape: {data.shape} (taxa × characters)")
            self.log(f"  Taxa count: {len(taxa)}")
            self.log(f"  Character count: {len(characters)}")
            self.log("")

            plot_params = get_plot_style_params(len(taxa))
            scatter_size = plot_params['scatter_fig_size']
            scatter_font = plot_params['scatter_font_size']
            dendro_size = plot_params['dendro_fig_size']
            dendro_font = plot_params['dendro_leaf_font_size']
            heatmap_size = plot_params['heatmap_fig_size']

            binary = data.values

            steps = []
            if self.chk_vars['similarity'].get(): steps.append('similarity')
            if self.chk_vars['clustering'].get(): steps.append('clustering')
            if self.chk_vars['pca'].get(): steps.append('pca')
            if self.chk_vars['nmds'].get(): steps.append('nmds')
            if self.chk_vars['heatmap'].get(): steps.append('heatmap')
            if self.chk_vars['umap'].get() and UMAP_AVAILABLE: steps.append('umap')
            elif self.chk_vars['umap'].get() and not UMAP_AVAILABLE:
                self.log("⚠ UMAP requested but 'umap' package not available. Skipping UMAP.")
            if self.chk_vars['tsne'].get(): steps.append('tsne')

            total = max(1, len(steps))
            completed = 0

            # 1) Similarity matrices
            if 'similarity' in steps:
                self.log("─" * 80)
                self.log("Computing Simple Matching Coefficient (SMC) similarity...")
                dist_smc = pdist(binary, metric='hamming')
                sim_smc = 1 - squareform(dist_smc)
                sim_smc_df = pd.DataFrame(sim_smc, index=taxa, columns=taxa)
                out_smc = os.path.join(self.outputs_dir, "similarity_matrix_SMC.csv")
                sim_smc_df.to_csv(out_smc)
                self.log(f"✓ Saved SMC similarity matrix → {os.path.basename(out_smc)}")

                self.log("Computing Jaccard similarity...")
                dist_jac = pdist(binary, metric='jaccard')
                sim_jac = 1 - squareform(dist_jac)
                sim_jac_df = pd.DataFrame(sim_jac, index=taxa, columns=taxa)
                out_jac = os.path.join(self.outputs_dir, "similarity_matrix_Jaccard.csv")
                sim_jac_df.to_csv(out_jac)
                self.log(f"✓ Saved Jaccard similarity matrix → {os.path.basename(out_jac)}")
            else:
                # Compute SMC anyway for other analyses
                dist_smc = pdist(binary, metric='hamming')
                sim_smc = 1 - squareform(dist_smc)
                sim_smc_df = pd.DataFrame(sim_smc, index=taxa, columns=taxa)

            completed += 1
            self.progress.set(completed / total)

            # 2) Hierarchical Clustering Dendrograms
            if 'clustering' in steps:
                self.log("─" * 80)
                self.log("Generating hierarchical clustering dendrograms based on SMC similarity...")
                dist_matrix_condensed = squareform(1 - sim_smc)
                methods = [
                    ('single', 'Single Linkage (Nearest Neighbor)'),
                    ('complete', 'Complete Linkage (Furthest Neighbor)'),
                    ('average', 'Average Linkage (UPGMA)'),
                    ('ward', 'Ward Linkage')
                ]
                for method, method_name in methods:
                    try:
                        if method == 'ward':
                            dist_input = pdist(binary, metric='euclidean')
                            ylabel = 'Euclidean Distance'
                            convert_to_similarity = False
                        else:
                            dist_input = dist_matrix_condensed
                            ylabel = 'SMC Similarity'
                            convert_to_similarity = True

                        link_mat = linkage(dist_input, method=method)
                        fig, ax = plt.subplots(figsize=dendro_size)
                        dendrogram(link_mat, labels=taxa, leaf_rotation=90,
                                   leaf_font_size=dendro_font, ax=ax)

                        if convert_to_similarity:
                            y_ticks = ax.get_yticks()
                            ax.set_yticklabels([f'{1-y:.3f}' if 0 <= y <= 1 else f'{1-y:.2f}'
                                                for y in y_ticks])

                        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
                        ax.set_xlabel('Taxa', fontsize=11, fontweight='bold')
                        ax.set_title(f"{method_name} Dendrogram", fontsize=12, fontweight='bold')
                        plt.tight_layout()
                        outname = os.path.join(self.outputs_dir, f"dendrogram_{method}.png")
                        fig.savefig(outname, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved {method_name} dendrogram → {os.path.basename(outname)}")
                        if method == 'average':
                            linkage_matrix_upgma = link_mat
                    except Exception as e:
                        self.log(f"✗ Failed {method_name} dendrogram: {e}")

                completed += 1
                self.progress.set(completed / total)
            else:
                dist_matrix_condensed = squareform(1 - sim_smc)
                linkage_matrix_upgma = linkage(dist_matrix_condensed, method='average')

            # 3) PCA
            if 'pca' in steps:
                self.log("─" * 80)
                self.log("Performing Principal Component Analysis (PCA)...")
                pca = PCA()
                pca_result = pca.fit_transform(binary)

                explained_var = pca.explained_variance_ratio_ * 100
                eigen_table = pd.DataFrame({
                    "Component": np.arange(1, len(explained_var)+1),
                    "Eigenvalue": pca.explained_variance_,
                    "Eigenvalue (%)": np.round(explained_var, 4),
                    "Cumulative (%)": np.round(np.cumsum(explained_var), 4)
                })
                out_eig = os.path.join(self.outputs_dir, "pca_eigenvalues.csv")
                eigen_table.to_csv(out_eig, index=False)
                self.log(f"✓ Saved PCA eigenvalues → {os.path.basename(out_eig)}")

                # Scree plot
                fig = plt.figure(figsize=(7,5))
                plt.bar(np.arange(1, len(explained_var)+1), explained_var)
                plt.plot(np.arange(1, len(explained_var)+1), explained_var, 'o-', color='navy')
                plt.xlabel("Principal Component")
                plt.ylabel("Variance Explained (%)")
                plt.title("PCA Scree Plot")
                plt.tight_layout()
                out_scree = os.path.join(self.outputs_dir, "pca_scree_plot.png")
                fig.savefig(out_scree, dpi=300)
                plt.close(fig)
                self.log(f"✓ Saved Scree plot → {os.path.basename(out_scree)}")

                # PCA 2D
                fig = plt.figure(figsize=scatter_size)
                plt.scatter(pca_result[:,0], pca_result[:,1], s=60, c='#e74c3c',
                            alpha=0.7, edgecolors='black', linewidth=0.5)
                for i, txt in enumerate(taxa):
                    plt.text(pca_result[i,0], pca_result[i,1], txt, fontsize=scatter_font)
                plt.xlabel(f"PC1 ({explained_var[0]:.2f}%)")
                plt.ylabel(f"PC2 ({explained_var[1]:.2f}%)")
                plt.title("PCA 2D Scatter Plot")
                plt.tight_layout()
                out_pca2 = os.path.join(self.outputs_dir, "pca_2d_plot.png")
                plt.savefig(out_pca2, dpi=300)
                plt.close(fig)
                self.log(f"✓ Saved PCA 2D plot → {os.path.basename(out_pca2)}")

                # PCA 3D
                if len(taxa) >= 3 and pca_result.shape[1] >= 3:
                    try:
                        fig = plt.figure(figsize=scatter_size)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2],
                                   s=60, c='#e74c3c', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            ax.text(pca_result[i,0], pca_result[i,1], pca_result[i,2],
                                    txt, fontsize=scatter_font)
                        ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}%)")
                        ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}%)")
                        ax.set_zlabel(f"PC3 ({explained_var[2]:.2f}%)")
                        plt.title("PCA 3D Scatter Plot")
                        plt.tight_layout()
                        out_pca3 = os.path.join(self.outputs_dir, "pca_3d_plot.png")
                        fig.savefig(out_pca3, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved PCA 3D plot → {os.path.basename(out_pca3)}")
                    except Exception as e:
                        self.log(f"⚠ Could not create PCA 3D plot: {e}")
                else:
                    self.log("⚠ Skipping PCA 3D plot: Requires at least 3 taxa")

                # Save character loadings
                loadings = pd.DataFrame(pca.components_.T, index=characters,
                                         columns=[f"PC{i+1}" for i in range(len(explained_var))])
                out_load = os.path.join(self.outputs_dir, "pca_character_loadings.csv")
                loadings.to_csv(out_load)
                self.log(f"✓ Saved PCA character loadings → {os.path.basename(out_load)}")

            completed += 1
            self.progress.set(completed / total)

            # 4) NMDS
            if 'nmds' in steps:
                self.log("─" * 80)
                self.log("Running Non-metric Multidimensional Scaling (NMDS)...")
                try:
                    dissim_matrix = 1 - sim_smc
                    n_dims_max = min(6, len(taxa))
                    stress_vals = []
                    for d in range(1, n_dims_max+1):
                        nmds = MDS(n_components=d, dissimilarity='precomputed',
                                   random_state=42, n_init=10)
                        nmds.fit(dissim_matrix)
                        stress_vals.append(nmds.stress_)

                    fig = plt.figure(figsize=(6,5))
                    plt.plot(range(1, n_dims_max+1), stress_vals, marker='o')
                    plt.xlabel("Number of Dimensions")
                    plt.ylabel("Stress")
                    plt.title("NMDS Stress Plot")
                    plt.tight_layout()
                    out_stress = os.path.join(self.outputs_dir, "nmds_stress_plot.png")
                    fig.savefig(out_stress, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved NMDS Stress plot → {os.path.basename(out_stress)}")

                    # NMDS 2D
                    nmds2 = MDS(n_components=2, dissimilarity='precomputed',
                                random_state=42, n_init=10)
                    nmds_2d = nmds2.fit_transform(dissim_matrix)
                    stress2 = nmds2.stress_
                    fig = plt.figure(figsize=scatter_size)
                    plt.scatter(nmds_2d[:,0], nmds_2d[:,1], s=60, c='#3498db',
                                alpha=0.7, edgecolors='black', linewidth=0.5)
                    for i, txt in enumerate(taxa):
                        plt.text(nmds_2d[i,0], nmds_2d[i,1], txt, fontsize=scatter_font)
                    plt.xlabel("NMDS1")
                    plt.ylabel("NMDS2")
                    plt.title(f"NMDS 2D Scatter Plot (Stress={stress2:.4f})")
                    plt.tight_layout()
                    out_nmds2 = os.path.join(self.outputs_dir, "nmds_2d_plot.png")
                    fig.savefig(out_nmds2, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved NMDS 2D plot → {os.path.basename(out_nmds2)} (Stress={stress2:.4f})")

                    # NMDS 3D
                    if len(taxa) >= 3:
                        nmds3d = MDS(n_components=3, dissimilarity='precomputed',
                                     random_state=42, n_init=10)
                        nmds_3d = nmds3d.fit_transform(dissim_matrix)
                        stress3 = nmds3d.stress_
                        fig = plt.figure(figsize=scatter_size)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(nmds_3d[:,0], nmds_3d[:,1], nmds_3d[:,2],
                                   s=60, c='#3498db', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            ax.text(nmds_3d[i,0], nmds_3d[i,1], nmds_3d[i,2],
                                    txt, fontsize=scatter_font)
                        ax.set_xlabel("NMDS1")
                        ax.set_ylabel("NMDS2")
                        ax.set_zlabel("NMDS3")
                        plt.title(f"NMDS 3D Scatter Plot (Stress={stress3:.4f})")
                        plt.tight_layout()
                        out_nmds3 = os.path.join(self.outputs_dir, "nmds_3d_plot.png")
                        fig.savefig(out_nmds3, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved NMDS 3D plot → {os.path.basename(out_nmds3)}")
                    else:
                        self.log("⚠ Skipping NMDS 3D plot: Requires at least 3 taxa")
                except Exception as e:
                    self.log(f"✗ NMDS failed: {e}")
            completed += 1
            self.progress.set(completed / total)

            # 5) Heatmap
            if 'heatmap' in steps:
                self.log("─" * 80)
                self.log("Creating similarity heatmap (SMC)...")
                try:
                    fig = plt.figure(figsize=heatmap_size)
                    annot_bool = scatter_font > 5 and len(taxa) <= 30
                    annot_kws = {"fontsize": scatter_font * 1.2} if annot_bool else {}
                    sns.heatmap(
                        sim_smc_df,
                        cmap='YlGnBu',
                        annot=annot_bool,
                        square=False,
                        cbar_kws={'label': 'Similarity', 'shrink': 0.65},
                        annot_kws=annot_kws,
                        xticklabels=True,
                        yticklabels=True,
                        linewidths=0.3,
                        linecolor='gray'
                    )
                    plt.tick_params(axis='x', labelsize=dendro_font)
                    plt.tick_params(axis='y', labelsize=dendro_font)
                    plt.title("Simple Matching Coefficient Similarity Heatmap")
                    plt.tight_layout()
                    out_heat = os.path.join(self.outputs_dir, "similarity_heatmap.png")
                    fig.savefig(out_heat, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved heatmap → {os.path.basename(out_heat)}")
                except Exception as e:
                    self.log(f"✗ Heatmap generation failed: {e}")
            completed += 1
            self.progress.set(completed / total)

            # 6) UMAP
            if 'umap' in steps:
                if UMAP_AVAILABLE:
                    self.log("─" * 80)
                    self.log("Running UMAP (2D + 3D)...")
                    try:
                        # UMAP 2D
                        umap_2d = umap.UMAP(n_neighbors=min(5, len(taxa)-1),
                                             min_dist=0.1, random_state=42, n_components=2)
                        umap_result_2d = umap_2d.fit_transform(binary)
                        fig = plt.figure(figsize=scatter_size)
                        plt.scatter(umap_result_2d[:,0], umap_result_2d[:,1],
                                    s=60, c='#9b59b6', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            plt.text(umap_result_2d[i,0], umap_result_2d[i,1],
                                     txt, fontsize=scatter_font)
                        plt.xlabel("UMAP1")
                        plt.ylabel("UMAP2")
                        plt.title("UMAP 2D Scatter Plot")
                        plt.tight_layout()
                        out_umap2d = os.path.join(self.outputs_dir, "umap_2d_plot.png")
                        fig.savefig(out_umap2d, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved UMAP 2D plot → {os.path.basename(out_umap2d)}")

                        # UMAP 3D
                        if len(taxa) >= 3:
                            try:
                                umap_3d = umap.UMAP(n_neighbors=min(5, len(taxa)-1),
                                                     min_dist=0.1, random_state=42, n_components=3)
                                umap_result_3d = umap_3d.fit_transform(binary)
                                fig = plt.figure(figsize=scatter_size)
                                ax = fig.add_subplot(111, projection='3d')
                                ax.scatter(umap_result_3d[:,0], umap_result_3d[:,1],
                                          umap_result_3d[:,2], s=60, c='#9b59b6',
                                          alpha=0.7, edgecolors='black', linewidth=0.5)
                                for i, txt in enumerate(taxa):
                                    ax.text(umap_result_3d[i,0], umap_result_3d[i,1],
                                            umap_result_3d[i,2], txt, fontsize=scatter_font)
                                ax.set_xlabel("UMAP1")
                                ax.set_ylabel("UMAP2")
                                ax.set_zlabel("UMAP3")
                                plt.title("UMAP 3D Scatter Plot")
                                plt.tight_layout()
                                out_umap3d = os.path.join(self.outputs_dir, "umap_3d_plot.png")
                                fig.savefig(out_umap3d, dpi=300)
                                plt.close(fig)
                                self.log(f"✓ Saved UMAP 3D plot → {os.path.basename(out_umap3d)}")
                            except Exception as e:
                                self.log(f"⚠ Could not create UMAP 3D plot: {e}")
                        else:
                            self.log("⚠ Skipping UMAP 3D plot: Requires at least 3 taxa")
                    except Exception as e:
                        self.log(f"✗ UMAP failed: {e}")
                else:
                    self.log("⚠ UMAP was requested but unavailable (package not installed)")
            completed += 1
            self.progress.set(completed / total)

            # 7) t-SNE
            if 'tsne' in steps:
                self.log("─" * 80)
                self.log("Running t-SNE (2D + 3D)...")
                try:
                    perplexity_val = min(30, len(taxa)-1)
                    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
                    tsne_result_2d = tsne_2d.fit_transform(binary)
                    fig = plt.figure(figsize=scatter_size)
                    plt.scatter(tsne_result_2d[:,0], tsne_result_2d[:,1],
                                s=60, c='#e67e22', alpha=0.7, edgecolors='black', linewidth=0.5)
                    for i, txt in enumerate(taxa):
                        plt.text(tsne_result_2d[i,0], tsne_result_2d[i,1],
                                 txt, fontsize=scatter_font)
                    plt.xlabel("t-SNE1")
                    plt.ylabel("t-SNE2")
                    plt.title("t-SNE 2D Scatter Plot")
                    plt.tight_layout()
                    out_tsne2d = os.path.join(self.outputs_dir, "tsne_2d_plot.png")
                    fig.savefig(out_tsne2d, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved t-SNE 2D plot → {os.path.basename(out_tsne2d)}")

                    # t-SNE 3D
                    if len(taxa) >= 3:
                        try:
                            tsne_3d = TSNE(n_components=3, random_state=42,
                                           perplexity=perplexity_val)
                            tsne_result_3d = tsne_3d.fit_transform(binary)
                            fig = plt.figure(figsize=scatter_size)
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(tsne_result_3d[:,0], tsne_result_3d[:,1],
                                      tsne_result_3d[:,2], s=60, c='#e67e22',
                                      alpha=0.7, edgecolors='black', linewidth=0.5)
                            for i, txt in enumerate(taxa):
                                ax.text(tsne_result_3d[i,0], tsne_result_3d[i,1],
                                        tsne_result_3d[i,2], txt, fontsize=scatter_font)
                            ax.set_xlabel("t-SNE1")
                            ax.set_ylabel("t-SNE2")
                            ax.set_zlabel("t-SNE3")
                            plt.title("t-SNE 3D Scatter Plot")
                            plt.tight_layout()
                            out_tsne3d = os.path.join(self.outputs_dir, "tsne_3d_plot.png")
                            fig.savefig(out_tsne3d, dpi=300)
                            plt.close(fig)
                            self.log(f"✓ Saved t-SNE 3D plot → {os.path.basename(out_tsne3d)}")
                        except Exception as e:
                            self.log(f"⚠ Could not create t-SNE 3D plot: {e}")
                    else:
                        self.log("⚠ Skipping t-SNE 3D plot: Requires at least 3 taxa")
                except Exception as e:
                    self.log(f"✗ t-SNE failed: {e}")
            completed += 1
            self.progress.set(completed / total)

            # Cluster summary
            try:
                clusters = fcluster(linkage_matrix_upgma, t=0.2, criterion='distance')
                cluster_table = pd.DataFrame({'Taxa': taxa, 'Cluster': clusters})
                out_cluster = os.path.join(self.outputs_dir, "taxa_cluster_summary.csv")
                cluster_table.to_csv(out_cluster, index=False)
                self.log(f"✓ Saved cluster summary → {os.path.basename(out_cluster)}")
            except Exception as e:
                self.log(f"⚠ Could not save cluster summary: {e}")

            self.log("")
            self.log("═" * 80)
            self.log("✓ All selected analyses completed successfully!")
            self.log(f"✓ Outputs are in the '{self.outputs_dir}' folder")
            self.log("═" * 80)
        except Exception as e:
            self.log("")
            self.log("═" * 80)
            self.log(f"✗ Pipeline error: {e}")
            self.log("═" * 80)
        finally:
            self.run_btn.configure(state="normal", fg_color="#27ae60")
            self.progress.set(1.0)


if __name__ == "__main__":
    app = PheneticaApp()
    app.root.mainloop()
