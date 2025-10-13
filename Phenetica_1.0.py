"""
Phenetica GUI (Tkinter)
Author: Sheikh Sunzid Ahmed and M. Oliur Rahman, 2025
Enhanced UI: Modern aesthetics and improved splash screen
"""

import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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


try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ---------------------------
# Helper functions
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
    """Enhanced splash screen that stays on top and displays properly"""
    splash = tk.Toplevel()
    splash.title("Phenetica 1.0")
    splash.overrideredirect(True)
    splash_width = 520
    splash_height = 280
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
    splash.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
    splash.configure(bg="#2c3e50")
    
    # Main container with border
    main_frame = tk.Frame(splash, bg="#ecf0f1", highlightbackground="#3498db", 
                          highlightthickness=3)
    main_frame.place(x=3, y=3, width=splash_width-6, height=splash_height-6)
    
    # Top colored bar
    top_bar = tk.Frame(main_frame, bg="#3498db", height=8)
    top_bar.pack(fill='x')
    
    # Logo/Icon
    logo_label = tk.Label(main_frame, text="🌿", font=("Segoe UI", 52), 
                          bg="#ecf0f1", fg="#27ae60")
    logo_label.pack(pady=(35,5))
    
    # Title
    title_label = tk.Label(main_frame, text="Phenetica 1.0", 
                           font=("Segoe UI", 26, "bold"), 
                           fg="#2c3e50", bg="#ecf0f1")
    title_label.pack(pady=(0,3))
    
    # Subtitle
    subtitle_label = tk.Label(main_frame, text="Advanced Morphometric Analysis Suite", 
                             font=("Segoe UI", 11), 
                             fg="#7f8c8d", bg="#ecf0f1")
    subtitle_label.pack(pady=(0,20))
    
    # Version
    version_label = tk.Label(main_frame, text="Version 1.0", 
                            font=("Segoe UI", 9), 
                            fg="#95a5a6", bg="#ecf0f1")
    version_label.pack(pady=(0,8))
    
    # Developers info
    dev_frame = tk.Frame(main_frame, bg="#ecf0f1")
    dev_frame.pack(pady=(5,0))
    
    dev_label = tk.Label(dev_frame, 
                         text="Sheikh Sunzid Ahmed & M. Oliur Rahman", 
                         font=("Segoe UI", 10, "bold"), 
                         fg="#34495e", bg="#ecf0f1")
    dev_label.pack()
    
    dept_label = tk.Label(dev_frame, 
                          text="Department of Botany, University of Dhaka", 
                          font=("Segoe UI", 9), 
                          fg="#7f8c8d", bg="#ecf0f1")
    dept_label.pack(pady=(2,0))
    
    # Progress indicator (animated dots)
    loading_label = tk.Label(main_frame, text="Loading", 
                            font=("Segoe UI", 9), 
                            fg="#3498db", bg="#ecf0f1")
    loading_label.pack(pady=(15,0))
    
    # Animated loading dots
    dots = [".", "..", "...", ""]
    dot_index = [0]
    
    def animate_dots():
        if splash.winfo_exists():
            loading_label.config(text=f"Loading{dots[dot_index[0]]}")
            dot_index[0] = (dot_index[0] + 1) % len(dots)
            splash.after(400, animate_dots)
    splash.attributes('-topmost', True)
    splash.lift()
    splash.focus_force()
    splash.update()
    animate_dots()
    def close_splash():
        if splash.winfo_exists():
            splash.destroy()
    
    splash.after(duration, close_splash)
    
    return splash

class PheneticaApp:
    def __init__(self, root):
        self.root = root
        
        
        root.withdraw()
        
        
        splash = show_splash_screen(root, duration=3500)
        
        
        root.after(3600, lambda: self.setup_main_window())
        
    def setup_main_window(self):
        """Setup the main application window"""
        root = self.root
        
        root.title("Phenetica 1.0 — Advanced Morphometric Analysis Suite")
        root.geometry("900x800")  # Increased height to ensure footer visibility
        root.minsize(850, 750)   # Set minimum size but allow resizing
        root.configure(bg="#ecf0f1")
        
        
        style = ttk.Style()
        style.theme_use('clam')
        
        
        style.configure('TFrame', background='#ecf0f1')
        style.configure('TLabel', background='#ecf0f1', foreground='#2c3e50', 
                       font=('Segoe UI', 10))
        style.configure('TLabelframe', background='#ecf0f1', foreground='#2c3e50',
                       font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe.Label', background='#ecf0f1', foreground='#2c3e50',
                       font=('Segoe UI', 10, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10), padding=6)
        style.configure('TCheckbutton', background='#ecf0f1', foreground='#2c3e50',
                       font=('Segoe UI', 10))
        style.map('TCheckbutton', background=[('active', '#ecf0f1')])
        
        # Configure Entry style
        style.configure('TEntry', fieldbackground='white', foreground='#2c3e50',
                       borderwidth=1, relief='solid')
        style.configure('TCombobox', fieldbackground='white', foreground='#2c3e50',
                       borderwidth=1, relief='solid')
        
        # Create main container with proper weight distribution
        main_container = tk.Frame(root, bg="#ecf0f1")
        main_container.pack(fill='both', expand=True)
        
        # Header frame with gradient effect
        header_frame = tk.Frame(main_container, bg="#27ae60", height=70)  # Changed to green
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        header_icon = tk.Label(header_frame, text="🌿", font=("Segoe UI", 28), 
                              bg="#27ae60", fg="white")  # Changed to green
        header_icon.pack(side='left', padx=(20,10), pady=10)

        header_text_frame = tk.Frame(header_frame, bg="#27ae60")  # Changed to green
        header_text_frame.pack(side='left', pady=10)

        header_title = tk.Label(header_text_frame, text="Phenetica 1.0", 
                               font=("Segoe UI", 18, "bold"), 
                               bg="#27ae60", fg="white")  # Changed to green
        header_title.pack(anchor='w')

                
        
        # Main content frame - this will expand
        content_frame = tk.Frame(main_container, bg="#ecf0f1")
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Top frame: File selection
        top_frame = ttk.LabelFrame(content_frame, text="  Input Configuration  ", padding=15)
        top_frame.pack(fill='x', pady=(0,12))
        
        # File selection row
        file_frame = tk.Frame(top_frame, bg="#ecf0f1")
        file_frame.pack(fill='x', pady=(0,10))
        
        ttk.Label(file_frame, text="Data Matrix File:", 
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0,10))
        
        self.file_entry = ttk.Entry(file_frame, width=55, font=('Segoe UI', 10))
        self.file_entry.pack(side='left', padx=(0,10), ipady=4)
        
        browse_btn = tk.Button(file_frame, text="📁 Browse", command=self.browse_file,
                              bg="#3498db", fg="white", font=('Segoe UI', 10, 'bold'),
                              relief='flat', padx=15, pady=6, cursor='hand2',
                              activebackground="#2980b9", activeforeground="white")
        browse_btn.pack(side='left')
        
        # Delimiter selection row
        delim_frame = tk.Frame(top_frame, bg="#ecf0f1")
        delim_frame.pack(fill='x')
        
        ttk.Label(delim_frame, text="Delimiter:", 
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0,10))
        
        self.delim_var = tk.StringVar(value="auto")
        delim_combo = ttk.Combobox(delim_frame, textvariable=self.delim_var, 
                                   values=["auto", "\\t", ","], width=15, 
                                   state='readonly', font=('Segoe UI', 10))
        delim_combo.pack(side='left', ipady=3)
        
        ttk.Label(delim_frame, text="(auto-detect, tab, or comma)", 
                 font=('Segoe UI', 9), foreground='#7f8c8d').pack(side='left', padx=(10,0))
        
        # Analysis selection frame
        analysis_frame = ttk.LabelFrame(content_frame, text="  Select Analyses  ", padding=15)
        analysis_frame.pack(fill='x', pady=(0,12))
        
        self.chk_vars = {}
        options = [
            ("📊 Similarity Matrices (SMC + Jaccard)", "similarity", True),
            ("🌳 UPGMA Dendrogram (average linkage)", "upgma", True),
            ("📈 PCA (2D, 3D, eigenvalues, loadings)", "pca", True),
            ("🎯 NMDS (2D + 3D)", "nmds", False),
            ("🔥 Similarity Heatmap (SMC)", "heatmap", False),
            ("🌲 Alternative Dendrograms (4 methods)", "alt_dend", False),
            ("🗺️ UMAP (2D + 3D) — optional", "umap", False),
            ("🎨 t-SNE (2D + 3D) — optional", "tsne", False)
        ]
        
        
        for i, (label, key, default) in enumerate(options):
            var = tk.BooleanVar(value=default)
            self.chk_vars[key] = var
            
            chk_frame = tk.Frame(analysis_frame, bg="#ecf0f1")
            chk_frame.grid(row=i//2, column=i%2, sticky='w', padx=15, pady=6)
            
            chk = ttk.Checkbutton(chk_frame, text=label, variable=var)
            chk.pack(anchor='w')
        
        
        control_frame = tk.Frame(content_frame, bg="#ecf0f1")
        control_frame.pack(fill='x', pady=(0,12))
        
        self.run_btn = tk.Button(control_frame, text="▶ Run Selected Analyses", 
                                command=self.run_clicked,
                                bg="#27ae60", fg="white", 
                                font=('Segoe UI', 12, 'bold'),
                                relief='flat', padx=25, pady=10, cursor='hand2',
                                activebackground="#229954", activeforeground="white")
        self.run_btn.pack(side='left', padx=(0,15))
        
        
        style.configure("Custom.Horizontal.TProgressbar", 
                       troughcolor='#bdc3c7', 
                       background='#27ae60',
                       borderwidth=0,
                       thickness=20)
        
        self.progress = ttk.Progressbar(control_frame, orient='horizontal', 
                                       mode='determinate', length=500,
                                       style="Custom.Horizontal.TProgressbar")
        self.progress.pack(side='left', fill='x', expand=True)
        
        
        log_frame = ttk.LabelFrame(content_frame, text="  Analysis Log  ", padding=10)
        log_frame.pack(fill='both', expand=True, pady=(0,10))
        
        
        log_container = tk.Frame(log_frame, bg="white")
        log_container.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side='right', fill='y')
        
        self.log_text = tk.Text(log_container, height=14, state='disabled', wrap='word',
                               bg='#2c3e50', fg='#ecf0f1', font=('Consolas', 9),
                               relief='flat', padx=10, pady=10,
                               yscrollcommand=scrollbar.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        
        footer_frame = tk.Frame(main_container, bg="#34495e", height=65)
        footer_frame.pack(side='bottom', fill='x')
        footer_frame.pack_propagate(False)
        
        
        top_border = tk.Frame(footer_frame, bg="#3498db", height=2)
        top_border.pack(fill='x')
        
        
        footer_content = tk.Frame(footer_frame, bg="#34495e")
        footer_content.pack(expand=True, pady=8)
        
        
        developed_label = tk.Label(footer_content, 
                                   text="Developed by", 
                                   font=("Times New Roman", 9), 
                                   fg="#95a5a6", bg="#34495e")
        developed_label.pack()
        
        
        authors_label = tk.Label(footer_content, 
                                text="Sheikh Sunzid Ahmed and M. Oliur Rahman, Department of Botany, University of Dhaka", 
                                font=("Times New Roman", 10, "bold"), 
                                fg="#ecf0f1", bg="#34495e")
        authors_label.pack(pady=(2,4))
        
        # Affiliation
        affiliation_label = tk.Label(footer_content, 
                                     text="Department of Botany, University of Dhaka", 
                                     font=("Times New Roman", 11), 
                                     fg="#bdc3c7", bg="#34495e")
        affiliation_label.pack()
        
        # Copyright notice
        copyright_label = tk.Label(footer_content, 
                                   text="© 2024 | All Rights Reserved", 
                                   font=("Times New Roman", 9), 
                                   fg="#7f8c8d", bg="#34495e")
        copyright_label.pack(pady=(3,0))
        
        # Initialize state
        self.input_path = None
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")
        
        
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)  # Content frame expands
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(3, weight=1)   # Log frame expands
        
        
        root.deiconify()
        root.lift()
        root.focus_force()
        
        # Log welcome message
        self.log("═" * 80)
        self.log("Welcome to Phenetica 1.0")
        self.log("Advanced Morphometric Analysis Suite")
        self.log("═" * 80)
        self.log("Ready to perform morphometric analyses.")
        self.log("Please select your input data file to begin.")
        self.log("")

    def log(self, message):
        t = timestamp()
        if message.startswith("═") or message.startswith("Advanced"):
            full = f"{message}\n"
        else:
            full = f"[{t}] {message}\n"
        self.log_text.configure(state='normal')
        self.log_text.insert('end', full)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')
        print(full, end='')

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Data Matrix File",
            filetypes=[("CSV/TSV files", "*.csv *.tsv *.txt"), 
                      ("All files", "*.*")]
        )
        if path:
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, path)
            self.input_path = path
            self.log(f"File selected: {os.path.basename(path)}")

    def run_clicked(self):
        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Input Required", 
                               "Please select a valid input file first.",
                               parent=self.root)
            return

        os.makedirs(self.outputs_dir, exist_ok=True)
        self.run_btn.configure(state='disabled', bg='#95a5a6')
        self.progress['value'] = 0
        self.log("═" * 80)
        thread = threading.Thread(target=self.execute_pipeline, args=(path,))
        thread.start()

    def execute_pipeline(self, path):
        try:
            self.log("Starting analyses pipeline...")
            delim_choice = self.delim_var.get()
            if delim_choice == "auto":
                try:
                    df = pd.read_csv(path, sep='\t', index_col=None)
                    if df.shape[1] <= 1:
                        raise Exception("Tab read gave 1 column")
                except Exception:
                    df = pd.read_csv(path, sep=',', index_col=None)
            else:
                sep = '\t' if delim_choice == "\\t" else ','
                df = pd.read_csv(path, sep=sep, index_col=None)

            df = df.fillna(0)
            try:
                numeric_df = df.apply(pd.to_numeric)
                data = numeric_df
            except Exception:
                data = df.T.apply(pd.to_numeric, errors='coerce').fillna(0)
            if data.shape[0] < data.shape[1]:
                data = data.T
            data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
            taxa = list(data.columns)
            characters = list(data.index)
            self.log(f"✓ Input read: {os.path.basename(path)}")
            self.log(f"  Data shape: {data.shape} (characters × taxa)")
            self.log(f"  Taxa count: {len(taxa)}")
            self.log("")
            
            plot_params = get_plot_style_params(len(taxa))
            scatter_size = plot_params['scatter_fig_size']
            scatter_font = plot_params['scatter_font_size']
            dendro_size = plot_params['dendro_fig_size']
            dendro_font = plot_params['dendro_leaf_font_size']
            heatmap_size = plot_params['heatmap_fig_size']
            binary = data.T.values  # taxa x characters

            steps = []
            if self.chk_vars['similarity'].get(): steps.append('similarity')
            if self.chk_vars['upgma'].get(): steps.append('upgma')
            if self.chk_vars['pca'].get(): steps.append('pca')
            if self.chk_vars['nmds'].get(): steps.append('nmds')
            if self.chk_vars['heatmap'].get(): steps.append('heatmap')
            if self.chk_vars['alt_dend'].get(): steps.append('alt_dend')
            if self.chk_vars['umap'].get() and UMAP_AVAILABLE: steps.append('umap')
            elif self.chk_vars['umap'].get() and not UMAP_AVAILABLE:
                self.log("⚠ UMAP requested but 'umap' package not available. Skipping UMAP.")
            if self.chk_vars['tsne'].get(): steps.append('tsne')

            total = max(1, len(steps))
            completed = 0

            # 1) Similarity matrices
            if 'similarity' in steps:
                self.log("─" * 80)
                self.log("Computing Simple Matching (SMC) distance and similarity...")
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
                dist_smc = pdist(binary, metric='hamming')
                sim_smc = 1 - squareform(dist_smc)
                sim_smc_df = pd.DataFrame(sim_smc, index=taxa, columns=taxa)

            completed += 1
            self.progress['value'] = (completed/total) * 100

            # 2) UPGMA dendrogram
            if 'upgma' in steps:
                self.log("─" * 80)
                self.log("Running UPGMA (average linkage) clustering...")
                linkage_matrix = linkage(dist_smc, method='average')
                fig = plt.figure(figsize=dendro_size) 
                dendrogram(linkage_matrix, labels=taxa, leaf_rotation=90, leaf_font_size=dendro_font)
                plt.title("UPGMA Dendrogram (SMC)")
                plt.tight_layout()
                out_png = os.path.join(self.outputs_dir, "upgma_dendrogram.png")
                fig.savefig(out_png, dpi=300)
                plt.close(fig)
                self.log(f"✓ Saved UPGMA dendrogram → {os.path.basename(out_png)}")
            else:
                linkage_matrix = linkage(dist_smc, method='average')
            completed += 1
            self.progress['value'] = (completed/total) * 100

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

                # PCA 2D with unique color (Red)
                fig = plt.figure(figsize=scatter_size)
                plt.scatter(pca_result[:,0], pca_result[:,1], s=60, c='#e74c3c', alpha=0.7, edgecolors='black', linewidth=0.5)
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

                # PCA 3D with unique color (Red)
                if len(taxa) >= 3:
                    try:
                        fig = plt.figure(figsize=scatter_size)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], s=60, c='#e74c3c', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            ax.text(pca_result[i,0], pca_result[i,1], pca_result[i,2], txt, fontsize=scatter_font) 
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
            self.progress['value'] = (completed/total) * 100

            # 4) NMDS (2D + 3D)
            if 'nmds' in steps:
                self.log("─" * 80)
                self.log("Running Non-metric Multidimensional Scaling (NMDS)...")
                try:
                    n_dims_max = min(6, len(taxa))
                    stress_vals = []
                    for d in range(1, n_dims_max+1):
                        nmds = MDS(n_components=d, dissimilarity='precomputed', random_state=42, n_init=10)
                        nmds.fit(1-sim_smc)
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

                    # NMDS 2D with unique color (Blue)
                    nmds2 = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10)
                    nmds_2d = nmds2.fit_transform(1 - sim_smc)
                    stress2 = nmds2.stress_
                    out_nmds2 = os.path.join(self.outputs_dir, "nmds_2d_plot.png")
                    fig = plt.figure(figsize=scatter_size) 
                    plt.scatter(nmds_2d[:,0], nmds_2d[:,1], s=60, c='#3498db', alpha=0.7, edgecolors='black', linewidth=0.5)
                    for i, txt in enumerate(taxa):
                        plt.text(nmds_2d[i,0], nmds_2d[i,1], txt, fontsize=scatter_font)
                    plt.xlabel("NMDS1")
                    plt.ylabel("NMDS2")
                    plt.title(f"NMDS 2D Scatter Plot (Stress={stress2:.4f})")
                    plt.tight_layout()
                    fig.savefig(out_nmds2, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved NMDS 2D plot → {os.path.basename(out_nmds2)} (Stress={stress2:.4f})")

                    # NMDS 3D with unique color (Blue)
                    if len(taxa) >= 3:
                        nmds3d = MDS(n_components=3, dissimilarity='precomputed', random_state=42, n_init=10)
                        nmds_3d = nmds3d.fit_transform(1 - sim_smc)
                        stress3 = nmds3d.stress_
                        fig = plt.figure(figsize=scatter_size)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(nmds_3d[:,0], nmds_3d[:,1], nmds_3d[:,2], s=60, c='#3498db', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            ax.text(nmds_3d[i,0], nmds_3d[i,1], nmds_3d[i,2], txt, fontsize=scatter_font)
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
            self.progress['value'] = (completed/total) * 100

            # 5) Heatmap of SMC similarity
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
                        cbar_kws={'label':'Similarity', 'shrink':0.65},
                        annot_kws=annot_kws,
                        xticklabels=True, 
                        yticklabels=True,
                        linewidths=0.3,
                        linecolor='gray'
                    )
                    plt.tick_params(axis='x', labelsize=dendro_font)
                    plt.tick_params(axis='y', labelsize=dendro_font)
                    plt.title("Simple Matching Similarity Heatmap")
                    plt.tight_layout()
                    out_heat = os.path.join(self.outputs_dir, "similarity_heatmap.png")
                    fig.savefig(out_heat, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved heatmap → {os.path.basename(out_heat)}")
                except Exception as e:
                    self.log(f"✗ Heatmap generation failed: {e}")
            completed += 1
            self.progress['value'] = (completed/total) * 100

            # 6) Alternative clustering dendrograms
            if 'alt_dend' in steps:
                self.log("─" * 80)
                self.log("Generating alternative dendrograms...")
                methods = ['single', 'complete', 'average', 'ward']
                for method in methods:
                    try:
                        dist_input = dist_smc if method != 'ward' else pdist(binary, metric='euclidean')
                        link_mat = linkage(dist_input, method=method)
                        fig = plt.figure(figsize=dendro_size)
                        dendrogram(link_mat, labels=taxa, leaf_rotation=90, leaf_font_size=dendro_font)
                        plt.title(f"Dendrogram ({method.capitalize()} linkage)")
                        plt.tight_layout()
                        outname = os.path.join(self.outputs_dir, f"dendrogram_{method}.png")
                        fig.savefig(outname, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved dendrogram ({method}) → {os.path.basename(outname)}")
                    except Exception as e:
                        self.log(f"✗ Failed {method} dendrogram: {e}")
            completed += 1
            self.progress['value'] = (completed/total) * 100

            # 7) UMAP (optional) - 2D and 3D
            if 'umap' in steps:
                if UMAP_AVAILABLE:
                    self.log("─" * 80)
                    self.log("Running UMAP (2D + 3D)...")
                    try:
                        # UMAP 2D with unique color (Purple)
                        umap_2d = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42, n_components=2)
                        umap_result_2d = umap_2d.fit_transform(binary)
                        fig = plt.figure(figsize=scatter_size)
                        plt.scatter(umap_result_2d[:,0], umap_result_2d[:,1], s=60, c='#9b59b6', alpha=0.7, edgecolors='black', linewidth=0.5)
                        for i, txt in enumerate(taxa):
                            plt.text(umap_result_2d[i,0], umap_result_2d[i,1], txt, fontsize=scatter_font)
                        plt.xlabel("UMAP1")
                        plt.ylabel("UMAP2")
                        plt.title("UMAP 2D Scatter Plot")
                        plt.tight_layout()
                        out_umap2d = os.path.join(self.outputs_dir, "umap_2d_plot.png")
                        fig.savefig(out_umap2d, dpi=300)
                        plt.close(fig)
                        self.log(f"✓ Saved UMAP 2D plot → {os.path.basename(out_umap2d)}")
                        
                        # UMAP 3D with unique color (Purple)
                        if len(taxa) >= 3:
                            try:
                                umap_3d = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42, n_components=3)
                                umap_result_3d = umap_3d.fit_transform(binary)
                                fig = plt.figure(figsize=scatter_size)
                                ax = fig.add_subplot(111, projection='3d')
                                ax.scatter(umap_result_3d[:,0], umap_result_3d[:,1], umap_result_3d[:,2], s=60, c='#9b59b6', alpha=0.7, edgecolors='black', linewidth=0.5)
                                for i, txt in enumerate(taxa):
                                    ax.text(umap_result_3d[i,0], umap_result_3d[i,1], umap_result_3d[i,2], txt, fontsize=scatter_font)
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
            self.progress['value'] = (completed/total) * 100

            # 8) t-SNE (optional) - 2D and 3D
            if 'tsne' in steps:
                self.log("─" * 80)
                self.log("Running t-SNE (2D + 3D)...")
                try:
                    # t-SNE 2D with unique color (Orange)
                    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(taxa)-1))
                    tsne_result_2d = tsne_2d.fit_transform(binary)
                    fig = plt.figure(figsize=scatter_size)
                    plt.scatter(tsne_result_2d[:,0], tsne_result_2d[:,1], s=60, c='#e67e22', alpha=0.7, edgecolors='black', linewidth=0.5)
                    for i, txt in enumerate(taxa):
                        plt.text(tsne_result_2d[i,0], tsne_result_2d[i,1], txt, fontsize=scatter_font)
                    plt.xlabel("t-SNE1")
                    plt.ylabel("t-SNE2")
                    plt.title("t-SNE 2D Scatter Plot")
                    plt.tight_layout()
                    out_tsne2d = os.path.join(self.outputs_dir, "tsne_2d_plot.png")
                    fig.savefig(out_tsne2d, dpi=300)
                    plt.close(fig)
                    self.log(f"✓ Saved t-SNE 2D plot → {os.path.basename(out_tsne2d)}")
                    
                    # t-SNE 3D with unique color (Orange)
                    if len(taxa) >= 3:
                        try:
                            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(taxa)-1))
                            tsne_result_3d = tsne_3d.fit_transform(binary)
                            fig = plt.figure(figsize=scatter_size)
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(tsne_result_3d[:,0], tsne_result_3d[:,1], tsne_result_3d[:,2], s=60, c='#e67e22', alpha=0.7, edgecolors='black', linewidth=0.5)
                            for i, txt in enumerate(taxa):
                                ax.text(tsne_result_3d[i,0], tsne_result_3d[i,1], tsne_result_3d[i,2], txt, fontsize=scatter_font)
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
            self.progress['value'] = (completed/total) * 100

            try:
                clusters = fcluster(linkage_matrix, t=0.2, criterion='distance')
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
            self.run_btn.configure(state='normal', bg='#27ae60')
            self.progress['value'] = 100

if __name__ == "__main__":
    root = tk.Tk()
    app = PheneticaApp(root)
    root.mainloop()