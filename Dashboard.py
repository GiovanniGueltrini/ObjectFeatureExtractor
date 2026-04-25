import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from function import threshold, estrazioni_feature_e_nomi, compute_pca_on_df_vars, run_kmeans_vars
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    def __init__(self, root):
        self.root: tk.Tk = root
        root.title("CSV -> Threshold automatico -> Feature")
        root.geometry("1200x700")

        self.input_csv_path: str = None # directory path
        self.paths = []         # vector of the paths of all images
        self.i = -1        # control variable for navigating files within the interface
        self.img = None         #global variable where to store the image
        self.img_bin = None        #global variable where to store the image mask
        self.tk1 = self.tk2 = None         #Reference to ImageTk.PhotoImage for the original preview and the binary mask preview
        self.last_features = None         # list of feature values extracted for the current image
        self.last_feature_names = None     # List of feature names
        self.csv_dir = None  # directory of the selected CSV file
        self.df_csv = None  # Pandas DataFrame from the uploaded CSV
        self._suspend_threshold = False # Flag for the threshold (True when setting IntVar from code)
        self.feature_cols = [] # List of “feature” columns in df_csv
        self.pca_model = None # PCA-based object
        self.pca_scores = None  #DataFrame of PCA scores for valid rows
        self.pca_feature_cols = []  # features actually used for the PCA fit
        self.kmeans_k = tk.IntVar(value=3)  #cluster number (k) chosen by the user for K-means
        self.kmeans_labels = None # array labels assigned by K-Means
        # soglie RGB
        self.rmin, self.rmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.gmin, self.gmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.bmin, self.bmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.use_pca = tk.BooleanVar( value=True)  # UI flag: if True, calculates PCA with n selected components; otherwise, calculates it with 2
        self.pca_n = tk.IntVar(value=2)  # number of required PCA components
        # Parameters LBP (Local Binary Patterns)
        self.raggio = tk.IntVar(value=3)  # LBP radius (in pixels): distance of points from the central pixel
        self.punti = tk.IntVar(value=4)  # number of LBP points/samples on the circle
        self._ui()  # builds and configures the entire graphical interface
        self._auto_bind_threshold()  # connect the tk.IntVar objects for the thresholds to the callbacks
        self.nomi_features_geometriche=["height", "equivalent_diameterwidth", "area", "aspect_ratio", "extent", "solidity",
                                     "",
                                     "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"] # names of geometric features
        self.nomi_feature_haralick = [ # names of textural features
            "mean_color",
            "variance_color",
            "Angular Second Moment",
            "Contrast",
            "Correlation",
            "Variance",
            "Inverse Difference Moment",
            "Sum Average",
            "Sum Variance",
            "Sum Entropy",
            "Entropy",
            "Difference Variance",
            "Difference Entropy",
            "Information Measure of Correlation 1",
            "Information Measure of Correlation 2"]
        self.nomi_canali = ["Red", "Green", "Blue"] # names of the channel
        # creation of the feature_names_haralick_channels vector
        nomi_features_haralick_canali=[]
        for nome in self.nomi_canali:
            nomi_features_haralick_canali = np.concatenate(
                [nomi_features_haralick_canali, [f"{n}_{nome}" for n in self.nomi_feature_haralick]])
        self.nomi_features_haralick_canali=nomi_features_haralick_canali


    def _ui(self):
        """ It builds the main UI (toolbar, threshold/LBP controls, image preview, feature text box).
        """
        #  Main controls + navigation + status
        # --- TOP BAR: left | center | right (fixed)
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        # 3 columns: Column 1 (in the center) fills the remaining space
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=0)

        left_bar = ttk.Frame(top)
        left_bar.grid(row=0, column=0, sticky="w")

        center_bar = ttk.Frame(top)
        center_bar.grid(row=0, column=1)  # no sticky: remains centered in the cell

        right_bar = ttk.Frame(top)
        right_bar.grid(row=0, column=2, sticky="e")

        # --- Left: CSV upload only
        ttk.Button(left_bar, text="Carica CSV", command=self.load_csv).pack(side="left", padx=4, pady=(0, 4))

        # --- Center: 3 buttons (centered)
        ttk.Button(center_bar, text="Estrai feature", command=self.extract_features).pack(side="left", padx=4,pady=(0, 4)) # extracts features from the current image
        ttk.Button(center_bar, text="Salva feature", command=self.save_features).pack(side="left", padx=4, pady=(0, 4))  # Save the features of the current image to a local file
        ttk.Button(center_bar, text="Salva feature dataset", command=self.save_features_all).pack(side="left", padx=4,pady=(0, 4))  # Save the features of all images to a CSV file

        # Right: Subwindow
        ttk.Button(right_bar, text="visualizza PCA", command=self.open_subwindow).pack(side="left", padx=4, pady=(0, 4))

        # Bottom bar: Prev / Next on the left, Status centered
        nav = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        nav.pack(fill="x")

        nav.grid_columnconfigure(0, weight=0)
        nav.grid_columnconfigure(1, weight=1)
        nav.grid_columnconfigure(2, weight=0)
        # Frame containing the Prev/Next buttons
        nav_left = ttk.Frame(nav)
        nav_left.grid(row=0, column=0, sticky="w")
        # The “main” frame where you enter your status
        nav_center = ttk.Frame(nav)
        nav_center.grid(row=0, column=1)  # centered

        # Pulsanti di navigazione
        ttk.Button(nav_left, text="<< Prev", command=self.prev).pack(side="left", padx=4)
        ttk.Button(nav_left, text="Next >>", command=self.next).pack(side="left", padx=4)

        self.status = ttk.Label(nav_center, text="Nessun CSV.")
        self.status.pack()

        #top.grid_columnconfigure(6, weight=1)

        #Control panel for threshold variables and linear binary patterns
        dash = ttk.LabelFrame(self.root, text="Threshold RGB (min/max) + LBP (raggio/punti)", padding=8)
        dash.pack(fill="x", padx=8, pady=6)

        # Helper for creating a row of min/max controls
        def row(r, name, vmin, vmax):
            ttk.Label(dash, text=name, width=2).grid(row=r, column=0, sticky="w", padx=(0, 8))
            ttk.Label(dash, text="min").grid(row=r, column=1, sticky="e")
            ttk.Entry(dash, textvariable=vmin, width=6).grid(row=r, column=2, padx=(4, 12))
            ttk.Label(dash, text="max").grid(row=r, column=3, sticky="e")
            ttk.Entry(dash, textvariable=vmax, width=6).grid(row=r, column=4, padx=(4, 18))

        # RGB thresholds: values read from self.rmin, self.rmax, etc.
        row(0, "R", self.rmin, self.rmax)
        row(1, "G", self.gmin, self.gmax)
        row(2, "B", self.bmin, self.bmax)

        # LBP parameters: Spin boxes linked to self.radius/self.points
        ttk.Label(dash, text="Raggio").grid(row=0, column=5, sticky="e")
        ttk.Spinbox(dash, from_=1, to=20, textvariable=self.raggio, width=6).grid(row=0, column=6, padx=(6, 18))
        ttk.Label(dash, text="Punti").grid(row=1, column=5, sticky="e")
        ttk.Spinbox(dash, from_=4, to=64, textvariable=self.punti, width=6).grid(row=1, column=6, padx=(6, 18))
        # Main window display area: image preview on the left, feature output on the right
        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        # Left section: two image panels side by side
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Original Preview
        f1 = ttk.LabelFrame(left, text="Originale")
        f1.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.p1 = ttk.Label(f1)  # label che ospita PhotoImage
        self.p1.pack(fill="both", expand=True, padx=8, pady=8)

        # Preview threshold
        f2 = ttk.LabelFrame(left, text="Threshold (binaria)")
        f2.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.p2 = ttk.Label(f2)
        self.p2.pack(fill="both", expand=True, padx=8, pady=8)

        # Right section: text box with features (names + values) and a scroll bar
        right = ttk.LabelFrame(body, text="Feature (nomi + valori)", padding=8, width=420)
        right.pack(side="right", fill="both")

        self.txt = tk.Text(right, wrap="none", width=55)  # wrap none: keep tabs/columns aligned
        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.txt.xview)
        yscroll = ttk.Scrollbar(right, orient="vertical", command=self.txt.yview)
        self.txt.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        # Pack scroll + text
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.txt.pack(side="left", fill="both", expand=True)

        # Monospace font for better alignment of “name = value”
        self.txt.configure(font=("Consolas", 10))
    def _pca_window_build(self, title, with_kmeans):
        #Create a “child” window dedicated to PCA
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("950x700")
        win.transient(self.root)
        win.grab_set()
        #Controls frame on top + chart frame on bottom
        ctrl = ttk.Frame(win, padding=10)
        ctrl.pack(fill="x")
        plot_frm = ttk.Frame(win, padding=10)
        plot_frm.pack(fill="both", expand=True)
        #TK variables that store the X/Y axis selection (PC1, PC2, ...)
        self._pca_x_var = tk.StringVar(value="PC1")
        self._pca_y_var = tk.StringVar(value="PC2")

        # X/Y axis controls: two read-only combo boxes
        ttk.Label(ctrl, text="Asse X:").grid(row=0, column=0, sticky="e", padx=(0, 4))
        self._pca_x_cb = ttk.Combobox(ctrl, textvariable=self._pca_x_var, values=["PC1", "PC2"], width=6, state="readonly")
        self._pca_x_cb.grid(row=0, column=1, sticky="w", padx=(0, 14))

        ttk.Label(ctrl, text="Asse Y:").grid(row=0, column=2, sticky="e", padx=(0, 4))
        self._pca_y_cb = ttk.Combobox(ctrl, textvariable=self._pca_y_var, values=["PC1", "PC2"], width=6, state="readonly")
        self._pca_y_cb.grid(row=0, column=3, sticky="w", padx=(0, 14))

        # 5) “n-component” check: spinbox displays self.pca_n
        ttk.Label(ctrl, text="numero componenti PCA:").grid(row=0, column=4, sticky="e", padx=(0, 4))
        self._pca_spn = ttk.Spinbox(ctrl, from_=2, to=999, textvariable=self.pca_n, width=6)
        self._pca_spn.grid(row=0, column=5, sticky="w", padx=(0, 14))

        #control k_means
        ttk.Label(ctrl, text="k cluster:").grid(row=0, column=6, sticky="e", padx=(0, 4))
        ttk.Spinbox(ctrl, from_=2, to=50, textvariable=self.kmeans_k, width=6).grid(row=0, column=7, sticky="w", padx=(0, 14))

        # Create a figure/axis using Matplotlib + a Tkinter canvas that hosts the plot
        self._pca_fig = plt.Figure()
        self._pca_ax = self._pca_fig.add_subplot(111)
        self._pca_canvas = FigureCanvasTkAgg(self._pca_fig, master=plot_frm)
        self._pca_canvas.get_tk_widget().pack(fill="both", expand=True)
        #When you change the X-axis or Y-axis, call _pca_redraw()
        self._pca_x_cb.bind("<<ComboboxSelected>>", lambda e: self._pca_redraw())
        self._pca_y_cb.bind("<<ComboboxSelected>>", lambda e: self._pca_redraw())
        # Enable/disable the PCA component spinbox when self.use_pca changes
        def _sync_state(*_):
            self._pca_spn.configure(state=("normal" if self.use_pca.get() else "disabled"))
        self.use_pca.trace_add("write", _sync_state)
        _sync_state()  # sincronizzo subito lo stato iniziale
        # Button bar: Close + Calculate/Refresh and k-means
        btns = ttk.Frame(ctrl)
        btns.grid(row=1, column=0, columnspan=8, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Chiudi", command=win.destroy).pack(side="right", padx=(0, 8))
        ttk.Button(btns, text="Calcola / Refresh", command=self._pca_compute_refresh).pack(side="right")
        ttk.Button(btns, text="K-Means", command=self._pca_run_kmeans).pack(side="left", padx=(0, 8))
        # Brings the window to the front and gives it focus
        win.lift()
        win.focus_force()
        return win

    def _pca_refresh_pc_dropdowns(self):

        # Update the PC dropdown menus based on the columns available in self.pca_scores.

        # If I haven't calculated the PCA yet (or the scores dataframe is empty), I can't update anything
        if self.pca_scores is None or self.pca_scores.empty:
            return
        # Retrieve the list of available components: [“PC1”, ‘PC2’, ..., “PCn”]
        pcs = list(self.pca_scores.columns)
        # Update the values displayed in the X and Y drop-down lists with the actual list of calculated PCs
        self._pca_x_cb.configure(values=pcs)
        self._pca_y_cb.configure(values=pcs)
        # If the current selection is no longer valid (e.g., I previously had PC5 and now I only have PC1–PC3),
        # I reset it to a valid value.
        if self._pca_x_var.get() not in pcs:
            self._pca_x_var.set(pcs[0])  # default: first available PC
        if self._pca_y_var.get() not in pcs:
            # default: second PC if it exists, otherwise the same as the first
            self._pca_y_var.set(pcs[1] if len(pcs) > 1 else pcs[0])

    def _pca_redraw(self):
        """
        Update the PCA plot based on the selected components.
        If available, also use K-means clusters to color the points.
        """
        # Calculate PCA internally from the DataFrame (column cleaning, standardization, PCA fit, scores).
        self._pca_refresh_pc_dropdowns()
        # I read which columns the user has selected for the X and Y axes
        xcol, ycol = self._pca_x_var.get(), self._pca_y_var.get()
        # I extract the values to be plotted
        xs = self.pca_scores[xcol].values
        ys = self.pca_scores[ycol].values
        # I retrieve the matplotlib axis and clear it
        ax = self._pca_ax
        ax.clear()
        # If I have already run K-Means and have labels that match the number of data points,
        # group the data points by cluster (c=labels). Otherwise, use a standard scatter plot.
        if self.kmeans_labels is not None and len(self.kmeans_labels) == len(self.pca_scores):
            ax.scatter(xs, ys, c=self.kmeans_labels)
            ax.set_title(f"{xcol} vs {ycol} (KMeans k={int(self.kmeans_k.get())})")
        else:
            ax.scatter(xs, ys)
            ax.set_title(f"{xcol} vs {ycol}")

        # Board labels
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        # Update the Tkinter canvas without blocking (draw_idle is more “gentle” than draw())
        self._pca_canvas.draw_idle()

    def _pca_compute_inline(self):
        """
        Calculate the PCA based on the loaded DataFrame and save the results.
        Clean the data, keep only valid numeric columns, and standardize the features.
        """
        # Prerequisite: I must have loaded a CSV file into `self.df_csv`
        if self.df_csv is None:
            return False, "Carica prima un CSV."

        # Columns to EXCLUDE from PCA
        exclude = {"path", "lbp_raggio", "lbp_punti",
                   "thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax"}

        # I take the candidate columns = all except the excluded ones
        cols = [c for c in self.df_csv.columns if c not in exclude]

        #  Convert to numeric
        #    Then I remove the columns that are entirely NaN
        tmp = self.df_csv[cols].apply(pd.to_numeric, errors="coerce")
        cols = [c for c in cols if not tmp[c].isna().all()]
        if not cols:
            return False, "Nessuna colonna numerica valida per PCA."

        # Create the X feature matrix
        X = self.df_csv[cols].apply(pd.to_numeric, errors="coerce")
        # I keep ONLY the complete rows (without NaN) to avoid errors and ensure consistent PCA
        valid = ~X.isna().any(axis=1)  # Boolean mask on rows
        Xv = X.loc[valid].values  # final matrix (n_valid_rows, n_features)
        if Xv.shape[0] < 2:
            return False, "Poche righe valide (NaN) per PCA."
        # I choose n components:
        #    - if use_pca=True, I use the value specified by the user
        #    - if use_pca=False, I set it to 2
        ncomp = int(self.pca_n.get()) if self.use_pca.get() else 2
        ncomp = max(2, min(ncomp, Xv.shape[1]))

        # PCA fit + transformation to scores
        #    scores has shape (n_valid_rows, ncomp)
        self.pca_scaler = StandardScaler().fit(Xv)
        Xs = self.pca_scaler.transform(Xv)

        # PCA fit + transformation to scores
        #    scores has shape (n_valid_rows, ncomp)
        self.pca_model = PCA(n_components=ncomp).fit(Xs)
        scores = self.pca_model.transform(Xs)

        # I put the scores into a DataFrame with columns PC1 through PCn,
        self.pca_scores = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, ncomp + 1)])

        #  Save metadata useful for debugging/reuse:
        self.pca_feature_cols = cols
        self._pca_valid_mask = valid
        self._pca_evr = self.pca_model.explained_variance_ratio_
        return True, ""

    def _pca_compute_from_helper(self):
        """
        Calculate the PCA using an external function and save the results in the app.
        If something goes wrong, return an error message.
        """
        #  Check: I must have loaded the CSV into self.df_csv
        if self.df_csv is None:
            return False, "Carica prima un CSV."

        #    returns:
        #    ok/msg: result and message
        #    pca: fitted PCA object
        #    scaler: fitted StandardScaler object (
        #    scores_df: DataFrame containing PCA scores
        #    cols: list of feature columns actually used
        #       valid_mask: mask of valid rows used for the fit
        #    evr: explained_variance_ratio_

        ok, msg, pca, scaler, scores_df, cols, valid_mask, evr = compute_pca_on_df_vars(self.df_csv,use_pca=bool(self.use_pca.get()), n_components=int(self.pca_n.get()),)
        # If the helper reports an error
        #    propagate the error to the caller.
        if not ok:
            return False, msg
        # If everything is OK, save the results to `self` like this:
        #    - `_pca_redraw()` can plot `self.pca_scores`
        #    - `KMeans` can operate on `self.pca_scores.values`
        #    - In the future, you can reuse the scaler/PCA to transform new data consistently
        self.pca_model = pca
        self.pca_scaler = scaler
        self.pca_scores = scores_df
        self.pca_feature_cols = cols
        self._pca_valid_mask = valid_mask
        self._pca_evr = evr

        # 5) Return OK
        return True, ""

    def _pca_compute_refresh(self):
        """
            Recalculate the PCA, reset the previous clusters, and update the graph.
            If the calculation fails, display a warning and stop.
        """
        #I choose which PCA “pipeline” to use based on how the window was opened.
        #    - _pca_mode == “helper”  -> use the external function compute_pca_on_df_vars (wraps everything)
        #    - otherwise             -> use the inline version (calculate PCA within this class)
        if getattr(self, "_pca_mode", None) == "helper":
            ok, msg = self._pca_compute_from_helper()
        else:
            ok, msg = self._pca_compute_inline()
        # If the calculation fails, I display a warning and stop: no redraw, no state update.
        if not ok:
            messagebox.showwarning("PCA", msg)
            return
        #When I recalculate the PCA, the old K-means labels are no longer valid
        self.kmeans_labels = None

        # I'm redrawing the scatter plot using the new PCA scores (self.pca_scores)
        self._pca_redraw()

    def _pca_run_kmeans(self):
        """
        Groups PCA points into clusters using K-means.
        If an error occurs, displays a message; otherwise, redraws the plot.
        """
        # Run K-Means in PCA space
        #    - k: number of clusters chosen by the user (spin box)
        #    - n_init: number of different initializations to try (higher = more robust)
        #    - random_state: repeatability of results
        ok, msg, labels, km, _ = run_kmeans_vars(self.pca_scores.values,k=int(self.kmeans_k.get()), n_init=10,random_state=0)
        #  If clustering fails
        if not ok:
            messagebox.showwarning("K-Means", msg)
            return

        # If everything is OK, save:
        #    - labels: to color the points in the scatter plot
        #    - km: K-means model (useful if you want inertia, centroids, etc.)
        self.kmeans_labels = labels
        self.kmeans_model = km
        # This time, I'm redrawing the scatter plot using colors by cluster.
        self._pca_redraw()

    def open_subwindow(self):
        """
        Opens the PCA window, sets up the controls, and displays the first graph.
        """
        self._pca_mode = "inline"
        # I create the window with KMeans controls enabled
        self._pca_window_build("PCA + Plot", with_kmeans=True)
        # I'll calculate the PCA right away and plot the initial scatter plot.
        self._pca_compute_refresh()

    def open_pca_plot_window(self):
        # This window uses PCA via the external helper `compute_pca_on_df_vars`
        self._pca_mode = "helper"
        #  Building the window without K-means
        self._pca_window_build("PCA scatter", with_kmeans=False)
        #  I'll calculate the PCA right away and plot the initial scatter plot.
        self._pca_compute_refresh()
    def _auto_bind_threshold(self):
        # Automatically apply the threshold when the values change
        for v in (self.rmin, self.rmax, self.gmin, self.gmax, self.bmin, self.bmax):
            v.trace_add("write", lambda *_: self.apply_threshold_safe())
    def _read_input_csv_as_df(self):
        # Reads the input CSV file
        df = pd.read_csv(self.input_csv_path, encoding="utf-8-sig")
        # Case 1: The “path” column already exists
        if "path" in df.columns:
            df["path"] = df["path"].astype(str).str.strip()
            return df
        # Case 2: CSV file created with a single column and no header
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "path"})
        df["path"] = df["path"].astype(str).str.strip()
        return df

    def load_csv(self):
        # select CSV
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fp:
            return
        # Save CSV file path and directory
        self.input_csv_path = fp
        self.csv_dir = os.path.dirname(fp)
        #  I'll try to read the CSV file
        try:
            # This function must:
            # - read the CSV file
            # - ensure that there is a “path” column
            # - strip spaces
            self.df_csv = self._read_input_csv_as_df()

        except Exception as e:
            messagebox.showerror("error", f"Unreadable CSV:\n{e}")
            self.df_csv = None
            self.feature_cols = []
            self.paths = []
            return

        #  I extract the paths from the “path” column
        if "path" not in self.df_csv.columns:
            messagebox.showwarning("Empty", "The CSV file does not contain a ‘path’ column.")
            self.paths = []
            self.feature_cols = []
            return
        self.paths = (self.df_csv["path"].astype(str).str.strip().replace("", np.nan).dropna().tolist())
        # If there are no valid paths, stop
        if not self.paths:
            messagebox.showwarning("Empty", "No path found in the ‘path’ column.")
            self.feature_cols = []
            return
        #  List of available feature columns
        self.feature_cols = [c for c in self.df_csv.columns if c != "path"]
        # I select the first image and upload it
        self.i = 0
        self.load_image()

    def prev(self):
        # If I don't have any paths loaded, I can't browse
        if not self.paths:
            return
        # Decrease the index
        self.i = max(0, self.i - 1)
        # I'm uploading the image corresponding to the new index
        self.load_image()

    def next(self):
        # If I don't have any paths loaded, I can't browse
        if not self.paths:
            return
        # Increase the index
        self.i = min(len(self.paths) - 1, self.i + 1)
        # Re-load
        self.load_image()

    def load_image(self):
        """Upload the image to be processed"""
        if not self.paths:
            return
        if self.i < 0 or self.i >= len(self.paths):
            self.i = 0

        # Current path to open
        path = self.paths[self.i]

        #  I'll try opening the image and converting it to RGB
        try:
            self.img = Image.open(path).convert("RGB")
        except Exception as e:
            # If it fails: I'll display a warning, clear the state, and stop
            messagebox.showerror("Errore", f"Impossibile aprire:\n{path}\n\n{e}")
            self.img = None
            self.img_bin = None

            # Update the status and clear the preview so the old image isn't left behind
            self.status.config(text=f"{self.i + 1}/{len(self.paths)} | ERRORE: {path}")
            self.p1.configure(image="")
            self.p2.configure(image="")
            self.tk1 = None
            self.tk2 = None
            return

        # Update status “x/y | path”
        self.status.config(text=f"{self.i + 1}/{len(self.paths)} | {path}")

        #  Show original image preview
        self._show(self.img, self.p1, which=1)

        # I'll try to load the thresholds associated with this image from the CSV file.
        #    If the thresholds (thr_*) are present in that row, I'll set them and apply the threshold once.
        loaded_thr = self.load_threshold_from_csv_for_current_image()

        # If I didn't find any thresholds in the CSV, I'll still apply the threshold using the current values in the UI
        if not loaded_thr:
            self.apply_threshold_safe()

        # If the CSV file already contains feature columns for this image, I'll display them in the text box.
        loaded = self.load_features_from_csv_for_current_image()

        # If there are no saved features, I clear the text box and reset the internal state
        if not loaded:
            self.txt.delete("1.0", tk.END)
            self.last_features = None
            self.last_feature_names = None

    def _show(self, img, label, which):
        # Fixed size for the preview
        W, H = 520, 520
        iw, ih = img.size
        s = min(W / iw, H / ih)
        img2 = img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.Resampling.NEAREST)
        tkimg = ImageTk.PhotoImage(img2)
        label.configure(image=tkimg)
        if which == 1:
            self.tk1 = tkimg
        else:
            self.tk2 = tkimg

    def load_threshold_from_csv_for_current_image(self):
        """
        Load the thresholds associated with the current image from the CSV file.
        If the correct values are found, update the interface and apply the threshold.
        """
        # Validation: I must have a loaded DataFrame, a list of paths, and a valid index
        if self.df_csv is None or not self.paths:
            return False
        if self.i < 0 or self.i >= len(self.paths):
            return False
        # Check: I need to have the “path” column in the CSV file in order to perform a match
        if "path" not in self.df_csv.columns:
            return False
        #  I take the current path and look for the corresponding row in the DataFrame
        img_path = self.paths[self.i]
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False
        #  I'll take the first line that matches
        row = match.iloc[0]
        # Columns required to reset the threshold
        needed = ["thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax"]
        #  If even a single thr_* column is missing from the CSV file, I can't upload anything
        if not all(c in self.df_csv.columns for c in needed):
            return False

        # Set the UI's IntVars using a suspension flag:
        try:
            self._suspend_threshold = True

            self.rmin.set(int(row["thr_rmin"]))
            self.rmax.set(int(row["thr_rmax"]))
            self.gmin.set(int(row["thr_gmin"]))
            self.gmax.set(int(row["thr_gmax"]))
            self.bmin.set(int(row["thr_bmin"]))
            self.bmax.set(int(row["thr_bmax"]))

            # LBP parameters: I only set them if they exist
            if "lbp_raggio" in self.df_csv.columns and not pd.isna(row.get("lbp_raggio", None)):
                self.raggio.set(int(row["lbp_raggio"]))
            if "lbp_punti" in self.df_csv.columns and not pd.isna(row.get("lbp_punti", None)):
                self.punti.set(int(row["lbp_punti"]))
        finally:
            # In any case, I'll reactivate the triggers
            self._suspend_threshold = False
        # I apply the threshold ONLY once with the final values set
        self.apply_threshold_safe()
        #  I confirm that I have successfully imported thresholds from the CSV file
        return True

    def apply_threshold_safe(self):
        #  When setting thresholds, I avoid recalculating the threshold every time I call .set() on the IntVars.
        if self._suspend_threshold:
            return
        #  If I don't have an image loaded, I can't apply the threshold
        if self.img is None:
            return
        try:
            # 3) I call your threshold function using the current IntVar values (R/G/B min/max)
            #    int(...) is needed because IntVars may contain temporary strings during editing
            self.img_bin = threshold(
                self.img,
                int(self.rmin.get()), int(self.rmax.get()),
                int(self.gmin.get()), int(self.gmax.get()),
                int(self.bmin.get()), int(self.bmax.get())
            )
            #  Here's a preview of the binary mask:
            self._show(self.img_bin.convert("RGB"), self.p2, which=2)

        except Exception:
            #    When entering data in the Entry fields, invalid input may occur
            self.img_bin = None
            self.p2.configure(image="")  # clears the preview
            self.tk2 = None  # Release reference to PhotoImage

    def extract_features(self):
        # If I don't have an image uploaded, I can't extract anything
        if self.img is None:
            return
        #  Features depend on the bitmask: if the threshold is not met or missing, issue a warning and stop
        if self.img_bin is None:
            messagebox.showwarning(
                "Threshold",
                "Threshold non valido: controlla min/max (0..255 e min<=max)."
            )
            return

        # I'll try to calculate the features by calling your external function.
        try:
            features, nomi = estrazioni_feature_e_nomi(
                self.img, self.img_bin,
                self.nomi_features_geometriche,
                self.nomi_features_haralick_canali,
                self.nomi_canali,
                raggio=int(self.raggio.get()),
                punti=int(self.punti.get())
            )
        except Exception as e:
            # If the extraction fails, I display the error and stop.
            messagebox.showerror("Errore feature", str(e))
            return

        # Save the latest result in the app's status
        self.last_features = features
        self.last_feature_names = nomi
        #  I clear the text box and type “name = value” in a way that's easy to read
        self.txt.delete("1.0", tk.END)
        #  If, for some reason, the lengths don't match, I'll print only the overlapping part
        n = min(len(features), len(nomi))
        for k in range(n):
            name = str(nomi[k])
            val = features[k]
            self.txt.insert(tk.END, f"\t{name}=\t{val}\n")

        #  If there is a mismatch, I'll flag it in the text box as a diagnostic warning
        if len(features) != len(nomi):
            self.txt.insert(
                tk.END,
                f"\n[WARN] len(features)={len(features)} diverso da len(nomi)={len(nomi)}\n"
            )
    def save_features(self):
        """except for the fields in the CSV"""
        if self.last_features is None or self.last_feature_names is None:
            messagebox.showwarning("Save", "First, extract the features (click the ‘Extract Features’ button)').")
            return
        if not self.paths or self.i < 0:
            messagebox.showwarning("Save", "No images selected.")
            return
        if not self.input_csv_path:
            messagebox.showwarning("Save", "First, upload an input CSV file.")
            return

        img_path = self.paths[self.i]

        # Build a feature dictionary
        nomi = list(self.last_feature_names)
        features = list(self.last_features)
        n = min(len(features), len(nomi))
        row_dict = {nomi[k]: features[k] for k in range(n)}
        row_dict["path"] = img_path
        row_dict["thr_rmin"] = int(self.rmin.get())
        row_dict["thr_rmax"] = int(self.rmax.get())
        row_dict["thr_gmin"] = int(self.gmin.get())
        row_dict["thr_gmax"] = int(self.gmax.get())
        row_dict["thr_bmin"] = int(self.bmin.get())
        row_dict["thr_bmax"] = int(self.bmax.get())
        row_dict["lbp_raggio"] = int(self.raggio.get())
        row_dict["lbp_punti"] = int(self.punti.get())
        try:
            df = self._read_input_csv_as_df()

            # create df row feature
            feat_df = pd.DataFrame([row_dict])

            # merge: adds/updates feature columns for that row (path)
            df = df.merge(feat_df, on="path", how="left", suffixes=("", "_new"))

            # If there are duplicate columns with “_new”, I'll take the new ones and overwrite them
            for c in feat_df.columns:
                if c == "path":
                    continue
                newc = f"{c}_new"
                if newc in df.columns:
                    df[c] = df[newc]
                    df.drop(columns=[newc], inplace=True)

            df.to_csv(self.input_csv_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Save", f"Features saved in the input CSV file:\n{self.input_csv_path}")

        except Exception as e:
                messagebox.showerror("Errore", f"Unable to save to the input CSV file:\n{e}")
        self.df_csv = df
        self.feature_cols = [c for c in self.df_csv.columns if c != "path"]



    def save_features_all(self):
        "Save the dimensions of all images in the CSV file"
        if not self.paths:
            messagebox.showwarning("Dataset", "First, upload a CSV file containing the paths.")
            return
        if not self.input_csv_path:
            messagebox.showwarning("Dataset", "First, upload an input CSV file.")
            return

        errors = []
        rows = []

        for idx, img_path in enumerate(self.paths, start=1):
            try:
                img = Image.open(img_path).convert("RGB")
                img_bin = threshold(
                    img,
                    int(self.rmin.get()), int(self.rmax.get()),
                    int(self.gmin.get()), int(self.gmax.get()),
                    int(self.bmin.get()), int(self.bmax.get())
                )

                features, nomi = estrazioni_feature_e_nomi(
                    img, img_bin,
                    self.nomi_features_geometriche,
                    self.nomi_features_haralick_canali,
                    self.nomi_canali,
                    raggio=int(self.raggio.get()),
                    punti=int(self.punti.get())
                )

                nomi = list(nomi)
                n = min(len(features), len(nomi))
                row = {nomi[k]: features[k] for k in range(n)}
                row["path"] = img_path
                row["thr_rmin"] = int(self.rmin.get())
                row["thr_rmax"] = int(self.rmax.get())
                row["thr_gmin"] = int(self.gmin.get())
                row["thr_gmax"] = int(self.gmax.get())
                row["thr_bmin"] = int(self.bmin.get())
                row["thr_bmax"] = int(self.bmax.get())
                row["lbp_raggio"] = int(self.raggio.get())
                row["lbp_punti"] = int(self.punti.get())
                rows.append(row)

                if idx % 10 == 0 or idx == len(self.paths):
                    self.status.config(text=f"Elaborate {idx}/{len(self.paths)}...")
                    self.root.update_idletasks()

            except Exception as e:
                errors.append((img_path, str(e)))

        if not rows and errors:
            messagebox.showerror("Errore", f"No features saved. Errors in {len(errors)} images.")
            return

        try:
            df = self._read_input_csv_as_df()
            feat_df = pd.DataFrame(rows)

            # merge sulle path
            df = df.merge(feat_df, on="path", how="left", suffixes=("", "_new"))

            # Overwrites any duplicate columns
            for c in feat_df.columns:
                if c == "path":
                    continue
                newc = f"{c}_new"
                if newc in df.columns:
                    df[c] = df[newc]
                    df.drop(columns=[newc], inplace=True)

            df.to_csv(self.input_csv_path, index=False, encoding="utf-8-sig")

        except Exception as e:
            messagebox.showerror("Errore", f"Unable to save to the input CSV file:\n{e}")
            return

        if errors:
            messagebox.showwarning("Completato con errori",f"Saved in the input CSV file:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini\nErrori: {len(errors)} immagini\n"   f"Esempio 1° errore:\n{errors[0][0]}\n{errors[0][1]}"    )
        else:
            messagebox.showinfo("Completato", f"Saved in the input CSV file:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini")
        self.status.config(text="Pronto.")
        self.df_csv = df
        thr_cols = {"thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax", "lbp_raggio", "lbp_punti"}
        self.feature_cols = [c for c in self.df_csv.columns if c != "path" and c not in thr_cols]
    def load_features_from_csv_for_current_image(self):
        """If the CSV file already contains feature columns, load and display the values for the current image."""

        #  Prerequisites: I must have:
        #    - df_csv loaded
        #    - a non-empty list of feature columns (feature_cols)
        #    - a list of paths
        #    - a valid index i
        if self.df_csv is None or not self.feature_cols or not self.paths:
            return False
        if self.i < 0 or self.i >= len(self.paths):
            return False
        # Path to the current image
        img_path = self.paths[self.i]

        #  I'm looking for the row in the DataFrame that corresponds to that path
        if "path" not in self.df_csv.columns:
            return False
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False

        row = match.iloc[0]

        #4) I'm just listing the features that are available:
        names = []
        vals = []
        for c in self.feature_cols:
            v = row.get(c, None)
            # Treat NaN or an empty string as “missing”
            if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
                continue
            names.append(str(c))
            vals.append(v)

        # If I haven't found any highlighted features, I won't upload anything
        if not names:
            return False
        #  Update the internal status
        self.last_feature_names = names
        self.last_features = vals
        #  Display the features in the text box
        self.txt.delete("1.0", tk.END)
        for n, v in zip(names, vals):
            self.txt.insert(tk.END, f"\t{n}=\t{v}\n")
        return True


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()