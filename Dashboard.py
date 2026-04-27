import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from function import (threshold, estrazioni_feature_e_nomi,
                      extract_paths, get_real_feature_columns, get_feature_columns, read_input_csv, load_rgb_image,
                      apply_rgb_threshold, resize_image_for_preview, update_dataset_with_feature_rows, save_csv
                      , build_feature_row,    extract_image_features,
    format_features_for_display,
    append_warning_to_display_text,compute_pca_from_dataframe, run_kmeans)
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
        self._build_ui()  # builds and configures the entire graphical interface
        self._auto_bind_threshold()  # connect the tk.IntVar objects for the thresholds to the callbacks
        self.nomi_features_geometriche=["height", "equivalent_diameter","width", "area", "aspect_ratio", "extent", "solidity",
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
    def _build_top_bar(self):
        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill="x")

        top_bar.grid_columnconfigure(0, weight=0)
        top_bar.grid_columnconfigure(1, weight=1)
        top_bar.grid_columnconfigure(2, weight=0)

        left_bar = ttk.Frame(top_bar)
        left_bar.grid(row=0, column=0, sticky="w")

        center_bar = ttk.Frame(top_bar)
        center_bar.grid(row=0, column=1)

        right_bar = ttk.Frame(top_bar)
        right_bar.grid(row=0, column=2, sticky="e")

        ttk.Button(left_bar, text="Load CSV", command=self.load_csv).pack(
            side="left", padx=4, pady=(0, 4)
        )
        ttk.Button(center_bar, text="Extract features", command=self.extract_features).pack(
            side="left", padx=4, pady=(0, 4)
        )
        ttk.Button(center_bar, text="Save features", command=self.save_features).pack(
            side="left", padx=4, pady=(0, 4)
        )
        ttk.Button(center_bar, text="Save dataset features", command=self.save_features_all).pack(
            side="left", padx=4, pady=(0, 4)
        )
        ttk.Button(right_bar, text="Show PCA", command=self.open_subwindow).pack(
            side="left", padx=4, pady=(0, 4)
        )

    def _build_navigation_bar(self):
        navigation_bar = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        navigation_bar.pack(fill="x")

        navigation_bar.grid_columnconfigure(0, weight=0)
        navigation_bar.grid_columnconfigure(1, weight=1)
        navigation_bar.grid_columnconfigure(2, weight=0)

        button_area = ttk.Frame(navigation_bar)
        button_area.grid(row=0, column=0, sticky="w")

        status_area = ttk.Frame(navigation_bar)
        status_area.grid(row=0, column=1)

        ttk.Button(button_area, text="<< Previous", command=self.prev).pack(
            side="left", padx=4
        )
        ttk.Button(button_area, text="Next >>", command=self.next).pack(
            side="left", padx=4
        )

        self.status_label = ttk.Label(status_area, text="No CSV loaded.")
        self.status_label.pack()

    def _build_settings_panel(self):
        settings_panel = ttk.LabelFrame(
            self.root,
            text="RGB threshold and LBP parameters",
            padding=8,
        )
        settings_panel.pack(fill="x", padx=8, pady=6)

        self._add_threshold_row(settings_panel, 0, "R", self.rmin, self.rmax)
        self._add_threshold_row(settings_panel, 1, "G", self.gmin, self.gmax)
        self._add_threshold_row(settings_panel, 2, "B", self.bmin, self.bmax)

        ttk.Label(settings_panel, text="Radius").grid(row=0, column=5, sticky="e")
        ttk.Spinbox(
            settings_panel,
            from_=1,
            to=20,
            textvariable=self.raggio,
            width=6,
        ).grid(row=0, column=6, padx=(6, 18))

        ttk.Label(settings_panel, text="Points").grid(row=1, column=5, sticky="e")
        ttk.Spinbox(
            settings_panel,
            from_=4,
            to=64,
            textvariable=self.punti,
            width=6,
        ).grid(row=1, column=6, padx=(6, 18))

    def _add_threshold_row(self, parent, row_index, channel_name, min_var, max_var):
        ttk.Label(parent, text=channel_name, width=2).grid(
            row=row_index, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Label(parent, text="min").grid(row=row_index, column=1, sticky="e")
        ttk.Entry(parent, textvariable=min_var, width=6).grid(
            row=row_index, column=2, padx=(4, 12)
        )
        ttk.Label(parent, text="max").grid(row=row_index, column=3, sticky="e")
        ttk.Entry(parent, textvariable=max_var, width=6).grid(
            row=row_index, column=4, padx=(4, 18)
        )

    def _build_image_preview_panel(self, parent):
        image_area = ttk.Frame(parent)
        image_area.pack(side="left", fill="both", expand=True, padx=(0, 6))

        original_frame = ttk.LabelFrame(image_area, text="Original")
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.original_image_label = ttk.Label(original_frame)
        self.original_image_label.pack(fill="both", expand=True, padx=8, pady=8)

        mask_frame = ttk.LabelFrame(image_area, text="Binary mask")
        mask_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.mask_image_label = ttk.Label(mask_frame)
        self.mask_image_label.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_feature_output_panel(self, parent):
        feature_area = ttk.LabelFrame(
            parent,
            text="Features",
            padding=8,
            width=420,
        )
        feature_area.pack(side="right", fill="both")

        self.feature_text = tk.Text(feature_area, wrap="none", width=55)

        x_scrollbar = ttk.Scrollbar(
            feature_area,
            orient="horizontal",
            command=self.feature_text.xview,
        )
        y_scrollbar = ttk.Scrollbar(
            feature_area,
            orient="vertical",
            command=self.feature_text.yview,
        )

        self.feature_text.configure(
            xscrollcommand=x_scrollbar.set,
            yscrollcommand=y_scrollbar.set,
            font=("Consolas", 10),
        )

        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar.pack(side="bottom", fill="x")
        self.feature_text.pack(side="left", fill="both", expand=True)
    def _build_main_content(self):
        main_content = ttk.Frame(self.root, padding=8)
        main_content.pack(fill="both", expand=True)

        self._build_image_preview_panel(main_content)
        self._build_feature_output_panel(main_content)
    def _build_ui(self):
        """Build the main application user interface."""
        self._build_top_bar()
        self._build_navigation_bar()
        self._build_settings_panel()
        self._build_main_content()

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
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right", padx=(0, 8))
        ttk.Button(btns, text=" Refresh", command=self._pca_compute_refresh).pack(side="right")
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
        Compute PCA from the loaded CSV DataFrame.
        """
        if self.df_csv is None:
            return False, "Load a CSV file first."

        try:
            result = compute_pca_from_dataframe(
                df=self.df_csv,
                n_components=int(self.pca_n.get()),
                use_pca=bool(self.use_pca.get()),
            )

        except ValueError as error:
            return False, str(error)

        except Exception as error:
            return False, f"Unexpected PCA error: {error}"

        self.pca_model = result.model
        self.pca_scaler = result.scaler
        self.pca_scores = result.scores
        self.pca_feature_cols = result.feature_columns
        self._pca_valid_mask = result.valid_mask
        self._pca_evr = result.explained_variance_ratio

        return True, ""

    def _pca_compute_refresh(self):
        """
        Recompute PCA, reset previous clusters and update the PCA plot.
        """
        ok, message = self._pca_compute_inline()

        if not ok:
            messagebox.showwarning("PCA", message)
            return

        self.kmeans_labels = None
        self.kmeans_model = None
        self.kmeans_centroids = None
        self.kmeans_inertia = None

        self._pca_redraw()
    def _pca_run_kmeans(self):
        """
        Run K-Means on the current PCA scores and redraw the PCA plot.
        """
        if self.pca_scores is None or self.pca_scores.empty:
            messagebox.showwarning(
                "K-Means",
                "Compute PCA before running K-Means."
            )
            return

        try:
            result = run_kmeans(
                data=self.pca_scores.values,
                n_clusters=int(self.kmeans_k.get()),
                n_init=10,
                random_state=0,
            )

        except ValueError as error:
            messagebox.showwarning("K-Means", str(error))
            return

        except Exception as error:
            messagebox.showerror(
                "K-Means",
                f"Unable to run K-Means:\n{error}"
            )
            return

        self.kmeans_labels = result.labels
        self.kmeans_model = result.model
        self.kmeans_centroids = result.centroids
        self.kmeans_inertia = result.inertia

        self._pca_redraw()
    def open_subwindow(self):
        """
        Open the PCA window and display the initial PCA scatter plot.
        """
        self._pca_window_build("PCA + Plot", with_kmeans=True)
        self._pca_compute_refresh

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

    def load_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])

        if not csv_path:
            return

        self.input_csv_path = csv_path
        self.csv_dir = os.path.dirname(csv_path)

        try:
            self.df_csv = read_input_csv(self.input_csv_path)

        except Exception as error:
            messagebox.showerror("Error", f"Unreadable CSV:\n{error}")
            self.df_csv = None
            self.feature_cols = []
            self.paths = []
            return

        self.paths = extract_paths(self.df_csv)

        if not self.paths:
            messagebox.showwarning("Empty", "No path found in the 'path' column.")
            self.feature_cols = []
            return

            self.feature_cols = get_feature_columns(self.df_csv)

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
            self.img = load_rgb_image(path)
        except Exception as error:
            messagebox.showerror("Error", f"Unable to open:\n{path}\n\n{error}")

            self.img = None
            self.img_bin = None

            self.status_label.config(text=f"{self.i + 1}/{len(self.paths)} | ERROR: {path}")
            self.original_image_label.configure(image="")
            self.mask_image_label.configure(image="")
            self.tk1 = None
            self.tk2 = None
            return

        # Update status “x/y | path”
        self.status_label.config(text=f"{self.i + 1}/{len(self.paths)} | {path}")

        #  Show original image preview
        self._show(self.img, self.original_image_label, which=1)

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
            self.feature_text.delete("1.0", tk.END)
            self.last_features = None
            self.last_feature_names = None

    def _show(self, image, label, which):
        """
        Show a resized image preview in a Tkinter label.
        """
        preview_image = resize_image_for_preview(image)
        tk_image = ImageTk.PhotoImage(preview_image)

        label.configure(image=tk_image)

        if which == 1:
            self.tk1 = tk_image
        else:
            self.tk2 = tk_image

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
        """
        Apply the current RGB threshold to the loaded image.
        """
        if self._suspend_threshold:
            return

        if self.img is None:
            return

        try:
            self.img_bin = apply_rgb_threshold(
                image=self.img,
                threshold_function=threshold,
                rmin=int(self.rmin.get()),
                rmax=int(self.rmax.get()),
                gmin=int(self.gmin.get()),
                gmax=int(self.gmax.get()),
                bmin=int(self.bmin.get()),
                bmax=int(self.bmax.get()),
            )

            self._show(self.img_bin.convert("RGB"), self.mask_image_label, which=2)

        except Exception:
            self.img_bin = None
            self.mask_image_label.configure(image="")
            self.tk2 = None

    def extract_features(self):
        """
        Extract features from the current image and display them in the GUI.
        """
        if self.img is None:
            return

        if self.img_bin is None:
            messagebox.showwarning(
                "Threshold",
                "Invalid threshold: check min/max values."
            )
            return

        try:
            result = extract_image_features(
                image=self.img,
                binary_image=self.img_bin,
                extraction_function=estrazioni_feature_e_nomi,
                geometric_feature_names=self.nomi_features_geometriche,
                haralick_channel_feature_names=self.nomi_features_haralick_canali,
                channel_names=self.nomi_canali,
                lbp_radius=int(self.raggio.get()),
                lbp_points=int(self.punti.get()),
            )

        except Exception as error:
            messagebox.showerror("Feature extraction error", str(error))
            return

        self.last_features = result.values
        self.last_feature_names = result.names

        display_text = format_features_for_display(
            result.names,
            result.values,
        )

        display_text = append_warning_to_display_text(
            display_text,
            result.warning,
        )

        self.feature_text.delete("1.0", tk.END)
        self.feature_text.insert(tk.END, display_text)
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
            df = read_input_csv(self.input_csv_path)

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

    def get_threshold_values(self):
        """
        Read the current RGB threshold values from the GUI.
        """
        return {
            "thr_rmin": int(self.rmin.get()),
            "thr_rmax": int(self.rmax.get()),
            "thr_gmin": int(self.gmin.get()),
            "thr_gmax": int(self.gmax.get()),
            "thr_bmin": int(self.bmin.get()),
            "thr_bmax": int(self.bmax.get()),
        }
    def save_features_all(self):
        """
        Extract and save features for all images listed in the input CSV file.
        """
        if not self.paths:
            messagebox.showwarning(
                "Dataset",
                "First, load a CSV file containing image paths."
            )
            return

        if not self.input_csv_path or self.df_csv is None:
            messagebox.showwarning(
                "Dataset",
                "First, load an input CSV file."
            )
            return

        errors = []
        feature_rows = []

        for idx, img_path in enumerate(self.paths, start=1):
            try:
                image = load_rgb_image(img_path)

                binary_image = apply_rgb_threshold(
                    image=image,
                    threshold_function=threshold,
                    rmin=int(self.rmin.get()),
                    rmax=int(self.rmax.get()),
                    gmin=int(self.gmin.get()),
                    gmax=int(self.gmax.get()),
                    bmin=int(self.bmin.get()),
                    bmax=int(self.bmax.get()),
                )

                result = extract_image_features(
                    image=image,
                    binary_image=binary_image,
                    extraction_function=estrazioni_feature_e_nomi,
                    geometric_feature_names=self.nomi_features_geometriche,
                    haralick_channel_feature_names=self.nomi_features_haralick_canali,
                    channel_names=self.nomi_canali,
                    lbp_radius=int(self.raggio.get()),
                    lbp_points=int(self.punti.get()),
                )

                feature_row = build_feature_row(
                    image_path=img_path,
                    feature_names=result.names,
                    feature_values=result.values,
                    threshold_values=self.get_threshold_values(),
                    lbp_radius=int(self.raggio.get()),
                    lbp_points=int(self.punti.get()),
                )

                feature_rows.append(feature_row)

                if idx % 10 == 0 or idx == len(self.paths):
                    self.status_label.config(
                        text=f"Processed {idx}/{len(self.paths)}..."
                    )
                    self.root.update_idletasks()

            except Exception as error:
                errors.append((img_path, str(error)))

        if not feature_rows and errors:
            messagebox.showerror(
                "Error",
                f"No features saved. Errors in {len(errors)} images."
            )
            return

        try:
            updated_df = update_dataset_with_feature_rows(
                self.df_csv,
                feature_rows,
            )

            save_csv(updated_df, self.input_csv_path)

        except Exception as error:
            messagebox.showerror(
                "Error",
                f"Unable to save to the input CSV file:\n{error}"
            )
            return

        self.df_csv = updated_df
        self.feature_cols = get_real_feature_columns(self.df_csv)

        if errors:
            messagebox.showwarning(
                "Completed with errors",
                (
                    f"Saved in the input CSV file:\n{self.input_csv_path}\n\n"
                    f"OK: {len(feature_rows)} images\n"
                    f"Errors: {len(errors)} images\n\n"
                    f"First error:\n{errors[0][0]}\n{errors[0][1]}"
                )
            )
        else:
            messagebox.showinfo(
                "Completed",
                (
                    f"Saved in the input CSV file:\n{self.input_csv_path}\n\n"
                    f"OK: {len(feature_rows)} images"
                )
            )

        self.status_label.config(text="Ready.")
    def load_features_from_csv_for_current_image(self):
        """If the CSV file already contains feature columns, load and display the values for the current image."""
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
        self.feature_text.delete("1.0", tk.END)
        for n, v in zip(names, vals):
            self.feature_text.insert(tk.END, f"\t{n}=\t{v}\n")
        return True


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()