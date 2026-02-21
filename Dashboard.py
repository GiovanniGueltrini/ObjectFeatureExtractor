import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
# >>> metti i tuoi import reali qui
from function import threshold, estrazioni_feature_e_nomi
import numpy as np
import pandas as pd
import os

class App:
    def __init__(self, root):
        self.root = root
        root.title("CSV -> Threshold automatico -> Feature")
        root.geometry("1200x700")
        self.input_csv_path = None
        self.paths = []
        self.i = -1
        self.img = None
        self.img_bin = None
        self.tk1 = self.tk2 = None
        self.last_features = None
        self.last_feature_names = None
        self.csv_dir=None
        self.df_csv = None
        self._suspend_threshold = False
        self.feature_cols = []
        # soglie RGB
        self.rmin, self.rmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.gmin, self.gmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.bmin, self.bmax = tk.IntVar(value=0), tk.IntVar(value=255)

        # parametri LBP
        self.raggio = tk.IntVar(value=3)
        self.punti = tk.IntVar(value=4)

        self._ui()
        self._auto_bind_threshold()
        self.nomi_features_geometriche=["height", "width", "area", "aspect_ratio", "extent", "solidity",
                                     "equivalent_diameter",
                                     "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]
        self.nomi_feature_haralick = [
            "mean_color",
            "variance_color",
            "Angular Second Moment (Energy)",
            "Contrast",
            "Correlation",
            "Variance",
            "Inverse Difference Moment (Homogeneity)",
            "Sum Average",
            "Sum Variance",
            "Sum Entropy",
            "Entropy",
            "Difference Variance",
            "Difference Entropy",
            "Information Measure of Correlation 1",
            "Information Measure of Correlation 2"]
        self.nomi_canali = ["Red", "Green", "Blue"]
        nomi_features_haralick_canali=[]
        for nome in self.nomi_canali:
            nomi_features_haralick_canali = np.concatenate(
                [nomi_features_haralick_canali, [f"{n}_{nome}" for n in self.nomi_feature_haralick]])
        self.nomi_features_haralick_canali=nomi_features_haralick_canali
    def _ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")



        ttk.Button(top, text="Carica CSV", command=self.load_csv).pack(side="left", padx=4)
        ttk.Button(top, text="<< Prev", command=self.prev).pack(side="left", padx=4)
        ttk.Button(top, text="Next >>", command=self.next).pack(side="left", padx=4)
        ttk.Button(top, text="Estrai feature", command=self.extract_features).pack(side="left", padx=12)
        ttk.Button(top, text="Salva feature", command=self.save_features).pack(side="left", padx=4)
        ttk.Button(top, text="Salva feature (tutto)", command=self.save_features_all).pack(side="left", padx=4)
        self.status = ttk.Label(top, text="Nessun CSV.")
        self.status.pack(side="left", padx=12)

        dash = ttk.LabelFrame(self.root, text="Threshold RGB (min/max) + LBP (raggio/punti)", padding=8)
        dash.pack(fill="x", padx=8, pady=6)

        def row(r, name, vmin, vmax):
            ttk.Label(dash, text=name, width=2).grid(row=r, column=0, sticky="w", padx=(0, 8))
            ttk.Label(dash, text="min").grid(row=r, column=1, sticky="e")
            ttk.Entry(dash, textvariable=vmin, width=6).grid(row=r, column=2, padx=(4, 12))
            ttk.Label(dash, text="max").grid(row=r, column=3, sticky="e")
            ttk.Entry(dash, textvariable=vmax, width=6).grid(row=r, column=4, padx=(4, 18))

        row(0, "R", self.rmin, self.rmax)
        row(1, "G", self.gmin, self.gmax)
        row(2, "B", self.bmin, self.bmax)

        ttk.Label(dash, text="Raggio").grid(row=0, column=5, sticky="e")
        ttk.Spinbox(dash, from_=1, to=20, textvariable=self.raggio, width=6).grid(row=0, column=6, padx=(6, 18))

        ttk.Label(dash, text="Punti").grid(row=1, column=5, sticky="e")
        ttk.Spinbox(dash, from_=4, to=64, textvariable=self.punti, width=6).grid(row=1, column=6, padx=(6, 18))

        ttk.Label(dash, text="(Threshold si aggiorna da solo quando cambi i valori)", foreground="#444").grid(
            row=2, column=5, columnspan=2, sticky="w"
        )

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        # immagini
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        f1 = ttk.LabelFrame(left, text="Originale")
        f1.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.p1 = ttk.Label(f1)
        self.p1.pack(fill="both", expand=True, padx=8, pady=8)

        f2 = ttk.LabelFrame(left, text="Threshold (binaria)")
        f2.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.p2 = ttk.Label(f2)
        self.p2.pack(fill="both", expand=True, padx=8, pady=8)

        # output feature
        right = ttk.LabelFrame(body, text="Feature (nomi + valori)", padding=8, width=420)
        right.pack(side="right", fill="both")

        self.txt = tk.Text(right, wrap="none", width=55)
        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.txt.xview)
        yscroll = ttk.Scrollbar(right, orient="vertical", command=self.txt.yview)
        self.txt.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.txt.pack(side="left", fill="both", expand=True)

        self.txt.configure(font=("Consolas", 10))

    def _auto_bind_threshold(self):
        # applica automaticamente il threshold quando cambiano i valori
        for v in (self.rmin, self.rmax, self.gmin, self.gmax, self.bmin, self.bmax):
            v.trace_add("write", lambda *_: self.apply_threshold_safe())

    def _read_input_csv_as_df(self):
        # legge il CSV di input e garantisce una colonna "path"
        df = pd.read_csv(self.input_csv_path, encoding="utf-8-sig")

        # Caso 1: esiste già la colonna path
        if "path" in df.columns:
            df["path"] = df["path"].astype(str).str.strip()
            return df

        # Caso 2: CSV creato con una sola colonna senza header
        # oppure header diverso: assumo che la prima colonna sia il path
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "path"})
        df["path"] = df["path"].astype(str).str.strip()
        return df
    def load_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        self.csv_dir = os.path.dirname(fp)
        self.input_csv_path = fp

        if not fp:
            return

        try:
            with open(fp, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row and row[0].strip()]

            if not rows:
                messagebox.showwarning("Vuoto", "CSV vuoto o senza righe valide.")
                return

            # Se è un CSV salvato da DataFrame, la prima riga è l'header: "path"
            first_cell = rows[0][0].strip().lower()
            start_idx = 1 if first_cell in ("path",) else 0

            self.paths = [row[0].strip() for row in rows[start_idx:] if row and row[0].strip()]

        except Exception as e:
            messagebox.showerror("Errore", f"CSV non leggibile:\n{e}")
            return

        if not self.paths:
            messagebox.showwarning("Vuoto", "Nessun percorso trovato (prima colonna).")
            return


        try:
            self.df_csv = self._read_input_csv_as_df()  # funzione che già avevi (o la aggiungi sotto)
            self.feature_cols = [c for c in self.df_csv.columns if c != "path"]
        except Exception:
            self.df_csv = None
            self.feature_cols = []
        self.i = 0
        self.load_image()
    def prev(self):
        if not self.paths:
            return
        self.i = max(0, self.i - 1)
        self.load_image()

    def next(self):
        if not self.paths:
            return
        self.i = min(len(self.paths) - 1, self.i + 1)
        self.load_image()

    def load_image(self):
        path = self.paths[self.i]
        try:
            self.img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire:\n{path}\n\n{e}")
            self.img = None
            self.img_bin = None
            return

        self.status.config(text=f"{self.i+1}/{len(self.paths)} | {path}")
        self._show(self.img, self.p1, which=1)
        loaded_thr = self.load_threshold_from_csv_for_current_image()
        if not loaded_thr:
            self.apply_threshold_safe()
        # se nel CSV ci sono già feature, le mostro subito
        loaded = self.load_features_from_csv_for_current_image()
        if not loaded:
            self.txt.delete("1.0", tk.END)
            self.last_features = None
            self.last_feature_names = None

    def _show(self, img, label, which):
        # dimensione fissa (grande) per la preview
        W, H = 520, 520

        iw, ih = img.size
        s = min(W / iw, H / ih)  # qui permetto anche upscaling
        img2 = img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.Resampling.NEAREST)

        tkimg = ImageTk.PhotoImage(img2)
        label.configure(image=tkimg)
        if which == 1:
            self.tk1 = tkimg
        else:
            self.tk2 = tkimg

    def load_threshold_from_csv_for_current_image(self):
        if self.df_csv is None or not self.paths or self.i < 0:
            return False

        img_path = self.paths[self.i]
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False

        row = match.iloc[0]
        needed = ["thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax"]

        # colonne presenti?
        if not all(c in self.df_csv.columns for c in needed):
            return False

        # valori presenti?
        if any(pd.isna(row[c]) for c in needed):
            return False

        try:
            self._suspend_threshold = True  # evita trigger durante i set
            self.rmin.set(int(row["thr_rmin"]))
            self.rmax.set(int(row["thr_rmax"]))
            self.gmin.set(int(row["thr_gmin"]))
            self.gmax.set(int(row["thr_gmax"]))
            self.bmin.set(int(row["thr_bmin"]))
            self.bmax.set(int(row["thr_bmax"]))

            # opzionali
            if "lbp_raggio" in self.df_csv.columns and not pd.isna(row.get("lbp_raggio", None)):
                self.raggio.set(int(row["lbp_raggio"]))
            if "lbp_punti" in self.df_csv.columns and not pd.isna(row.get("lbp_punti", None)):
                self.punti.set(int(row["lbp_punti"]))

        finally:
            self._suspend_threshold = False

        self.apply_threshold_safe()  # applica UNA volta sola
        return True
    def apply_threshold_safe(self):
        if self._suspend_threshold:
            return
        if self.img is None:
            return
        try:
            self.img_bin = threshold(
                self.img,
                int(self.rmin.get()), int(self.rmax.get()),
                int(self.gmin.get()), int(self.gmax.get()),
                int(self.bmin.get()), int(self.bmax.get())
            )
            self._show(self.img_bin.convert("RGB"), self.p2, which=2)  # solo per visualizzazione
        except Exception:
            # durante digitazione può esserci input non valido: non spammare popup
            self.img_bin = None
            self.p2.configure(image="")
            self.tk2 = None

    def extract_features(self):
        if self.img is None:
            return
        if self.img_bin is None:
            messagebox.showwarning("Threshold", "Threshold non valido: controlla min/max (0..255 e min<=max).")
            return

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
            messagebox.showerror("Errore feature", str(e))
            return

        self.last_features = features
        self.txt.delete("1.0", tk.END)
        self.last_feature_names=nomi
        # robusto: in caso di mismatch lunghezze
        n = min(len(features), len(nomi))

        for k in range(n):
            name = str(nomi[k])
            val = features[k]
            self.txt.insert(tk.END, f"\t{name}=\t{val}\n")

        # se qualcosa avanza (non dovrebbe), lo segnalo comunque
        if len(features) != len(nomi):
            self.txt.insert(tk.END, f"\n[WARN] len(features)={len(features)} diverso da len(nomi)={len(nomi)}\n")

    def save_features(self):
        if self.last_features is None or self.last_feature_names is None:
            messagebox.showwarning("Salva", "Prima estrai le feature (pulsante 'Estrai feature').")
            return
        if not self.paths or self.i < 0:
            messagebox.showwarning("Salva", "Nessuna immagine selezionata.")
            return
        if not self.input_csv_path:
            messagebox.showwarning("Salva", "Prima carica un CSV di input.")
            return

        img_path = self.paths[self.i]

        # costruisco dizionario feature
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

            # creo df riga feature
            feat_df = pd.DataFrame([row_dict])

            # merge: aggiunge/aggiorna colonne feature per quella riga (path)
            df = df.merge(feat_df, on="path", how="left", suffixes=("", "_new"))

            # se esistono colonne duplicate con _new, prendo quelle nuove e sovrascrivo
            for c in feat_df.columns:
                if c == "path":
                    continue
                newc = f"{c}_new"
                if newc in df.columns:
                    df[c] = df[newc]
                    df.drop(columns=[newc], inplace=True)

            df.to_csv(self.input_csv_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Salvato", f"Feature salvate nel CSV di input:\n{self.input_csv_path}")

        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare nel CSV di input:\n{e}")
        self.df_csv = df
        self.feature_cols = [c for c in self.df_csv.columns if c != "path"]
    def save_features_all(self):
        if not self.paths:
            messagebox.showwarning("Dataset", "Prima carica un CSV con i path.")
            return
        if not self.input_csv_path:
            messagebox.showwarning("Dataset", "Prima carica un CSV di input.")
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
            messagebox.showerror("Errore", f"Nessuna feature salvata. Errori su {len(errors)} immagini.")
            return

        try:
            df = self._read_input_csv_as_df()
            feat_df = pd.DataFrame(rows)

            # merge sulle path
            df = df.merge(feat_df, on="path", how="left", suffixes=("", "_new"))

            # sovrascrive eventuali colonne duplicate
            for c in feat_df.columns:
                if c == "path":
                    continue
                newc = f"{c}_new"
                if newc in df.columns:
                    df[c] = df[newc]
                    df.drop(columns=[newc], inplace=True)

            df.to_csv(self.input_csv_path, index=False, encoding="utf-8-sig")

        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare nel CSV di input:\n{e}")
            return

        if errors:
            messagebox.showwarning(
                "Completato con errori",
                f"Salvato nel CSV di input:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini\nErrori: {len(errors)} immagini\n"
                f"Esempio 1° errore:\n{errors[0][0]}\n{errors[0][1]}"
            )
        else:
            messagebox.showinfo("Completato",
                                f"Salvato nel CSV di input:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini")
        self.status.config(text="Pronto.")
        self.df_csv = df
        thr_cols = {"thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax", "lbp_raggio", "lbp_punti"}
        self.feature_cols = [c for c in self.df_csv.columns if c != "path" and c not in thr_cols]
    def load_features_from_csv_for_current_image(self):
        """Se nel CSV esistono già colonne feature, carica e mostra i valori per l'immagine corrente."""
        if self.df_csv is None or not self.feature_cols or not self.paths or self.i < 0:
            return False

        img_path = self.paths[self.i]

        # trova riga del path
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False

        row = match.iloc[0]

        # prendo solo colonne feature che non sono NaN
        names = []
        vals = []
        for c in self.feature_cols:
            v = row.get(c, None)
            # considera "mancante" NaN o stringa vuota
            if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
                continue
            names.append(str(c))
            vals.append(v)

        if not names:
            return False

        # aggiorna stato interno e textbox
        self.last_feature_names = names
        self.last_features = vals

        self.txt.delete("1.0", tk.END)
        for n, v in zip(names, vals):
            self.txt.insert(tk.END, f"\t{n}=\t{v}\n")
        return True


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()