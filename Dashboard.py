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

        self.input_csv_path: str = None # percorso della directory
        self.paths = []         # vettore dei percorsi di tutte le immagini
        self.i = -1        # variabile di controllo per la navigazione dei file dentro l'interfaccia
        self.img = None         #variabile globale dove allocare l'immagine
        self.img_bin = None        #variabile globale dove allocare la maschera dell'immagine
        self.tk1 = self.tk2 = None         #riferimento a ImageTk.PhotoImage per la preview originale e per la preview della maschera binaria
        self.last_features = None         # lista dei valori feature estratti per l'immagine corrente
        self.last_feature_names = None     # lista dei nomi feature
        self.csv_dir = None  # directory del CSV selezionato
        self.df_csv = None  # DataFrame pandas del CSV caricato
        self._suspend_threshold = False # flag per il thersholde (True mentre setti IntVar da codice )
        self.feature_cols = [] # lista colonne "feature" presenti in df_csv
        self.pca_model = None # oggetto PCA fittato
        self.pca_scores = None  #DataFrame dei punteggi PCA per le righe valide
        self.pca_feature_cols = []  # colonne feature effettivamente usate per il fit PCA
        self.kmeans_k = tk.IntVar(value=3)  #numero cluster(k)  scelto dall'utente per KMeans
        self.kmeans_labels = None # array labels  assegnati da KMeans
        # soglie RGB
        self.rmin, self.rmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.gmin, self.gmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.bmin, self.bmax = tk.IntVar(value=0), tk.IntVar(value=255)
        self.use_pca = tk.BooleanVar( value=True)  # flag UI: se True calcola PCA con n componenti scelte se no lo calcola con 2
        self.pca_n = tk.IntVar(value=2)  # numero di componenti PCA richieste
        # Parametri LBP (Local Binary Patterns)
        self.raggio = tk.IntVar(value=3)  # raggio LBP (in pixel): distanza dei punti dal pixel centrale
        self.punti = tk.IntVar(value=4)  # numero di punti/samples LBP sul cerchio
        self._ui()  # costruisce e imposta tutta l'interfaccia grafica
        self._auto_bind_threshold()  # collega i tk.IntVar delle soglie alle callback
        self.nomi_features_geometriche=["height", "width", "area", "aspect_ratio", "extent", "solidity",
                                     "equivalent_diameter",
                                     "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"] # nomi delle features geometriche
        self.nomi_feature_haralick = [ # nomi delle feature testurali
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
        self.nomi_canali = ["Red", "Green", "Blue"] # nomi dei canali
        # creazione del vettore nomi_features_haralick_canali
        nomi_features_haralick_canali=[]
        for nome in self.nomi_canali:
            nomi_features_haralick_canali = np.concatenate(
                [nomi_features_haralick_canali, [f"{n}_{nome}" for n in self.nomi_feature_haralick]])
        self.nomi_features_haralick_canali=nomi_features_haralick_canali


    def _ui(self):
        """ Costruisce la UI principale (toolbar, controlli threshold/LBP, preview immagini, textbox feature).
        """
        #  comandi principali + navigazione + stato
        # --- TOP BAR: sinistra | centro | destra (stabile)
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        # 3 colonne: la colonna 1 (centro) si prende lo spazio extra
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=0)

        left_bar = ttk.Frame(top)
        left_bar.grid(row=0, column=0, sticky="w")

        center_bar = ttk.Frame(top)
        center_bar.grid(row=0, column=1)  # niente sticky: resta centrato nella cella

        right_bar = ttk.Frame(top)
        right_bar.grid(row=0, column=2, sticky="e")

        # --- Sinistra: solo Carica CSV
        ttk.Button(left_bar, text="Carica CSV", command=self.load_csv).pack(side="left", padx=4, pady=(0, 4))

        # --- Centro: 3 bottoni (centrati)
        ttk.Button(center_bar, text="Estrai feature", command=self.extract_features).pack(side="left", padx=4,pady=(0, 4)) # estrae le feature della immagine corrente
        ttk.Button(center_bar, text="Salva feature", command=self.save_features).pack(side="left", padx=4, pady=(0, 4))  # salva in locale le feature della immagine corrente
        ttk.Button(center_bar, text="Salva feature dataset", command=self.save_features_all).pack(side="left", padx=4,pady=(0, 4))  # salva nel csv le feature di tutte le immagini

        # Destra: Sottofinestra
        ttk.Button(right_bar, text="visualizza PCA", command=self.open_subwindow).pack(side="left", padx=4, pady=(0, 4))

        # Riga sotto: Prev / Next a sinistra, Status centrato
        nav = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        nav.pack(fill="x")

        nav.grid_columnconfigure(0, weight=0)
        nav.grid_columnconfigure(1, weight=1)
        nav.grid_columnconfigure(2, weight=0)
        # Frame che contiene i pulsanti Prev/Next
        nav_left = ttk.Frame(nav)
        nav_left.grid(row=0, column=0, sticky="w")
        # Frame “centrale” dove metti lo status
        nav_center = ttk.Frame(nav)
        nav_center.grid(row=0, column=1)  # centrato

        # Pulsanti di navigazione
        ttk.Button(nav_left, text="<< Prev", command=self.prev).pack(side="left", padx=4)
        ttk.Button(nav_left, text="Next >>", command=self.next).pack(side="left", padx=4)

        self.status = ttk.Label(nav_center, text="Nessun CSV.")
        self.status.pack()

        # commenta dopo
        #top.grid_columnconfigure(6, weight=1)

        #Pannello controlli per le variabili di thresolde e dei Lineary Binary Pattern
        dash = ttk.LabelFrame(self.root, text="Threshold RGB (min/max) + LBP (raggio/punti)", padding=8)
        dash.pack(fill="x", padx=8, pady=6)

        # Helper per creare una riga di controlli min/max
        def row(r, name, vmin, vmax):
            ttk.Label(dash, text=name, width=2).grid(row=r, column=0, sticky="w", padx=(0, 8))
            ttk.Label(dash, text="min").grid(row=r, column=1, sticky="e")
            ttk.Entry(dash, textvariable=vmin, width=6).grid(row=r, column=2, padx=(4, 12))
            ttk.Label(dash, text="max").grid(row=r, column=3, sticky="e")
            ttk.Entry(dash, textvariable=vmax, width=6).grid(row=r, column=4, padx=(4, 18))

        # Soglie RGB: valori letti da self.rmin/self.rmax ecc.
        row(0, "R", self.rmin, self.rmax)
        row(1, "G", self.gmin, self.gmax)
        row(2, "B", self.bmin, self.bmax)

        # Parametri LBP: Spinbox legate a self.raggio/self.punti
        ttk.Label(dash, text="Raggio").grid(row=0, column=5, sticky="e")
        ttk.Spinbox(dash, from_=1, to=20, textvariable=self.raggio, width=6).grid(row=0, column=6, padx=(6, 18))
        ttk.Label(dash, text="Punti").grid(row=1, column=5, sticky="e")
        ttk.Spinbox(dash, from_=4, to=64, textvariable=self.punti, width=6).grid(row=1, column=6, padx=(6, 18))
        # Corpo principale della visualizzazione delle finestre, sinistra preview immagini, destra output feature
        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        # Sezione sinistra: due pannelli immagine affiancati
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Preview originale
        f1 = ttk.LabelFrame(left, text="Originale")
        f1.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.p1 = ttk.Label(f1)  # label che ospita PhotoImage
        self.p1.pack(fill="both", expand=True, padx=8, pady=8)

        # Preview threshold (binaria)
        f2 = ttk.LabelFrame(left, text="Threshold (binaria)")
        f2.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.p2 = ttk.Label(f2)
        self.p2.pack(fill="both", expand=True, padx=8, pady=8)

        # Sezione destra: textbox con feature (nomi + valori) e scrollbar
        right = ttk.LabelFrame(body, text="Feature (nomi + valori)", padding=8, width=420)
        right.pack(side="right", fill="both")

        self.txt = tk.Text(right, wrap="none", width=55)  # wrap none: mantenere tab/colonne allineate
        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.txt.xview)
        yscroll = ttk.Scrollbar(right, orient="vertical", command=self.txt.yview)
        self.txt.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        # Pack scroll + text (scrollbar vertical a destra, orizzontale sotto)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.txt.pack(side="left", fill="both", expand=True)

        # Font monospazio per allineare meglio "name = value"
        self.txt.configure(font=("Consolas", 10))
    def _pca_window_build(self, title, with_kmeans):
        #Creo una finestra "figlia" dedicata alla PCA
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("950x700")
        win.transient(self.root)
        win.grab_set()
        #frame dei controlli sopra + frame del grafico sotto
        ctrl = ttk.Frame(win, padding=10)
        ctrl.pack(fill="x")
        plot_frm = ttk.Frame(win, padding=10)
        plot_frm.pack(fill="both", expand=True)
        #Variabili Tk che memorizzano la selezione dell’asse X/Y (PC1, PC2, ...)
        self._pca_x_var = tk.StringVar(value="PC1")
        self._pca_y_var = tk.StringVar(value="PC2")

        # Controlli asse X/Y: due combobox "readonly"
        ttk.Label(ctrl, text="Asse X:").grid(row=0, column=0, sticky="e", padx=(0, 4))
        self._pca_x_cb = ttk.Combobox(ctrl, textvariable=self._pca_x_var, values=["PC1", "PC2"], width=6, state="readonly")
        self._pca_x_cb.grid(row=0, column=1, sticky="w", padx=(0, 14))

        ttk.Label(ctrl, text="Asse Y:").grid(row=0, column=2, sticky="e", padx=(0, 4))
        self._pca_y_cb = ttk.Combobox(ctrl, textvariable=self._pca_y_var, values=["PC1", "PC2"], width=6, state="readonly")
        self._pca_y_cb.grid(row=0, column=3, sticky="w", padx=(0, 14))

        # 5) Controllo "n componenti": spinbox mostra a self.pca_n
        ttk.Label(ctrl, text="numero componenti PCA:").grid(row=0, column=4, sticky="e", padx=(0, 4))
        self._pca_spn = ttk.Spinbox(ctrl, from_=2, to=999, textvariable=self.pca_n, width=6)
        self._pca_spn.grid(row=0, column=5, sticky="w", padx=(0, 14))

        #controllo k_means
        ttk.Label(ctrl, text="k cluster:").grid(row=0, column=6, sticky="e", padx=(0, 4))
        ttk.Spinbox(ctrl, from_=2, to=50, textvariable=self.kmeans_k, width=6).grid(row=0, column=7, sticky="w", padx=(0, 14))

        # Creo figura/assi matplotlib + canvas Tkinter che ospita il grafico
        self._pca_fig = plt.Figure()
        self._pca_ax = self._pca_fig.add_subplot(111)
        self._pca_canvas = FigureCanvasTkAgg(self._pca_fig, master=plot_frm)
        self._pca_canvas.get_tk_widget().pack(fill="both", expand=True)
        #quando cambi asse X o asse Y, richiamo _pca_redraw()
        self._pca_x_cb.bind("<<ComboboxSelected>>", lambda e: self._pca_redraw())
        self._pca_y_cb.bind("<<ComboboxSelected>>", lambda e: self._pca_redraw())
        # Abilito/disabilito la spinbox delle componenti PCA quando cambia self.use_pca
        def _sync_state(*_):
            self._pca_spn.configure(state=("normal" if self.use_pca.get() else "disabled"))
        self.use_pca.trace_add("write", _sync_state)
        _sync_state()  # sincronizzo subito lo stato iniziale
        # Barra pulsanti: Chiudi + Calcola/Refresh e k_means
        btns = ttk.Frame(ctrl)
        btns.grid(row=1, column=0, columnspan=8, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Chiudi", command=win.destroy).pack(side="right", padx=(0, 8))
        ttk.Button(btns, text="Calcola / Refresh", command=self._pca_compute_refresh).pack(side="right")
        ttk.Button(btns, text="K-Means", command=self._pca_run_kmeans).pack(side="left", padx=(0, 8))
        # Porta la finestra davanti e le dà il focus
        win.lift()
        win.focus_force()
        return win

    def _pca_refresh_pc_dropdowns(self):

        # Aggiorna le combobox PC in base alle colonne disponibili in self.pca_scores.

        # Se non ho ancora calcolato la PCA (o il dataframe punteggi è vuoto) non posso aggiornare nulla
        if self.pca_scores is None or self.pca_scores.empty:
            return
        # Ricavo la lista delle componenti disponibili: ["PC1", "PC2", ..., "PCn"]
        pcs = list(self.pca_scores.columns)
        # Aggiorno i valori mostrati nelle combobox X e Y con la lista reale delle PC calcolate
        self._pca_x_cb.configure(values=pcs)
        self._pca_y_cb.configure(values=pcs)
        # Se la selezione attuale non è più valida (es. prima avevo PC5 e ora ho solo PC1..PC3),
        # la resetto a un valore valido.
        if self._pca_x_var.get() not in pcs:
            self._pca_x_var.set(pcs[0])  # default: prima PC disponibile
        if self._pca_y_var.get() not in pcs:
            # default: seconda PC se esiste, altrimenti uguale alla prima
            self._pca_y_var.set(pcs[1] if len(pcs) > 1 else pcs[0])

    def _pca_redraw(self):
        """
        Aggiorna il grafico PCA in base alle componenti scelte.
        Se presenti, usa anche i cluster KMeans per colorare i punti.
        """
        # Calcola PCA internamente dal DataFrame (pulizia colonne, standardizzazione, fit PCA, scores).
        self._pca_refresh_pc_dropdowns()
        # Leggo quali colonne l’utente ha scelto per l’asse X e Y
        xcol, ycol = self._pca_x_var.get(), self._pca_y_var.get()
        # Estraggo i valori  da plottare
        xs = self.pca_scores[xcol].values
        ys = self.pca_scores[ycol].values
        # Recupero l'asse matplotlib e lo pulisco
        ax = self._pca_ax
        ax.clear()
        # Se ho già eseguito KMeans e ho labels coerenti con il numero di punti,
        # coloro i punti per cluster (c=labels). Altrimenti scatter standard.
        if self.kmeans_labels is not None and len(self.kmeans_labels) == len(self.pca_scores):
            ax.scatter(xs, ys, c=self.kmeans_labels)
            ax.set_title(f"{xcol} vs {ycol} (KMeans k={int(self.kmeans_k.get())})")
        else:
            ax.scatter(xs, ys)
            ax.set_title(f"{xcol} vs {ycol}")

        # Etichette assi
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        # Aggiorno il canvas Tkinter senza bloccare (draw_idle è più “gentile” di draw())
        self._pca_canvas.draw_idle()

    def _pca_compute_inline(self):
        """
        Calcola la PCA partendo dal DataFrame caricato e salva i risultati.
        Controlla i dati, tiene solo le colonne numeriche valide e standardizza le feature.
        """
        # Controllo : devo aver caricato un CSV in self.df_csv
        if self.df_csv is None:
            return False, "Carica prima un CSV."

        # Colonne da ESCLUDERE dalla PCA
        exclude = {"path", "lbp_raggio", "lbp_punti",
                   "thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax"}

        # Prendo le colonne candidate = tutte tranne quelle escluse
        cols = [c for c in self.df_csv.columns if c not in exclude]

        #  converto a numerico
        #    Poi elimino le colonne che risultano completamente NaN
        tmp = self.df_csv[cols].apply(pd.to_numeric, errors="coerce")
        cols = [c for c in cols if not tmp[c].isna().all()]
        if not cols:
            return False, "Nessuna colonna numerica valida per PCA."

        # Creo la matrice feature X
        X = self.df_csv[cols].apply(pd.to_numeric, errors="coerce")
        # Tengo SOLO le righe complete (senza NaN) per evitare errori e avere PCA consistente
        valid = ~X.isna().any(axis=1)  # maschera booleana sulle righe
        Xv = X.loc[valid].values  # matrice finale (n_righe_valide, n_feature)
        if Xv.shape[0] < 2:
            return False, "Poche righe valide (NaN) per PCA."
        # Scelgo n componenti:
        #    - se use_pca=True uso il valore richiesto dall'utente
        #    - se use_pca=False forzo 2
        ncomp = int(self.pca_n.get()) if self.use_pca.get() else 2
        ncomp = max(2, min(ncomp, Xv.shape[1]))

        # Standardizzo le feature (fondamentale per PCA):

        self.pca_scaler = StandardScaler().fit(Xv)
        Xs = self.pca_scaler.transform(Xv)

        # Fit della PCA + trasformazione in scores
        #    scores ha shape (n_righe_valide, ncomp)
        self.pca_model = PCA(n_components=ncomp).fit(Xs)
        scores = self.pca_model.transform(Xs)

        #  Metto i punteggi in un DataFrame con colonne PC1..PCn,
        self.pca_scores = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, ncomp + 1)])

        #  Salvo metadati utili per debug/riuso:
        self.pca_feature_cols = cols
        self._pca_valid_mask = valid
        self._pca_evr = self.pca_model.explained_variance_ratio_
        return True, ""

    def _pca_compute_from_helper(self):
        """
        Calcola la PCA usando una funzione esterna e salva i risultati nell'app.
        Se qualcosa non va, restituisce un messaggio di errore.
        """
        #  Controllo: devo aver caricato il CSV in self.df_csv
        if self.df_csv is None:
            return False, "Carica prima un CSV."

        #    ritorna:
        #    ok/msg: esito e messaggio
        #    pca: oggetto PCA fittato
        #    scaler: oggetto StandardScaler fittato (
        #    scores_df: DataFrame con i punteggi PCA
        #    cols: lista delle colonne feature effettivamente usate
        #    valid_mask: maschera righe valide usate per il fit
        #    evr: explained_variance_ratio_

        ok, msg, pca, scaler, scores_df, cols, valid_mask, evr = compute_pca_on_df_vars(self.df_csv,use_pca=bool(self.use_pca.get()), n_components=int(self.pca_n.get()),)
        # 3) Se l'helper segnala errore
        #    propago l'errore al chiamante.
        if not ok:
            return False, msg
        # Se tutto OK, salvo i risultati in self, così:
        #    - _pca_redraw() può plottare self.pca_scores
        #    - KMeans può lavorare su self.pca_scores.values
        #    - in futuro puoi riusare scaler/pca per trasformare nuovi dati coerentemente
        self.pca_model = pca
        self.pca_scaler = scaler
        self.pca_scores = scores_df
        self.pca_feature_cols = cols
        self._pca_valid_mask = valid_mask
        self._pca_evr = evr

        # 5) Ritorno OK
        return True, ""

    def _pca_compute_refresh(self):
        """
            Ricalcola la PCA, azzera i cluster precedenti e aggiorna il grafico.
            Se il calcolo fallisce, mostra un avviso e si ferma.
            """
        #Scelgo quale "pipeline" PCA usare in base a come è stata aperta la finestra.
        #    - _pca_mode == "helper"  -> uso la funzione esterna compute_pca_on_df_vars (wrappa tutto)
        #    - altrimenti             -> uso la versione inline (calcolo PCA qui dentro la classe)
        if getattr(self, "_pca_mode", None) == "helper":
            ok, msg = self._pca_compute_from_helper()
        else:
            ok, msg = self._pca_compute_inline()
        # Se il calcolo fallisce  mostro un warning e interrompo: niente redraw, niente stato aggiornato.
        if not ok:
            messagebox.showwarning("PCA", msg)
            return
        #Quando ricalcolo la PCA, le vecchie etichette di KMeans non sono più valide
        self.kmeans_labels = None

        # Ridisegno lo scatter usando i nuovi punteggi PCA (self.pca_scores)
        self._pca_redraw()

    def _pca_run_kmeans(self):
        """
        Raggruppa i punti della PCA in cluster con KMeans.
        In caso di errore mostra un messaggio, altrimenti ridisegna il plot.
        """
        # Eseguo KMeans nello spazio PCA
        #    - k: numero cluster scelto dall’utente (spinbox)
        #    - n_init: quante inizializzazioni diverse provare (più alto = più robusto)
        #    - random_state: ripetibilità dei risultati
        ok, msg, labels, km, _ = run_kmeans_vars(self.pca_scores.values,k=int(self.kmeans_k.get()), n_init=10,random_state=0)
        #  Se il clustering fallisce (es. k > n punti, dati non validi, ecc. mostro il motivo e stop.
        if not ok:
            messagebox.showwarning("K-Means", msg)
            return

        # Se tutto OK, salvo:
        #    - labels: per colorare i punti nello scatter
        #    - km: modello KMeans (utile se vuoi inertia, centroidi, ecc.)
        self.kmeans_labels = labels
        self.kmeans_model = km
        # Ridisegno lo scatter questa volta con colori per cluster.
        self._pca_redraw()

    def open_subwindow(self):
        """
        Apre la finestra della PCA, prepara i controlli e mostra il primo grafico.
        """
        self._pca_mode = "inline"
        # Costruisco la finestra con controlli KMeans attivi
        self._pca_window_build("PCA + Plot", with_kmeans=True)
        # Calcolo subito PCA e disegno lo scatter iniziale.
        self._pca_compute_refresh()

    def open_pca_plot_window(self):
        # Questa finestra usa la PCA via helper esterno compute_pca_on_df_vars
        self._pca_mode = "helper"
        #  Costruisco la finestra senza KMeans
        self._pca_window_build("PCA scatter", with_kmeans=False)
        #  Calcolo subito PCA e disegno lo scatter iniziale.
        self._pca_compute_refresh()
    def _auto_bind_threshold(self):
        # applica automaticamente il threshold quando cambiano i valori
        for v in (self.rmin, self.rmax, self.gmin, self.gmax, self.bmin, self.bmax):
            v.trace_add("write", lambda *_: self.apply_threshold_safe())
    def _read_input_csv_as_df(self):
        # legge il CSV di input
        df = pd.read_csv(self.input_csv_path, encoding="utf-8-sig")
        # Caso 1: esiste già la colonna path
        if "path" in df.columns:
            df["path"] = df["path"].astype(str).str.strip()
            return df
        # Caso 2: CSV creato con una sola colonna senza header
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "path"})
        df["path"] = df["path"].astype(str).str.strip()
        return df

    def load_csv(self):
        # seleziona CSV
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fp:
            return  # utente ha annullato
        # Salvo percorso e directory del CSV
        self.input_csv_path = fp
        self.csv_dir = os.path.dirname(fp)
        #  Provo a leggere il CSV
        try:
            # Questa funzione deve:
            # - leggere il CSV
            # - garantire che esista una colonna "path"
            # - fare strip sugli spazi
            self.df_csv = self._read_input_csv_as_df()

        except Exception as e:
            messagebox.showerror("Errore", f"CSV non leggibile:\n{e}")
            self.df_csv = None
            self.feature_cols = []
            self.paths = []
            return

        #  Estraggo i path dalla colonna "path"
        if "path" not in self.df_csv.columns:
            messagebox.showwarning("Vuoto", "Il CSV non contiene una colonna 'path'.")
            self.paths = []
            self.feature_cols = []
            return
        self.paths = (self.df_csv["path"].astype(str).str.strip().replace("", np.nan).dropna().tolist())
        # Se non ho path validi, stop
        if not self.paths:
            messagebox.showwarning("Vuoto", "Nessun percorso trovato nella colonna 'path'.")
            self.feature_cols = []
            return
        #  Elenco colonne feature disponibili
        self.feature_cols = [c for c in self.df_csv.columns if c != "path"]
        # Mi posiziono sulla prima immagine e carico
        self.i = 0
        self.load_image()

    def prev(self):
        # Se non ho path caricati, non posso navigare
        if not self.paths:
            return
        # Decremento l’indice
        self.i = max(0, self.i - 1)
        # Carico l’immagine corrispondente al nuovo indice
        self.load_image()

    def next(self):
        # Se non ho path caricati, non posso navigare
        if not self.paths:
            return
        # Incremento l’indice
        self.i = min(len(self.paths) - 1, self.i + 1)
        # Riarico
        self.load_image()

    def load_image(self):
        "carica l'immagine da elaborare"
        if not self.paths:
            return
        if self.i < 0 or self.i >= len(self.paths):
            self.i = 0  # fallback sicuro

        # Path corrente da aprire
        path = self.paths[self.i]

        #  Provo ad aprire l’immagine e convertirla in RGB
        try:
            self.img = Image.open(path).convert("RGB")
        except Exception as e:
            # Se fallisce: avviso, pulisco lo stato, e non proseguo
            messagebox.showerror("Errore", f"Impossibile aprire:\n{path}\n\n{e}")
            self.img = None
            self.img_bin = None

            # aggiorno status e pulisco preview per non lasciare la vecchia immagine
            self.status.config(text=f"{self.i + 1}/{len(self.paths)} | ERRORE: {path}")
            self.p1.configure(image="")
            self.p2.configure(image="")
            self.tk1 = None
            self.tk2 = None
            return

        # Aggiorno status “x/y | path”
        self.status.config(text=f"{self.i + 1}/{len(self.paths)} | {path}")

        #  Mostro preview immagine originale
        self._show(self.img, self.p1, which=1)

        # Provo a caricare dal CSV le soglie associate a questa immagine.
        #    Se ci sono le soglie (thr_*) in quella riga: le setto e applico il threshold una sola volta.
        loaded_thr = self.load_threshold_from_csv_for_current_image()

        # Se non ho trovato soglie nel CSV, applico comunque il threshold con i valori correnti in UI
        if not loaded_thr:
            self.apply_threshold_safe()

        # Se nel CSV ci sono già colonne feature per questa immagine, le mostro nella textbox.
        loaded = self.load_features_from_csv_for_current_image()

        # Se non ci sono feature salvate, pulisco la textbox e azzero lo stato interno
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
        """
        Carica dal CSV le soglie associate all’immagine corrente.
        Se trova i valori giusti, aggiorna l’interfaccia e applica il threshold.
        """
        # controllo: devo avere un DataFrame caricato, una lista di path e un indice valido
        if self.df_csv is None or not self.paths:
            return False
        if self.i < 0 or self.i >= len(self.paths):
            return False
        # controllo: Devo avere la colonna "path" nel CSV per poter fare match
        if "path" not in self.df_csv.columns:
            return False
        #  Prendo il path corrente e cerco la riga corrispondente nel DataFrame
        img_path = self.paths[self.i]
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False
        #  Prendo la prima riga che matcha
        row = match.iloc[0]
        # Colonne soglie necessarie per poter ripristinare il threshold
        needed = ["thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax"]
        #  Se manca anche solo una colonna thr_* nel CSV, non posso caricare nulla
        if not all(c in self.df_csv.columns for c in needed):
            return False

        #  Setto le IntVar della UI usando un flag di sospensione:
        try:
            self._suspend_threshold = True

            self.rmin.set(int(row["thr_rmin"]))
            self.rmax.set(int(row["thr_rmax"]))
            self.gmin.set(int(row["thr_gmin"]))
            self.gmax.set(int(row["thr_gmax"]))
            self.bmin.set(int(row["thr_bmin"]))
            self.bmax.set(int(row["thr_bmax"]))

            # Parametri  LBP: li setto solo se esistono
            if "lbp_raggio" in self.df_csv.columns and not pd.isna(row.get("lbp_raggio", None)):
                self.raggio.set(int(row["lbp_raggio"]))
            if "lbp_punti" in self.df_csv.columns and not pd.isna(row.get("lbp_punti", None)):
                self.punti.set(int(row["lbp_punti"]))
        finally:
            # In ogni caso  riattivo i trigger
            self._suspend_threshold = False
        # Applico il threshold UNA sola volta con i valori finali impostati
        self.apply_threshold_safe()
        #  Indico che ho effettivamente caricato soglie dal CSV
        return True

    def apply_threshold_safe(self):
        #  Se sto impostando le soglie   evito di ricalcolare il threshold ad ogni .set() delle IntVar.
        if self._suspend_threshold:
            return
        #  Se non ho un’immagine caricata, non posso applicare il threshold
        if self.img is None:
            return
        try:
            # 3) Chiamo la tua funzione threshold usando i valori correnti delle IntVar (R/G/B min/max)
            #    int(...) serve perché le IntVar possono contenere stringhe temporanee durante edit
            self.img_bin = threshold(
                self.img,
                int(self.rmin.get()), int(self.rmax.get()),
                int(self.gmin.get()), int(self.gmax.get()),
                int(self.bmin.get()), int(self.bmax.get())
            )
            #  La maschera binaria la mostro in preview:
            self._show(self.img_bin.convert("RGB"), self.p2, which=2)

        except Exception:
            #    Durante digitazione nelle Entry può capitare input non valido
            self.img_bin = None
            self.p2.configure(image="")  # pulisce la preview
            self.tk2 = None  # libera riferimento PhotoImage (evita immagini “fantasma”)

    def extract_features(self):
        # Se non ho un’immagine caricata, non posso estrarre nulla
        if self.img is None:
            return
        #  Le feature dipendono dalla maschera binaria: se il threshold è fallito/assente, avviso e stop
        if self.img_bin is None:
            messagebox.showwarning(
                "Threshold",
                "Threshold non valido: controlla min/max (0..255 e min<=max)."
            )
            return

        # Provo a calcolare le feature chiamando la tua funzione esterna.
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
            # Se l’estrazione fallisce  mostro l’errore e interrompo.
            messagebox.showerror("Errore feature", str(e))
            return

        # Salvo l’ultimo risultato nello stato dell’app
        self.last_features = features
        self.last_feature_names = nomi
        #  Pulisco la textbox e scrivo "nome = valore" in modo leggibile
        self.txt.delete("1.0", tk.END)
        #  se per qualche motivo le lunghezze non coincidono, stampo solo la parte comune
        n = min(len(features), len(nomi))
        for k in range(n):
            name = str(nomi[k])
            val = features[k]
            self.txt.insert(tk.END, f"\t{name}=\t{val}\n")

        #  Se c’è mismatch, lo segnalo nella textbox come warning diagnostico
        if len(features) != len(nomi):
            self.txt.insert(
                tk.END,
                f"\n[WARN] len(features)={len(features)} diverso da len(nomi)={len(nomi)}\n"
            )
    def save_features(self):
        "salvo le feature nel csv"
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
        "salva le feature di tutte le immagini del csv"
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
            messagebox.showwarning("Completato con errori",f"Salvato nel CSV di input:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini\nErrori: {len(errors)} immagini\n"   f"Esempio 1° errore:\n{errors[0][0]}\n{errors[0][1]}"    )
        else:
            messagebox.showinfo("Completato", f"Salvato nel CSV di input:\n{self.input_csv_path}\n\nOK: {len(rows)} immagini")
        self.status.config(text="Pronto.")
        self.df_csv = df
        thr_cols = {"thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax", "lbp_raggio", "lbp_punti"}
        self.feature_cols = [c for c in self.df_csv.columns if c != "path" and c not in thr_cols]
    def load_features_from_csv_for_current_image(self):
        """Se nel CSV esistono già colonne feature, carica e mostra i valori per l'immagine corrente."""

        #  controllo: devo avere:
        #    - df_csv caricato
        #    - una lista di colonne feature (feature_cols) non vuota
        #    - una lista di path
        #    - un indice i valido
        if self.df_csv is None or not self.feature_cols or not self.paths:
            return False
        if self.i < 0 or self.i >= len(self.paths):
            return False
        # Path dell'immagine corrente
        img_path = self.paths[self.i]

        #  Cerco nel DataFrame la riga che corrisponde a quel path
        if "path" not in self.df_csv.columns:
            return False
        match = self.df_csv[self.df_csv["path"] == img_path]
        if match.empty:
            return False

        row = match.iloc[0]

        #4) Raccolgo solo le feature presenti:
        names = []
        vals = []
        for c in self.feature_cols:
            v = row.get(c, None)
            # considera "mancante" NaN o stringa vuota
            if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
                continue
            names.append(str(c))
            vals.append(v)

        # Se non ho trovato nessuna feature valorizzata, non carico nulla
        if not names:
            return False
        #  Aggiorno lo stato interno
        self.last_feature_names = names
        self.last_features = vals
        #  Mostro le feature nella textbox
        self.txt.delete("1.0", tk.END)
        for n, v in zip(names, vals):
            self.txt.insert(tk.END, f"\t{n}=\t{v}\n")
        return True


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()