from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans


def threshold(
    img: Image.Image,
    r_min: int, r_max: int,
    g_min: int, g_max: int,
    b_min: int, b_max: int
) -> Image.Image:
    """
    RGB threshold with lower and upper limits for each channel.
    White pixel if the pixel is within all 3 ranges, otherwise black.

    Returns: Binary PIL Image mode ‘L’ (0/255).
    """
    # Range validation
    vals = [r_min, r_max, g_min, g_max, b_min, b_max]
    if any((v < 0 or v > 255) for v in vals):
        raise ValueError("Tutti i valori devono essere tra 0 e 255.")

    if r_min > r_max or g_min > g_max or b_min > b_max:
        raise ValueError("Per ogni canale deve valere min <= max.")

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    mask = (
        (r >= r_min) & (r <= r_max) &
        (g >= g_min) & (g <= g_max) &
        (b >= b_min) & (b <= b_max)
    )

    binary = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def estrazzione_features_geometriche(mask: Image.Image) -> np.ndarray:
    """
    Extracts geometric features from a binary mask.

    Returned features:
    [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter, hu1..hu7]
    """
    # PIL -> numpy grayscale uint8
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)

    # Binarization  (0/255)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Area (white pixels)
    area = float(np.count_nonzero(mask_bin))

    # If there is no object, it returns zero
    if area == 0:
        return np.zeros(14, dtype=float)  # 7 base + 7 Hu

    # Bounding box without outlines
    ys, xs = np.where(mask_bin > 0)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    height = float(ymax - ymin + 1)
    width = float(xmax - xmin + 1)

    # Aspect ratio
    aspect_ratio = float(width / height) if height > 0 else 0.0

    # Extent = area / area bounding box
    rect_area = float(width * height)
    extent = float(area / rect_area) if rect_area > 0 else 0.0

    # Solidity and Hu Moments need a side contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(contorno)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Equivalent diameter
    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))

    # Hu moments
    moments = cv2.moments(contorno)
    hu = cv2.HuMoments(moments).flatten().astype(float)
    hu = np.sign(hu) * np.log(np.abs(hu) + 1e-12)
    # Final features
    features_base = np.array(
        [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter],
        dtype=float)
    features = np.concatenate([features_base, hu])
    return features

def estrazione_feature_texturali(img, img_binary, raggio=3, punti=4):
    """
    This function will return a vector containing the textural features
    for each channel of the image.
    :param img: Image
    :param img_binary: Mask of the image
    :param raggio: parameter of lbp
    :param punti: parameter of lbp
    :return: vector of features
    """
    img_binary_np = np.array(img_binary.convert("L"), dtype=np.uint8)
    img = np.array(img, dtype=np.uint8)
    vettore_haralick=[]
    vettore_lib=[]
    # isolate color channel
    for i in range(3):
        img_np = img[:, :, i]
        # select the region of interest
        roi = cv2.bitwise_and(img_np, img_np, mask=img_binary_np)
        # pixel mean and variance
        media_pixel = np.mean(roi.flatten())
        varianza_pixel = np.var(roi.flatten())
        # haralick feature extraction
        features_haralick = mahotas.features.haralick(roi,ignore_zeros=True)
        mean_features_haralick = features_haralick.mean(axis=0)
        # extraction Binary linear pattern
        features_lib = mahotas.features.lbp(roi, raggio, punti, ignore_zeros=True)
        # concatenation of all values into a single vector
        mean_features_haralick = np.r_[varianza_pixel,mean_features_haralick]
        mean_features_haralick=np.r_[media_pixel,mean_features_haralick]
        vettore_haralick=np.concatenate([vettore_haralick,mean_features_haralick])
        vettore_lib=np.concatenate([vettore_lib,features_lib])
    return vettore_haralick,vettore_lib

def estrazioni_feature_e_nomi(img, img_binary,
                             nomi_features_geometriche,
                             nomi_features_haralick_canali,
                             nomi_canali,
                             raggio=3, punti=4):
    """
    This function calculates the complete vector of features
    and returns the names corresponding to each feature in parallel, in order to have
    an interpretable and traceable representation of the columns.

    The textural component consists of:
    - Haralick (with added pixel mean and variance) for each channel
    - LBP for each channel (with names generated dynamically based on length)

       :param img: Image (RGB). Input image from which to extract features.
    :param img_binary: Mask of the image. Binary mask defining the ROI.
    :param geometric_feature_names: list/array of strings containing the names of geometric features.
    :param haralick_feature_names_channels: list/array of strings containing the names of the Haralick features (already expanded by channel).
    :param channel_names: list of strings with the names of the channels.
    :param radius: parameter of lbp. Radius used for the calculation of LBP.
    :param points: parameter of lbp. Number of points used for the calculation of LBP.
    :return:
        - features: 1D ndarray with all features concatenated (geometric + haralick + lbp)
        - names: ndarray/list with names corresponding to features (same order and same length)
    """
    # Extraction of textural features (divided into Haralick and LBP)
    haralick, lib = estrazione_feature_texturali(img, img_binary, raggio=raggio, punti=punti)
    # Concatenation of textural features into a single vector
    feature_testurali = np.concatenate([haralick, lib])
    # Final concatenation: geometric features + textural features
    features = np.concatenate([estrazzione_features_geometriche(img_binary), feature_testurali])
    # Constructing LBP feature names:
    # lib contains the concatenated LBPs from all 3 channels, so the length per channel is len(lib)/3
    nomi_lib = []
    len_lib = len(lib) / 3
    # For each channel, I create the names LBP_1_<channel>, LBP_2_<channel>, ...
    for nome_canale in nomi_canali:
        for i in range(int(len_lib)):
            nomi_lib.append(f"LBP_{i+1}_{nome_canale}")
    # Final concatenation of names: geometric + haralick (already prepared) + lbp (generated here)
    nomi = np.concatenate([nomi_features_geometriche, nomi_features_haralick_canali, nomi_lib])
    return features, nomi

def directory_immagini_to_csv(directory_path: str, recursive: bool = True, csv_name: str = "paths_immagini.csv") -> pd.DataFrame:
    """
    Create a DataFrame with a single column ‘path’ containing the paths of the images
    present in a directory and save it as CSV in the same directory.

    Args:
        directory_path: path of the directory to scan
        recursive: if True, also scans subfolders
        csv_name: name of the CSV file to save in the directory

    Returns:
        df: DataFrame with ‘path’ column
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory non trovata: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Non è una directory: {dir_path}")

    # Common image file extensions (add more if needed)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    pattern = "**/*" if recursive else "*"
    image_paths = [
        str(p.resolve())
        for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in exts
    ]

    df = pd.DataFrame({"path": image_paths})

    out_csv = dir_path / csv_name
    df.to_csv(out_csv, index=False)

    return df
def compute_pca_on_df_vars(
        df: pd.DataFrame,
        *,
        use_pca: bool = True,
        n_components: int = 2,
        exclude: set[str] | None = None,
        require_complete_rows: bool = True,
        min_samples: int = 2,
):
    """
    PCA on DataFrame, without classes: returns variables.


    Returns:
      ok (bool),
      msg (str),
      pca (PCA|None),
      scaler (StandardScaler|None),
      scores_df (pd.DataFrame|None),
      feature_cols (list[str]),
      valid_mask (pd.Series|None),
      explained_variance_ratio (np.ndarray|None)
    """
    if df is None or len(df) == 0:
        return (False, "DataFrame vuoto o None.", None, None, None, [], None, None)

    if exclude is None:
        exclude = {
            "path",
            "lbp_raggio", "lbp_punti",
            "thr_rmin", "thr_rmax", "thr_gmin", "thr_gmax", "thr_bmin", "thr_bmax",
        }

    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        return (False, "Non trovo feature nel DataFrame (solo path/parametri?).", None, None, None, [], None, None)

    X = df[cols].apply(pd.to_numeric, errors="coerce")

    if require_complete_rows:
        valid_mask = ~X.isna().any(axis=1)
    else:
        valid_mask = pd.Series([True] * len(X), index=X.index)

    Xv = X.loc[valid_mask].values
    if Xv.shape[0] < min_samples:
        return (False, "Troppe righe con NaN: non ho abbastanza campioni validi per PCA.",
                None, None, None, cols, valid_mask, None)

    # number of components
    ncomp = int(n_components) if use_pca else 2
    ncomp = max(2, ncomp)  # per scatter 2D
    ncomp = min(ncomp, Xv.shape[1])  # non più delle feature
    if ncomp < 2:
        return (False, "Troppe poche feature per fare PCA 2D.", None, None, None, cols, valid_mask, None)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv)

    pca = PCA(n_components=ncomp)
    scores = pca.fit_transform(Xs)

    pc_cols = [f"PC{k}" for k in range(1, ncomp + 1)]
    scores_df = pd.DataFrame(scores, columns=pc_cols, index=X.loc[valid_mask].index)

    return (True, "OK", pca, scaler, scores_df, cols, valid_mask, pca.explained_variance_ratio_)
def run_kmeans_vars(
    X,
    *,
    k: int,
    n_init: int = 10,
    random_state: int = 0,
    min_samples: int = 2,
):
    """
    KMeans standalone: returns only variables.

    Input:
      - X: array-like (n_samples, n_features) oppure DataFrame
      - k: numero cluster

    Returns:
      ok (bool),
      msg (str),
      labels (np.ndarray|None),
      km (KMeans|None),
      Xc (np.ndarray|None)
    """
    if X is None:
        return (False, "Input X is None.", None, None, None)

    # convert to NumPy
    Xc = np.asarray(X)

    if Xc.ndim != 2:
        return (False, "X deve essere 2D (n_samples, n_features).", None, None, None)

    n_samples = Xc.shape[0]
    if n_samples < min_samples:
        return (False, "Non ho abbastanza campioni per fare clustering.", None, None, Xc)

    k = int(k)
    if k < 2:
        return (False, "k deve essere >= 2.", None, None, Xc)

    if k > n_samples:
        return (False, "k troppo grande rispetto al numero di campioni.", None, None, Xc)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(Xc)

    return (True, "OK", labels, km, Xc)
def main():

    nomi_feature_haralick = [
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
    nomi_features_haralick_canali=[]
    nomi_canali=["Red", "Green", "Blue"]
    for nome in nomi_canali:
        nomi_features_haralick_canali = np.concatenate([nomi_features_haralick_canali,[f"{n}_{nome}" for n in nomi_feature_haralick]])

    img=apri_immagine(r"C:\Users\Giovanni Gueltrini\Desktop\unibo\Tirocinio_cimbria\Prove_output_programma\dataset_prova\immagini_2.png")
    img_th=threshold(
            img,
            0, 50,
            0,50,
            80, 200)

    x,y=estrazioni_feature_e_nomi(img,img_th,nomi_features_geometriche, nomi_features_haralick_canali,nomi_canali, raggio=4, punti=7)

if __name__ == "__main__":
    directory_immagini_to_csv(r"C:\Users\Giovanni Gueltrini\Desktop\unibo\Tirocinio_cimbria\Prove_output_programma\dataset_prova")
    #main()