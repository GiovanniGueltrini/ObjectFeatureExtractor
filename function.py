from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas

from pathlib import Path
import pandas as pd

def apri_immagine(path: str) -> Image.Image:
    """
    Opens an image from a path and converts it to RGB.
    Works with any format supported by Pillow.
    (PNG, JPEG, BMP, TIFF, WEBP, GIF, ecc.).
    """
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {path}")
    except UnidentifiedImageError:
        raise ValueError(f"Formato non supportato o file corrotto: {path}")

def threshold(
    img: Image.Image,
    r_min: int, r_max: int,
    g_min: int, g_max: int,
    b_min: int, b_max: int
) -> Image.Image:
    """
    Threshold RGB con limite inferiore e superiore per ogni canale.
    Pixel bianco se il pixel è dentro tutti e 3 gli intervalli, altrimenti nero.

    Ritorna: PIL Image binaria mode 'L' (0/255).
    """
    # Validazione range
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
    Estrae feature geometriche da una maschera binaria.

    Feature ritornate (ordine):
    [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter, hu1..hu7]
    """
    # PIL -> numpy grayscale uint8
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)

    # Binarizzazione sicura (0/255)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Area (pixel bianchi)
    area = float(np.count_nonzero(mask_bin))

    # Se non c'è oggetto, ritorna zeri
    if area == 0:
        return np.zeros(14, dtype=float)  # 7 base + 7 Hu

    # Bounding box senza contorni
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

    # Per solidity e Hu moments serve un contorno
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(contorno)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Diametro equivalente
    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))

    # Hu moments
    moments = cv2.moments(contorno)
    hu = cv2.HuMoments(moments).flatten().astype(float)
    hu = np.sign(hu) * np.log(np.abs(hu) + 1e-12)
    # Feature finali
    features_base = np.array(
        [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter],
        dtype=float)
    features = np.concatenate([features_base, hu])
    return features

def estrazione_feature_texturali(img, img_binary, raggio=3, punti=4):
    """
    questa funzione restituirà un vettore dove saranno presenti le feature texturali
    per ogni canale dell'immagine.
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
    Questa funzione calcola il vettore completo delle feature (geometriche + testurali)
    e restituisce in parallelo i nomi corrispondenti a ciascuna feature, in modo da avere
    una rappresentazione interpretabile e tracciabile delle colonne (utile per CSV/DataFrame).

    La parte testurale è composta da:
    - Haralick (con media/varianza pixel aggiunte) per ciascun canale
    - LBP per ciascun canale (con nomi generati dinamicamente in base alla lunghezza)

    :param img: Image (RGB). Immagine di input su cui estrarre le feature.
    :param img_binary: Mask of the image. Maschera binaria che definisce la ROI.
    :param nomi_features_geometriche: lista/array di stringhe contenente i nomi delle feature geometriche.
    :param nomi_features_haralick_canali: lista/array di stringhe contenente i nomi delle feature haralick (già espansi per canale).
    :param nomi_canali: lista di stringhe con i nomi dei canali (es. ["R","G","B"]).
    :param raggio: parameter of lbp. Raggio usato per il calcolo delle LBP.
    :param punti: parameter of lbp. Numero di punti usati per il calcolo delle LBP.
    :return: (features, nomi)
             - features: ndarray 1D con tutte le feature concatenate (geometriche + haralick + lbp)
             - nomi: ndarray/lista con i nomi corrispondenti alle feature (stesso ordine e stessa lunghezza)
    """
    # Estrazione delle feature testurali (separate in haralick e lbp)
    haralick, lib = estrazione_feature_texturali(img, img_binary, raggio=raggio, punti=punti)
    # Concatenazione delle feature testurali in un unico vettore
    feature_testurali = np.concatenate([haralick, lib])
    # Concatenazione finale: feature geometriche + feature testurali
    features = np.concatenate([estrazzione_features_geometriche(img_binary), feature_testurali])
    # Costruzione dei nomi delle feature LBP:
    # lib contiene le LBP di tutti e 3 i canali concatenate, quindi la lunghezza per canale è len(lib)/3
    nomi_lib = []
    len_lib = len(lib) / 3
    # Per ogni canale, creo i nomi LBP_1_<canale>, LBP_2_<canale>, ...
    for nome_canale in nomi_canali:
        for i in range(int(len_lib)):
            nomi_lib.append(f"LBP_{i+1}_{nome_canale}")
    # Concatenazione finale dei nomi: geometriche + haralick (già preparate) + lbp (generate qui)
    nomi = np.concatenate([nomi_features_geometriche, nomi_features_haralick_canali, nomi_lib])
    return features, nomi

def directory_immagini_to_csv(directory_path: str, recursive: bool = True, csv_name: str = "paths_immagini.csv") -> pd.DataFrame:
    """
    Crea un DataFrame con una sola colonna 'path' contenente i percorsi delle immagini
    presenti in una directory (opzionalmente anche nelle sottodirectory) e lo salva come CSV
    nella stessa directory.

    Args:
        directory_path: percorso della directory da scandire
        recursive: se True scandisce anche le sottocartelle
        csv_name: nome del file csv da salvare nella directory

    Returns:
        df: DataFrame con colonna 'path'
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory non trovata: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Non è una directory: {dir_path}")

    # Estensioni immagini comuni (aggiungine se ti serve)
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