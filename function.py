from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
def apri_immagine(path: str) -> Image.Image:
    """
    Apre un'immagine da percorso e la converte in RGB.
    Funziona con qualsiasi formato supportato da Pillow
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
    print(np.unique(mask_bin))

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
    vettore=[]
    img_binary_np = np.array(img_binary.convert("L"), dtype=np.uint8)
    img_np = np.array(img.convert("L"), dtype=np.uint8)
    maschera_booleana = img_binary_np > 0
    #for i in range(3):
    pixel_selezionati = img_np[maschera_booleana]
    media_pixel = np.mean(pixel_selezionati)
    varianza_pixel = np.var(pixel_selezionati)
    roi = cv2.bitwise_and(img_np, img_np, mask=img_binary_np)
    features_haralick = mahotas.features.haralick(roi,ignore_zeros=True)
    mean_features_haralick = features_haralick.mean(axis=0)
    features_lib = mahotas.features.lbp(roi, raggio, punti, ignore_zeros=True)
    vettore.append(media_pixel)
    vettore.append(varianza_pixel)
    v_unico = np.concatenate([vettore, mean_features_haralick, features_lib])


def main():
    img=apri_immagine(r"C:\Users\Giovanni Gueltrini\Desktop\unibo\Tirocinio_cimbria\Prove_output_programma\dataset_prova\immagini_2.png")
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
    img_th=threshold(
            img,
            0, 50,
            0,50,
            80, 200
        )
    plt.figure()
    plt.imshow(img_th, cmap="gray")
    plt.axis("off")
    plt.show()
    #print(estrazzione_features(img_th))
    estrazione_feature_texturali(img,img_th)

if __name__ == "__main__":
    main()