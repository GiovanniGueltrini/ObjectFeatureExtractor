from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

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

def threshold(img: Image.Image, tr: int, tg: int, tb: int) -> Image.Image:
    """
    Applica threshold separato sui 3 canali RGB.
    Pixel bianco se: R >= tr AND G >= tg AND B >= tb
    altrimenti nero.

    Ritorna un'immagine PIL in scala di grigi ('L') binaria (0/255).
    """
    # Controllo soglie
    for t in (tr, tg, tb):
        if not (0 <= t <= 255):
            raise ValueError("Le soglie devono essere tra 0 e 255.")

    arr = np.array(img, dtype=np.uint8)  # img è già RGB grazie a apri_immagine
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    mask = (r >= tr) & (g >= tg) & (b >= tb)
    binary = np.where(mask, 255, 0).astype(np.uint8)

    return Image.fromarray(binary, mode="L")


def main():
    img=apri_immagine(r"C:\Users\Giovanni Gueltrini\Desktop\unibo\computer vision\Exam Computer Vsision\immagini\Accurancy_FLops_imagenet.PNG")
    img=threshold(img, 100,0,0)
    plt.figure()
    plt.imshow(img, cmap="gray")  # se è binaria/grayscale va benissimo
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()