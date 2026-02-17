from PIL import Image, UnidentifiedImageError

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



def main():
    apri_immagine(r"C:\Users\Giovanni Gueltrini\Desktop\unibo\computer vision\Exam Computer Vsision\immagini\accelerator_sample.PNG")

if __name__ == "__main__":
    main()