import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps


class SimpleImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Tkinter Image Viewer")

        # Stato immagini
        self.original_img = None      # PIL Image originale
        self.current_img = None       # PIL Image corrente (modificata)
        self.tk_img = None            # riferimento per Tkinter (evita garbage collection)

        # ===== Top bar =====
        top = tk.Frame(root, padx=8, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        btn_open = tk.Button(top, text="Apri", width=12, command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=4)

        btn_gray = tk.Button(top, text="Grayscale", width=12, command=self.to_grayscale)
        btn_gray.pack(side=tk.LEFT, padx=4)

        btn_invert = tk.Button(top, text="Inverti", width=12, command=self.invert_image)
        btn_invert.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(top, text="Reset", width=12, command=self.reset_image)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Slider threshold
        self.thresh_value = tk.IntVar(value=128)
        self.thresh_scale = tk.Scale(
            top,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            label="Threshold",
            variable=self.thresh_value,
            length=220
        )
        self.thresh_scale.pack(side=tk.LEFT, padx=10)

        btn_thresh = tk.Button(top, text="Applica Threshold", command=self.apply_threshold)
        btn_thresh.pack(side=tk.LEFT, padx=4)

        # ===== Area immagine =====
        self.image_label = tk.Label(root, text="Apri un'immagine per iniziare", bg="#f0f0f0")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[
                ("Immagini", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tutti i file", "*.*")
            ]
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire l'immagine:\n{e}")
            return

        self.original_img = img
        self.current_img = img.copy()
        self.show_image(self.current_img)

    def show_image(self, pil_img):
        # Ridimensionamento leggero per stare nella finestra (senza complicare)
        max_w, max_h = 1000, 700
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img, text="")

    def ensure_image_loaded(self):
        if self.current_img is None:
            messagebox.showwarning("Attenzione", "Prima apri un'immagine.")
            return False
        return True

    def to_grayscale(self):
        if not self.ensure_image_loaded():
            return
        self.current_img = ImageOps.grayscale(self.current_img)
        self.show_image(self.current_img)

    def invert_image(self):
        if not self.ensure_image_loaded():
            return

        # Se è in grayscale inverti diretto, altrimenti inverti RGB
        if self.current_img.mode == "L":
            self.current_img = ImageOps.invert(self.current_img)
        else:
            self.current_img = ImageOps.invert(self.current_img.convert("RGB"))

        self.show_image(self.current_img)

    def apply_threshold(self):
        if not self.ensure_image_loaded():
            return

        t = self.thresh_value.get()

        # Threshold su grayscale
        gray = self.current_img.convert("L")
        bw = gray.point(lambda p: 255 if p >= t else 0, mode="L")

        self.current_img = bw
        self.show_image(self.current_img)

    def reset_image(self):
        if self.original_img is None:
            return
        self.current_img = self.original_img.copy()
        self.show_image(self.current_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleImageApp(root)
    root.geometry("1100x800")
    root.mainloop()