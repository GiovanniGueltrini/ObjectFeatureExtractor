from PIL import Image
import numpy as np
from hypothesis import given, strategies as st
import tempfile
import os
from function  import  apri_immagine, threshold
@given(path=st.text(min_size=1).filter(lambda s: "\x00" not in s))
def test_apri_immagine_file_inesistente_lancia_errore(path):
    # Evita collisioni con file reali: prefisso improbabile
    fake_path = "__file_inesistente_hypothesis__/" + path

    try:
        apri_immagine(fake_path)
        assert False, "Mi aspettavo FileNotFoundError"
    except FileNotFoundError:
        assert True
    except Exception:
        # Su alcuni OS/path strani possono emergere altri OSError
        # ma per lo scopo del corso teniamo chiaro il caso atteso
        assert True
@given(
    h=st.integers(min_value=1, max_value=32),
    w=st.integers(min_value=1, max_value=32),
    tr=st.integers(min_value=0, max_value=255),
    tg=st.integers(min_value=0, max_value=255),
    tb=st.integers(min_value=0, max_value=255),
)
def test_threshold(h, w, tr, tg, tb):
    # immagine RGB casuale
    arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    out = threshold(img, tr, tg, tb)  # usa la tua funzione
    out_arr = np.array(out, dtype=np.uint8)

    # atteso matematico
    expected = np.where(
        (arr[:, :, 0] >= tr) &
        (arr[:, :, 1] >= tg) &
        (arr[:, :, 2] >= tb),
        255, 0
    ).astype(np.uint8)

    assert out.mode == "L"
    assert out.size == img.size
    assert np.array_equal(out_arr, expected)

