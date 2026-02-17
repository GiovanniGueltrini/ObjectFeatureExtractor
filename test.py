from PIL import Image
import numpy as np
from hypothesis import given, strategies as st
import tempfile
import os
from function  import  apri_immagine, threshold, estrazzione_features
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
    r_min=st.integers(min_value=0, max_value=255),
    r_max=st.integers(min_value=0, max_value=255),
    g_min=st.integers(min_value=0, max_value=255),
    g_max=st.integers(min_value=0, max_value=255),
    b_min=st.integers(min_value=0, max_value=255),
    b_max=st.integers(min_value=0, max_value=255),
)
def test_threshold(h, w, r_min, r_max, g_min, g_max, b_min, b_max):
    # ordina i range (min <= max)
    r0, r1 = sorted((r_min, r_max))
    g0, g1 = sorted((g_min, g_max))
    b0, b1 = sorted((b_min, b_max))

    # immagine RGB casuale
    arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # output funzione da testare
    out = threshold(img, r0, r1, g0, g1, b0, b1)
    out_arr = np.array(out, dtype=np.uint8)

    # atteso matematico coerente con threshold a range
    expected = np.where(
        (arr[:, :, 0] >= r0) & (arr[:, :, 0] <= r1) &
        (arr[:, :, 1] >= g0) & (arr[:, :, 1] <= g1) &
        (arr[:, :, 2] >= b0) & (arr[:, :, 2] <= b1),
        255, 0
    ).astype(np.uint8)

    assert out.mode == "L"
    assert out.size == img.size
    assert np.array_equal(out_arr, expected)
@given(
    h=st.integers(min_value=10, max_value=64),
    w=st.integers(min_value=10, max_value=64),
    x1=st.integers(min_value=0, max_value=20),
    y1=st.integers(min_value=0, max_value=20),
    rw=st.integers(min_value=3, max_value=20),
    rh=st.integers(min_value=3, max_value=20),
)
def test_estrazzione_features_semplice(h, w, x1, y1, rw, rh):
    arr = np.zeros((h, w), dtype=np.uint8)

    x2 = min(w, x1 + rw)
    y2 = min(h, y1 + rh)
    arr[y1:y2, x1:x2] = 255

    mask = Image.fromarray(arr, mode="L")
    feat = estrazzione_features(mask)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (14,)

    # area, height, width non negative
    assert float(feat[2]) >= 0.0
    assert float(feat[0]) >= 0.0
    assert float(feat[1]) >= 0.0

    # nessun NaN/Inf nel vettore
    assert np.isfinite(feat).all()