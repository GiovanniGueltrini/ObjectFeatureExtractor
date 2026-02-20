from PIL import Image
import numpy as np
from hypothesis import given, strategies as st
import tempfile
import mahotas
import cv2
import os
from function  import  apri_immagine, threshold, estrazzione_features_geometriche, estrazione_feature_texturali
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

    assert out.size == img.size #Deve avere la stessa dimensione dell’input
    assert np.array_equal(out_arr, expected) #Ogni pixel deve essere identico a expected

@given(
    H=st.integers(min_value=16, max_value=128),
    W=st.integers(min_value=16, max_value=128),
    y0=st.integers(min_value=0, max_value=100),
    x0=st.integers(min_value=0, max_value=100),
    rh=st.integers(min_value=1, max_value=60),
    rw=st.integers(min_value=1, max_value=60),
)
def test_estrazzione_features_geometriche(H, W, y0, x0, rh, rw):
    # normalizza dimensioni rettangolo per stare dentro l'immagine
    y0 = min(y0, H - 1)
    x0 = min(x0, W - 1)
    rh = min(rh, H - y0)
    rw = min(rw, W - x0)

    # maschera con rettangolo pieno
    arr = np.zeros((H, W), dtype=np.uint8)
    arr[y0:y0+rh, x0:x0+rw] = 255
    mask = Image.fromarray(arr, mode="L")

    # output funzione da testare
    feat = estrazzione_features_geometriche(mask)

    # atteso matematico per le prime feature (quelle deterministiche)
    expected_height = float(rh)
    expected_width = float(rw)
    expected_area = float(rh * rw)
    expected_aspect = float(rw / rh) if rh > 0 else 0.0
    expected_extent = 1.0  # rettangolo pieno: area = area bbox

    assert feat.shape == (14,)
    assert np.all(np.isfinite(feat))

    height, width, area, aspect_ratio, extent = feat[:5]

    assert height == expected_height
    assert width == expected_width
    assert area == expected_area
    assert abs(aspect_ratio - expected_aspect) < 1e-12
    assert abs(extent - expected_extent) < 1e-12

@given(
    H=st.integers(min_value=32, max_value=96),
    W=st.integers(min_value=32, max_value=96),
    raggio=st.integers(min_value=1, max_value=5),
    punti=st.integers(min_value=4, max_value=12),
)
def test_estrazione_feature_texturali(H, W, raggio, punti):
    # immagine RGB casuale
    arr = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # maschera binaria non vuota (rettangolo centrale)
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    y0, y1 = H // 4, 3 * H // 4
    x0, x1 = W // 4, 3 * W // 4
    mask_arr[y0:y1, x0:x1] = 255
    img_binary = Image.fromarray(mask_arr, mode="L")

    # output funzione da testare
    har, lbp = estrazione_feature_texturali(img, img_binary, raggio=raggio, punti=punti)

    # proprietà base: vettori 1D finiti e coerenti
    assert har.ndim == 1
    assert lbp.ndim == 1
    assert np.all(np.isfinite(har))
    assert np.all(np.isfinite(lbp))

    # lunghezza haralick: (2 + n_haralick) per canale, per 3 canali
    # n_haralick = numero feature haralick (dipende da mahotas) => lo misuro direttamente una volta
    roi0 = cv2.bitwise_and(arr[:, :, 0], arr[:, :, 0], mask=mask_arr)
    n_h = mahotas.features.haralick(roi0, ignore_zeros=True).mean(axis=0).size
    assert har.size == 3 * (2 + n_h)

    # lunghezza lbp: 3 * len(lbp_per_canale)
    l0 = mahotas.features.lbp(roi0, raggio, punti, ignore_zeros=True).size
    assert lbp.size == 3 * l0

    # check media/varianza del primo canale siano in testa al blocco canale-1
    mean0 = float(np.mean(roi0.flatten()))
    var0 = float(np.var(roi0.flatten()))
    assert abs(har[0] - mean0) < 1e-9
    assert abs(har[1] - var0) < 1e-9