from PIL import Image
import numpy as np
from hypothesis import given, strategies as st
import mahotas
import cv2
import pandas as pd
from function  import   threshold, estrazzione_features_geometriche, estrazione_feature_texturali
from function import (
    directory_immagini_to_csv,
    compute_pca_on_df_vars,
    run_kmeans_vars,
    threshold,
)
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
    """
    Tests:
    if the output image has the same size as the input image
    if each output pixel is 255 when the input pixel RGB values lie inside the given ranges
    if each output pixel is 0 when at least one RGB channel is outside the given ranges
    if the output binary mask matches exactly the mathematically expected result (pixel-wise)
        """
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
    """
    Tests:
    if the output is a 1D numpy array with the expected length of  14 features
    if all returned feature values are finite (no NaN or Inf)
    if the computed bounding-box height equals the rectangle height used to build the mask
    if the computed bounding-box width equals the rectangle width used to build the mask
    if the computed area equals the rectangle area
    if the computed aspect ratio equals width/height
    if the computed extent is 1.0 for a fully-filled rectangle
    """
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
    """
    Tests:
    if the function returns two 1D numpy arrays (Haralick features and LBP features)
    if both output vectors contain only finite values (no NaN or Inf)
    if the Haralick output length matches 3 * (2 + n_haralick), i.e. mean+variance plus Haralick features for each RGB channel
    if the LBP output length matches 3 * n_lbp, i.e. LBP histogram/features for each RGB channel
    if the first two elements of the Haralick block for the first channel are respectively the mean and the variance of the masked ROI
    """
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

def test_directory_immagini_to_csv(tmp_path) -> None:
    """
    Tests:
    if the function returns a pandas DataFrame
    if the DataFrame has exactly one column named "path"
    if only files with valid image extensions are included (e.g., .jpg, .png)
    if non-image files are excluded (e.g., .txt)
    if the number of rows equals the number of valid image files found in the directory
    if the output CSV file is created in the target directory with the expected name
    """
    # creo "immagini" finte: basta il file con estensione valida
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "nope.txt").write_text("x")

    df = directory_immagini_to_csv(str(tmp_path), recursive=False, csv_name="out.csv")

    assert list(df.columns) == ["path"]
    assert len(df) == 2
    assert (tmp_path / "out.csv").exists()

def test_compute_pca_on_df_vars() -> None:
    """
    Tests:
    if the function returns ok=True and the message is exactly "OK" for a valid numeric dataframe
    if the PCA scores dataframe is not None when use_pca=True
    if the scores dataframe has exactly the expected PCA component columns (["PC1", "PC2"])
    if the number of rows in the scores dataframe equals the number of input samples
    if the explained variance ratio (evr) is not None when PCA is computed
    if the explained variance ratio length equals n_components
    """

    df = pd.DataFrame({
        "path": ["a", "b", "c", "d"],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [4.0, 3.0, 2.0, 1.0],
        "f3": [0.1, 0.2, 0.1, 0.2],
    })

    ok, msg, pca, scaler, scores_df, cols, valid_mask, evr = compute_pca_on_df_vars(
        df, exclude={"path"}, use_pca=True, n_components=2
    )

    assert ok is True
    assert msg == "OK"
    assert scores_df is not None
    assert list(scores_df.columns) == ["PC1", "PC2"]
    assert len(scores_df) == 4
    assert evr is not None
    assert len(evr) == 2

def test_run_kmeans_vars() -> None:
    """
    Tests:
    if the function returns ok=True and the message is exactly "OK" for a valid input matrix
    if the returned labels array is not None
    if the number of labels equals the number of input samples
    if the produced labels belong to the expected set of cluster IDs {0, 1} when k=2
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 9.9],
    ])

    ok, msg, labels, km, Xc = run_kmeans_vars(X, k=2, random_state=0)

    assert ok is True
    assert msg == "OK"
    assert labels is not None
    assert len(labels) == 4
    assert set(labels) <= {0, 1}

def test_threshold() -> None:
    """
    Tests:
    if only pixels whose RGB channels all lie inside the given ranges are set to 255
    if pixels with at least one channel outside the given ranges are set to 0
    if the output mask matches exactly the expected 2x2 result for a controlled input image
    """
    # immagine 2x2 controllata
    arr = np.array([
        [[10, 10, 10], [200, 10, 10]],
        [[10, 200, 10], [10, 10, 200]],
    ], dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    out = threshold(img, 0, 50, 0, 50, 0, 50)  # passa solo pixel (10,10,10)
    out_arr = np.array(out, dtype=np.uint8)

    # atteso: solo [0,0] bianco, gli altri neri
    expected = np.array([
        [255, 0],
        [0, 0],
    ], dtype=np.uint8)

    assert np.array_equal(out_arr, expected)