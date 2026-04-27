from PIL import Image
import numpy as np
from hypothesis import given, strategies as st
import mahotas
import cv2
from pathlib import Path
import tkinter as tk
from Dashboard import App
import pandas as pd
import os
from function import (read_input_csv,
    estrazzione_features_geometriche,
    threshold,extract_paths, build_feature_row, update_dataset_with_feature_rows, resize_image_for_preview,extract_saved_features_from_row,
                      apply_rgb_threshold,  extract_image_features, FeatureExtractionResult, select_numeric_feature_columns,
                      compute_pca_from_dataframe,run_kmeans, estrazione_feature_texturali )


######################### CSV MANAGER #######################################

def test_read_input_csv_reads_dummy_csv_file(tmp_path):
    """
    read_input_csv should read a CSV file, keep the path column,
    and strip spaces from path values.
    """
    csv_path = tmp_path / "dummy_paths.csv"

    csv_path.write_text(
        "path\n"
        " image_1.png \n"
        "image_2.jpg\n",
        encoding="utf-8-sig",
    )

    df = read_input_csv(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["path"]
    assert df["path"].tolist() == [
        "image_1.png",
        "image_2.jpg",
    ]

def test_threshold_sets_only_pixels_inside_rgb_range_to_white():
    """
    The threshold function should return 255 only for pixels whose R, G and B
    values are all inside the selected ranges.
    """
    image_array = np.array([
        [[10, 10, 10], [200, 10, 10]],
        [[10, 200, 10], [10, 10, 200]],
    ], dtype=np.uint8)

    image = Image.fromarray(image_array, mode="RGB")

    result = threshold(
        image,
        0, 50,
        0, 50,
        0, 50,
    )

    result_array = np.array(result, dtype=np.uint8)

    expected = np.array([
        [255, 0],
        [0, 0],
    ], dtype=np.uint8)

    assert result.size == image.size
    assert np.array_equal(result_array, expected)

def test_extract_paths_returns_only_valid_non_empty_paths():
    """
    extract_paths should return only non-empty paths from the path column.
    """
    df = pd.DataFrame({
        "path": [
            "image_1.png",
            " image_2.jpg ",
            "",
            None,
            "image_3.tif",
        ]
    })

    paths = extract_paths(df)

    assert paths == [
        "image_1.png",
        "image_2.jpg",
        "image_3.tif",
    ]

def test_build_feature_row_returns_features_metadata_and_path():
    """
    build_feature_row should return a dictionary containing feature values,
    image path, threshold values and LBP parameters.
    """
    image_path = "images/sample_1.png"
    feature_names = ["area", "solidity", "contrast"]
    feature_values = [120.0, 0.85, 15.5]

    threshold_values = {
        "thr_rmin": 0,
        "thr_rmax": 255,
        "thr_gmin": 10,
        "thr_gmax": 200,
        "thr_bmin": 20,
        "thr_bmax": 180,
    }

    row = build_feature_row(
        image_path=image_path,
        feature_names=feature_names,
        feature_values=feature_values,
        threshold_values=threshold_values,
        lbp_radius=3,
        lbp_points=8,
    )

    assert row["path"] == "images/sample_1.png"

    assert row["area"] == 120.0
    assert row["solidity"] == 0.85
    assert row["contrast"] == 15.5

    assert row["thr_rmin"] == 0
    assert row["thr_rmax"] == 255
    assert row["thr_gmin"] == 10
    assert row["thr_gmax"] == 200
    assert row["thr_bmin"] == 20
    assert row["thr_bmax"] == 180

    assert row["lbp_raggio"] == 3
    assert row["lbp_punti"] == 8


def test_update_dataset_with_feature_rows_updates_matching_path():
    """
    update_dataset_with_feature_rows should update existing feature values
    and add new feature columns for matching image paths.
    """
    df = pd.DataFrame({
        "path": ["image_1.png", "image_2.png"],
        "area": [10.0, 20.0],
    })

    feature_rows = [
        {
            "path": "image_1.png",
            "area": 99.0,
            "solidity": 0.85,
        }
    ]

    updated_df = update_dataset_with_feature_rows(df, feature_rows)

    assert updated_df.loc[0, "path"] == "image_1.png"
    assert updated_df.loc[0, "area"] == 99.0
    assert updated_df.loc[0, "solidity"] == 0.85

    assert updated_df.loc[1, "path"] == "image_2.png"
    assert updated_df.loc[1, "area"] == 20.0
    assert pd.isna(updated_df.loc[1, "solidity"])




def test_extract_saved_features_from_row_ignores_missing_and_empty_values():
    """
    extract_saved_features_from_row should return only feature columns
    with non-missing and non-empty values.
    """
    row = pd.Series({
        "path": "image_1.png",
        "area": 120.0,
        "solidity": 0.85,
        "contrast": pd.NA,
        "entropy": "",
        "unused_column": 999,
    })

    feature_columns = [
        "area",
        "solidity",
        "contrast",
        "entropy",
        "missing_column",
    ]

    names, values = extract_saved_features_from_row(
        row=row,
        feature_columns=feature_columns,
    )

    assert names == ["area", "solidity"]
    assert values == [120.0, 0.85]

############# test image_manager#############################
@given(
    width=st.integers(min_value=1, max_value=3000),
    height=st.integers(min_value=1, max_value=3000),
    max_width=st.integers(min_value=1, max_value=1000),
    max_height=st.integers(min_value=1, max_value=1000),
)

def test_resize_image_for_preview_never_exceeds_max_size(
    width,
    height,
    max_width,
    max_height,
):
    """
    resize_image_for_preview should never return an image larger than
    the requested maximum width and height.
    """
    image = Image.new("RGB", (width, height))

    resized = resize_image_for_preview(
        image,
        max_width=max_width,
        max_height=max_height,
    )

    resized_width, resized_height = resized.size

    assert resized_width <= max_width
    assert resized_height <= max_height
    assert resized_width >= 1
    assert resized_height >= 1

def test_apply_rgb_threshold_calls_threshold_function_with_valid_values():
    """
    apply_rgb_threshold should validate RGB threshold values and call the
    provided threshold function with the expected arguments.
    """
    image = Image.new("RGB", (2, 2))

    calls = {}

    def fake_threshold_function(image_arg, rmin, rmax, gmin, gmax, bmin, bmax):
        calls["image"] = image_arg
        calls["values"] = (rmin, rmax, gmin, gmax, bmin, bmax)
        return Image.new("L", image_arg.size)

    result = apply_rgb_threshold(
        image=image,
        threshold_function=fake_threshold_function,
        rmin=0,
        rmax=255,
        gmin=10,
        gmax=200,
        bmin=20,
        bmax=180,
    )

    assert result.mode == "L"
    assert result.size == image.size
    assert calls["image"] is image
    assert calls["values"] == (0, 255, 10, 200, 20, 180)


#####################feature_service######################
@given(
    width=st.integers(min_value=1, max_value=128),
    height=st.integers(min_value=1, max_value=128),
)
def test_threshold_returns_binary_mask_with_same_size(width, height):
    """
    threshold should always return a binary mask with the same size as the
    input image.
    """
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode="RGB")

    result = threshold(
        image,
        0, 255,
        0, 255,
        0, 255,
    )

    result_array = np.array(result, dtype=np.uint8)

    assert result.mode == "L"
    assert result.size == image.size
    assert set(np.unique(result_array)) <= {0, 255}
@given(
    image_height=st.integers(min_value=16, max_value=128),
    image_width=st.integers(min_value=16, max_value=128),
    rectangle_height=st.integers(min_value=1, max_value=32),
    rectangle_width=st.integers(min_value=1, max_value=32),
)
def test_geometric_features_are_finite_for_random_rectangles(
    image_height,
    image_width,
    rectangle_height,
    rectangle_width,
):
    """
    estrazzione_features_geometriche should return a finite 14-element vector
    for valid filled rectangular masks of different sizes.
    """
    rectangle_height = min(rectangle_height, image_height)
    rectangle_width = min(rectangle_width, image_width)

    mask_array = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_array[0:rectangle_height, 0:rectangle_width] = 255

    mask = Image.fromarray(mask_array, mode="L")

    features = estrazzione_features_geometriche(mask)

    assert features.shape == (14,)
    assert np.all(np.isfinite(features))



def create_texture_test_image_and_mask():
    """
    Create a small RGB image and a non-empty binary mask for texture tests.
    """
    image_array = np.full((40, 40, 3), 100, dtype=np.uint8)

    mask_array = np.zeros((40, 40), dtype=np.uint8)
    mask_array[10:30, 10:30] = 255

    image = Image.fromarray(image_array, mode="RGB")
    mask = Image.fromarray(mask_array, mode="L")

    return image, mask

def test_texture_features_are_finite():
    """
    estrazione_feature_texturali should return finite numerical values.
    """
    image, mask = create_texture_test_image_and_mask()

    haralick_features, lbp_features = estrazione_feature_texturali(
        image,
        mask,
        raggio=3,
        punti=8,
    )

    assert np.all(np.isfinite(haralick_features))
    assert np.all(np.isfinite(lbp_features))
def fake_extraction_function(
        image_arg,
        binary_image_arg,
        geometric_feature_names,
        haralick_channel_feature_names,
        channel_names,
        raggio,
        punti,
    ):
        return [100.0, 0.85], ["area", "solidity"]
def test_extract_image_features_returns_structured_result():
    """
    extract_image_features should call the extraction function and return
    feature names and values inside a FeatureExtractionResult object.
    """
    image = Image.new("RGB", (10, 10))
    binary_image = Image.new("L", (10, 10))



    result = extract_image_features(
        image=image,
        binary_image=binary_image,
        extraction_function=fake_extraction_function,
        geometric_feature_names=["area", "solidity"],
        haralick_channel_feature_names=[],
        channel_names=["red", "green", "blue"],
        lbp_radius=3,
        lbp_points=8,
    )

    assert isinstance(result, FeatureExtractionResult)
    assert result.values == [100.0, 0.85]
    assert result.names == ["area", "solidity"]
    assert result.warning is None

def test_extract_image_features_truncates_output_and_returns_warning_on_length_mismatch():
    """
    extract_image_features should truncate names and values to the same length
    and return a warning when their lengths do not match.
    """
    image = Image.new("RGB", (10, 10))
    binary_image = Image.new("L", (10, 10))

    def fake_extraction_function_with_mismatch(
        image_arg,
        binary_image_arg,
        geometric_feature_names,
        haralick_channel_feature_names,
        channel_names,
        raggio,
        punti,
    ):
        return [100.0, 0.85, 15.2], ["area", "solidity"]

    result = extract_image_features(
        image=image,
        binary_image=binary_image,
        extraction_function=fake_extraction_function_with_mismatch,
        geometric_feature_names=["area", "solidity"],
        haralick_channel_feature_names=[],
        channel_names=["red", "green", "blue"],
        lbp_radius=3,
        lbp_points=8,
    )

    assert result.values == [100.0, 0.85]
    assert result.names == ["area", "solidity"]
    assert result.warning is not None
    assert "does not match" in result.warning
###################################PCA_servie######################à
def test_select_numeric_feature_columns_excludes_metadata_columns():
    """
    select_numeric_feature_columns should exclude path, threshold and LBP
    metadata columns and keep valid numeric feature columns.
    """
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png"],
        "thr_rmin": [0, 0],
        "thr_rmax": [255, 255],
        "lbp_raggio": [3, 3],
        "lbp_punti": [8, 8],
        "area": [100.0, 200.0],
        "solidity": [0.85, 0.90],
    })

    columns = select_numeric_feature_columns(df)

    assert columns == ["area", "solidity"]

def test_select_numeric_feature_columns_accepts_numeric_strings():
    """
    select_numeric_feature_columns should keep columns that contain numeric
    values stored as strings.
    """
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png"],
        "area": ["100.0", "200.0"],
        "solidity": ["0.85", "0.90"],
    })

    columns = select_numeric_feature_columns(df)

    assert columns == ["area", "solidity"]




def test_compute_pca_from_dataframe_returns_complete_result():
    """
    compute_pca_from_dataframe should return PCA scores, model, scaler,
    selected feature columns, valid row mask and explained variance ratio.
    """
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"],
        "thr_rmin": [0, 0, 0, 0],
        "lbp_raggio": [3, 3, 3, 3],
        "area": [10.0, 20.0, 30.0, 40.0],
        "solidity": [0.80, 0.85, 0.90, 0.95],
        "contrast": [5.0, 6.0, 7.0, 8.0],
    })

    result = compute_pca_from_dataframe(
        df=df,
        n_components=2,
        use_pca=True,
    )

    assert list(result.scores.columns) == ["PC1", "PC2"]
    assert result.scores.shape == (4, 2)

    assert result.feature_columns == ["area", "solidity", "contrast"]
    assert result.valid_mask.tolist() == [True, True, True, True]

    assert result.model.n_components == 2
    assert result.scaler is not None
    assert len(result.explained_variance_ratio) == 2

def test_compute_pca_from_dataframe_excludes_rows_with_missing_values():
    """
    compute_pca_from_dataframe should exclude rows containing NaN values
    in the selected feature columns.
    """
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"],
        "area": [10.0, None, 30.0, 40.0],
        "solidity": [0.80, 0.85, 0.90, 0.95],
        "contrast": [5.0, 6.0, 7.0, 8.0],
    })

    result = compute_pca_from_dataframe(
        df=df,
        n_components=2,
        use_pca=True,
    )

    assert result.scores.shape == (3, 2)
    assert result.valid_mask.tolist() == [True, False, True, True]
    assert result.feature_columns == ["area", "solidity", "contrast"]

###########################k-means#############################
def test_run_kmeans_returns_complete_clustering_result():
    """
    run_kmeans should return labels, fitted model, centroids and inertia
    for a valid input matrix.
    """
    data = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ])

    result = run_kmeans(
        data=data,
        n_clusters=2,
        n_init=10,
        random_state=0,
    )

    assert result.labels.shape == (4,)
    assert set(result.labels) <= {0, 1}

    assert result.centroids.shape == (2, 2)

    assert result.model.n_clusters == 2
    assert result.model.n_init == 10

    assert isinstance(result.inertia, float)
    assert result.inertia >= 0


########### Test  ui ###########################
def test_app_initializes_without_errors():
    """
    App should initialize the Tkinter interface without raising errors.
    """
    root = tk.Tk()
    root.withdraw()

    app = App(root)

    assert app.root is root

    root.destroy()


def test_app_creates_main_widgets():
    """
    App should create the main widgets used by the interface.
    """
    root = tk.Tk()
    root.withdraw()

    app = App(root)

    assert hasattr(app, "status_label")
    assert hasattr(app, "original_image_label")
    assert hasattr(app, "mask_image_label")
    assert hasattr(app, "feature_text")

    root.destroy()



