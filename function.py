from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
######################### CSV MANAGER #######################################
from pathlib import Path
from typing import Any

import pandas as pd
from dataclasses import dataclass


PATH_COLUMN = "path"

THRESHOLD_COLUMNS = [
    "thr_rmin",
    "thr_rmax",
    "thr_gmin",
    "thr_gmax",
    "thr_bmin",
    "thr_bmax",
]

LBP_COLUMNS = [
    "lbp_raggio",
    "lbp_punti",
]


def read_input_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Read the input CSV file and ensure that it contains a 'path' column.

    If the CSV does not contain a 'path' column, the first column is renamed
    to 'path'. Leading and trailing spaces in image paths are removed.
    """
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if PATH_COLUMN not in df.columns:
        first_column = df.columns[0]
        df = df.rename(columns={first_column: PATH_COLUMN})

    df[PATH_COLUMN] = df[PATH_COLUMN].astype(str).str.strip()

    return df


def extract_paths(df: pd.DataFrame) -> list[str]:
    """
    Extract valid image paths from the 'path' column.
    """
    if PATH_COLUMN not in df.columns:
        return []

    return (
        df[PATH_COLUMN]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all columns that are not the image path column.
    """
    return [column for column in df.columns if column != PATH_COLUMN]


def get_real_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return only feature columns, excluding path, threshold columns and LBP columns.
    """
    excluded_columns = set([PATH_COLUMN] + THRESHOLD_COLUMNS + LBP_COLUMNS)

    return [column for column in df.columns if column not in excluded_columns]


def find_row_by_path(df: pd.DataFrame, image_path: str) -> pd.Series | None:
    """
    Return the row corresponding to an image path.

    If no row is found, return None.
    """
    if PATH_COLUMN not in df.columns:
        return None

    matches = df[df[PATH_COLUMN] == str(image_path).strip()]

    if matches.empty:
        return None

    return matches.iloc[0]


def build_feature_row(
    image_path: str,
    feature_names: list[str],
    feature_values: list[Any],
    threshold_values: dict[str, int],
    lbp_radius: int,
    lbp_points: int,
) -> dict[str, Any]:
    """
    Build a dictionary containing path, features, threshold values and LBP parameters.

    This dictionary can later be merged into the dataset CSV.
    """
    n = min(len(feature_names), len(feature_values))

    row = {
        feature_names[index]: feature_values[index]
        for index in range(n)
    }

    row[PATH_COLUMN] = image_path
    row.update(threshold_values)
    row["lbp_raggio"] = lbp_radius
    row["lbp_punti"] = lbp_points

    return row


def update_dataset_with_feature_rows(
    df: pd.DataFrame,
    feature_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Merge extracted feature rows into the original dataset.

    Existing columns are overwritten with the new values when a matching
    image path is found.
    """
    if not feature_rows:
        return df.copy()

    feature_df = pd.DataFrame(feature_rows)

    if PATH_COLUMN not in feature_df.columns:
        raise ValueError("Feature rows must contain a 'path' field.")

    updated_df = df.merge(
        feature_df,
        on=PATH_COLUMN,
        how="left",
        suffixes=("", "_new"),
    )

    for column in feature_df.columns:
        if column == PATH_COLUMN:
            continue

        new_column = f"{column}_new"

        if new_column in updated_df.columns:
            updated_df[column] = updated_df[new_column]
            updated_df = updated_df.drop(columns=[new_column])

    return updated_df


def save_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    """
    Save the DataFrame to CSV.
    """
    csv_path = Path(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def extract_threshold_values(row: pd.Series) -> dict[str, int] | None:
    """
    Extract RGB threshold values from a CSV row.

    Return None if at least one threshold column is missing or empty.
    """
    if not all(column in row.index for column in THRESHOLD_COLUMNS):
        return None

    if row[THRESHOLD_COLUMNS].isna().any():
        return None

    return {
        "thr_rmin": int(row["thr_rmin"]),
        "thr_rmax": int(row["thr_rmax"]),
        "thr_gmin": int(row["thr_gmin"]),
        "thr_gmax": int(row["thr_gmax"]),
        "thr_bmin": int(row["thr_bmin"]),
        "thr_bmax": int(row["thr_bmax"]),
    }


def extract_lbp_values(row: pd.Series) -> dict[str, int] | None:
    """
    Extract LBP parameters from a CSV row.

    Return None if LBP columns are missing or empty.
    """
    if not all(column in row.index for column in LBP_COLUMNS):
        return None

    if row[LBP_COLUMNS].isna().any():
        return None

    return {
        "lbp_raggio": int(row["lbp_raggio"]),
        "lbp_punti": int(row["lbp_punti"]),
    }


def extract_saved_features_from_row(
    row: pd.Series,
    feature_columns: list[str],
) -> tuple[list[str], list[Any]]:
    """
    Extract already saved feature names and values from a CSV row.

    Missing values and empty strings are ignored.
    """
    names = []
    values = []

    for column in feature_columns:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        if isinstance(value, str) and value.strip() == "":
            continue

        names.append(column)
        values.append(value)

    return names, values
#######################image_manager.py#############################

def load_rgb_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk and convert it to RGB.

    Parameters
    ----------
    image_path:
        Path of the image to load.

    Returns
    -------
    Image.Image
        RGB image.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    OSError
        If the image cannot be opened by PIL.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def resize_image_for_preview(
    image: Image.Image,
    max_width: int = 520,
    max_height: int = 520,
    resample: int = Image.Resampling.NEAREST,
) -> Image.Image:
    """
    Resize an image while preserving its aspect ratio.

    This function is used only to create a GUI preview. It does not modify
    the original image used for feature extraction.

    Parameters
    ----------
    image:
        Input image.
    max_width:
        Maximum preview width.
    max_height:
        Maximum preview height.
    resample:
        PIL resampling method.

    Returns
    -------
    Image.Image
        Resized image.
    """
    original_width, original_height = image.size

    scale = min(
        max_width / original_width,
        max_height / original_height,
    )

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    return image.resize((new_width, new_height), resample)


def validate_threshold_values(
    rmin: int,
    rmax: int,
    gmin: int,
    gmax: int,
    bmin: int,
    bmax: int,
) -> None:
    """
    Validate RGB threshold values.

    Values must be between 0 and 255 and each minimum must be lower than
    or equal to the corresponding maximum.

    Raises
    ------
    ValueError
        If at least one threshold value is invalid.
    """
    values = {
        "rmin": rmin,
        "rmax": rmax,
        "gmin": gmin,
        "gmax": gmax,
        "bmin": bmin,
        "bmax": bmax,
    }

    for name, value in values.items():
        if value < 0 or value > 255:
            raise ValueError(f"{name} must be between 0 and 255.")

    if rmin > rmax:
        raise ValueError("rmin must be lower than or equal to rmax.")

    if gmin > gmax:
        raise ValueError("gmin must be lower than or equal to gmax.")

    if bmin > bmax:
        raise ValueError("bmin must be lower than or equal to bmax.")


def apply_rgb_threshold(
    image: Image.Image,
    threshold_function: Any,
    rmin: int,
    rmax: int,
    gmin: int,
    gmax: int,
    bmin: int,
    bmax: int,
) -> Image.Image:
    """
    Validate threshold values and apply the RGB threshold function.

    Parameters
    ----------
    image:
        Input RGB image.
    threshold_function:
        Function used to compute the binary mask. In your project this is
        the existing `threshold` function.
    rmin, rmax, gmin, gmax, bmin, bmax:
        RGB threshold limits.

    Returns
    -------
    Image.Image
        Binary mask image.
    """
    validate_threshold_values(
        rmin=rmin,
        rmax=rmax,
        gmin=gmin,
        gmax=gmax,
        bmin=bmin,
        bmax=bmax,
    )

    return threshold_function(
        image,
        rmin,
        rmax,
        gmin,
        gmax,
        bmin,
        bmax,
    )

######################feature_service.py################à
@dataclass
class FeatureExtractionResult:
    """
    Container for the output of feature extraction.

    Attributes
    ----------
    values:
        Extracted feature values.
    names:
        Feature names.
    warning:
        Optional warning message, for example when the number of names and
        values does not match.
    """
    values: list[Any]
    names: list[str]
    warning: str | None = None


def extract_image_features(
    image: Image.Image,
    binary_image: Image.Image,
    extraction_function,
    geometric_feature_names: list[str],
    haralick_channel_feature_names: list[str],
    channel_names: list[str],
    lbp_radius: int,
    lbp_points: int,
) -> FeatureExtractionResult:
    """
    Extract geometric and texture features from an image and its binary mask.

    This function wraps the existing feature extraction function used by the
    project and returns a structured result.

    Parameters
    ----------
    image:
        Original RGB image.
    binary_image:
        Binary mask obtained from thresholding.
    extraction_function:
        Existing feature extraction function. In this project it is
        `estrazioni_feature_e_nomi`.
    geometric_feature_names:
        Names of geometric features.
    haralick_channel_feature_names:
        Names of Haralick features for each RGB channel.
    channel_names:
        RGB channel names.
    lbp_radius:
        Radius used for Local Binary Patterns.
    lbp_points:
        Number of points used for Local Binary Patterns.

    Returns
    -------
    FeatureExtractionResult
        Extracted feature names, values and optional warning.
    """
    feature_values, feature_names = extraction_function(
        image,
        binary_image,
        geometric_feature_names,
        haralick_channel_feature_names,
        channel_names,
        raggio=lbp_radius,
        punti=lbp_points,
    )

    feature_values = list(feature_values)
    feature_names = list(feature_names)

    warning = None

    if len(feature_values) != len(feature_names):
        warning = (
            f"Number of feature values ({len(feature_values)}) does not match "
            f"number of feature names ({len(feature_names)})."
        )

        common_length = min(len(feature_values), len(feature_names))
        feature_values = feature_values[:common_length]
        feature_names = feature_names[:common_length]

    return FeatureExtractionResult(
        values=feature_values,
        names=feature_names,
        warning=warning,
    )


def format_features_for_display(
    feature_names: list[str],
    feature_values: list[Any],
) -> str:
    """
    Format feature names and values for display in the Tkinter text box.

    Parameters
    ----------
    feature_names:
        Names of the extracted features.
    feature_values:
        Values of the extracted features.

    Returns
    -------
    str
        Formatted text.
    """
    lines = []

    for name, value in zip(feature_names, feature_values):
        lines.append(f"\t{name}=\t{value}")

    return "\n".join(lines)


def append_warning_to_display_text(
    text: str,
    warning: str | None,
) -> str:
    """
    Append a warning message to feature display text, if a warning exists.
    """
    if warning is None:
        return text

    return f"{text}\n\n[WARNING] {warning}"

###################################PCA_servie######################à
from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


EXCLUDED_PCA_COLUMNS = {
    "path",
    "thr_rmin",
    "thr_rmax",
    "thr_gmin",
    "thr_gmax",
    "thr_bmin",
    "thr_bmax",
    "lbp_raggio",
    "lbp_punti",
}


@dataclass
class PCAResult:
    """
    Container for PCA output.

    Attributes
    ----------
    model:
        Fitted PCA model.
    scaler:
        Fitted StandardScaler model.
    scores:
        DataFrame containing PCA scores with columns PC1, PC2, ...
    feature_columns:
        Numeric feature columns actually used for PCA.
    valid_mask:
        Boolean mask identifying rows used for PCA.
    explained_variance_ratio:
        Explained variance ratio for each principal component.
    """
    model: PCA
    scaler: StandardScaler
    scores: pd.DataFrame
    feature_columns: list[str]
    valid_mask: pd.Series
    explained_variance_ratio: object


def select_numeric_feature_columns(
    df: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> list[str]:
    """
    Select numeric feature columns suitable for PCA.

    Columns listed in excluded_columns are ignored. Columns that cannot be
    converted to numeric values or that contain only NaN values are removed.

    Parameters
    ----------
    df:
        Input DataFrame.
    excluded_columns:
        Columns to exclude from PCA.

    Returns
    -------
    list[str]
        Valid numeric feature columns.
    """
    if excluded_columns is None:
        excluded_columns = EXCLUDED_PCA_COLUMNS

    candidate_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    if not candidate_columns:
        return []

    numeric_df = df[candidate_columns].apply(pd.to_numeric, errors="coerce")

    feature_columns = [
        column for column in numeric_df.columns
        if not numeric_df[column].isna().all()
    ]

    return feature_columns


def compute_pca_from_dataframe(
    df: pd.DataFrame,
    n_components: int = 2,
    use_pca: bool = True,
    excluded_columns: set[str] | None = None,
) -> PCAResult:
    """
    Compute PCA from the valid numeric feature columns of a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame containing feature columns.
    n_components:
        Number of PCA components requested.
    use_pca:
        If False, force the number of components to 2.
    excluded_columns:
        Columns that must not be used for PCA.

    Returns
    -------
    PCAResult
        Structured PCA result.

    Raises
    ------
    ValueError
        If no valid numeric feature columns are available or if there are not
        enough valid rows.
    """
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty.")

    if excluded_columns is None:
        excluded_columns = EXCLUDED_PCA_COLUMNS

    feature_columns = select_numeric_feature_columns(
        df,
        excluded_columns=excluded_columns,
    )

    if not feature_columns:
        raise ValueError("No valid numeric feature columns available for PCA.")

    numeric_df = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    valid_mask = ~numeric_df.isna().any(axis=1)
    valid_data = numeric_df.loc[valid_mask].values

    if valid_data.shape[0] < 2:
        raise ValueError("At least two valid rows are required for PCA.")

    max_components = min(valid_data.shape[0], valid_data.shape[1])

    if max_components < 2:
        raise ValueError("At least two valid numeric features or rows are required for PCA.")

    if use_pca:
        final_n_components = int(n_components)
    else:
        final_n_components = 2

    final_n_components = max(2, min(final_n_components, max_components))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(valid_data)

    model = PCA(n_components=final_n_components)
    scores_array = model.fit_transform(scaled_data)

    scores = pd.DataFrame(
        scores_array,
        columns=[
            f"PC{index}"
            for index in range(1, final_n_components + 1)
        ],
    )

    return PCAResult(
        model=model,
        scaler=scaler,
        scores=scores,
        feature_columns=feature_columns,
        valid_mask=valid_mask,
        explained_variance_ratio=model.explained_variance_ratio_,
    )
###########################k-means#############################
@dataclass
class KMeansResult:
    """
    Container for K-Means clustering output.

    Attributes
    ----------
    labels:
        Cluster label assigned to each sample.
    model:
        Fitted KMeans model.
    centroids:
        Cluster centroids.
    inertia:
        Sum of squared distances of samples to their closest cluster center.
    """
    labels: np.ndarray
    model: KMeans
    centroids: np.ndarray
    inertia: float


def validate_clustering_input(data: np.ndarray, n_clusters: int) -> None:
    """
    Validate input data and number of clusters for K-Means.

    Parameters
    ----------
    data:
        Input numeric matrix with shape (n_samples, n_features).
    n_clusters:
        Number of clusters requested.

    Raises
    ------
    ValueError
        If the input data or number of clusters is invalid.
    """
    if data is None:
        raise ValueError("Input data cannot be None.")

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D matrix.")

    n_samples = data.shape[0]

    if n_samples < 2:
        raise ValueError("At least two samples are required for clustering.")

    if n_clusters < 2:
        raise ValueError("The number of clusters must be at least 2.")

    if n_clusters > n_samples:
        raise ValueError(
            "The number of clusters cannot be greater than the number of samples."
        )

    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or infinite values.")


def run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    random_state: int = 0,
) -> KMeansResult:
    """
    Run K-Means clustering on a numeric matrix.

    Parameters
    ----------
    data:
        Input numeric matrix with shape (n_samples, n_features).
    n_clusters:
        Number of clusters.
    n_init:
        Number of K-Means initializations.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    KMeansResult
        Structured clustering result.
    """
    data = np.asarray(data)

    validate_clustering_input(data, n_clusters)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )

    labels = model.fit_predict(data)

    return KMeansResult(
        labels=labels,
        model=model,
        centroids=model.cluster_centers_,
        inertia=float(model.inertia_),
    )

##############################################################
def threshold(
    img: Image.Image,
    r_min: int, r_max: int,
    g_min: int, g_max: int,
    b_min: int, b_max: int
) -> Image.Image:
    """
    RGB threshold with lower and upper limits for each channel.
    White pixel if the pixel is within all 3 ranges, otherwise black.

    Returns: Binary PIL Image mode ‘L’ (0/255).
    """
    # Range validation
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
    Extracts geometric features from a binary mask.

    Returned features:
    [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter, hu1..hu7]
    """
    # PIL -> numpy grayscale uint8
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)

    # Binarization  (0/255)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Area (white pixels)
    area = float(np.count_nonzero(mask_bin))

    # If there is no object, it returns zero
    if area == 0:
        return np.zeros(14, dtype=float)  # 7 base + 7 Hu

    # Bounding box without outlines
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

    # Solidity and Hu Moments need a side contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(contorno)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Equivalent diameter
    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))

    # Hu moments
    moments = cv2.moments(contorno)
    hu = cv2.HuMoments(moments).flatten().astype(float)
    hu = np.sign(hu) * np.log(np.abs(hu) + 1e-12)
    # Final features
    features_base = np.array(
        [height, width, area, aspect_ratio, extent, solidity, equivalent_diameter],
        dtype=float)
    features = np.concatenate([features_base, hu])
    return features

def estrazione_feature_texturali(img, img_binary, raggio=3, punti=4):
    """
    This function will return a vector containing the textural features
    for each channel of the image.
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
    This function calculates the complete vector of features
    and returns the names corresponding to each feature in parallel, in order to have
    an interpretable and traceable representation of the columns.

    The textural component consists of:
    - Haralick (with added pixel mean and variance) for each channel
    - LBP for each channel (with names generated dynamically based on length)

       :param img: Image (RGB). Input image from which to extract features.
    :param img_binary: Mask of the image. Binary mask defining the ROI.
    :param geometric_feature_names: list/array of strings containing the names of geometric features.
    :param haralick_feature_names_channels: list/array of strings containing the names of the Haralick features (already expanded by channel).
    :param channel_names: list of strings with the names of the channels.
    :param radius: parameter of lbp. Radius used for the calculation of LBP.
    :param points: parameter of lbp. Number of points used for the calculation of LBP.
    :return:
        - features: 1D ndarray with all features concatenated (geometric + haralick + lbp)
        - names: ndarray/list with names corresponding to features (same order and same length)
    """
    # Extraction of textural features (divided into Haralick and LBP)
    haralick, lib = estrazione_feature_texturali(img, img_binary, raggio=raggio, punti=punti)
    # Concatenation of textural features into a single vector
    feature_testurali = np.concatenate([haralick, lib])
    # Final concatenation: geometric features + textural features
    features = np.concatenate([estrazzione_features_geometriche(img_binary), feature_testurali])
    # Constructing LBP feature names:
    # lib contains the concatenated LBPs from all 3 channels, so the length per channel is len(lib)/3
    nomi_lib = []
    len_lib = len(lib) / 3
    # For each channel, I create the names LBP_1_<channel>, LBP_2_<channel>, ...
    for nome_canale in nomi_canali:
        for i in range(int(len_lib)):
            nomi_lib.append(f"LBP_{i+1}_{nome_canale}")
    # Final concatenation of names: geometric + haralick (already prepared) + lbp (generated here)
    nomi = np.concatenate([nomi_features_geometriche, nomi_features_haralick_canali, nomi_lib])
    return features, nomi

def directory_immagini_to_csv(directory_path: str, recursive: bool = True, csv_name: str = "paths_immagini.csv") -> pd.DataFrame:
    """
    Create a DataFrame with a single column ‘path’ containing the paths of the images
    present in a directory and save it as CSV in the same directory.

    Args:
        directory_path: path of the directory to scan
        recursive: if True, also scans subfolders
        csv_name: name of the CSV file to save in the directory

    Returns:
        df: DataFrame with ‘path’ column
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory non trovata: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Non è una directory: {dir_path}")

    # Common image file extensions (add more if needed)
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

def run_kmeans_vars(
    X,
    *,
    k: int,
    n_init: int = 10,
    random_state: int = 0,
    min_samples: int = 2,
):
    """
    KMeans standalone: returns only variables.

    Input:
      - X: array-like (n_samples, n_features) oppure DataFrame
      - k: numero cluster

    Returns:
      ok (bool),
      msg (str),
      labels (np.ndarray|None),
      km (KMeans|None),
      Xc (np.ndarray|None)
    """
    if X is None:
        return (False, "Input X is None.", None, None, None)

    # convert to NumPy
    Xc = np.asarray(X)

    if Xc.ndim != 2:
        return (False, "X deve essere 2D (n_samples, n_features).", None, None, None)

    n_samples = Xc.shape[0]
    if n_samples < min_samples:
        return (False, "Non ho abbastanza campioni per fare clustering.", None, None, Xc)

    k = int(k)
    if k < 2:
        return (False, "k deve essere >= 2.", None, None, Xc)

    if k > n_samples:
        return (False, "k troppo grande rispetto al numero di campioni.", None, None, Xc)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(Xc)

    return (True, "OK", labels, km, Xc)
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