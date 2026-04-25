# Main structure and functionality of the application
This program implements a graphical interface in Tkinter for image analysis and visualization of extracted features.
The application allows you to load a dataset of images, view both the original image and the corresponding binary mask obtained using RGB thresholding for each element, and interactively modify the thresholding parameters. 
It also allows the extraction of geometric and textural features, the saving of calculated descriptors directly in the input CSV, and the automatic processing of the entire dataset.

The program also includes a section dedicated to exploratory data analysis, based on Principal Component Analysis (PCA) of the extracted features and the application of the K-Means algorithm to the PCA scores, in order to highlight any structures or groupings present in the data. 

Overall, the tool is designed to support the construction, organization, and initial quantitative analysis of datasets of features extracted from images.

## Feature extraction
As mentioned above, the program allows you to extract a set of features from images and save them directly to the CSV file associated with the dataset. The descriptors calculated belong to two main categories: **geometric features** and **textural features**. Both are calculated exclusively on the selected image region after applying the threshold, i.e., on the portion that remains within the binary mask obtained from the thresholding. The extracted features can be temporarily displayed within the interface and then saved in the CSV file for both the current single image and the entire dataset.
LThe extracted geometric features are numerous and describe both the size and shape of the segmented object (region contained in the binary mask). In particular, the descriptor vector includes:
- height: height of the bounding box;
- width: width of the bounding box;
- area: area of the segmented region;
- equivalent_diameter: equivalent diameter, i.e., the diameter of the circle with the same area as the region;
- aspect_ratio: ratio between the width and height of the bounding box
- extent: ratio between the area of the region and the area of the bounding box;
- solidity: ratio between the area of the region and the area of its convex hull, useful for quantifying concavity and irregularity;
Added to these are the 7 invariant moments of [Hu](https://it.wikipedia.org/wiki/Momento_(elaborazione_delle_immagini)) (hu1–hu7), which are descriptors of shape invariant to translation, rotation, and (in a first approximation) scale, and therefore allow for a more robust comparison between objects with different orientations.

Textural features describe the distribution of intensity levels, i.e., the “granularity” and heterogeneity of the segmented region.
In the program, they are calculated on the ROI defined by the binary mask and include:
- mean_color: mean intensity values in the ROI (per channel);
- variance_color: variance of intensity values in the ROI (per channel);
In addition to these, there are the features of [Haralick](https://mahotas.readthedocs.io/en/latest/) derived from the Gray-Level Co-occurrence Matrix (GLCM), most of which are calculated using the Mahotas library: Angular Second Moment (Energy), Contrast, Correlation, Variance, Inverse Difference Moment (Homogeneity), Sum Average, Sum Variance, Sum Entropy, Entropy, Difference Variance, Difference Entropy, Information Measure of Correlation 1, Information Measure of Correlation 2.
The texture descriptors also include features based on the Local Binary Pattern (LBP), calculated on the region of interest identified by the binary mask. The LBP allows the local microstructure of the image to be represented, describing patterns such as edges, uniform areas, and texture variations, depending on the radius and number of points selected by the user.

## PCA e clustering
Once the display window is open, **PCA** can be applied to the vector of previously extracted features. The user can choose the number of principal components to calculate and select which two components to represent in the graph, so that they can be viewed together in a two-dimensional plane. Subsequently, a clustering algorithm, specifically **K-Means**, can be applied to the scores obtained from PCA in order to divide the data into groups. The membership of each point to its cluster is then shown graphically using different colors.

# Usage
To use the application correctly, you need to have a properly structured CSV file containing the paths of the images to be analyzed.
An example of the folder structure and the corresponding CSV file is available in the [dataset_prova](./dataset_prova/) folder.
It is not necessary for all images to be contained in the same directory: it is sufficient for the CSV file to include the correct paths of all the images you want to proces
If all images are stored in a single folder, you can automatically generate this file using the `directory_images_to_csv` function. Simply call the function, providing the path to the folder containing the images as input, to obtain a CSV file compatible with the program.
```bash
from function import directory_immagini_to_csv

path = "your_own_path"
directory_immagini_to_csv(path)
```
Once the CSV file has been uploaded, you can apply the threshold to the current image; the result of the thresholding will be displayed as a binary mask in the panel on the left side of the interface. Next to the threshold parameters, there will also be two values dedicated to defining the parameters of a specific textural feature, the Local Binary Pattern.

The main top bar will contain the “Load CSV” button, which will allow you to navigate within the file system and select the input CSV file. In the central part of the same bar, there will be three buttons dedicated to feature management. The “Extract features” button will allow you to calculate and display the features related to the currently selected image. The “Salva feature” button allows you to save the features associated with the current image only to the CSV file. Finally, the “Save Feature Dataset” button extracts and saves the features for the entire set of images contained in the dataset.
The “Visualizza PCA” button is available on the right side of the interface.

#### View PCA
After clicking the button, a new window will open showing the PCA results. Using the “Asse X” and “Asse Y” menus, you can select which principal components to display in the graph. In addition, using the drop-down menu, you can change the number of PCA components to be calculated. Each time this value is updated, you will need to press the “Calcolare/Refresh” button again to reprocess the analysis and update the display.

By pressing the “K-means” button, the application will cluster the data, generating a number of clusters equal to the value selected in the “k-cluster” drop-down menu. The clusters obtained will then be represented in the graph below by different colors of dots, so that the membership of each observation to its group is immediately visible.

#  Using a Python interpreter (Recommended for Linux)
The program can be run directly from the source code using a Python interpreter. In this case, you must first download or clone the project to your machine and navigate to the main folder of the repository using the terminal.
```bash
git clone https://github.com/GiovanniGueltrini/ObjectFeatureExtractor.git
cd ObjectFeatureExtractor
```
Once the virtual environment has been created and activated, you can install all the necessary packages using the requirements.txt file by running the command:
```bash
pip install -r requirements.txt
```
Once the dependencies have been installed, the program can be started by running the main application file, for example:
```bash
python Dashboard.py
```
The graphical interface will then open and you can use the program by loading a CSV file compatible with the required structure. This mode is suitable for users who wish to examine, modify, or further develop the application's source code.

Compatibility with macOS has not been directly verified. The program is developed in Python with cross-platform libraries, so it may also be executable on macOS via a Python interpreter, but support is not currently guaranteed.


### GUI dependency (Tkinter)
This project uses **Tkinter** (Python standard library).  
- **Windows/macOS**: usually included with Python  
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`  
- Check: `python -c "import tkinter; print('tkinter OK')"`
  
# Use via executable file(solo Windows)
Alternatively, the program can be used via an .exe executable file, without the need to manually install Python or the libraries required by the project. 
In this case, simply download the dist.zip file from the Release section, extract its contents to a local folder, and start the program by double-clicking on the .exe file inside the extracted folder.

Once the graphical interface is open, the user can load the CSV file containing the paths of the images to be analyzed and use all the software's features as normal, including thresholding, feature extraction, descriptor saving, and PCA visualization with K-Means clustering.

However, it is important to note that the executable file does not replace the input data required for the program to function. Consequently, for correct use, it will still be necessary to have the CSV file and the images associated with it. In addition, the paths listed in the CSV file must also be valid on the computer where the program is running; otherwise, the images will not be able to be loaded correctly.

