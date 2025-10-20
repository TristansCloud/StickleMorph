# StickleMorph

StickleMorph is a Streamlit web application that allows users to predict, visualize and edit landmarks on images of three-spined stickleback fish. Users can upload their own images, select a shape predictor model, and interact with the canvas to update landmark coordinates. The application provides a simple and user-friendly interface.

## Features
- Upload images in PNG, JPG, or JPEG format.
- Visualize and edit landmark coordinates on images using an interactive canvas.
- Export landmark coordinates to a TPS file

## Notes
- There is a [pretrained predictor and example images](https://drive.google.com/drive/folders/1xho5l4bL07By11o5lE6RD4I_IxUXtyRP?usp=sharing) available for download. Place the predictor.dat file in the predictors folder and the images in the images folder.
- This app uses 0 indexed landmarks. The companion paper describes landmarks as 1 indexed. If you are referencing the companion paper while using this app, subtract 1 from the paper's landmarks to convert to the app's landmarks. For example, landmarks 1, 2, and 3 in the paper are landmarks 0, 1, and 2 in the app respectively.

## Installation

Clone this repository:

```bash
git clone https://github.com/TristansCloud/StickleMorph.git
cd StickleMorph
```
Create and activate a virtual environment (optional, but recommended):
```bash
conda create -y -n stmorph python=3.7
conda activate stmorph
```
Install the required packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```
If you are installing on windows, the dlib requirement will probably fail to install. You can install dlib with conda:
```bash
conda install -c conda-forge dlib
```
Then double check all your requirements are installed:
```bash
pip install -r requirements.txt
```
## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open a web browser and visit the URL displayed in the terminal (usually http://localhost:8501).

Upload an image using the file uploader, select a shape predictor model, and interact with the canvas to update landmark coordinates.

View and export the updated landmark coordinates as needed.
