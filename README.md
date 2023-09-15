# StickleMorph

StickleMorph is a Streamlit web application that allows users to predict, visualize and edit landmarks on images of three-spined stickleback fish. Users can upload their own images, select a shape predictor model, and interact with the canvas to update landmark coordinates. The application provides a simple and user-friendly interface.

## Features
- Upload images in PNG, JPG, or JPEG format.
- Visualize and edit landmark coordinates on images using an interactive canvas.
- Export landmark coordinates to a TPS file (not implemented yet)

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

A pretrained predictor and some example images are available at [this link.](https://drive.google.com/drive/folders/1zZofiIedbJbsMdgw2adOPyJuSJWei7vH?usp=sharing) Place the predictor.dat file in the predictors folder.

View and export the updated landmark coordinates as needed.