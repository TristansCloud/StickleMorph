import cv2
import dlib
import glob
import numpy as np
import os
import pandas as pd
import PIL
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_javascript import st_javascript
import time



PAGE_CONFIG = {"page_title": "StickleMorph", "page_icon": ":o", "layout": "wide"}
st.set_page_config(**PAGE_CONFIG)

def clear_cache():
    st.session_state.landmarks = False
    st.session_state.initial = []

def reorder_landmarks(landmarks):
    parts = map(str, range(0, len(landmarks)))
    ids = sorted(list(parts))
    num = [int(i) for i in ids]
    new_landmarks = [x for _, x in sorted(zip(num, landmarks))]
    return new_landmarks

def format_labels(predictor):
    """Format predictor labels."""
    return predictor.split("/")[-1].split(".")[0]
    
def download_csv(session, filter):
    """Download landmarks data as a CSV."""
    if session['scaled_landmarks'] is False:
        session['scaled_landmarks'] = session['landmarks']
    df = pd.DataFrame(session['scaled_landmarks'], columns=['x', 'y'])
    if filter:
        df = df.drop(index=filter)
    return df.to_csv(index=True).encode("utf-8")

def landmarks_to_fabric(landmarks, point_color, text_color):
    """Convert landmarks to fabric circles."""
    circles_and_numbers = []
    for i, (x, y) in enumerate(landmarks):
        circles_and_numbers.append({
            "type": "circle",
            "left": x - 3,
            "top": y - 3,
            "radius": 3,
            "fill": point_color,
            "selectable": True,
        })
        circles_and_numbers.append({
            "type": "text",
            "text": str(i),
            "left": x + 5,
            "top": y + 3,
            "fontSize": 16,
            "fill": text_color,
            "selectable": False,
        })

    return {"objects": circles_and_numbers, "background": "rgba(0, 0, 0, 0)"}

# Set up session state
if 'landmarks' not in st.session_state:
    st.session_state.landmarks = False
if 'initial' not in st.session_state:
    st.session_state.initial = []

inner_width = st_javascript("""window.innerWidth""")
# Load shape predictor models
predictors = glob.glob("predictors/*.dat")

# Sidebar
st.sidebar.image("resources/logo_v4.png", use_column_width=True)

st.sidebar.markdown("## Predictors")
selected_model = st.sidebar.selectbox("Choose a predictor model", options=predictors, format_func=format_labels, on_change=clear_cache)

st.sidebar.markdown("## Image Dimensions")
maximum = st.sidebar.slider("Maximum width", min_value=200, max_value=2000, value=1000, step=50)

# st.sidebar.markdown("## Set scale")
# scale = st.sidebar.checkbox("Set image scale using two points", value=True)
scale = True

st.sidebar.markdown("## Edit Landmarks")
# edit = st.sidebar.radio("Choose:", options=["Editable", "Locked"], label_visibility="collapsed")
edit = "Editable"
cola, colb = st.sidebar.columns(2)
submit = cola.button("Submit Edits")
clear = colb.button("Clear Edits")

st.sidebar.markdown("## Filter Output")
filter = st.sidebar.multiselect("Choose landmarks to remove from output", options=range(0, 68))


st.sidebar.markdown("## Color")
cola,colb = st.sidebar.columns(2)
landmark_color = cola.color_picker("Landmark: ", "#00ff00")
stroke_color = colb.color_picker("Text: ", "#ffffff")

# Main area

image_path = st.text_input("Path to images", value = "./images")
st.sidebar.markdown("## Images")
all_images = os.listdir(image_path)
selected_image = st.sidebar.selectbox("Choose and image.", options = all_images, format_func=format_labels, on_change=clear_cache)

uploaded_file = st.image(os.path.join(image_path, selected_image))
# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image", on_change = clear_cache) #, accept_multiple_files=True

if uploaded_file is not None:
    st.write(f"{uploaded_file}")
    # img_name = uploaded_file.name
    # img_name = img_name.split(".")

    # Main area
    if inner_width < maximum:
        maximum = inner_width

    img_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ratio = maximum / w if w != 0 else 1

    rect = dlib.rectangle(1, 1, int(image.shape[1]) - 1, int(image.shape[0]) - 1)
    predictor = dlib.shape_predictor(selected_model)
    shape = predictor(image,rect)
    landmarks = [(point.x, point.y) for point in shape.parts()]
    landmarks = reorder_landmarks(landmarks)
    st.session_state.initial = landmarks.copy()

    number = st.sidebar.number_input("Length (mm)", 0, 2000, 15, 5)
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ["point"])
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FF0000")
    image = cv2.resize(image, (int(w*ratio), int(h*ratio)))
    canvas_result_scale = st_canvas(
        stroke_width=3,
        stroke_color=stroke_color,
        background_image=PIL.Image.fromarray(image),
        update_streamlit=False,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode=drawing_mode,
        key="canvas_scale",
    )
    if canvas_result_scale.json_data is not None:
        objects = pd.json_normalize(canvas_result_scale.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        if objects.empty is False:
            if len(objects.left) == 2:
                length = (
                    (
                        (objects.left[0] - objects.left[1]) ** 2
                        + (objects.top[0] - objects.top[1]) ** 2
                    )
                    ** 0.5
                ) * 1/ratio

                scale_length = round(length/number, 2)
                st.write(
                    f"Success! Your scale is {scale_length} pixels per mm."
                )

                    # Create a canvas to draw and update landmarks
                if edit == 'Locked':

                    st.write(f"Please enable editing landmarks.")
                    
                else:
                    if clear:
                        st.session_state.landmarks = st.session_state.initial
                    if st.session_state['landmarks'] is not False:
                        landmarks = st.session_state['landmarks']

                    
                    landmarks = [(int(x*ratio), int(y*ratio)) for x, y in landmarks]
                    image = cv2.resize(image, (int(w*ratio), int(h*ratio)))

                    canvas_result = st_canvas(
                        background_image=PIL.Image.fromarray(image),
                        stroke_width=3,
                        stroke_color= stroke_color,
                        background_color="rgba(0, 0, 0, 0)",
                        update_streamlit=True,
                        height=image.shape[0],
                        width=image.shape[1],
                        drawing_mode=['transform'],
                        initial_drawing=landmarks_to_fabric(landmarks, landmark_color, stroke_color),
                        key="canvas",
                    )

                    # time.sleep(1)
                    objects = pd.json_normalize(canvas_result.json_data["objects"])

                    circles = objects[objects['type'] == 'circle']
                    texts = objects[objects['type'] == 'text']

                    # Get the text objects immediately below each circle object
                    circles.loc[:, 'landmark_id'] = texts['text'].values

                    # Create a new dataframe with the required columns
                    final_df = circles[['landmark_id', 'left', 'top']]
                    final_df.columns = ['landmark_id', 'x', 'y']
                    final_df.landmark_id = final_df.landmark_id.astype(int)

                    if submit:
                        final_df['x'] = final_df['x']
                        final_df['y'] = final_df['y']
                    
                    st.session_state.scaled_landmarks = [(round((x*1/ratio)/scale_length,5), round((y*1/ratio)/scale_length,5)) for x, y in final_df[['x', 'y']].values]
                    st.sidebar.markdown("## Download")

                    st.sidebar.download_button(
                        label="Download coordinates(CSV)",
                        data=download_csv(st.session_state, filter),
                        file_name= selected_image + ".csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.write(
                    f"Please select only two points. Use backward arrow to delete points."
                )
        else:
            st.write(
                f"Please select two points: one at the beginning and one at the end of the scale. Then press the leftmost button to submit to the app."
            )


