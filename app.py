# importing packages
import streamlit as st
from sklearn.pipeline import Pipeline
from keras.models import load_model
from sklearn.preprocessing import FunctionTransformer

# Importing custom function and class
from functions import preprocess_image, TrainedModelTransformer

# calling our class
custom_function = FunctionTransformer(preprocess_image)

# loading our model
model = load_model("sports_classification_model.h5")

# creating our image preprocessing and model pipeline
model_pipeline = Pipeline([
    ('preprocessing image', custom_function),
    ('keras-model', TrainedModelTransformer(model))
])


# class names
class_names = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
               'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
               'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
               'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping', 'horse racing',
               'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
               'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse',
               'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby', 'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing',
               'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze',
               'tug of war', 'ultimate', 'uneven bars', 'volleyball', 'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing', 'wingsuit flying']


## APP ##

st.title("Sports Image Classifier")
st.markdown("by: Jeremiah Chinyelugo")

st.write(" ")
st.write(" ")

# "=============================================================================="

# Introduction
st.write("""This app classifies sports images.
         Simply upload an image of a sport and it will classify it. It's the simple!
         """)

st.write("""The model: \n
         The model used in this app was trained using the base layer of a ResNet-50 model from tensorhub.
         """)

st.write(" ")
st.write(" ")


file = st.file_uploader("Upload your .jpg image", [
                        'jpg'], accept_multiple_files=False)

if file:
    st.image(file, caption="Image you uploaded")

    prediction = model_pipeline.transform(file)

    st.write(f"You uploaded an image of:")
    st.subheader(class_names[prediction].title())
