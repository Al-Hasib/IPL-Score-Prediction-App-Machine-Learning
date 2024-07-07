import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


data = pd.read_csv(r"C:\Users\abdullah\projects\Machine_Learning_projects\IPL-Score-Prediction-App-Machine-Learning\data\ipl_data.csv")

def get_options(feature_name):
    values = list(data[feature_name].unique())
    return values

# Title of the app
st.write("<h1 style='text-align: center;'>IPL Score Prediction App</h1>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.write("<h4 style='text-align: center;'>User Input For Predicting the Total Score</h4>", unsafe_allow_html=True)

# Display an image
st.image('ipl1.png', caption='IPL League', use_column_width=True)

venue = st.sidebar.selectbox('VENUE', options=get_options('venue'))
bat_team = st.sidebar.selectbox('BAT TEAM', options=get_options('bat_team'))
bowl_team = st.sidebar.selectbox('BOWL TEAM', options=get_options('bowl_team'))
batsman = st.sidebar.selectbox('BATSMAN', options=get_options('batsman'))
bowler = st.sidebar.selectbox('BOWLER', options=get_options('bowler'))

runs = int(st.sidebar.text_input('RUNS IN CURRENT OVER','0'))

wickets = int(st.sidebar.text_input('WICKETS IN CURRENT OVER','0'))

overs = float(st.sidebar.text_input('OVERS','0.0'))

runs_last_5 = int(st.sidebar.text_input('RUNS IN LAST 5 OVERS','0'))


wickets_last_5 = int(st.sidebar.text_input('WICKETS IN LAST 5 OVERS','0'))



customdata_object = CustomData(venue,bat_team,bowl_team,batsman,bowler, runs, wickets, overs, runs_last_5, wickets_last_5)

dataframe = customdata_object.get_data_as_dataframe()

prediction_object = PredictPipeline()

prediction = prediction_object.predict(dataframe)

if st.button('Predict'):
    st.header(f"The Prediction of Total Score : {int(prediction[0])}")








