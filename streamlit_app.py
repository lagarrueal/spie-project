import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, time
import plotly.express as px
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

model = tf.keras.models.load_model('./models/model2.h5')
scaler = pickle.load(open('./models/scaler_min_max.pkl', 'rb'))

# The code below is for the title and logo for this page.
st.set_page_config(page_title="SPIE project", page_icon="🧙")

st.image(
    "spie.png",
    width=160,
)

st.title("`SPIE project` 🧙 ")

st.write("")

st.markdown(
    """
    Alexandre LAGARRUE, Jules LEFEBVRE, Hugo COEUILLET, Baptiste VALENTIN, Victor FEUGA
"""
)

with st.expander("About this app"):

    st.write("")

    st.markdown(
        """
    A compléter :)
    """
    )

    consoDF = pd.read_csv('conso/consoDF.csv')
    consoDFmelt = consoDF.melt(ignore_index=False)

    fig = px.line(consoDFmelt, x="variable", y="value", color=consoDFmelt.index)

    fig.update_layout(
        title="Averaged daily electrical comsuption per habitation",
        xaxis_title="Time",
        yaxis_title="kW",
        legend_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

    series_names = ["Withtout LV LL SL", "LV LL SL averaged", "LV LL SL logical"]

    for idx, name in enumerate(series_names):
        fig.data[idx].name = name
        fig.data[idx].hovertemplate = name
        
    st.plotly_chart(fig, theme=None,use_container_width=True)


res_radio={
    "Oui":1,
    "Non":0
}

st.write("")
with st.form("prediction"):
    st.header("Prédire votre consomation 🔋")

    date_value = st.date_input("Pour quel jour ?", dt.date(2023, 1, 1), min_value=dt.date(2023, 1, 1))
    time_value = st.time_input("Pour quelle heure ?", dt.time(12, 0))

    if time_value.minute/15 in [1,3]:
        # Convert the time object to a datetime object
        datetime_time = datetime.combine(datetime.now().date(), time_value)

        # Subtract 15 minutes from the datetime object
        datetime_time = datetime_time - timedelta(minutes=15)

        # Convert the datetime object back to a time object
        time_value = datetime_time.time()

    options = ["Oui", "Non"]
    default_index = 1  # index of option 2
    appartement = st.radio("Vivez-vous dans un appartement ?", options, default_index)
    appartement = res_radio[appartement]

    maison = 0
    if appartement == 0:
        maison = 1

    surface = st.select_slider('Quelle est la taille de votre habitation (en m²) ?', 
                                options=[15, 25, 30, 50, 65, 80, 85, 90, 100, 110, 120, 130, 135, 140, 150, 160, 170, 180, 200, 250])

    nb_hab = st.select_slider("Combien y a-t-il d'occupants ?", 
                            options=[1,2,3,4,5,6])

    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write(date_value, time_value, appartement, maison, surface,nb_hab)
        st.metric(label="Prévision 🔎", value="XXX kW")

# input a date, extract the day and month, and do the prediction for the whole grid for this day
# display the graph of the prediction for the whole grid for this day per TOP30
# display the bar chart of the prediction for the whole grid for this day per TOP30
# display the bar chart prediction for the whole grid for this day 
with st.form("prediction_réseau"):
    
    st.header("Prédire la consommation par TOP30 sur l'ensemble du réseau pour la journée 🔋")
    date_value = st.date_input("Pour quel jour ?", dt.date(2023, 1, 1), min_value=dt.date(2022, 1, 1))
    
    day = date_value.day
    month = date_value.month
    
    list_housing = os.listdir('./data/housing_data/')
    nb_houses = []
    for housing in list_housing:
        nb_houses.append((housing, len(os.listdir('./data/housing_data/' + housing))))
        
    nb_foyers = 0
    for i in range(len(nb_houses)):
        nb_foyers += nb_houses[i][1]
        
    pred = []
    for i in range(48):
        pred.append([])
        
    for i in range(len(pred)):
        pred[i].append([i, day, month, 110, 5, 1, 0])
        pred[i].append([i, day, month, 170, 6, 0, 1])
        pred[i].append([i, day, month, 120, 5, 0, 1])
        pred[i].append([i, day, month, 150, 6, 1, 0])
        pred[i].append([i, day, month, 135, 3, 0, 1])
        pred[i].append([i, day, month, 150, 4, 0, 1])
        pred[i].append([i, day, month, 100, 3, 0, 1])
        pred[i].append([i, day, month, 130, 4, 1, 0])
        pred[i].append([i, day, month, 140, 5, 0, 1])
        pred[i].append([i, day, month, 250, 5, 0, 1])
        pred[i].append([i, day, month, 80, 2, 1, 0])
        pred[i].append([i, day, month, 50, 3, 1, 0])
        pred[i].append([i, day, month, 90, 4, 0, 1])
        pred[i].append([i, day, month, 95, 3, 0, 1])
        pred[i].append([i, day, month, 100, 3, 1, 0])
        pred[i].append([i, day, month, 120, 4, 1, 0])
        pred[i].append([i, day, month, 30, 2, 1, 0])
        pred[i].append([i, day, month, 200, 6, 0, 1])
        pred[i].append([i, day, month, 85, 3, 0, 1])
        pred[i].append([i, day, month, 160, 5, 0, 1])
        pred[i].append([i, day, month, 50, 2, 0, 1])
        pred[i].append([i, day, month, 25, 1, 1, 0])
        pred[i].append([i, day, month, 110, 4, 0, 1])
        pred[i].append([i, day, month, 180, 5, 0, 1])
        pred[i].append([i, day, month, 15, 1, 1, 0])
        pred[i].append([i, day, month, 50, 2, 1, 0])
    
    for i in range(len(pred)):
        pred[i] = scaler.transform(pred[i])
        
    for i in range(len(pred)):
        pred[i] = model.predict(pred[i])
    
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pred[i][j] = pred[i][j] * nb_houses[j][1]
    
    for i in range(len(pred)):
        pred[i] = sum(pred[i])
        pred[i] = pred[i][0]
        
    timestamp = ["0:00", "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30"]


    
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        fig = px.line(x=timestamp, y=pred, title=f"Prédiction de la consommation par TOP30 pour la journée du {day}/{month} sur l'ensemble du réseau ({nb_foyers} foyers)", markers=True)
        fig.update_xaxes(title_text="TOP30")
        fig.update_yaxes(title_text="Consommation (kW)")
        fig.update_layout(legend_title_text='TOP30')
        st.plotly_chart(fig)
        
        x = [f'{day}/{month}']
        y = [round(sum(pred),3)]
        colors = ['#1f77b4']
        fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
            width=0.25,
            marker_color=colors
        )])
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Consommation journalière totale (kW)")
        fig.update_layout(title_text=f"Prédiction de la consommation pour la journée du {day}/{month} sur l'ensemble du réseau ({nb_foyers} foyers)")
        st.plotly_chart(fig)






with st.form("optimisation"):
    st.header("Optimiser votre consomation 🧮")

    options = ["Oui", "Non"]
    default_index = 1  # index of option 2


    col1, col2, col3 = st.columns(3)

    with col1:
        LV = st.radio("Avez-vous un lave-vaisselle ?", options, default_index)
        LV = res_radio[LV]

    with col2:
        LL = st.radio("Avez-vous un lave-linge ?", options, default_index)
        LL = res_radio[LL]

    with col3:
        SL = st.radio("Avez-vous un sèche-linge ?", options, default_index)
        SL = res_radio[SL]

    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write(LV, LL, SL)
        st.metric(label="Prévision 🔎", value="XXX kW")
