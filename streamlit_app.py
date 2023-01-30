import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, time
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

dict_encoding = { 
    "0:0" : 1,
    "0:30" : 2,
    "1:0" : 3,
    "1:30" : 4,
    "2:0" : 5,
    "2:30" : 6,
    "3:0" : 7,
    "3:30" : 8,
    "4:0" : 9,
    "4:30" : 10,
    "5:0" : 11,
    "5:30" : 12,
    "6:0" : 13,
    "6:30" : 14,
    "7:0" : 15,
    "7:30" : 16,
    "8:0" : 17,
    "8:30" : 18,
    "9:0" : 19,
    "9:30" : 20,
    "10:0" : 21,
    "10:30" : 22,
    "11:0" : 23, 
    "11:30" : 24,
    "12:0" : 25, 
    "12:30" : 26, 
    "13:0" : 27, 
    "13:30" : 28, 
    "14:0" : 29, 
    "14:30" : 30, 
    "15:0" : 31, 
    "15:30" : 32, 
    "16:0" : 33, 
    "16:30" : 34, 
    "17:0" : 35, 
    "17:30" : 36, 
    "18:0" : 37, 
    "18:30" : 38, 
    "19:0" : 39, 
    "19:30" : 40, 
    "20:0" : 41, 
    "20:30" : 42, 
    "21:0" : 43, 
    "21:30" : 44, 
    "22:0" : 45, 
    "22:30" : 46, 
    "23:0" : 47, 
    "23:30" : 48
}

equipements = pd.read_csv('data/dataOPTI.csv',index_col="Type")

#load le scaler
import pickle
scaler = pickle.load(open('models/scalerMM.pkl', 'rb'))

model2 = tf.keras.models.load_model('models/model2.h5')

# The code below is for the title and logo for this page.
st.set_page_config(page_title="SPIE project", page_icon="🧙")

#For optimization
consoOPTI = pd.read_csv('conso/res_opti_alajulio.csv')
consoOPTI = consoOPTI.transpose().drop('Unnamed: 0')
consoOPTI = consoOPTI.round()
consoOPTI.rename(columns={0:'LV',1:'LL',2:'SL'}, inplace=True)


st.image(
    "spie.png",
    width=160,
)

st.title("`SPIE project` 🧙 ")

st.write("")

st.markdown(
    """
    By Hugo COEUILLET, Victor FEUGA, Alexandre LAGARRUE, Jules LEFEBVRE, Baptiste VALENTIN
"""
)
res_radio={
    "Oui":1,
    "Non":0
}

st.write("")

with st.form("prediction"):
    st.header("Prédire votre consomation 🔋")

    date_value = st.date_input("Pour quel jour ?", dt.date(2023, 1, 1), min_value=dt.date(2022, 12, 1))
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

    submitted = st.form_submit_button("Prévoir votre consommation")


    if submitted:


        val = np.array([dict_encoding[str(time_value.hour)+":"+str(time_value.minute)],
                        date_value.day,
                        date_value.month,
                        surface,
                        nb_hab,
                        appartement,
                        maison])

        val = val.reshape(1,7)

        val = scaler.transform(val)

        val = model2.predict(val)
        val = str(round(float(val),3))+" kW"
        st.metric(label="Prévision 🔎", value=val)


with st.form("optimisation"):
    st.header("Optimiser votre consommation 🧮")
    st.markdown(
        """
        _Il est important de penser à programmer les appareils qui le peuvent afin de réduire la consommation totale sur le réseau_ 🤗
    """
    )

    id_h = st.text_input('Quel est votre identifiant ?', 'A100-3-1')

    submitted = st.form_submit_button("Voir votre planning personnalisé")
    
    if submitted:

        LV = str(int(consoOPTI['LV'].loc[[id_h]].values))
        
        LL = str(int(consoOPTI['LL'].loc[[id_h]].values))

        SL = str(int(consoOPTI['SL'].loc[[id_h]].values))

        col1, col2, col3 = st.columns(3)

        with col1:
            if equipements['LV'].loc[id_h] == 1:
                st.metric(label="Lancez votre lave-vaisselle à ", value=LV +"h")
            else :
                st.metric(label="Lancez votre lave-vaisselle à", value="❌")

        with col2:
            if equipements['LL'].loc[id_h] == 1:
                st.metric(label="Lancez votre lave-linge à", value=LL +"h")
            else :
                st.metric(label="Lancez votre lave-linge à", value="❌")


        with col3:
            if equipements['SL'].loc[id_h] == 1:
                st.metric(label="Lancez votre sèche-linge à", value=SL +"h")
            else :
                st.metric(label="Lancez votre sèche-linge à", value="❌")

        st.markdown(
            """
            _Ces petits gestes permettent de réduire la sur consommation du réseau_ 🌳
        """
        )


with st.expander("Graphique optimisation"):

    st.write("")

    st.markdown(
        """
    Voici une illustration de nos optimisations
    """
    )

    consoDF = pd.read_csv('conso/consoDF.csv')
    consoDFmelt = consoDF.melt(ignore_index=False)

    fig = px.line(consoDFmelt, x="variable", y="value", color=consoDFmelt.index)

    fig.update_layout(
        title="Consommation journalière moyenne par habitation",
        xaxis_title="Temps",
        yaxis_title="kW",
        legend_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

    series_names = ["Sans LV LL SL","LV LL SL moyennés sur la journée","LV LL SL distribution logique", "Optimisation procédurale", "Optimisation non-linéaire"]

    for idx, name in enumerate(series_names):
        fig.data[idx].name = name
        fig.data[idx].hovertemplate = name
        
    st.plotly_chart(fig, theme=None,use_container_width=True)



with st.form("prediction_réseau"):
    
    st.header("Prédire la consommation sur l'ensemble du réseau 🧭")
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
        pred[i] = model2.predict(pred[i])
    
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pred[i][j] = pred[i][j] * nb_houses[j][1]
    
    for i in range(len(pred)):
        pred[i] = sum(pred[i])
        pred[i] = pred[i][0]
        
    timestamp = ["0:00", "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30"]

    
    submitted = st.form_submit_button("Prédire la consommation générale")
    
    if submitted:
        fig = px.line(x=timestamp, y=pred, title=f"Prédiction de la consommation par TOP30 pour la journée du {day}/{month} sur l'ensemble du réseau ({nb_foyers} foyers)", markers=True)
        fig.update_xaxes(title_text="TOP30")
        fig.update_yaxes(title_text="Consommation (kW)")
        fig.update_layout(legend_title_text='TOP30')
        fig.update_layout(
            title=f"Prédiction de la consommation par TOP30 pour la journée du {day}/{month}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig)
        
        st.metric(label=f"Prédiction de la consommation totale pour la journée du {day}/{month}", value=str(round(sum(pred),2))+" kW")
