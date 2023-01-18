import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, time
import plotly.express as px


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
