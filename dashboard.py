import streamlit as st
import requests

st.write("""
# Modèle de scoring""")


""""""

client_list = requests.get('http://127.0.0.1:3000/get_clients_id').json()['data']
sorted_client_list = sorted(client_list)
selected_client = st.sidebar.selectbox('Clients_id', sorted_client_list)


url = 'http://127.0.0.1:3000/predict'
client_id = str(selected_client)
x = requests.post(url, data = client_id)
st.write(x.text)