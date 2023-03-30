import streamlit as st
import requests

st.write("""
# Modèle de scoring""")


""""""

client_list = requests.get('http://127.0.0.1:3000/get_clients_id').json()['data']
print(client_list)
sorted_client_list = sorted(client_list)
selected_client = st.sidebar.selectbox('Clients_id', sorted_client_list)