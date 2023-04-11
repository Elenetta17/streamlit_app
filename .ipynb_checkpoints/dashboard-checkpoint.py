import pandas as pd
import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pickle
import plotly.graph_objects as go


X_test_reduced = pd.read_csv('../X_test_reduced.csv', index_col=0) 
#explainer = pickle.load(open('../explainer.pkl', 'rb'))
#shap_values = explainer.shap_values(X_test_reduced)

@st.cache_data  # 👈 Add the caching decorator
def load_shap_values(path, df):
    explainer = pickle.load(open(path, 'rb'))
    shap_values = explainer.shap_values(df)
    return explainer, shap_values

explainer, shap_values = load_shap_values('../explainer.pkl', X_test_reduced)

st.write("""
# Dashboard Scoring Credit""")


""""""


# création liste clients
client_list = sorted(X_test_reduced.index)
selected_client = st.sidebar.selectbox('Clients_id', client_list)


url = 'http://127.0.0.1:3000/predict'
client_id = str(selected_client)
prediction = requests.post(url, data = client_id)
print(prediction)
st.write("""
## Prédiction """)
col1, col2 = st.columns([3, 1])

"""Prédictions"""

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = int(prediction.text),
    number= { 'suffix': "%", 'font':{'color':'dimgrey'}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Probabilité de remboursement", 'font':{'color':'dimgrey' }},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "dimgrey", 'tickfont':{'color':'dimgrey' }},
        'bar': {'color': "dimgrey"},
        'steps': [
            {'range': [0, 45], 'color': 'red'},
            {'range': [45, 100], 'color': 'green'}],
    }))
fig.update_layout(
    margin=dict(l=30, t=50, r=30, b=0),
    height=250,
    width=300
    #margin={'b': 70},
    #height=150,
)
col1.plotly_chart(fig)

col2.header("""Client """ + client_id)



st.write("""
## Explication de la prediction:""")

shap_object = shap.Explanation(base_values = explainer.expected_value[0],
values = shap_values[0],
feature_names = X_test_reduced.columns,
data = X_test_reduced)
fig1, ax1 = plt.subplots(2,1, figsize=(15,20))
plt.subplot(2, 1 ,1)
shap.summary_plot(shap_values[0], X_test_reduced,  max_display=10, show=False)
plt.subplot(2, 1, 2)
shap.plots.waterfall(shap_object[X_test_reduced.index.get_loc(selected_client)], max_display=10)
plt.tight_layout()
st.pyplot(fig1)

# Store the list of columns
columns_to_plot = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'INSTAL_AMT_PAYMENT_SUM', 'DAYS_EMPLOYED',
                    'NAME_EDUCATION_TYPE_Higher education', 'INSTAL_DPD_MEAN','POS_COUNT', 'DAYS_BIRTH','AMT_CREDIT','AMT_ANNUITY'
                                 ]
st.write("""
## Positionnement du client par rapport à l'ensamble de clients """)
# boxplots
fig, ax = plt.subplots((len(columns_to_plot)+1)//2, 2, figsize=(15, 15))
count = 1
for col in columns_to_plot:
    plt.subplot((len(columns_to_plot)+1)//2, 2, count)
    plt.title(col)
    sns.boxplot(data=X_test_reduced, x=col, showfliers=False)
    plt.scatter(X_test_reduced.loc[selected_client, col],0, linewidths=5, c='red')
    plt.text(X_test_reduced.loc[selected_client, col],-0.1, "CLIENT")
    count += 1
plt.tight_layout()
st.pyplot(fig)