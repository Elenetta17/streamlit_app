import pandas as pd
import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pickle


X_test_reduced = pd.read_csv('../X_test_reduced.csv', index_col=0) 

st.write("""
# Modèle de scoring""")


""""""

client_list = requests.get('http://127.0.0.1:3000/get_clients_id').json()['data']
sorted_client_list = sorted(client_list)
selected_client = st.sidebar.selectbox('Clients_id', sorted_client_list)


url = 'http://127.0.0.1:3000/predict'
client_id = str(selected_client)
prediction = requests.post(url, data = client_id)
print(prediction)
st.write("""
## Probabilité de remboursement: """ + prediction.text + " %")

st.write("""
## Explication de la prediction:""")
explainer = pickle.load(open('../explainer.pkl', 'rb'))
shap_values = explainer.shap_values(X_test_reduced)
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