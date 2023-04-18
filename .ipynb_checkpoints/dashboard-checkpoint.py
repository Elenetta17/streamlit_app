from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.graph_objects as go
import requests
import shap
import streamlit as st

variables=pd.read_csv('description_variables.csv', sep = '\t', index_col=0)

@st.cache_data # caching
def load_dataframe(path):
    """This function take as input the path(string) of a csv
    file and return the dataframe"""
    data = pd.read_csv(path, index_col=0)
    return data

@st.cache_data  # 👈 Add the caching decorator
def load_shap_values(path, df):
    """This function take as input the path(string) of a shap explainer
    and a dataframe and return the explainer and the correspondent shap
    values of the dataframe"""
    explainer = pickle.load(open(path, 'rb'))
    shap_values = explainer.shap_values(df)
    return explainer, shap_values

@st.cache_data # caching
def load_client_list(data):
    """This function take as input ta dataframe
    and return the index in ascending order"""
    return sorted(data.index)

# chargement dataframe
data = load_dataframe('kaggle_reduced.csv') 

# chargement explainer et shap values
explainer, shap_values = load_shap_values('explainer.pkl', data)

shap_object = shap.Explanation(base_values = explainer.expected_value[0],
values = shap_values[0],
feature_names = data.columns,
data = data)

# categorical variables
categorical_columns=[
    'PREV_NAME_YIELD_GROUP_high_MEAN',
    'NAME_EDUCATION_TYPE_Higher education',
    'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
    'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
    'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
    'PREV_NAME_CLIENT_TYPE_New_MEAN',
    'PREV_NAME_PORTFOLIO_POS_MEAN',
    'PREV_NAME_YIELD_GROUP_high_MEAN',
    'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
    'PREV_NAME_YIELD_GROUP_middle_MEAN',
    'PREV_PRODUCT_COMBINATION_Cash_MEAN']

# application title
st.write("""
# Dashboard Scoring Credit""")

### SIDEBAR
# add client list to sidebar
client_list = load_client_list(data)
selected_client = st.sidebar.selectbox('Clients_id', client_list)

# add features to sidebar
features_to_show = st.sidebar.multiselect('Variables', sorted(data.columns),
                                          default = [
                                              'AMT_CREDIT',
 'BURO_DAYS_CREDIT_ENDDATE_MAX',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'NAME_EDUCATION_TYPE_Higher education',
 'PAYMENT_RATE',
 'PREV_CNT_PAYMENT_MEAN',
 'PREV_NAME_YIELD_GROUP_high_MEAN'],
                                         max_selections=10)
#=======================================================================#
### PREDICTION ####
st.write("""
## Prédiction """)

# get prediction
url = 'http://127.0.0.1:3000/predict'
# url= 'https://elena-openclassrooms-predict.herokuapp.com/predict'
client_id = str(selected_client)
prediction = requests.post(url, data=client_id)

col1, col2 = st.columns(2)

# Showing client state and probability 
if int(prediction.text) > 50:
    client_state = 'Client peu risqué'
    approval = 'Accorder le crédit'
else:
    client_state = 'Client à risque de défaut'
    approval = 'Refuser le crédit'

col1.subheader("""Client """ + client_id)
col1.write("""Probabilité de remboursement: """ + prediction.text +"%")
col1.write("""Etat client: **""" + client_state + """**""")
col1.write("""**""" + approval + """**""")

# Gauge graph
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = int(prediction.text),
    number= { 'suffix': "%", 'font':{'color':'dimgrey'}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Probabilité de remboursement", 'font':{'color':'dimgrey' }},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "dimgrey", 'tickfont':{'color':'dimgrey' }},
        'bar': {'color': "lightgrey"},
        'steps': [
            {'range': [0, 50], 'color': 'red'},
            {'range': [50, 100], 'color': 'green'}],
    }))
fig.update_layout(
    margin=dict(l=30, t=50, r=30, b=0),
    height=250,
    width=300
    #margin={'b': 70},
    #height=150,
)
col2.plotly_chart(fig)
#=======================================================================#
### CLIENT INFO ###

st.write("""
## Informations relatives au client """ + str(selected_client))

# list the values of selected features
for feature in features_to_show:
    if feature not in categorical_columns:
        st.markdown("""**"""+feature+""": """ + str(round(data.loc[selected_client, feature], 2))+ '** - <span style=font-size:14px>' + variables.loc[feature, 'Description'], unsafe_allow_html=True)
    else:  # if feature is categorical, display Yes or No
        if data.loc[selected_client, feature] > 0.5:
            st.markdown("""**"""+feature+""": Oui**"""+ ' - <span style=font-size:14px>' + variables.loc[feature, 'Description'], unsafe_allow_html=True)
        else:
            st.markdown("""**"""+feature+""":** Non"""  '<span style=font-size:14px>(' + variables.loc[feature, 'Description'] +')', unsafe_allow_html=True) 

            
#=======================================================================#
# MODEL EXPLICATION

st.write("""
## Explication du modèle""")

st.write("""
###  Caractéristiques influençant le score du client""")
st_shap(shap.plots.waterfall(shap_object[data.index.get_loc(selected_client)], max_display=10),
         width=800, height=300
)

# Store the list of columns
                                 
st.write("""
## Positionnement du client par rapport à l'ensamble de clients """)
# boxplots
fig, ax = plt.subplots((len(features_to_show)+1)//2, 2, figsize=(15, 15))
count = 1
for col in features_to_show:
    plt.subplot((len(features_to_show)+1)//2, 2, count)
    plt.title(col)
    plt.hist(data[col])
    plt.axvline(data.loc[selected_client, col], color='red', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    #plt.text(data.loc[selected_client, col], max_ylim*0.9, "CLIENT")
    count += 1
if len(features_to_show)%2 !=0:
    fig.delaxes(ax[(len(features_to_show)+1)//2-1][1])
plt.tight_layout()
st.pyplot(fig)