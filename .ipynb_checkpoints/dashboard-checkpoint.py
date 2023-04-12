from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.graph_objects as go
import requests
import seaborn as sns
import shap
import streamlit as st


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
X_test_reduced = load_dataframe('X_test_reduced.csv') 

# chargement explainer et shap values
explainer, shap_values = load_shap_values('explainer.pkl', X_test_reduced)

st.write("""
# Dashboard Scoring Credit""")

# création liste clients
client_list = load_client_list(X_test_reduced)
selected_client = st.sidebar.selectbox('Clients_id', client_list)

#creation liste variables pur le sidebar
features_to_show = st.sidebar.multiselect('Variables', sorted(X_test_reduced.columns), default = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'INSTAL_AMT_PAYMENT_SUM', 'DAYS_EMPLOYED',
                    'NAME_EDUCATION_TYPE_Higher education', 'INSTAL_DPD_MEAN'],
                                         max_selections=10)

categorical=['PREV_NAME_YIELD_GROUP_high_MEAN',
'NAME_EDUCATION_TYPE_Higher education',
'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
       'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
       'PREV_NAME_CLIENT_TYPE_New_MEAN', 'PREV_NAME_PORTFOLIO_POS_MEAN',
       'PREV_NAME_YIELD_GROUP_high_MEAN',
       'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
       'PREV_NAME_YIELD_GROUP_middle_MEAN',
       'PREV_PRODUCT_COMBINATION_Cash_MEAN']


url = 'http://127.0.0.1:3000/predict'
client_id = str(selected_client)
prediction = requests.post(url, data = client_id)
print(prediction)
st.write("""
## Prédiction """)

col1, col2 = st.columns(2)

# Prédictions

etat_client = 'Client peu risqué' if int(prediction.text) < 0.55 else 'Client à risque de defaut'
col1.subheader("""Client """ + client_id)
col1.write("""Probabilité de remboursement: """ + prediction.text +"%")
col1.write("""Etat client: **""" + etat_client + """**""")

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
            {'range': [0, 55], 'color': 'red'},
            {'range': [55, 100], 'color': 'green'}],
    }))
fig.update_layout(
    margin=dict(l=30, t=50, r=30, b=0),
    height=250,
    width=300
    #margin={'b': 70},
    #height=150,
)
col2.plotly_chart(fig)


st.write("""
## Information relatives au client """ + str(selected_client))

for feature in features_to_show:
    if feature not in categorical:
        st.write("""**"""+feature+""":** """ + str(X_test_reduced.loc[selected_client, feature]))
    else:
        if X_test_reduced.loc[selected_client, feature] > 0.5:
            st.write("""**"""+feature+""":** Yes""")
        else:
            st.write("""**"""+feature+""":** Non""")

shap_object = shap.Explanation(base_values = explainer.expected_value[0],
values = shap_values[0],
feature_names = X_test_reduced.columns,
data = X_test_reduced)

st.write("""
## Caractéristiques influençant le score""")
st_shap(shap.summary_plot(shap_values[0], X_test_reduced,  max_display=10, show=False))
st_shap(shap.plots.waterfall(shap_object[X_test_reduced.index.get_loc(selected_client)], max_display=10)
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
    plt.hist(X_test_reduced[col])
    plt.axvline(X_test_reduced.loc[selected_client, col], color='red', linestyle='dashed', linewidth=1)
    #plt.scatter(X_test_reduced.loc[selected_client, col],0, linewidths=5, c='red')
    min_ylim, max_ylim = plt.ylim()
    plt.text(X_test_reduced.loc[selected_client, col], max_ylim*0.9, "CLIENT")
    count += 1
if len(features_to_show)%2 !=0:
    fig.delaxes(ax[(len(features_to_show)+1)//2-1][1])
plt.tight_layout()
st.pyplot(fig)