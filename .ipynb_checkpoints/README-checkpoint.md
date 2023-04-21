# Project 7 - Création d'un Dashboard - OpenClassrooms
Elena Nardi, Avril 2023

## Run the dashboard app.
https://elena-openclassrooms-dashboard.herokuapp.com/

## General purposes.
Ce depot contient une application de dashboard à des fins éducatives et répondant au concours Kaggle :
"Risque de défaut de crédit à domicile" (https://www.kaggle.com/c/home-credit-default-risk)

L'ojectif principal de ce projet est de construire un dashboard interactif permettant d'afficher la propabilité de remboursement d'un crédit, 
 interpréter ces probabilités et améliorer la connaissance de l'entreprise sur ses clients.

Spécifications du tableau de bord :

- Permettre de visualiser le score (probabilité de remboursement) et l’interprétation de ce score pour chaque client de façon intelligible.
- Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients

## Implementation.

Ce dashboard interactif a été implémenté à l'aide du framework Streamlit et codé en Python. Les packages nécessaires pour exécuter cette application sont répertoriés dans le fichier requirements.txt. L'application principale dashbord.py est accompagnée des éléments suivants :
 - tests unitaires dans le dossier tests
 - description des variables en français dans le fichier description_variables.csv
 - données d'entree dans le fichier test_kaggle_reduced.csv
 - le fichier explainer.pkl, qui contient le modèle d'explication shap; Le package pickle a été utilisé pour exporter le modèle




