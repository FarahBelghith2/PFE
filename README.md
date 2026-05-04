# PFE — Prédiction de la demande de réservation avec XGBoost

## 📌 Description du projet
Ce projet de fin d’études (PFE) a pour objectif de développer un système intelligent capable de **prédire la demande de réservation** selon la **date**, **l’heure** et le **contexte externe**, afin d’aider l’administrateur à mieux gérer l’activité.

Le système permet notamment de :

- anticiper les périodes de forte et faible demande,
- améliorer la gestion des réservations,
- identifier les créneaux où une promotion peut être utile,
- proposer une décision marketing adaptée à la demande prévue.

---

## 🎯 Objectifs
Les objectifs principaux du projet sont :

- analyser les données de réservation,
- identifier les périodes à forte et faible demande,
- construire et comparer plusieurs modèles de régression,
- sélectionner un modèle performant,
- intégrer le modèle dans une API Flask,
- aider l’administrateur à prendre une décision marketing.

---

## 🤖 Modèles étudiés
Les modèles testés dans ce projet sont :

- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

Le modèle principal retenu pour l’exploitation est **XGBoost**.

---

## 🛠️ Technologies utilisées
- **Python**
- **Jupyter Notebook / Google Colab**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Flask**
- **Requests**

---

## 📂 Structure du projet
PFE/
│
├── README.md
├── app.py
├── features_utils.py
├── Nettoyage_données.ipynb
├── Préparation_des_données.ipynb
├── Business_Understanding_Data_Understanding.ipynb
├── XGBOOST2.ipynb
└── xgboost_model.pkl
Description des fichiers

README.md : documentation du projet
app.py : API Flask permettant d’utiliser le modèle en temps réel
features_utils.py : fonctions utilitaires partagées (temps, événements, formatage…)
Nettoyage_données.ipynb : nettoyage et préparation initiale des données
Préparation_des_données.ipynb : enrichissement des données
Business_Understanding_Data_Understanding.ipynb : compréhension métier et compréhension des données
xgboost_model.pkl : modèle final sauvegardé


⚙️ Étapes du projet
Le projet suit plusieurs étapes :

Collecte et préparation des données
Analyse exploratoire
Prétraitement des variables
Création de nouvelles variables (feature engineering)
Entraînement du modèle XGBoost
Évaluation des performances
Intégration du modèle dans une API Flask
Génération d’une recommandation marketing


🚀 Installation
1) Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/PFE.git
cd PFE
