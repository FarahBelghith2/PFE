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
---

Description des fichiers
README.md : documentation du projet
app.py : API Flask permettant d’utiliser le modèle en temps réel
features_utils.py : fonctions utilitaires partagées (temps, événements, formatage…)
Nettoyage_données.ipynb : nettoyage et préparation initiale des données
Préparation_des_données.ipynb : enrichissement des données
Business_Understanding_Data_Understanding.ipynb : compréhension métier et compréhension des données
xgboost_model.pkl : modèle final sauvegardé

---

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

---

🚀 Installation
1) Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/PFE.git
cd PFE
2) Installer les dépendances
pip install -r requirements.txt
3) Vérifier que les fichiers nécessaires sont présents
Assurez-vous d’avoir dans le dossier du projet :
app.py
features_utils.py
xgboost_model.pkl
---

▶️ Lancer l’API Flask
python app.py
L’API démarre localement sur :
http://127.0.0.1:5000

---

📡 Routes disponibles
1) /predict
Permet de prédire la demande et de proposer une décision marketing.
Exemple de requête :
{
  "date": "2025-08-04",
  "heure_num": 19,
  "type_terrain": "outdoor"
}
Exemple de réponse
{
  "date_demande": "2025-08-04",
  "heure_demande": "19h00",
  "type_terrain": "outdoor",
  "saison": "été",
  "partie_jour": "soir",
  "meteo_detectee": "nuageux",
  "jour_ferie": "Aucun",
  "vacances_scolaires": "Oui",
  "evenement_detecte": "Pécho sous les étoiles (+29 autres)",
  "nombre_evenements_detectes": 30,
  "type_evenement": "Événement local",
  "force_evenement": "Faible",
  "taux_remplissage_prevu": "52%",
  "predicted_reservations": 79,
  "decision_marketing": "Lancer promotion de -15%",
  "status": "success"
}

2) /meteo
Retourne la météo associée à une date et une heure.
Exemple
{
  "date": "2025-08-04",
  "heure_num": 19
}

3) /jour_ferie
Indique si une date est un jour férié et/ou tombe pendant les vacances scolaires.
Exemple
{
  "date": "2025-08-04"
}

4) /vacances
Indique si une date est en vacances scolaires.
Exemple
{
  "date": "2025-08-04"
}

5) /evenement
Retourne les événements détectés pour une date donnée.
Exemple
{
  "date": "2025-08-04"
}

---

🧠 Fonctionnement général
Le système fonctionne en plusieurs étapes :

1.L’utilisateur envoie une date, une heure et un type de terrain
2.L’API récupère les informations externes :
-météo
-jour férié
-vacances scolaires
-événements
3.Les variables nécessaires au modèle sont construites automatiquement
4.Le modèle prédit un taux de remplissage
5.Ce taux est transformé en nombre de réservations prévues
6.Une décision marketing est générée selon la demande estimée

🔧 Paramètres à modifier dans app.py
1) Localisation météo
Modifier :
LATITUDE = .....
LONGITUDE = .....
TIMEZONE = "Europe/Paris"
➡️ Remplacer par la latitude, la longitude et le fuseau horaire de la ville concernée.

2) Vacances scolaires
Modifier :
ZONE_SCOLAIRE = "C"
➡️ La logique actuelle est basée sur le calendrier scolaire français.
Si le pays n’utilise pas ce système, il faudra :
-remplacer la source
-ou modifier complètement la logique de vacances scolaires
3) Jours fériés
Modifier :
ZONE_FERIES = "metropole"
➡️ La logique actuelle correspond aux jours fériés français.
Pour un autre pays, il faudra utiliser une source locale adaptée.
4) Événements publics
Modifier :
DEPARTEMENT = "Paris"
VILLE = None
➡️ Ces paramètres servent à cibler les événements détectés.
Pour une autre utilisation, remplacer par :

-la ville concernée
-la région concernée
-ou adapter complètement la logique si le pays ne fonctionne pas avec ces champs

5) Mots-clés d’événements
Modifier:
BIG_EVENT_KEYWORDS = [...]
➡️ Cette liste contient les mots-clés utilisés pour repérer les événements importants.
Exemples actuels :

concert
festival
finale
tournoi
match
olympique

Si le projet est utilisé dans un autre pays, il faudra :

garder les mots-clés utiles,
supprimer ceux qui ne servent pas,
ajouter des mots-clés locaux
