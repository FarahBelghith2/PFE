# 📈 PFE — Prédiction de la demande de réservation avec XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-API-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)

## 📌 Description du projet
Ce projet de fin d’études (PFE) a pour objectif de développer un système intelligent capable de **prédire la demande de réservation** selon la **date**, **l’heure** et le **contexte externe** (météo, événements, jours fériés), afin d’aider un administrateur à mieux gérer son complexe sportif.

Le système permet notamment de :
- anticiper les périodes de forte et faible demande,
- améliorer la gestion des réservations,
- identifier les créneaux où une promotion peut être utile,
- proposer une **décision marketing automatisée** (Dynamic Pricing) adaptée à la demande prévue.

---

## 🎯 Objectifs
- Analyser les données de réservation historiques.
- Identifier les périodes à forte et faible demande.
- Construire et comparer plusieurs modèles de régression.
- Sélectionner un modèle performant (XGBoost).
- Intégrer le modèle dans une architecture micro-services (API Flask).
- Aider l’administrateur à prendre des décisions marketing éclairées.

---

## 🤖 Modèles étudiés
Les modèles testés dans ce projet lors de la phase de recherche sont :
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor** *(🏆 Modèle principal retenu pour l’exploitation en production)*

---

## 🛠️ Technologies utilisées
- **Langage :** Python
- **Data Science :** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning :** Scikit-learn, XGBoost
- **Back-End :** Flask, Requests (pour la connexion aux APIs Open Data)
- **Environnement :** Jupyter Notebook / Google Colab

---

## 📂 Structure du projet
```text
PFE/
│
├── README.md                                       # Documentation du projet
├── app.py                                          # API Flask (Serveur de production)
├── features_utils.py                               # Fonctions utilitaires partagées
├── Nettoyage_données.ipynb                         # Préparation initiale des données
├── Préparation_des_données.ipynb                   # Enrichissement des données
├── Business_Understanding_Data_Understanding.ipynb # Compréhension métier et données
├── XGBOOST2.ipynb                                  # Entraînement et évaluation du modèle
└── xgboost_model.pkl                               # Modèle IA final sauvegardé
```

⚙️Étapes du projet

1. Collecte et preparation des données
2. Analyse exploratoire
3. Pretraitement des variables
4. Création de nouvelles variables (Feature Engineering temporel et externe)
5. Entraînement du modèle XGBoost (avec gestion de la sur-représentation des pics)
6. Evaluation des performances (Tests internes et tests externes "Zero-Shot")
7. Integration du modele dans une API Flask
8. Génération d'une recommandation marketing (Règles métier)

---

🚀 Installation & Lancement
1.Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/PFE.git
cd PFE

2.Installer les dépendances
pip install -r requirements.txt

---

▶️ Lancer l’API Flask
python app.py
L’API démarre localement sur :
http://127.0.0.1:5000

---

📡 Routes d'API disponibles
L'architecture est construite en micro-services :

•POST /predict : Prédit la demande et propose une décision marketing (taux, niveau, promo).

•POST /meteo : Retourne la météo réelle ou prévue associée à une date et une heure.

•POST /jour_ferie : Indique si une date est un jour férié en France.

•POST /vacances : Indique si une date tombe pendant les vacances scolaires (Zone C).

•POST /evenement : Retourne les gros événements locaux détectés pour une date donnée.

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

4.Le modèle prédit un taux de demande / remplissage relatif

5.Ce taux est interprété sous forme de niveau de demande (faible, moyenne, forte, très forte)

6.Une décision marketing est générée selon la demande estimée

---

🔧 Configuration : Comment adapter ce projet pour votre ville ?

Le modèle actuel est configuré pour être un "Expert de Paris (France)".

Si vous forkez ce projet pour l'utiliser dans un autre pays ou une autre ville, vous devez modifier le bloc de Configuration situé tout en haut du fichier app.py.

Ouvrez app.py et modifiez les constantes suivantes :

1. Localisation et Météo
Par défaut, la météo interroge les coordonnées de Paris. Remplacez par vos coordonnées GPS :

LATITUDE = 48.8566         # Remplacez par la latitude de votre ville
LONGITUDE = 2.3522         # Remplacez par la longitude de votre ville
TIMEZONE = "Europe/Paris"  # Remplacez par votre fuseau horaire (ex: "America/New_York")


2. Jours Fériés et Vacances
L'API utilise les bases gouvernementales françaises.
   
ZONE_FERIES = "metropole"  # Options FR : "alsace-moselle", "guadeloupe", etc.
ZONE_SCOLAIRE = "C"        # Zones FR : "A", "B", ou "C"

(⚠️ Si vous déployez hors de France, il faudra modifier les URL des APIs dans les fonctions charger_jours_feries_france et charger_vacances_scolaires_officielles pour pointer vers l'Open Data de votre pays).

3. Détection des événements publics

Renseignez la ville pour surveiller l'OpenAgenda local :

DEPARTEMENT = "Paris"      # ex: "Rhône", "Gironde"...
VILLE = None               # Spécifier une ville précise si besoin (ex: "Lyon")

4. Mots-clés des événements majeurs

L'algorithme scanne le nom des événements pour évaluer leur impact. Ajoutez des mots-clés spécifiques à votre culture ou région :

BIG_EVENT_KEYWORDS = [
   "concert", "festival", "spectacle", "show",
    "champion", "champions league", "ligue des champions",
    "coupe", "cup", "finale", "tournoi", "match",
    "euro", "world cup", "coupe du monde",
    "olympique", "jeu olympique", "jeux olympiques"
    # Ajoutez vos événements locaux majeurs ici (ex: "super bowl", "carnaval")
]

---
## 📊 Résultats attendus
Le système permet d’obtenir automatiquement :

Lors d'un appel à /predict, le système génère :

•Un taux de remplissage prévu (ex: 82%).

•Un niveau de demande (faible, moyenne, forte, très forte).

•La détection du contexte (météo pluvieuse, vacances actives, match ce soir).

•Une décision marketing (Dynamic Pricing).

Exemple de logique métier:

si la demande prévue est faible → proposer une promotion,

si la demande est forte → ne pas proposer de réduction,

si le terrain est outdoor et que la météo est défavorable → ajuster la décision marketing.

---
✅ Avantages du système

mieux anticiper la demande,

mieux gérer les créneaux horaires,

améliorer l’occupation des terrains,

prendre des décisions plus rapides,

automatiser partiellement la logique marketing.

---
⚠️ Limites actuelles
•Le projet présente encore certaines limites :

•le modèle a été entraîné sur un contexte géographique précis,

•certaines règles métier sont encore définies manuellement,

•la logique des événements dépend des données disponibles dans les APIs externes,

•la météo future peut être indisponible trop loin dans le temps,

•la généralisation à d’autres villes ou pays nécessite des adaptations.

---
👩‍💻 Auteur

Projet réalisé dans le cadre d’un Projet de Fin d’Études (PFE).
