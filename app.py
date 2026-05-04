from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import requests
from functools import lru_cache

app = Flask(__name__)

# =========================================================
# CONFIG
# =========================================================
ZONE_FERIES = "metropole"
ZONE_SCOLAIRE = "C"
DEPARTEMENT = "Paris"
VILLE = None

LATITUDE = 48.8566
LONGITUDE = 2.3522
TIMEZONE = "Europe/Paris"

BIG_EVENT_KEYWORDS = [
    "concert", "festival", "spectacle", "show",
    "champion", "champions league", "ligue des champions",
    "coupe", "cup", "finale", "tournoi", "match",
    "euro", "world cup", "coupe du monde",
    "olympique", "jo", "jeu olympique", "jeux olympiques"
]

mapping_type_admin = {
    "none": "Aucun événement",
    "concert": "Concert / festival",
    "sport": "Événement sportif",
    "champions_league": "Grand match",
    "other": "Événement local"
}

mapping_impact_admin = {
    0: "Aucun",
    1: "Faible",
    2: "Moyen",
    3: "Fort"
}

# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================
print("Chargement du modèle...")
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Modèle chargé")

# =========================================================
# OUTILS
# =========================================================
def month_to_season_fr(month):
    if month in [12, 1, 2]:
        return "hiver"
    elif month in [3, 4, 5]:
        return "printemps"
    elif month in [6, 7, 8]:
        return "été"
    else:
        return "automne"


def partie_jour_from_hour(heure_num):
    if heure_num <= 5:
        return "nuit"
    elif heure_num <= 11:
        return "matin"
    elif heure_num <= 14:
        return "midi"
    elif heure_num <= 18:
        return "apres_midi"
    else:
        return "soir"


def extract_event_info(event_name):
    event_name = str(event_name).lower().strip()

    if event_name in ["", "aucun", "none", "nan"]:
        return "none", 0

    if any(k in event_name for k in ["champions league", "ligue des champions"]):
        event_type = "champions_league"
    elif any(k in event_name for k in ["concert", "festival", "spectacle", "show"]):
        event_type = "concert"
    elif any(k in event_name for k in ["match", "tournoi", "coupe", "finale", "euro", "world cup", "olympique", "jo"]):
        event_type = "sport"
    else:
        event_type = "other"

    if any(k in event_name for k in ["champions league", "ligue des champions", "world cup", "coupe du monde", "jo", "jeux olympiques"]):
        event_strength = 3
    elif any(k in event_name for k in ["concert", "festival", "finale", "tournoi", "match", "euro", "olympique"]):
        event_strength = 2
    else:
        event_strength = 1

    return event_type, event_strength


def extract_event_info_multi(event_text):
    txt = str(event_text).strip()

    if txt in ["", "Aucun", "none", "nan"]:
        return "none", 0

    evenements = [e.strip() for e in txt.split("|") if e.strip() != ""]

    types = []
    strengths = []

    for ev in evenements:
        ev_type, ev_strength = extract_event_info(ev)
        if ev_type != "none":
            types.append(ev_type)
            strengths.append(ev_strength)

    if not types:
        return "none", 0

    type_final = pd.Series(types).value_counts().idxmax()
    strength_final = max(strengths)

    return type_final, strength_final


def choisir_evenement_principal(event_text):
    txt = str(event_text).strip()

    if txt in ["", "Aucun", "none", "nan"]:
        return "Aucun", 0

    evenements = []
    for e in txt.split("|"):
        e = e.strip()
        if e and e.lower() not in ["aucun", "none", "nan"] and e not in evenements:
            evenements.append(e)

    if not evenements:
        return "Aucun", 0

    principal = max(evenements, key=lambda ev: extract_event_info(ev)[1])
    return principal, len(evenements)


def format_event_display(event_text):
    principal, nb = choisir_evenement_principal(event_text)

    if principal == "Aucun" or nb == 0:
        return "Aucun", 0

    if nb == 1:
        return principal, 1

    return f"{principal} (+{nb-1} autres)", nb


# =========================================================
# API JOURS FÉRIÉS
# =========================================================
@lru_cache(maxsize=8)
def charger_jours_feries_france(zone="metropole"):
    url = f"https://etalab.github.io/jours-feries-france-data/json/{zone}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    feries = pd.DataFrame(list(data.items()), columns=["date_ferie", "nom_jour_ferie"])
    feries["date_ferie"] = pd.to_datetime(feries["date_ferie"], errors="coerce").dt.normalize()
    feries["ferie"] = 1
    return feries


def get_jour_ferie_info(date_obj):
    feries = charger_jours_feries_france(ZONE_FERIES)
    date_norm = pd.Timestamp(date_obj).normalize()

    match = feries.loc[feries["date_ferie"] == date_norm]
    if match.empty:
        return 0, "Aucun"
    return 1, str(match["nom_jour_ferie"].iloc[0])


# =========================================================
# API VACANCES SCOLAIRES
# =========================================================
@lru_cache(maxsize=4)
def charger_vacances_scolaires_officielles(zone="C"):
    base_url = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"

    all_rows = []
    offset = 0
    limit = 100

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "select": "description,start_date,end_date,zones,population"
        }

        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        all_rows.extend(results)

        total_count = data.get("total_count", len(all_rows))
        offset += limit
        if offset >= total_count:
            break

    if not all_rows:
        return pd.DataFrame(columns=["description", "start_date", "end_date"])

    vac = pd.DataFrame(all_rows)

    for c in ["description", "start_date", "end_date", "zones", "population"]:
        if c not in vac.columns:
            vac[c] = None

    vac["population"] = vac["population"].astype(str).str.lower().str.strip()
    vac = vac[~vac["population"].str.contains("enseignant", na=False)].copy()

    vac["start_date"] = pd.to_datetime(vac["start_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    vac["end_date"] = pd.to_datetime(vac["end_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    vac = vac.dropna(subset=["start_date", "end_date"]).copy()

    vac["zones"] = vac["zones"].astype(str).str.lower().str.strip()
    vac["description"] = vac["description"].astype(str).str.lower().str.strip()

    zone_label = f"zone {str(zone).lower()}"
    mask_zone = (
        vac["zones"].str.contains(zone_label, na=False) |
        vac["zones"].str.contains("toutes les zones", na=False) |
        vac["zones"].str.contains("paris|versailles|créteil|creteil|toulouse|montpellier", na=False) |
        vac["description"].str.contains("noël|noel|été|ete", na=False)
    )
    vac = vac[mask_zone].copy()

    mots_cles = ["vacances", "toussaint", "noël", "noel", "hiver", "printemps", "été", "ete"]
    pattern = "|".join(mots_cles)
    vac = vac[vac["description"].str.contains(pattern, na=False)].copy()

    return vac[["description", "start_date", "end_date"]]


def est_vacances_scolaires(date_obj):
    vacances_df = charger_vacances_scolaires_officielles(zone=ZONE_SCOLAIRE)
    date_norm = pd.Timestamp(date_obj).normalize()

    for _, vac in vacances_df.iterrows():
        start = pd.to_datetime(vac["start_date"]).normalize()
        end = pd.to_datetime(vac["end_date"]).normalize()
        if start <= date_norm < end:
            return 1

    return 0


@app.route("/vacances", methods=["POST"])
def get_vacances():
    try:
        data = request.get_json()
        date_str = data.get("date")
        dt = pd.to_datetime(date_str)

        vacances = est_vacances_scolaires(dt)

        return jsonify({
            "date_consultee": date_str,
            "en_vacances_scolaires": "Oui" if vacances == 1 else "Non",
            "zone_scolaire": ZONE_SCOLAIRE
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# =========================================================
# API ÉVÉNEMENTS
# =========================================================
def charger_gros_evenements_locaux(date_start, date_end, departement=None, ville=None, limit=100):
    base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    start = pd.Timestamp(date_start).normalize()
    end = pd.Timestamp(date_end).normalize()

    all_rows = []

    for day in pd.date_range(start=start, end=end, freq="D"):
        day_start = day.normalize()
        day_end = day_start + pd.Timedelta(days=1)

        conditions = [
            f"firstdate_begin < date'{day_end:%Y-%m-%d}T00:00:00+00:00'",
            f"lastdate_end >= date'{day_start:%Y-%m-%d}T00:00:00+00:00'"
        ]

        if departement:
            dep = str(departement).replace("'", "''")
            conditions.append(f"location_department = '{dep}'")

        if ville:
            v = str(ville).replace("'", "''")
            conditions.append(f"location_city = '{v}'")

        where_clause = " AND ".join(conditions)

        offset = 0
        while True:
            params = {
                "select": "title_fr,keywords_fr,firstdate_begin,lastdate_end,location_city,location_department",
                "where": where_clause,
                "limit": limit,
                "offset": offset,
                "order_by": "firstdate_begin"
            }

            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            all_rows.extend(results)

            total_count = data.get("total_count", 0)
            offset += limit

            if offset >= total_count:
                break

            if offset + limit > 10000:
                print(f"⚠️ Trop d'événements le {day_start:%Y-%m-%d}, on coupe la pagination.")
                break

    events = pd.DataFrame(all_rows)

    if events.empty:
        return pd.DataFrame(columns=["date_event_local", "nom_event_local"])

    events["title_fr"] = events["title_fr"].fillna("").astype(str)
    events["keywords_fr"] = events["keywords_fr"].fillna("").astype(str)

    if BIG_EVENT_KEYWORDS:
        pattern = "|".join([kw.lower().replace("+", r"\+") for kw in BIG_EVENT_KEYWORDS])

        filtered = events[
            events["title_fr"].str.lower().str.contains(pattern, na=False, regex=True) |
            events["keywords_fr"].str.lower().str.contains(pattern, na=False, regex=True)
        ].copy()

        if not filtered.empty:
            events = filtered

    events["date_start_evt"] = (
        pd.to_datetime(events["firstdate_begin"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    events["date_end_evt"] = (
        pd.to_datetime(events["lastdate_end"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    events["date_end_evt"] = events["date_end_evt"].fillna(events["date_start_evt"])

    expanded_rows = []
    for _, row in events.iterrows():
        if pd.isna(row["date_start_evt"]) or pd.isna(row["date_end_evt"]):
            continue

        for d in pd.date_range(row["date_start_evt"], row["date_end_evt"], freq="D"):
            expanded_rows.append({
                "date_event_local": d.normalize(),
                "nom_event_local": row["title_fr"]
            })

    events_expanded = pd.DataFrame(expanded_rows)

    if events_expanded.empty:
        return pd.DataFrame(columns=["date_event_local", "nom_event_local"])

    events_grouped = events_expanded.groupby("date_event_local", as_index=False).agg({
        "nom_event_local": lambda x: " | ".join(sorted(set([str(i) for i in x if str(i).strip() != ""])))
    })

    return events_grouped


def get_evenement_info(date_obj):
    date_str = pd.Timestamp(date_obj).strftime("%Y-%m-%d")

    events_df = charger_gros_evenements_locaux(
        date_start=date_str,
        date_end=date_str,
        departement=DEPARTEMENT,
        ville=VILLE
    )

    if events_df.empty:
        return "Aucun", "none", 0

    event_name = str(events_df["nom_event_local"].iloc[0]).strip()
    if event_name == "":
        return "Aucun", "none", 0

    event_type, event_strength = extract_event_info_multi(event_name)
    return event_name, event_type, event_strength


@app.route("/evenement", methods=["POST"])
def get_evenement():
    try:
        data = request.get_json()
        date_str = data.get("date")
        dt = pd.to_datetime(date_str)

        nom_event_local, event_type, event_strength = get_evenement_info(dt)

        type_evenement_admin = mapping_type_admin.get(event_type, "Événement local")
        impact_admin = mapping_impact_admin.get(event_strength, "Faible")
        evenement_admin, nb_evenements = format_event_display(nom_event_local)

        return jsonify({
            "date_consultee": date_str,
            "evenement_detecte": evenement_admin,
            "nombre_evenements_detectes": nb_evenements,
            "type_evenement": type_evenement_admin,
            "force_impact": impact_admin
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# =========================================================
# MÉTÉO RÉELLE
# =========================================================
@lru_cache(maxsize=512)
def get_meteo_info_reelle(date_str, heure_num, lat=LATITUDE, lon=LONGITUDE, timezone=TIMEZONE):
    date_obj = pd.Timestamp(date_str).normalize()
    today = pd.Timestamp.today().normalize()

    # fallback si trop loin dans le futur
    if date_obj > today + pd.Timedelta(days=16):
        return {
            "meteo": "prévision_non_disponible",
            "temperature_c": 20.0,
            "humidite_pct": 60.0,
            "vent_kmh": 10.0
        }

    if date_obj <= today:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_obj.strftime("%Y-%m-%d"),
        "end_date": date_obj.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "timezone": timezone
    }

    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        return {
            "meteo": "nuageux",
            "temperature_c": 18.0,
            "humidite_pct": 60.0,
            "vent_kmh": 10.0
        }

    df_meteo = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "humidite_pct": data["hourly"]["relative_humidity_2m"],
        "vent_kmh": data["hourly"]["wind_speed_10m"],
        "weather_code": data["hourly"]["weather_code"]
    })

    ligne = df_meteo[df_meteo["time"].dt.hour == int(heure_num)]

    if ligne.empty:
        return {
            "meteo": "nuageux",
            "temperature_c": 18.0,
            "humidite_pct": 60.0,
            "vent_kmh": 10.0
        }

    ligne = ligne.iloc[0]

    def map_wmo(code):
        if pd.isna(code):
            return "nuageux"
        code = int(code)

        if code in [0, 1]:
            return "ensoleillé"
        elif code in [2, 3, 45, 48]:
            return "nuageux"
        elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
            return "pluvieux"
        elif code in [71, 73, 75, 77, 85, 86]:
            return "pluvieux"
        elif code in [95, 96, 99]:
            return "orageux"
        return "nuageux"

    meteo = map_wmo(ligne["weather_code"])

    if meteo == "ensoleillé" and int(heure_num) <= 5:
        meteo = "nuit_claire"

    return {
        "meteo": meteo,
        "temperature_c": float(ligne["temperature_c"]),
        "humidite_pct": float(ligne["humidite_pct"]),
        "vent_kmh": float(ligne["vent_kmh"])
    }


# =========================================================
# RÉCUPÉRATION DES INFOS EXTERNES
# =========================================================
def obtenir_infos_externes(date_obj, heure_num):
    meteo_info = get_meteo_info_reelle(
        date_str=pd.Timestamp(date_obj).strftime("%Y-%m-%d"),
        heure_num=int(heure_num)
    )

    ferie, nom_ferie = get_jour_ferie_info(date_obj)
    vacances = est_vacances_scolaires(date_obj)
    nom_event_local, event_type, event_strength = get_evenement_info(date_obj)

    return {
        "meteo": meteo_info["meteo"],
        "temperature_c": meteo_info["temperature_c"],
        "humidite_pct": meteo_info["humidite_pct"],
        "vent_kmh": meteo_info["vent_kmh"],
        "ferie": ferie,
        "nom_jour_ferie": nom_ferie,
        "is_vacances_scolaires": vacances,
        "event_type": event_type,
        "event_strength": event_strength,
        "nom_event_local": nom_event_local
    }


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def enrichir_donnees(date_str, heure_num):
    dt = pd.to_datetime(date_str)
    infos_ext = obtenir_infos_externes(dt, heure_num)

    jour_semaine_num = dt.weekday()
    mapping_jours = {
        0: "lundi",
        1: "mardi",
        2: "mercredi",
        3: "jeudi",
        4: "vendredi",
        5: "samedi",
        6: "dimanche"
    }

    features = {
        "heure_num": heure_num,
        "heure_sin": np.sin(2 * np.pi * heure_num / 24),
        "heure_cos": np.cos(2 * np.pi * heure_num / 24),
        "jour_semaine": mapping_jours[jour_semaine_num],
        "partie_jour": partie_jour_from_hour(heure_num),
        "est_weekend": 1 if jour_semaine_num in [5, 6] else 0,
        "is_peak_hour": 1 if heure_num in [8, 9, 12, 13, 18, 19, 20] else 0,
        "is_commute_hour": 1 if heure_num in [7, 8, 9, 17, 18, 19] else 0,
        "mois": dt.month,
        "semaine_annee": int(dt.isocalendar().week),
        "trimestre": (dt.month - 1) // 3 + 1,
        "jour_annee_sin": np.sin(2 * np.pi * dt.dayofyear / 365),
        "jour_annee_cos": np.cos(2 * np.pi * dt.dayofyear / 365),
        "saison": month_to_season_fr(dt.month),
        "ferie": infos_ext["ferie"],
        "jour_ouvrable": 0 if (jour_semaine_num in [5, 6] or infos_ext["ferie"] == 1) else 1,
        "is_vacances_scolaires": infos_ext["is_vacances_scolaires"],
        "meteo": infos_ext["meteo"],
        "meteo_severe": 1 if infos_ext["meteo"] in ["pluvieux", "orageux"] else 0,
        "temperature_c": infos_ext["temperature_c"],
        "humidite_pct": infos_ext["humidite_pct"],
        "vent_kmh": infos_ext["vent_kmh"],
        "event_type": infos_ext["event_type"],
        "event_strength": infos_ext["event_strength"],
        "temp_froide": 1 if infos_ext["temperature_c"] < 5 else 0,
        "temp_chaude": 1 if infos_ext["temperature_c"] > 28 else 0,
        "humidite_forte": 1 if infos_ext["humidite_pct"] > 80 else 0,
        "vent_fort": 1 if infos_ext["vent_kmh"] > 20 else 0,
        "site_id": "site_A",
        "annee": dt.year
    }

    return features, infos_ext


# =========================================================
# ROUTES
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return "API de Prédiction Opérationnelle ✅"


@app.route("/meteo", methods=["POST"])
def get_meteo():
    try:
        data = request.get_json()
        date_str = data.get("date")
        heure_num = int(data.get("heure_num", 12))
        dt = pd.to_datetime(date_str)

        infos = obtenir_infos_externes(dt, heure_num)

        return jsonify({
            "date_consultee": date_str,
            "heure_consultee": f"{heure_num}h00",
            "meteo_prevue": infos["meteo"],
            "temperature": f"{infos['temperature_c']} °C",
            "vent": f"{infos['vent_kmh']} km/h"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/jour_ferie", methods=["POST"])
def get_jour_ferie():
    try:
        data = request.get_json()
        date_str = data.get("date")
        dt = pd.to_datetime(date_str)

        ferie, nom_ferie = get_jour_ferie_info(dt)
        vacances = est_vacances_scolaires(dt)

        return jsonify({
            "date_consultee": date_str,
            "est_un_jour_ferie": "Oui" if ferie == 1 else "Non",
            "nom_jour_ferie": nom_ferie,
            "en_vacances_scolaires": "Oui" if vacances == 1 else "Non"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        date_str = data.get("date")
        heure_num = int(data.get("heure_num"))
        type_terrain = data.get("type_terrain", "outdoor")

        dictionnaire_complet, infos_ext = enrichir_donnees(date_str, heure_num)
        df = pd.DataFrame([dictionnaire_complet])

        prediction_taux = float(model.predict(df)[0])
        prediction_taux = max(0.0, min(1.0, prediction_taux))

        capacite_max = 150
        prediction_finale = int(round(prediction_taux * capacite_max))

        seuils = {"bas": 0.15, "haut": 0.65}

        promo_base = 0
        if prediction_taux <= seuils["bas"]:
            promo_base = 30
        elif prediction_taux < seuils["haut"]:
            ratio = (prediction_taux - seuils["bas"]) / (seuils["haut"] - seuils["bas"])
            promo_base = 30 * (1 - ratio)

        promo_base = int(round(promo_base / 5) * 5)

        promo_finale = promo_base
        jour = dictionnaire_complet["jour_semaine"]
        meteo = infos_ext["meteo"]

        if jour in ["mardi", "mercredi", "jeudi"] and 14 <= heure_num <= 17:
            promo_finale += 5

        if type_terrain == "outdoor":
            if meteo in ["pluvieux", "orageux", "venteux"]:
                promo_finale += 10
            elif meteo in ["ensoleillé", "clair"]:
                promo_finale -= 5

        if infos_ext["ferie"] == 1 or infos_ext["is_vacances_scolaires"] == 1:
            promo_finale -= 5

        promo_finale = max(0, min(promo_finale, 30))
        if promo_finale < 5:
            promo_finale = 0

        if promo_finale > 0:
            decision = f"Lancer promotion de -{promo_finale}%"
        else:
            decision = "Aucune promotion nécessaire"

        type_evenement_admin = mapping_type_admin.get(infos_ext["event_type"], "Événement local")
        impact_admin = mapping_impact_admin.get(infos_ext["event_strength"], "Faible")
        evenement_admin, nb_evenements = format_event_display(infos_ext["nom_event_local"])

        return jsonify({
            "date_demande": date_str,
            "heure_demande": f"{heure_num}h00",
            "type_terrain": type_terrain,
            "saison": dictionnaire_complet["saison"],
            "partie_jour": dictionnaire_complet["partie_jour"],
            "meteo_detectee": meteo,
            "jour_ferie": infos_ext["nom_jour_ferie"],
            "vacances_scolaires": "Oui" if infos_ext["is_vacances_scolaires"] == 1 else "Non",
            "evenement_detecte": evenement_admin,
            "nombre_evenements_detectes": nb_evenements,
            "type_evenement": type_evenement_admin,
            "force_evenement": impact_admin,
            "taux_remplissage_prevu": f"{int(prediction_taux * 100)}%",
            "predicted_reservations": prediction_finale,
            "decision_marketing": decision,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)