import time

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS 
import pickle
import requests
from functools import lru_cache
import ast
import re

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

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
    "olympique", "jeu olympique", "jeux olympiques"
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

from features_utils import (
    month_to_season_fr,
    partie_jour,
    extract_event_info,
    extract_event_info_multi,
    choisir_evenement_principal,
    format_event_display
)

# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================
print("Chargement du modèle...")
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Modèle chargé")


def get_niveau_demande(taux):
    if taux < 0.30:
        return "Faible"
    elif taux < 0.60:
        return "Moyenne"
    elif taux < 0.85:
        return "Forte"
    else:
        return "Très forte"


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
        params = {"limit": limit, "offset": offset, "select": "description,start_date,end_date,zones,population"}
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results: break
        all_rows.extend(results)
        total_count = data.get("total_count", len(all_rows))
        offset += limit
        if offset >= total_count: break

    if not all_rows:
        return pd.DataFrame(columns=["description", "start_date", "end_date"])

    vac = pd.DataFrame(all_rows)
    for c in ["description", "start_date", "end_date", "zones", "population"]:
        if c not in vac.columns: vac[c] = None

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
                "select": "title_fr,keywords_fr,firstdate_begin,lastdate_end,timings,location_city,location_department",
                "where": where_clause,
                "limit": limit,
                "offset": offset,
                "order_by": "firstdate_begin"
            }
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results: break
            all_rows.extend(results)
            total_count = data.get("total_count", 0)
            offset += limit
            if offset >= total_count: break
            if offset + limit > 10000: break

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

    def has_occurrence_on_day(row, target_day):
        timings = row.get("timings", None)
        if pd.notna(timings) and str(timings).strip() not in ["", "nan", "None"]:
            try:
                parsed = timings
                if isinstance(parsed, str): parsed = ast.literal_eval(parsed)
                if isinstance(parsed, list):
                    for occ in parsed:
                        begin = pd.to_datetime(occ.get("begin"), errors="coerce")
                        end = pd.to_datetime(occ.get("end"), errors="coerce")
                        if pd.notna(begin):
                            begin_day = begin.tz_localize(None).normalize() if begin.tzinfo is not None else begin.normalize()
                            if pd.notna(end):
                                end_day = end.tz_localize(None).normalize() if end.tzinfo is not None else end.normalize()
                            else:
                                end_day = begin_day
                            if begin_day <= target_day <= end_day: return True
            except Exception: pass

        first_begin = pd.to_datetime(row.get("firstdate_begin"), errors="coerce", utc=True)
        if pd.notna(first_begin):
            first_begin = first_begin.tz_convert(None).normalize()
            return first_begin == target_day
        return False

    expanded_rows = []
    for _, row in events.iterrows():
        for target_day in pd.date_range(start, end, freq="D"):
            if has_occurrence_on_day(row, target_day.normalize()):
                expanded_rows.append({"date_event_local": target_day.normalize(), "nom_event_local": row["title_fr"]})

    events_expanded = pd.DataFrame(expanded_rows)
    if events_expanded.empty:
        return pd.DataFrame(columns=["date_event_local", "nom_event_local"])

    events_grouped = events_expanded.groupby("date_event_local", as_index=False).agg({
        "nom_event_local": lambda x: " | ".join(sorted(set([str(i) for i in x if str(i).strip() != ""])))
    })

    return events_grouped


def get_evenement_info(date_obj):
    date_str = pd.Timestamp(date_obj).strftime("%Y-%m-%d")
    events_df = charger_gros_evenements_locaux(date_start=date_str, date_end=date_str, departement=DEPARTEMENT, ville=VILLE)
    if events_df.empty: return "Aucun", "none", 0
    event_name = str(events_df["nom_event_local"].iloc[0]).strip()
    if event_name == "": return "Aucun", "none", 0
    event_type, event_strength = extract_event_info_multi(event_name)
    return event_name, event_type, event_strength


@app.route("/evenement", methods=["POST"])
def get_evenement():
    try:
        data = request.get_json()
        date_str = data.get("date")
        dt = pd.to_datetime(date_str)
        nom_event_local, _, _ = get_evenement_info(dt)
        principal_event, nb_evenements = choisir_evenement_principal(nom_event_local)
        evenement_admin, _ = format_event_display(nom_event_local)
        event_label_clean = evenement_admin.split(" (+")[0].strip()
        event_type, event_strength = extract_event_info(event_label_clean)
        
        type_evenement_admin = mapping_type_admin.get(event_type, "Événement local")
        impact_admin = mapping_impact_admin.get(event_strength, "Faible")

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

    if date_obj > today + pd.Timedelta(days=16):
        return {"meteo": "prévision_non_disponible", "temperature_c": 20.0, "humidite_pct": 60.0, "vent_kmh": 10.0}

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

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("⚠️ Erreur API météo, utilisation valeurs par défaut:", e)
        return {
            "meteo": "nuageux",
            "temperature_c": 20.0,
            "humidite_pct": 60.0,
            "vent_kmh": 10.0}
    

    if "hourly" not in data or "time" not in data["hourly"]:
        return {"meteo": "nuageux", "temperature_c": 18.0, "humidite_pct": 60.0, "vent_kmh": 10.0}

    df_meteo = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "humidite_pct": data["hourly"]["relative_humidity_2m"],
        "vent_kmh": data["hourly"]["wind_speed_10m"],
        "weather_code": data["hourly"]["weather_code"]
    })

    ligne = df_meteo[df_meteo["time"].dt.hour == int(heure_num)]
    if ligne.empty:
        return {"meteo": "nuageux", "temperature_c": 18.0, "humidite_pct": 60.0, "vent_kmh": 10.0}

    ligne = ligne.iloc[0]

    def map_wmo(code):
        if pd.isna(code): return "nuageux"
        code = int(code)
        if code in [0, 1]: return "ensoleillé"
        elif code in [2, 3, 45, 48]: return "nuageux"
        elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]: return "pluvieux"
        elif code in [71, 73, 75, 77, 85, 86]: return "pluvieux"
        elif code in [95, 96, 99]: return "orageux"
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
# FEATURE ENGINEERING & ANCIENNE ROUTE PREDICT (COMPLÈTE)
# =========================================================
def obtenir_infos_externes(date_obj, heure_num):
    meteo_info = get_meteo_info_reelle(pd.Timestamp(date_obj).strftime("%Y-%m-%d"), int(heure_num))
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

def enrichir_donnees(date_str, heure_num):
    dt = pd.to_datetime(date_str)
    infos_ext = obtenir_infos_externes(dt, heure_num)
    jour_semaine_num = dt.weekday()
    mapping_jours = {0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi", 4: "vendredi", 5: "samedi", 6: "dimanche"}

    features = {
        "heure_num": heure_num,
        "heure_sin": np.sin(2 * np.pi * heure_num / 24),
        "heure_cos": np.cos(2 * np.pi * heure_num / 24),
        "jour_semaine": mapping_jours[jour_semaine_num],
        "partie_jour": partie_jour(heure_num),
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
            "humidite": f"{infos['humidite_pct']} %",   
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
    print("✅ predict appelé")
 
    start = time.time()

    # ton code ici

    print("⏱ temps:", time.time() - start)

    try:
        data = request.get_json()
        active_rules = data.get("active_rules", ["1", "2", "3", "4", "5"])
        rules_applied = []
        print("✅ Règles actives reçues:", active_rules)
        date_str = data.get("date")
        heure_num = float(data.get("heure_num"))
        type_terrain = data.get("type_terrain", "outdoor")

        dictionnaire_complet, infos_ext = enrichir_donnees(date_str, heure_num)
        df = pd.DataFrame([dictionnaire_complet])

        prediction_taux = float(model.predict(df)[0])
        prediction_taux = max(0.0, min(1.0, prediction_taux))

        meteo = infos_ext["meteo"]
        jour = dictionnaire_complet["jour_semaine"]

        taux_ajuste = prediction_taux
        # Règle 3 : météo outdoor
        if "3" in active_rules and heure_num >= 18 and type_terrain == "outdoor":
            if meteo == "nuageux":
                taux_ajuste -= 0.10
                rules_applied.append("Météo outdoor nuageuse")
            elif meteo in ["pluvieux", "orageux", "venteux"]:
                taux_ajuste -= 0.20
                rules_applied.append("Météo défavorable outdoor")
            elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
                taux_ajuste += 0.03
                rules_applied.append("Météo favorable outdoor")

        # Règle 4 : météo indoor
        if "4" in active_rules and heure_num >= 18 and type_terrain == "indoor":
            if meteo == "nuageux":
                taux_ajuste += 0.05
                rules_applied.append("Météo indoor nuageuse")
            elif meteo in ["pluvieux", "orageux", "venteux"]:
                taux_ajuste += 0.10
                rules_applied.append("Météo défavorable favorise indoor")
                # ✅ Jour férié + beau temps + outdoor = forte demande attendue
        # Règle 2 : jour férié + beau temps + outdoor
            # Règle 2 : jour férié + beau temps + outdoor
        if (
            "2" in active_rules
            and type_terrain == "outdoor"
            and infos_ext["ferie"] == 1
            and meteo in ["ensoleillé", "clair", "nuit_claire"]
        ):
            taux_ajuste += 0.20
            taux_ajuste = max(taux_ajuste, 0.65)
            rules_applied.append("Jour férié + beau temps + outdoor")
       # Règle 1 : weekend après-midi / début soirée
        if (
            "1" in active_rules
            and dictionnaire_complet["est_weekend"] == 1
            and 16 <= heure_num < 20
        ):
            taux_ajuste += 0.20
            taux_ajuste = max(taux_ajuste, 0.55)
            rules_applied.append("Weekend après-midi et soir")

        # Règle 1 : weekend soir
        if (
            "1" in active_rules
            and dictionnaire_complet["est_weekend"] == 1
            and heure_num >= 20
        ):
            taux_ajuste += 0.25
            taux_ajuste = max(taux_ajuste, 0.60)
            rules_applied.append("Weekend soir")
        # Règle 5 : vacances ou événement
        if (
            "5" in active_rules
            and (
                infos_ext["is_vacances_scolaires"] == 1
                or infos_ext["event_strength"] > 0
            )
        ):
            taux_ajuste += 0.10
            taux_ajuste = max(taux_ajuste, 0.50)
            rules_applied.append("Vacances ou événement")
            taux_ajuste = max(0.0, min(1.0, taux_ajuste))

        niveau_demande = get_niveau_demande(taux_ajuste)

        seuils = {"bas": 0.15, "haut": 0.65}
        promo_base = 0
        if taux_ajuste <= seuils["bas"]:
            promo_base = 50
        elif taux_ajuste < seuils["haut"]:
            ratio = (taux_ajuste - seuils["bas"]) / (seuils["haut"] - seuils["bas"])
            promo_base = 50 * (1 - ratio)

        promo_base = int(round(promo_base / 5) * 5)
        promo_finale = promo_base

        if jour in ["mardi", "mercredi", "jeudi"] and 14 <= heure_num <= 17:
            promo_finale += 5

        if type_terrain == "outdoor":
            if meteo in ["pluvieux", "orageux", "venteux"]: promo_finale += 10
            elif meteo == "nuageux": promo_finale += 5
            elif meteo in ["ensoleillé", "clair", "nuit_claire"]: promo_finale -= 5
        elif type_terrain == "indoor":
            if meteo in ["pluvieux", "orageux", "venteux"]: promo_finale -= 5
            elif meteo == "nuageux": promo_finale -= 2

        if infos_ext["ferie"] == 1 or infos_ext["is_vacances_scolaires"] == 1:
            promo_finale -= 5

        promo_finale = max(0,(promo_finale))
        promo_finale = int(round(promo_finale / 5) * 5)
        if promo_finale < 5: promo_finale = 0

        decision = f"{promo_finale}%" if promo_finale > 0 else "Aucune promotion nécessaire"

        principal_event, nb_evenements = choisir_evenement_principal(infos_ext["nom_event_local"])
        evenement_admin, _ = format_event_display(infos_ext["nom_event_local"])
        event_label_clean = evenement_admin.split(" (+")[0].strip()
        event_type_principal, event_strength_principal = extract_event_info(event_label_clean)

        type_evenement_admin = mapping_type_admin.get(event_type_principal, "Événement local")
        impact_admin = mapping_impact_admin.get(event_strength_principal, "Faible")
        prediction_finale = int(round(taux_ajuste * 150))
        print("✅ Règles appliquées:", rules_applied)
        print("✅ Taux ajusté:", taux_ajuste)
        print("✅ Décision promo:", decision)
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
            "taux_remplissage_prevu": f"{int(taux_ajuste * 100)}%",
            "predicted_reservations": prediction_finale,
            "decision_marketing": decision,
            "rules_applied": rules_applied,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# =========================================================
# NOUVELLES ROUTES : PRÉDICTIONS SUR-MESURE (À LA CARTE)
# =========================================================

def enrichir_donnees_sur_mesure(date_str, heure_num, use_meteo=False, use_vacances=False, use_ferie=False, use_event=False):
    dt = pd.to_datetime(date_str)
    jour_semaine_num = dt.weekday()
    mapping_jours = {0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi", 4: "vendredi", 5: "samedi", 6: "dimanche"}
    
    # 2. Valeurs neutres par défaut
    meteo_val = "nuageux"
    temp_val = 20.0
    hum_val = 50.0
    vent_val = 10.0
    ferie_val = 0
    vacances_val = 0
    event_type_val = "none"
    event_strength_val = 0
    nom_ferie = "Aucun"
    nom_event = "Aucun"

    # 3. Activation "À la carte"
    if use_meteo:
        m = get_meteo_info_reelle(date_str, heure_num)
        meteo_val, temp_val, hum_val, vent_val = m["meteo"], m["temperature_c"], m["humidite_pct"], m["vent_kmh"]
    if use_vacances:
        vacances_val = est_vacances_scolaires(dt)
    if use_ferie:
        ferie_val, nom_ferie = get_jour_ferie_info(dt)
    if use_event:
        nom_event, event_type_val, event_strength_val = get_evenement_info(dt)

    features = {
        "heure_num": heure_num,
        "heure_sin": np.sin(2 * np.pi * heure_num / 24),
        "heure_cos": np.cos(2 * np.pi * heure_num / 24),
        "jour_semaine": mapping_jours[jour_semaine_num],
        "partie_jour": partie_jour(heure_num),   # <-- UTILISATION DE LA FONCTION IMPORTÉE
        "est_weekend": 1 if jour_semaine_num in [5, 6] else 0,
        "is_peak_hour": 1 if int(heure_num) in [16, 17, 18, 19, 20, 21, 22, 23] else 0,        "is_commute_hour": 1 if heure_num in [7, 8, 9, 17, 18, 19] else 0,
        "mois": dt.month,
        "semaine_annee": int(dt.isocalendar().week),
        "trimestre": (dt.month - 1) // 3 + 1,
        "jour_annee_sin": np.sin(2 * np.pi * dt.dayofyear / 365),
        "jour_annee_cos": np.cos(2 * np.pi * dt.dayofyear / 365),
        "saison": month_to_season_fr(dt.month),  # <-- UTILISATION DE LA FONCTION IMPORTÉE
        "ferie": ferie_val,
        "jour_ouvrable": 0 if (jour_semaine_num in [5, 6] or ferie_val == 1) else 1,
        "is_vacances_scolaires": vacances_val,
        "meteo": meteo_val,
        "meteo_severe": 1 if meteo_val in ["pluvieux", "orageux"] else 0,
        "temperature_c": temp_val,
        "humidite_pct": hum_val,
        "vent_kmh": vent_val,
        "event_type": event_type_val,
        "event_strength": event_strength_val,
        "temp_froide": 1 if temp_val < 5 else 0,
        "temp_chaude": 1 if temp_val > 28 else 0,
        "humidite_forte": 1 if hum_val > 80 else 0,
        "vent_fort": 1 if vent_val > 20 else 0,
        "site_id": "site_A",
        "annee": dt.year
    }
    return features, meteo_val, ferie_val, vacances_val, nom_ferie, nom_event


def generer_prediction_et_promo(data, mode, use_meteo, use_vacances, use_ferie, use_event):
    date_str = data.get("date")
    heure_num = float(data.get("heure_num"))
    type_terrain = data.get("type_terrain", "outdoor")
    active_rules = data.get("active_rules", ["1", "2", "3", "4", "5"])
    rules_applied = []
    features, meteo, ferie, vacances, nom_ferie, nom_event = enrichir_donnees_sur_mesure(
        date_str, heure_num, use_meteo, use_vacances, use_ferie, use_event
    )
    
    df = pd.DataFrame([features])
    prediction_taux = float(model.predict(df)[0])
    prediction_taux = max(0.0, min(1.0, prediction_taux))

    # ✅ taux ajusté utilisé pour corriger les cas métier
    taux_ajuste = prediction_taux
    # Règle 3 : météo outdoor
    if "3" in active_rules and use_meteo and heure_num >= 18 and type_terrain == "outdoor":
        if meteo == "nuageux":
            taux_ajuste -= 0.10
            rules_applied.append("Météo outdoor nuageuse")
        elif meteo in ["pluvieux", "orageux", "venteux"]:
            taux_ajuste -= 0.20
            rules_applied.append("Météo défavorable outdoor")
        elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
            taux_ajuste += 0.03
            rules_applied.append("Météo favorable outdoor")

    # Règle 4 : météo indoor
    if "4" in active_rules and use_meteo and heure_num >= 18 and type_terrain == "indoor":
        if meteo == "nuageux":
            taux_ajuste += 0.05
            rules_applied.append("Météo indoor nuageuse")
        elif meteo in ["pluvieux", "orageux", "venteux"]:
            taux_ajuste += 0.10
            rules_applied.append("Météo défavorable favorise indoor")
    # ✅ Jour férié + beau temps + outdoor = forte demande attendue
    if (
        "2" in active_rules
        and type_terrain == "outdoor"
        and use_ferie
        and ferie == 1
        and meteo in ["ensoleillé", "clair", "nuit_claire"]
    ):
        taux_ajuste += 0.20
        taux_ajuste = max(taux_ajuste, 0.65)
        rules_applied.append("Férié + beau temps outdoor")

    if (
        "1" in active_rules
        and features["est_weekend"] == 1
        and 16 <= heure_num < 20
    ):
        taux_ajuste += 0.20
        taux_ajuste = max(taux_ajuste, 0.55)
        rules_applied.append("Weekend après-midi et soir")

    if (
        "1" in active_rules
        and features["est_weekend"] == 1
        and heure_num >= 20
    ):
        taux_ajuste += 0.25
        taux_ajuste = max(taux_ajuste, 0.60)
        rules_applied.append("Weekend soir")
    # ✅ sécuriser entre 0 et 1
    taux_ajuste = max(0.0, min(1.0, taux_ajuste))
    

    prediction_finale = int(round(taux_ajuste * 150))

    # ✅ Calcul promo basé sur taux_ajuste, pas prediction_taux
    promo_base = 0
    if taux_ajuste <= 0.15:
        promo_base = 30
    elif taux_ajuste < 0.65:
        ratio = (taux_ajuste - 0.15) / (0.65 - 0.15)
        promo_base = 30 * (1 - ratio)

    promo_finale = int(round(promo_base / 5) * 5)

    if use_meteo and type_terrain == "outdoor":
        if meteo in ["pluvieux", "orageux", "venteux"]:
            promo_finale += 10
        elif meteo in ["ensoleillé", "clair"]:
            promo_finale -= 5

    if use_ferie and ferie == 1:
        promo_finale -= 5

    if use_vacances and vacances == 1:
        promo_finale -= 5

    promo_finale = max(0,promo_finale)

    if promo_finale < 5:
        promo_finale = 0

    decision = f"{promo_finale}%" if promo_finale > 0 else "Aucune promotion nécessaire"

    reponse_json = {
        "date": date_str,
        "heure": f"{heure_num}h00",
        "taux_remplissage_prevu": f"{int(taux_ajuste * 100)}%",
        "predicted_reservations": prediction_finale,
        "decision_marketing": decision,
        "rules_applied": rules_applied
    }

    if use_vacances and vacances == 1:
         reponse_json["vacances_scolaires_detectees"] = "Oui"

    if use_ferie and ferie == 1:
         reponse_json["jour_ferie_detecte"] = nom_ferie

    if use_meteo:
        reponse_json["meteo_detectee"] = meteo

    if use_event and nom_event != "Aucun":
        reponse_json["evenement_detecte"] = nom_event

    return jsonify(reponse_json)
    

# =========================================================
# LES 5 NOUVELLES PORTES (ROUTES)
# =========================================================
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json() or {}
        active_rules = data.get("active_rules", ["1", "2", "3", "4", "5"])

        start_dt = pd.to_datetime(data.get("start_datetime"))
        end_dt   = pd.to_datetime(data.get("end_datetime"))
        step_minutes = int(data.get("step_minutes", 30))
        type_terrain = data.get("type_terrain", "outdoor")
        scenario = str(data.get("scenario", "all")).lower().strip()

        if pd.isna(start_dt) or pd.isna(end_dt):
            return jsonify({"status": "error", "message": "start_datetime/end_datetime requis"}), 400
        if end_dt <= start_dt:
            return jsonify({"status": "error", "message": "end_datetime doit être > start_datetime"}), 400
        if step_minutes <= 0:
            return jsonify({"status": "error", "message": "step_minutes doit être > 0"}), 400

        # Map scenario -> flags
        if scenario == "base":
            flags = dict(use_meteo=False, use_vacances=False, use_ferie=False, use_event=False)
        elif scenario == "meteo":
            flags = dict(use_meteo=True, use_vacances=False, use_ferie=False, use_event=False)
        elif scenario == "vacances":
            flags = dict(use_meteo=False, use_vacances=True, use_ferie=False, use_event=False)
        elif scenario == "ferie":
            flags = dict(use_meteo=False, use_vacances=False, use_ferie=True, use_event=False)
        elif scenario == "evenement":
            flags = dict(use_meteo=False, use_vacances=False, use_ferie=False, use_event=True)
        else:
            flags = dict(use_meteo=True, use_vacances=True, use_ferie=True, use_event=True)

        slots = pd.date_range(start=start_dt, end=end_dt, freq=f"{step_minutes}min", inclusive="left")

        results = []
        for ts in slots:
            current_date = ts.strftime("%Y-%m-%d")
            heure_num = ts.hour + ts.minute / 60.0
            heure_label = ts.strftime("%H:%M")

            payload = {
                "date": current_date,
                "heure_num": heure_num,
                "type_terrain": type_terrain,
                "active_rules": active_rules
            }

            resp = generer_prediction_et_promo(
                payload,
                "ALL",
                use_meteo=flags["use_meteo"],
                use_vacances=flags["use_vacances"],
                use_ferie=flags["use_ferie"],
                use_event=flags["use_event"]
            )
            j = resp.get_json()

            # taux "38%" -> 38
            taux_str = str(j.get("taux_remplissage_prevu", "0%"))
            try:
                taux_num = int(taux_str.replace("%", ""))
            except Exception:
                taux_num = 0

            # promo depuis decision "15%" sinon 0
            decision = str(j.get("decision_marketing", ""))
            m = re.search(r"\d+", decision)
            promo_num = int(m.group(0)) if m else 0

            detail = {
            "datetime": ts.strftime("%Y-%m-%d %H:%M"),
            "date": current_date,
            "heure": heure_label,
            "taux": taux_num,
            "reservations": int(j.get("predicted_reservations", 0)),
            "promo": promo_num,
            "rules_applied": j.get("rules_applied", [])
}

    # ✅ Ajouter seulement si existe
            if j.get("meteo_detectee"):
                detail["meteo"] = j.get("meteo_detectee")

            if j.get("jour_ferie_detecte"):
                detail["jour_ferie"] = j.get("jour_ferie_detecte")

            if j.get("vacances_scolaires_detectees"):
                detail["vacances"] = j.get("vacances_scolaires_detectees")

            if j.get("evenement_detecte"):
                detail["evenement"] = j.get("evenement_detecte")

            results.append(detail)

        if not results:
            return jsonify({"status": "success", "summary": {}, "details": [], "count": 0})

        avg_taux = int(round(sum(r["taux"] for r in results) / len(results)))
        total_res = int(sum(r["reservations"] for r in results))
        promo_max = int(max(r["promo"] for r in results))

        return jsonify({
            "status": "success",
            "count": len(results),
            "summary": {
                "avg_taux": avg_taux,
                "total_reservations": total_res,
                "promo_max": promo_max,
                "scenario": scenario,
                "step_minutes": step_minutes,
                "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M")
            },
            "details": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
@app.route("/predict/base", methods=["POST"])
def pred_base():
    try: return generer_prediction_et_promo(request.get_json(), "BASE", use_meteo=False, use_vacances=False, use_ferie=False, use_event=False)
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/vacances", methods=["POST"])
def pred_vacances():
    try: return generer_prediction_et_promo(request.get_json(), "VACANCES_UNIQUEMENT", use_meteo=False, use_vacances=True, use_ferie=False, use_event=False)
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/ferie", methods=["POST"])
def pred_ferie():
    try: return generer_prediction_et_promo(request.get_json(), "FERIE_UNIQUEMENT", use_meteo=False, use_vacances=False, use_ferie=True, use_event=False)
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/evenement", methods=["POST"])
def pred_evenement():
    try: return generer_prediction_et_promo(request.get_json(), "EVENEMENT_UNIQUEMENT", use_meteo=False, use_vacances=False, use_ferie=False, use_event=True)
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/meteo", methods=["POST"])
def pred_meteo():
    try: return generer_prediction_et_promo(request.get_json(), "METEO_UNIQUEMENT", use_meteo=True, use_vacances=False, use_ferie=False, use_event=False)
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
