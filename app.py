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
import os


# =========================================================
# SÉCURITÉ API EXTERNE + NORMALISATION
# =========================================================
# Ces helpers empêchent Flask de planter si une API externe ferme la connexion
# et empêchent l'erreur : argument of type 'int' is not a container or iterable.

_ORIGINAL_REQUESTS_GET = requests.get

class SafeEmptyResponse:
    """Réponse vide compatible avec requests.Response pour fallback API."""
    status_code = 200
    text = ""
    url = "URL non disponible"

    def json(self):
        return {"results": []}

    def raise_for_status(self):
        return None


def safe_requests_get(url, *args, **kwargs):
    """
    Version sécurisée de requests.get.
    Si une API externe coupe la connexion, on retourne une réponse vide
    au lieu de bloquer toute la route Flask.
    En cas d'erreur SSL, une deuxième tentative est faite avec verify=False.
    """
    try:
        kwargs.setdefault("timeout", 10)

        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("User-Agent", "Mozilla/5.0 PFE-Reservation-App/1.0")
        headers.setdefault("Accept", "application/json")

        return _ORIGINAL_REQUESTS_GET(
            url,
            *args,
            headers=headers,
            **kwargs
        )

    except requests.exceptions.SSLError as e:
        print("⚠️ Erreur SSL API externe :", e)

        try:
            kwargs.setdefault("timeout", 10)

            headers = kwargs.pop("headers", {}) or {}
            headers.setdefault("User-Agent", "Mozilla/5.0 PFE-Reservation-App/1.0")
            headers.setdefault("Accept", "application/json")

            return _ORIGINAL_REQUESTS_GET(
                url,
                *args,
                headers=headers,
                verify=False,
                **kwargs
            )

        except Exception as e2:
            print("⚠️ API externe toujours indisponible après fallback SSL :", e2)
            return SafeEmptyResponse()

    except requests.exceptions.RequestException as e:
        print("⚠️ API externe indisponible")
        print("⚠️ Détail :", e)
        return SafeEmptyResponse()


def normaliser_active_rules(active_rules):
    """
    Garantit que active_rules est toujours une liste de strings.
    Corrige : TypeError: argument of type 'int' is not iterable.
    """
    default_rules = ["1", "2", "3", "4", "5"]

    if active_rules is None:
        return default_rules

    if isinstance(active_rules, list):
        return [str(rule) for rule in active_rules]

    if isinstance(active_rules, (tuple, set)):
        return [str(rule) for rule in active_rules]

    if isinstance(active_rules, str):
        # Accepte aussi "1,2,3" ou "123"
        if "," in active_rules:
            return [x.strip() for x in active_rules.split(",") if x.strip()]
        return [active_rules]

    # Si active_rules arrive comme int, float, etc.
    return [str(active_rules)]


def get_infos_ext_defaut(date_str):
    """Fallback minimal si météo/vacances/fériés/événements sont indisponibles."""
    dt = pd.to_datetime(date_str)
    jour_semaine_num = dt.weekday()
    mapping_jours = {0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi", 4: "vendredi", 5: "samedi", 6: "dimanche"}

    dictionnaire_complet = {
        "heure_num": 12,
        "jour_semaine": mapping_jours[jour_semaine_num],
        "est_weekend": 1 if jour_semaine_num in [5, 6] else 0,
        "is_peak_hour": 0,
        "mois": dt.month,
        "annee": dt.year,
    }

    infos_ext = {
        "meteo": "nuageux",
        "temperature_c": 20.0,
        "humidite_pct": 50.0,
        "vent_kmh": 10.0,
        "ferie": 0,
        "nom_jour_ferie": "Aucun",
        "is_vacances_scolaires": 0,
        "is_vacances": 0,
        "nom_event_local": "Aucun",
        "event_type": "none",
        "event_strength": 0,
    }
    return dictionnaire_complet, infos_ext

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# CONFIG
# =========================================================
ZONE_FERIES = "metropole"
ZONE_SCOLAIRE = "B"
DEPARTEMENT = "Paris"
VILLE = None

LATITUDE = 48.8566
LONGITUDE = 2.3522
TIMEZONE = "Europe/Paris"
@lru_cache(maxsize=128)
def geocoder_adresse(adresse):
    """
    Transforme une adresse en latitude / longitude avec Nominatim.
    Si l'adresse est vide ou introuvable, on garde les coordonnées par défaut.
    """

    try:
        if not adresse or str(adresse).strip() == "":
            return LATITUDE, LONGITUDE, "Adresse par défaut"

        url = "https://nominatim.openstreetmap.org/search"

        params = {
            "q": adresse,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "fr"
        }

        headers = {
            "User-Agent": "PFE-Reservation-App/1.0"
        }

        response = safe_requests_get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            print("⚠️ Erreur géolocalisation:", response.status_code, response.text)
            return LATITUDE, LONGITUDE, "Adresse par défaut"

        data = response.json()

        if not data:
            print("⚠️ Adresse introuvable:", adresse)
            return LATITUDE, LONGITUDE, "Adresse introuvable"

        result = data[0]

        lat = float(result.get("lat"))
        lon = float(result.get("lon"))
        display_name = result.get("display_name", adresse)

        print("✅ Adresse géocodée:", adresse, "=>", lat, lon)

        return lat, lon, display_name

    except Exception as e:
        print("⚠️ Erreur géocodage adresse:", e)
        return LATITUDE, LONGITUDE, "Erreur géocodage"
# =========================================================
# API FOOTBALL-DATA : DÉTECTION AUTOMATIQUE CHAMPIONS LEAGUE
# =========================================================

FOOTBALL_DATA_API_KEY = "5a9541cfa4414609a51493b6b5fb3dd7"
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"


BIG_EVENT_KEYWORDS = [
    "concert", "festival", "spectacle", "show",
    "champion", "champions league", "ligue des champions",
    "coupe", "cup", "finale", "tournoi", "match",
    "euro", "world cup", "coupe du monde",
    "olympique", "jeu olympique", "jeux olympiques",
    "enfant", "enfants", "enfance",
    "journée des femmes", "journee des femmes",
"journée internationale des femmes", "journee internationale des femmes",
"droits des femmes",
    "mère", "mères", "mere", "meres",
    "père", "pères", "pere", "peres", "papa", "papas"
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
    response = safe_requests_get(url, timeout=30)
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
        resp = safe_requests_get(base_url, params=params, timeout=30)
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


def est_vacances_scolaires(date_obj, zone_scolaire=None):
    zone = zone_scolaire or ZONE_SCOLAIRE

    vacances_df = charger_vacances_scolaires_officielles(zone=zone)
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
            resp = safe_requests_get(base_url, params=params, timeout=30)
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

@lru_cache(maxsize=512)
def detecter_evenement_football_cached(date_str, competition_code):
    """
    Détecte un match de football pour une compétition donnée et une date donnée.
    Retourne toujours : nom_evenement, type_evenement, impact.
    """
    try:
        if not FOOTBALL_DATA_API_KEY:
            print("⚠️ FOOTBALL_DATA_API_KEY manquante.")
            return "Aucun", "none", 0

        headers = {
            "X-Auth-Token": FOOTBALL_DATA_API_KEY
        }

        url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"

        params = {
            "dateFrom": date_str,
            "dateTo": date_str
        }

        response = safe_requests_get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code == 429:
            print("⚠️ Limite API Football-Data atteinte.")
            return "Aucun", "none", 0

        if response.status_code != 200:
            print("⚠️ Erreur API Football-Data:", response.status_code, response.text)
            return "Aucun", "none", 0

        data = response.json()
        matches = data.get("matches", [])

        if not matches:
            return "Aucun", "none", 0

        match = matches[0]

        home_team = match.get("homeTeam", {}).get("name", "Équipe domicile")
        away_team = match.get("awayTeam", {}).get("name", "Équipe extérieure")
        competition = match.get("competition", {}).get("name", competition_code)
        stage = match.get("stage", "")

        event_name = f"{competition} : {home_team} vs {away_team}"

        if stage:
            event_name += f" ({stage})"

        if competition_code == "CL":
            return event_name, "champions_league", 3

        if competition_code == "WC":
            return event_name, "sport", 3

        return event_name, "sport", 2

    except Exception as e:
        print("⚠️ Erreur détection événement football:", e)
        return "Aucun", "none", 0


def detecter_evenement_football(date_obj):
    """
    Cherche un événement football important.
    Priorité : Coupe du Monde, puis Champions League.
    """
    date_str = pd.Timestamp(date_obj).strftime("%Y-%m-%d")

    competitions_prioritaires = ["WC", "CL"]

    for competition_code in competitions_prioritaires:
        nom_event, type_event, strength = detecter_evenement_football_cached(
            date_str,
            competition_code
        )

        if nom_event != "Aucun" and type_event != "none":
            return nom_event, type_event, strength

    return "Aucun", "none", 0
def get_evenement_info(date_obj):
    """
    Retourne uniquement l'événement local principal ou 'Aucun'.
    Ne détecte pas le football ici, car le sportif est séparé dans get_evenements_separes().
    """
    try:
        date_str = pd.Timestamp(date_obj).strftime("%Y-%m-%d")

        events_df = charger_gros_evenements_locaux(
            date_start=date_str,
            date_end=date_str,
            departement=DEPARTEMENT,
            ville=VILLE
        )

        if events_df is None or events_df.empty:
            return "Aucun", "none", 0

        if "nom_event_local" not in events_df.columns:
            return "Aucun", "none", 0

        event_name = str(events_df["nom_event_local"].iloc[0]).strip()

        if not event_name or event_name == "nan":
            return "Aucun", "none", 0

        try:
            event_type, event_strength = extract_event_info_multi(event_name)
        except Exception:
            event_type, event_strength = extract_event_info(event_name)

        return event_name, event_type, event_strength

    except Exception as e:
        print("⚠️ Erreur get_evenement_info :", e)
        return "Aucun", "none", 0


def get_evenements_separes(date_obj):
    """
    Retourne séparément :
    - evenement_local
    - evenement_sportif
    - evenement_principal
    """

    evenement_local = "Aucun"
    evenement_sportif = "Aucun"
    evenement_principal = "Aucun"

    type_local, strength_local = "none", 0
    type_sportif, strength_sportif = "none", 0

    # 1) Événement sportif
    try:
        nom_sportif, type_sportif, strength_sportif = detecter_evenement_football(date_obj)

        if nom_sportif and nom_sportif != "Aucun" and type_sportif != "none":
            evenement_sportif = nom_sportif
        else:
            evenement_sportif = "Aucun"
            type_sportif, strength_sportif = "none", 0

    except Exception as e:
        print("⚠️ Erreur événement sportif :", e)
        evenement_sportif = "Aucun"
        type_sportif, strength_sportif = "none", 0

    # 2) Événement local
    try:
        nom_local, type_local, strength_local = get_evenement_info(date_obj)

        if nom_local and nom_local != "Aucun" and type_local != "none":
            evenement_local_affiche, _ = format_event_display(nom_local)
            evenement_local = evenement_local_affiche.split(" (+")[0].strip()
        else:
            evenement_local = "Aucun"
            type_local, strength_local = "none", 0

    except Exception as e:
        print("⚠️ Erreur événement local :", e)
        evenement_local = "Aucun"
        type_local, strength_local = "none", 0

    # 3) Événement principal : priorité sportif puis local
    if evenement_sportif != "Aucun":
        evenement_principal = evenement_sportif
        event_type = type_sportif
        event_strength = strength_sportif

    elif evenement_local != "Aucun":
        evenement_principal = evenement_local
        event_type = type_local
        event_strength = strength_local

    else:
        evenement_principal = "Aucun"
        event_type = "none"
        event_strength = 0

    return {
        "evenement_local": evenement_local,
        "evenement_sportif": evenement_sportif,
        "evenement_principal": evenement_principal,
        "event_type": event_type,
        "event_strength": event_strength
    }


@app.route("/evenement", methods=["POST"])
def get_evenement():
    try:
        data = request.get_json() or {}
        date_str = data.get("date")

        if not date_str:
            return jsonify({
                "status": "error",
                "message": "date est obligatoire"
            }), 400

        dt = pd.to_datetime(date_str)
        evenements_sep = get_evenements_separes(dt)

        return jsonify({
            "status": "success",
            "date_consultee": date_str,
            "evenement_local": evenements_sep.get("evenement_local", "Aucun"),
            "evenement_sportif": evenements_sep.get("evenement_sportif", "Aucun"),
            "evenement": evenements_sep.get("evenement_principal", "Aucun")
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400



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
        resp = safe_requests_get(base_url, params=params, timeout=10)
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
def obtenir_infos_externes(date_obj, heure_num, zone_scolaire=None, adresse_site=None):
    lat, lon, adresse_geocodee = geocoder_adresse(adresse_site)

    meteo_info = get_meteo_info_reelle(
        pd.Timestamp(date_obj).strftime("%Y-%m-%d"),
        int(heure_num),
        lat=lat,
        lon=lon
    )

    ferie, nom_ferie = get_jour_ferie_info(date_obj)
    vacances = est_vacances_scolaires(date_obj, zone_scolaire)
    nom_event_local, event_type, event_strength = get_evenement_info(date_obj)

    return {
        "meteo": meteo_info["meteo"],
        "temperature_c": meteo_info["temperature_c"],
        "humidite_pct": meteo_info["humidite_pct"],
        "vent_kmh": meteo_info["vent_kmh"],
        "ferie": ferie,
        "nom_jour_ferie": nom_ferie,
        "is_vacances_scolaires": vacances,
        "is_vacances": vacances,
        "event_type": event_type,
        "event_strength": event_strength,
        "nom_event_local": nom_event_local,
        "latitude_site": lat,
        "longitude_site": lon,
        "adresse_geocodee": adresse_geocodee
    }

def enrichir_donnees(date_str, heure_num, zone_scolaire=None, adresse_site=None):
    dt = pd.to_datetime(date_str)
    infos_ext = obtenir_infos_externes(
        dt,
        heure_num,
        zone_scolaire,
        adresse_site
    )
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
        "is_vacances": infos_ext["is_vacances_scolaires"],
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

    try:
        data = request.get_json() or {}
        
        date_str = data.get("date")
        heure_num = float(data.get("heure_num"))
        type_terrain = data.get("type_terrain", "outdoor")

        zone_scolaire = data.get("zone_scolaire", ZONE_SCOLAIRE)
        adresse_site = data.get("adresse_site", "")

        active_rules = normaliser_active_rules(
            data.get("active_rules", ["1", "2", "3", "4", "5"])
        )

        rules_applied = []
        print("✅ Règles actives reçues:", active_rules)

        date_str = data.get("date")
        heure_num = float(data.get("heure_num"))
        type_terrain = data.get("type_terrain", "outdoor")

        dictionnaire_complet, infos_ext = enrichir_donnees(date_str, heure_num, zone_scolaire)
        df = pd.DataFrame([dictionnaire_complet])

        prediction_taux = float(model.predict(df)[0])
        prediction_taux = max(0.0, min(1.0, prediction_taux))

        meteo = infos_ext.get("meteo", "nuageux")
        jour = dictionnaire_complet.get("jour_semaine", "")
        ferie = infos_ext.get("ferie", 0)

        # ✅ AJOUT : événements séparés local / sportif
        evenements_sep = get_evenements_separes(pd.to_datetime(date_str))

        evenement_local = evenements_sep.get("evenement_local", "Aucun")
        evenement_sportif = evenements_sep.get("evenement_sportif", "Aucun")
        evenement_principal = evenements_sep.get("evenement_principal", "Aucun")
        event_strength_sep = evenements_sep.get("event_strength", 0)

        taux_ajuste = prediction_taux
        predicted_reservations = int(round(taux_ajuste * 150))
        ferie = infos_ext.get("ferie", 0)

# =========================================================
# Événements séparés : local / sportif
# =========================================================

        # =========================================================
        # Règle 2 : Jour férié selon météo et type de terrain
        # =========================================================
        if "2" in active_rules and ferie == 1:

            # Jour férié + beau temps + outdoor
            if (
                type_terrain == "outdoor"
                and meteo in ["ensoleillé", "clair", "nuit_claire"]
            ):
                taux_ajuste += 0.14
                taux_ajuste = max(taux_ajuste, 0.65)
                rules_applied.append("Jour férié avec météo favorable outdoor")

            # Jour férié + mauvais temps + outdoor
            elif (
                type_terrain == "outdoor"
                and meteo in ["pluvieux", "orageux", "venteux"]
            ):
                taux_ajuste -= 0.08
                rules_applied.append("Jour férié avec météo défavorable outdoor")

            # Jour férié + mauvais temps + indoor
            elif (
                type_terrain == "indoor"
                and meteo in ["pluvieux", "orageux", "venteux"]
            ):
                taux_ajuste += 0.12
                rules_applied.append("Jour férié avec météo défavorable favorisant indoor")

            # Jour férié normal
            else:
                taux_ajuste += 0.06
                rules_applied.append("Jour férié")

        # =========================================================
        # Règle 3 : Météo outdoor hors jour férié
        # =========================================================
        if (
            "3" in active_rules
            and ferie == 0
            and type_terrain == "outdoor"
        ):
            if meteo == "nuageux":
                taux_ajuste -= 0.10
                rules_applied.append("Météo outdoor nuageuse")

            elif meteo in ["pluvieux", "orageux", "venteux"]:
                taux_ajuste -= 0.20
                rules_applied.append("Météo défavorable outdoor")

            elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
                taux_ajuste += 0.03
                rules_applied.append("Météo favorable outdoor")

        # =========================================================
        # Règle 4 : Météo indoor hors jour férié
        # =========================================================
        if (
            "4" in active_rules
            and ferie == 0
            and type_terrain == "indoor"
        ):
            if meteo == "nuageux":
                taux_ajuste += 0.05
                rules_applied.append("Météo indoor nuageuse")

            elif meteo in ["pluvieux", "orageux", "venteux"]:
                taux_ajuste += 0.10
                rules_applied.append("Météo défavorable favorise indoor")

        # =========================================================
        # Règle 1 : Weekend après-midi / début soirée
        # =========================================================
        if (
            "1" in active_rules
            and dictionnaire_complet.get("est_weekend", 0) == 1
            and 16 <= heure_num < 20
        ):
            taux_ajuste += 0.20
            taux_ajuste = max(taux_ajuste, 0.55)
            rules_applied.append("Weekend après-midi et soir")

        # =========================================================
        # Règle 1 : Weekend soir
        # =========================================================
        if (
            "1" in active_rules
            and dictionnaire_complet.get("est_weekend", 0) == 1
            and heure_num >= 20
        ):
            taux_ajuste += 0.25
            taux_ajuste = max(taux_ajuste, 0.60)
            rules_applied.append("Weekend soir")

        # =========================================================
        # Règle 5 : Vacances scolaires / événements
        # =========================================================
        if "5" in active_rules:
            vacances = infos_ext.get(
                "is_vacances",
                infos_ext.get("is_vacances_scolaires", 0)
            )

            event_strength = event_strength_sep or infos_ext.get("event_strength", 0)

            if vacances == 1 and event_strength > 0:
                taux_ajuste += 0.10
                taux_ajuste = max(taux_ajuste, 0.55)
                rules_applied.append("Vacances scolaires + événement détecté")

            elif vacances == 1:
                taux_ajuste += 0.10
                taux_ajuste = max(taux_ajuste, 0.50)
                rules_applied.append("Vacances scolaires détectées")

            elif event_strength > 0:
                taux_ajuste += 0.10
                taux_ajuste = max(taux_ajuste, 0.50)
                rules_applied.append(
                f"Événement détecté : {evenement_principal}"
)

        # Sécurité : garder le taux entre 0 et 1
        taux_ajuste = max(0.0, min(1.0, taux_ajuste))

        niveau_demande = get_niveau_demande(taux_ajuste)

        # =========================================================
        # Calcul de la promotion
        # =========================================================
        seuils = {"bas": 0.15, "haut": 0.65}
        promo_base = 0

        if taux_ajuste <= seuils["bas"]:
            promo_base = 50

        elif taux_ajuste < seuils["haut"]:
            ratio = (taux_ajuste - seuils["bas"]) / (
                seuils["haut"] - seuils["bas"]
            )
            promo_base = 50 * (1 - ratio)

        promo_base = int(round(promo_base / 5) * 5)
        promo_finale = promo_base

        # Bonus promotion sur certains créneaux creux
        if jour in ["mardi", "mercredi", "jeudi"] and 14 <= heure_num <= 17:
            promo_finale += 5

        # Ajustement promo selon météo
        if type_terrain == "outdoor":
            if meteo in ["pluvieux", "orageux", "venteux"]:
                promo_finale += 10
            elif meteo == "nuageux":
                promo_finale += 5
            elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
                promo_finale -= 5

        elif type_terrain == "indoor":
            if meteo in ["pluvieux", "orageux", "venteux"]:
                promo_finale -= 5
            elif meteo == "nuageux":
                promo_finale -= 2

        # Si la demande est naturellement plus forte, on réduit la promo
        if (
            infos_ext.get("ferie", 0) == 1
            or infos_ext.get("is_vacances_scolaires", 0) == 1
        ):
            promo_finale -= 5

        promo_finale = max(0, promo_finale)
        promo_finale = int(round(promo_finale / 5) * 5)

        if promo_finale < 5:
            promo_finale = 0

        decision = (
            f"{promo_finale}%"
            if promo_finale > 0
            else "Aucune promotion nécessaire"
        )
       

        principal_event, nb_evenements = choisir_evenement_principal(
        evenement_principal
    )

        evenement_admin, _ = format_event_display(
        evenement_principal
    )

        event_label_clean = evenement_admin.split(" (+")[0].strip()
        event_type_principal, event_strength_principal = extract_event_info(
            event_label_clean
        )
        if evenement_sportif != "Aucun":
         event_type_principal = "sport"
        event_strength_principal = 3
        type_evenement_admin = mapping_type_admin.get(
            event_type_principal,
            "Événement local"
        )

        impact_admin = mapping_impact_admin.get(
            event_strength_principal,
            "Faible"
        )

        # ✅ Ciblage promotionnel selon l'événement détecté
        ciblage_event = get_ciblage_promo_evenement(evenement_admin)
        if ciblage_event.get("promo_bonus", 0) > 0:
            promo_finale = min(50, promo_finale + ciblage_event["promo_bonus"])
            promo_finale = int(round(promo_finale / 5) * 5)
            decision = f"{promo_finale}%" if promo_finale > 0 else "Aucune promotion nécessaire"
            rules_applied.append(ciblage_event["message_ciblage"])

        prediction_finale = int(round(taux_ajuste * 150))

        print("✅ Règles appliquées:", rules_applied)
        print("✅ Taux ajusté:", taux_ajuste)
        print("✅ Décision promo:", decision)
        print("⏱ temps:", time.time() - start)
        predicted_reservations = int(round(taux_ajuste * 150))
        return jsonify({
            "date_demande": date_str,
            "heure_demande": f"{heure_num}h00",
            "type_terrain": type_terrain,
            "saison": dictionnaire_complet.get("saison"),
            "partie_jour": dictionnaire_complet.get("partie_jour"),
            "meteo_detectee": meteo,
            "jour_ferie": infos_ext.get("nom_jour_ferie", "Aucun"),
            "vacances_scolaires": (
                "Oui"
                if infos_ext.get("is_vacances_scolaires", 0) == 1
                else "Non"
            ),
           "evenement_local": evenement_local,
            "evenement_sportif": evenement_sportif,
            "evenement": evenement_principal,
            "evenement_detecte": evenement_principal,
            "nombre_evenements_detectes": nb_evenements,
            "type_evenement": type_evenement_admin,
            "force_evenement": impact_admin,
            "taux_remplissage_prevu": f"{int(taux_ajuste * 100)}%",
            "niveau_demande": niveau_demande,
            "predicted_reservations": prediction_finale,
            "decision_marketing": decision,
            "target_gender": ciblage_event["target_gender"],
            "min_age": ciblage_event["min_age"],
            "max_age": ciblage_event["max_age"],
            "message_ciblage": ciblage_event["message_ciblage"],
            "promo_bonus_evenement": ciblage_event["promo_bonus"],
        })

    except Exception as e:
        print("Erreur /predict:", e)
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
        "is_vacances": vacances_val,
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
    active_rules = normaliser_active_rules(data.get("active_rules", ["1", "2", "3", "4", "5"]))
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
    

    predicted_reservations = int(round(taux_ajuste * 150))

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

    # ✅ Ciblage promotionnel selon l'événement détecté
    ciblage_event = get_ciblage_promo_evenement(nom_event)
    if ciblage_event.get("promo_bonus", 0) > 0:
        promo_finale = min(50, promo_finale + ciblage_event["promo_bonus"])
        promo_finale = int(round(promo_finale / 5) * 5)
        decision = f"{promo_finale}%" if promo_finale > 0 else "Aucune promotion nécessaire"
        rules_applied.append(ciblage_event["message_ciblage"])

    reponse_json = {
        "date": date_str,
        "heure": f"{heure_num}h00",
        "taux_remplissage_prevu": f"{int(taux_ajuste * 100)}%",
        "predicted_reservations": predicted_reservations,
        "decision_marketing": decision
    }

    if use_vacances and vacances == 1:
         reponse_json["vacances_scolaires_detectees"] = "Oui"

    if use_ferie and ferie == 1:
         reponse_json["jour_ferie_detecte"] = nom_ferie

    if use_meteo:
        reponse_json["meteo_detectee"] = meteo

    if use_event and nom_event != "Aucun":
        reponse_json["evenement_detecte"] = nom_event
        reponse_json["target_gender"] = ciblage_event["target_gender"]
        reponse_json["min_age"] = ciblage_event["min_age"]
        reponse_json["max_age"] = ciblage_event["max_age"]
        reponse_json["message_ciblage"] = ciblage_event["message_ciblage"]
        reponse_json["promo_bonus_evenement"] = ciblage_event["promo_bonus"]
    return jsonify(reponse_json)
    

# =========================================================
# LES 5 NOUVELLES PORTES (ROUTES)
# =========================================================
@app.route("/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json() or {}
        active_rules = normaliser_active_rules(
            data.get("active_rules", ["1", "2", "3", "4", "5"])
        )

        start_dt = pd.to_datetime(data.get("start_datetime"))
        end_dt = pd.to_datetime(data.get("end_datetime"))
        step_minutes = int(data.get("step_minutes", 60))
        type_terrain = data.get("type_terrain", "outdoor")
        scenario = str(data.get("scenario", "all")).lower().strip()

        if pd.isna(start_dt) or pd.isna(end_dt):
            return jsonify({
                "status": "error",
                "message": "start_datetime/end_datetime requis"
            }), 400

        if end_dt <= start_dt:
            return jsonify({
                "status": "error",
                "message": "end_datetime doit être > start_datetime"
            }), 400

        if step_minutes <= 0:
            return jsonify({
                "status": "error",
                "message": "step_minutes doit être > 0"
            }), 400

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

        slots = pd.date_range(
            start=start_dt,
            end=end_dt,
            freq=f"{step_minutes}min",
            inclusive="both"
        )

        results = []

        for ts in slots:
            current_date = ts.strftime("%Y-%m-%d")
            heure_num = ts.hour + ts.minute / 60.0
            heure_label = ts.strftime("%H:%M")

            payload = {
                "date": current_date,
                "heure_num": heure_num,
                "type_terrain": type_terrain,
                "active_rules": active_rules,
                "adresse_site": data.get("adresse_site"),
                "zone_scolaire": data.get("zone_scolaire")
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

            # ✅ Ajout événements séparés local / sportif
            evenements_sep = get_evenements_separes(ts)

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

                # ✅ Champs événements ajoutés ici
                "evenement_local": evenements_sep.get("evenement_local", "Aucun"),
                "evenement_sportif": evenements_sep.get("evenement_sportif", "Aucun"),
                "evenement": evenements_sep.get("evenement_principal", "Aucun"),
                "evenement_detecte": evenements_sep.get("evenement_principal", "Aucun"),

                "rules_applied": j.get("rules_applied", [])
            }

            # ✅ Ajouter seulement si existe
            if j.get("meteo_detectee"):
                detail["meteo"] = j.get("meteo_detectee")

            if j.get("jour_ferie_detecte"):
                detail["jour_ferie"] = j.get("jour_ferie_detecte")

            if j.get("vacances_scolaires_detectees"):
                detail["vacances"] = j.get("vacances_scolaires_detectees")

            # Ancien champ événement IA, mais on garde la priorité au nouveau champ
            if j.get("evenement_detecte") and detail["evenement"] == "Aucun":
                detail["evenement"] = j.get("evenement_detecte")
                detail["evenement_detecte"] = j.get("evenement_detecte")

            results.append(detail)

        if not results:
            return jsonify({
                "status": "success",
                "summary": {},
                "details": [],
                "count": 0
            })

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
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
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
# ROUTES MÉTIER SANS IA : règles dynamiques uniquement
# =========================================================
# Cette partie n'appelle PAS XGBoost. Elle applique uniquement des règles métier.

CAPACITE_REFERENCE_METIER = 150


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def normaliser_meteo(meteo):
    if meteo is None:
        return "nuageux"

    m = str(meteo).strip().lower()
    mapping = {
        "soleil": "ensoleillé",
        "ensoleille": "ensoleillé",
        "ensoleillé": "ensoleillé",
        "clair": "clair",
        "nuit claire": "nuit_claire",
        "nuit_claire": "nuit_claire",
        "nuage": "nuageux",
        "nuageux": "nuageux",
        "couvert": "nuageux",
        "pluie": "pluvieux",
        "pluvieux": "pluvieux",
        "averse": "pluvieux",
        "orage": "orageux",
        "orageux": "orageux",
        "vent": "venteux",
        "venteux": "venteux",
    }
    return mapping.get(m, m)


def base_taux_metier(date_str, heure_num, type_terrain):
    """
    Base variable selon heure + jour + terrain + saison.
    Cela évite d'avoir toujours 44% et la même promo.
    """
    dt = pd.to_datetime(date_str)
    jour = dt.weekday()
    mois = dt.month
    h = float(heure_num)
    terrain = str(type_terrain or "outdoor").lower()

    # Base selon l'heure
    if h < 8:
        taux = 0.18
    elif h < 11:
        taux = 0.28
    elif h < 14:
        taux = 0.36
    elif h < 16:
        taux = 0.42
    elif h < 18:
        taux = 0.53
    elif h < 21:
        taux = 0.68
    elif h < 23:
        taux = 0.57
    else:
        taux = 0.30

    # Effet jour de semaine
    effet_jour = {
        0: -0.06,  # lundi
        1: -0.04,  # mardi
        2: 0.02,   # mercredi
        3: 0.00,   # jeudi
        4: 0.08,   # vendredi
        5: 0.14,   # samedi
        6: 0.10,   # dimanche
    }
    taux += effet_jour.get(jour, 0.0)

    # Effet saison/terrain
    mois_chauds = [5, 6, 7, 8, 9]
    mois_froids = [11, 12, 1, 2]

    if terrain == "outdoor" and mois in mois_chauds:
        taux += 0.04
    elif terrain == "outdoor" and mois in mois_froids:
        taux -= 0.04
    elif terrain == "indoor" and mois in mois_froids:
        taux += 0.04

    return clamp(taux)


def charger_infos_pour_metier(date_str, heure_num, zone_scolaire=None, adresse_site=None):
    """
    Charge les infos externes avec fallback.
    Si une API externe échoue, on continue avec des valeurs par défaut.
    """
    try:
        dictionnaire_complet, infos_ext = enrichir_donnees(
    date_str,
    heure_num,
    zone_scolaire,
    adresse_site
)

        # Compatibilité avec les anciennes clés
        infos_ext["is_vacances"] = infos_ext.get("is_vacances", infos_ext.get("is_vacances_scolaires", 0))
        infos_ext["is_vacances_scolaires"] = infos_ext.get("is_vacances_scolaires", infos_ext.get("is_vacances", 0))
        infos_ext.setdefault("meteo", "nuageux")
        infos_ext.setdefault("ferie", 0)
        infos_ext.setdefault("nom_jour_ferie", "Aucun")
        infos_ext.setdefault("nom_event_local", "Aucun")
        infos_ext.setdefault("event_strength", 0)

        return dictionnaire_complet, infos_ext

    except Exception as e:
        print("⚠️ enrichir_donnees indisponible, fallback métier utilisé :", e)
        dictionnaire_complet, infos_ext = get_infos_ext_defaut(date_str)
        dictionnaire_complet["heure_num"] = heure_num
        return dictionnaire_complet, infos_ext


def calculer_prediction_metier(date_str, heure_num, type_terrain, active_rules=None, zone_scolaire=None, adresse_site=None):
    active_rules = normaliser_active_rules(active_rules)
    type_terrain = str(type_terrain or "outdoor").lower()

    dictionnaire_complet, infos_ext = charger_infos_pour_metier(
    date_str,
    heure_num,
    zone_scolaire,
    adresse_site
)

    meteo = normaliser_meteo(infos_ext.get("meteo"))
    ferie = int(infos_ext.get("ferie", 0) or 0)
    vacances = int(infos_ext.get("is_vacances_scolaires", infos_ext.get("is_vacances", 0)) or 0)
    event_strength = int(infos_ext.get("event_strength", 0) or 0)
    event_type = infos_ext.get("event_type", "none")
    nom_event = infos_ext.get("nom_event_local", "Aucun")
    nom_ferie = infos_ext.get("nom_jour_ferie", "Aucun")

    taux = base_taux_metier(date_str, heure_num, type_terrain)
    rules_applied = ["Base métier dynamique : heure + jour + terrain + saison"]

    # =========================================================
    # Règle 1 : Weekend
    # =========================================================
    if "1" in active_rules and dictionnaire_complet.get("est_weekend") == 1:
        if 16 <= heure_num < 20:
            taux += 0.08
            rules_applied.append("Weekend après-midi / début soirée")
        elif heure_num >= 20:
            taux += 0.10
            rules_applied.append("Weekend soir")
        else:
            taux += 0.05
            rules_applied.append("Weekend")

    # =========================================================
    # Règle 2 : Jour férié selon météo et type de terrain
    # =========================================================
    if "2" in active_rules and ferie == 1:

        if type_terrain == "outdoor" and meteo in ["ensoleillé", "clair", "nuit_claire"]:
            taux += 0.14
            rules_applied.append("Jour férié avec météo favorable outdoor")

        elif type_terrain == "outdoor" and meteo in ["pluvieux", "orageux", "venteux"]:
            taux -= 0.08
            rules_applied.append("Jour férié avec météo défavorable outdoor")

        elif type_terrain == "indoor" and meteo in ["pluvieux", "orageux", "venteux"]:
            taux += 0.12
            rules_applied.append("Jour férié avec météo défavorable favorisant indoor")

        else:
            taux += 0.06
            rules_applied.append("Jour férié")

    # =========================================================
    # Règle 3 : Météo outdoor hors jour férié
    # =========================================================
    if (
        "3" in active_rules
        and ferie == 0
        and type_terrain == "outdoor"
    ):
        if meteo == "nuageux":
            taux -= 0.06
            rules_applied.append("Météo outdoor nuageuse")

        elif meteo in ["pluvieux", "orageux", "venteux"]:
            taux -= 0.16
            rules_applied.append("Météo défavorable outdoor")

        elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
            taux += 0.05
            rules_applied.append("Météo favorable outdoor")

    # =========================================================
    # Règle 4 : Météo indoor hors jour férié
    # =========================================================
    if (
        "4" in active_rules
        and ferie == 0
        and type_terrain == "indoor"
    ):
        if meteo == "nuageux":
            taux += 0.04
            rules_applied.append("Météo nuageuse favorise légèrement indoor")

        elif meteo in ["pluvieux", "orageux", "venteux"]:
            taux += 0.10
            rules_applied.append("Météo défavorable favorise indoor")

        elif meteo in ["ensoleillé", "clair", "nuit_claire"]:
            taux -= 0.02
            rules_applied.append("Beau temps favorise moins indoor")

    # =========================================================
    # Règle spéciale : Champions League
    # =========================================================
    if event_type == "champions_league":
        taux += 0.15
        rules_applied.append(f"Grand match Champions League détecté : {nom_event}")

    # =========================================================
    # Règle 5 : Vacances / événements
    # =========================================================
    if "5" in active_rules:
        if vacances == 1 and event_strength > 0:
            taux += 0.14
            rules_applied.append("Vacances scolaires + événement détecté")

        elif vacances == 1:
            taux += 0.09
            rules_applied.append("Vacances scolaires")

        elif event_strength > 0:
            bonus_event = min(0.12, 0.04 * event_strength)
            taux += bonus_event
            rules_applied.append(f"Événement détecté : {nom_event}")

    # =========================================================
    # Sécurité : taux entre 0 et 1
    # =========================================================
    taux = clamp(taux)

    # =========================================================
    # Réservations prévues
    # =========================================================
    predicted_reservations = int(round(taux * CAPACITE_REFERENCE_METIER))

    # =========================================================
    # Calcul promotion
    # =========================================================
    if taux < 0.30:
        promo_finale = 30
    elif taux < 0.50:
        promo_finale = 20
    elif taux < 0.65:
        promo_finale = 10
    else:
        promo_finale = 0

    # Si la demande est naturellement boostée, réduire un peu la promo
    if ferie == 1 or vacances == 1 or event_strength > 0:
        promo_finale = max(0, promo_finale - 5)

    # ✅ Ciblage promotionnel selon l'événement détecté
    ciblage_event = get_ciblage_promo_evenement(nom_event)
    if ciblage_event.get("promo_bonus", 0) > 0:
        promo_finale = min(50, promo_finale + ciblage_event["promo_bonus"])
        rules_applied.append(ciblage_event["message_ciblage"])

    promo_finale = int(round(promo_finale / 5) * 5)
    decision = f"{promo_finale}%" if promo_finale > 0 else "Aucune promotion nécessaire"

    return {
        "date_demande": date_str,
        "heure_demande": f"{heure_num}00",
        "type_terrain": type_terrain,
        "meteo_detectee": meteo,
        "jour_ferie": nom_ferie,
        "vacances_scolaires": "Oui" if vacances == 1 else "Non",
        "evenement_detecte": nom_event,
        "taux_remplissage_prevu": f"{int(round(taux * 100))}%",
        "predicted_reservations": predicted_reservations,
        "decision_marketing": decision,
        "rules_applied": rules_applied,
        "target_gender": ciblage_event["target_gender"],
        "min_age": ciblage_event["min_age"],
        "max_age": ciblage_event["max_age"],
        "message_ciblage": ciblage_event["message_ciblage"],
        "promo_bonus_evenement": ciblage_event["promo_bonus"],
        "adresse_site": adresse_site,
        "zone_scolaire": zone_scolaire or ZONE_SCOLAIRE,
        "adresse_site": adresse_site,
        "adresse_geocodee": infos_ext.get("adresse_geocodee"),
        "latitude_site": infos_ext.get("latitude_site"),
        "longitude_site": infos_ext.get("longitude_site"),
        "zone_scolaire": zone_scolaire or ZONE_SCOLAIRE,
        "mode": "metier_sans_ia",
        "debug_metier": {
            "taux_decimal": round(taux, 4),
            "capacite_reference": CAPACITE_REFERENCE_METIER,
            "active_rules": active_rules,
            "source_infos": "api_ou_fallback"
        }
    }
def extraire_age_depuis_evenement(event_name):
    event = str(event_name or "").lower()

    # Cas : 2-6 ans / 5-10 ans / 6 à 12 ans
    matches = re.findall(r'(\d{1,2})\s*(?:-|à|a)\s*(\d{1,2})\s*ans', event)
    if matches:
        ages_min = [int(m[0]) for m in matches]
        ages_max = [int(m[1]) for m in matches]
        return min(ages_min), max(ages_max)

    # Cas : de 6 à 12 ans
    matches = re.findall(r'de\s*(\d{1,2})\s*(?:à|a)\s*(\d{1,2})\s*ans', event)
    if matches:
        ages_min = [int(m[0]) for m in matches]
        ages_max = [int(m[1]) for m in matches]
        return min(ages_min), max(ages_max)

    # Cas : moins de 18 ans
    match = re.search(r'moins de\s*(\d{1,2})\s*ans', event)
    if match:
        return None, int(match.group(1))

    # Cas : dès 16 ans / à partir de 16 ans
    match = re.search(r'(?:dès|à partir de|a partir de)\s*(\d{1,2})\s*ans', event)
    if match:
        return int(match.group(1)), None

    return None, None

def get_ciblage_promo_evenement(event_name):
    """
    Retourne un ciblage marketing selon le nom d'événement détecté.
    L'âge n'est pas fixé : il est extrait automatiquement depuis l'événement s'il existe.
    """
    event = str(event_name or "").lower()
    min_age_event, max_age_event = extraire_age_depuis_evenement(event)

    # Fête des mères
    if (
        "fête des mères" in event
        or "fete des meres" in event
        or "fête des meres" in event
        or "mères" in event
        or "meres" in event
        or "maman" in event
        or "mamans" in event
    ):
        return {
            "target_gender": "female",
            "min_age": min_age_event,
            "max_age": max_age_event,
            "promo_bonus": 10,
            "message_ciblage": "Promotion spéciale Fête des mères"
        }

    # Fête des pères
    if (
        "fête des pères" in event
        or "fete des peres" in event
        or "fête des peres" in event
        or "pères" in event
        or "peres" in event
        or "papa" in event
        or "papas" in event
    ):
        return {
            "target_gender": "male",
            "min_age": min_age_event,
            "max_age": max_age_event,
            "promo_bonus": 10,
            "message_ciblage": "Promotion spéciale Fête des pères"
        }

    # Journée des femmes : mots-clés précis pour éviter "Femme de chambre"
    if (
        "journée internationale des femmes" in event
        or "journee internationale des femmes" in event
        or "journée des femmes" in event
        or "journee des femmes" in event
        or "droits des femmes" in event
        or "égalité femmes hommes" in event
        or "egalite femmes hommes" in event
        or "fête des femmes" in event
        or "fete des femmes" in event
    ):
        return {
            "target_gender": "female",
            "min_age": min_age_event,
            "max_age": max_age_event,
            "promo_bonus": 10,
            "message_ciblage": "Promotion spéciale Journée des femmes"
        }

    # Enfance / enfants
    if (
        "journée de l'enfant" in event
        or "journee de l'enfant" in event
        or "journée des enfants" in event
        or "journee des enfants" in event
        or "enfance" in event
        or "enfant" in event
        or "enfants" in event
        or "montessori" in event
        or "stage nature" in event
        or "atelier enfant" in event
        or "atelier enfants" in event
    ):
        return {
            "target_gender": None,
            "min_age": min_age_event,
            "max_age": max_age_event,
            "promo_bonus": 15,
            "message_ciblage": "Promotion spéciale enfants"
        }

    # Vacances
    if (
        "vacances scolaires" in event
        or "vacances d'été" in event
        or "vacances ete" in event
        or "colonies de vacances" in event
        or "vacances" in event
    ):
        return {
            "target_gender": None,
            "min_age": min_age_event,
            "max_age": max_age_event,
            "promo_bonus": 5,
            "message_ciblage": "Promotion spéciale vacances"
        }

    return {
        "target_gender": None,
        "min_age": None,
        "max_age": None,
        "promo_bonus": 0,
        "message_ciblage": None
    }
@app.route("/predict/metier", methods=["POST"])
def predict_metier():
    try:
        data = request.get_json() or {}

        date_str = data.get("date")
        heure_num = float(data.get("heure_num"))
        type_terrain = data.get("type_terrain", "outdoor")
        active_rules = normaliser_active_rules(data.get("active_rules", ["1", "2", "3", "4", "5"]))
        zone_scolaire = data.get("zone_scolaire", ZONE_SCOLAIRE)
        adresse_site = data.get("adresse_site", "")

        print("✅ /predict/metier appelé")
        print("✅ Règles actives reçues côté backend :", active_rules)

        if not date_str:
            return jsonify({"status": "error", "message": "date est obligatoire"}), 400

        # Calcul métier normal
        result = calculer_prediction_metier(
            date_str,
            heure_num,
            type_terrain,
            active_rules,
            zone_scolaire,
            adresse_site
        )
        # Récupération séparée des événements
        evenements_sep = get_evenements_separes(pd.to_datetime(date_str))

        # Ajouter les deux événements dans la réponse
        result["evenement_local"] = evenements_sep["evenement_local"]
        result["evenement_sportif"] = evenements_sep["evenement_sportif"]

        # Garder aussi l'événement principal pour compatibilité avec ton front actuel
        result["evenement"] = evenements_sep["evenement_principal"]

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/geocode", methods=["POST"])
def geocode_adresse_route():
    try:
        data = request.get_json() or {}
        adresse_site = data.get("adresse_site", "")

        lat, lon, adresse_geocodee = geocoder_adresse(adresse_site)

        return jsonify({
            "status": "success",
            "adresse_site": adresse_site,
            "adresse_geocodee": adresse_geocodee,
            "latitude": lat,
            "longitude": lon
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route("/reverse-geocode", methods=["POST"])
def reverse_geocode_route():
    try:
        data = request.get_json() or {}

        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))

        url = "https://nominatim.openstreetmap.org/reverse"

        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": 18,
            "addressdetails": 1
        }

        headers = {
            "User-Agent": "PFE-Reservation-App/1.0"
        }

        response = safe_requests_get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "Impossible de convertir les coordonnées en adresse."
            }), 400

        result = response.json()
        adresse = result.get("display_name", "")

        return jsonify({
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "adresse": adresse
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route("/predict/metier/batch", methods=["POST", "OPTIONS"])
def predict_metier_batch():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json() or {}

        active_rules = normaliser_active_rules(
            data.get("active_rules", ["1", "2", "3", "4", "5"])
        )

        start_dt = pd.to_datetime(data.get("start_datetime"))
        end_dt = pd.to_datetime(data.get("end_datetime"))
        step_minutes = int(data.get("step_minutes", 60))

        type_terrain = data.get("type_terrain", "outdoor")
        zone_scolaire = data.get("zone_scolaire", ZONE_SCOLAIRE)
        adresse_site = data.get("adresse_site", "")

        if pd.isna(start_dt) or pd.isna(end_dt):
            return jsonify({
                "status": "error",
                "message": "start_datetime/end_datetime requis"
            }), 400

        if end_dt <= start_dt:
            return jsonify({
                "status": "error",
                "message": "end_datetime doit être > start_datetime"
            }), 400

        if step_minutes <= 0:
            return jsonify({
                "status": "error",
                "message": "step_minutes doit être > 0"
            }), 400

        slots = pd.date_range(
            start=start_dt,
            end=end_dt,
            freq=f"{step_minutes}min",
            inclusive="both"
        )

        results = []

        for ts in slots:
            current_date = ts.strftime("%Y-%m-%d")
            heure_num = ts.hour + ts.minute / 60.0
            heure_label = ts.strftime("%H:%M")

            # Calcul métier normal
            j = calculer_prediction_metier(
                current_date,
                heure_num,
                type_terrain,
                active_rules,
                zone_scolaire,
                adresse_site
            )

            # Séparation événement local / événement sportif
            evenements_sep = get_evenements_separes(ts)

            taux_str = str(j.get("taux_remplissage_prevu", "0%"))

            try:
                taux_num = int(taux_str.replace("%", ""))
            except Exception:
                taux_num = 0

            decision = str(j.get("decision_marketing", ""))
            match = re.search(r"\d+", decision)
            promo_num = int(match.group(0)) if match else 0

            detail = {
                "datetime": ts.strftime("%Y-%m-%d %H:%M"),
                "date": current_date,
                "heure": heure_label,

                "taux": taux_num,
                "reservations": int(j.get("predicted_reservations", 0)),
                "promo": promo_num,

                "zone_scolaire": zone_scolaire,
                "adresse_site": adresse_site,

                # Nouveaux champs séparés
                "evenement_local": evenements_sep.get("evenement_local", "Aucun"),
                "evenement_sportif": evenements_sep.get("evenement_sportif", "Aucun"),

                # Événement principal gardé pour compatibilité
                "evenement": evenements_sep.get("evenement_principal", "Aucun"),

                "target_gender": j.get("target_gender"),
                "min_age": j.get("min_age"),
                "max_age": j.get("max_age"),
                "message_ciblage": j.get("message_ciblage"),
                "promo_bonus_evenement": j.get("promo_bonus_evenement", 0),

                "rules_applied": j.get("rules_applied", [])
            }

            if j.get("meteo_detectee"):
                detail["meteo"] = j.get("meteo_detectee")

            if j.get("jour_ferie") and j.get("jour_ferie") != "Aucun":
                detail["jour_ferie"] = j.get("jour_ferie")

            if j.get("vacances_scolaires") == "Oui":
                detail["vacances"] = j.get("vacances_scolaires")

            results.append(detail)

        if results:
            avg_taux = round(sum(r["taux"] for r in results) / len(results))
            total_reservations = sum(r["reservations"] for r in results)
            promo_max = max(r["promo"] for r in results)
        else:
            avg_taux = 0
            total_reservations = 0
            promo_max = 0

        return jsonify({
            "status": "success",
            "mode": "metier_sans_ia_batch",
            "summary": {
                "avg_taux": avg_taux,
                "total_reservations": total_reservations,
                "promo_max": promo_max,
                "nb_slots": len(results),
                "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M")
            },
            "details": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
