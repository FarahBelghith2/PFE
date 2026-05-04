import pandas as pd
import numpy as np


def parse_heure_num(x):
    x = str(x).strip().lower()
    if x in ["nan", "none", ""]:
        return np.nan
    if "h" in x:
        return pd.to_numeric(x.split("h")[0], errors="coerce")
    if ":" in x:
        return pd.to_numeric(x.split(":")[0], errors="coerce")
    return pd.to_numeric(x, errors="coerce")


def month_to_season_fr(month):
    if month in [12, 1, 2]:
        return "hiver"
    elif month in [3, 4, 5]:
        return "printemps"
    elif month in [6, 7, 8]:
        return "été"
    else:
        return "automne"


def partie_jour(h):
    return "nuit" if h <= 5 else "matin" if h <= 11 else "midi" if h <= 14 else "apres_midi" if h <= 18 else "soir"


def extract_event_info(event_name):
    txt = str(event_name).strip().lower()

    if txt in ["", "aucun", "none", "nan"]:
        return "none", 0

    if "champions league" in txt or "ligue des champions" in txt:
        return "champions_league", 3
    elif "coupe du monde" in txt:
        return "world_cup", 3
    elif "finale" in txt:
        return "finale", 3
    elif "concert" in txt:
        return "concert", 2
    elif "festival" in txt:
        return "festival", 2
    elif "grand prix" in txt:
        return "grand_prix", 2
    elif "match" in txt or "derby" in txt:
        return "match", 1
    elif "tournoi" in txt:
        return "tournoi", 1
    elif "coupe" in txt:
        return "coupe", 1
    else:
        return "other", 1


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