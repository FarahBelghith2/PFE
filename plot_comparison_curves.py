"""
Génération des courbes de comparaison XGBoost vs autres modèles
Style exactement comme dans la thèse/mémoire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================================
# 1. FONCTION PRINCIPALE : Créer 2 courbes côte à côte (XGBoost vs autre)
# ============================================================================

def plot_xgboost_vs_model(
    y_true,
    y_pred_xgboost,
    y_pred_other,
    model_name_other="Gradient Boosting",
    title="Valeurs réelles vs Valeurs prédites",
    filename=None,
    figsize=(16, 8)
):
    """
    Crée 2 graphiques côte à côte exactement comme dans la thèse:
    - Gauche: XGBoost (meilleur)
    - Droite: Autre modèle
    
    Avec barres pour valeurs réelles + courbe pour valeurs prédites
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    
    x_axis = np.arange(len(y_true))
    
    # ===== GRAPHIQUE 1 : XGBoost =====
    ax1.bar(x_axis, y_true, alpha=0.7, label='Valeurs réelles', 
            color='#4A4E69', width=0.8, edgecolor='black', linewidth=0.5)
    ax1.plot(x_axis, y_pred_xgboost, color='#FF7F0E', linewidth=2.5, 
             label='Valeurs prédites', marker='o', markersize=4, alpha=0.85)
    
    # Zone d'erreur
    ax1.fill_between(x_axis, y_true, y_pred_xgboost, 
                     color='#FF7F0E', alpha=0.15)
    
    mae_xgb = mean_absolute_error(y_true, y_pred_xgboost)
    rmse_xgb = np.sqrt(mean_squared_error(y_true, y_pred_xgboost))
    r2_xgb = r2_score(y_true, y_pred_xgboost)
    
    textstr_xgb = f'MAE: {mae_xgb:.4f} | RMSE: {rmse_xgb:.4f} | R²: {r2_xgb:.4f}'
    ax1.text(0.02, 0.98, textstr_xgb, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.95, 
                     edgecolor='#1f77b4', linewidth=1.5))
    
    ax1.set_ylabel('Total des réservations', fontsize=11, fontweight='bold')
    ax1.set_title(f'Figure - Résultat graphique d\'algorithme XGBoost ⭐', 
                 fontsize=12, fontweight='bold', loc='left', pad=12)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0, top=max(y_true) * 1.05)
    
    # ===== GRAPHIQUE 2 : Autre modèle =====
    ax2.bar(x_axis, y_true, alpha=0.7, label='Valeurs réelles', 
            color='#4A4E69', width=0.8, edgecolor='black', linewidth=0.5)
    ax2.plot(x_axis, y_pred_other, color='#FF7F0E', linewidth=2.5, 
             label='Valeurs prédites', marker='o', markersize=4, alpha=0.85)
    
    # Zone d'erreur
    ax2.fill_between(x_axis, y_true, y_pred_other, 
                     color='#FF7F0E', alpha=0.15)
    
    mae_other = mean_absolute_error(y_true, y_pred_other)
    rmse_other = np.sqrt(mean_squared_error(y_true, y_pred_other))
    r2_other = r2_score(y_true, y_pred_other)
    
    textstr_other = f'MAE: {mae_other:.4f} | RMSE: {rmse_other:.4f} | R²: {r2_other:.4f}'
    ax2.text(0.02, 0.98, textstr_other, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FEF4E8', alpha=0.95,
                     edgecolor='#D97706', linewidth=1.5))
    
    ax2.set_ylabel('Total des réservations', fontsize=11, fontweight='bold')
    ax2.set_title(f'Figure - Résultat graphique d\'algorithme {model_name_other}', 
                 fontsize=12, fontweight='bold', loc='left', pad=12)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0, top=max(y_true) * 1.05)
    
    # Ajouter "Année-Mois" sous les axes
    ax1.set_xlabel('Année-Mois', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Année-Mois', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Courbe sauvegardée : {filename}")
    
    plt.show()
    
    # Afficher le résumé
    print("\n" + "="*70)
    print("COMPARAISON VISUELLE")
    print("="*70)
    print(f"\n{'XGBoost':^35} | {model_name_other:^35}")
    print("-" * 70)
    print(f"MAE:  {mae_xgb:.4f}{' ' * 28} | MAE:  {mae_other:.4f}")
    print(f"RMSE: {rmse_xgb:.4f}{' ' * 28} | RMSE: {rmse_other:.4f}")
    print(f"R²:   {r2_xgb:.4f}{' ' * 28} | R²:   {r2_other:.4f}")
    print("="*70 + "\n")
    
    return fig


# ============================================================================
# 2. FONCTION : Créer 3 graphiques en colonne (XGBoost vs GB vs RF)
# ============================================================================

def plot_three_models_vertical(
    y_true,
    y_pred_xgboost,
    y_pred_gb,
    y_pred_rf,
    title="Comparaison des trois modèles",
    filename=None,
    figsize=(14, 14)
):
    """
    Crée 3 graphiques en vertical : XGBoost, Gradient Boosting, Random Forest
    Style thèse/mémoire
    """
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    x_axis = np.arange(len(y_true))
    
    models = [
        ("XGBoost", y_pred_xgboost, "#1f77b4", "#E8F4F8", axes[0]),
        ("Gradient Boosting", y_pred_gb, "#D97706", "#FEF4E8", axes[1]),
        ("Random Forest", y_pred_rf, "#10B981", "#E8F5E9", axes[2])
    ]
    
    for model_name, y_pred, color_line, color_box, ax in models:
        # Barres pour valeurs réelles
        ax.bar(x_axis, y_true, alpha=0.7, label='Valeurs réelles', 
               color='#4A4E69', width=0.8, edgecolor='black', linewidth=0.5)
        
        # Courbe pour valeurs prédites
        ax.plot(x_axis, y_pred, color=color_line, linewidth=2.5, 
                label='Valeurs prédites', marker='o', markersize=4, alpha=0.85)
        
        # Zone d'erreur
        ax.fill_between(x_axis, y_true, y_pred, color=color_line, alpha=0.15)
        
        # Métriques
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        textstr = f'MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.95, 
                         edgecolor=color_line, linewidth=1.5))
        
        ax.set_ylabel('Total des réservations', fontsize=11, fontweight='bold')
        ax.set_title(f'Figure - Résultat graphique d\'algorithme {model_name}', 
                    fontsize=12, fontweight='bold', loc='left', pad=12)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0, top=max(y_true) * 1.05)
        ax.set_xlabel('Année-Mois', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Trois modèles sauvegardés : {filename}")
    
    plt.show()
    
    return fig


# ============================================================================
# 3. À EXÉCUTER DANS VOTRE NOTEBOOK
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  UTILISATION: Copier-coller dans votre notebook Colab         ║
    ╚════════════════════════════════════════════════════════════════╝
    
    # Importer les fonctions
    from plot_comparison_curves import (
        plot_xgboost_vs_model,
        plot_three_models_vertical
    )
    
    # ===== OPTION 1 : Comparaison XGBoost vs Gradient Boosting =====
    plot_xgboost_vs_model(
        y_true=y_true_float,
        y_pred_xgboost=y_pred_float,
        y_pred_other=y_pred_gb,
        model_name_other="Gradient Boosting",
        title="Valeurs réelles vs Valeurs prédites",
        filename="/content/drive/MyDrive/PFE/xgboost_vs_gb.png"
    )
    
    # ===== OPTION 2 : Les 3 modèles en vertical =====
    plot_three_models_vertical(
        y_true=y_true_float,
        y_pred_xgboost=y_pred_float,
        y_pred_gb=y_pred_gb,
        y_pred_rf=y_pred_rf,
        title="Comparaison des trois modèles",
        filename="/content/drive/MyDrive/PFE/three_models_comparison.png"
    )
    """)
