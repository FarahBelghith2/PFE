"""
Visualisations de comparaison XGBoost vs autres modèles
Style professionnel comme en thèse/mémoire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Style de la figure
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================================
# 1. COMPARAISON VISUELLE : XGBoost vs autres modèles (style annuel/mois)
# ============================================================================

def create_comparison_figure_annual(
    y_true, 
    y_pred_xgboost, 
    y_pred_gb, 
    y_pred_rf,
    dates=None,
    title="Valeurs réelles vs Valeurs prédites",
    filename=None
):
    """
    Crée une figure de comparaison annuelle style mémoire/thèse
    avec XGBoost clairement supérieur
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    x_axis = np.arange(len(y_true))
    
    models = [
        ("XGBoost", y_pred_xgboost, "#2E86AB", axes[0]),
        ("Gradient Boosting", y_pred_gb, "#A23B72", axes[1]),
        ("Random Forest", y_pred_rf, "#F18F01", axes[2])
    ]
    
    for model_name, y_pred, color, ax in models:
        # Tracer les valeurs réelles et prédites
        ax.bar(x_axis, y_true, alpha=0.6, label='Valeurs réelles', 
               color='#4A4E69', width=0.8)
        ax.plot(x_axis, y_pred, color=color, linewidth=2.5, 
                label='Valeurs prédites', marker='o', markersize=3, alpha=0.8)
        
        # Zone d'erreur
        ax.fill_between(x_axis, y_true, y_pred, alpha=0.15, color=color)
        
        # Calcul des métriques
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Texte des métriques
        textstr = f'MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}'
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel('Total des réservations', fontsize=11, fontweight='bold')
        ax.set_title(f'Figure - Résultat graphique d\'algorithme {model_name}', 
                    fontsize=12, fontweight='bold', loc='left', pad=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
    
    axes[2].set_xlabel('Année-Mois', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée : {filename}")
    
    plt.show()
    return fig


def create_single_comparison_figure(
    y_true, 
    y_pred_xgboost, 
    y_pred_other,
    model_name_other="Gradient Boosting",
    filename=None,
    sample_size=None
):
    """
    Figure de comparaison directe : XGBoost vs un autre modèle
    XGBoost en bleu (meilleur) et l'autre en orange (moins bon)
    """
    
    if sample_size:
        y_true = y_true[:sample_size]
        y_pred_xgboost = y_pred_xgboost[:sample_size]
        y_pred_other = y_pred_other[:sample_size]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Comparaison des modèles de prédiction', 
                 fontsize=16, fontweight='bold')
    
    x_axis = np.arange(len(y_true))
    
    # ===== XGBoost (Meilleur) =====
    ax1.plot(x_axis, y_true, label='Valeurs réelles', color='#1f77b4', 
             linewidth=2.5, alpha=0.9)
    ax1.plot(x_axis, y_pred_xgboost, label='Valeurs prédites', 
             color='#ff7f0e', linewidth=2.5, linestyle='--', alpha=0.9)
    
    # Zone d'erreur XGBoost
    ax1.fill_between(x_axis, y_true, y_pred_xgboost, 
                     color='red', alpha=0.1, label='Écart (Erreur)')
    
    mae_xgb = mean_absolute_error(y_true, y_pred_xgboost)
    rmse_xgb = np.sqrt(mean_squared_error(y_true, y_pred_xgboost))
    r2_xgb = r2_score(y_true, y_pred_xgboost)
    
    textstr_xgb = f'MAE: {mae_xgb:.3f} | RMSE: {rmse_xgb:.3f} | R²: {r2_xgb:.4f}'
    ax1.text(0.02, 0.95, textstr_xgb, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#C6DBEF', alpha=0.9, 
                     edgecolor='#2E86AB', linewidth=2))
    
    ax1.set_ylabel('Total des réservations', fontsize=12, fontweight='bold')
    ax1.set_title('Figure - Résultat graphique d\'algorithme XGBoost ⭐', 
                 fontsize=13, fontweight='bold', loc='left', pad=15, color='#2E86AB')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # ===== Modèle Comparé =====
    ax2.plot(x_axis, y_true, label='Valeurs réelles', color='#1f77b4', 
             linewidth=2.5, alpha=0.9)
    ax2.plot(x_axis, y_pred_other, label='Valeurs prédites', 
             color='#ff7f0e', linewidth=2.5, linestyle='--', alpha=0.9)
    
    # Zone d'erreur
    ax2.fill_between(x_axis, y_true, y_pred_other, 
                     color='red', alpha=0.1, label='Écart (Erreur)')
    
    mae_other = mean_absolute_error(y_true, y_pred_other)
    rmse_other = np.sqrt(mean_squared_error(y_true, y_pred_other))
    r2_other = r2_score(y_true, y_pred_other)
    
    textstr_other = f'MAE: {mae_other:.3f} | RMSE: {rmse_other:.3f} | R²: {r2_other:.4f}'
    ax2.text(0.02, 0.95, textstr_other, transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FED8B1', alpha=0.9,
                     edgecolor='#F18F01', linewidth=2))
    
    ax2.set_xlabel('Année-Mois', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total des réservations', fontsize=12, fontweight='bold')
    ax2.set_title(f'Figure - Résultat graphique d\'algorithme {model_name_other}', 
                 fontsize=13, fontweight='bold', loc='left', pad=15)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée : {filename}")
    
    plt.show()
    return fig


# ============================================================================
# 2. TABLE COMPARATIVE DES PERFORMANCES
# ============================================================================

def create_performance_comparison_table(
    y_true,
    predictions_dict,  # {"Model Name": y_pred_array}
    filename=None
):
    """
    Crée un tableau de comparaison des performances
    avec XGBoost mis en avant
    """
    
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        results.append({
            'Modèle': model_name,
            'MAE': f'{mae:.4f}',
            'RMSE': f'{rmse:.4f}',
            'R²': f'{r2:.4f}'
        })
    
    df_results = pd.DataFrame(results)
    
    # Créer une figure avec tableau
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    table = ax.table(
        cellText=df_results.values,
        colLabels=df_results.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style : XGBoost en vert clair, autres en gris
    for i, row in enumerate(df_results.itertuples(index=False)):
        if 'XGBoost' in row[0]:
            color = '#C6EFCE'
            text_weight = 'bold'
        else:
            color = '#E8E8E8'
            text_weight = 'normal'
        
        for j in range(len(row)):
            cell = table[(i+1, j)]
            cell.set_facecolor(color)
            cell.set_text_props(weight=text_weight)
    
    # En-tête en bleu
    for j in range(len(df_results.columns)):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Tableau Comparatif des Performances\n(Test Interne)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Tableau sauvegardé : {filename}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    return df_results


# ============================================================================
# 3. DIAGRAMME EN BARRES DES ERREURS
# ============================================================================

def create_error_bars_comparison(
    y_true,
    predictions_dict,
    metric='R2',
    filename=None
):
    """
    Diagramme en barres comparant les erreurs/performances
    XGBoost clairement en haut (meilleur R²) ou en bas (moins d'erreur)
    """
    
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        results.append({
            'model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparaison des Performances des Modèles', 
                fontsize=14, fontweight='bold')
    
    metrics = ['MAE', 'RMSE', 'R2']
    colors_base = ['#E74C3C', '#E74C3C', '#27AE60']  # Rouge pour erreurs, vert pour R2
    
    for ax, metric_name, color in zip(axes, metrics, colors_base):
        values = df[metric_name].values
        models = df['model'].values
        
        # XGBoost en surbrillance
        colors = [color if 'XGBoost' not in m else '#2E86AB' for m in models]
        
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'Métrique: {metric_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotation des labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {filename}")
    
    plt.show()


print("✅ Module de visualisations chargé avec succès!")
print("""
FONCTIONS DISPONIBLES:
  - create_comparison_figure_annual()     : Compare 3 modèles en figure annuelle
  - create_single_comparison_figure()     : Compare 2 modèles (XGBoost vs autre)
  - create_performance_comparison_table() : Tableau des métriques
  - create_error_bars_comparison()        : Diagramme en barres des erreurs
""")
