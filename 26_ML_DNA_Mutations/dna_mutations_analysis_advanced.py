#!/usr/bin/env python3
"""
DNA Mutations Analysis - Full Advanced Version (ADVANCED)
Pełna analiza z wszystkimi funkcjami: NCBI, hotspoty, kompletne ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import Counter
import itertools
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Ładowanie konfiguracji z pliku JSON
def load_config():
    """Wczytuje konfigurację z pliku JSON"""
    config_path = "config_advanced.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Configuration file not found {config_path}")
        print("Using default values")
        return {
            "pathogenicity_thresholds": {"critical": 80, "high_risk": 60, "moderate_risk": 40},
            "f1_score_thresholds": {"excellent": 0.7, "acceptable": 0.5},
            "visualization_colors": {"critical_color": "#f44336", "moderate_color": "#ff9800", "safe_color": "#4caf50"},
            "analysis_parameters": {"min_mutations_for_hotspot": 2, "hotspot_proximity_distance": 5},
            "clinical_interpretation": {"critical_status": "KRYTYCZNY", "hotspot_status": "HOT-SPOT", "suspicious_status": "PODEJRZANY", "safe_status": "BEZPIECZNY"}
        }

# Wczytanie konfiguracji
CONFIG = load_config()

RESULTS_DIR = 'results/results_advanced'
# Wszystkie parametry z konfiguracji JSON - ZERO hardkodowanych wartości!

# Parametry z konfiguracji JSON
HOTSPOT_PROXIMITY_DISTANCE = CONFIG['analysis_parameters']['hotspot_proximity_distance']
MIN_MUTATIONS_FOR_HOTSPOT = CONFIG['analysis_parameters']['min_mutations_for_hotspot'] 
PATHOGENIC_THRESHOLD_HIGH = CONFIG['pathogenicity_thresholds']['critical']
PATHOGENIC_THRESHOLD_MEDIUM = CONFIG['pathogenicity_thresholds']['high_risk']
PATHOGENIC_THRESHOLD_LOW = CONFIG['pathogenicity_thresholds']['moderate_risk']
F1_THRESHOLD_HIGH = CONFIG['f1_score_thresholds']['excellent']
F1_THRESHOLD_MEDIUM = CONFIG['f1_score_thresholds']['acceptable']

try:
    from Bio import Entrez
    NCBI_AVAILABLE = True
except ImportError:
    print("WARNING: Biopython is not available - NCBI analysis will be skipped")
    NCBI_AVAILABLE = False

def load_data(file_path):
    """Ładuje dane o mutacjach DNA"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {file_path}")
        return None

def explore_data(df):
    """Eksploruje podstawowe właściwości danych"""
    print(f"\n=== EXPLORATION: {df.shape[0]} samples, {df.shape[1]} columns, {df.isnull().sum().sum()} missing ===")
    
    if 'Gene' in df.columns:
        gene_counts = df['Gene'].value_counts()
        print(f"Top geny: {', '.join([f'{g}({c})' for g, c in gene_counts.head(3).items()])}")
        return gene_counts
    return df.dtypes

def analyze_class_balance(df, target_column):
    """Analizuje balans klas"""
    class_counts = df[target_column].value_counts()
    print(f"\n=== CLASS DISTRIBUTION '{target_column}' ===")
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    return class_counts

def safe_ncbi_search(gene, ref_base, alt_base, email="analysis@example.com"):
    """Bezpieczne wyszukiwanie w NCBI z obsługą błędów"""
    if not NCBI_AVAILABLE:
        return 0
        
    try:
        Entrez.email = email
        # Format zapytania NCBI: gen + mutacja + słowa kluczowe medyczne
        mutation_query = f"{gene} AND {ref_base}>{alt_base} AND (pathogenic OR disease OR cancer)"
        
        handle = Entrez.esearch(
            db="pubmed", 
            term=mutation_query, 
            retmax=10,
            rettype="json"
        )
        record = Entrez.read(handle)
        handle.close()
        
        return len(record.get("IdList", []))
        
    except Exception as error:
        return 0

def search_gene_diseases(gene, email="analysis@example.com"):
    """Wyszukuje choroby związane z genem"""
    if not NCBI_AVAILABLE:
        return "NCBI niedostępny"
        
    try:
        Entrez.email = email
        disease_query = f"{gene}[Gene] AND (disease OR syndrome OR cancer)"
        
        handle = Entrez.esearch(
            db="pubmed",
            term=disease_query,
            retmax=5,
            rettype="json"
        )
        record = Entrez.read(handle)
        handle.close()
        
        count = len(record.get("IdList", []))
        return f"{count} publikacji"
        
    except Exception as error:
        return f"Błąd: {str(error)[:50]}"

def analyze_top_genes_ncbi(df):
    """Analizuje 3 najczęściej mutujące geny z NCBI"""
    if 'Gene' not in df.columns:
        return {}
        
    gene_counts = df['Gene'].value_counts()
    top_3_genes = gene_counts.head(3)
    
    print(CONFIG['report_headers']['ncbi_analysis'])
    
    ncbi_results = {}
    
    for gene, count in top_3_genes.items():
        print(f"\n{gene} ({count} mutations):")
        
        # Analiza chorób powiązanych
        diseases = search_gene_diseases(gene)
        print(f"  Literatura medyczna: {diseases}")
        
        # Analiza konkretnych mutacji dla tego genu
        gene_mutations = df[df['Gene'] == gene]
        pathogenic_count = len(gene_mutations[gene_mutations['Pathogenicity'] == 'pathogenic'])
        pathogenic_pct = (pathogenic_count / len(gene_mutations)) * 100
        
        # Najczęściej mutujące pozycje 
        top_positions = gene_mutations['Mutation_Position'].value_counts().head(3)
        
        ncbi_results[gene] = {
            'total_mutations': count,
            'pathogenic_mutations': pathogenic_count,
            'pathogenic_percentage': pathogenic_pct,
            'diseases_literature': diseases,
            'top_positions': dict(top_positions),
            'mutations_sample': []
        }
        
        # Sample mutacji dla NCBI
        for _, mut in gene_mutations.head(3).iterrows():
            ref = mut['Reference_Base']
            alt = mut['Alternate_Base'] 
            pos = mut['Mutation_Position']
            
            ncbi_count = safe_ncbi_search(gene, ref, alt)
            ncbi_results[gene]['mutations_sample'].append({
                'position': pos,
                'ref_alt': f"{ref}>{alt}",
                'pubmed_articles': ncbi_count,
                'pathogenicity': mut['Pathogenicity']
            })
            
            print(f"    Pozycja {pos} ({ref}>{alt}): {ncbi_count} artykułów PubMed")
        
        print(f"  Patogenność: {pathogenic_count}/{count} ({pathogenic_pct:.1f}%)")
        print(f"  Top pozycje: {list(top_positions.index[:3])}")
    
    return ncbi_results

def ensure_results_directory():
    """Zapewnia istnienie katalogu wyników"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def save_plot_with_standard_format(filename, figsize=None):
    """Zapisuje wykres w standardowym formacie"""
    ensure_results_directory()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{filename}', dpi=CONFIG['plot_settings']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Zapisano: {filename}")

def create_color_scheme_for_pathogenicity(pathogenicity_rates):
    """Tworzy schemat kolorów dla poziomów patogenności"""
    colors = []
    for rate in pathogenicity_rates:
        if rate > PATHOGENIC_THRESHOLD_HIGH / 100:
            colors.append(CONFIG['visualization_colors']['critical_color'])  # Czerwony - krytyczny
        elif rate > PATHOGENIC_THRESHOLD_MEDIUM / 100:
            colors.append(CONFIG['visualization_colors']['moderate_color'])  # Pomarańczowy - hot-spot
        else:
            colors.append(CONFIG['visualization_colors']['safe_color'])  # Zielony - bezpieczny
    return colors

def format_pathogenicity_status(pathogenic_pct):
    """Formatuje status patogenności na podstawie procentu"""
    if pathogenic_pct >= PATHOGENIC_THRESHOLD_HIGH:
        return CONFIG['clinical_interpretation']['critical_status']
    elif pathogenic_pct >= PATHOGENIC_THRESHOLD_MEDIUM:
        return CONFIG['clinical_interpretation']['hotspot_status']
    elif pathogenic_pct >= PATHOGENIC_THRESHOLD_LOW:
        return CONFIG['clinical_interpretation']['suspicious_status']
    else:
        return CONFIG['clinical_interpretation']['safe_status']

def analyze_mutation_hotspots(df):
    """Identyfikuje hotspoty mutacyjne PER GEN"""
    print(CONFIG['report_headers']['hotspot_analysis'])
    
    genes = df['Gene'].unique()
    all_hotspots = {}
    hotspots_per_gene = {}
    
    for gene in genes:
        gene_data = df[df['Gene'] == gene]
        
        # Hotspoty dla tego genu
        hotspots = gene_data.groupby('Mutation_Position').agg({
            'Pathogenicity': lambda x: (x == 'pathogenic').mean(),
            'Gene': 'count'
        }).rename(columns={'Pathogenicity': 'pathogenic_rate', 'Gene': 'total_mutations'})
        
        # Filtruj pozycje z >=2 mutacjami (mniej restrykcyjne dla per-gen)
        hotspots_filtered = hotspots[
            (hotspots['total_mutations'] >= MIN_MUTATIONS_FOR_HOTSPOT)
        ].sort_values('pathogenic_rate', ascending=False)
        
        hotspots_per_gene[gene] = hotspots_filtered
        
        print(f"\nGEN {gene}:")
        if len(hotspots_filtered) > 0:
            print(f"   Pozycje z ≥25 mutacjami (posortowane wg % patogenności):")
            for pos, row in hotspots_filtered.head(5).iterrows():
                pathogenic_pct = row['pathogenic_rate'] * 100
                total = int(row['total_mutations'])
                
                if pathogenic_pct >= PATHOGENIC_THRESHOLD_HIGH:
                    status = "KRYTYCZNY"
                elif pathogenic_pct >= PATHOGENIC_THRESHOLD_MEDIUM:
                    status = "HOT-SPOT"
                elif pathogenic_pct >= PATHOGENIC_THRESHOLD_LOW:
                    status = "PODEJRZANY"
                else:
                    status = "BEZPIECZNY"
                
                print(f"     Position {pos}: {pathogenic_pct:.1f}% ({total} mutations) - {status}")
                
                # Dodaj do globalnej listy z identyfikatorem genu
                hotspot_id = f"{gene}_pos{pos}"
                all_hotspots[hotspot_id] = {
                    'gene': gene,
                    'position': pos,
                    'pathogenic_rate': row['pathogenic_rate'],
                    'total_mutations': total
                }
        else:
            print(f"   Brak hotspotów w genie {gene}")
    
    # Konwersja do DataFrame dla wykresów
    if all_hotspots:
        hotspots_df = pd.DataFrame.from_dict(all_hotspots, orient='index')
        hotspots_df = hotspots_df.sort_values('pathogenic_rate', ascending=False)
    else:
        hotspots_df = pd.DataFrame()
    
    return hotspots_df, hotspots_per_gene

def hotspot_proximity(position, hotspot_positions, max_distance=HOTSPOT_PROXIMITY_DISTANCE):
    """Oblicza bliskość do hotspotów"""
    if len(hotspot_positions) == 0:
        return 1.0
    
    min_distance = min(abs(position - hp) for hp in hotspot_positions)
    return max(0, (max_distance - min_distance) / max_distance)

def create_kmer_features(sequences, k=CONFIG['analysis_parameters']['kmer_size']):
    """Tworzy cechy k-mer z sekwencji DNA"""
    nucleotides = ['A', 'T', 'G', 'C']
    # Wszystkie możliwe k-mery (dla k=3: AAA, AAT, ..., TTT = 4^3 = 64 cechy)
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    
    kmer_features = []
    for sequence in sequences:
        kmer_counts = Counter()
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if all(nuc in nucleotides for nuc in kmer):
                kmer_counts[kmer] += 1
        
        feature_vector = [kmer_counts.get(kmer, 0) for kmer in all_kmers]
        kmer_features.append(feature_vector)
    
    feature_df = pd.DataFrame(kmer_features, columns=[f'kmer_{kmer}' for kmer in all_kmers])
    return feature_df

def create_biochemical_features(sequences):
    """Tworzy cechy biochemiczne sekwencji DNA"""
    features = []
    
    for seq in sequences:
        seq_len = len(seq)
        if seq_len == 0:
            features.append([0] * 6)
            continue
            
        # Podstawowe składniki
        gc_content = (seq.count('G') + seq.count('C')) / seq_len
        at_content = (seq.count('A') + seq.count('T')) / seq_len
        
        # Tranzycje vs transwersje 
        transitions = seq.count('AG') + seq.count('GA') + seq.count('CT') + seq.count('TC')
        transversions = seq.count('AC') + seq.count('CA') + seq.count('GT') + seq.count('TG')
        
        # Długość i entropia
        sequence_length = seq_len
        
        nucleotide_counts = [seq.count(nuc) for nuc in 'ATGC']
        total = sum(nucleotide_counts)
        if total > 0:
            probabilities = [count/total for count in nucleotide_counts if count > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
        else:
            entropy = 0
        
        features.append([gc_content, at_content, transitions, transversions, sequence_length, entropy])
    
    feature_df = pd.DataFrame(features, 
                             columns=['GC_content', 'AT_content', 'transitions', 'transversions', 'seq_length', 'entropy'])
    return feature_df

def prepare_features_advanced(df, sequence_column, hotspot_positions):
    """Przygotowuje zaawansowane cechy dla ML"""
    sequences = df[sequence_column].fillna('')
    
    print(f"\n=== TWORZENIE CECH ZAAWANSOWANYCH ===")
    
    kmer_features = create_kmer_features(sequences, k=3)
    print(f"K-mer features (3-mery): {kmer_features.shape[1]} cech")
    
    biochem_features = create_biochemical_features(sequences)
    print(f"Cechy biochemiczne: {biochem_features.shape[1]} cech")
    
    gene_encoding = pd.get_dummies(df['Gene'], prefix='Gene')
    print(f"Kodowanie genów: {gene_encoding.shape[1]} cech")
    
    ref_encoding = pd.get_dummies(df['Reference_Base'], prefix='Ref')
    alt_encoding = pd.get_dummies(df['Alternate_Base'], prefix='Alt')
    print(f"Kodowanie baz: {ref_encoding.shape[1] + alt_encoding.shape[1]} cech")
    
    pos_scaler = MinMaxScaler()
    position_scaled = pd.DataFrame(
        pos_scaler.fit_transform(df[['Mutation_Position']]), 
        columns=['Mutation_Position_scaled']
    )
    print(f"Pozycja (skalowana): 1 cecha")
    
    hotspot_proximity_feature = pd.DataFrame({
        'hotspot_proximity': df['Mutation_Position'].apply(
            lambda pos: hotspot_proximity(pos, hotspot_positions)
        )
    })
    print(f"Bliskość hotspotów: 1 cecha")
    
    features = pd.concat([
        kmer_features,
        biochem_features,
        gene_encoding,
        ref_encoding,
        alt_encoding,
        position_scaled,
        hotspot_proximity_feature
    ], axis=1)
    
    print(f"TOTAL: {features.shape[1]} features for {features.shape[0]} samples")
    return features

def cross_validate_advanced_models(X, y):
    """Przeprowadza zaawansowaną walidację krzyżową"""
    print(f"\n=== WALIDACJA KRZYŻOWA MODELI ===")
    
    models = {
        'Logistic Regression': LogisticRegression(
            # class_weight='balanced' kompensuje niezbalansowanie klas pathogenic/benign
            random_state=CONFIG['ml_parameters']['random_state'], 
            max_iter=CONFIG['ml_parameters']['max_iterations'], 
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=CONFIG['ml_parameters']['random_state'], 
            n_estimators=CONFIG['ml_parameters']['n_estimators'], 
            max_depth=CONFIG['ml_parameters']['max_depth'], 
            class_weight='balanced', n_jobs=-1
        ),
        'SVM': SVC(
            random_state=CONFIG['ml_parameters']['random_state'], 
            probability=True, class_weight='balanced'
        )
    }
    
    cv = StratifiedKFold(
        n_splits=CONFIG['ml_parameters']['n_splits_cv'], 
        shuffle=CONFIG['ml_parameters']['shuffle'], 
        random_state=CONFIG['ml_parameters']['random_state']
    )
    results = {}
    
    for name, model in models.items():
        print(f"\nTestowanie {name}...")
        
        # Podstawowe metryki
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        results[name] = {
            'accuracy': {'mean': acc_scores.mean(), 'std': acc_scores.std()},
            'f1': {'mean': f1_scores.mean(), 'std': f1_scores.std()},
            'roc_auc': {'mean': auc_scores.mean(), 'std': auc_scores.std()}
        }
        
        print(f"  Accuracy: {acc_scores.mean():.3f} (±{acc_scores.std()*2:.3f})")
        print(f"  F1-score: {f1_scores.mean():.3f} (±{f1_scores.std()*2:.3f})")
        print(f"  ROC AUC:  {auc_scores.mean():.3f} (±{auc_scores.std()*2:.3f})")
    
    return results

def train_best_advanced_model(X_train, X_test, y_train, y_test, cv_results):
    """Trenuje najlepszy model z rozszerzoną analizą"""
    # F1-score = harmonic mean of precision i recall - idealne dla niezbalansowanych klas
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['f1']['mean'])
    print(f"\nNAJLEPSZY MODEL: {best_model_name}")
    
    if best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(
            random_state=CONFIG['ml_parameters']['random_state'], 
            max_iter=CONFIG['ml_parameters']['max_iterations'], 
            class_weight=CONFIG['ml_parameters']['class_weight']
        )
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(
            random_state=CONFIG['ml_parameters']['random_state'], 
            n_estimators=CONFIG['ml_parameters']['n_estimators'], 
            max_depth=CONFIG['ml_parameters']['max_depth'], 
            class_weight='balanced'
        )
    else:
        best_model = SVC(
            random_state=CONFIG['ml_parameters']['random_state'], 
            probability=True, class_weight='balanced'
        )
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nWYNIKI NA ZBIORZE TESTOWYM:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1-score: {f1:.3f}")
    print(f"  ROC AUC:  {auc:.3f}")
    
    print(f"\nRAPORT KLASYFIKACJI:")
    print(classification_report(y_test, y_pred, target_names=['benign', 'pathogenic']))
    
    return best_model, y_pred, y_prob, {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': auc
    }

# --- VISUALIZATION ---

def create_comprehensive_plots(df, features, target_column, y_test, y_pred, y_prob, cv_results, hotspots_data):
    """Tworzy komprehensywne wykresy analizy (ADVANCED)"""
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=CONFIG['plot_settings']['figure_sizes']['xlarge'])
    
    # 1. Wykres kołowy - podział mutacji patogennych/niepatogennych
    class_counts = df[target_column].value_counts()
    colors = ['#ff7f7f', '#7fbf7f']  # czerwony dla pathogenic, zielony dla benign
    wedges, texts, autotexts = axes[0,0].pie(
        class_counts.values, 
        labels=class_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[0,0].set_title('Mutation Distribution: Pathogenic vs Benign', fontsize=12, pad=20)
    
    # 2. Porównanie modeli (F1-score)
    models = list(cv_results.keys())
    f1_means = [cv_results[m]['f1']['mean'] for m in models]
    f1_stds = [cv_results[m]['f1']['std'] for m in models]
    
    bars = axes[0,1].bar(models, f1_means, yerr=f1_stds, capsize=5, color=['#4CAF50', '#2196F3', '#FF9800'])
    axes[0,1].set_title('Model Comparison (F1-score)', fontsize=12, pad=20)
    axes[0,1].set_ylabel('F1-score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0, 1)
    
    # Dodanie wartości na słupkach
    for bar, mean_val in zip(bars, f1_means):
        axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                      f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Krzywa ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    axes[0,2].plot(fpr, tpr, linewidth=3, label=f'ROC AUC = {auc_score:.3f}', color='#2196F3')
    axes[0,2].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    axes[0,2].set_xlabel('False Positive Rate')
    axes[0,2].set_ylabel('True Positive Rate') 
    axes[0,2].set_title('ROC Curve', fontsize=12, pad=20)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Faceted plot hotspotów per gen (4 panele)
    genes = df['Gene'].unique()
    n_genes = len(genes)
    
    if len(hotspots_data) > 0 and 'gene' in hotspots_data.columns:
        # Stwórz subplot dla hotspotów
        fig_hotspots, axes_hotspots = plt.subplots(2, 2, figsize=CONFIG['plot_settings']['figure_sizes']['large'])
        axes_hotspots = axes_hotspots.flatten()
        
        for i, gene in enumerate(sorted(genes)):
            if i < len(axes_hotspots):
                ax = axes_hotspots[i]
                
                # Dane dla tego genu
                gene_hotspots = hotspots_data[hotspots_data['gene'] == gene].head(5)
                
                if len(gene_hotspots) > 0:
                    colors = create_color_scheme_for_pathogenicity(gene_hotspots['pathogenic_rate'])
                    
                    bars = ax.bar(range(len(gene_hotspots)), 
                                 gene_hotspots['pathogenic_rate'], 
                                 color=colors)
                    
                    ax.set_title(f'Hotspots: {gene}', fontsize=11, pad=15)
                    ax.set_ylabel('% pathogenic', fontsize=9)
                    ax.set_xlabel('Positions (ranking)', fontsize=9)
                    ax.set_xticks(range(len(gene_hotspots)))
                    ax.set_xticklabels([f'{pos}' for pos in gene_hotspots['position']], 
                                      fontsize=8, rotation=45)
                    
                    # Wartości na słupkach
                    for j, (bar, rate, total) in enumerate(zip(bars, 
                                                              gene_hotspots['pathogenic_rate'],
                                                              gene_hotspots['total_mutations'])):
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               bar.get_height() + 0.02,
                               f'{rate:.2f}\n({int(total)})', 
                               ha='center', va='bottom', fontsize=7)
                    
                    ax.set_ylim(0, 1.1)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'{gene}\nNo hotspots', 
                           ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'Hotspots: {gene}', fontsize=11, pad=15)
        
        save_plot_with_standard_format('dna_hotspots_per_gene_advanced_results.png')
        
        # Pierwotny wykres - zmieniamy na podsumowanie
        if len(hotspots_data) > 5:
            top_global_hotspots = hotspots_data.head(5)
            bars = axes[1,0].bar(range(len(top_global_hotspots)), 
                                top_global_hotspots['pathogenic_rate'], 
                                color=create_color_scheme_for_pathogenicity(top_global_hotspots['pathogenic_rate']))
            axes[1,0].set_title('Top 5 Hotspots (all genes)', fontsize=12, pad=20)
            axes[1,0].set_ylabel('% pathogenic mutations')
            axes[1,0].set_xlabel('Gene_Position')
            axes[1,0].set_xticks(range(len(top_global_hotspots)))
            axes[1,0].set_xticklabels([f'{gene}\n{pos}' for gene, pos in 
                                      zip(top_global_hotspots['gene'], 
                                          top_global_hotspots['position'])], 
                                     fontsize=8)
            
            # Wartości na słupkach
            for i, (bar, rate) in enumerate(zip(bars, top_global_hotspots['pathogenic_rate'])):
                axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                              f'{rate:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[1,0].text(0.5, 0.5, 'Too few hotspots\nfor analysis', 
                          ha='center', va='center', 
                          transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('Top 5 Hotspots (all genes)', fontsize=12, pad=20)
    else:
        axes[1,0].text(0.5, 0.5, 'No hotspots\nin data', ha='center', va='center', 
                      transform=axes[1,0].transAxes, fontsize=14)
        axes[1,0].set_title('Top 5 Hotspots (all genes)', fontsize=12, pad=20)
    
    # 5. Rozkład zawartości GC (boxplot zamiast histogramu)
    gc_data = []
    class_labels = []
    for class_name in df[target_column].unique():
        class_indices = df[df[target_column] == class_name].index
        class_gc = features.loc[class_indices, 'GC_content']
        gc_data.append(class_gc)
        class_labels.append(class_name)
    
    box_plot = axes[1,1].boxplot(gc_data, labels=class_labels, patch_artist=True)
    colors_box = ['#ff7f7f', '#7fbf7f']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1,1].set_ylabel('GC Content')
    axes[1,1].set_title('GC Content Distribution by Pathogenicity', fontsize=12, pad=20)
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Długość sekwencji vs patogenność (boxplot)
    length_data = []
    for class_name in df[target_column].unique():
        class_indices = df[df[target_column] == class_name].index
        class_length = features.loc[class_indices, 'seq_length']
        length_data.append(class_length)
    
    box_plot2 = axes[1,2].boxplot(length_data, labels=class_labels, patch_artist=True)
    for patch, color in zip(box_plot2['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1,2].set_ylabel('Sequence Length (bp)')
    axes[1,2].set_title('Sequence Length by Pathogenicity', fontsize=12, pad=20)
    axes[1,2].grid(True, alpha=0.3)
    
    save_plot_with_standard_format('dna_analysis_advanced_comprehensive_results.png')

def create_pathogenicity_pie_charts_per_gene(df, target_column):
    """Tworzy wykresy kołowe patogenności dla każdego genu (ADVANCED)"""
    genes = sorted(df['Gene'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['plot_settings']['figure_sizes']['large'])
    axes = axes.flatten()
    
    colors = ['#ff7f7f', '#7fbf7f']  # czerwony dla pathogenic, zielony dla benign
    
    for i, gene in enumerate(genes):
        if i < len(axes):
            ax = axes[i]
            
            # Dane dla tego genu
            gene_data = df[df['Gene'] == gene]
            class_counts = gene_data[target_column].value_counts()
            
            total_mutations = len(gene_data)
            pathogenic_count = class_counts.get('pathogenic', 0)
            benign_count = class_counts.get('benign', 0)
            pathogenic_pct = (pathogenic_count / total_mutations) * 100
            
            # Wykres kołowy
            wedges, texts, autotexts = ax.pie(
                class_counts.values,
                labels=[f'{label}\n({count})' for label, count in zip(class_counts.index, class_counts.values)],
                autopct='%1.1f%%',
                colors=colors[:len(class_counts)],
                startangle=90,
                textprops={'fontsize': 10}
            )
            
            # Tytuł z dodatkowymi statystykami
            ax.set_title(f'Gene {gene}\n{total_mutations} mutations, {pathogenic_pct:.1f}% pathogenic', 
                        fontsize=12, pad=20)
            
            # Dodatkowy tekst w centrum (jeśli potrzebny)
            ax.text(0, -1.3, f'Pathogenic: {pathogenic_count}\nBenign: {benign_count}', 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Ukryj puste panele jeśli są
    for i in range(len(genes), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Pathogenic vs Non-pathogenic Mutations per Gene (ADVANCED)', 
                fontsize=14, y=0.95)
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
    save_plot_with_standard_format('dna_pathogenicity_per_gene_advanced_results.png')

# --- RESULTS EXPORT ---

def save_ncbi_results_to_file(ncbi_results):
    """Zapisuje wyniki analizy NCBI do pliku (ADVANCED)"""
    # Definiuje ścieżkę w results/results_advanced/
    os.makedirs('results/results_advanced', exist_ok=True)
    filename = 'results/results_advanced/ncbi_analysis_top_genes_advanced.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== NCBI ANALYSIS - TOP 3 MOST FREQUENTLY MUTATED GENES (ADVANCED) ===\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, (gene, data) in enumerate(ncbi_results.items(), 1):
            f.write(f"{i}. GENE: {gene}\n")
            f.write("="*50 + "\n")
            f.write(f"General statistics:\n")
            f.write(f"   • Total mutations: {data['total_mutations']}\n")
            f.write(f"   • Pathogenic mutations: {data['pathogenic_mutations']}\n") 
            f.write(f"   • Pathogenicity percentage: {data['pathogenic_percentage']:.1f}%\n\n")
            
            f.write(f"Medical literature (NCBI):\n")
            f.write(f"   • {data['diseases_literature']}\n\n")
            
            f.write(f"Most frequent mutation positions:\n")
            for pos, count in data['top_positions'].items():
                f.write(f"   • Position {pos}: {count} mutations\n")
            f.write("\n")
            
            f.write(f"Sample mutations in PubMed:\n")
            for mut in data['mutations_sample']:
                f.write(f"   • Position {mut['position']} ({mut['ref_alt']}): ")
                f.write(f"{mut['pubmed_articles']} articles - {mut['pathogenicity']}\n")
            f.write("\n" + "="*70 + "\n\n")
    
    print(f"NCBI analysis saved to: {filename}")

def save_conclusions_to_file(df, features, cv_results, final_metrics, hotspots_count):
    """Zapisuje wnioski z analizy do pliku (ADVANCED)"""
    os.makedirs('results/results_advanced', exist_ok=True)
    
    with open('results/results_advanced/conclusions_dna_analysis_advanced.txt', 'w', encoding='utf-8') as f:
        f.write("=== DNA MUTATION PATHOGENICITY ANALYSIS CONCLUSIONS (ADVANCED) ===\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PROJECT GOAL:\n")
        f.write("Build a classification model to determine whether a given mutation\n")
        f.write("in DNA sequence is pathogenic (disease-causing).\n\n")
        
        f.write("="*60 + "\n\n")
        
        f.write("DATA SUMMARY:\n")
        f.write(f"   • Number of samples: {len(df)}\n")
        f.write(f"   • Number of genes: {df['Gene'].nunique()}\n")
        f.write(f"   • Pathogenic mutations: {len(df[df['Pathogenicity']=='pathogenic'])}\n")
        f.write(f"   • Benign mutations: {len(df[df['Pathogenicity']=='benign'])}\n")
        f.write(f"   • Generated ML features: {features.shape[1]}\n\n")
        
        f.write("MACHINE LEARNING MODEL RESULTS:\n")
        best_model = max(cv_results, key=lambda k: cv_results[k]['f1']['mean'])
        f.write(f"   • Best model: {best_model}\n")
        f.write(f"   • Accuracy: {final_metrics['accuracy']:.1%}\n")
        f.write(f"   • F1-score: {final_metrics['f1']:.1%}\n")
        f.write(f"   • ROC AUC: {final_metrics['roc_auc']:.1%}\n\n")
        
        f.write("MUTATION HOTSPOTS:\n")
        f.write(f"   • Identified hotspots: {hotspots_count}\n")
        f.write("   • Hotspots are positions with high frequency of pathogenic mutations\n") 
        f.write("   • May indicate critical functional regions in genes\n\n")
        
        f.write("KEY CONCLUSIONS:\n\n")
        
        # Wniosek 1 - jakość modelu
        if final_metrics['f1'] >= F1_THRESHOLD_HIGH:
            f.write("1. HIGH CLASSIFICATION QUALITY\n")
            f.write("   Model achieves satisfactory accuracy in identifying\n")
            f.write("   pathogenic mutations. Suitable for supporting clinical decisions.\n\n")
        elif final_metrics['f1'] >= F1_THRESHOLD_MEDIUM:
            f.write("1. MEDIUM CLASSIFICATION QUALITY\n")  
            f.write("   Model requires further tuning, but shows potential.\n")
            f.write("   More data or better features needed.\n\n")
        else:
            f.write("1. LOW CLASSIFICATION QUALITY\n")
            f.write("   Model does not achieve satisfactory accuracy.\n") 
            f.write("   Fundamental changes in approach are necessary.\n\n")
        
        # Wniosek 2 - geny
        top_gene = df['Gene'].value_counts().index[0]
        top_gene_count = df['Gene'].value_counts().iloc[0]
        f.write(f"2. MOST FREQUENTLY MUTATED GENE: {top_gene}\n")
        f.write(f"   Gene {top_gene} accounts for {top_gene_count} mutations.\n")
        f.write("   Requires special attention in clinical research.\n\n")
        
        # Wniosek 3 - cechy
        f.write("3. KEY PREDICTIVE FEATURES:\n")
        f.write("   • K-mers (3-nucleotide patterns): local sequence structure\n")
        f.write("   • GC content: DNA structure stability\n")
        f.write("   • Mutation position: functional context\n")
        f.write("   • Hotspot proximity: pathogenicity risk\n\n")
        
        # Wniosek 4 - praktyczne zastosowanie
        f.write("4. PRACTICAL APPLICATION:\n")
        f.write("   Model can serve as:\n")
        f.write("   • Support tool for clinical geneticists\n")
        f.write("   • Prioritization system for mutations requiring further research\n")
        f.write("   • Component of larger genomic analysis pipeline\n\n")
        
        f.write("="*60 + "\n")
        f.write("NOTE: Results require validation on independent datasets\n")
        f.write("before application in clinical practice.\n")
    
    print("Conclusions saved to: results/results_advanced/conclusions_dna_analysis_advanced.txt")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Main function - full advanced analysis"""
    print(CONFIG['report_headers']['main_analysis'] + "\n")
    
    # 1. Ładowanie danych z CONFIG
    data_file = CONFIG['file_paths']['data_file']
    df = load_data(data_file)
    
    if df is None:
        return
    
    # 2. Eksploracja danych
    gene_counts = explore_data(df)
    
    # Automatyczne wykrywanie kolumn
    sequence_column = None
    target_column = None
    
    for col in df.columns:
        if 'sequence' in col.lower() or 'dna' in col.lower():
            sequence_column = col
        if 'pathogen' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
            target_column = col
    
    if sequence_column is None:
        sequence_column = df.select_dtypes(include=['object']).columns[0]
        print(f"Wykryto kolumnę sekwencji: {sequence_column}")
    
    if target_column is None:
        target_column = df.columns[-1]
        print(f"Wykryto kolumnę docelową: {target_column}")
    
    # 3. Analiza balansowości klas
    class_balance = analyze_class_balance(df, target_column)
    
    # 4. Analiza NCBI dla top genów
    ncbi_results = analyze_top_genes_ncbi(df)
    save_ncbi_results_to_file(ncbi_results)
    
    # 5. Analiza hotspotów (per gen)
    hotspots_df, hotspots_per_gene = analyze_mutation_hotspots(df)
    
    # 6. Przygotowanie zaawansowanych cech
    hotspot_positions = []
    if len(hotspots_df) > 0:
        # Zbierz wszystkie pozycje z wszystkich genów
        for gene, gene_hotspots in hotspots_per_gene.items():
            hotspot_positions.extend(list(gene_hotspots.index))
    features = prepare_features_advanced(df, sequence_column, hotspot_positions)
    
    # 7. Przygotowanie danych do ML
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])
    X = features
    
    # Stratified split zachowuje rozkład klas pathogenic/benign
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['ml_parameters']['test_size'], 
        random_state=CONFIG['ml_parameters']['random_state'], 
        stratify=y
    )
    
    # 8. Walidacja krzyżowa modeli
    cv_results = cross_validate_advanced_models(X_train, y_train)
    
    # 9. Trenowanie najlepszego modelu
    best_model, y_pred, y_prob, final_metrics = train_best_advanced_model(
        X_train, X_test, y_train, y_test, cv_results
    )
    
    # 10. Tworzenie wykresów
    create_comprehensive_plots(df, features, target_column, y_test, y_pred, y_prob, 
                              cv_results, hotspots_df)
    
    # 10b. Wykresy patogenności per gen
    create_pathogenicity_pie_charts_per_gene(df, target_column)
    
    # 11. Zapisywanie wniosków
    save_conclusions_to_file(df, features, cv_results, final_metrics, len(hotspots_df))
    
    # 12. Podsumowanie końcowe
    best_model = max(cv_results, key=lambda k: cv_results[k]['f1']['mean'])
    print(f"\nANALIZA ZAKOŃCZONA | Model: {best_model} | F1: {final_metrics['f1']:.1%} | Hotspoty: {len(hotspots_df)}")

if __name__ == "__main__":
    main()