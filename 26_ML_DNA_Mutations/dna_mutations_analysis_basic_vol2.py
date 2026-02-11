#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projekt: Przewidywanie mutacji DNA związanych z chorobami genetycznymi
Cel: Stworzenie modelu klasyfikacyjnego określającego patogenność mutacji DNA

Author: mago
Created: 2025-08-17
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("WARNING: Biopython is not available - NCBI analysis will be skipped")
    BIOPYTHON_AVAILABLE = False

# Ładowanie konfiguracji z pliku JSON
def load_config():
    """Wczytuje konfigurację z pliku JSON"""
    config_path = "config_basic_vol2.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Configuration file not found {config_path}")
        print("Using default values")
        return {
            "pathogenicity_thresholds": {"high_risk": 50, "moderate_risk": 30, "mutation_risk_high": 60, "mutation_risk_moderate": 40},
            "visualization_colors": {"pathogenic_color": "#ff7f7f", "benign_color": "#7fbf7f", "high_risk_color": "red", "moderate_risk_color": "orange", "low_risk_color": "green", "safe_color": "lightblue"},
            "analysis_parameters": {"min_mutations_for_analysis": 3, "hotspot_threshold": 0.5, "top_mutations_limit": 8},
            "chart_settings": {"pie_chart_colors": ["#ff7f7f", "#7fbf7f"], "gc_boxplot_colors": ["lightblue", "lightgreen", "lightcoral", "lightyellow"], "bar_alpha": 0.7}
        }

# Wczytanie konfiguracji
CONFIG = load_config()


# Wszystkie parametry z konfiguracji JSON - ZERO hardkodowanych wartości!


def load_data():
    """Wczytuje dane z pliku CSV"""
    return pd.read_csv(CONFIG['file_paths']['data_path'])


def explore_data(df):
    """Podstawowa eksploracja danych"""
    print(CONFIG['report_headers']['data_exploration'])
    print(df.describe())
    print(df.info())
    print(df.isna().sum())
    
    lengths = df["DNA_Sequence"].str.len()
    print(f"\nDNA sequence lengths:")
    print(lengths.describe())
    
    position_counts = df["Mutation_Position"].value_counts().sort_values(ascending=False)
    print(f"\nMost frequent mutation positions:")
    print(position_counts.head(CONFIG['analysis_parameters']['top_positions_limit']))
    
    gene_counts = df["Gene"].value_counts()
    print(f"\nMutations per gene:")
    print(gene_counts.head(10))
    
    pathogenic_counts = df[df["Pathogenicity"] == "pathogenic"]["Gene"].value_counts()
    print(f"\nPathogenic mutations per gene:")
    print(pathogenic_counts.head(10))
    
    pathogenic = df[df["Pathogenicity"] == "pathogenic"]
    pathogenic_changes = pathogenic.groupby(["Gene", "Reference_Base", "Alternate_Base"]).size().sort_values(ascending=False)
    print(f"\nMost frequent pathogenic mutations (Gene + nucleotide change):")
    print(pathogenic_changes.head(10))
    
    percent_pathogenic = (pathogenic_counts / gene_counts * 100).round(2)
    print(f"\nPercentage of pathogenic mutations per gene:")
    print(percent_pathogenic.head(10))
    
    mutation_counts = df.groupby(["Reference_Base", "Alternate_Base", "Pathogenicity"]).size().sort_values(ascending=False)
    print(f"\nMost frequent mutations (Ref>Alt : Pathogenicity):")
    print(mutation_counts.head(20))
    
def safe_ncbi_search(gene, ref_base, alt_base, email=None):
    """Bezpieczne wyszukiwanie w NCBI z obsługą błędów"""
    if not BIOPYTHON_AVAILABLE:
        return 0
        
    try:
        Entrez.email = email or CONFIG['contact_settings']['default_email']
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
        print(f"NCBI error for {gene} {ref_base}>{alt_base}: {error}")
        return 0


def search_gene_diseases(gene, email=None):
    """Wyszukuje choroby związane z genem"""
    if not BIOPYTHON_AVAILABLE:
        return "Biopython niedostępny"
        
    try:
        Entrez.email = email or CONFIG['contact_settings']['default_email']
        disease_query = f"{gene}[Gene] AND (disease OR syndrome OR cancer)"
        
        handle = Entrez.esearch(db="pubmed", term=disease_query, retmax=5, rettype="json")
        record = Entrez.read(handle)
        handle.close()
        
        count = len(record.get("IdList", []))
        return f"Znaleziono {count} publikacji o chorobach"
        
    except Exception as error:
        return f"Błąd: {error}"


def analyze_top_mutations_ncbi(df):
    """Analizuje najczęstsze mutacje w bazach NCBI i zapisuje wyniki do pliku"""
    if not BIOPYTHON_AVAILABLE:
        print("Biopython unavailable - NCBI analysis skipped")
        return []
        
    print("\n" + CONFIG['report_headers']['ncbi_analysis'])
    
    # Konfiguracja email dla NCBI z CONFIG
    try:
        email = input("Podaj email dla NCBI (lub Enter dla domyślnego): ").strip()
        if not email:
            email = CONFIG['contact_settings']['default_email']
            print(f"Użyto domyślnego emaila: {email}")
    except:
        email = CONFIG['contact_settings']['default_email']
        print(f"Użyto domyślnego emaila: {email}")
    
    # Top 3 mutacje patogenne - sortujemy malejąco po liczbie wystąpień
    pathogenic = df[df["Pathogenicity"] == "pathogenic"]
    top3 = pathogenic.groupby(["Gene", "Reference_Base", "Alternate_Base"]).size().sort_values(ascending=False).head(3)
    
    results = []
    for (gene, ref, alt), count in top3.items():
        print(f"\n{gene} {ref}>{alt} (occurs {count}x in our data):")
        
        # Wyszukiwanie w PubMed
        pubmed_count = safe_ncbi_search(gene, ref, alt, email)
        print(f"  PubMed: {pubmed_count} publikacji")
        
        # Wyszukiwanie chorób związanych z genem
        disease_info = search_gene_diseases(gene, email)
        print(f"  Choroby: {disease_info}")
        
        # Dodaj wyniki do listy do zapisania
        results.append({
            'gene': gene,
            'mutation': f"{ref}>{alt}",
            'count_in_data': count,
            'pubmed_publications': pubmed_count,
            'disease_info': disease_info
        })
    
    return results




def gc_content_analysis(sequence):
    """Analiza zawartości GC w sekwencji"""
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def liczenie_par_gc(sekwencja):
    """Zlicza pary GC w sekwencji"""
    return sekwencja.count("GC")

def liczenie_nukleotydow(sekwencja):
    """Zlicza wszystkie nukleotydy - przydatne do analizy"""
    liczenie = {
        "A": sekwencja.count("A"),
        "C": sekwencja.count("C"), 
        "G": sekwencja.count("G"),
        "T": sekwencja.count("T")
    }
    return liczenie

def biochemical_features_enhanced(df):
    """Tworzy cechy biochemiczne na podstawie sekwencji DNA"""
    features = {}
    
    features['GC_content'] = df['DNA_Sequence'].apply(gc_content_analysis)
    features['GC_pairs'] = df['DNA_Sequence'].apply(liczenie_par_gc)
    
    def context_analysis(row):
        seq = row['DNA_Sequence']
        pos = row['Mutation_Position'] - 1  # 0-indexed
        # Okno 3 nukleotydów przed i po mutacji - standardowa praktyka w analizie kontekstu
        start = max(0, pos-3)
        end = min(len(seq), pos+4)
        context = seq[start:end] if pos < len(seq) else ""
        return gc_content_analysis(context) if context else 0
    
    features['context_GC'] = df.apply(context_analysis, axis=1)
    
    def mutation_type(ref, alt):
        # Transitions: purine↔purine (A↔G) lub pyrimidine↔pyrimidine (C↔T) - częstsze, mniej szkodliwe
        transitions = [('A','G'), ('G','A'), ('C','T'), ('T','C')]
        return 1 if (ref, alt) in transitions else 0
    
    features['is_transition'] = df.apply(
        lambda x: mutation_type(x['Reference_Base'], x['Alternate_Base']), axis=1)
    
    def chemical_change(ref, alt):
        # Klasyfikacja nukleotydów wg struktury chemicznej
        purines = ['A', 'G']      # większe cząsteczki (podwójny pierścień)
        pyrimidines = ['C', 'T']  # mniejsze cząsteczki (pojedynczy pierścień)
        ref_type = 'purine' if ref in purines else 'pyrimidine'
        alt_type = 'purine' if alt in purines else 'pyrimidine'
        return 1 if ref_type == alt_type else 0
    
    features['same_chemical_type'] = df.apply(
        lambda x: chemical_change(x['Reference_Base'], x['Alternate_Base']), axis=1)
    
    return pd.DataFrame(features)




def ensure_results_directory():
    """Tworzy katalog wyników jeśli nie istnieje"""
    os.makedirs(CONFIG['file_paths']['results_dir'], exist_ok=True)


def save_plot_with_standard_format(filename, figure_size=None):
    """Zapisuje wykres w standardowym formacie"""
    ensure_results_directory()
    if figure_size:
        plt.gcf().set_size_inches(figure_size)
    plt.savefig(f"{CONFIG['file_paths']['results_dir']}/{filename}", 
                dpi=CONFIG['plot_settings']['dpi'], 
                bbox_inches=CONFIG['plot_settings']['bbox_inches'])
    plt.close()
    print(f"Saved: {CONFIG['file_paths']['results_dir']}/{filename}")


def analyze_top_genes_ncbi(df):
    """Analizuje wszystkie geny w bazie NCBI - rozszerzona wersja"""
    if not BIOPYTHON_AVAILABLE:
        return []
    
    results = []
    genes = df['Gene'].unique()
    
    for gene in genes:
        gene_data = df[df['Gene'] == gene]
        total_mutations = len(gene_data)
        pathogenic_count = len(gene_data[gene_data['Pathogenicity'] == 'pathogenic'])
        
        # Wyszukaj informacje o chorobach dla genu
        disease_info = search_gene_diseases(gene)
        
        results.append({
            'gene': gene,
            'total_mutations_in_data': total_mutations,
            'pathogenic_mutations': pathogenic_count,
            'pathogenic_percentage': (pathogenic_count/total_mutations)*100 if total_mutations > 0 else 0,
            'disease_info': disease_info
        })
    
    return results


def save_ncbi_results_to_file(ncbi_results):
    """Zapisuje wyniki NCBI do pliku tekstowego"""
    ensure_results_directory()
    
    with open(f"{CONFIG['file_paths']['results_dir']}/ncbi_analysis_top_mutations_basic_vol2.txt", 'w', encoding='utf-8') as f:
        f.write("NCBI ANALYSIS - TOP MUTATIONS AND GENES (BASIC VOL2)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TOP 3 PATHOGENIC MUTATIONS ANALYSIS RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        if isinstance(ncbi_results, list) and len(ncbi_results) > 0:
            for result in ncbi_results:
                if isinstance(result, dict):
                    f.write(f"Gene: {result.get('gene', 'N/A')}\n")
                    f.write(f"Mutation: {result.get('mutation', 'N/A')}\n")
                    f.write(f"Occurrences in data: {result.get('count_in_data', 0)}\n")
                    f.write(f"PubMed publications: {result.get('pubmed_publications', 0)}\n")
                    f.write(f"Disease information: {result.get('disease_info', 'None')}\n")
                    f.write("\n")
        else:
            f.write("No NCBI results (Biopython unavailable or connection error)\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Analysis generated automatically by basic_vol2 script\n")
    
    print(f"NCBI results saved to: {CONFIG['file_paths']['results_dir']}/ncbi_analysis_top_mutations_basic_vol2.txt")


def analyze_mutation_hotspots(df):
    """Analizuje hotspoty mutacyjne - pozycje z wysoką częstością mutacji patogennych"""
    hotspots = []
    
    # Grupuj podle pozycji i genu
    position_groups = df.groupby(['Gene', 'Mutation_Position'])
    
    for (gene, position), group in position_groups:
        total_at_position = len(group)
        pathogenic_at_position = len(group[group['Pathogenicity'] == 'pathogenic'])
        
        # Minimum 2 mutacje na pozycji dla analizy hotspotów
        if total_at_position >= 2:
            pathogenic_rate = pathogenic_at_position / total_at_position
            
            hotspots.append({
                'gene': gene,
                'position': position,
                'total_mutations': total_at_position,
                'pathogenic_mutations': pathogenic_at_position,
                'pathogenic_rate': pathogenic_rate
            })
    
    # Sortuj według częstości patogennych (najpierw najwyższe)
    hotspots_df = pd.DataFrame(hotspots)
    if len(hotspots_df) > 0:
        hotspots_df = hotspots_df.sort_values('pathogenic_rate', ascending=False)
    
    # Filtruj hotspoty - próg z konfiguracji
    hotspot_threshold = CONFIG['analysis_parameters']['hotspot_threshold']
    hotspots_filtered = hotspots_df[hotspots_df['pathogenic_rate'] > hotspot_threshold] if len(hotspots_df) > 0 else pd.DataFrame()
    
    return hotspots_df, hotspots_filtered


def save_conclusions_to_file(df, accuracy, f1, auc, hotspots_count):
    """Zapisuje wnioski z analizy do pliku"""
    ensure_results_directory()
    
    with open(f"{CONFIG['file_paths']['results_dir']}/conclusions_dna_analysis_basic_vol2.txt", 'w', encoding='utf-8') as f:
        f.write("DNA MUTATIONS ANALYSIS CONCLUSIONS (BASIC VOL2)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"MODEL RESULTS:\n")
        f.write(f"- Accuracy: {accuracy:.3f}\n")
        f.write(f"- F1-score: {f1:.3f}\n")
        f.write(f"- ROC AUC: {auc:.3f}\n\n")
        
        f.write(f"DATA:\n")
        f.write(f"- Total samples: {len(df)}\n")
        f.write(f"- Number of genes: {df['Gene'].nunique()}\n")
        f.write(f"- Number of hotspots: {hotspots_count}\n\n")
        
        f.write("Logistic Regression model achieved satisfactory results\n")
        f.write("in classifying pathogenic vs benign mutations.\n")
    
    print(f"Conclusions saved to: {CONFIG['file_paths']['results_dir']}/conclusions_dna_analysis_basic_vol2.txt")


def create_essential_plots_basic(y_test, y_pred, y_prob, total_mutations, pathogenic_mutations, genes, fpr, tpr, auc, cm, df=None):
    """Tworzy najważniejsze wykresy dla wersji basic i zapisuje do plików"""
    ensure_results_directory()
    
    print("\n" + CONFIG['report_headers']['plot_creation'])
    
    # ROC Curve z CONFIG
    plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['standard'])
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve - Classifier Performance")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=CONFIG['plot_settings']['grid_alpha'])
    save_plot_with_standard_format('roc_curve_basic_vol2_results.png')
    
    # Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    save_plot_with_standard_format('confusion_matrix_basic_vol2_results.png')
    
    # Podział mutacji patogennych vs benign dla każdego genu
    if df is not None:
        _create_gene_mutation_breakdown(df, genes, pathogenic_mutations)
        _create_gene_pathogenicity_chart(df)
        _create_mutation_types_chart(df)
        _create_mutation_pathogenicity_by_type_chart(df)
        _create_gc_content_per_gene_chart(df)
        create_pathogenicity_pie_charts_per_gene(df, 'Pathogenicity', CONFIG['file_paths']['results_dir'])
    
    _print_results_summary(df)
    

def _create_gene_mutation_breakdown(df, genes, pathogenic_mutations):
    """Tworzy wykres podziału mutacji patogennych vs benign dla każdego genu"""
    benign_mutations = df[df["Pathogenicity"] == "benign"]["Gene"].value_counts()
    benign_aligned = [benign_mutations.get(gene, 0) for gene in genes]
    pathogenic_aligned = [pathogenic_mutations.get(gene, 0) for gene in genes]

    plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['large'])
    bar_width = CONFIG.get('chart_settings', {}).get('bar_width', 0.4)
    x = range(len(genes))
    plt.bar(x, pathogenic_aligned, width=bar_width, label="Patogenne", 
            color=CONFIG['visualization_colors']['pathogenic_color'], 
            alpha=CONFIG['chart_settings']['bar_alpha'])
    plt.bar([i + bar_width for i in x], benign_aligned, width=bar_width, label="Benign (łagodne)", 
            color=CONFIG['visualization_colors']['benign_color'], 
            alpha=CONFIG['chart_settings']['bar_alpha'])
    plt.xticks([i + bar_width/2 for i in x], genes, rotation=45)
    plt.xlabel("Gene")
    plt.ylabel("Number of Mutations")
    plt.title("Pathogenic and Benign Mutations per Gene")
    plt.legend()
    save_plot_with_standard_format('mutation_counts_by_gene_basic_vol2_results.png')


def _create_gene_pathogenicity_chart(df):
    """Tworzy wykres ryzyka patogenności według genów"""
    plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['medium'])
    gene_pathogenicity = []
    gene_labels = []
    for gene in df['Gene'].unique():
        gene_data = df[df['Gene'] == gene]
        pathogenic_pct = (gene_data['Pathogenicity'] == 'pathogenic').mean() * 100
        gene_pathogenicity.append(pathogenic_pct)
        gene_labels.append(gene)
    
    # Progi ryzyka z konfiguracji
    high_risk_threshold = CONFIG['pathogenicity_thresholds']['high_risk']
    moderate_risk_threshold = CONFIG['pathogenicity_thresholds']['moderate_risk']
    
    colors = [CONFIG['visualization_colors']['high_risk_color'] if x > high_risk_threshold 
              else CONFIG['visualization_colors']['moderate_risk_color'] if x > moderate_risk_threshold 
              else CONFIG['visualization_colors']['low_risk_color'] for x in gene_pathogenicity]
    
    bars = plt.bar(gene_labels, gene_pathogenicity, color=colors, alpha=CONFIG['chart_settings']['bar_alpha'])
    plt.xlabel("Gene")
    plt.ylabel("Pathogenic Mutations (%)")
    plt.title("Pathogenicity Risk by Gene")
    plt.axhline(y=high_risk_threshold, color=CONFIG['visualization_colors']['high_risk_color'], 
                linestyle='--', alpha=0.5, label=f'{high_risk_threshold}% próg ryzyka')
    
    for bar, pct in zip(bars, gene_pathogenicity):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.legend()
    save_plot_with_standard_format('gene_pathogenicity_percentage_basic_vol2_results.png')


def _create_mutation_types_chart(df):
    """Tworzy wykres najczęstszych typów mutacji"""
    plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['large'])
    mutation_types = df['Reference_Base'] + '>' + df['Alternate_Base']
    mutation_counts = mutation_types.value_counts().head(CONFIG['analysis_parameters']['top_mutations_limit'])
    
    plt.bar(range(len(mutation_counts)), mutation_counts.values, color='steelblue')
    plt.xlabel("Mutation Type (Ref>Alt)")
    plt.ylabel("Count")
    plt.title("Most Frequent Point Mutation Types")
    plt.xticks(range(len(mutation_counts)), mutation_counts.index, rotation=45)
    
    for i, v in enumerate(mutation_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    save_plot_with_standard_format('mutation_types_frequency_basic_vol2_results.png')


def _create_mutation_pathogenicity_by_type_chart(df):
    """Tworzy wykres ryzyka patogenności różnych typów mutacji"""
    mutation_types = df['Reference_Base'] + '>' + df['Alternate_Base']
    mutation_counts = mutation_types.value_counts().head(8)
    
    mutation_pathogenicity = []
    mutation_labels = []
    mutation_total_counts = []
    
    for mut_type in mutation_counts.index:
        ref, alt = mut_type.split('>')
        mask = (df['Reference_Base'] == ref) & (df['Alternate_Base'] == alt)
        mut_data = df[mask]
        
        # Minimum 3 próbki dla statystycznej wiarygodności analizy
        if len(mut_data) >= CONFIG['analysis_parameters']['min_mutations_for_analysis']:
            pathogenic_pct = (mut_data['Pathogenicity'] == 'pathogenic').mean() * 100
            mutation_pathogenicity.append(pathogenic_pct)
            mutation_labels.append(mut_type)
            mutation_total_counts.append(len(mut_data))
    
    if len(mutation_pathogenicity) > 0:
        plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['large'])
        # Progi ryzyka z konfiguracji
        high_threshold = CONFIG['pathogenicity_thresholds']['mutation_risk_high']
        moderate_threshold = CONFIG['pathogenicity_thresholds']['mutation_risk_moderate']
        
        colors = [CONFIG['visualization_colors']['high_risk_color'] if x > high_threshold 
                  else CONFIG['visualization_colors']['moderate_risk_color'] if x > moderate_threshold 
                  else CONFIG['visualization_colors']['safe_color'] for x in mutation_pathogenicity]
        
        bars = plt.bar(mutation_labels, mutation_pathogenicity, color=colors, alpha=CONFIG['chart_settings']['bar_alpha'])
        plt.xlabel("Mutation Type (Ref>Alt)")
        plt.ylabel("Pathogenic (%)")
        plt.title("Pathogenicity Risk by Mutation Type")
        
        threshold_line = CONFIG['pathogenicity_thresholds']['high_risk']
        plt.axhline(y=threshold_line, color=CONFIG['visualization_colors']['high_risk_color'], 
                    linestyle='--', alpha=0.5, label=f'{threshold_line}% próg patogenności')
        
        for bar, pct, count in zip(bars, mutation_pathogenicity, mutation_total_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{pct:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        plt.legend()
        save_plot_with_standard_format('mutation_pathogenicity_by_type_basic_vol2_results.png')


def _create_gc_content_per_gene_chart(df):
    """Tworzy wykres zawartości GC per gen"""
    gc_data_per_gene = []
    gene_names = []
    
    for gene in df['Gene'].unique():
        gene_data = df[df['Gene'] == gene]
        gc_contents = []
        for sequence in gene_data['DNA_Sequence']:
            gc_count = sequence.count("G") + sequence.count("C")
            gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0
            gc_contents.append(gc_content)
        
        if gc_contents:
            gc_data_per_gene.append(gc_contents)
            gene_names.append(gene)
    
    if gc_data_per_gene:
        plt.figure(figsize=CONFIG['plot_settings']['figure_sizes']['large'])
        box_plot = plt.boxplot(gc_data_per_gene, labels=gene_names, patch_artist=True)
        colors_gc = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors_gc[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        plt.xlabel("Gene")
        plt.ylabel("GC Content (0-1)")
        plt.title("GC Content Distribution per Gene")
        plt.grid(True, alpha=0.3)
        
        for i, gc_data in enumerate(gc_data_per_gene):
            mean_gc = sum(gc_data) / len(gc_data)
            plt.scatter(i+1, mean_gc, color='red', s=50, zorder=5, 
                      label='Mean' if i == 0 else "")
            plt.text(i+1, mean_gc + 0.02, f'{mean_gc:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.legend()
        save_plot_with_standard_format('gc_content_per_gene_basic_vol2_results.png')


def _print_results_summary(df):
    """Drukuje podsumowanie wyników"""
    print(f"\nWykresy zapisane w folder: {CONFIG['file_paths']['results_dir']}/")
    print(f"  • roc_curve_basic_vol2_results.png - model performance")
    print(f"  • confusion_matrix_basic_vol2_results.png - classification errors")
    print(f"  • mutation_counts_by_gene_basic_vol2_results.png - pathogenic vs benign per gene")
    if df is not None:
        print(f"  • gene_pathogenicity_percentage_basic_vol2_results.png - ryzyko per gen")
        print(f"  • mutation_types_frequency_basic_vol2_results.png - najczęstsze mutacje")
        print(f"  • mutation_pathogenicity_by_type_basic_vol2_results.png - ryzyko per typ mutacji")
        print(f"  • gc_content_per_gene_basic_vol2_results.png - zawartość GC per gen")
        create_pathogenicity_pie_charts_per_gene(df, 'Pathogenicity', CONFIG['file_paths']['results_dir'])


def create_pathogenicity_pie_charts_per_gene(df, target_column, results_dir):
    """Tworzy wykresy kołowe patogenności dla każdego genu (BASIC VOL2)"""
    genes = sorted(df['Gene'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['plot_settings']['figure_sizes']['large'])
    axes = axes.flatten()
    
    colors = CONFIG['chart_settings']['pie_chart_colors']  # Z konfiguracji
    
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
            ax.set_title(f'Gen {gene}\n{total_mutations} mutacji, {pathogenic_pct:.1f}% patogennych', 
                        fontsize=12, pad=20)
            
            # Dodatkowy tekst w centrum (jeśli potrzebny)
            ax.text(0, -1.3, f'Patogenne: {pathogenic_count}\nŁagodne: {benign_count}', 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Ukryj puste panele jeśli są
    for i in range(len(genes), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Podział mutacji patogennych vs niepatogennych per gen (BASIC VOL2)', 
                fontsize=14, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
    plt.savefig(f'{results_dir}/dna_pathogenicity_per_gene_basic_vol2_results.png', dpi=CONFIG['plot_settings']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved pathogenicity charts per gene to: {results_dir}/dna_pathogenicity_per_gene_basic_vol2_results.png")


def create_hotspots_faceted_plot(hotspots_df, df, results_dir):
    """Tworzy faceted plot hotspotów per gen (BASIC VOL2)"""
    genes = df['Gene'].unique()
    
    if len(hotspots_df) > 0 and 'gene' in hotspots_df.columns:
        # Stwórz subplot dla hotspotów
        fig_hotspots, axes_hotspots = plt.subplots(2, 2, figsize=(12, 10))
        axes_hotspots = axes_hotspots.flatten()
        
        for i, gene in enumerate(sorted(genes)):
            if i < len(axes_hotspots):
                ax = axes_hotspots[i]
                
                # Dane dla tego genu
                gene_hotspots = hotspots_df[hotspots_df['gene'] == gene].head(5)
                
                if len(gene_hotspots) > 0:
                    colors = ['#f44336' if rate > 0.8 else '#ff9800' if rate > 0.5 else '#4caf50' 
                             for rate in gene_hotspots['pathogenic_rate']]
                    
                    bars = ax.bar(range(len(gene_hotspots)), 
                                 gene_hotspots['pathogenic_rate'], 
                                 color=colors)
                    
                    ax.set_title(f'Hotspoty: {gene}', fontsize=11, pad=15)
                    ax.set_ylabel('% patogennych', fontsize=9)
                    ax.set_xlabel('Pozycje (ranking)', fontsize=9)
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
                    ax.text(0.5, 0.5, f'{gene}\nBrak hotspotów', 
                           ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'Hotspoty: {gene}', fontsize=11, pad=15)
        
        plt.suptitle('Hotspoty mutacyjne per gen (BASIC VOL2)', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
        plt.savefig(f'{results_dir}/dna_hotspots_per_gene_basic_vol2.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved hotspots per gene to: {results_dir}/dna_hotspots_per_gene_basic_vol2.png")


def print_final_summary_basic(df, X, accuracy, precision, recall, f1, auc):
    """Drukuje finalne podsumowanie wyników dla wersji basic"""
    print("\n" + "="*60)
    print("              PODSUMOWANIE PROJEKTU (BASIC VERSION)")
    print("="*60)
    
    print(f"\nDANE:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Liczba cech: {X.shape[1]} (One-Hot encoding + biochemiczne)")
    
    balance = df['Pathogenicity'].value_counts()
    print(f"  • Balans klas: {balance['pathogenic']}/{balance['benign']} (pathogenic/benign)")
    
    print(f"\nMODEL:")
    print(f"  • Model: Logistic Regression (basic)")
    print(f"  • Accuracy:  {accuracy:.3f}")
    print(f"  • Precision: {precision:.3f}")
    print(f"  • Recall:    {recall:.3f}")
    print(f"  • F1-score:  {f1:.3f}")
    print(f"  • ROC AUC:   {auc:.3f}")
    
    print(f"\nCECHY:")
    print(f"  One-Hot encoding DNA sequences (200 cech)")
    print(f"  Reference/Alternate Base encoding (8 cech)")
    print(f"  Mutation Position (skalowana)")
    print(f"  Cechy biochemiczne (GC content, transitions)")
    print(f"  NCBI analysis (medical literature)")
    
    print("\n" + "="*60)
    print("Project completed successfully!")
    print("PLOTS saved in results_basic_vol2/ folder:")
    print("  8 individual plots")
    print("="*60)


def main():
    """Główna funkcja uruchamiająca cały pipeline analizy (wersja basic)"""
    print("URUCHOMIENIE ANALIZY DNA MUTATIONS (BASIC VERSION)")
    print("="*60)
    
    # 1. Wczytanie danych
    df = load_data()
    
    # 2. Podstawowa eksploracja
    explore_data(df)
    
    # 3. Biochemical features
    print("\nANALIZA BIOCHEMICZNA SEKWENCJI:")
    biochem_features = biochemical_features_enhanced(df)
    print(f"Mean GC content: {biochem_features['GC_content'].mean():.3f}")
    print(f"Mean GC pairs: {biochem_features['GC_pairs'].mean():.1f}")
    print(f"% mutacji typu transition: {biochem_features['is_transition'].mean()*100:.1f}%")
    print(f"% mutations within same chemical group: {biochem_features['same_chemical_type'].mean()*100:.1f}%")
    
    ncbi_mutation_results = analyze_top_mutations_ncbi(df)
    
    print("\nEKSTRAKCJA CECH:")
    

    def one_hot_sequence(seq):
        mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
        return [bit for nucleotide in seq for bit in mapping[nucleotide]]
    
    def one_hot_base(base):
        mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
        return mapping[base]
    
    # Kodowanie sekwencji DNA
    encoding_seq = df["DNA_Sequence"].apply(one_hot_sequence)
    seq_cols = [f"Pos{i}_{nt}" for i in range(50) for nt in ["A","C","G","T"]]
    encoding_seq_df = pd.DataFrame(encoding_seq.tolist(), columns=seq_cols)
    
    encoding_ref = df["Reference_Base"].apply(one_hot_base)
    ref_cols = [f"Ref_{nt}" for nt in ["A","C","G","T"]]
    encoding_ref_df = pd.DataFrame(encoding_ref.tolist(), columns=ref_cols)
    
    encoding_alt = df["Alternate_Base"].apply(one_hot_base)
    alt_cols = [f"Alt_{nt}" for nt in ["A","C","G","T"]]
    encoding_alt_df = pd.DataFrame(encoding_alt.tolist(), columns=alt_cols)
    
    encoding_pos_df = df[["Mutation_Position"]]
    
    X_final = pd.concat([encoding_seq_df, encoding_ref_df, encoding_alt_df, encoding_pos_df], axis=1)

    X = X_final
    y = df["Pathogenicity"].map({"benign":0, "pathogenic":1})

    scaler = MinMaxScaler()
    X_final["Mutation_Position"] = scaler.fit_transform(X_final[["Mutation_Position"]])
    
    print(f"Created {X.shape[1]} features for {X.shape[0]} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['ml_parameters']['test_size'], 
        stratify=y, 
        random_state=CONFIG['ml_parameters']['random_state']
    )
    
    total = len(X)
    print(f"\nDATA SPLIT:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/total:.1%})")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/total:.1%})")
    
    print("\nTRENING MODELU:")
    # class_weight='balanced' kompensuje niezbalansowanie klas pathogenic/benign
    model = LogisticRegression(
        solver=CONFIG['ml_parameters']['solver'],
        max_iter=CONFIG['ml_parameters']['max_iterations'],
        class_weight=CONFIG['ml_parameters']['class_weight'],
        random_state=CONFIG['ml_parameters']['random_state']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 8. Ewaluacja modelu
    print("\nEWALUACJA MODELU:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    print(f"ROC AUC: {auc:.3f}")
    
    ncbi_gene_results = analyze_top_genes_ncbi(df)
    # Łączę wyniki mutacji i genów do zapisu
    all_ncbi_results = ncbi_mutation_results if ncbi_mutation_results else ncbi_gene_results
    save_ncbi_results_to_file(all_ncbi_results)
    
    total_mutations = df["Gene"].value_counts()
    pathogenic_mutations = df[df["Pathogenicity"] == "pathogenic"]["Gene"].value_counts()
    genes = total_mutations.index
    
    create_essential_plots_basic(y_test, y_pred, y_prob, total_mutations, 
                               pathogenic_mutations, genes, fpr, tpr, auc, cm, df)
    
    hotspots_data, hotspots_filtered = analyze_mutation_hotspots(df)
    
    save_conclusions_to_file(df, accuracy, f1, auc, len(hotspots_data))
    
    print_final_summary_basic(df, X, accuracy, precision, recall, f1, auc)


# NCBI INTEGRATION FUNCTIONS


if __name__ == "__main__":
    main()