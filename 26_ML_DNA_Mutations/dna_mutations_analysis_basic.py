#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 21:16:44 2025

@author: mago
"""

"""
Projekt 2: Przewidywanie mutacji w sekwencjach DNA związanych z chorobami genetycznymi

Cel projektu: Celem jest stworzenie modelu klasyfikacyjnego, który określi, czy dana mutacja w sekwencji DNA jest patogenna.\
    
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Import dla NCBI API
try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Biopython nie jest zainstalowany. Funkcje NCBI będą niedostępne.")
    print("Zainstaluj: pip install biopython")
    BIOPYTHON_AVAILABLE = False


# Pobranie danych (wczytanie pliku z danymi)
df = pd.read_csv("/Users/mago/Desktop/Projekty VS/26_ML_DNA_Mutations/data/data_rare/data_DNA_mutations.csv")


# Eksploracja danych
# Wyświetlenie podstawowych statystyk
print(df.describe())
print(df.info())

# Ile braków w każdej kolumnie
print(df.isna().sum())

# Długości sekwencji DNA
lengths = df["DNA_Sequence"].str.len()
print(lengths.head())       # Podejrzenie kilku pierwszych
print(lengths.describe())       # Statystyki

# Ile braków w każdej kolumnie
print(df.isna().sum())

# Najczęstsze pozycje mutacji
position_counts = df["Mutation_Position"].value_counts().sort_values(ascending=False)
print("\nNajczęstsze pozycje mutacji:\n", position_counts.head(15))

# Najrzadsze pozycje mutacji
rare_positions = df["Mutation_Position"].value_counts()
print("\nNajrzadsze pozycje mutacji:\n", rare_positions.tail(15))

# Całkowita liczba mutacji w każdym genie
gene_counts = df["Gene"].value_counts()
print("\nLiczba mutacji w każdym genie:\n", gene_counts.head(10))

# Liczba mutacji patogennych w każdym genie
pathogenic_counts = df[df["Pathogenicity"] == "pathogenic"]["Gene"].value_counts()
print("\nLiczba mutacji patogennych w każdym genie: \n", pathogenic_counts.head(10))

# Liczba mutacji patogennych w każdym genie
pathogenic = df[df["Pathogenicity"] == "pathogenic"]        # Filtrujemy mutacje patogenne

pathogenic_changes = pathogenic.groupby(["Gene", "Reference_Base", "Alternate_Base"]).size().sort_values(ascending=False)       # Grupujemy po genie i zmianie nukleotydu, liczymy wystąpienia

# Wyświetlamy top 10
print("\nNajczęstsze mutacje patogenne (Gen + zmiana nukleotydu):\n", pathogenic_changes.head(10))

# Procent mutacji patogennych w każdym genie
percent_pathogenic = (pathogenic_counts / gene_counts * 100).round(2)
print("\nProcent mutacji patogennych w każdym genie:\n", percent_pathogenic.head(10))

# Liczba wystąpień każdej mutacji (Ref>Alt) oraz patogenność
mutation_counts = df.groupby(["Reference_Base", "Alternate_Base", "Pathogenicity"]).size().sort_values(ascending=False)
print("\nNajczęstsze mutacje (Ref>Alt : Pathogenicity):\n", mutation_counts.head(20))


# ===== FUNKCJE BIOCHEMICZNE (GC CONTENT ANALYSIS) =====

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
    """Tworzy dodatkowe cechy biochemiczne"""
    features = {}
    
    # GC content dla całej sekwencji
    features['GC_content'] = df['DNA_Sequence'].apply(gc_content_analysis)
    
    # Pary GC w sekwencji  
    features['GC_pairs'] = df['DNA_Sequence'].apply(liczenie_par_gc)
    
    # Analiza kontekstu mutacji (3 nukleotydy przed i po)
    def context_analysis(row):
        seq = row['DNA_Sequence']
        pos = row['Mutation_Position'] - 1  # 0-indexed
        
        # 3 nukleotydy przed i po mutacji
        start = max(0, pos-3)
        end = min(len(seq), pos+4)
        context = seq[start:end] if pos < len(seq) else ""
        
        gc_in_context = gc_content_analysis(context) if context else 0
        return gc_in_context
    
    features['context_GC'] = df.apply(context_analysis, axis=1)
    
    # Typ mutacji (transition vs transversion)
    def mutation_type(ref, alt):
        transitions = [('A','G'), ('G','A'), ('C','T'), ('T','C')]
        return 1 if (ref, alt) in transitions else 0  # 1=transition, 0=transversion
    
    features['is_transition'] = df.apply(lambda x: mutation_type(x['Reference_Base'], x['Alternate_Base']), axis=1)
    
    # Zmiana chemiczna (purine <-> pyrimidine)
    def chemical_change(ref, alt):
        purines = ['A', 'G']
        pyrimidines = ['C', 'T']
        ref_type = 'purine' if ref in purines else 'pyrimidine'
        alt_type = 'purine' if alt in purines else 'pyrimidine'
        return 1 if ref_type == alt_type else 0  # 1=same type, 0=different type
    
    features['same_chemical_type'] = df.apply(lambda x: chemical_change(x['Reference_Base'], x['Alternate_Base']), axis=1)
    
    return pd.DataFrame(features)

# Zastosowanie funkcji biochemicznych
print("\nANALIZA BIOCHEMICZNA SEKWENCJI:")
biochem_features = biochemical_features_enhanced(df)
print(f"Średnia zawartość GC: {biochem_features['GC_content'].mean():.3f}")
print(f"Średnia liczba par GC: {biochem_features['GC_pairs'].mean():.1f}")
print(f"% mutacji typu transition: {biochem_features['is_transition'].mean()*100:.1f}%")
print(f"% mutacji w obrębie tej samej grupy chemicznej: {biochem_features['same_chemical_type'].mean()*100:.1f}%")


# ===== FUNKCJE NCBI I ERROR HANDLING =====

def safe_ncbi_search(gene, ref_base, alt_base, email="analysis@example.com"):
    """Bezpieczne wyszukiwanie w NCBI z obsługą błędów"""
    if not BIOPYTHON_AVAILABLE:
        return 0
        
    try:
        Entrez.email = email
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
        print(f"Błąd NCBI dla {gene} {ref_base}>{alt_base}: {error}")
        return 0

def search_gene_diseases(gene, email="analysis@example.com"):
    """Wyszukuje choroby związane z genem"""
    if not BIOPYTHON_AVAILABLE:
        return "Biopython niedostępny"
        
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
        return f"Znaleziono {count} publikacji o chorobach"
        
    except Exception as error:
        return f"Błąd: {error}"

def analyze_top_mutations_ncbi(df):
    """Analizuje najczęstsze mutacje w bazach NCBI"""
    if not BIOPYTHON_AVAILABLE:
        print("Biopython niedostępny - analiza NCBI pominięta")
        return
        
    print("\nANALIZA TOP 3 MUTACJI W NCBI:")
    
    # Konfiguracja email dla NCBI
    try:
        email = input("Podaj email dla NCBI (lub Enter dla domyślnego): ").strip()
        if not email:
            email = "analysis@example.com"
            print(f"Użyto domyślnego emaila: {email}")
    except:
        email = "analysis@example.com"
        print(f"Użyto domyślnego emaila: {email}")
    
    # Top 3 mutacje patogenne
    pathogenic = df[df["Pathogenicity"] == "pathogenic"]
    top3 = pathogenic.groupby(["Gene", "Reference_Base", "Alternate_Base"]).size().head(3)
    
    for (gene, ref, alt), count in top3.items():
        print(f"\n{gene} {ref}>{alt} (występuje {count}x w naszych danych):")
        
        # Wyszukiwanie w PubMed
        pubmed_count = safe_ncbi_search(gene, ref, alt, email)
        print(f"  PubMed: {pubmed_count} publikacji")
        
        # Wyszukiwanie chorób związanych z genem
        disease_info = search_gene_diseases(gene, email)
        print(f"  Choroby: {disease_info}")
    
    # Znane mutacje kliniczne (literatura)
    clinical_mutations = {
        'BRCA1': {
            'c.68_69delAG': 'Breast/Ovarian cancer - founder mutation',
            'c.185delAG': 'Breast cancer - frameshift', 
            'c.5266dupC': 'Hereditary breast-ovarian cancer'
        },
        'TP53': {
            'R175H': 'Li-Fraumeni syndrome - hotspot',
            'R248W': 'Cancer predisposition',
            'R273H': 'Multiple cancer types'
        },
        'CFTR': {
            'F508del': 'Cystic Fibrosis - najczęstsza mutacja',
            'G542X': 'Cystic Fibrosis - nonsense',
            'N1303K': 'Cystic Fibrosis - missense'
        },
        'MYH7': {
            'R403Q': 'Hypertrophic cardiomyopathy - first described',
            'R719W': 'Dilated cardiomyopathy',
            'E927K': 'Left ventricular noncompaction'
        }
    }
    
    print("\nZNANE MUTACJE KLINICZNE Z LITERATURY:")
    genes_in_data = df['Gene'].unique()
    for gene in genes_in_data:
        if gene in clinical_mutations:
            print(f"\n{gene}:")
            for mutation, disease in clinical_mutations[gene].items():
                print(f"   • {mutation}: {disease}")

# Uruchomienie analizy NCBI
analyze_top_mutations_ncbi(df)


# Ekstrakcja cech - przetwarzanie danych

# Funkcja do one-hot encoding dla DNA_Sequence (binarne kodowanie nukleotydów)
def one_hot_sequence(seq):
    mapping = {"A": [1,0,0,0],
               "C": [0,1,0,0],
               "G": [0,0,1,0],
               "T": [0,0,0,1]}
    return [bit for nucleotide in seq for bit in mapping[nucleotide]]

# One-hot encoding dla Reference_Base i Alternate_Base
def one_hot_base(base):
    mapping = {"A": [1,0,0,0],
               "C": [0,1,0,0],
               "G": [0,0,1,0],
               "T": [0,0,0,1]}
    return mapping[base]

# Kodowanie sekwencji DNA
encoding_seq = df["DNA_Sequence"].apply(one_hot_sequence)
seq_cols = [f"Pos{i}_{nt}" for i in range(50) for nt in ["A","C","G","T"]]
encoding_seq_df = pd.DataFrame(encoding_seq.tolist(), columns=seq_cols)

# Kodowanie Reference_Base
encoding_ref = df["Reference_Base"].apply(one_hot_base)
ref_cols = [f"Ref_{nt}" for nt in ["A","C","G","T"]]
encoding_ref_df = pd.DataFrame(encoding_ref.tolist(), columns=ref_cols)

# Kodowanie Alternate_Base
encoding_alt = df["Alternate_Base"].apply(one_hot_base)
alt_cols = [f"Alt_{nt}" for nt in ["A","C","G","T"]]
encoding_alt_df = pd.DataFrame(encoding_alt.tolist(), columns=alt_cols)

# Kodowanie Mutation_Position - zostawiamy jako liczbę
encoding_pos_df = df[["Mutation_Position"]]

# Łączymy wszystkie cechy 
X_final = pd.concat([encoding_seq_df, encoding_ref_df, encoding_alt_df, encoding_pos_df], axis=1)
print(X_final.head())

# Kodowanie Pathogenicity
encoding_pathogenicity = pd.get_dummies(df["Pathogenicity"], prefix="Patho")

# Przypisujemy cechy X i etykiety y
X = X_final
y = df["Pathogenicity"].map({"benign":0, "pathogenic":1})

# Skalowanie Mutation_Position
scaler = MinMaxScaler()
X_final["Mutation_Position"] = scaler.fit_transform(X_final[["Mutation_Position"]])


# Podział na zbiory treningowy i testowy (70% treningowe + walidacyjne, 30% testowe)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=62)

# Wypisanie informacji
total = len(X)
print(f"\nTrain: {len(X_train)} próbek ({len(X_train)/total:.1%})")
print(f"\nTest:  {len(X_test)} próbek ({len(X_test)/total:.1%})")


# Budowa modelu ML
model = LogisticRegression(solver='liblinear', max_iter=800)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Ewaluacja modelu
print("\nAccuracy:", accuracy_score(y_test, y_pred))      # Dokładność
print("\nPrecision:", precision_score(y_test, y_pred))        # Precyzja
print("\nRecall:", recall_score(y_test, y_pred))      # Czułość
print("\nF1-score:", f1_score(y_test, y_pred))        # Średnia harmoniczna
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))      # Macierz pomyłek
print("\nClassification Report:\n", classification_report(y_test, y_pred))        # Raport klasyfikacji


# Miara zdolności modelu do rozróżniania klas
auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("ROC AUC:", auc)

# Predykcje prawdopodobieństwa dla klasy 1 (pathogenic)
y_prob = model.predict_proba(X_test)[:,1]

# AUC
auc = roc_auc_score(y_test, y_prob)
print("\nAUC:", auc)

# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)


# Interpretacja wyników
# Rysowanie wykresu "Ilość mutacji z podziałem patogenne i nie"
# Dane do wykresu
total_mutations = df["Gene"].value_counts()                  # wszystkie mutacje
pathogenic_mutations = df[df["Pathogenicity"] == "pathogenic"]["Gene"].value_counts()  # mutacje patogenne
genes = total_mutations.index 

# Tworzenie wykresu słupkowego
plt.figure(figsize=(12,6))
bar_width = 0.4
x = range(len(genes))
plt.bar(x, total_mutations, width=bar_width, label="Całkowite mutacje", color="purple")
plt.bar([i + bar_width for i in x], pathogenic_mutations, width=bar_width, label="Mutacje patogenne", color="green")
plt.xticks([i + bar_width/2 for i in x], genes, rotation=45)
plt.xlabel("Gen")
plt.ylabel("Liczba mutacji")
plt.title("Ilość mutacji z podziałem patogenne i nie")
plt.legend()
plt.tight_layout()
plt.show()


# Wizualizacja macierzy pomyłek
# Dane do wykresu
cm = confusion_matrix(y_test, y_pred)

# Tworzenie wykresu - heatmap'y
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predykcja')
plt.ylabel('Prawdziwa klasa')
plt.title('Macierz pomyłek')
plt.show()


# Tworzenie wykresu "ROC Curve - skuteczność klasyfikatora"
# Dane do wykresu
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Twoerzenie wykresu
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], 'k--')  # losowa klasyfikacja
plt.xlabel("False Positive Rate (1 - specyficzność)")
plt.ylabel("True Positive Rate (czułość)")
plt.title("ROC Curve - skuteczność klasyfikatora")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()