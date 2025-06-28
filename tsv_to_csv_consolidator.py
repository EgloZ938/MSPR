import pandas as pd
import os
import glob
import re
from pathlib import Path

def clean_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convertir en string si ce n'est pas déjà le cas
    text = str(text)
    
    # Supprimer les caractères spéciaux
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Nettoyer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_year_from_filename(filename):
    """Extrait l'année du nom de fichier - logique adaptée aux données démographiques"""
    # Cherche un pattern comme (1), (2), etc.
    match = re.search(r'\((\d+)\)', filename)
    if match:
        file_number = int(match.group(1))
        # Les données semblent être des estimations par région/pays
        # On va créer des années fictives pour différencier les datasets
        return 2024 + file_number  # Commence à 2025, 2026, etc.
    else:
        # Fichier de base sans numéro
        return 2024

def identify_dataset_type(content_lines):
    """Identifie le type de dataset basé sur le contenu"""
    # Regarder les premières lignes de données pour identifier le type
    for line in content_lines[3:8]:  # Lignes de données après le header
        if 'Eastern Africa' in line or 'Middle Africa' in line:
            return "Regional"
        elif 'Brunei' in line or 'Cambodia' in line:
            return "Southeast_Asia"
        elif 'Armenia' in line or 'Azerbaijan' in line:
            return "Western_Asia"
        elif 'Argentina' in line or 'Bolivia' in line:
            return "Latin_America"
        elif 'Austria' in line or 'Belgium' in line:
            return "Europe"
        elif 'Algeria' in line or 'Angola' in line:
            return "Africa_Countries"
    
    return "Unknown"

def process_tsv_file(file_path):
    """Traite un fichier TSV et retourne un DataFrame nettoyé"""
    try:
        print(f"Traitement du fichier: {os.path.basename(file_path)}")
        
        # Lire le fichier avec différents encodages
        content = None
        encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except:
                continue
        
        if content is None:
            print(f"  ❌ Impossible de lire le fichier")
            return pd.DataFrame()
        
        # Séparer en lignes
        lines = content.strip().split('\n')
        
        # Extraire l'année du nom de fichier
        year = extract_year_from_filename(os.path.basename(file_path))
        
        # Identifier le type de dataset
        dataset_type = identify_dataset_type(lines)
        
        # Trouver la ligne de header (celle qui contient "Countries")
        header_line_idx = None
        for i, line in enumerate(lines):
            if 'Countries' in line and 'Total population' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            print(f"  ❌ Pas de header trouvé")
            return pd.DataFrame()
        
        # Extraire le header et nettoyer
        header_line = lines[header_line_idx]
        headers = [h.strip() for h in header_line.split('\t')]
        
        # Traiter les lignes de données
        data_rows = []
        for line in lines[header_line_idx + 1:]:
            line = line.strip()
            if not line:  # Ignorer les lignes vides
                continue
            
            # Ignorer les lignes de source/note (elles contiennent souvent "Institut", "Source", etc.)
            if any(keyword in line.upper() for keyword in ['INSTITUT NATIONAL', 'SOURCE', 'NOTE FOR THE READER', 'WORLD POPULATION', 'THE PUBLICATION']):
                continue
            
            # Si la ligne contient des tabs et semble être des données
            if '\t' in line and len(line.split('\t')) >= 3:
                # Séparer par tabs
                values = line.split('\t')
                
                # Vérifier que la première colonne n'est pas vide (nom du pays/région)
                if values[0].strip():
                    # Nettoyer les valeurs
                    cleaned_values = []
                    for val in values:
                        val = clean_text(val)
                        # Essayer de convertir en nombre si possible
                        try:
                            # Si ça contient un point et des chiffres, essayer float
                            if '.' in val and any(c.isdigit() for c in val):
                                val = float(val)
                            # Si c'est que des chiffres, essayer int
                            elif val.replace(',', '').replace('.', '').replace('-', '').isdigit():
                                val = val.replace(',', '')  # Supprimer les virgules des milliers
                                if '.' not in str(val):
                                    if val.startswith('-'):
                                        val = -int(val[1:])
                                    else:
                                        val = int(val)
                                else:
                                    val = float(val)
                        except:
                            pass  # Garder comme string si conversion impossible
                        
                        cleaned_values.append(val)
                    
                    # S'assurer qu'on a le bon nombre de colonnes
                    while len(cleaned_values) < len(headers):
                        cleaned_values.append('')
                    
                    data_rows.append(cleaned_values[:len(headers)])
        
        if not data_rows:
            print(f"  ❌ Aucune donnée valide trouvée")
            return pd.DataFrame()
        
        # Créer le DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Ajouter les métadonnées
        df['Year'] = year
        df['Dataset_Type'] = dataset_type
        df['Source_File'] = os.path.basename(file_path)
        
        # Réorganiser les colonnes
        meta_cols = ['Year', 'Dataset_Type', 'Source_File']
        data_cols = [col for col in df.columns if col not in meta_cols]
        df = df[meta_cols + data_cols]
        
        print(f"  ✅ {len(df)} lignes extraites (Type: {dataset_type})")
        return df
        
    except Exception as e:
        print(f"  ❌ Erreur: {str(e)}")
        return pd.DataFrame()

def main():
    """Fonction principale"""
    # Définir les chemins
    excel_folder = "covid-dashboard-vue/data/excel"
    output_folder = "covid-dashboard-vue/data/dataset"
    output_file = "consolidated_demographics_data.csv"
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Trouver tous les fichiers
    tsv_files = glob.glob(os.path.join(excel_folder, "*.xls*"))
    
    if not tsv_files:
        print(f"Aucun fichier trouvé dans {excel_folder}")
        return
    
    print(f"🔄 Traitement de {len(tsv_files)} fichiers TSV...")
    print("=" * 60)
    
    # Liste pour stocker tous les DataFrames
    all_dataframes = []
    
    # Traiter chaque fichier
    for file_path in sorted(tsv_files):
        df = process_tsv_file(file_path)
        if not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("\n❌ Aucune donnée extraite des fichiers")
        return
    
    # Consolider tous les DataFrames
    print(f"\n🔗 Consolidation de {len(all_dataframes)} datasets...")
    consolidated_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Nettoyer le DataFrame final
    consolidated_df = consolidated_df.dropna(how='all')
    
    # Sauvegarder en CSV
    output_path = os.path.join(output_folder, output_file)
    consolidated_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n🎉 CONSOLIDATION TERMINÉE !")
    print("=" * 60)
    print(f"📁 Fichier de sortie: {output_path}")
    print(f"📊 Total des lignes: {len(consolidated_df):,}")
    print(f"📊 Colonnes: {len(consolidated_df.columns)}")
    
    # Statistiques par type de dataset
    print(f"\n📈 Répartition par type de dataset:")
    if 'Dataset_Type' in consolidated_df.columns:
        type_counts = consolidated_df['Dataset_Type'].value_counts()
        for dataset_type, count in type_counts.items():
            print(f"  {dataset_type}: {count} lignes")
    
    # Statistiques par année
    if 'Year' in consolidated_df.columns:
        print(f"\n📅 Répartition par année:")
        year_counts = consolidated_df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count} lignes")
    
    # Aperçu des données
    print(f"\n📋 Aperçu des 5 premières lignes:")
    print(consolidated_df.head().to_string())
    
    print(f"\n✅ Données sauvegardées dans: {output_path}")

if __name__ == "__main__":
    main()