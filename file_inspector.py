import os
import glob
from pathlib import Path

def inspect_file_format(file_path):
    """Inspecte le format réel d'un fichier"""
    try:
        with open(file_path, 'rb') as f:
            # Lire les premiers octets pour identifier le format
            header = f.read(512)
            
        # Vérifier différents formats
        if header.startswith(b'PK'):
            return "ZIP/XLSX (Excel moderne)"
        elif header.startswith(b'\xd0\xcf\x11\xe0'):
            return "OLE2/XLS (Excel classique)"
        elif header.startswith(b'<html') or header.startswith(b'<!DOCTYPE'):
            return "HTML"
        elif header.startswith(b'<?xml'):
            return "XML"
        elif b'<table' in header[:100]:
            return "HTML avec tableaux"
        elif header.startswith(b'\xef\xbb\xbf'):
            return "UTF-8 BOM (probablement CSV/TXT)"
        elif b',' in header and b'\n' in header:
            return "CSV"
        elif b'\t' in header and b'\n' in header:
            return "TSV (Tab Separated)"
        else:
            # Essayer de décoder comme texte
            try:
                text_content = header.decode('utf-8', errors='ignore')
                if 'Countries' in text_content or 'Total population' in text_content:
                    return "Texte avec données démographiques"
                else:
                    return f"Format inconnu - début: {text_content[:50]}..."
            except:
                return f"Format binaire inconnu - hex: {header[:20].hex()}"
                
    except Exception as e:
        return f"Erreur: {str(e)}"

def show_file_content_sample(file_path, max_lines=10):
    """Affiche un échantillon du contenu du fichier"""
    try:
        # Essayer différents encodages
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read(2000)  # Lire les premiers 2000 caractères
                print(f"✅ Lecture réussie avec l'encodage: {encoding}")
                break
            except:
                continue
        
        if content is None:
            print("❌ Impossible de lire le fichier avec les encodages testés")
            return
        
        lines = content.split('\n')[:max_lines]
        print(f"📝 Aperçu ({len(lines)} premières lignes):")
        print("-" * 60)
        for i, line in enumerate(lines, 1):
            print(f"{i:2}: {line[:80]}{'...' if len(line) > 80 else ''}")
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture: {str(e)}")

def main():
    """Fonction principale d'inspection"""
    excel_folder = "covid-dashboard-vue/data/excel"
    
    # Trouver tous les fichiers
    excel_files = glob.glob(os.path.join(excel_folder, "*.xls*"))
    
    if not excel_files:
        print(f"Aucun fichier trouvé dans {excel_folder}")
        return
    
    print(f"🔍 Inspection de {len(excel_files)} fichiers...\n")
    
    # Inspecter chaque fichier
    for i, file_path in enumerate(sorted(excel_files)[:3]):  # Limiter à 3 fichiers pour commencer
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        print(f"📁 Fichier {i+1}: {filename}")
        print(f"📏 Taille: {file_size:,} octets")
        
        # Détecter le format
        format_detected = inspect_file_format(file_path)
        print(f"🔬 Format détecté: {format_detected}")
        
        # Afficher un échantillon du contenu
        show_file_content_sample(file_path)
        
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()