import kagglehub 
import os 
import pandas as pd
import shutil

# test=kagglehub.dataset_download("imdevskp/corona-virus-report")
# destination = "C:/Users/matte/Desktop/Cours_B3/MSPR_Datascience/final/MSPR/covid/dataset"
# os.rename(test, destination)
# liste = os.listdir(destination)
# for element in liste : 
#     path = destination + "/" + element
#     print(path)
#     df = pd.read_csv(path)
#     df_clean = df.dropna()
#     df_clean = df_clean.drop_duplicates()



# test = kagglehub.dataset_download("imdevskp/corona-virus-report")
# destination = "C:/Users/medyv/Desktop/EPSI/MSPR bloc 1/MSPR/dataset"
# os.rename(test, destination)


# liste = os.listdir(destination)


# for element in liste:
#     if element.endswith('.csv'):  
        
#         path = os.path.join(destination, element)
        
        
#         clean_filename = element.replace('.csv', '_clean.csv')
#         clean_path = os.path.join(destination, clean_filename)
        
#         print(f"Nettoyage du fichier : {element}")
        
   
#         df = pd.read_csv(path)
        
  
#         print(f"Avant nettoyage - Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
        
    
#         df_clean = df.dropna()
        
   
#         df_clean = df_clean.drop_duplicates()
        
   
#         print(f"Après nettoyage - Lignes : {df_clean.shape[0]}, Colonnes : {df_clean.shape[1]}")
       
#         df_clean.to_csv(clean_path, index=False)
#         print(f"Fichier {clean_filename} créé\n")







destination = os.path.join(os.path.dirname(__file__), 'dataset')
clean_destination = os.path.join(os.path.dirname(__file__), 'dataset_clean')


os.makedirs(clean_destination, exist_ok=True)


liste = os.listdir(destination)


for element in liste:
    if element.endswith('.csv'):
        path = os.path.join(destination, element)
        
      
        clean_filename = element.replace('.csv', '_clean.csv')
        clean_path = os.path.join(clean_destination, clean_filename)
        
        print(f"Nettoyage du fichier : {element}")
        
   
        df = pd.read_csv(path)
        

        print(f"Avant nettoyage - Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
        

        df_clean = df.dropna()
        
 
        df_clean = df_clean.drop_duplicates()
        
       
        print(f"Après nettoyage - Lignes : {df_clean.shape[0]}, Colonnes : {df_clean.shape[1]}")
        
      
        df_clean.to_csv(clean_path, index=False)
        print(f"Fichier {clean_filename} créé\n")