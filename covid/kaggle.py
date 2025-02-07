import kagglehub
import os 
import pandas as pd

test=kagglehub.dataset_download("imdevskp/corona-virus-report")
destination = "C:/Users/matte/Desktop/Cours_B3/MSPR_Datascience/final/MSPR/covid/dataset"
os.rename(test, destination)
liste = os.listdir(destination)
# for element in liste : 
#     path = destination + "/" + element
#     print(path)
#     df = pd.read_csv(path)
#     df_clean = df.dropna()
#     df_clean = df_clean.drop_duplicates()



