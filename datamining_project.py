# Datamining Project Joris FELZINES - Guillaume MANTZARIDES

#dependencies for displaying map
import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data2.csv"
csv_file_clean = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_clean.csv"

def cleaning(csv_file, csv_file_clean):
    data = pd.read_table(csv_file, sep=",", low_memory=False)

    # On supprime certaines colonnes, les dates d'upload et les 3 dernières colonnes quasiment vides car pas utiles pour l'analyse
    data.drop(data.columns[[11,12,13,14,15,16,17,18]], axis=1, inplace=True)

    # On renomme les colonnes avec des espaces
    data.rename(columns={' user': 'user',
                        ' lat': 'lat',
                        ' long': 'long',
                        ' tags': 'tags',
                        ' title': 'title',
                        ' date_taken_year': 'date_taken_year',
                        ' date_taken_month': 'date_taken_month',
                        ' date_taken_day': 'date_taken_day',
                        ' date_taken_hour': 'date_taken_hour',
                        ' date_taken_minute': 'date_taken_minute'} ,
                        inplace=True)

    # On supprime les duplicats
    data.drop_duplicates(keep='first', inplace=True)

    # On supprime les lignes ou les valeurs manquantes dérangent
    columns_to_check = ['id', 'user', 'lat', 'long', 'date_taken_year', 'date_taken_month', 'date_taken_day', 'date_taken_hour', 'date_taken_minute']
    data.dropna(subset=columns_to_check, inplace=True)

    # On remet les dates de prises de photos dans le bon sens
    cols_to_shift = ['date_taken_minute', 'date_taken_hour', 'date_taken_day', 'date_taken_month', 'date_taken_year']
    shifted_order_minute = cols_to_shift[1:] + cols_to_shift[:1]
    shifted_order_hour = cols_to_shift[2:] + cols_to_shift[:2]
    shifted_order_day = cols_to_shift[3:] + cols_to_shift[:3]
    shifted_order_month = cols_to_shift[4:] + cols_to_shift[:4]

    # Pour chaque ligne, on vérifie si les valeurs sont cohérentes
    for index, row in data.iterrows():
        if row['date_taken_minute'] > 999:
            data.loc[index, cols_to_shift] = row[shifted_order_minute].values

        if row['date_taken_hour'] > 999:
            data.loc[index, cols_to_shift] = row[shifted_order_hour].values

        if row['date_taken_day'] > 999:
            data.loc[index, cols_to_shift] = row[shifted_order_day].values

        if row['date_taken_month'] > 999:
            data.loc[index, cols_to_shift] = row[shifted_order_month].values

        if row['date_taken_year'] > 2025:
            data.drop(index, inplace=True)

    # On convertit les minutes en int
    data['date_taken_minute'] = data['date_taken_minute'].astype(int)

    # Mettre les données nettoyées dans un fichier csv
    data.to_csv(csv_file_clean, index=False, sep=",")

    return True

# cleaning(csv_file, csv_file_clean)

data = pd.read_table(csv_file_clean, sep=",", low_memory=False)

#################################AFFICHAGE AVEC SEABORN
# cols=['lat','long']
# # Plot the pairplot
# fig = plt.figure(figsize = (20,20))
# # pair plot
# g = sns.pairplot(data=data[cols], diag_kind='hist')
# # add a title to the figure
# g.figure.suptitle('Pairplot', y=1.04)
# # Add new legend
# g.add_legend(loc='upper right')
# # Adjust the layout to prevent title overlap
# plt.tight_layout()
# plt.show()

##############################
#AFFICHER LA CARTE ET UN MARQUEUR, POUR LA VOIR L'OUVRIR À LA MAIN DANS UN NAVIGATEUR
##############################

my_map = folium.Map(location=(45.75, 4.83))
for i in range(len(data)):
    folium.CircleMarker(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip="Afficher titre", popup=data.at[i,"title"], radius = 3).add_to(my_map)
my_map.save("map.html")