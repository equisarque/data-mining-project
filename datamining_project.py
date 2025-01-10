# Datamining Project Joris FELZINES - Guillaume MANTZARIDES

#dependencies for displaying map
import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_table("C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data2.csv", sep=",", low_memory=False)
data = pd.read_table("flickr_data2_sample.csv", sep=",", low_memory=False)

# Clean : on supprime certaines colonnes, les dates d'upload et les 3 dernières colonnes quasiment vides car pas utiles pour l'analyse
# Certaines lignes ont également des valeurs décalées au niveau des dates de prises de photos, on les remet dans le bon sens
# Enfin, il manque des données de date pour certaines photos, on les supprime afin de garder une cohérence dans les données car elles sont peu nombreuses (moins de 5 pour 420 000 lignes)
data.drop(data.columns[[11,12,13,14,15,16,17,18]], axis=1, inplace=True)
print(len(data))

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
print(len(data))

# On supprime les lignes ou les valeurs manquantes dérangent
columns_to_check = ['id', 'user', 'lat', 'long', 'date_taken_year', 'date_taken_month', 'date_taken_day', 'date_taken_hour', 'date_taken_minute']
data.dropna(subset=columns_to_check)
print(len(data))

# print les rows avec des valeurs manquantes dans columns_to_check
print(data[data[columns_to_check].isnull().any(axis=1)][columns_to_check])

# datatmp = data.head(10)

# for row in datatmp.iterrows():
#     print(row)
#     if row["date_taken_minute"] not in range(0,60):
#         if row["date_taken_minute"] > 1000:
#             print('a')

# print(data.info())
# print(data.describe()



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
    folium.CircleMarker(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip="Afficher titre", popup=data.at[i,"title"], radius = 3,).add_to(my_map)
my_map.save("map.html")