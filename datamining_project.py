# Datamining Project Joris FELZINES - Guillaume MANTZARIDES


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dependencies for displaying map
import folium

data = pd.read_table("C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data2.csv", sep=",", low_memory=False)

# On supprime certaines colonnes : les dates d'upload et les 3 dernières colonnes quasiment vides
data.drop(data.columns[[11,12,13,14,15,16,17,18]], axis=1, inplace=True)

# minute_values = [i for i in range(0,60)]
# data_mask = data["date_taken_minute"] not in [i for i in range(0,60)]
# data = data[data_mask]

print(data.info())
print(data.describe())

##############################
#AFFICHER LA CARTE ET UN MARQUEUR, POUR LA VOIR L'OUVRIR À LA MAIN DANS UN NAVIGATEUR
##############################

# my_map = folium.Map(location=(45.75, 4.83))

# folium.CircleMarker(
#     location=[45.75, 4.83],
#     tooltip="Click me!",
#     popup="Mt. Hood Meadows",
#     radius = 3,
# ).add_to(my_map)

# my_map.save("map.html")