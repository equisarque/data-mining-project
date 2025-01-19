# Datamining Project Joris FELZINES - Guillaume MANTZARIDES

#dependencies for displaying map
import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import cluster
#k-means
from sklearn.cluster import KMeans
import numpy as np
#hierarchical cluster
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
# silhouette scores
from sklearn.metrics import silhouette_score, silhouette_samples

csv_file = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data2.csv"
csv_file_clean = "./flickr_data_clean.csv"
csv_file_clean_sample = "./flickr_data_clean-SAMPLE.csv"

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

# Randomly select 500 rows
random_data = data.sample(n=500, random_state=42)  # 'random_state' ensures reproducibility

# Optionally, reset the index
data = random_data.reset_index(drop=True)

#################################AFFICHAGE AVEC SEABORN DU PAIRPLOT

## NE SERT À RIEN SUR LE PROJET PUISQUE REVIENT À UNE CARTE
# cols=['lat','long']
## pair plot
# pairplt = sns.pairplot(data=data[cols], diag_kind='hist')
## ne fonctionne pas
# add a title to the figure
#pairplt.figure.suptitle('Pairplot', y=1.04)
## Add new legend
#pairplt.add_legend(loc='upper right')
## Adjust the layout to prevent title overlap
# plt.tight_layout()
# plt.show()

# scaler
#on ne scale pas parce que lat et long sont comparable

#drop column except long and lat
data_cluster = data.drop(columns=["id","user","tags","title","date_taken_minute","date_taken_hour","date_taken_day","date_taken_month","date_taken_year"])

################################
# K MEANS
################################

## range of k
#range_k = range(1, 50)
## a list of intertia scores
#inertias = []

# # vary k and apply k-means
# for i in range_k:
#     # apply k-means with i clusters
#     kmeans = KMeans(n_clusters=i, init='k-means++')
#     # fit data 
#     kmeans.fit(data_cluster)
#     # append inertia to the list
#     inertias.append(kmeans.inertia_)

# # visualise
# n = len(inertias)
# xticks_new = np.arange(1, n+1)
# plt.plot(xticks_new, inertias[0:n], 'bx-')
# plt.title('Finding the optimal number of clusters')
# plt.xlabel('# clusters')
# plt.ylabel('Sum of squared distances')
# plt.show()
# #finding the good number of cluster with elbow, result = 6

# kmeans = KMeans(n_clusters=6, init='k-means++')
# kmeans.fit(data_cluster)
# labels = kmeans.labels_
# data['cluster kmeans'] = labels

############################
# Calculate silhouette scores for k-means
############################

# silhouette_avg = silhouette_score(data_cluster, labels, metric='euclidean')
# sample_silhouette_values = silhouette_samples(data_cluster, labels, metric='euclidean')
# #data['silhouette kmeans'] = sample_silhouette_values

# print(f"Average silhouette score: {silhouette_avg}")
# print(f"Sample Silhouette values: {sample_silhouette_values}")
#data.groupby('cluster kmeans')['silhouette kmeans'].mean()

# def plot_silhouette(sample_silhouette_values, silhouette_avg, labels, n_clusters=3):
#     # Create the plot
#     fig = plt.figure(figsize=(10, 6))
    
#     y_lower = 10
#     for i in range(n_clusters):
#         # Get silhouette scores for cluster i
#         ith_cluster_values = sample_silhouette_values[labels == i]
#         ith_cluster_values.sort()
        
#         size_cluster_i = ith_cluster_values.shape[0]
#         y_upper = y_lower + size_cluster_i
        
#         # Fill the silhouette
#         plt.fill_betweenx(np.arange(y_lower, y_upper),
#                          0, ith_cluster_values,
#                          alpha=0.7)
        
#         # Label the silhouette plots
#         plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
        
#         y_lower = y_upper + 10
    
#     # Add vertical line for average silhouette score
#     plt.axvline(x=silhouette_avg, color='red', linestyle='--', 
#                 label=f'Average Silhouette: {silhouette_avg:.3f}')
    
#     plt.title('Silhouette Plot')
#     plt.xlabel('Silhouette Coefficient')
#     plt.ylabel('Cluster')
#     plt.legend(loc='best')
#     plt.tight_layout()
    
#     return fig

############################
# Hierarchical Clustering
############################



##########################
########################### ILS FAUT RÉDUIRE LE NOMBRE DE POINT À UNE ZONE GÉOGRAPHIQUE RESTREINTE,CAR ÇA MARCHE PAS POUR TOUTES LES DATA ET COMPARER AVEC AUTRE ALGO

def plot_dendrogram(model, lbls, title='Hierarchical Clustering Dendrogram', x_title='coordinates', **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([
           model.children_,
           model.distances_,
           counts
       ]).astype(float)

    fig = plt.figure(figsize=(12, 8))
    
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=lbls, leaf_rotation=90)
    
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Distance')
    
    fig.show()
    
    return fig

def hierarchical(data, labels, metric='euclidean', linkage='average', n_clusters=None, dist_thres=None):
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True, compute_distances=True)
    model = model.fit(data)
    
    txt_title = 'Hierarchical Clustering Dendrogram' + ', linkage: ' + linkage
    f = plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='Flowers')
    
    return model, f

# types of linkage to consider
linkage = ['complete', 'average', 'single']

for link in linkage:
    m, f = hierarchical(data_cluster, list(data_cluster.index), metric='euclidean', linkage=link, n_clusters=6, dist_thres=None)
    
    data['cluster ' + link] = m.labels_
    silhouette_avg = silhouette_score(data_cluster, m.labels_, metric='euclidean')
    sample_silhouette_values = silhouette_samples(data_cluster, m.labels_, metric='euclidean')
    data['silhouette ' + link] = sample_silhouette_values
    
    print(f"Linkage: {link}, silhouette score: {silhouette_avg}")

plt.show()
# for link in linkage:
#     m, f = hierarchical(scaled_data, list(scaled_data_df.index), metric='euclidean', linkage=link, n_clusters=3, dist_thres=None)
    
#     data_cleaned['cluster ' + link] = m.labels_
#     silhouette_avg = silhouette_score(scaled_data, m.labels_, metric='euclidean')
#     sample_silhouette_values = silhouette_samples(scaled_data, m.labels_, metric='euclidean')
#     data_cleaned['silhouette ' + link] = sample_silhouette_values
    
#     print(f"Linkage: {link}, silhouette score: {silhouette_avg}")

##############################
#AFFICHER LA CARTE ET UN MARQUEUR, POUR LA VOIR L'OUVRIR À LA MAIN DANS UN NAVIGATEUR
##############################

my_map = folium.Map(location=(45.75, 4.83))
liste_color = [
    "AliceBlue", "AntiqueWhite", "Aqua", "Aquamarine", "Azure", "Beige", "Bisque", "Black", "BlanchedAlmond", "Blue",
    "BlueViolet", "Brown", "BurlyWood", "CadetBlue", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Cornsilk", "Crimson",
    "Cyan", "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray", "DarkGreen", "DarkKhaki", "DarkMagenta", "DarkOliveGreen", "DarkOrange"
]

# # for i in range(len(data_cluster)):
# #     folium.Circle(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip=data.at[i,"title"], radius = 3,color=liste_color[data.at[i,"cluster kmeans"]%len(liste_color)]).add_to(my_map)
# # my_map.save("map.html")

# Average silhouette score: 0.9824358047317638
# Sample Silhouette values: [0.9855359  0.97559046 0.98166426 ... 0.97620527 0.98779591 0.98793261]


##############################
# Description of areas of interest using text pattern mining
##############################

# Preprocessing of tags and title

# csv_file_to_be_processed = csv_file_clean_sample
# csv_file_processed = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_sample_processed.csv"

# def preprocessing(csv_file, csv_file_processed):
#     data = pd.read_table(csv_file, sep=",", low_memory=False)

#     # On supprime les lignes dont les tags et title sont vides
#     data = data.dropna(subset=['tags', 'title'])

#     # On supprime les tags et title qui ne sont pas des mots
#     data['tags'] = data['tags'].str.replace(r'[^a-zA-Z ]', '')
#     data['title'] = data['title'].str.replace(r'[^a-zA-Z ]', '')

#     # On met tout en minuscule
#     data['tags'] = data['tags'].str.lower()

#     # On met les tags dans une liste
#     data['tags'] = data['tags'].str.split()

#     # On enlève les "stopwords" anglais
#     stopwords_english = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 't', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', 't', 'd', 'did', 'didn', 't', 'do', 'does', 'doesn', 't', 'doing', 'don', 't', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', 't', 'has', 'hasn', 't', 'have', 'haven', 't', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 't', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 't', 'more', 'most', 'mustn', 't', 'my', 'myself', 'needn', 't', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 't', 'she', 'should', 'should', 've']
