# Datamining Project Joris FELZINES - Guillaume MANTZARIDES

#dependencies for displaying map
import folium
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px 

#import cluster
#k-means
from sklearn.cluster import KMeans
import numpy as np
#hierarchical cluster
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
# silhouette scores
from sklearn.metrics import silhouette_score, silhouette_samples
# DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
# scaler
from sklearn.preprocessing import StandardScaler

# Librairies pour le text mining
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#########################
#CONFIGURER LE PROGRAMME#
#########################

#indiquer position fichier
csv_file = "./flickr_data_clean.csv"
csv_file_clean = "./flickr_data_cleaned.csv"
# csv_file_clean_with_cluster = "./flickr_data_clean_with_csv.csv"
csv_file_clean_with_cluster = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_clean_with_csv.csv" # Pour windows

#demander nettoyage des données
data_to_clean = 0 # 0 : don't clean, 1 : clean
#/!\ si le nettoyage n'est pas demandé donner un fichier nettoyé à la variable csv_file ci-dessus

# choisir le nombres de ligne aléatoire du fichier, 0 = toutes les lignes
nb_line = 500

# choisir l'algorithme de clusterisation

#clustering_algo = "kmeans"
clustering_algo = "hierarchical all_linkage"
#clustering_algo = "hierarchical average"
#clustering_algo = "hierarchical single"
#clustering_algo = "hierarchical complete"
#clustering_algo = "dbscan"

#ne pas manipuler
nb_cluster_current = 0

#####################
#Liste des Fonctions#
#####################

# Fonction de nettoyage des données
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

# Calcul du silhouette score
def silhouette(current_algo, n_clusters, data):
    global nb_cluster_current
    if (current_algo == "average"):
        current_algo = "hierarchical average"
    elif (current_algo == "single"):
        current_algo = "hierarchical single"
    elif (current_algo == "complete"):
        current_algo = "hierarchical complete"
    
    #indication pour la partie data mining
    nb_cluster_current = n_clusters

    #retrait des lignes contenant des points présents dans aucun cluster
    new_data, new_labels, data_cluster_new = clean_data_without_cluster(data, current_algo)

    silhouette_avg = silhouette_score(data_cluster_new, new_labels, metric='euclidean')
    sample_silhouette_values = silhouette_samples(data_cluster_new, new_labels, metric='euclidean')
    new_data['silhouette ' + current_algo] = sample_silhouette_values

    # silhouette score par cluster
    data_by_cluster = new_data.groupby('cluster '+ current_algo)['silhouette ' + current_algo].mean()
    
    # nombre d'élément par cluster
    nb_by_cluster = new_data['cluster ' + current_algo].value_counts()

    #affiche le graphique lié aux silhouettes scores
    plot_silhouette(sample_silhouette_values, silhouette_avg, new_labels, data_by_cluster, n_clusters, nb_by_cluster, current_algo)

def clean_data_without_cluster(data, current_algo):
    #retirer les lignes hors des cluster
    new_data = data[data['cluster '+ current_algo] != -1]
    data_cluster_new = new_data[['lat','long']]
    new_labels = new_data['cluster ' + current_algo]
    new_data.to_csv(csv_file_clean_with_cluster, index=False, sep=",")

    return new_data, new_labels, data_cluster_new

# Affichage du silhouette plot 
def plot_silhouette(sample_silhouette_values, silhouette_avg, labels, data_by_cluster, n_clusters, nb_by_cluster, current_algo):
    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    
    y_lower = 10
    for i in range(n_clusters):
        # Récupérer le score pour le cluster i
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Remplissage de la silhouette
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_values,
                        alpha=0.7)
        
        # Nommer l'affichage de la silhouette
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}, moy {data_by_cluster[i]:0.2f}, nb {nb_by_cluster[i]}')
        
        y_lower = y_upper + 10
    
    # Ajouter une ligne vertical pour le silhouette score moyen
    plt.axvline(x=silhouette_avg, color='red', linestyle='--', 
                label=f'Silhouette score moyen : {silhouette_avg:.3f}')
    
    plt.title('Silhouette affichage - ' + current_algo)
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.legend(loc='best')
    plt.tight_layout()
    return fig

# Définition de K-means
def k_means():
    kmeans = KMeans(n_clusters=6, init='k-means++')
    kmeans.fit(data_cluster)
    labels = kmeans.labels_
    data['cluster kmeans'] = labels
    silhouette(clustering_algo, 6, data)

# Affichage de Hierarchical Clustering
def plot_dendrogram(model, lbls, title='Hierarchical Clustering Dendrogram', x_title='coordinates', **kwargs):

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
    
    # Affiche le dendogramme associé
    dendrogram(linkage_matrix, labels=lbls, leaf_rotation=90)
    
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Distance')
    fig.show()
    return fig

# Définition de Hierarchical Clustering
def hierarchical(data, labels, metric='euclidean', linkage='average', n_clusters=None, dist_thres=None):
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True, compute_distances=True)
    model = model.fit(data)
    
    txt_title = 'Hierarchical Clustering Dendrogram, linkage: ' + linkage
    f = plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='coordinates')
    
    return model, f

def choosing_linkage():
    if (clustering_algo == "hierarchical average"):
        linkage = ['average']
    elif (clustering_algo == "hierarchical single"):
        linkage = ['single']
    elif (clustering_algo == "hierarchical single"):
        linkage = ['complete']
    else:
        linkage = ['complete', 'average', 'single']

    for link in linkage:
        m, f = hierarchical(data_cluster, list(data_cluster.index), metric='euclidean', linkage=link, n_clusters=6, dist_thres=None)
        labels = m.labels_
        data['cluster hierarchical ' + link] = m.labels_
        silhouette(link, 6, data)

# Définition de DBSCAN
def find_optimal_eps(data, min_pts):
    # calcul la distance avec le kième voisin le plus proche
    neigh = NearestNeighbors(n_neighbors=min_pts)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)
    
    # trie les distances aux kième voisins
    k_distances = np.sort(distances[:, min_pts-1])

    # Créer l'affichage
    plt.plot(range(len(k_distances)), k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {min_pts}th nearest neighbor')

def applied_DBscan(best_eps, best_min_samples):
    # Applique DBscan avec les meilleurs paramètres
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_labels = best_dbscan.fit_predict(scaled_data)
    data['cluster dbscan'] = best_labels
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    silhouette(clustering_algo, n_clusters, data)

# Création de la carte
def creer_map(algo, my_map, liste_color, csv_file):
    data = pd.read_table(csv_file, sep=",", low_memory=False)
    for i in range(len(data)):
        if data.at[i,'cluster ' + algo] == -1:
            continue
        folium.Circle(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip=data.at[i,"temporal_cluster"], popup=data.at[i,"important_words"], radius = 15,color =liste_color[data.at[i,'cluster ' + algo]%len(liste_color)]).add_to(my_map)
    my_map.save(algo + "_map.html")


# Prétraitement des données pour le text mining
def preprocessing(csv_file, csv_file_processed):
    data = pd.read_table(csv_file, sep=",", low_memory=False)

    # On enlève les caractères spéciaux et les chiffres dans tags et title, mais pas les accents ou autres caractères courants
    data['tags'] = data['tags'].str.replace(r'[^a-zA-Z éèôàëêöâäîïùûüç]', ' ', regex=True)
    data['title'] = data['title'].str.replace(r'[^a-zA-Z éèôàëêöâäîïùûüç]', ' ', regex=True)

    # On met tout en minuscule
    data['tags'] = data['tags'].str.lower()
    data['title'] = data['title'].str.lower()

    # Lorsque tags ou title est de longueur 0, on le remplace par le contenu de l'autre
    data['tags'].fillna('', inplace=True)
    data['title'].fillna('', inplace=True)

    # On enlève les "stopwords" anglais et français dans title 
    stop_words = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    data['title'] = data['title'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    data['title'] = data['title'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words_fr]))

    # On enlève les mots trop fréquents ou qui les contiennent ou lettres seules dans title et tags
    frequent_words = ['photo', 'picture', 'image', 'photography', 'photograph', 'photographie','lyon', 'france', 'flickr', 'photographer', 'photographie', 'streetphotography','iphone','lesphotosdevoyage','img','jpg','jpeg','png']
    solo_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    data['title'] = data['title'].apply(lambda x: ' '.join([item for item in x.split() if item not in frequent_words and item not in solo_letters]))
    data['tags'] = data['tags'].apply(lambda x: ' '.join([item for item in x.split() if item not in frequent_words and item not in solo_letters]))

    # On refait un cleaning des data
    data['tags'].fillna('', inplace=True)
    data['title'].fillna('', inplace=True)

    # On joint les tags et les titres dans une nouvelle colonne
    data['text'] = data['title'] + ' ' + data['tags']

    # On met les données dans un fichier csv
    data.to_csv(csv_file_processed, index=False, sep=",")


# Création d'un word cloud avec les données preprocessées
def wordcloud(csv_file_processed):
    data = pd.read_table(csv_file_processed, sep=",", low_memory=False)

    # On essaye de tokeniser la colonne text, mais la tokenization est inefficace pour les tags et title
    # car les mots sont souvent collés les uns aux autres, et il n'y a pas de ponctuation, verbes, etc.
    # La tokenization est donc inutile pour ce dataset en particulier
    data['text'] = data['text'].apply(lambda x: word_tokenize(x))
    
    # On crée une liste de mots avec les titres et les tags
    words = []
    for i in range(len(data)):
        words += data.at[i, 'text']

    # On crée un word cloud
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(' '.join(words))

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()


# On suppose que le fichier a été nettoyé, prétraité et avec une colonne d'entiers représentant à quel cluster appartient chaque ligne
# On cherche donc maintenant à trouver les meilleurs mots pour décrire chaque cluster en utilisant la méthode "term frequency and inverse document frequency"
def TF_IDF(csv_file_processed, nb_clusters, csv_file_tfidf):
    data = pd.read_table(csv_file_processed, sep=",", low_memory=False)

    data['text'] = data['text'].apply(lambda x: word_tokenize(x))

    # On crée un dictionnaire avec les mots et leur fréquence
    words = {}
    for i in range(len(data)):
        for word in data.at[i, 'text']:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    
    # On crée un dictionnaire avec les mots et leur fréquence par cluster
    words_by_cluster = {}
    for i in range(nb_clusters):
        words_by_cluster[i] = {}
    for i in range(len(data)):
        for word in data.at[i, 'text']:
            if word in words_by_cluster[data.at[i, 'cluster dbscan']]:
                words_by_cluster[data.at[i, 'cluster dbscan']][word] += 1
            else:
                words_by_cluster[data.at[i, 'cluster dbscan']][word] = 1

    # On calcule le TF-IDF pour chaque mot
    tf_idf = {}
    for i in range(nb_clusters):
        tf_idf[i] = {}
    for i in range(len(data)):
        for word in data.at[i, 'text']:
            tf = words_by_cluster[data.at[i, 'cluster dbscan']][word] / len(data)
            idf = len(data) / words[word]
            tf_idf[data.at[i, 'cluster dbscan']][word] = tf * idf
        data.at[i, 'text'] = ' '.join(data.at[i, 'text'])

    # On affiche les mots les plus importants pour chaque cluster
    # for i in range(nb_clusters):
    #     print("Cluster", i)
    #     print(sorted(tf_idf[i], key=tf_idf[i].get, reverse=True)[:10])

    # On ajoute une colonne pour chaque ligne avec les 10 mots les plus importants du cluster auquel elle appartient
    for i in range(len(data)):
        data.at[i, 'important_words'] = ' '.join(sorted(tf_idf[data.at[i, 'cluster dbscan']], key=tf_idf[data.at[i, 'cluster dbscan']].get, reverse=True)[:10])

    # On met les données dans un fichier csv
    data.to_csv(csv_file_tfidf, index=False, sep=",")


# Étude sur l'axe temporel
# Pour chaque cluster, on détermine l'intervalle de temps où les photos ont été prises
# Si cet intervalle est très court, cela signifie que c'est un cluster temporel ponctuel
# Si cet intervalle est très long, cela signifie que c'est un cluster temporel permanent
# On a ces colonnes : date_taken_minute,date_taken_hour,date_taken_day,date_taken_month,date_taken_year
# Il faut en faire une seule date sur lesquelles on peut calculer l'intervalle entre 
# Date max et date min pour chaque cluster
def analyse_temporelle(csv_file, nb_clusters, csv_file_temporel):
    data = pd.read_table(csv_file, sep=",", low_memory=False)

    # On crée une colonne date
    data.rename(columns={'date_taken_year': 'year', 'date_taken_month': 'month', 'date_taken_day': 'day', 'date_taken_hour': 'hour', 'date_taken_minute': 'minute'}, inplace=True)
    data['date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    # On crée un dictionnaire avec les dates min et max pour chaque cluster
    dates = {}
    for i in range(nb_clusters):
        dates[i] = {'min': None, 'max': None}
    for i in range(len(data)):
        if dates[data.at[i, 'cluster dbscan']]['min'] is None or data.at[i, 'date'] < dates[data.at[i, 'cluster dbscan']]['min']:
            dates[data.at[i, 'cluster dbscan']]['min'] = data.at[i, 'date']
        if dates[data.at[i, 'cluster dbscan']]['max'] is None or data.at[i, 'date'] > dates[data.at[i, 'cluster dbscan']]['max']:
            dates[data.at[i, 'cluster dbscan']]['max'] = data.at[i, 'date']

    # On affiche les dates min et max pour chaque cluster
    # for i in range(nb_clusters):
    #     print("Cluster", i)
    #     print("Date min:", dates[i]['min'])
    #     print("Date max:", dates[i]['max'])

    # On regarde maintenant la différence entre les dates min et max pour chaque cluster
    # Si la différence est très grande (arbitrairement plus de 1 mois), cela signifie que c'est un cluster temporel permanent
    # On ajoute donc une autre colonne pour chaque cluster qui indique si c'est un cluster temporel ponctuel ou permanent
    for i in range(len(data)):
        if (dates[data.at[i, 'cluster dbscan']]['max'] - dates[data.at[i, 'cluster dbscan']]['min']).days > 7:
            data.at[i, 'temporal_cluster'] = 'permanent'
        else:
            data.at[i, 'temporal_cluster'] = 'ponctuel'
    
    # On affiche pour chaque cluster si il est temporel ponctuel ou permanent
    # for i in range(nb_clusters):
    #     print("Cluster", i)
    #     print(data[data['cluster dbscan'] == i]['temporal_cluster'].value_counts())
    
    # On met les données dans un fichier csv
    data.to_csv(csv_file_temporel, index=False, sep=",")


#################
# Préparer data #
#################

if (data_to_clean == 1):
    cleaning(csv_file, csv_file_clean)
    file_to_read = csv_file_clean
else:
    file_to_read = csv_file

data = pd.read_table(file_to_read, sep=",", low_memory=False)

if (nb_line == 0):
    pass
else:
    # Select aléatoirement un nombre de ligne
    random_data = data.sample(n=nb_line, random_state=42)  # 'random_state' permet assure la reproductibilitée
    data = random_data.reset_index(drop=True)

#supprime les colonnes excepté long and lat
data_cluster = data.drop(columns=["id","user","tags","title","date_taken_minute","date_taken_hour","date_taken_day","date_taken_month","date_taken_year"])

# Rééchelonne les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_cluster)
# create a DataFrame
data_cluster = pd.DataFrame(data=scaled_data, columns=data_cluster.columns)

#########
#k-means#
#########
if (clustering_algo == "kmeans"):
    def elbow_method():
        # échelle de k
        range_k = range(1, 50)
        # list des scores d'inertie
        inertias = []

        # faire varier k et appliquer k-means
        for i in range_k:
            # appliquer k-means avec i cluster
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(data_cluster)
            # rajouter l'inertie à la liste
            inertias.append(kmeans.inertia_)

        # visualiser
        n = len(inertias)
        xticks_new = np.arange(1, n+1)
        plt.plot(xticks_new, inertias[0:n], 'bx-')
        plt.title('Finding the optimal number of clusters')
        plt.xlabel('# clusters')
        plt.ylabel('Sum of squared distances')

    k_means()

#########################
#Hierarchical Clustering#
#########################
elif (clustering_algo == "hierarchical average" or clustering_algo == "hierarchical single" or clustering_algo == "hierarchical complete" or clustering_algo == "hierarchical all_linkage"): 
    choosing_linkage()

########
#DBScan#
########
elif (clustering_algo == "dbscan"):
    best_eps = 0.008
    best_min_samples = 10
    # #trouver le meilleur paramètre
    # min_pnts = 15 # on trouve 0,25
    # find_optimal_eps(scaled_data, min_pnts)
    applied_DBscan(best_eps,best_min_samples)


############################################################################
#Description des zones d'intérêt en utilisant des techniques de text mining#
############################################################################

# Preprocessing des données
after_clustering = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_clustered.csv"
csv_file_processed_sample = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_processed-SAMPLE.csv"

preprocessing(after_clustering, csv_file_processed_sample)
wordcloud(csv_file_processed_sample)

csv_file_tfidf = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_tfidf.csv"
TF_IDF(csv_file_processed_sample, nb_cluster_current, csv_file_tfidf)

csv_file_temporel = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_final.csv"
analyse_temporelle(csv_file_tfidf, nb_cluster_current, csv_file_temporel)

##############################
#AFFICHER LA CARTE ET LES MARQUEURS, POUR LA VOIR L'OUVRIR À LA MAIN DANS UN NAVIGATEUR
##############################

my_map = folium.Map(location=(45.75, 4.83))
liste_color = [
    "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray", "DarkGreen", "DarkKhaki", "DarkMagenta", "DarkOliveGreen", "DarkOrange",
    "AliceBlue", "AntiqueWhite", "Aqua", "Aquamarine", "Azure", "Beige", "Bisque", "Black", "BlanchedAlmond", "Blue",
    "BlueViolet", "Brown", "BurlyWood", "CadetBlue", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Cornsilk", "Crimson",
    "Cyan"
]

if (clustering_algo == "hierarchical all_linkage"):
    creer_map('hierarchical average', my_map, liste_color)
    creer_map('hierarchical single', my_map, liste_color)
    creer_map('hierarchical complete', my_map, liste_color)

else:
    creer_map(clustering_algo, my_map, liste_color, csv_file_temporel)

#afficher tous les schémas
plt.show()

