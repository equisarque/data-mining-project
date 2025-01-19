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


#########################
#CONFIGURER LE PROGRAMME#
#########################

# activer la partie Data Mining (le code de la Fougère)
data_mining = 0 # 1 = activé, 0 = désactivé


# activer la partie Clusterisation (le code du lutin)
clusterisation = 1 # 1 = activé, 0 = désactivé

#indiquer position fichier
csv_file = "./flickr_data_clean.csv"
csv_file_clean = "./flickr_data_clean.csv"

#demander nettoyage des données
data_to_clean = 0 # 0 : don't clean, 1 : clean
#/!\ si le nettoyage n'est pas demandé donner un fichier nettoyé à la variable csv_file ci-dessus

# choisir le nombres de ligne aléatoire du fichier, 0 = toutes les lignes
nb_line = 500

# choisir l'algorithme de clusterisation

clustering_algo = "cluster kmeans"
# clustering_algo = "cluster hierarchical average"
# clustering_algo = "cluster hierarchical single"
# clustering_algo = "cluster hierarchical complete"
# clustering_algo = "cluster dbscan"

#################
# Preparing data#
#################
if (clusterisation==1):

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

    if (data_to_clean == 1):
        cleaning(csv_file, csv_file_clean)
        file_to_read = csv_file_clean
    else:
        file_to_read = csv_file

    data = pd.read_table(file_to_read, sep=",", low_memory=False)

    if (nb_line == 0):
        pass
    else:
        # Randomly select nb_line rows
        random_data = data.sample(n=nb_line, random_state=42)  # 'random_state' ensures reproducibility
        # Optionally, reset the index
        data = random_data.reset_index(drop=True)

    #drop column except long and lat
    data_cluster = data.drop(columns=["id","user","tags","title","date_taken_minute","date_taken_hour","date_taken_day","date_taken_month","date_taken_year"])

    print(data)
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
        ################################
        # K MEANS
        ################################
    if (clustering_algo == "cluster kmeans"):
        ################################
        # ELBOW METHOD
        ################################

        # # range of k
        # range_k = range(1, 50)
        # # a list of intertia scores
        # inertias = []

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

        kmeans = KMeans(n_clusters=6, init='k-means++')
        kmeans.fit(data_cluster)
        labels = kmeans.labels_
        data['cluster kmeans'] = labels

        ############################
        # Calculate silhouette scores for k-means
        ############################

        silhouette_avg = silhouette_score(data_cluster, labels, metric='euclidean')
        sample_silhouette_values = silhouette_samples(data_cluster, labels, metric='euclidean')
        data['silhouette kmeans'] = sample_silhouette_values

        print(f"Sample Silhouette values: {sample_silhouette_values}")
        print(data.groupby('cluster kmeans')['silhouette kmeans'].mean())
        # number of elements per cluster
        print(data['cluster kmeans'].value_counts())
        def plot_silhouette(sample_silhouette_values, silhouette_avg, labels, n_clusters=6):
            # Create the plot
            fig = plt.figure(figsize=(10, 6))
            
            y_lower = 10
            for i in range(n_clusters):
                # Get silhouette scores for cluster i
                ith_cluster_values = sample_silhouette_values[labels == i]
                ith_cluster_values.sort()
                
                size_cluster_i = ith_cluster_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                # Fill the silhouette
                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_values,
                                alpha=0.7)
                
                # Label the silhouette plots
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
                
                y_lower = y_upper + 10
            
            # Add vertical line for average silhouette score
            plt.axvline(x=silhouette_avg, color='red', linestyle='--', 
                        label=f'Average Silhouette: {silhouette_avg:.3f}')
            
            plt.title('Silhouette Plot')
            plt.xlabel('Silhouette Coefficient')
            plt.ylabel('Cluster')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
            return fig

        plot_silhouette(sample_silhouette_values, silhouette_avg, labels, n_clusters=6)
    ############################
    # Hierarchical Clustering
    ############################



    ##########################
    ########################### ILS FAUT RÉDUIRE LE NOMBRE DE POINT À UNE ZONE GÉOGRAPHIQUE RESTREINTE,CAR ÇA MARCHE PAS POUR TOUTES LES DATA ET COMPARER AVEC AUTRE ALGO

    # def plot_dendrogram(model, lbls, title='Hierarchical Clustering Dendrogram', x_title='coordinates', **kwargs):
    #     # Create linkage matrix and then plot the dendrogram

    #     # create the counts of samples under each node
    #     counts = np.zeros(model.children_.shape[0])
    #     n_samples = len(model.labels_)
    #     for i, merge in enumerate(model.children_):
    #         current_count = 0
    #         for child_idx in merge:
    #             if child_idx < n_samples:
    #                 current_count += 1
    #             else:
    #                 current_count += counts[child_idx - n_samples]
    #         counts[i] = current_count

    #     linkage_matrix = np.column_stack([
    #            model.children_,
    #            model.distances_,
    #            counts
    #        ]).astype(float)

    #     fig = plt.figure(figsize=(12, 8))
        
    #     # Plot the corresponding dendrogram
    #     dendrogram(linkage_matrix, labels=lbls, leaf_rotation=90)
        
    #     plt.title(title)
    #     plt.xlabel(x_title)
    #     plt.ylabel('Distance')
        
    #     fig.show()
        
    #     return fig

    # def hierarchical(data, labels, metric='euclidean', linkage='average', n_clusters=None, dist_thres=None):
    #     model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True, compute_distances=True)
    #     model = model.fit(data)
        
    #     txt_title = 'Hierarchical Clustering Dendrogram' + ', linkage: ' + linkage
    #     f = plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='Flowers')
        
    #     return model, f

    # # types of linkage to consider
    # linkage = ['complete', 'average', 'single']

    # for link in linkage:
    #     m, f = hierarchical(data_cluster, list(data_cluster.index), metric='euclidean', linkage=link, n_clusters=6, dist_thres=None)
        
    #     data['cluster ' + link] = m.labels_
    #     silhouette_avg = silhouette_score(data_cluster, m.labels_, metric='euclidean')
    #     sample_silhouette_values = silhouette_samples(data_cluster, m.labels_, metric='euclidean')
    #     data['silhouette ' + link] = sample_silhouette_values
        
    #     print(f"Linkage: {link}, silhouette score: {silhouette_avg}")

    # plt.show()

    ##############################
    #AFFICHER LA CARTE ET LES MARQUEURS, POUR LA VOIR L'OUVRIR À LA MAIN DANS UN NAVIGATEUR
    ##############################

    my_map = folium.Map(location=(45.75, 4.83))
    liste_color = [
        "AliceBlue", "AntiqueWhite", "Aqua", "Aquamarine", "Azure", "Beige", "Bisque", "Black", "BlanchedAlmond", "Blue",
        "BlueViolet", "Brown", "BurlyWood", "CadetBlue", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Cornsilk", "Crimson",
        "Cyan", "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray", "DarkGreen", "DarkKhaki", "DarkMagenta", "DarkOliveGreen", "DarkOrange"
    ]

    for i in range(len(data_cluster)):
        folium.Circle(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip=data.at[i,"title"], radius = 3,color=liste_color[data.at[i,clustering_algo]%len(liste_color)]).add_to(my_map)
    my_map.save("map.html")

if (data_mining == 1):
    ##############################
    # Description des zones d'intérêt en utilisant des techniques de text mining
    ##############################

    # Librairies pour le text mining
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Preprocessing des données

    csv_file_processed = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_processed.csv"
    csv_file_processed_sample = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_processed-SAMPLE.csv"

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
        frequent_words = ['photo', 'picture', 'image', 'photography', 'photograph', 'photographie','lyon', 'france', 'flickr', 'photographer', 'photographie', 'streetphotography','iphone','lesphotosdevoyage']
        solo_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        data['title'] = data['title'].apply(lambda x: ' '.join([item for item in x.split() if item not in frequent_words and item not in solo_letters]))
        data['tags'] = data['tags'].apply(lambda x: ' '.join([item for item in x.split() if item not in frequent_words and item not in solo_letters]))

        # On refait un cleaning des data
        data['tags'].fillna('', inplace=True)
        data['title'].fillna('', inplace=True)

        # On met les données dans un fichier csv
        data.to_csv(csv_file_processed, index=False, sep=",")

    # Création d'un word cloud avec les données preprocessées

    def wordcloud(csv_file_processed):
        data = pd.read_table(csv_file_processed, sep=",", low_memory=False)

        data['tags'].fillna('', inplace=True)
        data['title'].fillna('', inplace=True)

        # On crée une liste de mots avec les titres et les tags
        words = []
        for i in range(len(data)):
            words += data.at[i, 'title'].split()
            words += data.at[i, 'tags'].split()

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
    def TF_IDF(csv_file_processed, nb_clusters):
        data = pd.read_table(csv_file_processed, sep=",", low_memory=False)

        # On garantit que les tags et les titres sont bien des strings
        data['tags'] = data['tags'].astype(str)
        data['title'] = data['title'].astype(str)

        # On commence par joindre les tags et les titres dans une nouvelle colonne puis on tokenise
        data['text'] = data['title'] + ' ' + data['tags']
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
                if word in words_by_cluster[data.at[i, 'cluster']]:
                    words_by_cluster[data.at[i, 'cluster']][word] += 1
                else:
                    words_by_cluster[data.at[i, 'cluster']][word] = 1

        # On calcule le TF-IDF pour chaque mot
        tf_idf = {}
        for i in range(nb_clusters):
            tf_idf[i] = {}
        for i in range(len(data)):
            for word in data.at[i, 'text']:
                tf = words_by_cluster[data.at[i, 'cluster']][word] / len(data)
                idf = len(data) / words[word]
                tf_idf[data.at[i, 'cluster']][word] = tf * idf

        # On affiche les mots les plus importants pour chaque cluster
        for i in range(nb_clusters):
            tf_idf[i].pop('nan')
            print("Cluster", i)
            print(sorted(tf_idf[i], key=tf_idf[i].get, reverse=True)[:10])
    

    # preprocessing(csv_file_clean_sample, csv_file_processed_sample)
    # wordcloud(csv_file_processed_sample)

    csv_file_processed_sample_cluster = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_processed-SAMPLE_fake_clusters.csv"

    TF_IDF(csv_file_processed_sample_cluster, 3)
