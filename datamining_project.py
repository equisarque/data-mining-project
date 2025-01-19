# Datamining Project Joris FELZINES - Guillaume MANTZARIDES

#dependencies for displaying map
import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data2.csv"
csv_file_clean = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_clean.csv"
csv_file_clean_sample = "C:/Users/felzi/Desktop/INSA/4IF/S1/DataMining/flickr_data_clean-SAMPLE.csv"

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

# data = pd.read_table(csv_file_clean_sample, sep=",", low_memory=False)

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

# my_map = folium.Map(location=(45.75, 4.83))
# for i in range(len(data)):
#     folium.CircleMarker(location=[data.at[i,"lat"], data.at[i,"long"]], tooltip="Afficher titre", popup=data.at[i,"title"], radius = 3).add_to(my_map)
# my_map.save("map.html")



##############################
# Description des zones d'intérêt en utilisant des techniques de text mining
##############################

# Librairies for text mining
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
