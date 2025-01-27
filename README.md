# data-mining-project

# hypothèses : les colonnes d'upload des données et les trois dernières colonnes sont inutiles pour l'usage principale des données

# les datas semblent bonnes (non nulles) pour id, user, lat, lon

# pour regarder une photo flickr.com/photos/id_user/id_photo ou regarder le sujet

La tokenization est inefficace pour les tags et title car les mots sont souvent collés les uns aux autres, et il n'y a pas de ponctuation, verbes, etc.
La tokenization est donc inutile pour ce dataset en particulier.
# on a utilisé fourvière pour savoir quelle était le bon epsilon

# il faudrait essayer de faire un DBSCAN récursif pour couper les gros cluster



