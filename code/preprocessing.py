import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None  # default='warn'


def preprocessing_for_classification(dataframe):
    # Per la feature "Key" convertiamo le 12 chiavi in un numeri, utilizzando l'indice.
    list_of_keys = dataframe['key'].unique()
    for i in range(len(list_of_keys)):
        dataframe.loc[dataframe['key'] == list_of_keys[i], 'key'] = i

    # Per la feature "Mode" convertiamo le Major in 1 e Minor in 0.
    dataframe.loc[dataframe["mode"] == 'Major', "mode"] = 1
    dataframe.loc[dataframe["mode"] == 'Minor', "mode"] = 0

    # Per la feature "time_signature" convertiamo i battiti in numeri, utilizzando l'indice.
    list_of_time_signatures = dataframe['time_signature'].unique()
    for i in range(len(list_of_time_signatures)):
        dataframe.loc[dataframe['time_signature'] == list_of_time_signatures[i], 'time_signature'] = i

    # Per la feature "popularity" la rendiamo una binaria, dove una canzone é popolare se ha
    # uno score maggiore o uguale a 75. Non é popolare altrimenti.
    dataframe.loc[dataframe['popularity'] < 75, 'popularity'] = 0
    dataframe.loc[dataframe['popularity'] >= 75, 'popularity'] = 1

    return dataframe


def preprocessing_for_clustering(data):
    # Rimuoviamo le feature "key" e "time_signature",
    # e trasferiamo track_name e artist_name in un'altra tabella.
    indx = data[['track_name', 'artist_name']]
    attributes = data.drop(['track_id', 'time_signature', 'track_name', 'artist_name', 'key'], axis=1)

    # Trasformiamo i valori genre e mode in valori binari,
    # aggiungendo ogni tipologia di genere alle feature, in modo
    # che ogni canzone abbia 1 al proprio genere.
    ordinal_encoder = OrdinalEncoder()
    object_cols = ['mode']
    attributes[object_cols] = ordinal_encoder.fit_transform(attributes[object_cols])

    attributes = pd.get_dummies(attributes)
    attributes.insert(loc=0, column='track_name', value=indx.track_name)
    attributes.insert(loc=1, column='artist_name', value=indx.artist_name)

    genres_names = ['genre_A Capella', 'genre_Alternative', 'genre_Anime', 'genre_Blues',
                    "genre_Children's Music", "genre_Children’s Music", 'genre_Classical',
                    'genre_Comedy', 'genre_Country', 'genre_Dance', 'genre_Electronic',
                    'genre_Folk', 'genre_Hip-Hop', 'genre_Indie', 'genre_Jazz',
                    'genre_Movie', 'genre_Opera', 'genre_Pop', 'genre_R&B', 'genre_Rap',
                    'genre_Reggae', 'genre_Reggaeton', 'genre_Rock', 'genre_Ska',
                    'genre_Soul', 'genre_Soundtrack', 'genre_World']

    genres = attributes.groupby(['track_name', 'artist_name'])[genres_names].sum()

    column_names = ['track_name', 'artist_name']
    for i in genres_names:
        column_names.append(i)

    genres.reset_index(inplace=True)
    genres.columns = column_names

    attributes = attributes.drop(genres_names, axis=1)

    atts_cols = attributes.drop(['track_name', 'artist_name'], axis=1).columns
    scaler = StandardScaler()
    attributes[atts_cols] = scaler.fit_transform(attributes[atts_cols])

    songs = pd.merge(genres, attributes, how='inner', on=['track_name', "artist_name"])
    songs = songs.drop_duplicates(['track_name', 'artist_name']).reset_index(drop=True)

    return data, songs
