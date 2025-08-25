import numpy as np
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity

# Funzione che data una canzone, genera una playlist di canzoni simili.
def playlist_song(name, artist, songs, n_songs=10):
    list_songs = find_similar(str(name), str(artist), songs, n_songs)

    if type(list_songs) != type(None):

        print(colored('Playlist basata sulla canzone "' + str(name) + '" di ' + str(artist), "green"))
        print()

        for i in np.arange(0, len(list_songs)):
            track_name = list_songs.track_name[i]
            artist_name = list_songs.artist_name[i]

            print(colored(str(track_name) + ' - ' + str(artist_name), "green"))

        print()

    return None


# Funzione che data una canzone, ne trova delle canzoni simili.
def find_similar(name, artist, songs, top_n=5):
    database = songs[songs.popularity > 0.5].reset_index(drop=True)
    indx_names = database[['track_name', 'artist_name', 'Cluster']]
    songs_train = database.drop(['track_name', 'artist_name', 'Cluster'], axis=1)

    song = find_song_database(str(name), str(artist), database)

    if type(song) != type(None):
        indx_song = song.index

        cos_dists = cosine_similarity(songs_train, songs_train)
        indx_names.loc[:, ['result']] = cos_dists[indx_song[0]]

        indx_names = indx_names.sort_values(by=['result'], ascending=False)

        return indx_names[1:top_n].reset_index(drop=True)

    else:
        print("Song not found")
        return None


# Funzione che cerca una canzone nel dataset.
def find_song_database(name, artist, songs):
    result = songs[(songs.artist_name == str(artist)) & (songs.track_name == str(name))]
    if len(result) == 0:
        return None
    return result.drop(['track_name', 'artist_name', 'Cluster'], axis=1)
