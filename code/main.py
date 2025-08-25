import pandas as pd
import preprocessing
import graphics
import clustering
import classification
import functions
import prediction
import os

def main():
    graphics.print_logo()
    bol = False
    while True:
        if bol:
            graphics.print_menu()

        else:
            graphics.print_welcome_menu()
        response = input()
        # Creazione della playlist in base a nome della canzone e nome dell'artista
        if response == '1':
            # richiesta nome canzone
            graphics.song_request()
            track_name = input()
            # richiesta nome artista
            graphics.artist_request()
            artist_name = input()
            # numero di canzoni da aggiungere alla playlist
            graphics.number_request()
            n_song = int(input())

            data = pd.read_csv('../dataset/spotify_features.csv')
            # Fase di preprocessing
            data_frame, songs = preprocessing.preprocessing_for_clustering(data)
            # Fase di clustering
            df = clustering.clustering(data_frame, songs)
            # Creazione della playlist
            functions.playlist_song(track_name, artist_name, songs, n_song + 1)
            print("\n")
            os.system("pause")
            bol = True
            print("\n")

        # Prediction della popolarità di una canzone
        elif response == '2':
            # richiesta nome canzone
            graphics.song_request_for_prediction()
            name = input()
            # richiesta nome artista
            graphics.artist_request()
            artist = input()

            data = pd.read_csv('../dataset/spotify_features.csv')
            # Selezioniamo la riga della canzone selezionata dall'utente
            row = data.loc[(data['track_name'] == name) & (data['artist_name'] == artist)]
            # Controllo dell'esistenza della canzone nel dataset
            if row.empty:
                graphics.no_song_matched()
            else:
                graphics.song_mathced()
                # Scelta del classificatore
                graphics.choose_classifier()
                scelta = int(input())
                match scelta:
                    case 1:
                        if prediction.rfc_prediction(name, artist) == 0:
                            graphics.song_not_popular()
                        else:
                            graphics.song_is_popular()
                    case 2:
                        if prediction.knn_prediction(name, artist) == 0:
                            graphics.song_not_popular()
                        else:
                            graphics.song_is_popular()
                    case 3:
                        if prediction.dt_prediction(name, artist) == 0:
                            graphics.song_not_popular()
                        else:
                            graphics.song_is_popular()
                    case 4:
                        if prediction.lr_prediction(name, artist) == 0:
                            graphics.song_not_popular()
                        else:
                            graphics.song_is_popular()

            print("\n")
            os.system("pause")
            bol = True
            print("\n")

        # Uscita dal sistema
        elif response == '3':
            graphics.print_goodbye()
            break

        elif response == '4':
            #print("Esecuzione di tutti i classificatori sul dataset Spotify...")
            
            #classification.various_classification()

            #classification.stampa()
            
            classification.cross_validation_evaluation()
            print("\n")
            os.system("pause")
            bol = True
            print("\n")

if __name__ == '__main__':
    main()
