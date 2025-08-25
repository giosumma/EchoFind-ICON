from termcolor import colored


def print_logo():
    print(colored("\nEchoFind: Scopri l’eco dei tuoi gusti musicali\n", "green"))


def print_welcome_menu():
    print(colored("Ciao, benvenuto nel sistema di raccomandazione di Playlist basato sulle canzoni di Spotify!\n"
                  + "Vuoi che ti suggerisca una playlist? - Premi 1\n" \
                  + "Vuoi sapere se una canzone è popolare? - Premi 2\n" \
                  + "Vuoi uscire? - Premi 3", "green"))


def print_menu():
    print(colored("Vuoi che ti suggerisca una playlist? - Premi 1\n" \
                  + "Vuoi sapere se una canzone è popolare? - Premi 2\n" \
                  + "Vuoi uscire? - Premi 3", "green"))


def print_goodbye():
    print(colored("\nArrivederci!", "green"))


def song_request():
    print(colored("\nAdesso dovrai suggerirmi su quale canzone basare la tua playlist!\n" \
                  + "Qual'é il nome di una traccia che ti piace?", "green"))


def song_request_for_prediction():
    print(colored("\nAdesso dovrai suggerirmi la canzone su cui predire la popolaritá!\n" \
                  + "Qual'é il nome della traccia?", "green"))


def artist_request():
    print(colored("\nAdesso dimmi il nome dell'artista che ha scritto la traccia.", "green"))


def number_request():
    print(colored("\nQuante canzoni vuoi inserire nella tua playlist?", "green"))


def no_song_matched():
    print(colored("\nNessuna canzone trovata nel dataset!", "green"))


def song_mathced():
    print(colored("\nCanzone trovata nel dataset!", "green"))


def choose_classifier():
    print(colored("\nQuale classificatore vuoi utilizzare?\n"
                  + "Random Forest Classifier - Premi 1\n" \
                  + "K-Nearest Neighbors Classifier - Premi 2\n" \
                  + "Decision Tree Classifier - Premi 3\n" \
                  + "Logistic Regression - Premi 4\n", "green"))


def song_is_popular():
    print(colored("\nLa canzone é popolare!", "green"))


def song_not_popular():
    print(colored("\nLa canzone non é popolare!", "green"))
