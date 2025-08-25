import pandas as pd
import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# Random Forest Classifier
def rfc_prediction(name_song, name_artist):
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    dataframe = preprocessing.preprocessing_for_classification(dataframe)
    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                "mode", "speechiness", "tempo", "time_signature", "valence"]

    row = dataframe.loc[(dataframe['track_name'] == name_song) & (dataframe['artist_name'] == name_artist)]
    row = row.drop(['track_id', 'track_name', 'artist_name', 'loudness', 'genre', 'popularity'], axis=1)

    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['popularity']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train, y_train)
    rfc_predict = rfc_model.predict(row)
    return rfc_predict.all()

# Logistic Regression
def lr_prediction(name_song, name_artist):
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    dataframe = preprocessing.preprocessing_for_classification(dataframe)
    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                "mode", "speechiness", "tempo", "time_signature", "valence"]

    row = dataframe.loc[(dataframe['track_name'] == name_song) & (dataframe['artist_name'] == name_artist)]
    row = row.drop(['track_id', 'track_name', 'artist_name', 'loudness', 'genre', 'popularity'], axis=1)

    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['popularity']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_predict = lr_model.predict(row)
    return lr_predict.all()

# KNN
def knn_prediction(name_song, name_artist):
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    dataframe = preprocessing.preprocessing_for_classification(dataframe)
    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                "mode", "speechiness", "tempo", "time_signature", "valence"]

    row = dataframe.loc[(dataframe['track_name'] == name_song) & (dataframe['artist_name'] == name_artist)]
    row = row.drop(['track_id', 'track_name', 'artist_name', 'loudness', 'genre', 'popularity'], axis=1)

    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['popularity']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_predict = knn_model.predict(row)
    return knn_predict.all()

# Decision Tree
def dt_prediction(name_song, name_artist):
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    dataframe = preprocessing.preprocessing_for_classification(dataframe)
    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                "mode", "speechiness", "tempo", "time_signature", "valence"]

    row = dataframe.loc[(dataframe['track_name'] == name_song) & (dataframe['artist_name'] == name_artist)]
    row = row.drop(['track_id', 'track_name', 'artist_name', 'loudness', 'genre', 'popularity'], axis=1)

    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['popularity']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_predict = dt_model.predict(row)
    return dt_predict.all()

