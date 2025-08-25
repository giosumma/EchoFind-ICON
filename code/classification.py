import pandas as pd
import preprocessing
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Funzione principale per valutare tutti i classificatori
def various_classification():
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    features = ["acousticness", "danceability", "duration_ms", "energy", 
                "instrumentalness", "key", "liveness", "mode", "speechiness", 
                "tempo", "time_signature", "valence"]

    dataframe = preprocessing.preprocessing_for_classification(dataframe)

    # --- Divisione Training (80%) e Test set (20%) ---
    train_set = dataframe.sample(frac=0.8, random_state=420)
    test_set = dataframe.drop(train_set.index)

    X_train = train_set[features]
    y_train = train_set['popularity']
    X_test = test_set[features]
    y_test = test_set['popularity']

    # --- Divisione Training interno / Validation set (20% del training) ---
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=420
    )



    # --- Stampare dimensioni dei set ---
    print("\n--- Suddivisione dati ---")
    print(f"Training set: {len(X_train)} esempi")
    print(f"Validation set: {len(X_valid)} esempi")
    print(f"Test set: {len(X_test)} esempi")
    print("\nNota:")
    print("Training set: usato per addestrare il modello")
    print("Validation set: usato per ottimizzare e scegliere il modello")
    print("Test set: usato solo alla fine per valutare le performance finali\n")

    # Creazione dei modelli
    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (KNeighborsClassifier(), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree")
    ]

    # Valutazione di tutti i modelli
    for model, name in models:
        evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, name)

def stampa():
    dataframe = pd.read_csv('../dataset/spotify_features.csv')
    features = ["acousticness", "danceability", "duration_ms", "energy", 
                "instrumentalness", "key", "liveness", "mode", "speechiness", 
                "tempo", "time_signature", "valence"]

    dataframe = preprocessing.preprocessing_for_classification(dataframe)

    # --- Divisione Training (80%) e Test set (20%) ---
    train_set = dataframe.sample(frac=0.8, random_state=420)
    test_set = dataframe.drop(train_set.index)

    X_train = train_set[features]
    y_train = train_set['popularity']
    X_test = test_set[features]
    y_test = test_set['popularity']

    # --- Divisione Training interno / Validation set (20% del training) ---
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=420
    )

    # Creazione dei modelli
    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (KNeighborsClassifier(), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree")
    ]

    # Valutazione di tutti i modelli
    for model, name in models:
        evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, name)

    print(f"Training set: {len(X_train)} righe")
    print(f"Validation set: {len(X_valid)} righe")
    print(f"Test set: {len(X_test)} righe")

# Funzione generica per valutare i modelli su validation e test set
def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, name):
    model.fit(X_train, y_train)

    # --- Validation set ---
    y_val_pred = model.predict(X_valid)
    if hasattr(model, "predict_proba"):
        y_val_prob = model.predict_proba(X_valid)[:,1]
    else:
        y_val_prob = y_val_pred

    print(f"{name} - Validation Set:")
    print(f"Accuracy = {accuracy_score(y_valid, y_val_pred):.4f}, AUC = {roc_auc_score(y_valid, y_val_prob):.4f}")

    # --- Test set ---
    y_test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:,1]
    else:
        y_test_prob = y_test_pred

    print(f"{name} - Test Set:")
    print(f"Accuracy = {accuracy_score(y_test, y_test_pred):.4f}, AUC = {roc_auc_score(y_test, y_test_prob):.4f}")
    print("-"*50)

# Logistic Regression
def logistic_regression(X_train, y_train, X_valid, y_valid):
    LR_Model = LogisticRegression()
    LR_Model.fit(X_train, y_train)
    LR_Predict = LR_Model.predict(X_valid)
    LR_Accuracy = accuracy_score(y_valid, LR_Predict)
    print("Logistic Regression.")
    print("Accuracy: " + str(LR_Accuracy))

    LR_AUC = roc_auc_score(y_valid, LR_Predict)
    print("AUC: " + str(LR_AUC))

# Random Forest
def random_forest_classifier(X_train, y_train, X_valid, y_valid):
    RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
    print("Random Forest Classifier.")
    print("Accuracy: " + str(RFC_Accuracy))

    RFC_AUC = roc_auc_score(y_valid, RFC_Predict)
    print("AUC: " + str(RFC_AUC))

# KNN
def k_neighbors_classifier(X_train, y_train, X_valid, y_valid):
    KNN_Model = KNeighborsClassifier()
    KNN_Model.fit(X_train, y_train)
    KNN_Predict = KNN_Model.predict(X_valid)
    KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    print("K Neighbor Classifier.")
    print("Accuracy: " + str(KNN_Accuracy))

    KNN_AUC = roc_auc_score(y_valid, KNN_Predict)
    print("AUC: " + str(KNN_AUC))

# Decision Tree
def decision_tree_classifier(X_train, y_train, X_valid, y_valid):
    DT_Model = DecisionTreeClassifier()
    DT_Model.fit(X_train, y_train)
    DT_Predict = DT_Model.predict(X_valid)
    DT_Accuracy = accuracy_score(y_valid, DT_Predict)
    print("Decision Tree Classifier.")
    print("Accuracy: " + str(DT_Accuracy))

    DT_AUC = roc_auc_score(y_valid, DT_Predict)
    print("AUC: " + str(DT_AUC))





def cross_validation_evaluation():
    # --- Caricamento dataset ---
    dataframe = pd.read_csv('../dataset/spotify_features.csv')

    # --- Preprocessing ---
    # Feature "key"
    list_of_keys = dataframe['key'].unique()
    for i in range(len(list_of_keys)):
        dataframe.loc[dataframe['key'] == list_of_keys[i], 'key'] = i

    # Feature "mode"
    dataframe.loc[dataframe["mode"] == 'Major', "mode"] = 1
    dataframe.loc[dataframe["mode"] == 'Minor', "mode"] = 0

    # Feature "time_signature"
    list_of_time_signatures = dataframe['time_signature'].unique()
    for i in range(len(list_of_time_signatures)):
        dataframe.loc[dataframe['time_signature'] == list_of_time_signatures[i], 'time_signature'] = i

    # --- Binarizzazione target sul 25° percentile ---
    threshold = dataframe['popularity'].quantile(0.25)
    dataframe['popularity'] = (dataframe['popularity'] >= threshold).astype(int)
    print("Distribuzione classi dopo binarizzazione:")
    print(dataframe['popularity'].value_counts())

    # --- Feature e target ---
    features = ["acousticness", "danceability", "duration_ms", "energy", 
                "instrumentalness", "key", "liveness", "mode", "speechiness", 
                "tempo", "time_signature", "valence"]
    X = dataframe[features]
    y = dataframe['popularity']

    # --- Standardizzazione ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Definizione modelli ---
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    # --- Stratified K-Fold ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # --- Valutazione modelli ---
    for name, model in models.items():
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        results[name] = {'accuracy': acc_scores, 'auc': auc_scores}
        print(f"{name}: Accuracy media = {acc_scores.mean():.4f}, Deviazione standard = {acc_scores.std():.4f}")
        print(f"{name}: AUC media = {auc_scores.mean():.4f}, Deviazione standard = {auc_scores.std():.4f}")
        print("-"*50)

    # --- Tabella riepilogativa ---
    cv_results = pd.DataFrame({
        'Modello': list(results.keys()),
        'Accuracy Media': [results[m]['accuracy'].mean() for m in results],
        'Deviazione Std Accuracy': [results[m]['accuracy'].std() for m in results],
        'AUC Media': [results[m]['auc'].mean() for m in results],
        'Deviazione Std AUC': [results[m]['auc'].std() for m in results]
    })
    print("\nTabella riassuntiva cross-validation:")
    print(cv_results)

    # --- Grafico comparativo Accuracy ---
    plt.figure(figsize=(8,5))
    plt.bar(cv_results['Modello'], cv_results['Accuracy Media'], 
            yerr=cv_results['Deviazione Std Accuracy'], capsize=5, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Confronto Accuracy - 5-Fold Cross Validation')
    plt.ylim(0,1)
    plt.show()

    # --- Grafico comparativo AUC ---
    plt.figure(figsize=(8,5))
    plt.bar(cv_results['Modello'], cv_results['AUC Media'], 
            yerr=cv_results['Deviazione Std AUC'], capsize=5, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    plt.ylabel('AUC')
    plt.title('Confronto AUC - 5-Fold Cross Validation')
    plt.ylim(0,1)
    plt.show()
