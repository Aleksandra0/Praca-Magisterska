#Biblioteki potrzebne w przetwarzaniu i wizualizacji
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

#Biblioteki potrzebne dla metod uczenia maszynowego
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef, \
    roc_auc_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

#Biblioteki potrzebne dla undersamplingu
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, NeighbourhoodCleaningRule, NearMiss, EditedNearestNeighbours, CondensedNearestNeighbour

#Biblioteki potrzebne dla oversamplingu
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE

#Biblioteki potrzebne dla metod hybrydowych
from imblearn.combine import SMOTETomek, SMOTEENN

#Biblioteki potrzebne dla sieci neuronowej
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


random.seed(402974)
np.random.seed(402974)


print("Wczytanie danych:")

#Zbiór danych: Loan Data Evaluation
ld_evaluation = pd.read_csv('dane/loan_data_evaluation.csv')

#Usunięcie nieopisanej zmiennej
ld_evaluation = ld_evaluation.drop('not.fully.paid', axis=1)

print(ld_evaluation.head())
print(ld_evaluation.info())
print(ld_evaluation.tail())

print("\nCzy w danych występują wartości NA lub null?")
print(ld_evaluation.isna().any().any())
print(ld_evaluation.isnull().any().any())

#Zamiana wartości 0 i 1 dla zmiennej credit.policy
ld_evaluation['credit.policy'] = ld_evaluation['credit.policy'] ^ 1

print("\nStopień niezbalansowania zbioru danych")
print(sum(ld_evaluation['credit.policy'])/len(ld_evaluation['credit.policy']))

#WSTĘPNA ANALIZA DANYCH

#Podstawowowe statystyki
print((ld_evaluation.iloc[:, 0:5]).describe())
print((ld_evaluation.iloc[:, 5:9]).describe())
print((ld_evaluation.iloc[:, 10:13]).describe())
print(ld_evaluation["purpose"].value_counts())

#Zmienne liczbowe
ld_ev_il = ld_evaluation.drop('purpose', axis=1)

#Współczynniki zmienności
for col_name in ld_ev_il.columns:
    mean_ld_ev_il = np.mean(ld_ev_il[col_name])
    std_ld_ev_il= np.std(ld_ev_il[col_name])

    cv = (mean_ld_ev_il/std_ld_ev_il) * 100
    print("WSP Zmiennosci:\n" + col_name + ": ", cv)

#Zbadanie zależności między zmiennymi

"""correlation_matrix = ld_ev_il.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()"""

#Wykresy pudełkowe w relacji do zmiennej credit.policy

"""for col_name in ld_ev_il.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='credit.policy', y=col_name, data=ld_ev_il)
    plt.xlabel('credit.policy')
    plt.ylabel(col_name)
    plt.show()

cd = ld_evaluation.groupby(['purpose', 'credit.policy']).size().unstack(fill_value=0)
cd.plot(kind='bar', stacked=True)
plt.xlabel('purpose')
plt.ylabel('Count')
plt.legend(title='credit.policy')
plt.show()"""

#Usunięcie nieistotnych zmiennych
ld_evaluation = ld_evaluation.drop('log.annual.inc', axis=1)
ld_evaluation = ld_evaluation.drop('delinq.2yrs', axis=1)
ld_evaluation = ld_evaluation.drop('pub.rec', axis=1)


#Utworzenie zmiennych 0-1 dla zmiennej kategorycznej
purposes = pd.get_dummies(pd.Series(ld_evaluation["purpose"]), prefix='purpose')

def code_tf_01(status):
    if status == True:
        return 1
    else:
        return 0

for col_name in purposes.columns:
    purposes[col_name] = purposes[col_name].apply(code_tf_01)

#Usunięcie kolumny purpose
ld_evaluation = ld_evaluation.drop('purpose', axis=1)

ld_evaluation  = pd.concat([ld_evaluation, purposes], axis=1)
print(ld_evaluation.head(25))
print(ld_evaluation.info())

print("\nCzy w danych występują wartości NA lub null?")
print(ld_evaluation.isna().any().any())
print(ld_evaluation.isnull().any().any())

'''fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

axs = axs.flatten()
for i, column in enumerate(ld_evaluation.columns[9:16]):
    if i < len(axs):
        axs[i].hist(ld_evaluation[column])
        axs[i].set_title(column)
        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Frequency')
for ax in axs[len(ld_evaluation.columns):]:
    ax.axis('off')
plt.tight_layout()
plt.show()'''

#ZAKOŃCZENIE WSTĘPNEJ ANALIZY DANYCH

#-----------------------------------------------------------------------------------------------------------------------

#METODY UCZENIA MASZYNOWEGO

names = ['Spłata', 'Brak spłaty']

#Przygotowanie Danych

X = ld_evaluation.drop(columns=['credit.policy']) #zmienne objaśniające
y = ld_evaluation['credit.policy'] #zmienna objaśniana

def split_and_normalize_data(X, y):
    #Podział na zbiór uczący i zbiór testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=402974, stratify=y)

    #Normalizacja
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_and_normalize_data(X, y)

#Metoda - wypisywanie miar dla poszczególnych modeli
def miary_jakosci(y_test, y_pred, y_prob):
    # Obliczanie miar
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    sensitivity = recall_score(y_test, y_pred)
    fs = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f'Dokładność: {acc:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Czułość: {sensitivity:.2f}')
    print(f'F-Score: {fs:.2f}')
    print(f'Wsp. Korelacji Matthews: {mcc:.2f}')

    print("Macierz pomyłek:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

#1) METODA KNN

def knn_dataset(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test) #Predykcja na zbiorze testowym
    y_prob = knn.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    #Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

    '''# Wizualizacja macierzy pomyłek
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.xlabel('Wartosć Prognozowana')
    plt.ylabel('Wartość Rezczywista')
    plt.title('Macierz pomyłek')
    plt.show()'''

#2) REGRESJA LOGISTYCZNA

def regresja_logistyczna_dataset(X_train, X_test, y_train, y_test, maxiter):
    #Tworzenie modelu - regresja logistyczna
    model_LR = LogisticRegression(max_iter=maxiter)
    model_LR.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_LR.predict(X_test)
    y_prob = model_LR.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#3) DRZEWA KLASYFIKACYJNE

def drzewa_klasyfikacyjne_dataset(X_train, X_test, y_train, y_test):

    #Tworzenie modelu - drzewa klasyfikacyjne
    model_DTC = DecisionTreeClassifier(random_state=402974)
    model_DTC.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_DTC.predict(X_test)
    y_prob = model_DTC.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#4) METODA SVM

def svm_dataset(X_train, X_test, y_train, y_test):

    #Tworzenie modelu - metoda SVM

    model_svm = svm.SVC(kernel='linear', probability=True) # Inicjalizacja klasyfikatora SVM
    model_svm.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_svm.predict(X_test)
    y_prob = model_svm.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#5) Bagging + Drzewa Decyzyjne
def bagging_DC_dataset(X_train, X_test, y_train, y_test, n_classifiers):

    #Tworzenie modelu

    # Inicjalizacja klasyfikatora bazowego - drzewa decyzyjnego
    base_DTC = DecisionTreeClassifier()

    #Inicjalizacja klasyfikatora dla Baggingu
    model_bagging = BaggingClassifier(base_DTC, n_estimators=n_classifiers, random_state=402974)
    model_bagging.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_bagging.predict(X_test)
    y_prob = model_bagging.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#6) Boosting + Drzewa Decyzyjne
def boosting_DC_dataset(X_train, X_test, y_train, y_test, n_classifiers):

    #Tworzenie modelu

    # Inicjalizacja klasyfikatora bazowego - drzewa decyzyjnego
    base_DTC = DecisionTreeClassifier()

    # Inicjalizacja klasyfikatora AdaBoost
    model_adaboost = AdaBoostClassifier(estimator=base_DTC, n_estimators=n_classifiers, random_state=402974)
    model_adaboost.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_adaboost.predict(X_test)
    y_prob = model_adaboost.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#7) Bagging + SVM
def bagging_SVM_dataset(X_train, X_test, y_train, y_test, n_classifiers):

    #Tworzenie modelu

    # Inicjalizacja klasyfikatora bazowego - SVM
    base_SVM = svm.SVC(kernel='linear', probability=True)

    #Inicjalizacja klasyfikatora dla Baggingu
    model_bagging = BaggingClassifier(base_SVM, n_estimators=n_classifiers, random_state=402974)
    model_bagging.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_bagging.predict(X_test)
    y_prob = model_bagging.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#8) Boosting + SVM
def boosting_SVM_dataset(X_train, X_test, y_train, y_test, n_classifiers):

    #Tworzenie modelu

    # Inicjalizacja klasyfikatora bazowego - SVM
    base_SVM = svm.SVC(kernel='linear', probability=True)

    # Inicjalizacja klasyfikatora AdaBoost
    model_adaboost = AdaBoostClassifier(estimator=base_SVM, n_estimators=n_classifiers, random_state=402974)
    model_adaboost.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model_adaboost.predict(X_test)
    y_prob = model_adaboost.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

    # Ocena modelu
    miary_jakosci(y_test, y_pred, y_prob)

#Wywołanie metod

def learn_on_datasets(X_train, X_test, y_train, y_test, k, maxiter, n_classifiers_dt, n_classifiers_svm):
    print("\n------------------------------------\nMetoda KNN")
    knn_dataset(X_train, X_test, y_train, y_test, k)

    print("\n------------------------------------\nRegresja Logistyczna")
    knn_dataset(X_train, X_test, y_train, y_test, maxiter)

    print("\n------------------------------------\nDrzewa klasyfikacyjne")
    drzewa_klasyfikacyjne_dataset(X_train, X_test, y_train, y_test)

    print("\n------------------------------------\nMetoda SVM")
    svm_dataset(X_train, X_test, y_train, y_test)

    print("\n------------------------------------\nMetoda Bagging + Drzewa Decyzyjne")
    bagging_DC_dataset(X_train, X_test, y_train, y_test, n_classifiers_dt)

    print("\n------------------------------------\nMetoda Boosting + Drzewa Decyzyjne")
    boosting_DC_dataset(X_train, X_test, y_train, y_test, n_classifiers_dt)

    #print("\n------------------------------------\nMetoda Bagging + SVM")
    #bagging_SVM_dataset(X_train, X_test, y_train, y_test, n_classifiers_svm)

    #print("\n------------------------------------\nMetoda Boosting + SVM")
    #boosting_SVM_dataset(X_train, X_test, y_train, y_test, n_classifiers_svm)

k = 5  # parametr określający liczbę sąsiadów
maxiter = 50 #maksymalna liczba iteracji dla regresji logistycznej
n_classifiers_dt = 20 #Liczba drzew decyzyjnych tworzących zespół
n_classifiers_svm = 10 #Liczba modeli SVM tworzących zespół

#Metody uczenia maszynowego - bez modyfikacji
#learn_on_datasets(X_train, X_test, y_train, y_test, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#MODYFIKACJA ZBIORU TAK ABY BYŁ SILNIE NIEZBALANSOWANY

#Wyfiltrowanie wierszy z wartością 1 dla zmiennej objaśnianej
ones = ld_evaluation[ld_evaluation['credit.policy'] == 1]
zeros = ld_evaluation[ld_evaluation['credit.policy'] != 1]

#Losowe usunięcie 75% wierszy z wartością 1
np.random.seed(402974)
remove = ones.sample(frac=0.85)
left = ones.drop(remove.index)

#Ponowne połączenie zbioru danych
ld_evaluation_modified = pd.concat([left, zeros]).sort_index().reset_index(drop=True)

print("\nStopień niezbalansowania zbioru danych")
print(sum(ld_evaluation_modified['credit.policy'])/len(ld_evaluation_modified['credit.policy']))

ld_evaluation = ld_evaluation_modified
X = ld_evaluation.drop(columns=['credit.policy']) #zmienne objaśniające
y = ld_evaluation['credit.policy'] #zmienna objaśniana
X_train, X_test, y_train, y_test = split_and_normalize_data(X, y)

X_train_original, X_test_original, y_train_original, y_test_original = split_and_normalize_data(X, y)

#Metody uczenia maszynowego - dla zmodyfikowanego zbioru
learn_on_datasets(X_train, X_test, y_train, y_test, k, maxiter, n_classifiers_dt, n_classifiers_svm)


#-----------------------------------------------------------------------------------------------------------------------

#ZASTOSOWANIE METOD UNDERSAMPLINGU

#1) Random Undersampling

rus = RandomUnderSampler(random_state=402974)
X_rus, y_rus = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_rus, y_rus)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA RANDOM UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#2) Cluster Centroid Undersampling

cc = ClusterCentroids(random_state=402974)
X_cc, y_cc = cc.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_cc, y_cc)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA CLUSTER CENNTROID UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#3) Tomek Links Undersampling

tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_tl, y_tl)

#Metody uczenia maszynowego
print("\nWYNIKI DLA TOMEK LINKS UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#4) Neighbourhood Cleaning Rule

ncr = NeighbourhoodCleaningRule()
X_ncr, y_ncr = ncr.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_ncr, y_ncr)

#Metody uczenia maszynowego
print("\nWYNIKI DLA NCR (Neighbourhood Cleaning Rule):")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#5) Near Miss Undersampling (Wersja 1 - odległość Euklidesowa)

nm = NearMiss(version=1)
X_nm, y_nm = nm.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_nm, y_nm)

#Metody uczenia maszynowego
print("\nWYNIKI DLA NEARMISS-1 UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#6) Near Miss Undersampling (Wersja 2 - odległość Euklidesowa + różnice w etykietach klas)

nm2 = NearMiss(version=2)
X_nm2, y_nm2 = nm2.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_nm2, y_nm2)

#Metody uczenia maszynowego
print("\nWYNIKI DLA NEARMISS-2 UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#7) Edited Nearest Neighbours Undersampling

enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_enn, y_enn)

#Metody uczenia maszynowego
print("\nWYNIKI DLA EDITED NESREST NEIGHBOURS UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#8) Condensed Nearest Neighbours Undersampling

cnn = CondensedNearestNeighbour(random_state=402974)
X_cnn, y_cnn = cnn.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_cnn, y_cnn)

#Metody uczenia maszynowego
print("\nWYNIKI DLA CONDENSED NESREST NEIGHBOURS UNDERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#-----------------------------------------------------------------------------------------------------------------------

#ZASTOSOWANIE METOD OVERSAMPLINGU

#1) Random Oversampling

ros = RandomOverSampler(random_state=402974)
X_ros, y_ros = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_ros, y_ros)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA RANDOM OVERSAMPLING:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#2) SMOTE

smote = SMOTE(random_state=402974)
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_smote, y_smote)

#Metody uczenia maszynowego
print("\nWYNIKI DLA SMOTE:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#3) Borderline-SMOTE

borderline_smote = BorderlineSMOTE(random_state=402974)
X_borderline_smote, y_borderline_smote = borderline_smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_borderline_smote, y_borderline_smote)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA Bordeline-SMOTE:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#4) SVM-SMOTE

svm_smote = SVMSMOTE(random_state=402974)
X_svm_smote, y_svm_smote = svm_smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_svm_smote, y_svm_smote)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA SVM-SMOTE:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#5) ROSE

rose = RandomOverSampler(random_state=402974, shrinkage=0.5) #ROSE - ROS z zastoswanie smoothed bootstrap
X_rose, y_rose = rose.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_rose, y_rose)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA ROSE:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#6) ADASYN

adasyn = ADASYN(random_state=402974)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_adasyn, y_adasyn)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA ADASYN:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#-----------------------------------------------------------------------------------------------------------------------

#ZASTOSOWANIE METOD HYBRYDOWYCH

#1) SMOTE-TOMEK

smote_tomek = SMOTETomek(random_state=402974)
X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_smote_tomek, y_smote_tomek)

#Metodu uczenia maszynowego
print("\nWYNIKI DLA SMOTE-TOMEK:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#2) SMOTE-ENN

smote_enn = SMOTEENN(random_state=402974)
X_smote_enn, y_smote_enn = smote_enn.fit_resample(X, y)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_smote_enn, y_smote_enn)

#Metody uczenia maszynowego
print("\nWYNIKI DLA SMOTE-ENN:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#3) Random Oversampling + NCR

# Random oversampling
ros = RandomOverSampler(random_state=402974)
X_ros, y_ros = ros.fit_resample(X, y)

# Neighbourhood Cleaning Rule
ncl = NeighbourhoodCleaningRule()
X_ros_ncl, y_ros_ncl = ncl.fit_resample(X_ros, y_ros)

X_train, X_test, y_train, y_test = split_and_normalize_data(X_ros_ncl, y_ros_ncl)

#Metody uczenia maszynowego
print("\nWYNIKI DLA ROS + NCL:")
learn_on_datasets(X_train, X_test_original, y_train, y_test_original, k, maxiter, n_classifiers_dt, n_classifiers_svm)

#-----------------------------------------------------------------------------------------------------------------------

#MODYFIKACJA WAG KLAS

print('----- MODYFIKACJA WAG KLAS -----')

X_train, X_test, y_train, y_test = split_and_normalize_data(X, y)

class_weight_fixed = {0: 1, 1: 5}
class_weights_computed = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_computed = dict(enumerate(class_weights_computed))

# Tworzenie modelu z wagami klas - regresja logistyczna
model_LR = LogisticRegression(max_iter=maxiter, class_weight=class_weights_computed)
model_LR.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model_LR.predict(X_test)
y_prob = model_LR.predict_proba(X_test)[:, 1]  # Prawdopodobieństwa dla klasy pozytywnej

print('\nRegresja logistyczna:')
# Ocena modelu
miary_jakosci(y_test, y_pred, y_prob)

#Tworzenie modelu z wagami klas - drzewa klasyfikacyjne
model_DTC = DecisionTreeClassifier(random_state=402974, class_weight=class_weights_computed)
model_DTC.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model_DTC.predict(X_test)
y_prob = model_DTC.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

print('\nDrzewa Decyzyjne:')
# Ocena modelu
miary_jakosci(y_test, y_pred, y_prob)

#Tworzenie modelu z wagami klas - metoda SVM

model_svm = svm.SVC(kernel='linear', probability=True, class_weight=class_weights_computed)  # Inicjalizacja klasyfikatora SVM
model_svm.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model_svm.predict(X_test)
y_prob = model_svm.predict_proba(X_test)[:, 1]  # Prawdopodobieństwa dla klasy pozytywnej


print('\nMetoda SVM:')
# Ocena modelu
miary_jakosci(y_test, y_pred, y_prob)

#Bagging z wagami klas
# Inicjalizacja klasyfikatora bazowego - drzewa decyzyjnego
base_DTC = DecisionTreeClassifier(class_weight=class_weights_computed)

# Inicjalizacja klasyfikatora dla Baggingu
model_bagging = BaggingClassifier(base_DTC, n_estimators=n_classifiers_dt, random_state=402974)
model_bagging.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model_bagging.predict(X_test)
y_prob = model_bagging.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

print('\nBagging:')
# Ocena modelu
miary_jakosci(y_test, y_pred, y_prob)

#Boosting z wagami klas
# Inicjalizacja klasyfikatora bazowego - drzewa decyzyjnego
base_DTC = DecisionTreeClassifier(class_weight=class_weights_computed)

# Inicjalizacja klasyfikatora AdaBoost
model_adaboost = AdaBoostClassifier(estimator=base_DTC, n_estimators=n_classifiers_dt, random_state=402974)
model_adaboost.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model_adaboost.predict(X_test)
y_prob = model_adaboost.predict_proba(X_test)[:, 1]  #Prawdopodobieństwa dla klasy pozytywnej

print('\nBoosting:')
# Ocena modelu
miary_jakosci(y_test, y_pred, y_prob)

#-----------------------------------------------------------------------------------------------------------------------

#SIEĆ NEURONOWA

print('----- SIEĆ NEURONOWA -----')

#Podzielenie i przetasowanie zbioru danych: zbior treningowy, walidacyjny, testowy
train_data, test_data = train_test_split(ld_evaluation, test_size=0.2)

#Stworzenie tablic typu np.array
train_y = np.array(train_data.pop('credit.policy'))
test_y = np.array(test_data.pop('credit.policy'))

train_X = np.array(train_data)
test_X = np.array(test_data)

#Normalizacja danych
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

#Zdefiniowanie metryk
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.Recall(name='recall')
]

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10, mode='max', restore_best_weights=True)

# Obliczanie wag klas
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
class_weights_dict = dict(enumerate(class_weights))
#class_w = {0: 1., 1: 5.}

print("Wagi klas:", class_weights_dict)

# Definiowanie modelu
model = Sequential()
model.add(Dense(32, input_shape=(train_X.shape[1],), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #Wyjście binarne
optimizer = Adam(learning_rate=0.001) #Optimizer Adam

# Kompilacja modelu
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

# Trening modelu
history = model.fit(train_X, train_y, epochs=50, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, callbacks=[early_stopping])

# Ocena modelu
accuracy, auc, loss, recall = model.evaluate(test_X, test_y)
print(model.evaluate(test_X, test_y))
print("Test loss:", loss)
print("Test recall:", recall)