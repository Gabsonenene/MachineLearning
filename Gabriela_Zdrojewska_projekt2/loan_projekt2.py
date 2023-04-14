import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# ustawiamy katalogi pracy
import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"loan_dataset")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"dane")
os.makedirs(KATALOG_DANYCH, exist_ok=True)



def load_loan_data(loan_path=KATALOG_DANYCH):
    csv_path = os.path.join(loan_path, "loan_data_set.csv")
    return pd.read_csv(csv_path)
loan = load_loan_data()

print('\n Początek bazy \n',loan.head())
#zestaw atrybutow bazy
print('\n Nazwy kolumn: \n', loan.columns)
#ogolna informacja o zawartosci bazy : zwroc uwage na kompletnosc informacji
print('\n informacja ogolna\n')
loan.info()
# ogólna informacja kompletnoci danych
print('\nBRAK DANYCH  \n', loan.isnull().sum())
print('\nDuplikaty  \n',loan.duplicated().sum())


print('\n Plec i ich ilosc \n',loan["Gender"].value_counts())
print('\nEdukacja i ich ilosc\n',loan["Education"].value_counts())

statystyka = loan.describe()
print('\nstatystyka opisowa\n', statystyka)

for item in statystyka:
    print (item, '\tmean =  %.3f'%statystyka[item][1], '\tmin max =  %.3f,  %.3f'%(statystyka[item][3],statystyka[item][7])   )
    
print('\nIlosc osbo, ktore dostaly lub nie dostaly pozyczki\n',loan["Loan_Status"].value_counts())
#loan['Loan_Status'].value_counts().plot.bar(title='loan_status')
#dano pozyczke dla 422 osob z 614

#loan['Gender'].value_counts().plot.bar(title='gender')
#plt.show()
plt.figure(figsize=(5,5))
s=sns.countplot(x="Gender", data=loan)

plt.figure(figsize=(5,5))
s=sns.countplot(x="Loan_Status", data=loan)
    
plt.figure(figsize=(5,5))
s=sns.countplot(x="Credit_History", data=loan)

plt.figure(figsize=(5,5))
s=sns.countplot(x="Education", data=loan)

plt.figure(figsize=(5,5))
s=sns.countplot(x="Property_Area", data=loan)

loan.hist(bins=50, figsize=(9,6))
plt.tight_layout()
plt.title("attribute_histogram_plots")
plt.show() # sprawdzić jak się plik zapisal


corr_matrix = loan.corr()  #obejrzyj macierz
#korelacje z mediana wartosci domu
print('\nKorelacje z LoanAmount') 
print( corr_matrix["LoanAmount"].sort_values(ascending=False))


plt.figure(figsize=(12,4))
sns.heatmap(corr_matrix,cmap="Blues", annot = True)

fig, (ax_1, ax_2) = plt.subplots(1,2)
loan['CoapplicantIncome'].hist(ax=ax_1, color='red', label='CoapplicantIncome', bins=50)
loan['CoapplicantIncome'].hist(ax=ax_2, color='green', label='CoapplicantIncome', bins=10)
plt.suptitle('CoapplicantIncome in distints bins')
plt.tight_layout()
plt.show()

fig, (ax_1, ax_2) = plt.subplots(1,2)
loan['ApplicantIncome'].hist(ax=ax_1, color='red', label='Applicant income', bins=50)
loan['ApplicantIncome'].hist(ax=ax_2, color='green', label='applicnat income', bins=10)
plt.suptitle('applicant income in distints bins')
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
sns.countplot(x="Married", hue="Loan_Status", data=loan)

plt.figure(figsize=(5,5))
sns.countplot(x="Gender", hue="Loan_Status", data=loan)

plt.figure(figsize=(5,5))
sns.countplot(x="Dependents", hue="Loan_Status", data=loan)

plt.figure(figsize=(5,5))
sns.countplot(x="Credit_History", hue="Loan_Status", data=loan)
plt.show()


"""
Uzupelniam none w string zeby moc zamienic na int
"""

loan['Gender'].fillna(loan['Gender'].mode()[0], inplace=True)
loan['Married'].fillna(loan['Married'].mode()[0], inplace=True)
loan['Dependents'].fillna(loan['Dependents'].mode()[0], inplace=True)
loan['Self_Employed'].fillna(loan['Self_Employed'].mode()[0], inplace=True)
#loan['Gender'] = loan['Gender'].apply(lambda x: x.fillna(x.value_counts().index[0]))

"""
zamiana strng na int
"""
"""
Algorytmy ML dzialaja na liczbach .
preprocessing:
     (1) dane nienumeryczne trzeba zamienić na liczby
"""

#tu sa dane nienumeryczne
loan_category = loan[['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']]
#tusa dane tylko numeryczne
loan_num = loan.drop(['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'],axis=1)

#wygodna maszynka do kodowania warto
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
loan[['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']] = ordinal_encoder.fit_transform(loan[['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']])
print('kategorie \n',ordinal_encoder.categories_)



'''
preprocessing:  
    (2)uzupelnianie/edycja danych 
dobrze zrozumiec zlozenie operacji .isnull().any()

'''

print('total nan in data', loan.isnull().sum().sum() )
print('total nan in particular columns', loan.isnull().sum() )
see_incomplete_rows = loan[loan.isnull().any(axis=1)]
print('wybrakowane wiersze', see_incomplete_rows.shape)


#loan_dropna = loan.dropna(subset=['LoanAmount'])
#loan_drop = loan.drop('LoanAmount', axis=1)
median = loan['LoanAmount'].median()
loan['LoanAmount'].fillna(median, inplace=True)

median = loan['Loan_Amount_Term'].median()
loan['Loan_Amount_Term'].fillna(median, inplace=True)

median = loan['Credit_History'].median()
loan['Credit_History'].fillna(median, inplace=True)

loan=loan.drop('Loan_ID',axis=1)

"""
Algorytmy ML pracuja na zestawach trenujacych i testujacych
     (3) podzial danych na trenujace i testujace
"""

# tworzymy zestawy danych do trenowania i testowania
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(loan, test_size=0.2, random_state=42)
print('train_set', train_set.shape,'test_set', test_set.shape)

#!!!!!zbiory atrybutow i etykiet  do ML
loan_data =train_set.drop('Loan_Status', axis=1)
loan_labels = train_set['Loan_Status'].copy()

loan_data_test = test_set.drop('Loan_Status', axis=1)
loan_labels_test = test_set['Loan_Status'].copy()



"""
Szukane rozwiazanie ML problemu to regresja, czyli  wyznaczenie funkcji 
zadana  na atrybutach i  wyliczjaca wartosc domu.

Stosujemy rozne modele numeryczne dla regresji: 
     kolejno liniowy, drzewa decyzyjnego i losowego lasu
Poki co nie interesuje nas co i jak  te algorytmy pracuja.
pracujemy na domyslnych ustawieniach

Zauwaz, podobne uzycie procedur
(1) wczytujemy odpowiednia biblioteke
(2) budujemy obiekt danego obliczenia
(3) rozwiazujemy w tym obiekcie zadanie
(4) oceniamy rozwiazanie 
"""
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(loan_data,loan_labels)
predictions = log_reg.predict(loan_data_test)
LogisticRegression = accuracy_score(loan_labels_test,predictions)
print("accuracy",LogisticRegression)
#ocena wyniku poprzez blad sredni kwadratowy: mse
from sklearn.metrics import mean_squared_error

expected = loan_labels
predicted = log_reg.predict(loan_data)   #see examples 

difference= expected-predicted

lin_mse = mean_squared_error(expected, predicted)
lin_rmse = np.sqrt(lin_mse)
print("blad rmse regresji liniowej ", '%.3f'%lin_rmse)


'''
decision tree
'''

from sklearn.tree import DecisionTreeClassifier
Dec_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
Dec_tree.fit(loan_data, loan_labels)
predictions = Dec_tree.predict(loan_data_test)
DecisionTree = accuracy_score(loan_labels_test,predictions)
print(DecisionTree)

importance = Dec_tree.feature_importances_
plt.bar([x for x in range(len(importance))], importance)
x=range(11)
plt.xticks(x, loan_data.columns, rotation='vertical')
plt.show()
tree_mse = mean_squared_error(expected, predicted)
tree_rmse = np.sqrt(tree_mse)
print("blad rmse regresji z drzewem decyzji", '%.3f'%tree_rmse)
print('waznosc cech \n', Dec_tree.feature_importances_ )
'''
randomforest
'''

from sklearn.ensemble import RandomForestClassifier
Rand_forest = RandomForestClassifier(n_estimators=10,random_state=42)
Rand_forest.fit(loan_data,loan_labels)
predictions = Rand_forest.predict(loan_data_test)
RandomForest = accuracy_score(loan_labels_test,predictions)
print("ac",RandomForest)

predicted = Rand_forest.predict(loan_data)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z losowym lasem decyzji ", '%.3f'%forest_rmse)
print('waznosc cech \n', Rand_forest.feature_importances_ )

importance = Rand_forest.feature_importances_
plt.bar([x for x in range(len(importance))], importance)
x=range(11)
plt.xticks(x, loan_data.columns, rotation='vertical')
plt.show()

'''
KNeighbor
'''

from sklearn.neighbors import KNeighborsClassifier
K_neighbors = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
K_neighbors.fit(loan_data,loan_labels)
predictions = K_neighbors.predict(loan_data_test)
KNeighbors = accuracy_score(loan_labels_test,predictions)
print("ac",KNeighbors)

predicted = K_neighbors.predict(loan_data)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z kneighbor ", '%.3f'%forest_rmse)


"""
SVM
"""



from sklearn.svm import SVC
Svc1 = SVC( random_state = 42) #kernel = 'linear',
Svc1.fit(loan_data,loan_labels)
predictions = Svc1.predict(loan_data_test)
SVCC = accuracy_score(loan_labels_test,predictions)
print("ac",SVCC)

predicted = Svc1.predict(loan_data)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z losowym lasem decyzji ", '%.3f'%forest_rmse)



score = [LogisticRegression,DecisionTree,RandomForest,KNeighbors,SVCC]
Models = pd.DataFrame({
    'Model': ["LogisticRegression","DecisionTree","RandomForest","KNeighbors","SVCC"],
    'Accuracy': score})
Models.sort_values(by='Accuracy', ascending=False)
print(Models)




# DANE PRZESKALOWANE: z transformerem
# zapamietuje transformacje - latwo wiec zastosowac do zbioru testujacego
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(loan_data)
print('srednie', scaler.mean_)
print('odchylenia', scaler.scale_)


train_norma = scaler.transform(loan_data)
test_norma = scaler.transform(loan_data_test)

scaler_minmax = preprocessing.MinMaxScaler().fit(loan_data)

train_minmax= scaler_minmax.transform(loan_data)
test_minmax= scaler_minmax.transform(loan_data_test)




#regresja logistyczna dla danych znormalizowanych 

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
Log_reg = LogisticRegression()
Log_reg.fit(train_norma,loan_labels)
predictions = Log_reg.predict(test_norma)
LogisticRegression = accuracy_score(loan_labels_test,predictions)
print("accuracy",LogisticRegression)
#ocena wyniku poprzez blad sredni kwadratowy: mse
from sklearn.metrics import mean_squared_error

expected = loan_labels
predicted = Log_reg.predict(train_norma)   #see examples 

lin_mse = mean_squared_error(expected, predicted)
lin_rmse = np.sqrt(lin_mse)
print("blad rmse regresji liniowej ", '%.3f'%lin_rmse)


'''

randomforest

'''
from sklearn.ensemble import RandomForestClassifier
Rand_forest = RandomForestClassifier(n_estimators=10,random_state=42)
Rand_forest.fit(train_norma,loan_labels)
predictions = Rand_forest.predict(test_norma)
RandomForest = accuracy_score(loan_labels_test,predictions)
print("ac",RandomForest)

predicted = Rand_forest.predict(train_norma)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z losowym lasem decyzji ", '%.3f'%forest_rmse)
print('waznosc cech \n', Rand_forest.feature_importances_ )

importance = Rand_forest.feature_importances_
plt.bar([x for x in range(len(importance))], importance)
x=range(11)
plt.xticks(x, ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Propperty_Area'], rotation='vertical')
plt.show()



"""

organizowanie przetwarzania potokowego PIPELINE dla powtarzanych  operacji

"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


numerical_pipeline = Pipeline( [
        ('uzupelnij braki', SimpleImputer(strategy='median')),
        ('znormalizuj dane', StandardScaler()),
        
        ])
    
num_wyciek= numerical_pipeline.fit_transform(loan_data)

#porownanie zawartosc tabic num_wyciek z train_norma
roznica = num_wyciek - train_norma
print(roznica)
print(roznica.sum())



"""
Jak porownywac modele? Jak modele nie przetrenowac?

#cross-VALIDATION:  VALIDATION  to select the best model 

"""

from sklearn.model_selection import cross_val_score

Dec_tree_scores = cross_val_score(Dec_tree, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
Dec_tree_rmse_scores = np.sqrt(-Dec_tree_scores)

print("\nscore for decission tree regressor")
print(Dec_tree_rmse_scores)
print('srednia', Dec_tree_rmse_scores.mean())
print('ochylenie stand', Dec_tree_rmse_scores.std())


Log_reg_scores = cross_val_score(Log_reg, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
Log_reg_rmse_scores = np.sqrt(-Log_reg_scores)
print("\nscore for linear regression ")
print(Log_reg_rmse_scores)
print('srednia', Log_reg_rmse_scores.mean())
print('ochylenie stand', Log_reg_rmse_scores.std())
#the linear regression is better than Decision Tree!


Rand_forest_scores = cross_val_score(Rand_forest, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
Rand_forest_rmse_scores = np.sqrt(-Rand_forest_scores)

print("\nscore for random forest regressor")
print(Rand_forest_rmse_scores)
print('srednia', Rand_forest_rmse_scores.mean())
print('ochylenie stand', Rand_forest_rmse_scores.std())
#the RandomForest is the best: the smallest error and the smallest std!


"""
WYSZUKIWANIE NAKLEPSZYCH PARAMETROW NAJLEPSZEGO MODELU

najlepszy model to RandomForestClasifier()
teraz tuning hyperparametrow tego oblicznia.
"""
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(Rand_forest, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(loan_data, loan_labels)
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#HURRA FINAL: NAJLEPSZY MODEL Z NAJLEPSZYMI PARAMETRAMI STOSUJEMY 
# do zestawow testowych
from sklearn.metrics import mean_squared_error
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(loan_data_test)

final_mse = mean_squared_error(loan_labels_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Dokladnosc najlepszego rozwiazania  ML ',"%.1f"%final_rmse)


'''






#regresja liniowa


from sklearn.linear_model import LinearRegression  # wczytanie biblioteka
lin_reg = LinearRegression()                       # stworznie obiektu obliczenia
lin_reg.fit(loan_data, loan_labels)          # rozwiazanie na naszych danych  

print('wartosc stala', '%.3f'%lin_reg.intercept_ ) # wyniki 
wspolczynniki=lin_reg.coef_
print('wspolczynniki od kolejnych atrybutow \n', wspolczynniki) 
atrybuty = loan_data.columns
print("\n WYNIK REGRESJI LINIOWEJ ( estetycznie?)")
print('Loan_Status =',  '%.3f'%lin_reg.intercept_  )
cus ='  '
for i in range(len(wspolczynniki)):
    cus += "  "
    print(cus, "+  %.2f"%wspolczynniki[i],'*', atrybuty[i])
    
    
#ocena wyniku poprzez blad sredni kwadratowy: mse
from sklearn.metrics import mean_squared_error

expected = loan_labels
predicted = lin_reg.predict(loan_data)   #see examples 

difference= expected-predicted

lin_mse = mean_squared_error(expected, predicted)
lin_rmse = np.sqrt(lin_mse)
print("blad rmse regresji liniowej ", '%.3f'%lin_rmse)
#sprawdzamy jak to dziala
        
from statsmodels.api import OLS
print(OLS(loan_labels,loan_data).fit().summary())

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(loan_data, loan_labels)

predicted = tree_reg.predict(loan_data)
tree_mse = mean_squared_error(expected, predicted)
tree_rmse = np.sqrt(tree_mse)
print("blad rmse regresji z drzewem decyzji", '%.3f'%tree_rmse)
print('waznosc cech \n', tree_reg.feature_importances_ )
## CO SIE DZIEJE?? DLaczego blad= 0?
    
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(loan_data, loan_labels)

predicted = forest_reg.predict(loan_data)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z losowym lasem decyzji ", '%.3f'%forest_rmse)
print('waznosc cech \n', forest_reg.feature_importances_ )

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(loan_data)
print('srednie', scaler.mean_)
print('odchylenia', scaler.scale_)


train_norma = scaler.transform(loan_data)
test_norma = scaler.transform(loan_data_test)

scaler_minmax = preprocessing.MinMaxScaler().fit(loan_data)

train_minmax= scaler_minmax.transform(loan_data)
test_minmax= scaler_minmax.transform(loan_data_test)

lin_reg = LinearRegression()
lin_reg.fit(train_norma, loan_labels)

print('wartosc stala', '%.3f'%lin_reg.intercept_ )
wspolczynniki=lin_reg.coef_
print('wspolczynniki od kolejnych atrybutow \n', wspolczynniki)
atrybuty = loan_data.columns

print("\n WYNIK REGRESJI LINIOWEJ")
print('cena domu =',  '%.3f'%lin_reg.intercept_  )
cus ='  '
for i in range(len(wspolczynniki)):
    cus += "  "
    print(cus, "+  %.2f"%wspolczynniki[i],'*', atrybuty[i])
    
lin_mse = mean_squared_error(loan_labels, lin_reg.predict(train_norma))
lin_rmse = np.sqrt(lin_mse)
print("blad rmse regresji liniowej ", '%.3f'%lin_rmse)

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(train_norma, loan_labels)

predicted = forest_reg.predict(train_norma)
forest_mse = mean_squared_error(expected, predicted)
forest_rmse = np.sqrt(forest_mse)
print("blad rmse regresji z losowym lasem decyzji ", '%.3f'%forest_rmse)
print('waznosc cech \n', forest_reg.feature_importances_ )



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


numerical_pipeline = Pipeline( [
        ('uzupelnij braki', SimpleImputer(strategy='median')),
        ('znormalizuj dane', StandardScaler()),
        
        ])
    
num_wyciek= numerical_pipeline.fit_transform(loan_data)

#porownanie zawartosc tabic num_wyciek z train_norma
roznica = num_wyciek - train_norma
print(roznica)
print(roznica.sum())




from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_reg, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

print("\nscore for decission tree regressor")
print(tree_rmse_scores)
print('srednia', tree_rmse_scores.mean())
print('ochylenie stand', tree_rmse_scores.std())


lin_scores = cross_val_score(lin_reg, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("\nscore for linear regression ")
print(lin_rmse_scores)
print('srednia', lin_rmse_scores.mean())
print('ochylenie stand', lin_rmse_scores.std())
#the linear regression is better than Decision Tree!


forest_scores = cross_val_score(forest_reg, loan_data, loan_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("\nscore for random forest regressor")
print(forest_rmse_scores)
print('srednia', forest_rmse_scores.mean())
print('ochylenie stand', forest_rmse_scores.std())    
    
    
    
    
"""
WYSZUKIWANIE NAKLEPSZYCH PARAMETROW NAJLEPSZEGO MODELU

najlepszy model to RandomForestRegressor()
teraz tuning hyperparametrow tego oblicznia.
"""
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(loan_data, loan_labels)
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
from sklearn.metrics import mean_squared_error
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(loan_data_test)

final_mse = mean_squared_error(loan_labels_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Dokladnosc najlepszego rozwiazania  ML ',"%.1f"%final_rmse)
    
''' 
    
    
    
    
    
    
    
    
    
    
    

