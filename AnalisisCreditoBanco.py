from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd  # Tratamiento de datos
import matplotlib.pyplot as plt  # Herramienta para graficar
import seaborn as sns  # Herramienta para graficar
# Declaracion de funciones


def procesar_datos(df_banco):
    df_banco = df_banco.drop_duplicates() if df_banco.duplicated(
    ).any() else df_banco  # asegurar que no hayan datos duplicados
    df_banco = df_banco.dropna() if df_banco.isnull().values.any(
    ) else df_banco  # asegurar que no hayan datos nulos
    a = {'no checking account': 4,
         '>= 200 DM / salary assignments for at least 1 year': 3,
         '0 <= ... < 200 DM': 2,
         '< 0 DM': 1
         }
    df_banco['account_check_status'] = df_banco['account_check_status'].map(a)

    a = {'no credits taken/ all credits paid back duly': 1,
         'all credits at this bank paid back duly': 2,
         'existing credits paid back duly till now': 3,
         'delay in paying off in the past': 4,
         'critical account/ other credits existing (not at this bank)': 5
         }
    df_banco['credit_history'] = df_banco['credit_history'].map(a)

    a = {'car (new)': 1,
         'car (used)': 2,
         'furniture/equipment': 3,
         'radio/television': 4,
         'domestic appliances': 5,
         'repairs': 6,
         'education': 7,
         '(vacation - does not exist?)': 8,
         'retraining': 9,
         'business': 10,
         'others': 11
         }
    df_banco['purpose'] = df_banco['purpose'].map(a)

    a = {'unknown/ no savings account': 1,
         '.. >= 1000 DM ': 2,
         '500 <= ... < 1000 DM ': 3,
         '100 <= ... < 500 DM': 4,
         '... < 100 DM': 5
         }
    df_banco['savings'] = df_banco['savings'].map(a)

    a = {'.. >= 7 years': 1,
         '4 <= ... < 7 years': 2,
         '1 <= ... < 4 years': 3,
         '... < 1 year ': 4,
         'unemployed': 5
         }
    df_banco['present_emp_since'] = df_banco['present_emp_since'].map(a)

    a = {'male : divorced/separated': 1,
         'female : divorced/separated/married': 2,
         'male : single': 3,
         'male : married/widowed': 4,
         'female : single': 5
         }
    df_banco['personal_status_sex'] = df_banco['personal_status_sex'].map(a)

    a = {'none': 1,
         'co-applicant': 2,
         'guarantor': 3
         }
    df_banco['other_debtors'] = df_banco['other_debtors'].map(a)

    a = {'real estate': 1,
         'if not A121 : building society savings agreement/ life insurance': 2,
         'if not A121/A122 : car or other, not in attribute 6': 3,
         'unknown / no property': 4
         }
    df_banco['property'] = df_banco['property'].map(a)

    a = {'bank': 1,
         'stores': 2,
         'none': 3
         }
    df_banco['other_installment_plans'] = df_banco['other_installment_plans'].map(
        a)

    a = {'rent': 1,
         'own': 2,
         'for free': 3
         }
    df_banco['housing'] = df_banco['housing'].map(a)

    a = {'unemployed/ unskilled - non-resident': 1,
         'unskilled - resident': 2,
         'skilled employee / official': 3,
         'management/ self-employed/ highly qualified employee/ officer': 4
         }
    df_banco['job'] = df_banco['job'].map(a)

    a = {'yes, registered under the customers name ': 1,
         'none': 0
         }
    df_banco['telephone'] = df_banco['telephone'].map(a)

    a = {'yes': 1,
         'no': 0
         }
    df_banco['foreign_worker'] = df_banco['foreign_worker'].map(a)
    return df_banco


def feature_engineering(df_banco):
    dic_sexo = {2: 1, 5: 1, 1: 0, 3: 0, 4: 0}
    dic_est_civil = {3: 1, 5: 1, 1: 0, 2: 0, 4: 0}
    df_banco['sexo'] = df_banco['personal_status_sex'].map(dic_sexo)
    df_banco['estado_civil'] = df_banco['personal_status_sex'].map(
        dic_est_civil)
    df_banco['rango_edad'] = pd.cut(x=df_banco['age'],
                                    bins=[18, 30, 40, 50, 60, 70, 80],
                                    labels=[1, 2, 3, 4, 5, 6]).astype(int)
    df_banco['rango_plazos_credito'] = pd.cut(x=df_banco['duration_in_month'],
                                              bins=[1, 12, 24, 36, 48, 60, 72],
                                              labels=[1, 2, 3, 4, 5, 6]).astype(int)
    df_banco['rango_valor_credito'] = pd.cut(x=df_banco['credit_amount'],
                                             bins=[1, 1000, 2000, 3000, 4000,
                                                   5000, 6000, 7000, 8000, 9000,
                                                   10000, 11000, 12000, 13000,
                                                   14000, 15000, 16000, 17000,
                                                   18000, 19000, 20000],
                                             labels=[1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                     10, 11, 12, 13, 14, 15, 16,
                                                     17, 18, 19, 20]).astype(int)
    df_banco = df_banco.drop(columns=['personal_status_sex', 'age',
                                      'duration_in_month', 'credit_amount'])


def analisis_exploratorio(df_banco):

    histogramas = ['sexo', 'estado_civil',
                   'rango_plazos_credito', 'rango_edad', 'default']
    lista_histogramas = list(enumerate(histogramas))
    # plt.figure(figsize=(8, 5))
    # plt.title('Histogramas')
    for i in lista_histogramas:
        # plt.subplot(3, 2, i[0]+1)
        plt.figure(figsize=(4, 3))
        plt.title(f'Histograma de variable: {histogramas[i[0]]}')
        sns.countplot(x=i[1], data=df_banco)
        plt.xlabel(i[1])
        plt.ylabel('Total')

    plt.show()


def visualiza_resultados():
    # global df_banco, resultados
    results_df = pd.DataFrame(resultados)
    results_df.set_index('Model', inplace=True)

    # Transponer el DataFrame para facilitar la representación
    results_df = results_df.T
    colors = ['#0077b6', '#CDDBF3', '#9370DB', '#DDA0DD']

    # Gráfico de barras agrupadas para cada métrica
    results_df.plot(kind='bar', figsize=(12, 6),
                    colormap='viridis', rot=0, color=colors)
    plt.title('Comparación de Métricas por Modelo')
    plt.xlabel('Métricas')
    plt.ylabel('Puntuación')
    plt.legend(title='Modelos')
    plt.tight_layout()
    plt.show()

    # @title Texto de título predeterminado
    # from IPython.display import HTML, display

    # Texto que quieres centrar
    # texto = "¿Cuál de estos modelos seleccionarías y por qué?"

    # Crear una celda HTML con el texto centrado
    # display(HTML(f"<center><h2>{texto}</h2></center>"))


def crea_modelos(df_banco):
    y = df_banco['default']
    x = df_banco.drop(columns='default')
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.30, random_state=77)

    models = {
        'Regresión Logística': LogisticRegression(),
        'Árbol de Decisión': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = {'Model': [], 'Accuracy': [], 'Precision': [],
               'Recall': [], 'F1-score': [], 'AUC-ROC': []}

    for name, model in models.items():
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions)
        recall = recall_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_x)
            roc_auc = roc_auc_score(test_y, proba[:, 1])
        else:
            roc_auc = None

        results['Model'].append(name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-score'].append(f1)
        results['AUC-ROC'].append(roc_auc)

    resultados = results
    return resultados

# Declaracion de variables globales


pd.set_option('display.max_columns', None)
df_banco = pd.read_csv('german_credit.csv')

# Indice de cada uno de los datos presentes en la columna

columnas = list(df_banco.select_dtypes(include=['object']).columns)

# Analisis de cada columna

print('Analisis de cada columna')
for columna in columnas:
    print(f'El nombre de la columna: {columna}')
    print(list(df_banco[f'{columna}'].value_counts().index))
    print('\n')

# Funcion para procesar los datos extraidos
procesar_datos(df_banco)

# Procesa la base de datos por rango
feature_engineering(df_banco)

# Configurar el estilo de Seaborn (opcional)
sns.set(style="whitegrid")

# Funcion para imprimir analisis sobre las columnas
analisis_exploratorio(df_banco)

# Mapa de calor
plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(df_banco.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlaciones', fontsize=18)
plt.show()


# Dividir los datos en conjuntos de entrenamiento y prueba
X = df_banco.drop('default', axis=1)
y = df_banco['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evaluar el modelo
predicciones = modelo.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Accuracy: {accuracy}')


resultados = crea_modelos(df_banco)
for i, model in enumerate(resultados['Model']):
    print(model)
    print(resultados['Accuracy'][i])
    print(resultados['Precision'][i])
    print(resultados['Recall'][i])
    print(resultados['F1-score'][i])
    print(resultados['AUC-ROC'][i])
    print('\n')


visualiza_resultados()
