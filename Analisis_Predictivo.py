import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from matplotlib import colors as mcolors


class Analisis_Predictivo:

    def __init__(self, datos: DataFrame, predecir: str, predictoras=[],
                 modelo=None, train_size=80, random_state=None):
        '''
        datos: Datos completos y listos para construir un modelo

        modelo: Instancia de una Clase de un método de clasificación(KNN,Árboles,SVM,etc).
        Si no especifica un modelo no podrá utilizar el método fit_n_review()

        predecir: Nombre de la variable a predecir

        predictoras: Lista de los nombres de las variables predictoras.
        Si vacío entonces utiliza todas las variables presentes excepto la variable a predecir.

        train_size: Proporción de la tabla de entrenamiento respecto a la original.

        random_state: Semilla aleatoria para la división de datos(training-testing).
        '''
        self.datos = datos
        self.predecir = predecir
        self.predictoras = predictoras
        self.modelo = modelo
        self.random_state = random_state
        if modelo != None:
            self.train_size = train_size
            self._training_testing()

    def training_testing_con_reja_corte(self, corte=None, model=None):
        model.fit(self.X_train, self.y_train.ravel())
        probabilidad = model.predict_proba(self.X_test)[:, 1]
        for c in corte:
            print("===========================")
            print("Probabilidad de Corte: ", c)
            prediccion = np.where(probabilidad > c, "Si", "No")
            MC = confusion_matrix(self.y_test, prediccion)
            indices = self.indices_general(MC, list(np.unique(self.y)))
            for k in indices:
                print("\n%s:\n%s" % (k, str(indices[k])))

    def _training_testing(self):
        if len(self.predictoras) == 0:
            X = self.datos.drop(columns=[self.predecir])
        else:
            X = self.datos[self.predictoras]

        y = self.datos[self.predecir].values

        self.y = y
        self.x = X

        train_test = train_test_split(X, y, train_size=self.train_size,
                                      random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test

    def fit_predict(self):
        if self.modelo != None:
            self.modelo.fit(self.X_train, self.y_train)
            return self.modelo.predict(self.X_test)

    def fit_predict_resultados(self, imprimir=True):
        if (self.modelo != None):
            y = self.datos[self.predecir].values
            prediccion = self.fit_predict()
            MC = confusion_matrix(self.y_test, prediccion)
            indices = self.indices_general(MC, list(np.unique(y)))
            if imprimir == True:
                for k in indices:
                    print("\n%s:\n%s" % (k, str(indices[k])))

            return indices

    def indices_general(self, MC, nombres=None):
        "Método para calcular los índices de calidad de la predicción"
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres != None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión": MC,
                "Precisión Global": precision_global,
                "Error Global": error_global,
                "Precisión por categoría": precision_categoria}

    def distribucion_variable_predecir(self):
        "Método para graficar la distribución de la variable a predecir"
        variable_predict = self.predecir
        data = self.datos
        colors = list(dict(**mcolors.CSS4_COLORS))
        df = pd.crosstab(index=data[variable_predict], columns="valor") / data[variable_predict].count()
        fig = plt.figure(figsize=(10, 9))
        g = fig.add_subplot(111)
        countv = 0
        titulo = "Distribución de la variable %s" % variable_predict
        for i in range(df.shape[0]):
            g.barh(1, df.iloc[i], left=countv, align='center', color=colors[11 + i], label=df.iloc[i].name)
            countv = countv + df.iloc[i]
        vals = g.get_xticks()
        g.set_xlim(0, 1)
        g.set_yticklabels("")
        g.set_title(titulo)
        g.set_ylabel(variable_predict)
        g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        countv = 0
        for v in df.iloc[:, 0]:
            g.text(np.mean([countv, countv + v]) - 0.03, 1, '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
        g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)

    def poder_predictivo_categorica(self, var: str):
        "Método para ver la distribución de una variable categórica respecto a la predecir"
        data = self.datos
        variable_predict = self.predecir
        df = pd.crosstab(index=data[var], columns=data[variable_predict])
        df = df.div(df.sum(axis=1), axis=0)
        titulo = "Distribución de la variable %s según la variable %s" % (var, variable_predict)
        g = df.plot(kind='barh', stacked=True, legend=True, figsize=(10, 9), \
                    xlim=(0, 1), title=titulo, width=0.8)
        vals = g.get_xticks()
        g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
        for bars in g.containers:
            plt.setp(bars, width=.9)
        for i in range(df.shape[0]):
            countv = 0
            for v in df.iloc[i]:
                g.text(np.mean([countv, countv + v]) - 0.03, i, '{:.1%}'.format(v), color='black', fontweight='bold')
                countv = countv + v

    def poder_predictivo_numerica(self, var: str):
        "Función para ver la distribución de una variable numérica respecto a la predecir"
        sns.FacetGrid(self.datos, hue=self.predecir, height=6).map(sns.kdeplot, var, shade=True).add_legend()

    def graficar_validacion_error(self, repeticion: int):
        error_tt = []
        error_cv = []

        for i in range(0, repeticion):
            self._training_testing()
            self.modelo.fit(self.X_train, self.y_train.ravel())
            error_tt.append(1 - self.modelo.score(self.X_test, self.y_test.ravel()))

        for i in range(0, repeticion):
            kfold = KFold(n_splits=10, shuffle=True)
            error_folds = []
            folds = kfold.split(self.x, self.y)

            for train, test in folds:
                self.modelo.fit(self.x.iloc[train], self.y[train].ravel())
                error_folds.append((1 - self.modelo.score(self.x.iloc[test], self.y[test].ravel())))
            error_cv.append(np.mean(error_folds))

        plt.figure(figsize=(15, 10))
        plt.plot(error_tt, 'o-', lw=2)
        plt.plot(error_cv, 'o-', lw=2)
        plt.xlabel("Número de Iteración", fontsize=15)
        plt.ylabel("Error Cometido", fontsize=15)
        plt.title("Variación del Error", fontsize=20)
        plt.grid(True)
        plt.legend(['Training Testing', 'K-Fold CV'], loc='upper right',
                   fontsize=15)

    def graficar_validacion_error_validacion_kfold_vs_testing(self, repeticion: int):
        error_tt = []
        error_cv = []

        for i in range(0, repeticion):
            self._training_testing()
            self.modelo.fit(self.X_train, self.y_train.ravel())
            error_tt.append(1 - self.modelo.score(self.X_test, self.y_test.ravel()))

        for i in range(0, repeticion):
            kfold = KFold(n_splits=10, shuffle=True)
            error_folds = []
            folds = kfold.split(self.x, self.y)

            for train, test in folds:
                self.modelo.fit(self.x.iloc[train], self.y[train].ravel())
                error_folds.append((1 - self.modelo.score(self.x.iloc[test], self.y[test].ravel())))
            error_cv.append(np.mean(error_folds))

        plt.figure(figsize=(25, 15))
        plt.plot(error_tt, 'o-', lw=2)
        plt.plot(error_cv, 'o-', lw=2)
        plt.xlabel("Número de Iteración", fontsize=15)
        plt.ylabel("Error Cometido", fontsize=15)
        plt.title("Variación del Error", fontsize=20)
        plt.grid(True)
        plt.legend(['Training Testing', 'K-Fold CV'], loc='upper right',
                   fontsize=15)


class Graficar:
    def __init__(self, models, label, color, x, y):
        self.__color = color
        self.__label = label
        self.__models = models
        self.__porcentajes = []
        self.__x = x
        self.__y = y

    def getPorcentajeKFoald(self):
        for model in self.__models:
            instancia_kfold = KFold(n_splits=10, shuffle=True)
            crossScore = cross_val_score(model, self.__x, self.__y, cv=instancia_kfold)
            self.__porcentajes.append(crossScore.mean())

    def barras(self):
        self.getPorcentajeKFoald()
        plt.figure(figsize=(13, 9))
        barras = self.__label
        y_pos = np.arange(len(barras))
        plt.bar(y_pos, self.__porcentajes, color=self.__color)
        plt.xticks(y_pos, barras)
        plt.show()
