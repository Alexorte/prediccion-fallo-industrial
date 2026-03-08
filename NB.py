import math
from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import norm

# Implementación propia de Naive Bayes que hereda de BaseEstimator siguiendo la estructura de scikit-learn
# Las guias de scikit-learn para la implementación de algoritmos se encuentran en https://scikit-learn.org/stable/developers/develop.html
class NB(BaseEstimator):
    # Constructor
    def __init__(self, categorical_features='all'):
        self.categorical_features = categorical_features
        return

    # To string
    def __str__(self):
        result = "NB(): Naive Bayes with Categorical and Gaussian Features\n\n"
        result += "P(Y):\n" + str(np.round(self.class_prob_, 4)) + "\n\n"
        
        for i in range(len(self.tables_)):
            if self.is_categorical_[i]:  # Característica categórica
                result += f"P(X{i}|Y):\n"
                for cls, probs in self.tables_[i].items():
                    result += f"{cls}:\t{np.round(probs, 4)}\n"
                result += "\n"
            else:  # Característica gaussiana
                result += f"P(X{i}|Y):\n"
                for cls, (mean, var) in self.tables_[i].items():
                    result += f"{cls}:\tμ = {mean:.4f}, σ² = {var:.4f}\n"
                result += "\n"
        return result


    # Función fit para entrenar el modelo a partir de un conjunto de datos X y sus etiquetas y
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Determinar características categóricas
        if self.categorical_features == 'all':
            self.categorical_features_ = list(range(n_features))
        elif self.categorical_features == 'none':
            self.categorical_features_ = []
        else:
            self.categorical_features_ = self.categorical_features
        self.is_categorical_ = np.isin(range(n_features), self.categorical_features_)

        # Información de las clases
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)    
        self.class_prob_ = class_counts / n_samples  # P(Y)

        # Información de las categorías para las características categóricas
        self.categories_ = []
        for i in range(n_features):
            if self.is_categorical_[i]:
                self.categories_.append(np.unique(X[:, i]))
            else:
                self.categories_.append(None)

        self.tables_ = [] 
        for i in range(n_features):
            # Probabilidades categóricas con suavizado Laplace 
            if self.is_categorical_[i]:
                self.tables_.append({ })
                # TODO: Calcular las probabilidades categóricas
                x = X[:,i] 
                for j in self.classes_:
                    x_class = x[y == j]
                    self.tables_[i][j] = []
                    for k in self.categories_[i]:
                        self.tables_[i][j].append( (( np.sum(x_class == k) + 1 ) / (len(x_class) + self.n_classes_) ) )
                        
            # Parámetros gaussianos 
            else:
                # TODO: Calcular los parámetros gaussianos (media, varianza)
                x = X[:,i]
                self.tables_.append({})
                for index,c in enumerate(self.classes_):
                    self.tables_[i][c] = []
                    x1 = x[y == c]
                    self.tables_[i][c].append( (np.mean(x1)) )
                    self.tables_[i][c].append( (np.var(x1)) )
        return self

    # Función predict_proba para predecir las probabilidades de las etiquetas de un conjunto de datos X
    def predict_proba(self, X):
        num_instances = X.shape[0]
        # Inicializar la matriz de probabilidades (una fila por cada instancia de X y una columna por cada posible valor de Y)
        prob = np.zeros((num_instances, len(self.classes_)))
        
        for i in range(num_instances):
            x = X[i, :]
            # TODO: Calcular la probabilidad de cada clase para la instancia x
            for c in range(self.n_classes_):
                for index,valor in enumerate(x):
                    if self.is_categorical_[index]:
                            valor_idx = np.where(self.categories_[index] == valor)[0][0] #Cogemos posicion donde se encuentre el valor a calcular
                            prob[i][c] += np.log(self.tables_[index][self.classes_[c]][valor_idx]) 
                    else:  
                        prob[i][c] += np.log( norm.pdf(valor,self.tables_[index][self.classes_[c]][0], np.sqrt( self.tables_[index][self.classes_[c]][1] )) ) 
                prob[i][c] = prob[i][c] + np.log( self.class_prob_[c] )
                
            prob[i] = np.power(np.e,prob[i])
            suma_t = sum(prob[i])
            prob[i] = prob[i] / suma_t
                         
        return prob
    

    # TODO: Función predict para predecir las etiquetas de un conjunto de datos X
    def predict(self, X): #calcular el mayor valor para cada fila de X
        probs = self.predict_proba(X)
        return np.argmax(probs,axis=1)

    # TODO: Función score para calcular el porcentaje de acierto del modelo a partir de un conjunto de datos X y sus etiquetas y
    def score(self, X, y):
        pr = self.predict(X)
        return sum(self.classes_[pr] == y) / len(y) 