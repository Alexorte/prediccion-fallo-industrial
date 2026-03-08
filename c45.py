import random
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from collections import Counter
import numpy as np

class Node:
    def __init__(self):
        # Indica si el nodo es una hoja, o no
        self.is_leaf = False

        # Atributos relacionados con la variable que representa el nodo
        self.is_num = True      # Indica si la variable es numérica (True) o categórica (False)
        self.cat_dict = {}    # Diccionario para variables categóricas con formato {valor: indice}
        
        # Atributos cuando el objeto es una raíz
        self.var = None         # Nombre de la variable de corte
        self.var_index = -1     # Índice de la variable de corte
        self.cut_value = 0      # Valor de la variable de corte, en caso de ser numérica
        self.children = []      # Lista de hijos

        # Atributos cuando el objeto es una hoja
        self.class_value = -1       # Valor de la clase si el nodo es hoja
        self.class_count = (0,0)    # Tupla con el formato (casos con valor class_value, casos totales en la hoja)

        # Profundidad del nodo
        self.depth = -1

    def __str__(self):
        output = ''
        if(self.is_leaf):
            output += 'Class value: ' + str(self.class_value) + '\tCounts: ' + str(self.class_count)
        else:
            output += 'Feature '+ str(self.var)
            for i in range(len(self.children)):
                output += '\n'+'\t'*(self.depth+1)+str(self.cut_value)+': '+str(self.children[i]) 
            
        return output
    
    # Esta función nos servirá para hacer predicciones recursivamente hasta llegar a un nodo hoja. Debe ser completada
    def predict(self,x):
        if self.is_leaf:
            return self.class_value
        else:
            if self.is_num:
                #pass # TODO: Completar aquí
                if x[self.var_index] <= self.cut_value:
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)
            else:
                #pass # TODO: Completar aquí
                if x[self.var_index] in self.cat_dict:
                    ind = self.cat_dict[x[self.var_index]]
                else:
                    ind = random.choice(list(self.cat_dict.values()))
                return self.children[ind].predict(x)

class C45Classifier(BaseEstimator, ClassifierMixin):

    # Constructor de la clase, aquí se definen e inicializan las variables de la clase.
    def __init__(self, vars, disc, cont, max_depth=2, criterion='classification_error', prune=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.prune = prune

        self.vars = vars
        self.disc = disc
        self.cont = cont

        # Diccionario que nos permitirá convertir el nombre de la variable en su índice.
        self.features_dict = {feat: i for i, feat in enumerate(self.vars)}

        # Raíz del árbol
        self.tree = Node()   



    # Función para entrenar el modelo.
    def fit(self, X, y):

        self.var_clases = np.unique(y) # para generalizar el árbol.
        # Llamada a la función recursiva que aprende el árbol.
        self._partial_fit(X, y, self.tree, 0, set([]))

        if self.prune:
            self._prune_tree(self.tree)
        
        return self
    

    # Función para hacer predicciones.
    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])
    

    # Función recursiva que busca la variable y corte que maximiza la ganancia de información.
    # - Las variables continuas se tratan con un corte binario, lo que quiere decir que pueden ser usadas multiples veces. 
    # - Las variables discretas ramifican tantas veces como valores tengan, asi que solo pueden ser usadas una vez por camino, 
    #   debiendo almacenarlas en el conjunto `borradas`. 
    def _partial_fit(self, X, y, current_tree, current_depth, borradas):
        def _make_leaf():
            current_tree.is_leaf = True
            counts = Counter(y)
            max_value = counts.most_common(1) # most_common(1) devuelve una lista con el elemento más común y su frecuencia.
            current_tree.class_value = max_value[0][0]
            current_tree.class_count = (max_value[0][1], len(y))
            return
        
        # Antes de nada, si hemos alcanzado la profundidad máxima, el nodo se convierte en hoja.
        if current_depth >= self.max_depth:
            _make_leaf()
            return

        # Primero obtenemos el mejor punto de corte para el nodo actual dependiendo del criterio.
        best_var, cut_value, is_num = self._split(X, y, borradas, self.criterion)

        # Si no hay ninguna partición que mejore la actual, el nodo se convierte en hoja.
        if best_var is None:
            _make_leaf()
            return
    
        # Antes de llamar a la función recursiva, hay que actualizar los valores del árbol.
        borradas_copy = borradas.copy()
        if not is_num:    # Solo borramos las variables categóricas ya que estarán totalmente particionadas.
            borradas_copy.add(best_var)
            current_tree.is_num = False

        if len(y)==0: # Si no hay ejemplos, el nodo se convierte en hoja.
            current_tree.is_leaf = True
            current_tree.class_value = random.choice(self.var_clases)  # Elige aleatoriamente una clase
            current_tree.class_count = (0, 0)  # Cuenta vacía
            return

        _make_leaf() # Hacemos que todo nodo tenga la informacion tengan la tupla (casos con valor class_value, casos totales en la hoja).
        current_tree.is_leaf = False
        current_tree.depth = current_depth
        current_tree.var = best_var
        current_tree.var_index = self.features_dict[best_var]

        # Finalmente, se hace la llamada recursiva en función de si es numérica o categórica.
        if is_num:
            #pass # TODO: Completar aquí.
            left = Node()
            right = Node() #Creamos las particiones
            current_tree.cut_value = cut_value #nos guardamos el valor del punto de corte
            current_tree.children = [left, right]
            X_left = X[:,current_tree.var_index] <= cut_value 
            X_right = X[:,current_tree.var_index] > cut_value #nos guardamos una lista con true o false dependiendo si cumple las condiciones
            self._partial_fit(X[X_left], y[X_left], left, current_depth+1, borradas_copy)
            self._partial_fit(X[X_right], y[X_right], right, current_depth+1, borradas_copy) #hacemos la llamada recursiva con la particion correspondiente
        else:
            #pass # TODO: Completar aquí.
            particiones = np.unique(X[:,current_tree.var_index])
            for i,val in enumerate(particiones):
                child = Node()
                current_tree.children.append(child)
                current_tree.cat_dict[val] = i
                X_index = X[:,current_tree.var_index] == val
                self._partial_fit(X[X_index],y[X_index],child,current_depth+1,borradas_copy)
        return


    # Cálculo del mejor punto de corte en función de: Error de clasificación.
    def _split(self, X, y, borradas, criterion='classification_error'):
        # Error actual (sin partición)
        error_best = self._compute_split_criterion(y, criterion)

        best_var = None
        is_num = True
        cut_value = None    # Para variables categóricas no hay valor de corte (devolvemos None).
        
        for var in self.vars:
            index = self.features_dict[var]
            error = 0

            if var in self.disc:
                #pass # TODO: Completar aquí. 
                if var not in borradas:
                    x = np.unique(X[:,index])
                    yt = len(X[:,index])
                    for i in x:
                        X_index = X[:,index] == i
                        y1 = y[X_index]
                        error += (len(y1)/yt) * self._compute_split_criterion(y1,criterion) 
                    if error < error_best:
                        best_var = var
                        error_best = error
                        is_num = False

            elif var in self.cont:
                x = X[:,index]
                x_ord = np.unique(x)      

                if len (x_ord) < 50:
                    for i in x_ord[:-1]:
                        #corte con los valores unicos del conjunto
                        cut = (i+(i+1))/2   
                        y_left = y[x <= cut]
                        y_right = y[x > cut]
                        error = (self._compute_split_criterion(y_left, criterion)*(len(y_left)/len(x))) + (self._compute_split_criterion(y_right, criterion)*(len(y_right)/len(x)))
                        if error < error_best:
                            best_var = var
                            cut_value = cut
                            error_best = error
                            is_num = True

                else:

                    cuts = np.percentile(x_ord,np.arange(10,100,10))

                    for cut in cuts:
                        y_left = y[x <= cut]
                        y_right = y[x > cut]
                        error = (self._compute_split_criterion(y_left, criterion)*(len(y_left)/len(x))) + (self._compute_split_criterion(y_right, criterion)*(len(y_right)/len(x)))
                        if error < error_best:
                            best_var = var
                            cut_value = cut
                            error_best = error
                            is_num = True

            # Si conseguimos un error de 0 (óptimo), terminamos
            if error_best == 0:
                break

        return best_var, cut_value, is_num
    
    # TODO: Cálculo del mejor punto de corte en función de: Error de clasificación; Entropía; Índice Gini.
    def _compute_split_criterion(self, y, criterion='classification_error'):
        # TODO: Completar aquí si tenéis código común a los tres criterios.

        if criterion == 'classification_error':
            v,c = np.unique(y,return_counts=True) #v son los valores unicos, c es la proporcion de cada uno.
            return 1 - max(c)/len(y)
        
        elif criterion == 'entropy':
            #pass # TODO: Completar aquí.
            v,c = np.unique(y,return_counts=True)
            p = c/len(y)
            return -np.sum(p*np.log2(p))
        
        elif criterion == 'gini':
            #pass # TODO: Completar aquí.
            v,c = np.unique(y,return_counts=True)
            p = c/len(y)
            return 1 - np.sum(p**2)

        else:
            raise ValueError('Criterio no válido.')

    
    # TODO: Completar esta función para realizar la poda del modelo.
    def _prune_tree(self,node): 

        errors = 0
        error_padre = 0
        error = 0
        
        
        if (node.is_leaf):
            return

        for i in node.children:
            self._prune_tree(i)
            if i.is_leaf:
                error += ((i.class_count[1] - i.class_count[0] + len(self.var_clases) - 1) / (i.class_count[1] + len(self.var_clases))) * i.class_count[1]
            else:
                return
        errors = error / node.class_count[1] 
        error_padre = (node.class_count[1]  - node.class_count[0] + len(self.var_clases) - 1) / (node.class_count[1] + len(self.var_clases))
        if error_padre <= errors:
            node.is_leaf = True
            node.children = []
        else:
            return

    # Función para imprimir el modelo.
    def __str__(self):
        return str(self.tree)    