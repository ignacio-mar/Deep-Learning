import numpy as np
import time

from matplotlib import pylab as plt
from IPython import display

from grafica import *

class NeuronaGradiente(object):
    """
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    cotaE : float
        minimum error threshold
    FUN : string
        activation function: 'sigmoid', 'tanh', otherwise linear
    random_state : int
        Random number generator seed for random weight initialization.
    draw : int
        1 si dibuja -  0 si no
    title : list con 2 elementos
        titulos de los ejes - sólo 2D
        
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, alpha=0.01, n_iter=50, cotaE=10e-07, FUN='sigmoid', COSTO='ECM', random_state=None, draw=0, title=['X1','X2']):
        self.alpha = alpha
        self.n_iter = n_iter
        self.cotaE = cotaE
        self.FUN = FUN
        self.COSTO = COSTO
        self.random_state = random_state #-- asignar el valor 1 para fijar la semilla por defecto es aleatorio
        self.draw = draw
        self.title = title

    def fit(self, X, y):
        """Fit training data (VERSIÓN CORREGIDA)."""
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.uniform(-0.5, 0.5, size= X.shape[1]) 
        self.b_ = rgen.uniform(-0.5, 0.5)
        self.errors_ = []
        self.accuracy_ = []
        
        ph = 0 
        ErrorAnt = 0
        ErrorAct = 1
        
        i = 0
        while ((i<self.n_iter) and (np.absolute(ErrorAnt- ErrorAct) > self.cotaE)):
            ErrorAnt = ErrorAct
            ErrorAct = 0
            
            # Mezclamos los datos en cada época (SGD Real)
            indices = np.random.permutation(len(y))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Bucle estocástico
            for xi, target in zip(X_shuffled, y_shuffled):
                
                salida = self.predict_nOut(xi) # ŷ (predicción)
                errorXi = (target - salida)    # (y - ŷ)
                
                # ---------------------------------------------------------
                # --- ¡AQUÍ ESTÁ LA CORRECCIÓN DE LÓGICA! ---
                # ---------------------------------------------------------
                
                if self.COSTO == 'EC_binaria' and self.FUN == 'sigmoid':
                    # Gradiente de Cross-Entropy + Sigmoid (Regresión Logística)
                    # La derivada se simplifica a (salida - target), o sea -errorXi
                    gradiente_neta = -errorXi 
                else:
                    # Gradiente de ECM (u otros)
                    # (target - salida) * derivada_de_activacion
                    gradiente_neta = errorXi * self.derivar(salida)

                # El 'update' es alpha * gradiente_neta
                # (El signo negativo ya está incluido en el gradiente_neta si es BCE)
                
                # --- ACTUALIZACIÓN DE PARÁMETROS ---
                # W_nuevo = W_viejo - (alpha * (dJ/dW))
                # dJ/dW = dJ/dneta * dneta/dW = gradiente_neta * xi
                
                self.w_ = self.w_ - (self.alpha * gradiente_neta * xi)
                self.b_ = self.b_ - (self.alpha * gradiente_neta * 1)
                
                # ---------------------------------------------------------
                
                ErrorAct += self.fCosto(target, salida)
                
            self.errors_.append(ErrorAct / len(y)) # Guardamos el ECM
            self.accuracy_.append(self.accuracy(X,y))
            
            # graficar la recta
            if (self.draw):
                ph = dibuPtosRecta(X,y, self.w_, self.b_, self.title, ph)
            
            i = i + 1
        return self
      
    def fCosto(self,y, y_hat):
        #-- y es el valor esperado e y_hat el valor obtenido (ambos escalares)
        EPS = EPS = np.finfo(float).eps
        if (self.COSTO=='ECM'):
            return np.mean(((y-y_hat)**2))
        if (self.COSTO=='EC_binaria'):
            return(-y*np.log(y_hat+EPS)-(1-y)*np.log(1-y_hat+EPS))
        if (self.COSTO=='EC'):
            return(-y*np.log(y_hat+EPS))
        else:
            return(np.absolute(y-y_hat))


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def evaluar(self, x):
        if (self.FUN=='tanh'):
            return (2.0 / (1+np.exp(-2*x)) - 1)
        elif (self.FUN=='sigmoid'):
            return (1.0/(1+np.exp(-x)))
        else:
            return(x)
        
    def derivar(self,x):
        if (self.FUN=='tanh'):
            return (1-x**2)
        elif (self.FUN=='sigmoid'):
            return (x*(1-x))
        else:
            return(1)    

    def predict_nOut(self, X):
        """Return class label after unit step"""
        return self.evaluar(self.net_input(X))
    
    def predict(self, X):
        """Retorna un entero con el índice de la clase más probable """
        y_hat = self.predict_nOut(X)
        if (self.FUN=='tanh'):
            return (2*(y_hat>0)*1-1)
        elif (self.FUN=='sigmoid'):
            return ((y_hat>0.5)*1)
        else:
            return(X)
            
    def accuracy(self, X, y):
        y_hat = self.predict(X)
        OK = np.sum(y_hat==y)
        return (OK/X.shape[0])
        
    
    