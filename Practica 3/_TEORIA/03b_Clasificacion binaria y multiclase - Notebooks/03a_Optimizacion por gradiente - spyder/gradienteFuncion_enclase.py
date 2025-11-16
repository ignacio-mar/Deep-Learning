import numpy as np
import grafica_Grad as gr
import math

# z = 3*x^2 + y^2;
[x, y, h] = gr.graficoGradientePy(1)
z = 3*x**2 + y**2

PtoAnt = [x, y, z]   # guardamos la posición para dibujar

# cambiamos x e y (nos movemos)
x = x-1  
y = y-2

# evaluamos la función en la nueva ubicación
z = 3*x**2 + y**2;

# dibujamos una línea uniendo la posición anterior y la actual
gr. graficarPaso(PtoAnt, [x, y, z], h)
    
    
    
    
    


