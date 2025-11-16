import grafica_Grad as gr
import math
# (2,3), (1,1), (-1,-3)
X = [2, 1, -1]
T = [3, 1, -3]

[w0, w1, dibu] = gr.graficoGradientePy(4)

alfa = 0.05
MAX_ITE = 5000  
COTA = 0.0001
ite = 0
E_ant = 1
E = (1/3)*((3-2*w1-w0)**2+(1-w1-w0)**2+(-3+w1-w0)**2)

while ((ite<MAX_ITE) and (math.fabs(E_ant - E) > COTA)):
    for p in range(len(X)):
        E_ant=E
        PtoAnt = [w0, w1, E]
        y = w1 * X[p] + w0
        Error = T[p]-y
        
        grad_w0 = -2*Error
        grad_w1 = -2*Error*X[p]
    
        w0 = w0 - alfa * grad_w0
        w1 = w1 - alfa * grad_w1
        E = (1/3)*((3-2*w1-w0)**2+(1-w1-w0)**2+(-3+w1-w0)**2)
       
        gr.graficarPaso(PtoAnt, [w0, w1, E], dibu)
    ite = ite + 1
    print ("ite= %d   w0= %8.5f   w1=%8.5f   E=%.8f" % (ite,w0,w1,E)) 
