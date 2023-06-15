#----------------------------------------------
#Garcia Junquera Luis Eduardo
#Agrupamiento usando conjuntos de particulas
#------------------------------------------------

import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm

#------------------------
#Lee los datos
#(hoja de datos panda)
#-------------------------
plot = True
#--------------------------------------------------
#Pasar a un arreglo de numpy (dataframe a numpy)
#--------------------------------------------------
data_points = pd.read_csv('iris.txt', sep=',', header=None)

clusters = data_points[4].values
#-----------------------------------------------------------
#Remover columna 4 de los datos (metoddo drop de pandas)
#-----------------------------------------------------------
data_points = data_points.drop([4], axis=1)
#--------------------------------------------------------------
# Usar columna 0 y 1 como(x,y) para puntos en 2D
#-----------------------------------------------------------------
if plot:
    data_points = data_points[[0, 1]]
#--------------------------------
#convierte a areglo  2d numpy
#--------------------------------

data_points = data_points.values
#----------------------
#Algoritmo PSO-clustering
#----------------------
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)

#---------------------------------------------
# Mapeo de colores o elementos de los grupos
#----------------------------------------------
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
print('Actual classes = ', clusters)
