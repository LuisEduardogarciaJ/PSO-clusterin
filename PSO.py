# -*- coding: utf-8 -*-
"""Untitled31.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RPpldcWPEOzxrYAUsNEF4KfWwDqHeoVO
"""

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

#---------------------------------
#Clase particle
#---------------------------------
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


class Particle:
    def __init__(self, n_clusters, data, use_kmeans=True, w=0.72, c1=1.49, c2=1.49):
        self.n_clusters = n_clusters
        if use_kmeans:
            k_means = KMeans(n_clusters=self.n_clusters)
            k_means.fit(data)
            self.centroids_pos = k_means.cluster_centers_
        else:
            self.centroids_pos = data[np.random.choice(list(range(len(data))), self.n_clusters)]

        #---------------------------------------------------------------------------------
        #Cada agrupamiento tiene un centoide que es el punto que lo representa
        #se asignan k datos aleatorios a k centroides
        #---------------------------------------------------------------------------------
        self.pb_val = np.inf
        #-----------------------------------------------------------------
        # Mejor posicion personal para todos los centroides hasta aqui
        #-----------------------------------------------------------------
        self.pb_pos = self.centroids_pos.copy()
        #--------------------------------------------------------------------------
        #Parametro del PSO(particle swarm optimization)
        #(optimizacion usando enjambres de particulas)
        #--------------------------------------------------------------------------
        self.velocity = np.zeros_like(self.centroids_pos)

        self.pb_clustering = None
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_pb(self, data: np.ndarray):
        """
        Actualizacion el mejor puntaje basando en la funcion de aptitud(ecuacion 4)
        """
        # --------------------------------------------------------------------
        #Encuentra los datos (puntos) que permanecen a cada agrupamiento
        #utilizando distancias a los centroides
        #---------------------------------------------------------------------
        distances = self._get_distances(data=data)
        #---------------------------------------------------------------------------------------------
        #La distancia minima entre los datos y un centroide indica que permanece a ese agrupamiento
        #---------------------------------------------------------------------------------------------
        clusters = np.argmin(distances, axis=0)
        clusters_ids = np.unique(clusters)
        while len(clusters_ids) != self.n_clusters:
            deleted_clusters = np.where(np.isin(np.arange(self.n_clusters), clusters_ids) == False)[0]
            self.centroids_pos[deleted_clusters] = data[np.random.choice(list(range(len(data))), len(deleted_clusters))]
            distances = self._get_distances(data=data)
            clusters = np.argmin(distances, axis=0)
            clusters_ids = np.unique(clusters)

        new_val = self._fitness_function(clusters=clusters, distances=distances)
        if new_val < self.pb_val:
            self.pb_val = new_val
            self.pb_pos = self.centroids_pos.copy()
            self.pb_clustering = clusters.copy()

    def update_velocity(self, gb_pos: np.ndarray):
        """
        Actualiza la nueva velocidad en función de la velocidad actual, la mejor posición personal hasta el momento y la mejor del enjambre (global)
         posición hasta ahora.
         :param gb_pos: vector de las mejores posiciones centroides entre todas las partículas hasta el momento
         :devolver:
        """
        self.velocity = self.w * self.velocity + \
                        self.c1 * np.random.random() * (self.pb_pos - self.centroids_pos) + \
                        self.c2 * np.random.random() * (gb_pos - self.centroids_pos)

    def move_centroids(self, gb_pos):
        self.update_velocity(gb_pos=gb_pos)
        new_pos = self.centroids_pos + self.velocity
        self.centroids_pos = new_pos.copy()

    def _get_distances(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclidiana entre los datos y los centroides
         :datos de parámetro:
         :retorno: distancias: una serie de distancias numpy (len(centroides) x len(datos))
        """
        distances = []
        for centroid in self.centroids_pos:
            # calcula la distancia euclidiana --> raiz de la suma de los cuadrados
            d = np.linalg.norm(data - centroid, axis=1)
            distances.append(d)
        distances = np.array(distances)
        return distances

    def _fitness_function(self, clusters: np.ndarray, distances: np.ndarray) -> float:
        """
        Calcula la función de fitness (Ecuación 4)
         i es el índice de partícula
         j es el índice de agrupaciones en la partícula i
         p es el vector de los índices de datos de entrada que pertenecen al grupo [ij]
         z[p] es el vector de los datos de entrada pertenecientes al clúster[ij]
         d es un vector de distancias entre z(p) y el centroide j
         :param clústeres:
         :param distancias:
         :regresar: J:"""
        J = 0.0
        for i in range(self.n_clusters):
            p = np.where(clusters == i)[0]
            if len(p):
                d = sum(distances[i][p])
                d /= len(p)
                J += d
        J /= self.n_clusters
        return J

#-------------------------------------------------------
#Clase de optimizacion usando enjambres de particulas
#-------------------------------------------------------
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from particle import Particle


class PSOClusteringSwarm:
    def __init__(self, n_clusters: int, n_particles: int, data: np.ndarray, hybrid=True, w=0.72, c1=1.49, c2=1.49):
        """
        Inicializa el enjambre.
         :param n_clusters: número de clústeres
         :param n_particles: número de partículas
         :param data: (numero_de_puntos x dimensiones)
         :param hybrid: bool : usar o no kmsignifica como inicialización
         :parámetro w:
         :parámetro c1:
         :parámetro c2:
         """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data

        self.particles = []
        # guarda el mejor enjambre
        self.gb_pos = None
        self.gb_val = np.inf
        # el mejor agrupamiento hasta aqui
        # para cada dato contiene el numero de agrupamiento
        self.gb_clustering = None

        self._generate_particles(hybrid, w, c1, c2)

    def _print_initial(self, iteration, plot):
        print('*** Initialing swarm with', self.n_particles, 'PARTICLES, ', self.n_clusters, 'CLUSTERS with', iteration,
              'MAX ITERATIONS and with PLOT =', plot, '***')
        print('Data=', self.data.shape[0], 'points in', self.data.shape[1], 'dimensions')

    def _generate_particles(self, hybrid: bool, w: float, c1: float, c2: float):
        """
        Genera particulas con k agrupamientos y puntos en t dimensiones :return:
        """
        for i in range(self.n_particles):
            particle = Particle(n_clusters=self.n_clusters, data=self.data, use_kmeans=hybrid, w=w, c1=c1, c2=c2)
            self.particles.append(particle)

    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.pb_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        """
         :param plot: = True trazará los mejores grupos de datos globales
         :param iteración: número de iteraciones máximas
         :return: (mejor grupo, mejor valor de fitness)
        """
        self._print_initial(iteration, plot)
        progress = []
        # Iterar hasta la iteración máxima
        for i in range(iteration):
            if i % 200 == 0:
                clusters = self.gb_clustering
                print('iteration', i, 'GB =', self.gb_val)
                print('best clusters so far = ', clusters)
                if plot:
                    centroids = self.gb_pos
                    if clusters is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=clusters, cmap='viridis')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:  # si aún no hay grupos ( iteración = 0 ) trace los datos sin grupos
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_pb(data=self.data)
                self.update_gb(particle=particle)

            for particle in self.particles:
                particle.move_centroids(gb_pos=self.gb_pos)
            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])

        print('Finished!')
        return self.gb_clustering, self.gb_val