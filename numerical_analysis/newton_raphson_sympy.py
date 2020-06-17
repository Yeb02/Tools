import sys
import numpy as np
from sympy import sin, cos, Matrix
from sympy.abc import rho, phi
import os
from sympy import *
import random


# http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html

def Newton_Raphson(points, indices, u, v, x1, y1, vecteurs):  # utiliser des pts des images d'avant pour obtenir la constante de distance.
    t1 = []

    if len(indices) != 0:  #○ faire les minis-batches  ♀♂
        for i in indices:
            t1 += [vecteurs[-1][i]]
        l = len(indices)
    else:
        t1 += [1]
        l = 1

    t1 += [random.random() for k in range(n - l)]
    t2 = [random.random() for k in range(n)]  # variables d'intersection, à priori positives.
    phi, theta, psi = 0, 0, 0 #angles d'euler de cam2 dans la base de cam1, l' objectif regarde les +z
    X2, Y2, Z2 = 0, 0, 0 #coo de cam 2 par rapport à cam 1 dans la base de cam1

    def descente_de_gradient(Xn0, x1, y1, u, v, l, t1):
        Xn = sympy.IndexedBase('Xn')
        J = Jacobien(Xn, x1, y1, u, v, l, t1)
        epsilon = .5
        Xn1 = [0] * (2*n + 6)  #doit etre (tres ?) different de Xn0.
        cpt = 0
        while sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))) > epsilon and cpt < 20:
            # print(Xn0, len(Xn0), type(Xn0), sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))))
            print(sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))))
            if cpt != 0:
                Xn0 = list(Xn1)
            J1 = J.subs([(Xn[i], Xn0[i]) for i in range(l, 2*n + 6, 1)])
            taille = J1.shape
            J2 = np.zeros(taille)
            for i in range(taille[0]):
                for j in range(taille[1]):
                    J2[i,j]=sympy.N(J1[i,j])
            J2 = np.matrix(J2)
            J3 = (((J2.T).dot(J2)).I).dot(J2.T)
            fXn = f(Xn, x1, y1, u, v, l, t1)
            fXn = [fXn[j].subs([(Xn[i], Xn0[i]) for i in range(l, 2*n + 6, 1)]) for j in range(len(fXn))]
            Xn1 = t1[:l] + list(np.array(Xn0[l::] - J3.dot(fXn))[0])
            cpt += 1
        return(Xn0)

    Xn0 = t1 + t2
    Xn0 += [phi, theta, psi, X2, Y2, Z2]

    return descente_de_gradient(Xn0, x1, y1, u, v, l, t1)


def f(Xn, x1, y1, u, v, l, t1):
    n = len(x1)
    fct = [0] * 3 * n


    for i in range(l): #Vecteur colonne de f

        fct[3*i] = t1[i]*x1[i] - Xn[2*n + 3] - Xn[n + i]*( u[i] * (cos(Xn[2*n +2])*cos(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-cos(Xn[2*n +2])*sin(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) + sin(Xn[2*n +2])*sin(Xn[2*n + 1])  )

        fct[3*i + 1] = t1[i]*y1[i] - Xn[2*n + 4] - Xn[n + i]*( u[i] * (sin(Xn[2*n +2])*cos(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-sin(Xn[2*n +2])*sin(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) - cos(Xn[2*n +2])*sin(Xn[2*n +1])  )

        fct[3*i + 2] = t1[i] - Xn[2*n + 5] - Xn[n + i]*( u[i] * sin(Xn[2*n +1])*sin(Xn[2*n]) + v[i] * sin(Xn[2*n +1]) * cos(Xn[2*n]) + cos(Xn[2*n +1]) )


    for i in range(l, n, 1): #Vecteur colonne de f

        fct[3*i] = Xn[i]*x1[i] - Xn[2*n + 3] - Xn[n + i]*( u[i] * (cos(Xn[2*n +2])*cos(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-cos(Xn[2*n +2])*sin(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) + sin(Xn[2*n +2])*sin(Xn[2*n + 1])  )

        fct[3*i + 1] = Xn[i]*y1[i] - Xn[2*n + 4] - Xn[n + i]*( u[i] * (sin(Xn[2*n +2])*cos(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-sin(Xn[2*n +2])*sin(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) - cos(Xn[2*n +2])*sin(Xn[2*n +1])  )

        fct[3*i + 2] = Xn[i] - Xn[2*n + 5] - Xn[n + i]*( u[i] * sin(Xn[2*n +1])*sin(Xn[2*n]) + v[i] * sin(Xn[2*n +1]) * cos(Xn[2*n]) + cos(Xn[2*n +1]) )
    return fct


def Jacobien(Xn, x1, y1, u, v, l, t1):  #Vrai_Xn : (t11,t21,t12,t22...t2n,phi,theta,psi,X2,Y2,Z2) en valeurs numériques
    n = len(x1)
    fct = f(Xn, x1, y1, u, v, l, t1)
    F = Matrix([fct[i] for i in range(3*n)])
    X_n = Matrix([Xn[i] for i in range(l, 2*n + 6, 1)])
    return F.jacobian(X_n)