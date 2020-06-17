import sys
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D
from pylab import *




def multip_int(mod, n):
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    def exp_cmplx(i):
        return(np.exp((2j * np.pi * i / mod) + 1j * np.pi /2))
    for k in range(mod):
        r = (k * n)%mod
        plt.plot((np.real(exp_cmplx(k)), np.real(exp_cmplx(r))), (np.imag(exp_cmplx(k)), np.imag(exp_cmplx(r))), linewidth=.5, color='black')

def multi_float(inf, sup, pas, mod):
    for k in range(int((sup - inf)/pas)):
        plt.clf()
        u = inf + k * pas
        multip_int(mod, u)
        plt.pause(.005)
    plt.show()


def multip_int_FPS(mod, n, k):
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    def exp_cmplx(i):
        return(np.exp((2j * np.pi * i / mod) + 1j * np.pi /2))

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    for i in range(mod):
        r = (i * n)%mod
        plt.plot((np.real(exp_cmplx(i)), np.real(exp_cmplx(r))), (np.imag(exp_cmplx(i)), np.imag(exp_cmplx(r))), linewidth=1, color='black')

    plt.savefig(r'C:\Users\alpha\OneDrive\Bureau\Informatique\divers\images\compilees\im%06d.png' % k)
    plt.close(fig)

def multi_float_FPS(inf, sup, pas, mod):  #à n' executer qu'une fois
    l = int((sup - inf)/pas)
    for k in range(l):

        u = inf + k * pas
        multip_int_FPS(mod, u, k)
        print(k)

    fps = 25
    frame_width, frame_height = 640, 480
    vid = cv2.VideoWriter(r'C:\Users\alpha\OneDrive\Bureau\Informatique\launay_tables\video_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    for k in range(l):
        frame = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\divers\images\compilees\im%06d.png' % k)
        vid.write(frame)
        print(k)
    vid.release()




