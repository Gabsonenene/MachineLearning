"""
Przygotowanie danych potrzebnych do przeprowadzenia eksperymentu eksplozja
"""

import numpy as np
import sys

#Funkcja służy do obliczenia kwadratu odległości między dwoma punktami w dwóch wymiarach:
def odl_do_kwad(a, b, dim = 2):
    rob = [(a[i] - b[i])**2 for i in range(dim)]
    return sum(rob)

#Funkcja słyżu do losowanie wspólrzędnych N punktów na okręgu o r=1:
def losuj_pkt_pom(N):
    losuj = []
    for i in range(0,N):
        losuj.append(np.random.rand()*np.pi*2)
    katy = sorted(losuj)
    x = np.array([np.sin(katy[i]) for i in range(N)])
    y = np.array([np.cos(katy[i]) for i in range(N)])
    punkty_pomiarowe = []
    for i in range(0,N):
        rob = [x[i], y[i]]
        punkty_pomiarowe.append(rob)
    return punkty_pomiarowe

#Funkcja służy do losowanie miejsca wybuchu dla e = [ex, ey]:
def losuj_pkt_ekspl():
    ex = 2
    ey = 2
    odleglosc_e = odl_do_kwad([ex, ey], [0,0])
    while (odleglosc_e > 1):
        ex = np.random.rand()
        ey = np.random.rand()
        if (np.random.rand() < 0.5):
            ex = -ex
        if (np.random.rand() < 0.5):
            ey = -ey
        odleglosc_e = odl_do_kwad([ex, ey], [0,0])
    e = [ex, ey]
    return e

#Zaburzenie danych obserwacyjnych:
def obserw(punkty, e, N, sigma2):
    d2 = [odl_do_kwad(punkty[i], e) for i in range(N)]
    obserwacja = [1/(d2[i]+0.1) for i in range(N)]
    zaburzenie = [obserwacja[i] + np.random.normal(0,sigma2) for i in range(N)]
    if __name__ == '__main__':
        for i in range(N):
            print('punkt %d: (%.3f, %.3f)'%(i+1, punkt[i][0], punkt[i][1]), '\tsygnal czysty %.3f'%(obserwacha[i]), '\tpo zaburzeniu:%.3f'%(zaburzenie[i]))
    return zaburzenie
