"""
Gabriela Zdrojewska
Projekt nr 1 - Eksplozja
Inferencja położenia wybuchu
"""

import numpy as np
import sys
import eksplozja_dane as dane
import matplotlib.pyplot as plt
from class_variable import variable
from class_potential import potential

N = 10 #ilość punktów pomiarowych 
punkt = dane.losuj_pkt_pom((N))
e = dane.losuj_pkt_ekspl()
print("\nMiejsce wybuchu: (%.3f, %.3f)" %(e[0], e[1]))

sigma2 = 0.2 #odchylenie standardowe 
pomiar = dane.obserw(punkt, e, N, sigma2)

print("\nDane do inferencji")
for i in range(N):
    print('punkt %d: (%.3f, %.3f)' %(i+1, punkt[i][0], punkt[i][1]),
    '\tsygnal %.3f'%(pomiar[i]))

punkty = np.array(punkt)
e_x = 0
e_y = 1
delta = 0.1
stany = np.arange(-1, 1.001, delta)
ile_stanow = len(stany)

print('\n\nStanow dla wspolrzednej wybuchu jest', ile_stanow)
print('Tablica ze współrzędnymi:\n', stany)

zmienna = [variable(None, None) for i in range(2)]
zmienna[e_x].name = 'e_x'; zmienna[e_x].domain = stany
zmienna[e_y].name = 'e_y'; zmienna[e_y].domain = stany

pot = potential()
pot.variables = np.array([e_x, e_y])
table = np.zeros((ile_stanow, ile_stanow))
stala_1 = 1/(2*sigma2)
stala_2 = np.sqrt(np.pi*stala_1)
for j in range(ile_stanow):
    for k in range(ile_stanow):
        rob_e=[-1+delta*j,-1+delta*k]
        p=1; odl_e=dane.odl_do_kwad(rob_e, [0,0])
        if (odl_e<1.0):
            rob_d2=[dane.odl_do_kwad(punkt[i], rob_e) for i in range(N)]
            d2=[1/(rob_d2[i]+0.1) for i in range(N)]
            rob_p=[np.exp((-1/stala_1)*(pomiar[i]-d2[i])**2)/stala_2 for i in range(N)]
            for i in range(N):
                p=p*rob_p[i]
        else:
            p=0
        table[j,k]=p

pot.table = table
suma = np.sum(pot.table)
pot.table = pot.table/suma
suma=np.sum(pot.table)
maxi = np.amax(pot.table)
argument_maxi = np.argmax(pot.table)
indeksy = np.unravel_index(np.argmax(pot.table), pot.table.shape)
esty_ex = -1 + delta * indeksy[0]
esty_ey = -1 + delta * indeksy[1]

print('\n\nMaksimum wynosi %.3f' %(maxi),'dla', indeksy)
print("Estymacja punktu: (%.3f," %(esty_ex) + " %.3f)\n" %(esty_ey))

#Wykres:
x = np.arange(-1, 1.001, delta)
y = np.arange(-1, 1.001, delta)
X,Y = np.meshgrid(x,y)
plt.figure(num=1, figsize=(8,8))
level = 25
plt.contourf(Y, X, pot.table, level)
plt.colorbar()
plt.plot(e[0], e[1], color = 'darkgreen', label = 'rzeczywiste epicentrum')
plt.plot(esty_ex, esty_ey, 'x', color = 'deeppink', label = 'zarejestrowane epicentrum')
plt.plot(punkty[:,0], punkty[:,1], 'ro', label = 'punkty pomiarowe')
plt.legend(loc = 'upper right')
plt.ylabel('y')
plt.xlabel('x')
tytul = "Wybuch w: [" + str(round(e[0],2)) + ", " + str(round(e[1],2)) + ("], rozpoznany w: [%.2f " %esty_ex) + (", %.2f]" %esty_ey)
plt.title(tytul)
plt.show()