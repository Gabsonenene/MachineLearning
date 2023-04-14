"""
Gabriela Zdrojewska
Projekt nr 1 - Klinika
"""

import numpy as np
from class_potential import potential
from class_variable import variable

#Liczba zmiennych losowych
N = 8

#Pusta lista zmiennych
clinic_variables = [variable(None, None) for i in range(N)]

#Kategoryzowanie zmiennych
x_ray = 7
dyspnea = 6
either_t_or_lc = 5
tuberculosis = 4
lung_cancer = 3
bronchitis = 2
visited_asia = 1
smoker = 0

#Kategoryzowanie stanów zmiennych
tr = 1
fa = 0

#Wypełnianie listy zgodnie z parametrami problemu
clinic_variables[x_ray].name = 'X-ray'
clinic_variables[x_ray].domain = ['false', 'true']
clinic_variables[dyspnea].name = 'Dyspnea'
clinic_variables[dyspnea].domain = ['false', 'true']
clinic_variables[either_t_or_lc].name = 'Either Tuberculosis or Lung Cancer'
clinic_variables[either_t_or_lc].domain = ['false', 'true']
clinic_variables[tuberculosis].name = 'Tuberculosis'
clinic_variables[tuberculosis].domain = ['false', 'true']
clinic_variables[lung_cancer].name = 'Lung cancer'
clinic_variables[lung_cancer].domain =['false', 'true']
clinic_variables[bronchitis].name = 'Bronchitis'
clinic_variables[bronchitis].domain = ['false', 'true']
clinic_variables[visited_asia].name = 'Visited Asia'
clinic_variables[visited_asia].domain = ['false', 'true']
clinic_variables[smoker].name = 'Smoker'
clinic_variables[smoker].domain = ['false', 'true']

print("\nUtworzona lista zmiennych:")
for i in range(N):
    print('\t random variable ', clinic_variables[i].name,
    "with domain: ", clinic_variables[i].domain)

#Potencjały
clinic_potentials = [potential() for i in range(N)]

print('\nPusta lista dla potencjałów:')
for i in range(N):
    print('\t', i, clinic_potentials[i].variables, clinic_potentials[i].table)

print("\n\nTablice prawdopodobieństw:")

clinic_potentials[x_ray].variables = np.array([x_ray, either_t_or_lc])
table = np.zeros((2,2))
table[tr, tr] = 0.98
table[tr, fa] = 0.05
clinic_potentials[x_ray].table = table
clinic_potentials[x_ray].table[fa][:][:]=1-clinic_potentials[x_ray].table[tr][:][:]
print("\nPrawdopodobieństwa x_ray w zależności od stanu either_t_or_lc:")
print('(', clinic_variables[x_ray].name, '=', clinic_variables[x_ray].domain[1], '|', clinic_variables[either_t_or_lc].name,'=', clinic_variables[either_t_or_lc].domain[1],') =', clinic_potentials[x_ray].table[1][1])
print('(', clinic_variables[x_ray].name, '=', clinic_variables[x_ray].domain[1], '|', clinic_variables[either_t_or_lc].name,'=', clinic_variables[either_t_or_lc].domain[0],') =', clinic_potentials[x_ray].table[1][0])
print('(', clinic_variables[x_ray].name, '=', clinic_variables[x_ray].domain[0], '|', clinic_variables[either_t_or_lc].name,'=', clinic_variables[either_t_or_lc].domain[1],') =', clinic_potentials[x_ray].table[0][1])
print('(', clinic_variables[x_ray].name, '=', clinic_variables[x_ray].domain[0], '|', clinic_variables[either_t_or_lc].name,'=', clinic_variables[either_t_or_lc].domain[0],') =', clinic_potentials[x_ray].table[0][0])

clinic_potentials[dyspnea].variables = np.array([dyspnea, either_t_or_lc, bronchitis]) 
table = np.zeros((2,2,2))
table[tr, fa, fa] = 0.1
table[tr, fa, tr] = 0.8
table[tr, tr, fa] = 0.7
table[tr, tr, tr] = 0.9
clinic_potentials[dyspnea].table = table
clinic_potentials[dyspnea].table[fa][:][:]=1-clinic_potentials[dyspnea].table[tr][:][:]
print("\nPrawdopodobieństwa dyspnea w zależności od stanów zmiennych either_t_or_lc i bronchitis:")
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[1], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], ') = ', clinic_potentials[dyspnea].table[1][1][1])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[1], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], ') = ', clinic_potentials[dyspnea].table[1][0][1])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[1], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], ') = ', clinic_potentials[dyspnea].table[1][1][0])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[1], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], ') = ', clinic_potentials[dyspnea].table[1][0][0])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[0], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], ') = ', clinic_potentials[dyspnea].table[0][1][1])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[0], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], ') = ', clinic_potentials[dyspnea].table[0][0][1])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[0], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], ') = ', clinic_potentials[dyspnea].table[0][1][0])
print('(', clinic_variables[dyspnea].name, '=', clinic_variables[dyspnea].domain[0], '|', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], ') = ', clinic_potentials[dyspnea].table[0][0][0])

clinic_potentials[either_t_or_lc].variables = np.array([either_t_or_lc, tuberculosis, lung_cancer]) 
table = np.zeros((2,2,2))
table[tr, fa, fa] = 0
table[tr, fa, tr] = 1
table[tr, tr, fa] = 1
table[tr, tr, tr] = 1
clinic_potentials[either_t_or_lc].table = table
clinic_potentials[either_t_or_lc].table[fa][:][:]=1-clinic_potentials[either_t_or_lc].table[tr][:][:]
print("\nPrawdopodobieństwa either_t_or_lc w zależności od stanów zmiennych tuberculosis i lung_cancer:")
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], ') = ', clinic_potentials[either_t_or_lc].table[1][1][1])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], ') = ', clinic_potentials[either_t_or_lc].table[1][0][1])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], ') = ', clinic_potentials[either_t_or_lc].table[1][1][0])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[1], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], ') = ', clinic_potentials[either_t_or_lc].table[1][0][0])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], ') = ', clinic_potentials[either_t_or_lc].table[0][1][1])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], ') = ', clinic_potentials[either_t_or_lc].table[0][0][1])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], ') = ', clinic_potentials[either_t_or_lc].table[0][1][0])
print('(', clinic_variables[either_t_or_lc].name, '=', clinic_variables[either_t_or_lc].domain[0], '|', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], ') = ', clinic_potentials[either_t_or_lc].table[0][0][0])

clinic_potentials[tuberculosis].variables = np.array([tuberculosis, visited_asia])
table = np.zeros((2,2))
table[tr, tr] = 0.05
table[tr, fa] = 0.01
clinic_potentials[tuberculosis].table = table
clinic_potentials[tuberculosis].table[fa][:][:]=1-clinic_potentials[tuberculosis].table[tr][:][:]
print("\nPrawdopodobieństwa tuberculosis w zależności od stanu visited_asia:")
print('(', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[visited_asia].name,'=', clinic_variables[visited_asia].domain[1],') =', clinic_potentials[tuberculosis].table[1][1])
print('(', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[1], '|', clinic_variables[visited_asia].name,'=', clinic_variables[visited_asia].domain[0],') =', clinic_potentials[tuberculosis].table[1][0])
print('(', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[visited_asia].name,'=', clinic_variables[visited_asia].domain[1],') =', clinic_potentials[tuberculosis].table[0][1])
print('(', clinic_variables[tuberculosis].name, '=', clinic_variables[tuberculosis].domain[0], '|', clinic_variables[visited_asia].name,'=', clinic_variables[visited_asia].domain[0],') =', clinic_potentials[tuberculosis].table[0][0])

clinic_potentials[lung_cancer].variables = np.array([lung_cancer, smoker])
table = np.zeros((2,2))
table[tr, tr] = 0.1
table[tr, fa] = 0.01
clinic_potentials[lung_cancer].table = table
clinic_potentials[lung_cancer].table[fa][:][:]=1-clinic_potentials[lung_cancer].table[tr][:][:]
print("\nPrawdopodobieństwa lung_cancer w zależności od stanu smoker:")
print('(', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[1],') =', clinic_potentials[lung_cancer].table[1][1])
print('(', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[1], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[0],') =', clinic_potentials[lung_cancer].table[1][0])
print('(', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[1],') =', clinic_potentials[lung_cancer].table[0][1])
print('(', clinic_variables[lung_cancer].name, '=', clinic_variables[lung_cancer].domain[0], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[0],') =', clinic_potentials[lung_cancer].table[0][0])

clinic_potentials[bronchitis].variables = np.array([bronchitis, smoker])
table = np.zeros((2,2))
table[tr, tr] = 0.6
table[tr, fa] = 0.3
clinic_potentials[bronchitis].table = table
clinic_potentials[bronchitis].table[fa][:][:]=1-clinic_potentials[bronchitis].table[tr][:][:]
print("\nPrawdopodobieństwa bronchitis w zależności od stanu smoker:")
print('(', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[1],') =', clinic_potentials[bronchitis].table[1][1])
print('(', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[1], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[0],') =', clinic_potentials[bronchitis].table[1][0])
print('(', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[1],') =', clinic_potentials[bronchitis].table[0][1])
print('(', clinic_variables[bronchitis].name, '=', clinic_variables[bronchitis].domain[0], '|', clinic_variables[smoker].name,'=', clinic_variables[smoker].domain[0],') =', clinic_potentials[bronchitis].table[0][0])

clinic_potentials[visited_asia].variables = np.array([visited_asia])
table = np.zeros(2)
table[tr] = 0.01
table[fa] = 0.99
clinic_potentials[visited_asia].table = table
print("\nVisited_asia z aproiri prawdopodobieństwa bycia w Azji:")
print('(', clinic_variables[visited_asia].name, '=', clinic_variables[visited_asia].domain[0],') =', clinic_potentials[visited_asia].table[0])
print('(', clinic_variables[visited_asia].name, '=', clinic_variables[visited_asia].domain[1],') =', clinic_potentials[visited_asia].table[1])

clinic_potentials[smoker].variables = np.array([smoker])
table = np.zeros(2)
table[tr] = 0.5
table[fa] = 0.5
clinic_potentials[smoker].table = table
print("\nSmoker z aproiri prawdopodobieństwem palenia:")
print('(', clinic_variables[smoker].name, '=', clinic_variables[smoker].domain[0],') =', clinic_potentials[smoker].table[0])
print('(', clinic_variables[smoker].name, '=', clinic_variables[smoker].domain[1],') =', clinic_potentials[smoker].table[1])


#Prawdopodobieństwo łączne
multpot = potential()
multpot.variables = np.array([dyspnea, either_t_or_lc, tuberculosis, lung_cancer, bronchitis, visited_asia, smoker])
table = np.zeros((2,2,2,2,2,2,2))
print("\n\nPrawdopodobieństwo łączne:")

st = [tr, fa]
for i in st:#d
    for j in st:#e
        for k in st:#b
            for l in st:#l
                for m in st:#t
                    for n in st:#s
                        for o in st: #a
                            table[i, j, k, l, m, n, o] = clinic_potentials[dyspnea].table[i,j,k]\
                                *clinic_potentials[either_t_or_lc].table[j,m,l]\
                                *clinic_potentials[tuberculosis].table[m,o]\
                                *clinic_potentials[lung_cancer].table[l,n]\
                                *clinic_potentials[bronchitis].table[k,n]\
                                *clinic_potentials[smoker].table[n]\
                                *clinic_potentials[visited_asia].table[o]
                            #print("Prawdopodobieństwo d = {}, gdy e = {}, b = {},  l = {}, t = {}, s = {}, a = {} wynosi: {:5f} ".format(i, j, k, l, m, n, o, table[i, j, k, l, m, n, o]))
multpot.table = table


#Obliczenia
print("\nWyniki obliczeń:")

#p(d)
numerator = 0
for i in st:
    for j in st:
        for k in st:
            for l in st:
                for m in st:
                    for n in st:
                        numerator += multpot.table[tr, i, j, k, l, m, n]                      
print("Prawdopodbieństwo p(d) = %.5f"%numerator)

#p(d|s=1)
numerator = 0
for i in st:
    for j in st:
        for k in st:
            for l in st:
                for m in st:
                    numerator += multpot.table[tr, i, j, k, l, tr, m] 

denumerator = 0
for i in st:
    for j in st:
        for k in st:
            for l in st:
                for m in st:
                    for n in st:
                        denumerator += multpot.table[i, j, k, l, m, tr, n]
print("Prawdopodobieństwo wystąpienia duszności, jeżeli osoba była palaczem: p(d|s=1) = %.5f"%numerator,'/%.5f'%denumerator,' = %.5f'%(numerator/denumerator))

#p(d|s=0)
numerator = 0
for i in st:
    for j in st:
        for k in st:
            for l in st:
                for m in st:
                    numerator += multpot.table[tr, i, j, k, l, fa, m] 

denumerator = 0
for i in st:
    for j in st:
        for k in st:
            for l in st:
                for m in st:
                    for n in st:
                        denumerator += multpot.table[i, j, k, l, m, fa, n]  
print("Prawdopodobieństwo wystąpienia duszności, jeżeli osoba nie była palaczem: p(d|s=0) = %.5f"%numerator,'/%.5f'%denumerator,' = %.5f'%(numerator/denumerator))

#Test sprawdzający, czy p(t,s|d) jest równe p(t|d)*p(s|d) dla różnych wartości t, s, d
print('\nCzy t jest niezależne od s przy zadanym d?')
ptsd=0
for i in st:#x
    for j in st:#b
        for k in st:#e
            for l in st:#l
                for m in st:#a                    
                    for n in [1]:#s
                        for o in [1]:#t
                            for p in [1]:#d
                                ptsd += clinic_potentials[visited_asia].table[m]\
                                *clinic_potentials[either_t_or_lc].table[k,l,o]\
                                *clinic_potentials[lung_cancer].table[l,n]\
                                *clinic_potentials[bronchitis].table[k,n]\
                                *clinic_potentials[x_ray].table[i,k]\
                                *clinic_potentials[smoker].table[n]\
                                *clinic_potentials[dyspnea].table[p,k,j]\
                                *clinic_potentials[tuberculosis].table[o,m]
print('p(t,s|d) = ', '%.5f'%ptsd)

ptd=0
for i in st:#x
    for j in st:#b
        for k in st:#e
            for l in st:#l
                for m in st:#a                    
                    for n in st:#s
                        for o in [1]:#t
                            for p in [1]:#d
                                ptd += clinic_potentials[visited_asia].table[m]\
                                *clinic_potentials[either_t_or_lc].table[k,l,o]\
                                *clinic_potentials[lung_cancer].table[l,n]\
                                *clinic_potentials[bronchitis].table[k,n]\
                                *clinic_potentials[x_ray].table[i,k]\
                                *clinic_potentials[smoker].table[n]\
                                *clinic_potentials[dyspnea].table[p,k,j]\
                                *clinic_potentials[tuberculosis].table[o,m]
print('p(t|d) =', '%.5f'%ptd)

psd=0
for i in st:#x
    for j in st:#b
        for k in st:#e
            for l in st:#l
                for m in st:#a                    
                    for n in [1]:#s
                        for o in st:#t
                            for p in [1]:#d
                                psd += clinic_potentials[visited_asia].table[m]\
                                *clinic_potentials[either_t_or_lc].table[k,l,o]\
                                *clinic_potentials[lung_cancer].table[l,n]\
                                *clinic_potentials[bronchitis].table[k,n]\
                                *clinic_potentials[x_ray].table[i,k]\
                                *clinic_potentials[smoker].table[n]\
                                *clinic_potentials[dyspnea].table[p,k,j]\
                                *clinic_potentials[tuberculosis].table[o,m]
print('p(s|d) =', '%.5f'%psd)
print('p(t|d)*p(s|d) =', '%.5f'%(ptd*psd))
print('p(t,s|d) != p(t|d)*p(s|d)')
print('%.5f'%ptsd, '!=', '%.5f'%(ptd*psd))

#%%
#Graf
import networkx as nx
import matplotlib.pyplot as plt

print("\nDiGraf zależności pomiędzy potencjałami ")
G = nx.DiGraph()
G.add_nodes_from(['b', 's', 'a', 't', 'l', 'e', 'x', 'd'])
elist = [('b', 'd'), ('t', 'e'), ('e', 'x'), ('s', 'l'), ('a', 't'), ('l', 'e'), ('e', 'd'), ('s', 'b')]
G.add_edges_from(elist)
print("Graf o", G.number_of_nodes(), 'wierzchołkach i o',  G.number_of_edges(), 'krawedziach')

pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels = True, font_size = 35, node_color='b',
        node_size = 2500, arrowsize = 30, arrowstyle = 'fancy',
        node_shape = 'o', width = 2)
