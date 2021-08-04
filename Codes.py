# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 09:52:02 2021

@author: dh.cuellar
"""

import numpy as np
import pulp as lp
import matplotlib.pyplot as plt
import networkx as nx


#Define una semilla para la construccion de los parametros
#np.random.seed(180)

def Carga_Datos():
    
    Na_K = {0:'Arroz', 1:'Papa', 2:'Tomates', 3:'Camisas', 4:'Zapatos', 5:'Cubiertos', 6:'Plancha'}
    
    Na_M = {0:'Hogar', 1:'Exito', 2:'Falabella', 3:'Outlet las Americas',
                   4:'Ktronix', 5:'Carulla', 6:'Zara', 7:'Corabastos', 8:'San Andresito'}
    
    Abre_M = {'Hogar': 'H', 'Exito':'E', 'Falabella':'F', 'Outlet las Americas':'OA', 'Ktronix':'K', 'Carulla':'CA', 
               'Zara': 'Z', 'Corabastos': 'CO', 'San Andresito': 'SA'}
    
    #1: Minimiza ambas
    #2: Minimiza compra
    #3: Minimiza ruta
    
    #Define el tamaño del problema 
    Vertex = 9
    Products = 7
    
    Mk = {0:[1,5,7], 1:[1,5,7], 2:[1,5,7], 3:[1,2,3,6,8], 4:[1,2,3,6,8], 5:[1,2,4],6:[1,2,4] } #Subconjunto de mercados que ofrecen el producto k 
    dk = {k:1 for k in range(Products)} #Demanda del producto k 
    
    fik = {(1,0):3500,(1,1):2500,(1,2):1500,(1,3):40000,(1,4):60000, (1,5):40000,(1,6):55000,
           (2,0):0,   (2,1):0,   (2,2):0,   (2,3):45000,(2,4):65000, (2,5):50000,(2,6):70000,
           (3,0):0,   (3,1):0,   (3,2):0,   (3,3):30000,(3,4):52000, (3,5):0,    (3,6):0, 
           (4,0):0,   (4,1):0,   (4,2):0,   (4,3):0,    (4,4):0,     (4,5):60000,(4,6):39000, 
           (5,0):6000,(5,1):4000,(5,2):2500,(5,3):0,    (5,4):0,     (5,5):0,    (5,6):0,
           (6,0):0,   (6,1):0,   (6,2):0,   (6,3):75000,(6,4):90000,(6,5):0,    (6,6):0, 
           (7,0):2000,(7,1):1000,(7,2):800, (7,3):0,    (7,4):0,     (7,5):0,    (7,6):0, 
           (8,0):0,   (8,1):0,   (8,2):0,   (8,3):30000,(8,4):50000, (8,5):0,    (8,6):0} #Precio de compra del producto k en el mercado i 
    
    qik = {(1,0):1, (1,1):1, (1,2):1, (1,3):1, (1,4):1, (1,5):1, (1,6):1,
           (2,0):0, (2,1):0, (2,2):0, (2,3):1, (2,4):1, (2,5):1, (2,6):1,
           (3,0):0, (3,1):0, (3,2):0, (3,3):1, (3,4):1, (3,5):0, (3,6):0, 
           (4,0):0, (4,1):0, (4,2):0, (4,3):0, (4,4):0, (4,5):1, (4,6):1, 
           (5,0):1, (5,1):1, (5,2):1, (5,3):0, (5,4):0, (5,5):0, (5,6):0,
           (6,0):0, (6,1):0, (6,2):0, (6,3):1, (6,4):1, (6,5):0, (6,6):0, 
           (7,0):1, (7,1):1, (7,2):1, (7,3):0, (7,4):0, (7,5):0, (7,6):0, 
           (8,0):0, (8,1):0, (8,2):0, (8,3):1, (8,4):1, (8,5):0, (8,6):0} #Disponibilidad del producto k en el mercado i 
    
    coor = {0:(20, 9), 1: (7, 15), 2: (17,15),3:(31,13), 4:(23,7),5:(23,11), 6:(17,9), 7:(33,19), 8:(10,6) }
    
    
    coor1 = {'H':(20, 9), 'E': (7, 15), 'F': (17,15), 'OA':(31,13), 'K':(23,7), 'CA':(23,11), 'Z':(17,9), 'CO':(33,19), 'SA':(10,6) }
    
    Cij = {}
    coor_x  = []
    coor_y  = []
    
    for i in range(Vertex):
        for j in range(Vertex):
            if i != j:
                dist = np.sqrt(((coor[i][0]-coor[j][0])**2)+((coor[i][1]-coor[j][1])**2))
                #Cij[i,j]=np.around(np.random.normal(dist, 2),0)
                Cij[i,j] = dist*675.6
        
    #Conjuntos
    V = range(Vertex)
    M = range(1, Vertex)
    K = range(Products)
    
    fwd = {l:[(i,j) for (i,j) in Cij.keys() if i==l] for l in V}
    rev = {l:[(i,j) for (i,j) in Cij.keys() if j==l] for l in V}
    
    return V, M, K, fwd, rev, Cij, fik, qik, Mk, dk, coor1

def Corre_Modelo_Matematico(V, M, K, fwd, rev, Cij, fik, qik, Mk, dk, objetivo):
    
    Na_K = {0:'Arroz', 1:'Papa', 2:'Tomates', 3:'Camisas', 4:'Zapatos', 5:'Cubiertos', 6:'Plancha'}

    Na_M = {0:'Hogar', 1:'Exito', 2:'Falabella', 3:'Outlet las Americas',
               4:'Ktronix', 5:'Carulla', 6:'Zara', 7:'Corabastos', 8:'San Andresito'}

    Abre_M = {'Hogar': 'H', 'Exito':'E', 'Falabella':'F', 'Outlet las Americas':'OA', 'Ktronix':'K', 'Carulla':'CA', 
           'Zara': 'Z', 'Corabastos': 'CO', 'San Andresito': 'SA'}
    

    prob = lp.LpProblem("TPP_MTZ",lp.LpMinimize)
    
    #Variables de decisión
    x=lp.LpVariable.dicts("x_var", [(i,j) for (i,j) in Cij.keys()],lowBound=0,upBound=1,cat="Integer")
    y=lp.LpVariable.dicts("y_var", [i for i in M], lowBound=0,upBound=1,cat="Integer")
    z=lp.LpVariable.dicts("z_var", [(i,k) for (i,k) in fik.keys()],lowBound=0,cat="Continuous")
    u=lp.LpVariable.dicts("u_var", [i for i in M],lowBound=0,upBound=None,cat="Integer")
    
    for k in K:
        prob += sum(z[i,k] for i in Mk[k]) == dk[k], "attend_"+str(k)
        for i in Mk[k]:
            prob += z[i,k] <= qik[i,k]*y[i], "respect_"+str((i,k))
            
    for h in M:
        prob += sum(x[i,j] for i,j in fwd[h]) == y[h], "out_"+str(h)
        prob += sum(x[i,j] for i,j in rev[h]) == y[h], "in_"+str(h)
        
    for i in M:
        for j in M:
            if i!=j:
                prob += u[i] - u[j] + len(M)*x[i,j] <= len(M)-1, "MTZ_"+str((i,j))
    

    if objetivo == 1:                
        prob += lp.lpSum(Cij[i,j]*x[i,j] for (i,j) in Cij.keys()) + lp.lpSum(fik[i,k]*z[i,k] for k in K for i in Mk[k]), "OF"
    elif objetivo == 2:
        prob += lp.lpSum(Cij[i,j]*x[i,j] for (i,j) in Cij.keys()) + 10000*lp.lpSum(fik[i,k]*z[i,k] for k in K for i in Mk[k]), "OF"
    else:
        prob += 10000*lp.lpSum(Cij[i,j]*x[i,j] for (i,j) in Cij.keys()) + lp.lpSum(fik[i,k]*z[i,k] for k in K for i in Mk[k]), "OF"
    
    
    #prob.solve(lp.PULP_CBC_CMD(msg=0))
    prob.solve()
    
    #print("\t\tOF = "+str(lp.value(prob.objective))+"$")
    FO = lp.value(prob.objective)
    Info_Nodes = {}
    Info_Nodes[0]=[]
    Info_Nodes[1]=[]
    
    Compra = {Na_M[i]:[] for i in M}

    for i in M:
        if y[(i)].varValue > 0:
            #print(f'Prov select {i}: ' + str(y[(i)].varValue))
            
            Info_Nodes[0].append(Abre_M[Na_M[i]])
        else:
            Info_Nodes[1].append(Abre_M[Na_M[i]])
            
    FOcompra = 0
    for k in range(len(K)):
        for i in Mk[k]:
            if z[(i,k)].varValue > 0:
                #print(f'Prov {i} vende {k}: ' + str(z[(i,k)].varValue))
                FOcompra+= z[(i,k)].varValue*fik[(i,k)]
                Compra[Na_M[i]].append(Na_K[k])
    
    FORuta = 0
    Route = []
    Route1 = []
    for (i,j) in Cij.keys():
        if x[(i,j)].varValue > 0:
            #print(f'De {i} voy a {j}: ' + str(x[(i,j)].varValue))
            FORuta+=x[(i,j)].varValue*Cij[(i,j)]
            Route.append((Abre_M[Na_M[i]],Abre_M[Na_M[j]]))
            Route1.append((Na_M[i],Na_M[j]))
            
    Nombres = ["Costo Compra", "Costo Ruta", "Costo Total"]
    Costos = [round(FOcompra,2), round(FORuta,2), round(FORuta+FOcompra,2)]
    
    Valores = {0: Nombres, 1: Costos}
    
    print(f'El costo de la compra es: $' + str(round(FOcompra,2)))
    print(f'El costo de la ruta es: $' + str(round(FORuta,2)))
    print(f'El costo total es: $' + str(round(FORuta+FOcompra,2)))
    
    Secuencia = ['Hogar']
    inicio = 'Hogar'
    bandera = 0
    while bandera == 0:
        for i in Route1:
            if i[0] == inicio:
                Secuencia.append(i[1])
                inicio = i[1]
                if i[1] == 'Hogar':
                    bandera = 1
                
                break
            
            
    print("\nLa secuencia es: ")
    for i in range(len(Secuencia)):
        if i != len(Secuencia)-1:
            print(f"{Secuencia[i]} -- ", end = '')
        else:
            print(f"{Secuencia[i]}")
            
    print("\nEl plan de compra es: ")
    for i in Compra.keys():
        if len(Compra[i])>0:
            print(f"-{i}: ", end='')
            for j in range(len(Compra[i])):
                if j != len(Compra[i])-1:
                    print(f"{Compra[i][j]}, ", end = '')
                    
                else:
                    print(f"{Compra[i][j]}")
                
    
    return Valores, Route, Compra, Info_Nodes

def Dibuja_Ruta(Route, Nodes_out, coor, Valores):   
    G = nx.DiGraph(Route)
    G.add_nodes_from(Nodes_out)
    options = {
        "font_size": 10,
        "node_size": 250,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    fig, ax = plt.subplots()
    nx.draw_networkx(G, coor, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('Longitud') 
    plt.ylabel('Latitud')
    plt.title('Ruta óptima') 
    plt.show()  
    
    plt.bar(Valores[0], Valores[1])
    plt.show()  
