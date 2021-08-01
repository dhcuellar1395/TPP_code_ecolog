import pulp as lp
import matplotlib.pyplot as plt
import networkx as nx

def Corre_Modelo_Matematico(V, M, K, fwd, rev, Cij, fik, qik, Mk, dk):
    

    prob = lp.LpProblem("TPP_MTZ",lp.LpMinimize)
    
    #Variables de decisión
    x=lp.LpVariable.dicts("x_var", [(i,j) for (i,j) in Cij.keys()],lowBound=0,upBound=1,cat="Integer")
    y=lp.LpVariable.dicts("y_var", [i for i in M], lowBound=0,upBound=1,cat="Integer")
    z=lp.LpVariable.dicts("z_var", [(i,k) for (i,k) in fik.keys()],lowBound=0,upBound=None,cat="Continuous")
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
                
    prob += lp.lpSum(Cij[i,j]*x[i,j] for (i,j) in Cij.keys()) + lp.lpSum(fik[i,k]*z[i,k] for k in K for i in Mk[k]), "OF"
    #prob.solve(lp.PULP_CBC_CMD(msg=0))
    prob.solve()
    
    print("\t\tOF = "+str(lp.value(prob.objective))+"$")
    FO = lp.value(prob.objective)
    Info_Nodes = {}
    Info_Nodes[0]=[]
    Info_Nodes[1]=[]
    
    Compra = {i:[] for i in M}

    for i in M:
        if y[(i)].varValue > 0:
            print(f'Prov select {i}: ' + str(y[(i)].varValue))
            Info_Nodes[0].append(i)
        else:
            Info_Nodes[1].append(i)
            
            
    for (i,k) in fik.keys():
        if z[(i,k)].varValue > 0:
            print(f'Prov {i} vende {k}: ' + str(z[(i,k)].varValue))
            Compra[i].append(k)
    
    Route = []
    for (i,j) in Cij.keys():
        if x[(i,j)].varValue > 0:
            print(f'De {i} voy a {j}: ' + str(x[(i,j)].varValue))
            Route.append((i,j))
    
    return FO, Route, Compra, Info_Nodes

def Dibuja_Ruta(Route, Nodes_out, coor):   
    G = nx.DiGraph(Route)
    G.add_nodes_from(Nodes_out)
    options = {
        "font_size": 10,
        "node_size": 300,
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
