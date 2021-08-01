from pulp import LpVariable, LpProblem, LpMaximize, LpInteger
#Podemos importar la libreria bajo un alias pero debemos llamarlo 
#cada vez que usemos un objeto de este módulo
import pulp as lp

#Conjuntos
PRODUCTOS=['Soldados','Trenes','Barcos','Aviones']
ACTIVIDADES=['Carpinteria','Acabados']

#Parámetros
v={'Soldados':27000,
   'Trenes':21000,
   'Barcos':20000,
   'Aviones':28000}

c={'Soldados':24000,
   'Trenes':19000,
   'Barcos':15000,
   'Aviones':23000}

h={('Carpinteria','Soldados'):1,('Carpinteria','Trenes'):1,('Carpinteria','Barcos'):1,('Carpinteria','Aviones'):2,
   ('Acabados','Soldados'):2,('Acabados','Trenes'):1,('Acabados','Barcos'):1,('Acabados','Aviones'):2}

k={'Soldados':40,
   'Barcos':30,
   'Aviones':50}

n={'Carpinteria':80,
   'Acabados':100}

#Variables de decisión
x=lp.LpVariable.dict("Num_",PRODUCTOS,0,None,LpInteger)

#Crear el problema
prob=lp.LpProblem("Yepeto",lp.LpMaximize)

#Restricciones
#Horas disponibles de cada actividad
for a in ACTIVIDADES:
    prob += lp.lpSum(h[(a,p)]*x[(p)] for p in PRODUCTOS)<=n[a], "Horas_"+str(a)

#No producir por encima de la demanda de productos
for p in PRODUCTOS: 
    if p != 'Trenes':
        prob += x[(p)] <= k[p], "Demanda_"+str(p)

#Función Objetivo
prob += lp.lpSum((v[p]-c[p])*x[(p)] for p in PRODUCTOS)

#Imprime el problema explícito
print(prob)

#Resolver el problema
prob.solve()

#Impresión de resultados
#Estado del problema
print("Status: ",lp.LpStatus[prob.status])

print('\n')

#Valor de las variables
#Forma 1
for v in prob.variables():
    print(v.name,"=",v.varValue)

print('\n')


unidades_prod = []
#Forma 2
for p in PRODUCTOS:
    print(f'Unidades producidas de {p}: ' + str(x[(p)].varValue))
    unidades_prod.append(x[(p)].varValue)

print('\n')

#Forma 3
print(f"Unidades producidas de Soldados: {lp.value(x['Soldados'])}")

print('\n')

#Forma 4
print('Producto\tCantidad')
for p in PRODUCTOS:
    print(f'{p}\t{lp.value(x[(p)])}')

print('\n')

#Valor de la función objetivo
print("La utilidad de Yepeto es: ",lp.value(prob.objective))
