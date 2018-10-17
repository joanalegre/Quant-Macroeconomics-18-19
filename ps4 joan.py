#%% PS4, Joan Alegre
#%% Packages
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.optimize import fsolve
from numpy import *
import matplotlib.pyplot as plt
import timeit

#%% Exercice 1:
# Parameters:
ht = 1 # labor
o = 0.679 #Theta
b = 0.988 # beta
d = 0.013 # delta

#First of all I will find the stady state of this economy.
def ss(k):
    return 1-b*((1-o)*pow(k,-o)*pow(ht,o)+(1-d))
kguess = 1
kss = fsolve(ss,kguess)
# hence k stationary is 42.552

# STEP 1: DEFINE GRID (using stationary k inside the set) k today vs k tomorrow
k = np.linspace(1, 200, 300)
x, y = np.meshgrid(k,k)

# STEP 2: GUESS A SOLUTION. I guess V = 0
tic = timeit.default_timer()
V0 = np.zeros(300)

# STEP3: DEFINE MATRIX OF RETURNS, NON-NEGATIVE RESTRICCION (STEP4).
def ret(k1,k2):
    m = pow(k1,1-o)*pow(ht,o)-k2+(1-d)*k1
    if m<=0:
        a = -1000000000000000000
    else:
        a = np.log(m)
    return a

vecret = np.vectorize(ret)
M = vecret(y,x) # matrix of returns

# STEP 5: Compute the matrix X.
X = M+b*V0

#STEP 5.2: I compute the maximun value of every row to create a new vecto V:
V1 = np.amax(X, axis=1)
kp = np.ones(300) # policy function zero vectors

# STEP 6: I create the loop to iterate the bellman equation.
def bellman(M,V1):
    C = 0
    M = np.matrix(M)
    V1 = np.matrix(V1).reshape(1,300)
    while C<=2000:
        X = M+b*V1
        n = np.argmax(X, axis=1) 
        V1 = np.matrix(np.amax(X, axis=1)).reshape(1,300)
        C = C+1
#        for i in range(300):
#            kp[i] = k[n[i]] # you can remove this to be faster, it is usefull in the future.
    V1 = V1.reshape(300,1)
    return V1, n
V , n  = bellman(M,V1)
toc = timeit.default_timer()

Tbellman = toc-tic

for i in range(300):
    kp[i] = k[n[i]]
    
plt.plot(k,V, label = 'Value function')
plt.legend()
plt.title('value function', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()

plt.plot(k,kp, label = 'policy function')
plt.plot(k,k, color = "black", label = '45 degree line')
plt.axvline(x=42.5, color='black', linestyle='dashed', label='Steady state')
plt.xlabel('capital today', size=10)
plt.ylabel('capital tomorrow', size=10)
plt.title('Decition rule')
plt.legend()
plt.show()

#%%Exercice b: Taking into account monotonicity.
tic = timeit.default_timer()
V1 = np.zeros(300)
M = vecret(y,x)
X = M+b*V1
def bellmanmono(M,V1):
    I = np.zeros(300)
    C = 0
    V1 = np.matrix(V1).reshape(1,300)
    while C<=2000:
        X = M+b*V1
        X = np.array(X)
        I[0] = np.argmax(X[0])
        V1 = np.array(V1)
        V1[0][0] = X[0][int(I[0])]
        for j in range(1,300):
            I[j] = np.argmax(X[j][int(I[j-1]):300])+I[j-1]
            V1[0][j] = X[j][int(I[j])]
        V1 = np.matrix(V1).reshape(1,300)
        C = C+1
#        for i in range(300):
#            kp[i] = k[n[i]] # you can remove this to be faster, it is usefull in the future.
    X = np.array(X)
    V1 = np.array(V1.reshape(300,1))
   
    return V1, I, C, X
V2, I, C, X  = bellmanmono(M,V1)
toc = timeit.default_timer()
Tmonotone = toc-tic

kp1 = np.ones(300)
for i in range(300):
    kp1[i] = k[int(I[i])]

#%% Exercice c: Taking into account concavity.
tic = timeit.default_timer()
V1 = np.zeros(300)
def bellmanconca(M,V1):
    V1 = np.matrix(V1).reshape(1,300)
    X = M+b*V1
    V1 = np.array(V1)
    V2 = np.matrix(np.zeros(300)).reshape(1,300)
    V2 = np.array(V2)
    C=0
    while C<300:
        X = M+b*V1
        V2[0][0] = np.amax(X[0])
        for i in range(1,300):#filas
            
            for j in range(1,300):# columnas
                
                if M[i][j]+b*V1[0][j]<M[i][j-1]+b*V1[0][j-1]:
                    V2[0][i] = M[i][j-1]+b*V1[0][j-1]
                    break
        V1 = np.array(V2)        
        C=C+1
    return V1 
V1 = bellmanconca(M,V1)
toc = timeit.default_timer()
Tconcavity = toc-tic

X = M+b*V1
V1 = V1.reshape(300,1)
n = np.argmax(X, axis=1)         
for i in range(300):
    kp[i] = k[n[i]]   

plt.plot(k,V1, label = 'Value function')
plt.legend()
plt.title('value function', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()

plt.plot(k,kp, label = 'policy function')
plt.plot(k,k, color = "black", label = '45 degree line')
plt.axvline(x=42.5, color='black', linestyle='dashed', label='Steady state')
plt.xlabel('capital today', size=10)
plt.ylabel('capital tomorrow', size=10)
plt.title('Decition rule')
plt.legend()
plt.show()
    
#%%Exercice d: Local search.
tic = timeit.default_timer()
k = np.linspace(1, 200, 150)
x1, y1 = np.meshgrid(k,k)
V0 = np.zeros(150)
def bellman2(M,V1):
    C = 0
    M = np.matrix(M)
    V1 = np.matrix(V1).reshape(1,150)
    while C<=2000:
        X = M+b*V1
        n = np.argmax(X, axis=1) 
        V1 = np.matrix(np.amax(X, axis=1)).reshape(1,150)
        C = C+1
#        for i in range(300):
#            kp[i] = k[n[i]] # you can remove this to be faster, it is usefull in the future.
    V1 = V1.reshape(150,1)
    return V1, n
vecret = np.vectorize(ret)
M = vecret(y1,x1) # matrix of returns
X = M+b*V0
V1 = np.amax(X, axis=1)
kp = np.ones(150) # policy function zero vectors
V , n  = bellman2(M,V1)
toc = timeit.default_timer()
Tlocal = toc-tic
for i in range(150):
    kp[i] = k[n[i]]
#%% Exercice e: Concavity+monocity.
#NON-computational feasible.



#%% Exercice f: Hordward policy.
# Howard when iteration = 30, reassigment every 30 iterations.
    
tic = timeit.default_timer()
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=30:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)

toc = timeit.default_timer()
Thowin30 = toc-tic    
X = M+b*V0
I = np.argmax(X,axis=1)

#%%
# Howard when iteration = 1, reassigment every 30 iterations.
tic = timeit.default_timer()
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=1:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
toc = timeit.default_timer()
Thowreain1 = toc-tic
X = M+b*V0
I = np.argmax(X,axis=1)


#%% Exercice f: Hordward policy.
# Howard when iteration = 80, reassigment every 30 iterations.
tic = timeit.default_timer()
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=80:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<30:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
toc = timeit.default_timer()
Thowin80 = toc-tic    

X = M+b*V0
I = np.argmax(X,axis=1)
#%% exercice g Howard with reassements:
tic = timeit.default_timer()
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=30:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
while C<5:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
toc = timeit.default_timer()
Thowrea5 = toc-tic    


X = M+b*V0
I = np.argmax(X,axis=1)
#%% first iterations=30 and reassigment every 10
tic = timeit.default_timer()
 
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=30:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<10:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<10:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<10:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<10:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
toc = timeit.default_timer()
Thowre10 = toc-tic        
X = M+b*V0
I = np.argmax(X,axis=1)

#%% first iterations=30 and reassigment every 20
tic = timeit.default_timer()

V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=30:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<20:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<20:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
toc = timeit.default_timer()
Thowire20 = toc-tic    
    
X = M+b*V0
I = np.argmax(X,axis=1)

#%% first iterations=30 and reassigment every 50
tic = timeit.default_timer()
  
V0 = np.zeros(300)
M = vecret(y,x)
V0 = np.matrix(V0).reshape(1,300)
V0 = np.array(V0)
X = M +b*V0
C = 0 #First itearation
while C<=30:
    X = M+b*V0
    V0 = np.amax(X, axis=1)
    C = C+1

Iguess = np.argmax(X, axis=1)
C = 0#Howard first time
while C<50:
    X = M+b*V0
    V0 = V0.reshape(300,1)
    for i in range(300):
        V0[i] = X[i][int(Iguess[i])]
    V0 = np.matrix(V0).reshape(1,300)
    V0 = np.array(V0)
    C = C+1
toc = timeit.default_timer()
Thowire50 = toc-tic    
X = M+b*V0
I = np.argmax(X,axis=1)

#%% Exercice 1.2, with labor:
#First step is to find S.S:
ht = 1 # labor
o = 0.679 #Theta
b = 0.988 # beta
d = 0.013 # delta

def sslabor(x):
    F = np.zeros(2)
    kt = x[0]
    ht = x[1]
    F[0] = 1/b-(1-o)*pow(kt,-o)*pow(ht,o)-(1-d) #First condition.
    F[1] = o*pow(kt,-o)*pow(ht,o-1.5)-5.24*(pow(kt,-o)*pow(ht,o)-d) #second condition.
    return F
khguess = np.ones(2)
kss, hss = fsolve(sslabor,khguess)


#second step making the grid:
k = np.linspace(10,210,200)
h = np.linspace(0.15,0.6,50)
k1k1, hh, k2k2 = np.meshgrid(k,h,k)

def retlabor(k1,h,k2):
    c = pow(k1,1-o)*pow(h,o)-k2+(1-d)*k1
    l = 5.24*pow(h,1.5)/1.5
    if c<0:
        c = 0.0000000001
    return np.log(c)-l

vectrect = np.vectorize(retlabor)
M = vectrect(k1k1,hh,k2k2)
V0 = np.zeros(200)

def bellmanlabor(M,V0):
    X = np.ones(2000000).reshape(50,200,200)
    for i in range(50):
            X[i] = M[i]+b*V0
    C=0
    while C<1000:
        V0 = np.amax(np.amax(X, axis= 2), axis=0)
        for i in range(50):
            X[i] = M[i]+b*V0
        C = C+1
    for i in range(50):
            X[i] = M[i]+b*V0
    nh = np.argmax(np.amax(X, axis=2), axis=0)# Policy for labor
    nk = np.zeros(200)
    for i in range(200):
        nk[i] = np.argmax(X[int(nh[i])][i])# Policy for capital
    X = np.ones(2000000).reshape(50,200,200)
    for i in range(50):
            X[i] = M[i]+b*V0
    return V0, nh, nk, X
      
V0, nh, nk, X = bellmanlabor(M,V0) 

plt.plot(k,V0, label = 'Value function')
plt.legend()
plt.title('value function', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()

#%%Exercice 1.3 Chebycheff:

#%% Exercice 2.1: Bellman with shocks.
ht = 1 # labor
o = 0.679 #Theta
b = 0.988 # beta
d = 0.013 # delta
# Markow chain: 
PI = np.array([[0.49751, 0.50248] for i in range(2)])
#step 1, I make grid:
k = np.linspace(1, 200, 300)
x, y = np.meshgrid(k,k)
z = np.array([1.01,1/1.01])

# STEP2: DEFINE MATRIX OF RETURNS, NON-NEGATIVE RESTRICCION (STEP4).
def ret(k1,k2,z):
    m = z*pow(k1,1-o)*pow(ht,o)-k2+(1-d)*k1
    if m<=0:
        a = -1000000000000000000
    else:
        a = np.log(m)
    return a
vecret = np.vectorize(ret)
# STEP3: DEFINE VALUE VECTOR, V 2Px1:
V1 = np.zeros(300).reshape(1,300)
M1 = vecret(y,x,z[0])
M2 = vecret(y,x,z[1])
M = np.array([M1,M2]).reshape(600,300)
C=0

while C<2000:
    X = M+b*V1
    V1 = 0.497*np.amax(X[0:300],axis=1)+0.502*np.amax(X[300:600], axis=1)
    C= C+1
    
V1 = np.amax(X[0:300],axis=1)
V2 = np.amax(X[300:600], axis=1)
    
plt.plot(k,V1, label = 'Value function when z = 1.01')
plt.plot(k,V2, label = 'value function when z=(1/1.01)')
plt.legend()
plt.title('value function', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()

#%%I try wirh larger shocks:
z = np.array([1.2,1/1.2])

def ret(k1,k2,z):
    m = z*pow(k1,1-o)*pow(ht,o)-k2+(1-d)*k1
    if m<=0:
        a = -1000000000000000000
    else:
        a = np.log(m)
    return a
vecret = np.vectorize(ret)
# STEP3: DEFINE VALUE VECTOR, V 2Px1:
V1 = np.zeros(300).reshape(1,300)
M1 = vecret(y,x,z[0])
M2 = vecret(y,x,z[1])
M = np.array([M1,M2]).reshape(600,300)
C=0

while C<2000:
    X = M+b*V1
    V1 = 0.497*np.amax(X[0:300],axis=1)+0.502*np.amax(X[300:600], axis=1)
    C= C+1
    
V1 = np.amax(X[0:300],axis=1)
V2 = np.amax(X[300:600], axis=1)
    
plt.plot(k,V1, label = 'Value function when z = 100')
plt.plot(k,V2, label = 'value function when z=(1/100)')
plt.legend()
plt.title('value function', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()
#%%I try with different Markow matrix:

z = np.array([10,(1/10)])
V1 = np.zeros(300).reshape(1,300)
V2 = np.zeros(300).reshape(1,300)
M1 = vecret(y,x,z[0])
M2 = vecret(y,x,z[1])
C=0
while C<500:
    X1 = M1+b*V1
    X2 = M2+b*V2
    V1 = 0.9*np.amax(M1,axis=1)+0.1*np.amax(M2, axis=1)
    V2 = 0.1*np.amax(M1,axis=1)+0.9*np.amax(M2, axis=1)
    C = C+1

   
plt.plot(k,V1, label = 'Value function when z = 10')
plt.plot(k,V2, label = 'value function when z=(1/10)')
plt.legend()
plt.title('Economy with a markow chain [0.9,0.1],[0.1,0.9]', size =20)
plt.xlabel('capital today', size=10)
plt.ylabel('value', size=10)
plt.show()

