# ps5 Joan alegre cantón
#%% Packacges
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.optimize import fsolve
import quantecon as qe
import seaborn as sns

#%% We define a function for the markow process:
# Ns is the number of states, gamma is the correlation between current state and the following.
def ma(gamma,Ns): # Function that given a correlation gives a markow NsxNs dimension
    P = np.ones(Ns*Ns).reshape(Ns,Ns)
    P = P*((1-gamma)/(2*(Ns-1)))
    for i in range(Ns):
        P[i][i] = (1+gamma)/2
    return P

def path(P, init, N):
    '''This function gives the path of the states
    what it does is taking the markov matrix, make
    the distribution of every state, and use it 
    to take draws from a random variable.'''
    # === make sure P is a NumPy array === #
    P = np.asarray(P)
    # === allocate memory === #
    X = np.empty(N, dtype=int)
    X[0] = init
    # === convert each row of P into a distribution === #
    # In particular, P_dist[i] = the distribution corresponding to P[i, :]
    n = len(P)
    P_dist = [np.cumsum(P[i, :]) for i in range(n)] #CDF of every vector, in every row.

    # === generate the sample path === #
    for t in range(N - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])

    return X
    
def shocksimulation(gamma,init,N,S):
    '''This function gives the vector of shocks.
    using the markwov process with correlations, gamma
    is the correlation parameter, init is the initial 
    state, N is the number of shocks, and S is the
    vector of states.'''
    
    Ns = len(S)
    P = ma(gamma,Ns)
    X = path(P,init,N)
    Xs = np.zeros(N)
    for i in range(N):
        Xs[i] = S[int(X[i])]
        
    return Xs
#%% Exercice II.2: The infinitely-lived households economy
#I initialize all the paramaters:
varY = 0.05
Y = np.asarray([1-varY, 1+varY]) #vector of states
gamma = 0 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
#Two posible borrowing limits.
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit

#Making the grid:
a = np.linspace(Bnat,30,300)
a1a1,a2a2 = np.meshgrid(a,a)
#2nd step, we make returns matrix
s = len(Y)
Na = len(a)
def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=Bnat:
        c = 0.000000000000000000000000000000000000000000000000000000001
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-1000,2)#quadratic
    else:
        u = np.log(c)
    return u

vecut = np.vectorize(ut)
#This function add different returns matrix due to different shocks.
def ret(Y,a2a2,a1a1,i):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y)
        M =  np.concatenate([M, A])
    return M
P = ma(gamma,2)# I create markov chain matrix

'''this is done for two states variables I should generalize it for N states'''
#Probability vector from shock i to shock 1.
    #Initialazing values
X = np.zeros(Na*s*Na).reshape(Na*s,Na) #total values matrix
Ve = np.zeros(Na*s).reshape(1,Na*s) #Expectation vector
M = ret(Y,a2a2,a1a1,1) #returns matrix
V = np.zeros(Na*s).reshape(1,Na*s) #value function
P = ma(gamma,2) #markov matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)
gc2 = np.zeros(Na)
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

 
#%%II.3 life-Cicle economy:
M = ret(Y,a2a2,a1a1,1)
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
#%% II.4 Partial equilibrium with certanity
#I redifine returns function:
def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=Bnat:
        c = 0.000000000000000000000000000000000000000000000000000000001
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-100,2)#quadratic
    else:
        u = (pow(c,1-sigma)-1)/(1-sigma)
    return u
vecut = np.vectorize(ut)

def ret(Y,a2a2,a1a1,i):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y)
        M =  np.concatenate([M, A])
    return M
#iniialazing:
sigma = 2 #risk adversion
gamma = 0
varY = 0
Y = np.asarray([1-varY, 1+varY])
gamma = 0 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit
a = np.linspace(Bnat,30,300)
a1a1,a2a2 = np.meshgrid(a,a)
s = len(Y)
Na = len(a)
X = np.zeros(Na*s*Na).reshape(Na*s,Na) #total values matrix
Ve = np.zeros(Na*s).reshape(1,Na*s) #Expectation vector
V = np.zeros(Na*s).reshape(1,Na*s) #value function



#%% Finding policies for utility CRRA infinite horizont:
M = ret(Y,a2a2,a1a1,1) #returns matrix
P = ma(gamma,2) #markov matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
#taking sequence of assets:
NA1 = np.zeros(Na) #sequence of index assets, y1
GA1 = np.zeros(Na) #sequence of assets y1
NA2 = np.zeros(Na) #sequence of index assets, y2
GA2 = np.zeros(Na) #sequence of assets, y2
GC1 = np.zeros(Na-1) #sequence of consumption, y1
GC2 = np.zeros(Na-1) #sequence of consumption, y2
NA1[0] = 200 #starting with a = 11.45
NA2[0] = 200 #starting with a = 11.45
GA1[0] = a[int(NA1[0])]
GA2[0] = a[int(NA2[0])]

for k in range(1,Na):
    NA1[k] = na1[int(NA1[k-1])]
    GA1[k] = a[int(NA1[k])]
    NA2[k] = na1[int(NA2[k-1])]
    GA2[k] = a[int(NA2[k])]
    GC1[k-1] = Y[0]+(1+r)*GA1[k-1]-GA2[k]
    GC2[k-1] = Y[1]+(1+r)*GA2[k-1]-GA2[k]

plt.plot(a[2:Na],gc1[2:Na], label = 'consumption policy function y1')
plt.plot(a[2:Na],gc2[2:Na], label = 'consumption policy function y2')
plt.xlabel('Assets', size = 20)
plt.ylabel('Consumption', size = 20)
plt.legend()
plt.title('Consumption policies function with certanity CRRA CASE', size = 20)
plt.show()

plt.plot(a[2:Na],ga1[2:Na], label = 'asset policy function y1')
plt.plot(a[2:Na],ga2[2:Na], label = 'asset policy function y2')
plt.xlabel('Assets',size = 20)
plt.ylabel('Asset+1', size = 20)
plt.legend()
plt.title('Asset policies function with certanity CRRA CASE', size = 20)
plt.show()


time = np.linspace(1,Na,Na)
plt.plot(time,GA1, label = 'asset sequence y1')
plt.plot(time,GA2, label = 'asset sequence y2')
plt.xlabel('Time', size=20)
plt.ylabel('Assets', size = 20)
plt.legend()
plt.title('Sequence of assets, starting at a0=11.45 CRRA CASE', size = 20)
plt.show()

time = np.linspace(1,Na-1,Na-1)
plt.plot(time,GC1, label = 'Consumption y1')
plt.plot(time,GC2, label = 'consumption sequence y2')
plt.xlabel('Time', size=20)
plt.ylabel('Consumption', size = 20)
plt.legend()
plt.title('Sequence of consumption, starting at a0=11.45 CRRA CASE', size = 20)
plt.show()

#%% Finding policies for CRRA finite horizont T = 45.
M = ret(Y,a2a2,a1a1,1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1

# We want to plot policies for ages 5 and 40:
#For age 5:
na5 = storepolicy[4][0:600]
na5y1 = na5[2:300]
na5y2 = na5[302:600]
# for age 40:
na40 = storepolicy[40][0:600]
na40y1 = na40[2:300]
na40y2 = na40[302:600]
ga5y1 = np.zeros(Na-2)
ga40y1 = np.zeros(Na-2)
ga5y2 = np.zeros(Na-2)
ga40y2 = np.zeros(Na-2)
for n in range(Na-2):
    ga5y1[n] = a[int(na5y1[n])] 
    ga40y1[n] = a[int(na40y1[n])]
    ga5y2[n] = a[int(na5y2[n])] 
    ga40y2[n] = a[int(na40y2[n])]
    
#Corrige indices    

plt.plot(a[2:Na],na5y1, label = 'asset policy year5 shock 1')
plt.plot(a[2:Na],na5y2, label = 'asset policy year5 shock 2')
plt.plot(a[2:Na],na40y1, label = 'asset policy year40 shock 1')
plt.plot(a[2:Na],na40y2, label = 'asset policy year40 shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Asset t+1', size = 20)
plt.title('Asset policy different years and shocks CRRA CASE T=40 and T=5')
plt.show()
#%% Finding policies for utility QUADRATIC infinite horizont:
def ut1(a1,a2,y):
    c = y +(1+r)*a1-a2
    if a2<=Bnat:
        c = -1000000000000000000000
    if c<=0:
        c = -600000000000000000000000000
    u = -0.5*pow(c-100,2)#quadratic
    return u

vecut1 = np.vectorize(ut1)

def ret1(Y,a2a2,a1a1):#i is the type of utility
    y = Y[0]
    M = vecut1(a2a2,a1a1,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut1(a2a2,a1a1,y)
        M =  np.concatenate([M, A])
    return M
M = ret1(Y,a2a2,a1a1) #returns matrix
P = ma(gamma,2) #markov matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
#taking sequence of assets:
NA1 = np.zeros(60) #sequence of index assets, y1
GA1 = np.zeros(60) #sequence of assets y1
NA2 = np.zeros(60) #sequence of index assets, y2
GA2 = np.zeros(60) #sequence of assets, y2
GC1 = np.zeros(59) #sequence of consumption, y1
GC2 = np.zeros(59) #sequence of consumption, y2
NA1[0] = 200 #starting with a = 11.45
NA2[0] = 200 #starting with a = 11.45
GA1[0] = a[int(NA1[0])]
GA2[0] = a[int(NA2[0])]

for k in range(1,60):
    NA1[k] = na1[int(NA1[k-1])]
    GA1[k] = a[int(NA1[k])]
    NA2[k] = na1[int(NA2[k-1])]
    GA2[k] = a[int(NA2[k])]
    GC1[k-1] = Y[0]+(1+r)*GA1[k-1]-GA2[k]
    GC2[k-1] = Y[1]+(1+r)*GA2[k-1]-GA2[k]

plt.plot(a[2:300],gc1[2:300], label = 'consumption policy function y1')
plt.plot(a[2:300],gc2[2:300], label = 'consumption policy function y2')
plt.xlabel('Assets', size = 20)
plt.ylabel('Consumption', size = 20)
plt.legend()
plt.title('Consumption policies function with certanity QUADRATIC CASE', size = 20)
plt.show()

plt.plot(a[2:300],ga1[2:300], label = 'asset policy function y1')
plt.plot(a[2:300],ga2[2:300], label = 'asset policy function y2')
plt.xlabel('Assets',size = 20)
plt.ylabel('Asset+1', size = 20)
plt.legend()
plt.title('Asset policies function with certanity QUADRATIC CASE', size = 20)
plt.show()


time = np.linspace(1,60,60)
plt.plot(time,GA1, label = 'asset sequence y1')
plt.plot(time,GA2, label = 'asset sequence y2')
plt.xlabel('Time', size=20)
plt.ylabel('Assets', size = 20)
plt.legend()
plt.title('Sequence of assets, starting at a0=11.45 QUADRATIC CASE', size = 20)
plt.show()

time = np.linspace(1,60-1,60-1)
plt.plot(time,GC1, label = 'Consumption y1')
plt.plot(time,GC2, label = 'consumption sequence y2')
plt.xlabel('Time', size=20)
plt.ylabel('Consumption', size = 20)
plt.legend()
plt.title('Sequence of consumption, starting at a0=11.45 QUADRATIC CASE', size = 20)
plt.show()

#%% Finding policies for QUADRATIC finite horizont T = 45.
M = ret1(Y,a2a2,a1a1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1

# We want to plot policies for ages 5 and 40:
#For age 5:
na5 = storepolicy[4][0:600]
na5y1 = na5[2:300]
na5y2 = na5[302:600]
# for age 40:
na40 = storepolicy[40][0:600]
na40y1 = na40[2:300]
na40y2 = na40[302:600]
ga5y1 = np.zeros(Na-2)
ga40y1 = np.zeros(Na-2)
ga5y2 = np.zeros(Na-2)
ga40y2 = np.zeros(Na-2)
for n in range(Na-2):
    ga5y1[n] = a[int(na5y1[n])] 
    ga40y1[n] = a[int(na40y1[n])]
    ga5y2[n] = a[int(na5y2[n])] 
    ga40y2[n] = a[int(na40y2[n])]
    
    

plt.plot(a[2:Na],na5y1, label = 'asset policy year5 shock 1')
plt.plot(a[2:Na],na5y2, label = 'asset policy year5 shock 2')
plt.plot(a[2:Na],na40y1, label = 'asset policy year40 shock 1')
plt.plot(a[2:Na],na40y2, label = 'asset policy year40 shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Asset t+1', size = 20)
plt.title('Asset policy different years and shocks QUADRATIC CASE')
plt.show()

#%% II.4.2: With uncertanity.A)
'''From now on I will change the borrowing constraint up to 0, in order to
make binding the constraint as often as one may prefer to fall in debt. '''
##########CRRA COMPARATION WITH CERTANITY AND UNCERTANITY T = INFINITE:##############
def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = y+(1+r)*a1
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-1000,2)#quadratic
    else:
        u = np.log(c)
    return u

vecut = np.vectorize(ut)
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2]) #vector of states
gamma = 0 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
#Two posible borrowing limits.
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit
P = ma(gamma,2)

############     CRRA with certanity: ######################
M = ret(Y,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of certanity.
ga1Cert = ga1[139:Na]
ga2Cert = ga2[139:Na]
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

############     CRRA with certanity: ######################
M = ret(Y2,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of uncertanity.
ga1Uncert = ga1[139:Na]
ga2Uncert = ga2[139:Na]
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

########### What we want to comparate is: ############


plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 CRRA, T=Infinite CASE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 CRRA, T=Infinite, CASE')
plt.show()

##########CRRA COMPARATION WITH CERTANITY AND UNCERTANITY T = 45:##############

#Certanity:
M = ret(Y,a2a2,a1a1,1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
naC5 = storepolicy[4][0:Na*s]
na5Cy1 = naC5[0:Na]
na5Cy2 = naC5[Na:s*Na]

ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na5Cy1[n])] 
    ga2[n] = a[int(na5Cy2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
    
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

########## Uncertanity: ##########
M = ret(Y2,a2a2,a1a1,1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
naC5 = storepolicy[4][0:Na*s]
na5Cy1 = naC5[0:Na]
na5Cy2 = naC5[Na:s*Na]

ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na5Cy1[n])] 
    ga2[n] = a[int(na5Cy2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
    
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 CRRA, T=45, age=5 CASE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 CRRA, T=45, age=5 CASE')
plt.show()


#%%I make this part with Quadratic:
##########QUADRATIC COMPARATION WITH CERTANITY AND UNCERTANITY T = INFINITE:##############
def ut1(a1,a2,y):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = y+(1+r)*a1
    if c<=0:
        c = -600000000000000000000000000
    u = -0.5*pow(c-100,2)#quadratic
    return u

vecut1 = np.vectorize(ut1)

def ret1(Y,a2a2,a1a1):#i is the type of utility
    y = Y[0]
    M = vecut1(a2a2,a1a1,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut1(a2a2,a1a1,y)
        M =  np.concatenate([M, A])
    return M
M = ret1(Y,a2a2,a1a1) #returns matrix
P = ma(gamma,2) #markov matrix

vecut = np.vectorize(ut)
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2]) #vector of states
gamma = 0 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
#Two posible borrowing limits.
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit
P = ma(gamma,2)

############     CRRA with certanity: ######################
M = ret1(Y,a2a2,a1a1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of certanity.
ga1Cert = ga1[139:Na]
ga2Cert = ga2[139:Na]
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

############     Quadratic wit certanity: ######################
M = ret1(Y2,a2a2,a1a1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of uncertanity.
ga1Uncert = ga1[139:Na]
ga2Uncert = ga2[139:Na]
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

########### What we want to comparate is: ############


plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 Quadratic, T=Infinite, CASE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 Quadratic, T=Infinite, CASE')
plt.show()

##########QUADRATIC COMPARATION WITH CERTANITY AND UNCERTANITY T = 45:##############
#Certanity:
M = ret1(Y,a2a2,a1a1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
naC5 = storepolicy[4][0:Na*s]
na5Cy1 = naC5[0:Na]
na5Cy2 = naC5[Na:s*Na]

ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na5Cy1[n])] 
    ga2[n] = a[int(na5Cy2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
    
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

########## Uncertanity: ##########
M = ret1(Y2,a2a2,a1a1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V
T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
naC5 = storepolicy[4][0:Na*s]
na5Cy1 = naC5[0:Na]
na5Cy2 = naC5[Na:s*Na]

ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na5Cy1[n])] 
    ga2[n] = a[int(na5Cy2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]
    
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 QUADRATIC, T=45, age=5 CASE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 QUADRATIC, T=45, age=5 CASE')
plt.show()

#%% II.4.2: With uncertanity.B) #SIMULATION, T=45 QUADRATIC
#Shocks:
import random
random.seed( 3 )
N = 45
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2])
init = 0
gamma = 0
p = ma(gamma,2)
paths = path(p, init, N)
shocks = np.zeros(N)
for i in range(N):
    shocks[i] = Y2[int(paths[i])]
# Certanity:
M = ret1(Y,a2a2,a1a1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue1 = StoreValue
#simulation: #SOLUCIONAR ESTO DE LAS STORE POLICIES:
simu = np.zeros(45) #I will start with a = 200
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu1 = simu

#Uncertanity:
M = ret1(Y2,a2a2,a1a1)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue2 = StoreValue
simu = np.zeros(45) #I will start with a = 299
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu2 = simu


gac = np.zeros(45) 
gaun = np.zeros(45) 
gcc = np.zeros(45) 
gcun = np.zeros(45) 

gac[0] = a[int(simu1[0])]
gaun[0] = a[int(simu2[0])]
for n in range(1,45):
    gac[n] = a[int(simu1[n])] 
    gaun[n] = a[int(simu2[n])] 
    gcc[n] = Y[int(paths[n])]+(1+r)*gac[n-1]-gac[n]
    gcun[n] = Y2[int(paths[n])]+(1+r)*gaun[n-1]-gaun[n]

gcc[0] = Y[int(paths[0])]+(1+r)*gac[0]-gac[1]
gcun[0] = Y2[int(paths[0])]+(1+r)*gaun[0]-gaun[1]

time = np.linspace(1,45,45)
plt.plot(time, gcc, label = 'simulation with certanity')
plt.plot(time, gcun, label = 'simulation with uncertanity')
plt.plot(time,shocks, label = 'shocks')
plt.legend()
plt.ylabel('consumption')
plt.xlabel('asset t')
plt.title('Simulations Quadratic utility, cert vs uncert, consumption, t=45, EXERCICE II.4.2')
plt.show()

#%% II.4.2: With uncertanity.B) #SIMULATION, T=45 CRRA
#Shocks:
i = 1
N = 45
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2])
init = 0
gamma = 0
p = ma(gamma,2)
#paths = path(p, init, N)
#shocks = np.zeros(N)
#for i in range(N):
#    shocks[i] = Y2[int(paths[i])]
# Certanity:
M = ret(Y,a2a2,a1a1,i)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue1 = StoreValue
#simulation: #SOLUCIONAR ESTO DE LAS STORE POLICIES:
simu = np.zeros(45) #I will start with a = 200
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu1 = simu

#Uncertanity:
M = ret(Y2,a2a2,a1a1,i)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue2 = StoreValue
simu = np.zeros(45) #I will start with a = 299
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu2 = simu


gac = np.zeros(45) 
gaun = np.zeros(45) 
gcc = np.zeros(45) 
gcun = np.zeros(45) 

gac[0] = a[int(simu1[0])]
gaun[0] = a[int(simu2[0])]
for n in range(1,45):
    gac[n] = a[int(simu1[n])] 
    gaun[n] = a[int(simu2[n])] 
    gcc[n] = Y[int(paths[n])]+(1+r)*gac[n-1]-gac[n]
    gcun[n] = Y2[int(paths[n])]+(1+r)*gaun[n-1]-gaun[n]

gcc[0] = Y[int(paths[0])]+(1+r)*gac[0]-gac[1]
gcun[0] = Y2[int(paths[0])]+(1+r)*gaun[0]-gaun[1]

time = np.linspace(1,45,45)
plt.plot(time, gcc, label = 'simulation with certanity')
plt.plot(time, gcun, label = 'simulation with uncertanity')
plt.plot(time,shocks, label = 'shocks')
plt.legend()
plt.ylabel('consumption')
plt.xlabel('asset t')
plt.title('Simulations CRRA utility, cert vs uncert, consumption, t=45, EXERCICE II.4.2')
plt.show()

#%% II.4.2.3 sigma 2,5,20:

def ut(a1,a2,i,y,sigma):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = y+(1+r)*a1
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-1000,2)#quadratic
    else:
        u = (pow(c,1-sigma)-1)/(1-sigma)
    return u
vecut = np.vectorize(ut)

def ret(Y,a2a2,a1a1,i,sigma):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y,sigma)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y,sigma)
        M =  np.concatenate([M, A])
    return M
sigma = 2
i = 1
N = 45
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2])
init = 0
gamma = 0
p = ma(gamma,2)
#paths = path(p, init, N)
#shocks = np.zeros(N)
#for i in range(N):
#    shocks[i] = Y2[int(paths[i])]
# Certanity:
M = ret(Y,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue1 = StoreValue

simu = np.zeros(45) #I will start with a = 200
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu1 = simu

#Uncertanity:
M = ret(Y2,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue2 = StoreValue
simu = np.zeros(45) #I will start with a = 299
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu2 = simu


gac = np.zeros(45) 
gaun = np.zeros(45) 
gcc = np.zeros(45) 
gcun = np.zeros(45) 

gac[0] = a[int(simu1[0])]
gaun[0] = a[int(simu2[0])]
for n in range(1,45):
    gac[n] = a[int(simu1[n])] 
    gaun[n] = a[int(simu2[n])] 
    gcc[n] = Y[int(paths[n])]+(1+r)*gac[n-1]-gac[n]
    gcun[n] = Y2[int(paths[n])]+(1+r)*gaun[n-1]-gaun[n]

gcc[0] = Y[int(paths[0])]+(1+r)*gac[0]-gac[1]
gcun[0] = Y2[int(paths[0])]+(1+r)*gaun[0]-gaun[1]

time = np.linspace(1,45,45)
plt.plot(time, gcc, label = 'simulation with certanity')
plt.plot(time, gcun, label = 'simulation with uncertanity')
plt.plot(time,shocks, label = 'shocks')
plt.legend()
plt.ylabel('consumption')
plt.xlabel('asset t')
plt.title('Simulations CRRA utility SIGMA 2, cert vs uncert, consumption, t=45, EXERCICE II.4.3')
plt.show()

#%% SIGMA 5
sigma = 5
i = 1
N = 45
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2])
init = 0
gamma = 0
p = ma(gamma,2)
#paths = path(p, init, N)
#shocks = np.zeros(N)
#for i in range(N):
#    shocks[i] = Y2[int(paths[i])]
# Certanity:
M = ret(Y,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue1 = StoreValue
#simulation: #SOLUCIONAR ESTO DE LAS STORE POLICIES:
simu = np.zeros(45) #I will start with a = 200
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu1 = simu

#Uncertanity:
M = ret(Y2,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue2 = StoreValue
simu = np.zeros(45) #I will start with a = 299
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu2 = simu


gac = np.zeros(45) 
gaun = np.zeros(45) 
gcc = np.zeros(45) 
gcun = np.zeros(45) 

gac[0] = a[int(simu1[0])]
gaun[0] = a[int(simu2[0])]
for n in range(1,45):
    gac[n] = a[int(simu1[n])] 
    gaun[n] = a[int(simu2[n])] 
    gcc[n] = Y[int(paths[n])]+(1+r)*gac[n-1]-gac[n]
    gcun[n] = Y2[int(paths[n])]+(1+r)*gaun[n-1]-gaun[n]

gcc[0] = Y[int(paths[0])]+(1+r)*gac[0]-gac[1]
gcun[0] = Y2[int(paths[0])]+(1+r)*gaun[0]-gaun[1]

time = np.linspace(1,45,45)
plt.plot(time, gcc, label = 'simulation with certanity')
plt.plot(time, gcun, label = 'simulation with uncertanity')
plt.plot(time,shocks, label = 'shocks')
plt.legend()
plt.ylabel('consumption')
plt.xlabel('asset t')
plt.title('Simulations CRRA utility SIGMA 5, cert vs uncert, consumption, t=45, EXERCICE II.4.3')
plt.show()

#%% SIGMA 6
sigma = 6
i = 1
N = 45
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.1
Y2 = np.asarray([1-varY2, 1+varY2])
init = 0
gamma = 0
p = ma(gamma,2)
#paths = path(p, init, N)
# Certanity:
M = ret(Y,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue1 = StoreValue
#simulation: #SOLUCIONAR ESTO DE LAS STORE POLICIES:
simu = np.zeros(45) #I will start with a = 200
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu1 = simu

#Uncertanity:
M = ret(Y2,a2a2,a1a1,i,sigma)
P = ma(gamma,2) #markov matrix
ve = np.zeros(Na*s).reshape(1,Na*s)
V = np.amax(M, axis = 1).reshape(1,Na*s)
na = np.argmax(X, axis=1)
storepolicy = na.reshape(1,Na*s)
StoreValue = V

T = 0
while T<=45:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    StoreValue = np.concatenate([V, StoreValue])
    na = np.argmax(X, axis=1).reshape(1,Na*s)
    storepolicy = np.concatenate([na, storepolicy])
    T = T+1
StoreValue2 = StoreValue
simu = np.zeros(45) #I will start with a = 299
simu[0] = 299
for t in range(1,45):
    for n in range(45):
        simu[t] = storepolicy[t][int(simu[t-1])+int(paths[t])*300]
simu2 = simu


gac = np.zeros(45) 
gaun = np.zeros(45) 
gcc = np.zeros(45) 
gcun = np.zeros(45) 

gac[0] = a[int(simu1[0])]
gaun[0] = a[int(simu2[0])]
for n in range(1,45):
    gac[n] = a[int(simu1[n])] 
    gaun[n] = a[int(simu2[n])] 
    gcc[n] = Y[int(paths[n])]+(1+r)*gac[n-1]-gac[n]
    gcun[n] = Y2[int(paths[n])]+(1+r)*gaun[n-1]-gaun[n]

gcc[0] = Y[int(paths[0])]+(1+r)*gac[0]-gac[1]
gcun[0] = Y2[int(paths[0])]+(1+r)*gaun[0]-gaun[1]

time = np.linspace(1,45,45)
plt.plot(time, gcc, label = 'simulation with certanity')
plt.plot(time, gcun, label = 'simulation with uncertanity')
plt.plot(time,shocks, label = 'shocks')
plt.legend()
plt.ylabel('consumption')
plt.xlabel('asset t')
plt.title('Simulations CRRA utility SIGMA 6, cert vs uncert, consumption, t=45, EXERCICE II.4.3')
plt.show()

#nOT POSIBLE FOR SIGMA 20
''' I could not compute the simulation in the case of sigma 20 due to a tecnichal issue of the computer
 That says that it is too large to be able to compute a value, which I do not
 understand since if we increase sigma it should not be any problem about well-defined
 function, at leas for sigma 20, that is neither super large nor close to 1.'''

#%% II.4.2.4: Variance of 0.5:
''' I will now plot for CRRA case consumption policies of both uncertanity case and
certanity case. I will use CRRA with sigma 1, and variance of 0.5, gamma = 0 as ususal.
In the case of certanity both policies should be equal, since shocks should be smoothed
First I will plot consumption policy function with certanity and uncertanity in the case
of varY= 0.5, separating different shocks in different plots. '''

def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = y+(1+r)*a1
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-1000,2)#quadratic
    else:
        u = np.log(c)
    return u
vecut = np.vectorize(ut)

def ret(Y,a2a2,a1a1,i):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y)
        M =  np.concatenate([M, A])
    return M

vecut = np.vectorize(ut)
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.5
Y2 = np.asarray([1-varY2, 1+varY2]) #vector of states
gamma = 0 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
#Two posible borrowing limits.
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit
P = ma(gamma,2)

############     CRRA with certanity: ######################
M = ret(Y,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of certanity.
ga1Cert = ga1[139:Na]
ga2Cert = ga2[139:Na]
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

############     CRRA with certanity: ######################
M = ret(Y2,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of uncertanity.
ga1Uncert = ga1[139:Na]
ga2Uncert = ga2[139:Na]
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

########### What we want to comparate is: ############


plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 CRRA,VarY=0.5, T=Infinite CASE, II.4.2.4 EXERCICE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 CRRA,VarY=0.5, T=Infinite, CASE II.4.2.4 EXERCICE')
plt.show()

'''We can see that differences between certanity and uncertanity are higher than the previous cases,
we can also see that this is consistent with the idea of precautionary savings since
when we have a good shock (the second plot) we have a lower consumption than
the certanity one, genereting and excess of asset that we will consume on the bad shock
genereting that in the first plot we consume more than the certain case. '''

#%%II.4.2.5: VarY=0.5 and gamma = 0.95, 
def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = y+(1+r)*a1
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-1000,2)#quadratic
    else:
        u = np.log(c)
    return u
vecut = np.vectorize(ut)

def ret(Y,a2a2,a1a1,i):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y)
        M =  np.concatenate([M, A])
    return M

vecut = np.vectorize(ut)
varY = 0
Y = np.asarray([1-varY, 1+varY])
varY2 = 0.5
Y2 = np.asarray([1-varY2, 1+varY2]) #vector of states
gamma = 0.95 #correlation between shocks.
r = 0.04 # interest rate
rho = 0.06
b = 1/(1+rho) #♠discounting term
#Two posible borrowing limits.
Bnat = -((1+r)/r)*Y[0] #natural borrowing limit
B = 0 #Non-borrowing limit
P = ma(gamma,2)

############     CRRA with certanity: ######################
M = ret(Y,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of certanity.
ga1Cert = ga1[139:Na]
ga2Cert = ga2[139:Na]
gc1Cert = gc1[139:Na]
gc2Cert = gc2[139:Na]

############     CRRA with certanity: ######################
M = ret(Y2,a2a2,a1a1,1) #returns matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=1000:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#Taking from value function the policies functions:
    
na = np.argmax(X, axis=1)
na1 = na[0:Na]# index element asset policy when shock 1
na2 = na[Na:2*Na]# index element asset policy when shock 2
ga1 = np.zeros(Na)
ga2 = np.zeros(Na)  
gc1 = np.zeros(Na)#policie function consumption, y1
gc2 = np.zeros(Na)#policie function consumption, y2
for n in range(Na):
    ga1[n] = a[int(na1[n])] 
    ga2[n] = a[int(na2[n])] 
    gc1[n] = Y[0]+(1+r)*a[n]-ga1[n]
    gc2[n] = Y[1]+(1+r)*a[n]-ga2[n]

#This is the important part, policies of uncertanity.
ga1Uncert = ga1[139:Na]
ga2Uncert = ga2[139:Na]
gc1Uncert = gc1[139:Na]
gc2Uncert = gc2[139:Na]

########### What we want to comparate is: ############


plt.plot(a[139:Na],gc1Cert, label = 'Consumption policy certanity shock 1')
plt.plot(a[139:Na],gc1Uncert, label = 'Consumption policy uncertanity shock 1', color='black')
plt.legend()
plt.xlabel('asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock1 CRRA,VarY=0.5,, gamma=0.95, T=Infinite CASE, II.4.2.4 EXERCICE')
plt.show()

plt.plot(a[139:Na],gc2Cert, label = 'Consumption policy certanity shock 2')
plt.plot(a[139:Na],gc2Uncert, label = 'Consumption policy uncertanity shock 2', color='black')
plt.legend()
plt.xlabel('Asset t', size = 20)
plt.ylabel('Consumption', size = 20)
plt.title('Consumption policy uncert vs cert shock2 CRRA,VarY=0.5, gamma=0.95, T=Infinite, CASE II.4.2.5 EXERCICE')
plt.show()

#%% ################### AGAGARY HUGGET MODEL######################
#%% II5.1: The simple ABHI model.
d = 0.01
rho = 0.06
alpha = 0.5
gamma = 0.5

#Firm problem:
'''We use solve the problem of the firm with the guess, and assuming that in 
equilibrium aggregate labor is 1, to make this true I need a distribution of 
people such that if we integrate y(labor shock) we achieve 1. Since I will use
100 individuals y that will recieve every individual is 1/100 plus(menus) the
variance.

An important detail, I will use borrowing limit equal 0, nobody can fall in debt
and CRRA utility function.


Another issue is that I should design a distribution over assets such that I  hold
market clearing condition associated to my r guess. Nevertheless is much more easy
to guess a distribution of K and fin the r and w associated to this. And that is 
what I will do.'''
Na = 200
a = np.linspace(0,5,Na) #GRID OF ASSETS
a0 = a[0:100]
na0 = np.linspace(0,99,100)# index of initial asset.
K = sum(a0)
s = len(Y)
Na = len(a)
a1a1,a2a2 = np.meshgrid(a,a)


################# start problem ################
def firm1(A):
    K = sum(a0)
    d = 0.01
    alpha = 0.5
    r = A[0]
    w = A[1]
    F = np.zeros(2) 
    F[0] = alpha*pow(K,alpha-1)-r-d
    F[1] = (1-alpha)*pow(K,alpha)-w
    return F
Aguess = np.ones(2)
A = fsolve(firm1,Aguess)
r = A[0] 
w = A[1]
varY = 0.5/100
Y = np.asarray([1/100-varY,1/100+varY])
#bn = -((1+r)/r)*Y[0] #Natural borrowing limit


'''Now that we have initial distribution, K, r, w we can solve bellman equation.
take the policy function, and with this + markow chain take a new distribution'''
#Functions:
def ut(a1,a2,i,y):
    c = y +(1+r)*a1-a2
    if a2<=0:
        c = 0.000000000000000000000000000000000000000000000000000000001
    if c<=0:
        c = 0.0000000000000000000000000000000000000000000000000000000001
    if i == 0:
        u = -0.5*pow(c-100,2)#quadratic
    else:
        u = (pow(c,1-sigma)-1)/(1-sigma)
    return u
vecut = np.vectorize(ut)
#RETURNS MATRIX M:
def ret(Y,a2a2,a1a1,i):#i is the type of utility
    y = Y[0]
    M = vecut(a2a2,a1a1,i,y)
    for j in range(1,len(Y)):
        y = Y[j]
        A = vecut(a2a2,a1a1,i,y)
        M =  np.concatenate([M, A])
    return M

############# BELLMAN LOOOP: ##########################

X = np.zeros(Na*s*Na).reshape(Na*s,Na) #total values matrix
Ve = np.zeros(Na*s).reshape(1,Na*s) #Expectation vector
M = ret(Y,a2a2,a1a1,1) #returns matrix
V = np.zeros(Na*s).reshape(1,Na*s) #value function
P = ma(gamma,2) #markov matrix

#Starting to construct Bellman loop:
for k in range(Na): #with this loop I construct expectation value vector
    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
V1 = Ve[0][0:Na]
V2 = Ve[0][Na:Na*2]
X[0:Na] = M[0:Na]+b*V1
X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
C = 0
while C<=500:
    for k in range(Na):
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2
    V = np.amax(X,axis=1).reshape(1,Na*s)
    C = C+1
#POLICIES FUNCTION:
na = np.argmax(X, axis=1)
#na1 = na[0:Na]# index element asset policy when shock 1
#na2 = na[Na:2*Na]# index element asset policy when shock 2
#ga1 = np.zeros(Na)
#ga2 = np.zeros(Na)  
#for n in range(Na):
#    ga1[n] = a[int(na1[n])] 
#    ga2[n] = a[int(na2[n])]

#WITH THIS I CREATE INDEX DISTRIBUTION ASSETS FOR TOMORROW
na1 = np.zeros(100)
a1 = np.zeros(100)
q = np.random.binomial(1, 0.5, 100)
for i in range(100):
    na1[i] = na[int(na0[i])+int(q[i])*200]
    a1[i] = a[int(na1[i])]
na0 = na1
a0 = a1

##########
##
#def firm1(A):
#    K = sum(a0)
#    d = 0.01
#    alpha = 0.5
#    r = A[0]
#    w = A[1]
#    F = np.zeros(2) 
#    F[0] = alpha*pow(K,alpha-1)-r-d
#    F[1] = (1-alpha)*pow(K,alpha)-w
#    return F
#Aguess = np.ones(2)
#A = fsolve(firm1,Aguess)
#r = A[0] 
#w = A[1]
#    
#def ut(a1,a2,i,y):
#    c = y +(1+r)*a1-a2
#    if a2<=0:
#        c = 0.000000000000000000000000000000000000000000000000000000001
#    if c<=0:
#        c = 0.0000000000000000000000000000000000000000000000000000000001
#    if i == 0:
#        u = -0.5*pow(c-100,2)#quadratic
#    else:
#        u = (pow(c,1-sigma)-1)/(1-sigma)
#    return u
#vecut = np.vectorize(ut)
##RETURNS MATRIX M:
#def ret(Y,a2a2,a1a1,i):#i is the type of utility
#    y = Y[0]
#    M = vecut(a2a2,a1a1,i,y)
#    for j in range(1,len(Y)):
#        y = Y[j]
#        A = vecut(a2a2,a1a1,i,y)
#        M =  np.concatenate([M, A])
#    return M
#X = np.zeros(Na*s*Na).reshape(Na*s,Na) #total values matrix
#Ve = np.zeros(Na*s).reshape(1,Na*s) #Expectation vector
#M = ret(Y,a2a2,a1a1,1) #returns matrix
#V = np.zeros(Na*s).reshape(1,Na*s) #value function
#P = ma(gamma,2) #markov matrix
#
##Starting to construct Bellman loop:
#for k in range(Na): #with this loop I construct expectation value vector
#    Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
#    Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
#V1 = Ve[0][0:Na]
#V2 = Ve[0][Na:Na*2]
#X[0:Na] = M[0:Na]+b*V1
#X[Na:Na*s] = M[Na:Na*2]+b*V2 #I have create a new total values matrix incorporing expectations
#C = 0
#while C<=1000:
#    for k in range(Na):
#        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
#        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
#    V1 = Ve[0][0:Na]
#    V2 = Ve[0][Na:Na*2]
#    X[0:Na] = M[0:Na]+b*V1
#    X[Na:Na*s] = M[Na:Na*2]+b*V2
#    V = np.amax(X,axis=1).reshape(1,Na*s)
#    C = C+1
##POLICIES FUNCTION:
#na = np.argmax(X, axis=1)
#for i in range(100):
#    if q[i]==0:
#        q[i] = np.random.binomial(1,P[0][1])
#    if q[i]==1:
#        q[i] = np.random.binomial(1,P[1][1])
#for i in range(100):
#    na1[i] = na[int(na0[i])+int(q[i])*200]
#    a1[i] = a[int(na1[i])]
#a0 = a1    
    
T=0
while T<=80:
    def firm1(A):
        K = sum(a0)
        d = 0.01
        alpha = 0.5
        r = A[0]
        w = A[1]
        F = np.zeros(2)
        F[0] = alpha*pow(K,alpha-1)-r-d
        F[1] = (1-alpha)*pow(K,alpha)-w
        return F
    Aguess = np.ones(2)
    A = fsolve(firm1,Aguess)
    r = A[0]
    w = A[1]
    
    def ut(a1,a2,i,y):
        c = y +(1+r)*a1-a2
        if a2<=0:
            c = 0.000000000000000000000000000000000000000000000000000000001
        if c<=0:
            c = 0.0000000000000000000000000000000000000000000000000000000001
        if i == 0:
            u = -0.5*pow(c-100,2)#quadratic
        else:
            u = (pow(c,1-sigma)-1)/(1-sigma)
        return u
    vecut = np.vectorize(ut)
    #RETURNS MATRIX M:
    def ret(Y,a2a2,a1a1,i):#i is the type of utility
        y = Y[0]
        M = vecut(a2a2,a1a1,i,y)
        for j in range(1,len(Y)):
            y = Y[j]
            A = vecut(a2a2,a1a1,i,y)
            M =  np.concatenate([M, A])
        return M
    X = np.zeros(Na*s*Na).reshape(Na*s,Na) #total values matrix
    Ve = np.zeros(Na*s).reshape(1,Na*s) #Expectation vector
    M = ret(Y,a2a2,a1a1,1) #returns matrix
    V = np.zeros(Na*s).reshape(1,Na*s) #value function
    P = ma(gamma,2) #markov matrix
    #Starting to construct Bellman loop:
    for k in range(Na): #with this loop I construct expectation value vector
        Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
        Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
    V1 = Ve[0][0:Na]
    V2 = Ve[0][Na:Na*2]
    X[0:Na] = M[0:Na]+b*V1
    X[Na:Na*s] = M[Na:Na*2]+b*V2#I have create a new total values matrix incorporing expectations
    C = 0
    while C<=500:
        for k in range(Na):
            Ve[0][k] = P[0][0]*V[0][k]+P[0][1]*V[0][k+Na]
            Ve[0][k+Na] = P[1][0]*V[0][k]+P[1][1]*V[0][k+Na]
        V1 = Ve[0][0:Na]
        V2 = Ve[0][Na:Na*2]
        X[0:Na] = M[0:Na]+b*V1
        X[Na:Na*s] = M[Na:Na*2]+b*V2
        V = np.amax(X,axis=1).reshape(1,Na*s)
        C = C+1
    #POLICIES FUNCTION:
    na = np.argmax(X, axis=1)
    for i in range(100):
        if q[i]==0:
            q[i] = np.random.binomial(1,P[0][1])
        if q[i]==1:
                q[i] = np.random.binomial(1,P[1][1])
    for i in range(100):
        na1[i] = na[int(na0[i])+int(q[i])*200]
        a1[i] = a[int(na1[i])]
    na0 = na1
    a0 = a1   
    T=T+1
'''Initial asset really affects the stationary distribution since there are multiple 
stationary points where policy funtion is equal to the present asset.
If we are really rich (above than 3) then we decrease either we have a good
shock or a bad shock, nevertheless when if initial asset is between 2 and 3, good
shock do not make the individual reduce its wealth. In the end, everybody will be 
under 2, where all the wealth perfils are steady states. This means that if you
starte being under 2 you will be always under 2 no matter the shock you have. '''
sns.distplot(a0, hist=True, rug=True, label='asset distribution')
plt.title('Stationary distribution of weatlh ')
plt.legend()
plt.xlabel('asset')
plt.show()

c = np.zeros(100)
y = np.zeros(100)
#Finding consumption:
for i in range(100):
    c[i] = Y[int(q[i])]+r*a1[i]
    y[i] = w*Y[int(q[i])]

'''Consumption is affected by shocks, even if people that is poor will not move 
from its wealth status, income will affect the consumption, and it happens to be that
in this case I simulate a shock vector where most part of the bad shock was taken
by the people that would bi on the second histogram of assets ( people that have 
between asset of 1 and 1.5), nevertheless what we see is fairly consistent,
having that the poorer people consume less than the richest people. '''    
sns.distplot(c, hist=True, rug=True, label='consumption distribution')
plt.title('distribution of weatlh ')
plt.legend()
plt.xlabel('consumption')
plt.show()

'''For income distribution we can see that almost half of the people have had
a bad shock and half a good shock, this is due to the vectors of the markow chain
are really similar, having [a, 1-a],[b, 1-b] and a = 1-b and 1-a = b  '''
sns.distplot(y, hist=True, rug=True, label='income distribution')
plt.title('Stationary distribution of weatlh ')
plt.legend()
plt.xlabel('income')
plt.show()


