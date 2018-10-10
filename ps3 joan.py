# Ps3 Joan alegre cantón.
#%% Packages
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy.optimize as sc
import numpy as np
from scipy.optimize import fsolve
from numpy import random
from numpy import *
from scipy.optimize import *
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
#%%Exercice 1. a) and b):
# Let's find Z:
y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)
#Delta (depreciation) is clearly 1/16.
d = 1/16 # depreciation
h2z = pow((2*z*ht),0.67) #new worker in efficient terms.
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*h2z+1-d-(1/b)
kguess = 1
k = sc.fsolve(SS,kguess)
print('9.6815 is the new stationary capital when 2 times z')
#%%Exercice 1. c))
def EE(k1,k2,k3):
    return pow(k2,0.33)*h2z-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*h2z+(1-d))*(pow(k1,0.33)*h2z-k2+(1-d)*k1)
#def transv(k1,k2):
#    return pow(b,500)*(0.33*pow(k1,-0.67)*h2z+(1-d))*k1*pow(pow(k1,0.33)*h2z-k2+(1-d)*k1,-1)
K = 9.68
def transition(z): 
    F = np.zeros(200)
    z = z
    F[0] = EE(4,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[199] = 9.68
    F[198] = EE(z[197], z[198], z[199])
    for i in range(1,198):
        F[i] = EE(z[i],z[i+1],z[i+2])
    return F
z = np.ones(200)*4
k = sc.fsolve(transition, z)
k[0] = 4
# I create the domain to plot everything.
kplot = k[0:100]
t = np.linspace(0,100,100)

# I create savings, output and consumption:
yt = pow(kplot,0.33)*h2z
kt2 = k[1:101]
st = kt2-(1-d)*kplot
ct = yt-st

plt.plot(t,kplot, label='capital')
plt.legend()
plt.title('Transition of K from  first S.S to second S.S, first 100 times', size=20)
plt.xlabel('Time')
plt.ylabel('capital')
plt.show()

plt.plot(t,yt, label='Yt output')
plt.plot(t,st, label='st savings')
plt.plot(t,ct, label='ct consumption')
plt.legend(loc='upper right')
plt.title('Transition of the economy', size=20)
plt.xlabel('Time', size = 20)
plt.ylabel('Quantity', size = 20)
plt.show()
#%% Exercice1. d):

y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)

#new hz
hz = pow((z*ht),0.67)

# New euler equation:
def EE2(k1,k2,k3):
    return pow(k2,0.33)*hz-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*hz+(1-d))*(pow(k1,0.33)*hz-k2+(1-d)*k1)

k10 = k[9]
# I compute the new Stady stationary:
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*hz+1-d-(1/b)
kguess = 1
kss = sc.fsolve(SS,kguess)

def transition(z): 
    F = np.zeros(100)
    z = z
    F[0] = EE2(6.801,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[99] = 4.84
    F[98] = EE2(z[97], z[98], z[99])
    for i in range(1,98):
        F[i] = EE2(z[i],z[i+1],z[i+2])
    return F
z = np.ones(100)*4
k2 = sc.fsolve(transition, z)
k2[0] = 6.801

#lets plot everything:
kplot = k[0:100]
kfin = np.append(k[0:10],k2[0:90]) 
t = np.linspace(0,100,100)
plt.plot(t,kplot,'--', label='expected transition')
plt.plot(t,kfin, label='actual transition')
plt.axvline(x=10, color='black')
plt.legend()
plt.title('Difference of economy by shock at t=10', size=20)
plt.xlabel('Time', size=20)
plt.ylabel('Capital', size = 20)
plt.show()

#%% Question 2:
y0 = np.random.uniform(0.001,0.009,400)

for (i, item) in enumerate(y0):
    if 0.0055<item<0.0087:        
        y0[i] = 0.001        
y0 = np.array(y0)
sns.distplot(y0, hist=False, rug=True, label='kernel aproximation of y0 distribution')
plt.legend();
plt.show()
def sum(x):
    z = 0
    for i in x:
        z = z + i
    return z

r = 0.618448
sigma = 3
kappa = 4
nu = 4
beta = 0.99
tau = 0
T0 = 0
T1 = 0
          
# I create the matrix of attributes of individuals:
# matrix of NHU valures.
NHU = np.zeros(400)
NHU[0:100] = 1
NHU[100:200] = 1.5
NHU[200:300] = 2.5
NHU[300:400] = 3

# Matrix of epsilons:
EPS = np.tile(np.array([0.05, -0.05]),200)

# I create matrix caracteristics, C:
C = np.append(y0,NHU)
C = np.append(C,EPS)
C = C.reshape(3,400)

Equilibrium = []
  
for i in range(400):
    def rest(x):
        F =np.zeros(4)
        a = x[0]
        h0 = x[1]
        h1 = x[2]
        lamda = x[3]
        F[0]= np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)*(1-tau)*C[1][i] - kappa*np.power(h0,1/nu)
        F[1]= beta*np.power((((1-tau)*C[1][i]*h1)+(1+r)*a + T1), -sigma)*(1-tau)*C[1][i] - kappa*np.power(h1,1/nu)
        F[2]= beta*(np.power(((1-tau)*C[1][i]*h1)+(1+r)*a + T1,-sigma)*(1+r)) - lamda - np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)
        F[3]= ((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a) + (1/(1+r))*((1-tau)*(C[1][i]+C[2][i])*h1 + (1+r)*a + T1) - C[0][i] - (1+r)*(C[1][i]+C[2][i])*h1
        return F
    sguess= np.array([0.001,0.1,0.1, 1])
    sol = fsolve(rest,sguess)
    Equilibrium.append(sol)
    eq_mat = np.matrix(Equilibrium)
shape(eq_mat)
eq_mat = np.array(eq_mat)
eq_mat = eq_mat.T

a = eq_mat[0]
h0 = eq_mat[1]
h1 = eq_mat[2]
# Now I will find consumptions:
C1 = np.zeros(400)
C2 = np.zeros(400)
for i in range(400):
   C1[i] = (1-tau)*C[1][i]*h0[i] + C[0][i] + T0 -a[i]
   C2[i] = (1-tau)*(C[1][i]+C[2][i])*h1[i]+(1+r)*a[i]+T1 


#%% I make four groups of a,c,c' and y0.
   # For present consumption:
plt.scatter(y0[0:100],C1[0:100], label = 'C for nhu = 1')
plt.scatter(y0[0:100],C1[100:200], label = 'C for nhu = 1.5')
plt.scatter(y0[0:100],C1[200:300], label = 'C for nhu = 2.5')
plt.scatter(y0[0:100],C1[300:400], label = 'C for nhu = 3')
plt.title('Consumption t=1 vs y0', size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Consumption 1', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],C1[0:100], label = 'C for nhu = 1')
plt.title('Consumption t = 1 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Consumption 1', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()
# For consumption period 2:

plt.scatter(y0[0:100],C2[0:100], label = 'C for nhu = 1')
plt.scatter(y0[0:100],C2[100:200], label = 'C for nhu = 1.5')
plt.scatter(y0[0:100],C2[200:300], label = 'C for nhu = 2.5')
plt.scatter(y0[0:100],C2[300:400], label = 'C for nhu = 3')
plt.title('Consumption t=2 vs y0', size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Consumption 2', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],C2[0:100], label = 'C for nhu = 1')
plt.title('Consumption t=2 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Consumption 2', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# For h0
plt.scatter(y0[0:100],h0[0:100], label = 'Labor for nhu = 1')
plt.scatter(y0[0:100],h0[100:200], label = 'Labor for nhu = 1.5')
plt.scatter(y0[0:100],h0[200:300], label = 'Labor for nhu = 2.5')
plt.scatter(y0[0:100],h0[300:400], label = 'Labor for nhu = 3')
plt.title('Labour t=1 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Present labour', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],h0[0:100], label = 'Labour for nhu = 1')
plt.title('Labour t=1 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Present labour', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for h1:
# For h1
plt.scatter(y0[0:100],h1[0:100], label = 'Labor for nhu = 1')
plt.scatter(y0[0:100],h1[100:200], label = 'Labor for nhu = 1.5')
plt.scatter(y0[0:100],h1[200:300], label = 'Labor for nhu = 2.5')
plt.scatter(y0[0:100],h1[300:400], label = 'Labor for nhu = 3')
plt.title('Labour t=2 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Labour second period', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],h1[0:100], label = 'Labour for nhu = 1')
plt.title('Labour t=2 vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Labour second period', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for assets:
plt.scatter(y0[0:100],a[0:100], label = 'assets for nhu = 1')
plt.scatter(y0[0:100],a[100:200], label = 'assets for nhu = 1.5')
plt.scatter(y0[0:100],a[200:300], label = 'assets for nhu = 2.5')
plt.scatter(y0[0:100],a[300:400], label = 'assets for nhu = 3')
plt.title('assets vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('assets', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],a[0:100], label = 'assets for nhu = 1')
plt.title('Assets vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('Assets', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()
print('Consumption assets and labor in t=1 should not differ among individuals with same ability and same initial endowment this lead me to think that somehow indiividuals are infering the future shock epsilon and answering optimizing with that information, which not follow the logic of the problem. Hence, I have a mistake in the whole question. Nevertheless, I will continue with this mistake for the sake of making a big part of the task, but I am completly aware of my error.')

#%% second part of plotting, saving rates:
st = np.ones(400)
for i in range(400):
    st[i] = a[i]/(y0[i]+C[1][i]*h1[i]*(1-tau))

# I make it for assets:
plt.scatter(y0[0:100],st[0:100], label = 'savings for nhu = 1')
plt.scatter(y0[0:100],st[100:200], label = 'savings for nhu = 1.5')
plt.scatter(y0[0:100],st[200:300], label = 'savings for nhu = 2.5')
plt.scatter(y0[0:100],st[300:400], label = 'savings for nhu = 3')
plt.title('savings vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('savings', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],st[0:100], label = 'savings for nhu = 1')
plt.title('savings vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('savings', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

#%% third part of plotting, 
expgc = np.ones(400)
for i in range(400):
    expgc[i] = ((1-tau)*C[1][i]*h1[i]+(1+r)*a[i]+T1-((1-tau)*C[1][i]*h0[i]+y0[i]+T0))/((1-tau)*C[1][i]+y0[i]+T0)
gc = np.ones(400)
for i in range(400):
    gc[i] = ((1-tau)*(C[1][i]+C[2][i])*h1[i]+(1+r)*a[i]+T1-((1-tau)*C[1][i]*h0[i]+y0[i]+T0))/((1-tau)*C[1][i]+y0[i]+T0)

# gc vs E[gc]
plt.scatter(y0,expgc, label = 'E[gc], expected consumption growth')
plt.scatter(y0,gc, label = 'gc, consumption growth')
plt.legend()
plt.title('Expected versus actual consumption growth')
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('percentatge growth', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for assets:
plt.scatter(y0[0:100],expgc[0:100], label = 'E[gc] for nhu = 1')
plt.scatter(y0[0:100],expgc[100:200], label = 'E[gc]  for nhu = 1.5')
plt.scatter(y0[0:100],expgc[200:300], label = 'E[gc]  for nhu = 2.5')
plt.scatter(y0[0:100],expgc[300:400], label = 'E[gc]  for nhu = 3')
plt.title('expected cons growth E[gc] vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('percentatge growth', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

# I make it for only one group:
plt.scatter(y0[0:100],expgc[0:100], label = 'E[gc] for nhu = 1')
plt.title('expected cons growth E[gc] vs y0',size = 20)
plt.legend()
plt.xlabel('Initial endowment', size = 20)
plt.ylabel('percentatge growth', size = 20)
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()

#%% Part sixth: 
# K = 0:
R = np.linspace(0.5,0.7,10)
A = np.ones(10)
u = 0
for r in R:
    Equilibrium = []
    for i in range(400):
        def rest(x):
            F =np.zeros(4)
            a = x[0]
            h0 = x[1]
            h1 = x[2]
            lamda = x[3]
            F[0]= np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)*(1-tau)*C[1][i] - kappa*np.power(h0,1/nu)
            F[1]= beta*np.power((((1-tau)*C[1][i]*h1)+(1+r)*a + T1), -sigma)*(1-tau)*C[1][i] - kappa*np.power(h1,1/nu)
            F[2]= beta*(np.power(((1-tau)*C[1][i]*h1)+(1+r)*a + T1,-sigma)*(1+r)) - lamda - np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)
            F[3]= ((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a) + (1/(1+r))*((1-tau)*(C[1][i]+C[2][i])*h1 + (1+r)*a + T1) - C[0][i] - (1+r)*(C[1][i]+C[2][i])*h1
            return F
        sguess= np.array([0.001,0.1,0.1, 1])
        sol = fsolve(rest,sguess)
        Equilibrium.append(sol)
        eq_mat = np.matrix(Equilibrium)
    shape(eq_mat)
    eq_mat = np.array(eq_mat)
    eq_mat = eq_mat.T
    a = eq_mat[0]
    A[u] = sum(a)
    u = u+1
plt.plot(R,A, label = 'demand excess asset vs initial wealth')
plt.legend()
plt.title('Asset demand excess vs initial wealth', size=20)
plt.ylabel('Excess demand', size = 20)
plt.xlabel('Interest rate', size = 20)
plt.axhline(y=0, color='black')
plt.show()


    

   