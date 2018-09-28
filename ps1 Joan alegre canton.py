#%% Ps1 Joan alegre cantón, with help of german and Maria.
from sympy import diff;
from sympy import symbols;
import numpy as np;
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
from matplotlib.pyplot import xlim
from matplotlib.pyplot import legend
from matplotlib.pyplot import title
from scipy.interpolate import interp1d
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.chebyshev import chebroots
from numpy.polynomial.chebyshev import chebval2d
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import math as mt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sympy import solve

#factorial function:
def factorial(n):
    o = 1
    while n>=1:
        o = o*n
        n -= 1
    return o
#general function of taylor's expansion:

def taylor(f,a,n):
    i = 0
    t = 0
    while i<=n:
        t = t+(((f.diff(x,i).subs(x,a))*(x-a)**i)/factorial(i))
        i += 1
    return t

#Finding taylor expressions for f.
x = symbols('x')
f = x**.321
a = 1
t1 = taylor(f,a,1)
t2 = taylor(f,a,2)
t5 = taylor(f,a,5)
t20 = taylor(f,a,20)

# Stating function expressions and domain to plot them all:
x = np.linspace(0,4,50)
ylim(top=7)
ylim(bottom=-2)
t1 = 0.321*x + 0.679
t2 = 0.321*x - 0.1089795*(x - 1)**2 + 0.679
t5 = 0.321*x + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
t20 = 0.321*x - 0.00465389246518441*(x - 1)**20 + 0.00498302100239243*(x - 1)**19 - 0.00535535941204005*(x - 1)**18 + 0.00577951132662155*(x - 1)**17 - 0.00626645146709397*(x - 1)**16 + 0.00683038514023459*(x - 1)**15 - 0.00749000490558658*(x - 1)**14 + 0.0082703737422677*(x - 1)**13 - 0.00920582743809231*(x - 1)**12 + 0.0103445949299661*(x - 1)**11 - 0.0117564360191783*(x - 1)**10 + 0.0135458417089277*(x - 1)**9 - 0.0158761004532294*(x - 1)**8 + 0.0190161406836106*(x - 1)**7 - 0.0234395113198229*(x - 1)**6 + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
f = x**.321
#ion
plt.plot(x, t1, label='t1')
plt.plot(x, t2, label='t2')
plt.plot(x, t5, label='t5')
plt.plot(x, t20, label='t20')
plt.plot(x, f, label='$x^{0.321}$')
plt.legend()
plt.title('taylor expansion aproximation of $f(x)=x^{0.321}$ at $x=1$')
plt.show()
print('We can see that any taylor aproximation is quite good for values near to 1')
print('Nevertheless, the more far away we go from 1. the worse is the aproximation')
print('Beside if error will grow up strongly when we increase the degrees of the taylor')
print('at the same time that we go away from 1.')
print('Hence, may be, it is a good thing to use more degrees in order to aproximate a point of f')
print('But in this case is not a good idea increase the degrees if what we want is aproximate')
print('the whole function')
#%%
#Exercice 2:
#ramp function:
x = symbols('x')
f = x
a = 2
#finding taylor's expansions:
t1 = taylor(f,a,1)
t2 = taylor(f,a,2)
t5 = taylor(f,a,5)
t20 = taylor(f,a,20)

#taylor expressions and plots:
x = np.linspace(-4,6,30)
f = (x+abs(x))/2
t1 = x
t2 = x
t5 = x
t20 = x

plt.plot(x, t1, label='t1')
plt.plot(x, t2, label='t2')
plt.plot(x, t5, label='t5')
plt.plot(x, t20, label='t20')
plt.plot(x, f, label='ramp function')
plt.legend()
plt.title('taylor expansion aproximation of ramp function at $x=2$')
plt.show()
print('We can see that any taylor aproximation is perfect for values near 2')
print('Nevertheless, since we have a sudden kink in x = 0, we have that taylor aproximations')
print('no matter the degree they are will generate and increasing error of aproximation, since')
print('the actual function is slope 0 and all the taylor expansions are slope 1.')
#%% 3. Evenly spaced interpolation nodes and a cubic polynomial

# Import packages
from sympy import symbols
from sympy import diff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
import math
from scipy import interpolate
from scipy.interpolate import interp1d

#%% First function:

# Interpolations with monomials:

x = np.linspace(-1, 1, num = 20, endpoint = True) # If we increase the range, we could see better the behaviour of that function
y = np.exp(1/x)

# First part. Interpolations with monomials:

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return np.exp(1/x)

vector = np.linspace(-1, 1, num=20, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2, 5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomial:

def y(x):
    return np.exp(1/x)

vector = np.linspace(-1, 1, num=20, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

# With chebfit we obtain the coefficients of the Chevyshev polynomial, and chebval constructs the polynomial

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 1', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% Second function:

# First part. Interpolations with monomials:

x = np.linspace(-1, 1, num = 40, endpoint = True)
y = 1/(1+25*x**2)

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, 'o', label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return 1/(1+25*x**2)

vector = np.linspace(-1, 1, num=40, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2 ,5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomial:

def y(x):
    return 1/(1+25*x**2)

vector = np.linspace(-1, 1, num=40, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left', fontsize = 9)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 2', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.ylim(ymin = 0, ymax = 0.55)
plt.title('Errors Chebyshev - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% Third function:

# First part. Interpolations with monomials:

x = np.linspace(-1, 1, num = 40, endpoint = True)
y = (x+abs(x))/2

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, 'o', label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return (x+abs(x))/2

vector = np.linspace(-1, 1, num=40, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2, 5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomials:

def y(x):
    return (x+abs(x))/2

vector = np.linspace(-1, 1, num=40, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 3', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chevyshev - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%%
# Exercice 4
print('Exercice 4')
#p = 5
e = mt.exp(1)
x0 = np.linspace(0,10,15)
x = np.linspace(0,10,400)
p1 = 1/(5*(e**x0)+0.01)
options = (3,5,10)
p = 1/(5*(e**x)+0.01)
plt.plot(x,p, label='f(x)')
for i in options:
        c = chebfit(x0,p1,i)
        f = chebval(x,c)
        plt.plot(x,f, label = i)
plt.legend()
plt.title('chebicheff when $p1=1/0.2$',size=20)
plt.show()
#p = 4
e = mt.exp(1)
x0 = np.linspace(0,10,15)
x = np.linspace(0,10,400)
p1 = 1/(4*(e**x0)+0.01)
options = (3,5,10)
p = 1/(4*(e**x)+0.01)
plt.plot(x,p, label='f(x)')
for i in options:
        c = chebfit(x0,p1,i)
        f = chebval(x,c)
        plt.plot(x,f, label = i)
plt.legend()
plt.title('chebicheff when $p1=1/0.25$',size=20)
plt.show()

#errors
#errors of p1 = 5
e = mt.exp(1)
x0 = np.linspace(0,10,15)
x = np.linspace(0,10,400)
p1 = 1/(5*(e**x0)+0.01)
g = 1/(5*(e**x)+0.01)
options = (3,5,10)
for i in options:
        c = chebfit(x0,p1,i)
        f = chebval(x,c)
        n = g-f
        plt.plot(x,n, label = i)
plt.legend()
plt.title('errors with $p1=1/0.2$',size=20)
plt.show()

#errors of p1 = 4
e = mt.exp(1)
x0 = np.linspace(0,10,15)
x = np.linspace(0,10,400)
p1 = 1/(4*(e**x0)+0.01)
g = 1/(4*(e**x)+0.01)
options = (3,5,10)
for i in options:
        c = chebfit(x0,p1,i)
        f = chebval(x,c)
        n = g-f
        plt.plot(x,n, label = i)
plt.legend()
plt.title('errors with $p1=1/0.25$',size=20)
plt.show()

# Question 2:
#second part question 2
print('Question 2 second part: labour sharein function of capital per capita')
k = symbols('k')
share = 1/(3*(k**(-3))+1)
k = np.linspace(0,10,500)
share = 1/(3*(k**(-3))+1)
plt.plot(k,share, label='share function')
plt.legend()
plt.title('share of labour on the economy with respect to capital per capita', size=20)
plt.ylabel('% of share', size =20)
plt.xlabel('capital per capita', size=20)
#%%
#third part question 2:
print('Question 2 third part: Chebycheff aproximation to a CES')
#I will make a grid of 4x5 and it will be evenly spaced
x = np.linspace(-1,1,4)
y = np.linspace(-1,1,5)
xx, yy = np.meshgrid(x, y)

 
def vect(n,x):    
    z = np.ones(n-2)
    g = [1,x]
    vector = np.append(g,z) # vector of 1,x,2*x**2-1,...
    if n==1:# for n lower than 2 we use these choices.
        vector = 1
    else:
        if n==2:
            vector = [1,x]
        else:
            for i in range(2,n):#when n larger than 2 we compute the rest of the vector like a fibbonachi sequence.
                vector[i] = 2*x*vector[i-1]-vector[i-2]                        
    return vector
n = 3
def cheb2d(x,y,n):#x and y are coordenates of the cartesian space.
    T=len(x)
    Ax = np.transpose(np.matrix(vect(n,x[0])))# this is the chebycheff polynomial for coordinate y
    Ay = np.matrix(vect(n,y[0]))# this is the chebycheff polynomial for coordinate x
    A = np.array(np.matrix.flatten(np.multiply(Ax,Ay)))# All posible combinations of both polynomials
    for t in range(1,T):
        cx = np.transpose(np.matrix(vect(n,x[t])))
        cy = np.matrix(vect(n,y[t]))
        cA = np.matrix.flatten(np.multiply(cx,cy))
        cA = np.array(cA)
        A = np.vstack((A,cA))# vector of chebicheff parts of polynomials for an individual t.
    return A #Dimension of matrix is Individuals X n^2 

#Chebycheff nodes  
x = np.ones(4)
for k in range(1,5):
    x[k-1]=5+5*mt.cos(((2*k-1)*mt.pi)/(2*4))#Chebycheff nodes function    
y = np.ones(5)
for k in range(1,6):
    y[k-1]=5+5*mt.cos(((2*k-1)*mt.pi)/(2*5))#Chebycheff nodes function
xx, yy = np.meshgrid(x, y)
xx = xx.reshape(20,1)# X coordinate
yy = yy.reshape(20,1)# Y coordinate
D = cheb2d(xx,yy,n)#Explicative variables.
f = ((3/4)*(xx)**(-3)+((1/4)*(yy)**(-3)))**(-(1/3))
model = sm.OLS(f,D)
results = model.fit()
p = results.params

x = np.linspace(0,10,40)
y = np.linspace(0,10,40)
xx, yy = np.meshgrid(x, y)
f = ((3/4)*(xx)**(-3)+((1/4)*(yy)**(-3)))**(-(1/3))
w = chebval2d(xx,yy,p)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, f, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
print('Production function CES, axes X = K, axes Y = L and axes Z = Output')

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, w, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
print('Production function CES chebycheff aproximation, axes X = K, axes Y = L and axes Z = Output')

# Question 2 isoquants:
print('Question 2 isoquants')
h=symbols('h')
k=symbols('k')
a=.5
s=.25

y=((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))

def f(k,h):
    return ((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))
max= f(10,10)

per5=max*0.05 #percentil 5
hs=solve(y-0.5,h)



k=np.linspace(0,10,num=100,endpoint=True)
plt.title('Output percentiles',size=20)
plt.plot(k,((0.5**3)/2-k**3)**(1/3),label='P5') #per 5
plt.plot(k,((1**3)/2-k**3)**(1/3),label='P10') #per10
plt.plot(k,((2.5**3)/2-k**3)**(1/3),label='P25') #per25
plt.plot(k,((5**3)/2-k**3)**(1/3),label='P50') #per50
plt.plot(k,((7.5**3)/2-k**3)**(1/3),label='P75') #per75
plt.plot(k,((9**3)/2-k**3)**(1/3),label='P90') #per90
plt.plot(k,((9.5**3)/2-k**3)**(1/3),label='P95') #per95
plt.legend(loc='upper right')
plt.xlabel('h',size=15)
plt.ylabel('k',size=15)
plt.show()
