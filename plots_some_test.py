import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])

#from math import *


def prvi(k,m,om,t,g):
	return (k/(m*om**2*(g**2-om**2)))*(g**2-om**2+sy.exp(-2.0*g*t)*(om**2-g**2*sy.cosh(2.0*sy.sqrt(g**2-om**2)*t)-g*sy.sqrt(g**2-om**2)*sy.sinh(2.0*sy.sqrt(g**2-om**2)*t)))

def drugi(x,t,om,g):
	return (x/(g**2-om**2))*sy.exp(-2.0*g*t)*(-om**2*sy.cosh(sy.sqrt(g**2-om**2)*t)**2+g**2*sy.cosh(2.0*sy.sqrt(g**2-om**2)*t)+g*sy.sqrt(g**2-om**2)*sy.sinh(2.0*sy.sqrt(g**2-om**2)*t))

def treci(p,m,t,om,g):
	return (p/(m**2*(g**2-om**2)))*sy.exp(-2.0*g*t)*sy.sinh(sy.sqrt(g**2-om**2)*t)**2

def cetvrti(s,m,t,om,g):
	return (s/(2.0*m*(g**2-om**2)))*sy.exp(-2.0*g*t)*(2.0*g*sy.sinh(sy.sqrt(g**2-om**2)*t)**2+sy.sqrt(g**2-om**2)*sy.sinh(2.0*sy.sqrt(g**2-om**2)*t))

def funkcija(x,p,s,k,m,t,om,g):
	return prvi(k,m,om,t,g)+ drugi(x,t,om,g) + treci(p,m,t,om,g) + cetvrti(s,m,t,om,g)


def klasicna(k,m,om,t,g):
	return (k/(m*om**2*(g**2-om**2)))*(g**2-om**2+sy.exp(-2.0*g*t)*(om**2-g**2*sy.cosh(2.0*t*sy.sqrt(g**2-om**2))-g*sy.sqrt(g**2-om**2)*sy.sinh(2.0*t*sy.sqrt(g**2-om**2))))

x=10**(-7)
p=10**7
s=0.01

om=10.0
m=10.0
t=10.0
k=0.1

import sympy as sy





g = sy.symbols("g", real=True)

z=funkcija(x,p,s,k,m,t,om,g)
y = (k/(m*om**2*(g**2-om**2)))*(g**2-om**2+sy.exp(-2.0*g*t)*(om**2-g**2*sy.cosh(2.0*t*sy.sqrt(g**2-om**2))-g*sy.sqrt(g**2-om**2)*sy.sinh(2.0*t*sy.sqrt(g**2-om**2))))


xdata=np.linspace(0.01,100000000,1000001)
ydata=[]
zdata=[]

for i in xdata:
	print(i)
	ydata.append(y.subs(g, i).evalf())
	zdata.append(z.subs(g, i).evalf())


plt.figure()
plt.loglog(xdata,ydata,color='red',label='classical')
plt.loglog(xdata,zdata,color='blue',label='exact')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\langle x^{2} \rangle$')
plt.legend(loc='best')
plt.savefig('fig.pdf',dpi=400)
plt.show()
"""
plt.figure()
#plt.loglog(data,funkcija(10**(-7),10**7,0.01,0.1,10.0,10.0,10.0,data))
plt.loglog(data,klasicna(0.1,10.0,10.0,10.0,data))
plt.show()
"""