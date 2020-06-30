from scipy.integrate import odeint
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import fludata

exec(compile(open("fludata.py", "rb").read(), "fludata.py", 'exec'))

# The SIR model differential equations.
def sir_model(y, t, params):
    S, I, R = y
    dSdt = -params[1] * S * I / (params[0]*N)
    dIdt = params[1] * S * I / (params[0]*N) - params[2] * I
    dRdt = params[2] * I
    return (dSdt, dIdt, dRdt)

def run_model(init, p):
    f = lambda y,t: sir_model(y, t, p)
    y0 = p[0]*N, init, 0
    sol = odeint(f, y0, t)
    S, I, R = sol.T
    return I,t

def error_fn(real,model):
    error = real-model
    # Since the peak week of a epidemic is an important feature, 
    # we want to make sure this value is accurate
    error[np.argmax(real)]=error[np.argmax(real)]*1000
    return error

def real_mod_comp(p,real):
    modelI,t = run_model(real[0], p)
    err=error_fn(real,modelI)
    # we want to make sure our parameters are biologically plausible
    if p[0]<0 or p[1]<0 or p[2]<0 or p[0]>1:
        err = err+10**6
    return err

import warnings
warnings.filterwarnings("ignore")

N=300000000 #approximate US pop.
t = np.linspace(0, 52, 52)
p0=[.5,.7, 1./14]

#get parameters from fitting to last season
fn = lambda p: real_mod_comp(p,seasons[-2])
(c,kvg) = optimize.leastsq(fn, p0) 
p_prev=c

I,t=run_model(seasons[-1][0], p_prev)

fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
y_pos = np.arange(len(weeks))
plt.xticks(y_pos, weeks)

plt.plot(I, label="Previous Season fit SIR", linestyle='-', linewidth=3, color = 'blue')
plt.plot(seasons[-1],label="real data", linewidth=3,color = 'black')
plt.title("Predictions from Fitting Previous Season",fontsize=30)
plt.legend(prop={'size': 25})
plt.yticks(fontsize=14)
plt.xlabel("Weeks",fontsize=30)
plt.ylabel("Number of New Cases",fontsize=30)
plt.show()

# since we now have only 10 weeks of data, we need to change the error functions
def error_fn_partial(real,model):
    error = real-model
    error[-1]= error[-1]*100 #weight last data point
    return error
def real_mod_comp_partial(p,real):
    c=0
    if p[0]<0 or p[1]<0 or p[2]<0:
        c = 10**10
    model,t = run_model(real[0], p)
    err=error_fn_partial(real,model)+c
    return err

N=300000000
p0=[.5,.7, 1./14]

fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
y_pos = np.arange(len(weeks))
plt.xticks(y_pos, weeks)
plt.plot(seasons[-1],label="real data", linewidth =3,color = 'black')

numwks=10
t = np.linspace(0, numwks,numwks)
fn = lambda p: real_mod_comp_partial(p,seasons[-1][0:numwks])
(c,kvg) = optimize.leastsq(fn, p_prev) 

t = np.linspace(0, 52, 52)
I2,t=run_model(seasons[-1][0], c)
plt.plot(I2, label="Smart 10wks SIR", linestyle='-.', linewidth =3, color='aqua')

t = np.linspace(0, numwks,numwks)
fn = lambda p: real_mod_comp_partial(p,seasons[-1][0:numwks])
(c,kvg) = optimize.leastsq(fn, p0) 

t = np.linspace(0, 52, 52)
I3,t=run_model(seasons[-1][0], c)
plt.plot(I3, label="Naive 10wks SIR", linestyle='--', linewidth =3, color = 'navy')
plt.legend()

plt.plot(I, label="Previous Season fit SIR", linestyle='-', linewidth =3,color = 'blue')
plt.title("Comparing SIR Fits",fontsize=30 )
plt.legend(prop={'size': 20})
plt.yticks(fontsize=14)
plt.xlabel("Weeks",fontsize=30)
plt.ylabel("Number of New Cases",fontsize=30)
plt.show()

print("Difference in Peak Value:")
print("Smart 10wks SIR:",abs(I2.max() - seasons[-1].max()))
print("Previous Season fit SIR:",abs(I.max() - seasons[-1].max()))
print("")
print("Difference in Peak Week:")
print("Smart 10wks SIR:",abs(np.argmax(I2) - np.argmax(seasons[-1])))
print("Previous Season fit SIR:",abs(np.argmax(I) - np.argmax(seasons[-1])))