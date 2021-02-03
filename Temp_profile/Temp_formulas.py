#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file we will have the equations for all coefficients. 
They are arranged in the same way as the input_temp. All temperature is in
Kelvin
@author: shikhar
"""
import numpy as np
import math
THI=157807
THeI=285335
THeII=631515
kb=1.3807e-16 #in cgs units
#Defining a dictionary in which I will have the input values and 
#the rest of the stuff will be the name, lower temp limit, upper temp limit and accuracy
#From cases where no information is given we will take default Upper,lower and accuracy defined in def_data.

def_data=[10,10**9,5]
    
datadict={
    1:["HIalpha_A",[1,10**9,2]],
    2:["HeIalpha_A",[1,10**9,10]],
    3:["HeIIalpha_A",[1,10**9,2]],
    4:["HI_CR_RC_A",[3,10**9,2]],
    5:["HeI_CR_RC_A",[5*(10**3),5*(10**5),10]],
    6:["HeII_CR_RC_A",[1,(10**9),2]],
    7:["HIalpha_B",[1,(10**9),0.7]],
    8:["HeIalpha_B",[1,10**9,10]],
    9:["HeIIalpha_B",[1,10**9,2]],
    10:["HI_CR_RC_B",[1,(10**9),2]],
    11:["HeI_CR_RC_B",[5*(10**3),5*(10**5),10]],
    12:["HeII_CR_RC_B",[3,10**9,2]],
    13:["HI_CI",[10**4,10**9,3]],
    14:["HeI_CI",[10**4,10**9,3]],
    15:["HeII_CI",[10**4,10**9,3]],
    16:["HeII_Xi",[3*(10**4),10**6,5]],
    17:["HI_CR_CI",[10**4,10**9,3]],
    18:["HeI_CR_CI",[10**4,10**9,3]],
    19:["HeII_CR_CI",[10**4,10**9,3]],
    20:["HeII_CR_xi",[3*(10**4),10**6,5]],
    21:["HI_CR_LE",[5*(10**3),5*(10**5),10]],
    22:["HeII_CR_LE",[5*(10**3),5*(10**5),10]],
    23:["Bremsstrahlung",[10**5,10**9,10]],
    24:["HI_CR_CI_ATON",[10**4,10**9,3]],
    25:["HI_RC_CR_A_ATON",[3,10**9,2]],
    26:["HI_RC_CR_B_ATON",[1,(10**9),2]],
    27:["Bremsstrahlung_ATON",[10**5,10**9,10]],
    28:["HI_collisional_excitation_ATON",[5*(10**3),5*(10**5),10]],
    }

def whichplot(plot_key):
    
    curve_name=datadict.get(plot_key)
  
    return curve_name
#1
def HIalpha_A(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=1.269*math.pow(10,-13)*math.pow(lambdaHI, 1.503)/math.pow((1+math.pow(lambdaHI/0.522,0.470)),1.923)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     
#2           
def HeIalpha_A(s,plot_key,data_add):
      
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=3*math.pow(10,-14)*math.pow(lambdaHeI,0.654)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
         
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#3   
def HeIIalpha_A(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=2*1.269*math.pow(10,-13)*math.pow(lambdaHeII,1.503)/math.pow(1+math.pow(lambdaHeII/0.522,0.470),1.923)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#4  
def HI_CR_RC_A(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=1.778*math.pow(10,-29)*x*math.pow(lambdaHI,1.965)/math.pow(1+math.pow(lambdaHI/0.541,0.502),2.697)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#5    
def HeI_CR_RC_A(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=3*math.pow(10,-14)*math.pow(lambdaHeI,0.654)*x*kb
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#6
def HeII_CR_RC_A(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=8*1.778*math.pow(10,-29)*x*math.pow(lambdaHeII,1.965)/math.pow(1+math.pow(lambdaHeII/0.541,0.502),2.697)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])

        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#7
def HIalpha_B(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=2.753*math.pow(10,-14)*math.pow(lambdaHI, 1.500)/math.pow((1+math.pow(lambdaHI/2.740,0.407)),2.242)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
       
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#8          
def HeIalpha_B(s,plot_key,data_add):
      
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=1.26*math.pow(10,-14)*math.pow(lambdaHeI,0.750)
        
        return value
   
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#9   
def HeIIalpha_B(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=2*2.753*math.pow(10,-14)*math.pow(lambdaHeII,1.500)/math.pow(1+math.pow(lambdaHeII/2.740,0.407),2.242)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#10 
def HI_CR_RC_B(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=3.435*math.pow(10,-30)*x*math.pow(lambdaHI,1.970)/math.pow(1+math.pow(lambdaHI/2.250,0.376),3.720)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#11
def HeI_CR_RC_B(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=1.26*math.pow(10,-14)*math.pow(lambdaHeI,0.750)*x*kb
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])

        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#12
def HeII_CR_RC_B(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=8*3.435*math.pow(10,-30)*x*math.pow(lambdaHeII,1.970)/math.pow(1+math.pow(lambdaHeII/2.250,0.376),3.720)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#13
def HI_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=21.11*math.pow(x,-3.2)*math.exp(-lambdaHI/2)*math.pow(lambdaHI,-1.089)/math.pow(1+math.pow(lambdaHI/0.354,0.874),1.101)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#14
def HeI_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=32.38*math.pow(x,-3/2)*math.exp(-lambdaHeI/2)*math.pow(lambdaHeI,-1.146)/math.pow(1+math.pow(lambdaHeI/0.416,0.987),1.056)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#15
def HeII_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=19.95*math.pow(x,-3/2)*math.exp(-lambdaHeII/2)*math.pow(lambdaHeII,-1.089)/math.pow(1+math.pow(lambdaHeII/0.553,0.735),1.275)
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#16
def HeII_Xi(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=1.90*math.pow(10,-3)*math.pow(x,-3/2)*math.exp(-0.75*lambdaHeII/2)*(1+0.3*math.exp(-0.15*lambdaHeII/2))
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#17
def HI_CR_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=21.11*math.pow(x,-3/2)*math.exp(-lambdaHI/2)*math.pow(lambdaHI,-1.089)/math.pow(1+math.pow(lambdaHI/0.354,0.874),1.101)*kb*THI
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#18
def HeI_CR_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeI=2*THeI/x
        value=32.38*math.pow(x,-3/2)*math.exp(-lambdaHeI/2)*math.pow(lambdaHeI,-1.146)/math.pow(1+math.pow(lambdaHeI/0.416,0.987),1.056)*kb*THeI
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#19
def HeII_CR_CI(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=19.95*math.pow(x,-3/2)*math.exp(lambdaHeII/2)*math.pow(lambdaHeII,-1.089)/math.pow(1+math.pow(lambdaHeII/0.553,0.735),1.275)*kb*THeII
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#20
def HeII_CR_xi(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=0.75*1.90*math.pow(10,-3)*math.pow(x,-3/2)*math.exp(-0.75*lambdaHeII/2)*(1+0.3*math.exp(-0.15*lambdaHeII/2))*kb*THeII
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#21
def HI_CR_LE(s,plot_key,data_add):
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHI=2*THI/x
        value=7.5*math.pow(10,-19)*math.exp(-0.75*lambdaHI/2)/(1+math.pow(x/(10**5),1/2))
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y
#22
def HeII_CR_LE(s,plot_key,data_add):
    name=datadict.get(int(plot_key))
    
    def function_value(x):
        lambdaHeII=2*THeII/x
        value=5.54*math.pow(10,-17)*math.pow(x,-0.397)*math.exp(-0.75*lambdaHeII/2)/(1+math.pow(x/(10**5),1/2))
        
        return value
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y

   
#23
def Bremsstrahlung(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=1.42*10**(-27)*math.pow(x, 0.5)
                
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     

#24
def HI_CR_CI_ATON(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=math.exp(-157809.1/x)*1.27*math.pow(10,-21)*math.pow(x,1/2)/(1+math.pow(x/10**5,1/2))
                
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     

#25
def HI_RC_CR_A_ATON(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=1.778*math.pow(10,-29)*x*math.pow(2*157807/x,1.965)/math.pow(1+math.pow(2*157807/x/0.541,0.502),2.697)                
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     
        
        
#26
def HI_RC_CR_B_ATON(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=3.435*math.pow(10,-30)*x*math.pow(2*157807/x,1.970)/math.pow(1+(math.pow(2*157807/x/2.250,0.376)),3.72)                
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     
    
#27
def Bremsstrahlung_ATON(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=1.42*10**(-27)*math.pow(x, 0.5)
                
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     
        
#28
def HI_collisional_excitation_ATON(s,plot_key,data_add):
    
    
    name=datadict.get(int(plot_key))
    
    def function_value(x):
      
        value=math.exp(-118348/x)*7.5*10**(-19)/(1+math.pow(x/10**5,1/2))
               
        return value
    
    if s==0: 
        steps=1000
       
        X=np.logspace(np.log10(name[1][0]),np.log10(name[1][1]),steps)#X is the temperature
        Y=np.zeros(steps)        
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return X,Y,steps
    
    elif s==1:
        
        return name[1][1],name[1][0]
    
    elif s==2:
        
        X=np.logspace(np.log10(data_add[0]),np.log10(data_add[1]),data_add[2])
        Y=np.zeros(data_add[2])
        
        for l in X:
            Y[list(X).index(l)]=function_value(l)
            
        return Y     
        
        

        
