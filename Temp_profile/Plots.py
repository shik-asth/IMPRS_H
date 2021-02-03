#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the functions that will be used to plot the curves.
It has two functions in it. One is single_plot and the other is multi_plot
@author: shikhar
"""

import Temp_formulas as tf
import numpy as np
import matplotlib.pyplot as plt

def single_plot(plotkey):
    
    for pt in plotkey:
        
        if int(pt)!=0:
            name=tf.whichplot(int(pt))
            value=getattr(tf,name[0])(0,pt,0) #The first argument is used to distinguish what part of the called function is run. The last variable is the inputs to add curves 
       
            plt.figure()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('log(Temp in K)')
            plt.ylabel(name[0]) 
            plt.plot(value[0],value[1],label=name[0]+"_"+str(value[2]))
            plt.legend(loc="upper right")
            plt.savefig(pt+"_"+name[0]+"_"+str(value[2]))
            plt.show()
        
def multi_plot(plotkey):
    
    filename=""
    
    for pt in plotkey:
        
        if pt!='add' and pt!='subtract':
            name=tf.whichplot(int(pt))
            value=getattr(tf,name[0])(0,pt,0)
            filename=filename+pt+"_"
           
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('log(Temp in K)')
            plt.plot(value[0],value[1],label=name[0]+"_"+str(value[2]))
            plt.legend(loc="upper left")
            
            
        elif pt=='add':
            upperlimit_temp=[]
            lowerlimit_temp=[]
           
            filename="add_"
            for pt in plotkey:
                if pt!='add':
                   name=tf.whichplot(int(pt)) 
                   value=getattr(tf,name[0])(1,pt,0)
                   
                   upperlimit_temp.append(value[0])
                   lowerlimit_temp.append(value[1])
                   
            
            upperlimit_temp.sort() 
            upperlimit=upperlimit_temp[-1]
            
            lowerlimit_temp.sort()
            lowerlimit=lowerlimit_temp[0]
          
            steps=1000
            resolution=(upperlimit-lowerlimit)/steps
            
            if lowerlimit<100:
                lowerlimit=lowerlimit+resolution
                resolution=(upperlimit-lowerlimit)/steps
            
            add_data=[lowerlimit,upperlimit,steps]
            X=np.arange(lowerlimit,upperlimit,resolution)
            Y=np.zeros(len(X))
            
            for pt in plotkey:
                if pt!='add':
                    name=tf.whichplot(int(pt))
                    filename=filename+pt+"_"
                    value=getattr(tf,name[0])(2,pt,add_data)
                    Y=Y+value
                    
            
                           
            plt.xscale('log')
            plt.xlabel('log(Temp in K)')
            plt.yscale('log')
            plt.plot(X,Y,label=filename+str(steps))
            plt.legend(loc="upper left")
            break         
        
        elif pt=='subtract':
            upperlimit_temp=[]
            lowerlimit_temp=[]
           
            filename="subtract_"
            for pt in plotkey:
                if pt!='subtract':
                   name=tf.whichplot(int(pt)) 
                   value=getattr(tf,name[0])(1,pt,0)
                   
                   upperlimit_temp.append(value[0])
                   lowerlimit_temp.append(value[1])
                   
            
            upperlimit_temp.sort() 
            upperlimit=upperlimit_temp[-1]
            
            lowerlimit_temp.sort()
            lowerlimit=lowerlimit_temp[0]
          
            steps=1000
            resolution=(upperlimit-lowerlimit)/steps
            
            if lowerlimit<100:
                lowerlimit=lowerlimit+resolution
                resolution=(upperlimit-lowerlimit)/steps
            
            add_data=[lowerlimit,upperlimit,steps]
            X=np.arange(lowerlimit,upperlimit,resolution)
            Y=np.zeros(len(X))
            l=0
            for pt in plotkey:
                if pt!='subtract':
                    if l==0:
                        name=tf.whichplot(int(pt))
                        filename=filename+pt+"_"
                        value=getattr(tf,name[0])(2,pt,add_data)
                        Y=Y+value
                        l=l+1
                    elif l==1:
                        name=tf.whichplot(int(pt))
                        filename=filename+pt+"_"
                        value=getattr(tf,name[0])(2,pt,add_data)
                        Y=Y-value
            
                           
            plt.xscale('log')
            plt.xlabel('log(Temp in K)')
           # plt.yscale('log')
            plt.plot(X,Y,label=filename+str(steps))
            plt.legend(loc="upper left")
            break    
    plt.savefig(filename)
    plt.show()
        
