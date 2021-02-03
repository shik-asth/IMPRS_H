#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code we will be studying the temperature dependence of various parameters 
used in the differential equations that are solved by ATON.

A file will be taken to read which plots have to be created.

The arrangement will be straightforward. A function to calculate the value of the 
equation.

Another function will be used to create the plot. 

@author: shikhar
"""

import sys
import Plots

f=open(sys.argv[1],"r")
#Read all lines in the file. Since we know the order of the arguments we will write
#the code in that manner.
plotdata=[]
for line in f:
    li=line.strip()
    if not li.startswith("#"): #Checks which lines are commented out
        plotdata.append(list(li.split(" ")))#plotdata has now all the information regarding which plots to make

for counter in range(0,len(plotdata)):
       
    if counter==0:
        Plots.single_plot(plotdata[counter])
    else :
     
        Plots.multi_plot(plotdata[counter])
        
