import numpy as np
from scipy.integrate import quad
from scipy.constants import physical_constants

#### Input Variables

Tbb = 7e4 # temperature of blackbody spectrum in K
nbins = 9
bins = 'log'


##### Sigma variables 
#ev 
epsilon0_HI=0.4298
epsilon0_HeI=0.1361
epsilon0_HeII=1.720

#m^2
sigma0_HI=5.475e-18
sigma0_HeI=9.492e-20
sigma0_HeII=1.369e-18

#
P_HI=2.963
P_HeI=3.188
P_HeII=2.963

#
ya_HI=32.88
ya_HeI=1.469
ya_HeII=32.88

#
yw_HI=0
yw_HeI=2.039
yw_HeII=0

#
y0_HI=0
y0_HeI=0.4434
y0_HeII=0

#
y1_HI=0
y1_HeI=2.136
y1_HeII=0


#####

mystr=str(' ')
mystr2=str(' ')
c = physical_constants['speed of light in vacuum'][0] * 1.0e2 # cm s^-1 
h = physical_constants['Planck constant in eV/Hz'][0]/(2*np.pi) # eV s
# Also the lower limits for intergral
Egy_HI = physical_constants['Rydberg constant times hc in eV'][0] # eV 
Egy_HeI= 24.59 #ev 
Egy_HeII= 54.42 #ev 

kB = physical_constants['Boltzmann constant in eV/K'][0] # eV K^-1 


# Below, "h" is actually hbar.  Change variable name later.

def bb(x):
	#blackbody spectrum
	return 2. * Tbb**3 * x**3 / h**2 / c**2 / (np.exp(x, dtype=np.float128) - 1.) 

def sigma_HI(x):
	epsilon=photon_egy(x)
	l=epsilon/epsilon0_HI-y0_HI
	y=(l**2+y1_HI**2)**0.5
		
	#HI photoionization cross section from Hui & Gnedin
	return sigma0_HI * ( ( l - 1.)**2+yw_HI**2) * y ** (0.5 * P_HI - 5.5) * (1. + np.sqrt(y/ya_HI))**(-1. * P_HI)

def sigma_HeI(x):
	epsilon=photon_egy(x)
	l=epsilon/epsilon0_HeI-y0_HeI
	y=(l**2+y1_HeI**2)**0.5
		
	#HeI photoionization cross section from Hui & Gnedin
	return sigma0_HeI * ( ( l - 1.)**2+yw_HeI**2) * y ** (0.5 * P_HeI - 5.5) * (1. + np.sqrt(y/ya_HeI))**(-1. * P_HeI)

def sigma_HeII(x):
	epsilon=photon_egy(x)
	l=epsilon/epsilon0_HeII-y0_HeII
	y=(l**2+y1_HeII**2)**0.5
		
	#HeII photoionization cross section from Hui & Gnedin
	return sigma0_HeII * ( ( l - 1.)**2+yw_HeII**2) * y ** (0.5 * P_HeII - 5.5) * (1. + np.sqrt(y/ya_HeII))**(-1. * P_HeII)

def photon_egy(x):
	return x * kB * Tbb

def int_Energy_nume(x, Emin):
	return bb(x) * (photon_egy(x) - Emin) / photon_egy(x)

def int_Energy_deno(x):
	return bb(x) / photon_egy(x)

def int_sigma_number(x,sig):
	if sig=="HI":
		return bb(x) * sigma_HI(x) / photon_egy(x)
	elif sig=="HeI":
		return bb(x) * sigma_HeI(x) / photon_egy(x)
	elif sig=="HeII":
		return bb(x) * sigma_HeII(x) / photon_egy(x)
	
def int_sigma_energy(x,sig):
	if sig=="HI":
		return bb(x) * sigma_HI(x)
	if sig=="HeI":
		return bb(x) * sigma_HeI(x)
	if sig=="HeII":
		return bb(x) * sigma_HeII(x)
		
def energy(bin_lower,mystr,mystr2):
	#In this we will calculate the average energy of "absorbed photons", photoionization cross section, and fraction of photons capable of ionizing with respect to each species for the specific frequency bin
	
	#For Hydrogen
	hnu_HI=Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],xrange[bin_lower+1],args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],xrange[bin_lower+1])[0]
	mystr+= 'hnu_HI['+str(bin_lower)+']='+str(hnu_HI)+'*1.6022e-19;' +"\n"
	
	alphai_HI=quad(int_sigma_number, xrange[bin_lower], xrange[bin_lower+1],args=('HI'))[0]/quad(int_Energy_deno, xrange[bin_lower], xrange[bin_lower+1])[0]
	mystr += 'alphai_HI['+str(bin_lower)+']='+str(alphai_HI)+'*c;'   +"\n"
	
	alphae_HI=quad(int_sigma_energy, xrange[bin_lower], xrange[bin_lower+1],args=('HI'))[0]/quad(bb, xrange[bin_lower], xrange[bin_lower+1])[0]
	mystr += 'alphae_HI['+str(bin_lower)+']='+str(alphae_HI)+'*c;'   +"\n"

	factgrp_HI = quad(int_Energy_deno, xrange[bin_lower],xrange[bin_lower+1])[0]/quad(int_Energy_deno, xrange[0],np.inf)[0]
	mystr2 += 'factgrp_HI['+str(bin_lower)+']='+str(factgrp_HI)+';' +"\n"
	
	#For HeI, the minimum energy should be greater than Egy_HeI for absorbtion to take place. If no absorption then no photons and average energy of absorbed photons will be 0. Also the integration will take place if the ionization energy lies in the middle of a particular bin. In that case the lower limit will be ionization energy of Helium and the upper limit we will get from the variable x.  
	
	if Egy[bin_lower+1]>=Egy_HeI:
		if Egy[bin_lower]>=Egy_HeI: # this means that the lower limit is above the ionization energy so the limits will be same as hydrogen
			hnu_HeI= Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],xrange[bin_lower+1],args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],xrange[bin_lower+1])[0]
			alphai_HeI= quad(int_sigma_number,xrange[bin_lower],xrange[bin_lower+1],args=('HeI'))[0]/quad(int_Energy_deno,xrange[bin_lower],xrange[bin_lower+1])[0]
			alphae_HeI= quad(int_sigma_energy,xrange[bin_lower],xrange[bin_lower+1],args=('HeI'))[0]/quad(bb,xrange[bin_lower],xrange[bin_lower+1])[0]
			factgrp_HeI = quad(int_Energy_deno, xrange[bin_lower],xrange[bin_lower+1])[0]/quad(int_Energy_deno, Egy_HeI/kB/Tbb,np.inf)[0]

		else : # the lower limit will be the ionization energy of Helium.
			hnu_HeI= Egy_HeI+quad(int_Energy_nume,Egy_HeI/ kB / Tbb,xrange[bin_lower+1],args=(Egy_HeI))[0]/quad(int_Energy_deno,Egy_HeI/ kB / Tbb,xrange[bin_lower+1])[0]
			alphai_HeI= quad(int_sigma_number,Egy_HeI/ kB / Tbb,xrange[bin_lower+1],args=('HeI'))[0]/quad(int_Energy_deno,Egy_HeI/ kB / Tbb,xrange[bin_lower+1])[0]
			alphae_HeI= quad(int_sigma_energy,Egy_HeI/ kB / Tbb,xrange[bin_lower+1],args=('HeI'))[0]/quad(bb,Egy_HeI/ kB / Tbb,xrange[bin_lower+1])[0]
			factgrp_HeI = quad(int_Energy_deno, Egy_HeI/ kB / Tbb,xrange[bin_lower+1])[0]/quad(int_Energy_deno, Egy_HeI/kB/Tbb,np.inf)[0]
			
		mystr+= 'hnu_HeI['+str(bin_lower)+']='+str(hnu_HeI)+ '*1.6022e-19;' +"\n"
		mystr+= 'alphai_HeI['+str(bin_lower)+']='+str(alphai_HeI)+'*c;' +"\n"
		mystr+= 'alphae_HeI['+str(bin_lower)+']='+str(alphae_HeI)+'*c;'  +"\n"
		mystr2 += 'factgrp_HeI['+str(bin_lower)+']='+str(factgrp_HeI)+';'  +"\n"
		
	else :
		mystr+= 'hnu_HeI['+str(bin_lower)+']=0;'  +"\n"# If the upper limit is lesser than the ionization energy then no absorption. 
		mystr+= 'alphai_HeI['+str(bin_lower)+']=0;'   +"\n"
		mystr+= 'alphae_HeI['+str(bin_lower)+']=0;'   +"\n"
		mystr2 += 'factgrp_HeI['+str(bin_lower)+']=0;' +"\n"
		
	# The same arguments as HeI but now the limits are corresponding to HeII
	
	if Egy[bin_lower+1]>Egy_HeII:
		if Egy[bin_lower]>Egy_HeII: # this means that the lower limit is above the ionization energy so the limits will be same as hydrogen
			hnu_HeII= Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],xrange[bin_lower+1],args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],xrange[bin_lower+1])[0]
			alphai_HeII= quad(int_sigma_number,xrange[bin_lower],xrange[bin_lower+1],args=('HeII'))[0]/quad(int_Energy_deno,xrange[bin_lower],xrange[bin_lower+1])[0]
			alphae_HeII= quad(int_sigma_energy,xrange[bin_lower],xrange[bin_lower+1],args=('HeII'))[0]/quad(bb,xrange[bin_lower],xrange[bin_lower+1])[0]
			factgrp_HeII = quad(int_Energy_deno, xrange[bin_lower],xrange[bin_lower+1])[0]/quad(int_Energy_deno, Egy_HeII/kB/Tbb,np.inf)[0]
		
		else : # the lower limit will be the ionization energy of Helium.
			hnu_HeII= Egy_HeII+quad(int_Energy_nume,Egy_HeII/ kB / Tbb,xrange[bin_lower+1],args=(Egy_HeII))[0]/quad(int_Energy_deno,Egy_HeII/ kB / Tbb,xrange[bin_lower+1])[0]
			alphai_HeII= quad(int_sigma_number,Egy_HeII/ kB / Tbb,xrange[bin_lower+1],args=('HeII'))[0]/quad(int_Energy_deno,Egy_HeII/ kB / Tbb,xrange[bin_lower+1])[0]
			alphae_HeII= quad(int_sigma_energy,Egy_HeII/ kB / Tbb,xrange[bin_lower+1],args=('HeII'))[0]/quad(bb,Egy_HeII/ kB / Tbb,xrange[bin_lower+1])[0]
			factgrp_HeII = quad(int_Energy_deno, Egy_HeII/ kB / Tbb,xrange[bin_lower+1])[0]/quad(int_Energy_deno, Egy_HeII/kB/Tbb,np.inf)[0]
			
		mystr+= 'hnu_HeII['+str(bin_lower)+']='+str(hnu_HeII)+ '*1.6022e-19;' +"\n"
		mystr+= 'alphai_HeII['+str(bin_lower)+']='+str(alphai_HeII)+'*c;'   +"\n"
		mystr+= 'alphae_HeII['+str(bin_lower)+']='+str(alphae_HeII)+'*c;'  +"\n"
		mystr2 += 'factgrp_HeII['+str(bin_lower)+']='+str(factgrp_HeII)+';'   +"\n"
	else :
		mystr+= 'hnu_HeII['+str(bin_lower)+']=0;'   +"\n"
		mystr+= 'alphai_HeII['+str(bin_lower)+']=0;'   +"\n"
		mystr+= 'alphae_HeII['+str(bin_lower)+']=0;'   +"\n"
		mystr2 += 'factgrp_HeII['+str(bin_lower)+']=0;' +"\n"
	
	return mystr, mystr2
	

def energy_in(bin_lower,mystr,mystr2):
	
	#For Hydrogen
	
	hnu_HI=Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],np.inf,args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],np.inf)[0]
	mystr+= 'hnu_HI['+str(bin_lower)+']='+str(hnu_HI)+'*1.6022e-19;' +"\n"
	
	alphai_HI=quad(int_sigma_number, xrange[bin_lower], np.inf,args=('HI'))[0]/quad(int_Energy_deno, xrange[bin_lower], np.inf)[0]
	mystr += 'alphai_HI['+str(bin_lower)+']='+str(alphai_HI)+'*c;'  +"\n"
	 
	alphae_HI=quad(int_sigma_energy, xrange[bin_lower], np.inf,args=('HI'))[0]/quad(bb, xrange[bin_lower], np.inf)[0]
	mystr += 'alphae_HI['+str(bin_lower)+']='+str(alphae_HI)+'*c;'   +"\n"
	
	factgrp_HI = quad(int_Energy_deno, xrange[bin_lower],np.inf)[0]/quad(int_Energy_deno, xrange[0],np.inf)[0]
	mystr2 += 'factgrp_HI['+str(iegy)+']='+str(factgrp_HI)+';' +"\n"
	
	if Egy[bin_lower]>=Egy_HeI: # this means that the lower limit is above the ionization energy so the limits will be same as hydrogen
		hnu_HeI= Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],np.inf,args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],np.inf)[0]
		alphai_HeI= quad(int_sigma_number,xrange[bin_lower],np.inf,args=('HeI'))[0]/quad(int_Energy_deno,xrange[bin_lower],np.inf)[0]
		alphae_HeI= quad(int_sigma_energy,xrange[bin_lower],np.inf,args=('HeI'))[0]/quad(bb,xrange[bin_lower],np.inf)[0]
		factgrp_HeI = quad(int_Energy_deno, xrange[bin_lower],np.inf)[0]/quad(int_Energy_deno, xrange[0],np.inf)[0]
	

	else : # the lower limit will be the ionization energy of Helium.
		hnu_HeI= Egy_HeI+quad(int_Energy_nume,Egy_HeI/kB/Tbb,np.inf,args=(Egy_HeI))[0]/quad(int_Energy_deno,Egy_HeI/kB/Tbb,np.inf)[0]
		alphai_HeI= quad(int_sigma_number,Egy_HeI/kB/Tbb,np.inf,args=('HeI'))[0]/quad(int_Energy_deno,Egy_HeI/kB/Tbb,np.inf)[0]
		alphae_HeI= quad(int_sigma_energy,Egy_HeI/kB/Tbb,np.inf,args=('HeI'))[0]/quad(bb,Egy_HeI/kB/Tbb,np.inf)[0]
		factgrp_HeI = quad(int_Energy_deno,Egy_HeI/kB/Tbb,np.inf)[0]/quad(int_Energy_deno, Egy_HeI/kB/Tbb,np.inf)[0]
	
	mystr+= 'hnu_HeI['+str(bin_lower)+']='+str(hnu_HeI)+ '*1.6022e-19;' +"\n"
	mystr+= 'alphai_HeI['+str(bin_lower)+']='+str(alphai_HeI)+'*c;' +"\n"
	mystr+= 'alphae_HeI['+str(bin_lower)+']='+str(alphae_HeI)+'*c;'   +"\n"
	mystr2 += 'factgrp_HeI['+str(bin_lower)+']='+str(factgrp_HeI)+';' +"\n"
	
		
			
	
	if Egy[bin_lower]>Egy_HeII: # this means that the lower limit is above the ionization energy so the limits will be same as hydrogen
		hnu_HeII= Egy[bin_lower]+quad(int_Energy_nume,xrange[bin_lower],np.inf,args=(Egy[bin_lower]))[0]/quad(int_Energy_deno,xrange[bin_lower],np.inf)[0]
		alphai_HeII= quad(int_sigma_number,xrange[bin_lower],np.inf,args=('HeII'))[0]/quad(int_Energy_deno,xrange[bin_lower],np.inf)[0]
		alphae_HeII= quad(int_sigma_energy,xrange[bin_lower],np.inf,args=('HeII'))[0]/quad(bb,xrange[bin_lower],np.inf)[0]
		factgrp_HeII = quad(int_Energy_deno, xrange[bin_lower],np.inf)[0]/quad(int_Energy_deno, xrange[0],np.inf)[0]
			
	else : # the lower limit will be the ionization energy of Helium.
		hnu_HeII= Egy_HeII+quad(int_Energy_nume,Egy_HeII/kB/Tbb,np.inf,args=(Egy_HeII))[0]/quad(int_Energy_deno,Egy_HeII/kB/Tbb,np.inf)[0]
		alphai_HeII= quad(int_sigma_number,Egy_HeII/kB/Tbb,np.inf,args=('HeII'))[0]/quad(int_Energy_deno,Egy_HeII/kB/Tbb,np.inf)[0]
		alphae_HeII= quad(int_sigma_energy,Egy_HeII/kB/Tbb,np.inf,args=('HeII'))[0]/quad(bb,Egy_HeII/kB/Tbb,np.inf)[0]
		factgrp_HeII = quad(int_Energy_deno,Egy_HeII/kB/Tbb,np.inf)[0]/quad(int_Energy_deno, Egy_HeII/kB/Tbb,np.inf)[0]
			
	mystr+= 'hnu_HeII['+str(bin_lower)+']='+str(hnu_HeII)+ '*1.6022e-19;' +"\n"
	mystr+= 'alphai_HeII['+str(bin_lower)+']='+str(alphai_HeII)+'*c;'   +"\n"
	mystr+= 'alphae_HeII['+str(bin_lower)+']='+str(alphae_HeII)+'*c;'   +"\n"
	mystr2 += 'factgrp_HeII['+str(bin_lower)+']='+str(factgrp_HeII)+';' +"\n"
	
	return mystr, mystr2
	


# The intervals will be same for all the species. The interval will start for taking into account the least value which is the ionization energy for Hydrogen. 

if bins == 'log':
	# log spaced bins up to 10 Ryd
	Egy = np.logspace(np.log10(Egy_HI), np.log10(10.*Egy_HI), nbins)
else:
	# linearly spaced bins up to HeII ionization energy
	Egy = np.linspace(Egy_HI, 4*Egy_HI, nbins) 


xrange = Egy / kB / Tbb  # do integration in terms of this variable


for iegy in np.arange(len(Egy)):
	if iegy < len(Egy)-1:
		print("Energy",iegy,".",Egy[iegy])
		mystr,mystr2=energy(iegy,mystr,mystr2)
			
	else:
		print("Energy",iegy,".",Egy[iegy])
		mystr,mystr2=energy_in(iegy,mystr,mystr2)

		        
print (mystr)
print (mystr2)


