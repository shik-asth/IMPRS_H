#include "params.h"
#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "GPU.h"
#include "bnd.h"
#include "Atomic.h"

#define ONE 0.9999999f
#define ZERO 0.0000001f
#define NCELL4 ((NCELLZ+NBOUND2)*(NCELLY+NBOUND2)*(NCELLX+NBOUND2))
#define TINY 1e-26
#define FLAGSOURCE 5.
#define KBOLTZ 1.3806e-23 //J/K is the unit
#define EPSCOOL 0.0001

#define T_HI 157807e0
#define T_HeI 285335e0
#define T_HeII 631515e0
#define REAL float
#define mass_p 1.6726219e-27 //in kg
#define mass_He 6.6464731e-27 // in kg
#define frac_He 0.25
#define frac_H 0.75
#define HI_ion 13.6*1.6022e-19
#define HeI_ion 24.59*1.6022e-19
#define HeII_ion 54.42*1.6022e-19
#define DEFAULT_LOW 1e-15 // in case ionization fraction becomes less than 0
#define DEFAULT_HIGH 0.9999 // in case ionization fraction becomes greater than 1

//**********************************************************************************
//**********************************************************************************

__device__ float cuCompute_FaceLF(float fl, float fr, float ul, float ur)
{
	return (fl+fr-ur+ul)*0.5f;
}



//**********************************************************************************
__global__ void cuComputeELF(float *cuegy, float *cuflx, float *cuegy_new, float c, float dx, float dt, int iter, float aexp, float egy_min)
{
	int tx=threadIdx.x+NBOUND;
	int bx=blockIdx.x +NBOUND;
	int by=blockIdx.y +NBOUND;

	REAL um1,up1,fm1,fp1,u0;


	REAL res;
	REAL dtsurdx=dt/dx;

	// Divergence along Z

	int baseidu=(tx)+bx*(NCELLX+NBOUND2);
	um1=cuegy[baseidu+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)];
	u0 =cuegy[baseidu+(by  )*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)];
	up1=cuegy[baseidu+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)];

	int baseidf=2*NCELL4+(tx)+bx*(NCELLX+NBOUND2);
	fm1=cuflx[baseidf+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)];
	fp1=cuflx[baseidf+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)];

	res=u0-0.5*((fp1-fm1)+c*(2*u0-um1-up1))*dtsurdx; // 10 FLOP

	// Divergence along Y

	baseidu=(tx)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);
	um1=cuegy[baseidu+(bx-1)*(NCELLX+NBOUND2)];
	u0 =cuegy[baseidu+(bx  )*(NCELLX+NBOUND2)];
	up1=cuegy[baseidu+(bx+1)*(NCELLX+NBOUND2)];

	baseidf=1*NCELL4+(tx)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);
	fm1=cuflx[baseidf+(bx-1)*(NCELLX+NBOUND2)];
	fp1=cuflx[baseidf+(bx+1)*(NCELLX+NBOUND2)];

	res=res-0.5*((fp1-fm1)+c*(2*u0-um1-up1))*dtsurdx; // 10 FLOP


	//Divergence along X

	__shared__ float u[NCELLX+NBOUND2],f[NCELLX+NBOUND2];

	baseidu=bx*(NCELLX+NBOUND2)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);
	baseidf=0*NCELL4+bx*(NCELLX+NBOUND2)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);

	u[tx]=cuegy [baseidu+tx];
	f[tx]=cuflx [baseidf+tx];



	if(tx-NBOUND==0) 
	{
		u[NBOUND-1]=cuegy[baseidu+tx-1];
		f[NBOUND-1]=cuflx[baseidf+tx-1];
	}

	if(tx-NBOUND==blockDim.x-1) 
	{
		u[NCELLX+NBOUND]=cuegy[baseidu+tx+1];
		f[NCELLX+NBOUND]=cuflx[baseidf+tx+1];
	}

	__syncthreads();

	res=res-0.5*((f[tx+1]-f[tx-1])+c*(2*u[tx]-u[tx+1]-u[tx-1]))*dtsurdx; // 10 FLOP

	cuegy_new[baseidu+tx]=fmaxf(res,egy_min);

}





//**********************************************************************************

__device__ float Eddington(float fx, float fy, float fz, float ee, float c,int i,int j)
{

	float c2e=ee*c*c; // 2 flop
	float ff=0.;
	float arg,chi,res=0.;
	float n[3];
	#ifdef ISOTROP

	if(i==j) res=1./3.;

	#else
	n[0]=0.;n[1]=0.;n[2]=0.;

	if(ee>0)
	{
		ff=sqrtf(fx*fx+fy*fy+fz*fz); // 6 flop
		if(ff>0)
		{
			n[0]=fx/ff; 
			n[1]=fy/ff;
			n[2]=fz/ff; 
		}
		ff=ff/(c*ee); // 2flop
	}

	arg=fmaxf(4.-3.*ff*ff,0.); // 4 flop
	chi=(3.+4.*ff*ff)/(5.+2.*sqrtf(arg)); // 7 flops

	if(i==j) res=(1.-chi)/2.*c2e; // 1 flops on average
	arg=(3.*chi-1.)/2.*c2e;
	res+=arg*n[i]*n[j];
	#endif

	return res;
}




//**********************************************************************************
__global__ void cuComputeF_TOTAL_LF(float *cuflx, float *cuflx_new, float c, float dx, float dt, int iter, float *cuegy, float aexp)
{
	int tx=threadIdx.x+NBOUND;
	int bx=blockIdx.x +NBOUND;
	int by=blockIdx.y +NBOUND;

	float fm1,fp1;

	// REMINDER LF flux : (fl+fr-ur+ul)*0.5f;
	//  f_icjcks_p =cuCompute_FaceLF(f[2+idx*3],f[2+idxp*3],c*e[idx],c*e[idxp]);

	float resfx, resfy, resfz;

	__shared__ float u[(NCELLX+NBOUND2)*3],fp[(NCELLX+NBOUND2)*3],fm[(NCELLX+NBOUND2)*3],ep[(NCELLX+NBOUND2)],em[(NCELLX+NBOUND2)];

	//================================================ Z DIRECTION =============================================

	int baseidu=0*NCELL4+(tx)+bx*(NCELLX+NBOUND2);

	u[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)]; // FX local cell
	u[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+NCELL4]; // FX local cell
	u[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+2*NCELL4]; // FX local cell

	ep[tx]=cuegy[baseidu+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)]; // E Cell+1
	em[tx]=cuegy[baseidu+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)]; // E Cell+1

	fm[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)]; // E Cell+1
	fm[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+NCELL4]; // E Cell+1
	fm[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by-1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+NCELL4*2]; // E Cell+1


	fp[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)]; // E Cell+1
	fp[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+NCELL4]; // E Cell+1
	fp[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(by+1)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2)+NCELL4*2]; // E Cell+1


	__syncthreads();

	// FX Divergence along Z

	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,0,2);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,0,2);

	resfx=u[tx+0*(NCELLX+NBOUND2)]-0.5*((fp1-fm1)+c*(2*u[tx+0*(NCELLX+NBOUND2)]-fm[tx+0*(NCELLX+NBOUND2)]-fp[tx+0*(NCELLX+NBOUND2)]))/dx*dt; 


	// FY Divergence along Z


	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,1,2);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,1,2);
	resfy=u[tx+1*(NCELLX+NBOUND2)]-0.5*((fp1-fm1)+c*(2*u[tx+1*(NCELLX+NBOUND2)]-fm[tx+1*(NCELLX+NBOUND2)]-fp[tx+1*(NCELLX+NBOUND2)]))/dx*dt; 


	// FZ Divergence along Z

	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,2,2);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,2,2);
	resfz=u[tx+2*(NCELLX+NBOUND2)]-0.5*((fp1-fm1)+c*(2*u[tx+2*(NCELLX+NBOUND2)]-fm[tx+2*(NCELLX+NBOUND2)]-fp[tx+2*(NCELLX+NBOUND2)]))/dx*dt; 

	__syncthreads();


	//================================================ Y DIRECTION =============================================

	baseidu=0*NCELL4+(tx)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);

	u[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx)*(NCELLX+NBOUND2)]; // FX local cell
	u[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx)*(NCELLX+NBOUND2)+NCELL4]; // FX local cell
	u[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx)*(NCELLX+NBOUND2)+2*NCELL4]; // FX local cell

	ep[tx]=cuegy[baseidu+(bx+1)*(NCELLX+NBOUND2)]; // E Cell+1
	em[tx]=cuegy[baseidu+(bx-1)*(NCELLX+NBOUND2)]; // E Cell+1

	fm[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx-1)*(NCELLX+NBOUND2)]; // E Cell+1
	fm[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx-1)*(NCELLX+NBOUND2)+NCELL4]; // E Cell+1
	fm[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx-1)*(NCELLX+NBOUND2)+NCELL4*2]; // E Cell+1

	fp[0*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx+1)*(NCELLX+NBOUND2)]; // E Cell+1
	fp[1*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx+1)*(NCELLX+NBOUND2)+NCELL4]; // E Cell+1
	fp[2*(NCELLX+NBOUND2)+tx]=cuflx[baseidu+(bx+1)*(NCELLX+NBOUND2)+NCELL4*2]; // E Cell+1

	__syncthreads();

	// FX Divergence along Y

	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,0,1);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,0,1);
	resfx=resfx-0.5*((fp1-fm1)+c*(2*u[tx+0*(NCELLX+NBOUND2)]-fm[tx+0*(NCELLX+NBOUND2)]-fp[tx+0*(NCELLX+NBOUND2)]))/dx*dt; 

	// FY Divergence along Y

	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,1,1);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,1,1);
	resfy=resfy-0.5*((fp1-fm1)+c*(2*u[tx+1*(NCELLX+NBOUND2)]-fm[tx+1*(NCELLX+NBOUND2)]-fp[tx+1*(NCELLX+NBOUND2)]))/dx*dt; 

	// FZ Divergence along Y

	fp1=Eddington(fp[0*(NCELLX+NBOUND2)+tx],fp[1*(NCELLX+NBOUND2)+tx],fp[2*(NCELLX+NBOUND2)+tx],ep[tx],c,2,1);
	fm1=Eddington(fm[0*(NCELLX+NBOUND2)+tx],fm[1*(NCELLX+NBOUND2)+tx],fm[2*(NCELLX+NBOUND2)+tx],em[tx],c,2,1);
	resfz=resfz-0.5*((fp1-fm1)+c*(2*u[tx+2*(NCELLX+NBOUND2)]-fm[tx+2*(NCELLX+NBOUND2)]-fp[tx+2*(NCELLX+NBOUND2)]))/dx*dt; 



	__syncthreads();


	//================================================ X DIRECTION =============================================

	baseidu=0*NCELL4+bx*(NCELLX+NBOUND2)+by*(NCELLX+NBOUND2)*(NCELLY+NBOUND2);

	u[0*(NCELLX+NBOUND2)+tx]=cuflx[tx+baseidu]; // FX local cell
	u[1*(NCELLX+NBOUND2)+tx]=cuflx[tx+baseidu+NCELL4]; // FX local cell
	u[2*(NCELLX+NBOUND2)+tx]=cuflx[tx+baseidu+2*NCELL4]; // FX local cell
	ep[tx]=cuegy[tx+baseidu]; // E Cell+1

	if(tx-NBOUND==0) 
	{
		u[NBOUND-1+0*(NCELLX+NBOUND2)]=cuflx[baseidu+tx-1];
		u[NBOUND-1+1*(NCELLX+NBOUND2)]=cuflx[baseidu+tx-1+NCELL4];
		u[NBOUND-1+2*(NCELLX+NBOUND2)]=cuflx[baseidu+tx-1+2*NCELL4];
		ep[NBOUND-1]=cuegy[tx-1+baseidu]; 
	}

	if(tx-NBOUND==blockDim.x-1) 
	{
		u[NCELLX+NBOUND+0*(NCELLX+NBOUND2)]=cuflx[baseidu+tx+1];
		u[NCELLX+NBOUND+1*(NCELLX+NBOUND2)]=cuflx[baseidu+tx+1+NCELL4];
		u[NCELLX+NBOUND+2*(NCELLX+NBOUND2)]=cuflx[baseidu+tx+1+2*NCELL4];
		ep[NCELLX+NBOUND]=cuegy[tx+1+baseidu]; 
	}

	__syncthreads();


	// FX Divergence along X

	fp1=Eddington(u[0*(NCELLX+NBOUND2)+tx+1],u[1*(NCELLX+NBOUND2)+tx+1],u[2*(NCELLX+NBOUND2)+tx+1],ep[tx+1],c,0,0);
	fm1=Eddington(u[0*(NCELLX+NBOUND2)+tx-1],u[1*(NCELLX+NBOUND2)+tx-1],u[2*(NCELLX+NBOUND2)+tx-1],ep[tx-1],c,0,0);
	resfx=resfx-0.5*((fp1-fm1)+c*(2*u[tx+0*(NCELLX+NBOUND2)]-u[tx+1+0*(NCELLX+NBOUND2)]-u[tx-1+0*(NCELLX+NBOUND2)]))/dx*dt;


	// FY Divergence along X

	fp1=Eddington(u[0*(NCELLX+NBOUND2)+tx+1],u[1*(NCELLX+NBOUND2)+tx+1],u[2*(NCELLX+NBOUND2)+tx+1],ep[tx+1],c,1,0);
	fm1=Eddington(u[0*(NCELLX+NBOUND2)+tx-1],u[1*(NCELLX+NBOUND2)+tx-1],u[2*(NCELLX+NBOUND2)+tx-1],ep[tx-1],c,1,0);
	resfy=resfy-0.5*((fp1-fm1)+c*(2*u[tx+1*(NCELLX+NBOUND2)]-u[tx+1+1*(NCELLX+NBOUND2)]-u[tx-1+1*(NCELLX+NBOUND2)]))/dx*dt;


	// FZ Divergence along X

	fp1=Eddington(u[0*(NCELLX+NBOUND2)+tx+1],u[1*(NCELLX+NBOUND2)+tx+1],u[2*(NCELLX+NBOUND2)+tx+1],ep[tx+1],c,2,0);
	fm1=Eddington(u[0*(NCELLX+NBOUND2)+tx-1],u[1*(NCELLX+NBOUND2)+tx-1],u[2*(NCELLX+NBOUND2)+tx-1],ep[tx-1],c,2,0);
	resfz=resfz-0.5*((fp1-fm1)+c*(2*u[tx+2*(NCELLX+NBOUND2)]-u[tx+1+2*(NCELLX+NBOUND2)]-u[tx-1+2*(NCELLX+NBOUND2)]))/dx*dt;


	cuflx_new[baseidu+tx]=resfx;
	cuflx_new[baseidu+tx+NCELL4]=resfy;
	cuflx_new[baseidu+tx+2*NCELL4]=resfz;

}



//**********************************************************************************
__device__ float cufindroot3_2(float a,float b,float c,float d,float xorg)
{

	int i;
	float f,df,x;
	x=xorg;

	for(i=0;i<=10;i++)
	{
		f=a*x*x*x+b*x*x+c*x+d;
		df=3*a*x*x+2*b*x+c;
		if(fabsf(f/(df*x))<0.00001) break;
		x=x-f/df;
	}

	if(x>ONE) x=ONE;
	if(x<ZERO) x=ZERO;
	return x;
	

}
__device__ float cufindroot3(float a,float b,float c,float d,float xorg)
{
	float Q,R;
	float A,B,th;
	float x1,x2,x3;
	float x0;

	Q=((b/a)*(b/a)-3.*(c/a))/9.;
	R=(2*(b/a)*(b/a)*(b/a)-9*b*c/a/a+27*d/a)/54.;

	if(R*R<Q*Q*Q)
	{
		th=acosf(R/sqrtf(Q*Q*Q));
		x1=-2.*sqrtf(Q)*cosf(th/3.)-b/3./a;
		x2=-2.*sqrtf(Q)*cosf((th+2*M_PI)/3.)-b/3./a;
		x3=-2.*sqrtf(Q)*cosf((th-2*M_PI)/3.)-b/3./a;
		if((x1>=0)&&(x1<=1))
		{
			x0=x1;
		}
		else if((x2>=0)&&(x2<=1))
		{
			x0=x2;
		}
		else if((x3>=0)&&(x3<=1))
		{
			x0=x3;
		}
		else
		{
			x0=xorg;
		}

	}
	else
	{

		A=-copysignf(1.,R)*powf(fabsf(R)+sqrtf(R*R-Q*Q*Q),1./3.);
		B=0.;
		if(A!=0) B=Q/A;
		x0=(A+B)-b/3./a;
		if((x0>1)||(x0<0))
		{
			#ifdef DEVICE_EMULATION
				puts("ERROR in root finder, anormal ion fraction");
				abort();
			#endif
		}
	}

	//  if(idx==154869){printf("a=%e b=%e c=%e d=%e\n",a,b,c,d); printf("R=%e Q=%e x0=%e\n",R,Q,x0);}

	//x0=x3;
	return x0;
}


//**********************************************************************************
//**********************************************************************************

__device__ float cufindrootacc(float a,float b,float c)
{
	float q=-0.5*(b+copysign(1.,b)*sqrt(b*b-4*a*c));
	float x1=q/a;
	float x2=c/q;
	float x0=x1;
	if((x2<=ONE)&&(x2>=ZERO)) x0=x2;
	return x0;
}


//=========================================================
//=========================================================
#ifdef HELIUM
__device__ float cucompute_alpha_b_HII(float temp, float unit_number, float aexp)
{
	// CASE B recombination rate m**3 s*-1
	// temperature should be given in Kelvin

	float alpha_b,lambda;
	lambda=2e0*T_HI/temp;
	alpha_b=2.753e-14*powf(lambda,1.5)/powf(1e0+powf(lambda/2.740,0.407),2.242); //cm3/s
	#ifdef COSMO
		alpha_b=alpha_b*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#else
		alpha_b=alpha_b*1e-6*unit_number; //m3/s
	#endif
	return alpha_b;
}

__device__ float cucompute_alpha_b_HeII(float temp, float unit_number, float aexp)
{
	// CASE B recombination rate m**3 s*-1
	// temperature should be given in Kelvin

	float alpha_b,lambda;
	lambda=2e0*T_HeI/temp;
	alpha_b=1.26e-14*powf(lambda,0.750); //cm3/s
	#ifdef COSMO
		alpha_b=alpha_b*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#else
		alpha_b=alpha_b*1e-6*unit_number; //m3/s
	#endif
	return alpha_b;
}

__device__ float cucompute_alpha_b_HeIII(float temp, float unit_number, float aexp)
{
	// CASE B recombination rate m**3 s*-1
	// temperature should be given in Kelvin

	float alpha_b,lambda;
	lambda=2e0*T_HeII/temp;
	alpha_b=2e0*2.753e-14*powf(lambda,1.5)/powf(1e0+powf(lambda/2.740,0.407),2.242); //cm3/s
	#ifdef COSMO
		alpha_b=alpha_b*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#else
		alpha_b=alpha_b*1e-6*unit_number; //m3/s
	#endif
	return alpha_b;
}

//=========================================================
//=========================================================

__device__ float cucompute_alpha_a_HII(float temp, float unit_number, float aexp)
{
// CASE A recombination rate m**3 s*-1
// temperature should be given in Kelvin

	float alpha_a,lambda;
	lambda=2e0*T_HI/temp;
	alpha_a=1.269e-13*powf(lambda,1.503)/powf(1e0+powf(lambda/0.522,0.470),1.923); //cm3/s
	#ifdef COSMO
		alpha_a=alpha_a*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#endif
	return alpha_a;
}

__device__ float cucompute_alpha_a_HeII(float temp, float unit_number, float aexp)
{
// CASE A recombination rate m**3 s*-1
// temperature should be given in Kelvin

	float alpha_a,lambda;
	lambda=2e0*T_HeI/temp;
	alpha_a=3.0e-14*powf(lambda,0.654); //cm3/s
	#ifdef COSMO
		alpha_a=alpha_a*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#endif
	return alpha_a;
}

__device__ float cucompute_alpha_a_HeIII(float temp, float unit_number, float aexp)
{
// CASE A recombination rate m**3 s*-1
// temperature should be given in Kelvin

	float alpha_a,lambda;
	lambda=2e0*T_HeII/temp;
	alpha_a=2.0*1.269e-13*powf(lambda,1.503)/powf(1e0+powf(lambda/0.522,0.470),1.923); //cm3/s
	#ifdef COSMO
		alpha_a=alpha_a*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
	#endif
	return alpha_a;
}
//=========================================================
//=========================================================

__device__ float cucompute_beta_HI(float temp, float unit_number, float aexp)
{
	// Collizional ionization rate m**3 s*-1
	// temperature in Kelvin
	float beta,T5;
	T5=temp/1e5;
	beta=5.85e-11*sqrtf(temp)/(1+sqrtf(T5))*expf(-(T_HI/temp)); //cm3/s
	#ifdef COSMO
		beta=beta*1e-6*unit_number/(aexp*aexp*aexp); // !m3/s
	#endif
	return beta;
}
__device__ float cucompute_beta_HeI(float temp, float unit_number, float aexp)
{
	// Collizional ionization rate m**3 s*-1
	// temperature in Kelvin
	float beta,T5;
	T5=temp/1e5;
	beta=2.38e-11*sqrtf(temp)/(1+sqrtf(T5))*expf(-(T_HeI*4/temp)); //cm3/s
	#ifdef COSMO
		beta=beta*1e-6*unit_number/(aexp*aexp*aexp); // !m3/s
	#endif
	return beta;
}

__device__ float cucompute_beta_HeII(float temp, float unit_number, float aexp)
{
	// Collizional ionization rate m**3 s*-1
	// temperature in Kelvin
	float beta,T5;
	T5=temp/1e5;
	beta=5.68e-12*sqrtf(temp)/(1+sqrtf(T5))*expf(-(T_HeII/temp)); //cm3/s
	#ifdef COSMO
		beta=beta*1e-6*unit_number/(aexp*aexp*aexp); // !m3/s
	#endif
	return beta;
}


__device__ void cuCompCooling(float temp, float ne, float nHI, float nHII, float nHeI, float nHeII, float nHeIII, float *lambda, float aexp,float CLUMPF)
{

	//all n are in m-3
	
	float coll_HI,coll_HeI,coll_HeII, recom_HII, recom_HeII, recom_HeIII, collexc_HI, collexc_HeII, brem,dielec,compton;
	float lambda_HI,lambda_HeI,lambda_HeII;
	
	
	lambda_HI=2e0*T_HI/temp;
	lambda_HeI=2e0*T_HeI/temp;
	lambda_HeII=2e0*T_HeII/temp;
	// All units are erg cm3 s-1 m-6 
	// Collisional Ionization Cooling

	coll_HI=(expf(-T_HI/temp)*1.27e-21*sqrtf(temp)/(1+sqrtf(temp/1e5)))*ne*nHI*CLUMPF ; 
	coll_HeI=(expf(-(T_HeI*4)/temp)*9.38e-22*sqrtf(temp)/(1+sqrtf(temp/1e5)))*ne*nHeI*CLUMPF ;
	coll_HeII=(expf(-(T_HeII)/temp)*4.95e-22*sqrtf(temp)/(1+sqrtf(temp/1e5)))*ne*nHeII*CLUMPF; 

	// Case A Recombination Cooling

	recom_HII=1.778e-29*temp*powf(lambda_HI,1.965e0)/(powf(1.f+powf(lambda_HI/0.541e0,0.502e0),2.697e0))*ne*nHII*CLUMPF;
	recom_HeII=KBOLTZ*temp*3e-14*powf(lambda_HeI,0.654)*ne*nHeII*CLUMPF;
	recom_HeIII=8*1.778e-29*temp*powf(lambda_HeII,1.965e0)/(powf(1.f+powf(lambda_HeII/0.541e0,0.502e0),2.697e0))*ne*nHeIII*CLUMPF;


	// Collisional excitation cooling
	
	collexc_HI=(7.5e-19*expf(-118348e0/temp)/(1+sqrtf(temp/1e5)))*ne*nHI*CLUMPF;
	collexc_HeII=(5.54e-17*powf(temp,-0.397)*expf(-473638e0/temp)/(1+sqrtf(temp/1e5)))*ne*nHeII*CLUMPF;


	// Bremmsstrahlung

	brem=1.42e-27*sqrtf(temp)*ne*(nHII+nHeII+nHeIII)*CLUMPF;

	//dielectric
	
	dielec=1.24e-13*powf(temp,-1.5)*expf(-470000e0/temp)*(1+0.3*expf(-94000e0/temp))*ne*nHeII*CLUMPF; // erg cm3 s-1 m-6
	
	// Compton Cooling

	compton=1.017e-37*powf(2.727/aexp,4)*(temp-2.727/aexp)*(nHII+nHeII+nHeIII); //erg s-1 m-3

	// Overall Cooling

	*lambda=(coll_HI+coll_HeI+coll_HeII+recom_HII+recom_HeII+recom_HeIII+collexc_HI+collexc_HeII+brem+dielec)*1e-6+compton;//  erg m-3 s-1 


	// Unit Conversion

	*lambda=(*lambda)*1e-7;// ! J*m-3*s-1
	
}



//**********************************************************************************
//**********************************************************************************

//**********************************************************************************


__global__ void cuComputeTemp(float *cuxHII,float *cuxHeII,float *cuxHeIII, float *cudensity, float *cutemperature, float *cuegy_new, float fudgecool, float c, float dt,float unit_number, int ncvgcool, float aexp, float hubblet, float *cuflx_new, float CLUMPF, float egy_min, float fesc, float boost, float *cusrc)
{
	int 	tx=threadIdx.x,
	bx=blockIdx.x,
        by=blockIdx.y,
	idx1=tx+bx*blockDim.x+by*gridDim.x*blockDim.x,
	k=idx1/(NCELLX*NCELLY),
	j=(idx1-k*(NCELLX*NCELLY))/NCELLX,
	i=idx1-k*(NCELLX*NCELLY)-j*(NCELLX),
	idx=(i+NBOUND)+(j+NBOUND)*(NCELLX+NBOUND2)+(k+NBOUND)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2), // following a convention a[k,j,i] where i varies the first,
	idloc=tx,
	idloc3=3*idloc,
	igrp,
	nitcool=0;

	float Cool,
	Heat,
	tcool,
	dtcool,
	currentcool_t=0.f,alpha_a_HII=0.f,alpha_a_HeII=0.f,alpha_a_HeIII=0.f,
	alpha_b_HII=0.f,alpha_b_HeII=0.f,alpha_b_HeIII=0.f,
	beta_HI=0.f,beta_HeI=0.f,beta_HeII=0.f,
	eint=0.f,
	xHII_new=0.f,// These 3 will store the updated variables
	xHeII_new=0.f,
	xHeIII_new=0.f,
	temp_new=0.f,
	ai_tmp1=0.f, //3 temporary variables to store data
	ai_tmp2=0.f,
	ai_tmp3=0.f,
	numerator=0.f, // Few more temp variables 
	denominator=0.f,
	  hnu_HI[NGRP],hnu_HeI[NGRP], hnu_HeII[NGRP],		// ! Average Photon Energy (J)
	factgrp_HI[NGRP],factgrp_HeI[NGRP],factgrp_HeII[NGRP],		
	alphae_HI[NGRP],alphae_HeI[NGRP],alphae_HeII[NGRP],
	alphai_HI[NGRP],alphai_HeI[NGRP],alphai_HeII[NGRP],		
	et[NGRP],
	p[NGRP],
	//
	nHI=0.f,nHII=0.f,nHeI=0.f,nHeII=0.f,nHeIII=0.f,
	ntot=0.f,ne=0.f;//total number density and electron number density
	//
	
	__shared__ float
	egy_loc[BLOCKCOOL*NGRP],
	floc[3*BLOCKCOOL*NGRP],
	xHII[BLOCKCOOL],
	//
	xHeII[BLOCKCOOL],
	xHeIII[BLOCKCOOL],
	//
	nH[BLOCKCOOL],
	//
	nHe[BLOCKCOOL],
	//
	tloc[BLOCKCOOL],
	srcloc[BLOCKCOOL];

	#ifdef S_X
		float N2[NGRP];
		float F2[NGRP];
		float E0overI[NGRP];
	#endif

	c=c*aexp; 			// switch back to physical velocity
	SECTION_EFFICACE;
	FACTGRP;

	xHII[idloc]=cuxHII[idx];
	//
	xHeII[idloc]=cuxHeII[idx];
	xHeIII[idloc]=cuxHeIII[idx];
	//
	nH[idloc]=cudensity[idx]*unit_number/(aexp*aexp*aexp); //unit is m-3
	//
	nHe[idloc]=(nH[idloc]*mass_p*frac_He)/(frac_H*mass_He); // The Hydrogen fraction as seen in the mkdens file is 0.75
	//
	tloc[idloc]=cutemperature[idx]; 
	srcloc[idloc]=cusrc[idx]*unit_number/(aexp*aexp*aexp); 




	for (igrp=0;igrp<NGRP;igrp++)
	{			// switch to physical units, chemistry remains unchanged with and without cosmo
		egy_loc[idloc+igrp*BLOCKCOOL]=cuegy_new[idx+igrp*NCELL4]*unit_number/(aexp*aexp*aexp); 
		floc[0+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[0*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
		floc[1+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[1*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
		floc[2+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[2*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
	}

	__syncthreads();

	

	#ifdef WCLUMP
		float CLUMPF2=fminf(fmaxf(powf(nH[idloc]/6.,0.7),1.),40.);
		float CLUMPI=1.;
	#else
		float CLUMPF2=1.;
		float CLUMPI=1.;
	#endif

	for(igrp=0;igrp<NGRP;igrp++)
	{
		alphai_HI[igrp] *= CLUMPI;
		alphae_HI[igrp] *= CLUMPI;
		alphai_HeI[igrp] *= CLUMPI;
		alphae_HeI[igrp] *= CLUMPI;
		alphai_HeII[igrp] *= CLUMPI;
		alphae_HeII[igrp] *= CLUMPI;

	}

	while(currentcool_t<dt)
	{
		nitcool++;
		
		nHI=(1-xHII[idloc])*nH[idloc];
		nHII=xHII[idloc]*nH[idloc];
		nHeI=(1-xHeII[idloc]-xHeIII[idloc])*nHe[idloc];
		nHeII=xHeII[idloc]*nHe[idloc];
		nHeIII=xHeIII[idloc]*nHe[idloc];
		
		ne=nHII+nHeII+2*nHeIII;
		
		ntot=nH[idloc]+nHe[idloc]+ne;
		
		eint=1.5*KBOLTZ*ntot*tloc[idloc];//J m-3
		//== Getting a timestep
		cuCompCooling(tloc[idloc],ne,nHI,nHII,nHeI,nHeII,nHeIII,&Cool,aexp,CLUMPF2);

		ai_tmp1=0.; ai_tmp2=0.; ai_tmp3=0.;
		for (igrp=0;igrp<NGRP;igrp++) 
		{
			ai_tmp1 += ((alphae_HI[igrp])*hnu_HI[igrp]-(alphai_HI[igrp])*HI_ion)*egy_loc[idloc+igrp*BLOCKCOOL];
			ai_tmp2 += ((alphae_HeI[igrp])*hnu_HeI[igrp]-(alphai_HeI[igrp])*HeI_ion)*egy_loc[idloc+igrp*BLOCKCOOL];
			ai_tmp3 += ((alphae_HeII[igrp])*hnu_HeII[igrp]-(alphai_HeII[igrp])*HeII_ion)*egy_loc[idloc+igrp*BLOCKCOOL];
		}
		Heat=nHI*ai_tmp1+nHeI*ai_tmp2+nHeII*ai_tmp3;
		tcool=fabsf(eint/(Heat-Cool));
		
		ai_tmp1=0.;ai_tmp2=0.;ai_tmp3=0.;
		
		dtcool=fminf(fudgecool*tcool,dt-currentcool_t);
		
		//****************** previous temperature alpha and beta values
		alpha_a_HII=cucompute_alpha_a_HII(tloc[idloc],1.,1.)*CLUMPF2;
		alpha_b_HII=cucompute_alpha_b_HII(tloc[idloc],1.,1.)*CLUMPF2;
		beta_HI=cucompute_beta_HI(tloc[idloc],1.,1.)*CLUMPF2;
		
		alpha_a_HeII=cucompute_alpha_a_HeII(tloc[idloc],1.,1.)*CLUMPF2;
		alpha_b_HeII=cucompute_alpha_b_HeII(tloc[idloc],1.,1.)*CLUMPF2;
		beta_HeI=cucompute_beta_HeI(tloc[idloc],1.,1.)*CLUMPF2;
		
		alpha_a_HeIII=cucompute_alpha_a_HeIII(tloc[idloc],1.,1.)*CLUMPF2;
		alpha_b_HeIII=cucompute_alpha_b_HeIII(tloc[idloc],1.,1.)*CLUMPF2;
		beta_HeII=cucompute_beta_HeII(tloc[idloc],1.,1.)*CLUMPF2;
		
		//**********************
		
		bool test = 0;
		for(igrp=0;igrp<NGRP;igrp++)
		{
			// Updating Energy density
			numerator=egy_loc[idloc+igrp*BLOCKCOOL]+srcloc[idloc]*dtcool;
			numerator=numerator+ne*dtcool*(nHII*(alpha_a_HII-alpha_b_HII)+nHeII*(alpha_a_HeII-alpha_b_HeII)+nHeIII*(alpha_a_HeIII-alpha_b_HeIII));
			
			denominator=1.f+dtcool*(3*hubblet+nHI*alphai_HI[igrp]+nHeI*alphai_HeI[igrp]+nHeII*alphai_HeII[igrp]);
			
			et[igrp]=numerator/denominator;
			//-----------
			if(et[igrp]<0)	{test=1;}
			// Updating Flux
			p[igrp]=(1.f+(alphai_HI[igrp]*nHI+nHeI*alphai_HeI[igrp]+nHeII*alphai_HeII[igrp]+2*hubblet)*dtcool);
		}


		if (et[igrp]<0) 
		{
			printf("energy density is less than 0 \n");
			fudgecool/=10.f; 
			continue;	
		} 

		// IONISATION
		ai_tmp1=0.;ai_tmp2=0.;ai_tmp3=0.;
		#ifndef S_X // This runs
			for(igrp=0;igrp<NGRP;igrp++) {
				ai_tmp1 += alphai_HI[igrp]*et[igrp];
				ai_tmp2 += alphai_HeI[igrp]*et[igrp];
				ai_tmp3 += alphai_HeII[igrp]*et[igrp];
			}
		#else
			N2[0]=1.0f;
			float pp=(1.f-powf(x0[idloc],0.4092f)); 
			if(pp<0.f) pp=0.f; 

			N2[1]=1.0f+0.3908f*powf(pp,1.7592f)*E0overI[1]; 
			if(N2[1]<1.0f) N2[1]=1.0f; 

			//N2[1]=1.0f;

			for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += alphai[igrp]*et[igrp]*N2[igrp];}
		#endif
		
		//Updating ionizied fraction of Hydrogen		
		numerator=xHII[idloc]+dtcool*(beta_HI*ne+ai_tmp1);
		denominator=1+dtcool*alpha_a_HII*ne+dtcool*(beta_HI*ne+ai_tmp1);
		xHII_new=numerator/denominator;
		//------------------- This is new 
		
		if(xHII_new>1.f)
		{
		  //printf("xHII_new is > than 1 = %f",xHII_new);
			xHII_new=DEFAULT_HIGH;
		}	
		if(xHII_new<0.f)
		{
		  //	printf("xHII_new is < than 0 = %f",xHII_new);
			xHII_new=DEFAULT_LOW;
		}	
		
		//-------------------
		
		//Updating singley ionizied Helium fraction
		numerator=xHeII[idloc]+dtcool*(beta_HeI*ne+ai_tmp2)-dtcool*xHeIII[idloc]*(beta_HeI*ne+ai_tmp2+alpha_a_HeIII*ne);
		denominator=1.f+dtcool*(beta_HeI*ne+ai_tmp2+beta_HeII*ne+alpha_a_HeII*ne+ai_tmp2);
		xHeII_new=numerator/denominator;
		
		if(xHeII_new>1.f)
		{
		  //	printf("xHeII_new is > than 1 = %f",xHeII_new);
			xHeII_new=DEFAULT_HIGH;
		}	
		if(xHeII_new<0.f)
		{
		  //	printf("xHeII_new is < than 0 = %f",xHeII_new);
			xHeII_new=DEFAULT_LOW;
		}	
		
		
		//-------------------
			
		//Updating doubly ionized Helium fraction
		numerator=xHeIII[idloc]+dtcool*xHeII_new*(beta_HeII*ne+ai_tmp3);
		denominator=1.f+dtcool*alpha_a_HeIII*ne;
		xHeIII_new=numerator/denominator;
				
		
		
		if(xHeIII_new>1.f)
		{
		  //	printf("xHeIII_new is > than 1 = %f",xHeIII_new);
			xHeIII_new=DEFAULT_HIGH;
		}	
		if(xHeIII_new<0.f)
		{
		  //	printf("xHeIII_new is < than 0 = %f",xHeIII_new);
			xHeIII_new=DEFAULT_LOW;
		}	
		
		
		//-------------------	
	
	
		//------------------------------------------
		cuCompCooling(tloc[idloc],ne,nHI, nHII, nHeI, nHeII, nHeIII,&Cool,aexp,CLUMPF2);
		// Updating the temperature
	
		nHI=(1-xHII_new)*nH[idloc];
		nHII=xHII_new*nH[idloc];
		nHeI=(1-xHeII_new-xHeIII_new)*nHe[idloc];
		nHeII=xHeII_new*nHe[idloc];
		nHeIII=xHeIII_new*nHe[idloc];
		
		ne=nHII+nHeII+2*nHeIII;
		
		ntot=nH[idloc]+nHe[idloc]+ne;
		
		ai_tmp1=0.; ai_tmp2=0.; ai_tmp3=0.;
		#ifdef COOLING
			// HEATING
			#ifndef S_X //This runs
				for (igrp=0;igrp<NGRP;igrp++) 
				{
					ai_tmp1 += ((alphae_HI[igrp])*hnu_HI[igrp]-(alphai_HI[igrp])*HI_ion)*et[igrp];
					ai_tmp2 += ((alphae_HeI[igrp])*hnu_HeI[igrp]-(alphai_HeI[igrp])*HeI_ion)*et[igrp];
					ai_tmp3 += ((alphae_HeII[igrp])*hnu_HeII[igrp]-(alphai_HeII[igrp])*HeII_ion)*et[igrp];
				}
				
			#else
				float pp2;
				F2[0]=1.0f;
				F2[1]=1.0f;
				pp2=1.0f-powf(xt,0.2663f); 
				if(pp2<0.f) pp2=0.f; 
				F2[1]=0.9971f*(1.0f-powf(pp2,1.3163f)); 

				if(F2[1]>1.0f) F2[1]=1.0f; 
				if(F2[1]<0.0f) F2[1]=0.0f; 

				for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += et[igrp]*(alphae[igrp]*hnu[igrp]-(alphai[igrp]*hnu0))*F2[igrp];}
			#endif

			eint=(eint+dtcool*(nHI*ai_tmp1+nHeI*ai_tmp2+nHeII*ai_tmp3)-dtcool*Cool)/(1.f+dtcool*2*hubblet);
			temp_new=eint/(1.5*KBOLTZ*ntot);
			
			ai_tmp1=0; ai_tmp2=0; ai_tmp3=0;

			if(eint<0.f) 
			
			{	
				printf("Internal energy is less than 0 \n");
				fudgecool/=10.f; 
				continue;
			} 
		#endif
		//******************
		xHII[idloc]=xHII_new;
		xHeII[idloc]=xHeII_new;
		xHeIII[idloc]=xHeIII_new;
		tloc[idloc]=temp_new;
		
		for(igrp =0;igrp<NGRP;igrp++)
		{
			egy_loc[idloc+igrp*BLOCKCOOL]=et[igrp];
			floc[0+idloc3+igrp*BLOCKCOOL*3]=floc[0+idloc3+igrp*BLOCKCOOL*3]/p[igrp];
			floc[1+idloc3+igrp*BLOCKCOOL*3]=floc[1+idloc3+igrp*BLOCKCOOL*3]/p[igrp];
			floc[2+idloc3+igrp*BLOCKCOOL*3]=floc[2+idloc3+igrp*BLOCKCOOL*3]/p[igrp];	
		}

		
		currentcool_t+=dtcool;
		if((nitcool==ncvgcool)&&(ncvgcool!=0)) break;
	}

	for(igrp=0;igrp<NGRP;igrp++)
	{
		cuegy_new[idx+igrp*NCELL4]=fmax(egy_loc[idloc+igrp*BLOCKCOOL]*aexp*aexp*aexp,egy_min);
		cuflx_new[0*NCELL4+idx+igrp*NCELL4*3]=floc[0+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
		cuflx_new[1*NCELL4+idx+igrp*NCELL4*3]=floc[1+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
		cuflx_new[2*NCELL4+idx+igrp*NCELL4*3]=floc[2+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
	}

	
	cutemperature[idx]=tloc[idloc];
	cuxHII[idx]=xHII[idloc];
	cuxHeII[idx]=xHeII[idloc];
	cuxHeIII[idx]=xHeIII[idloc];
	__syncthreads();

}


#else

__device__ float cucompute_alpha_b(float temp, float unit_number, float aexp)
{
  // CASE B recombination rate m**3 s*-1
  // temperature should be given in Kelvin
  
  float alpha_b,lambda;
  lambda=2e0*157807e0/temp;
  alpha_b=2.753e-14*powf(lambda,1.5)/powf(1e0+powf(lambda/2.740,0.407),2.242); //cm3/s
#ifdef COSMO
  alpha_b=alpha_b*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
#else
  alpha_b=alpha_b*1e-6*unit_number; //m3/s
#endif
  return alpha_b;
}

//=========================================================
//=========================================================

__device__ float cucompute_alpha_a(float temp, float unit_number, float aexp)
{
  // CASE A recombination rate m**3 s*-1
  // temperature should be given in Kelvin
  
  float alpha_a,lambda;
  lambda=2e0*157807e0/temp;
  alpha_a=1.269e-13*powf(lambda,1.503)/powf(1e0+powf(lambda/0.522,0.470),1.923); //cm3/s
#ifdef COSMO
  alpha_a=alpha_a*1e-6*unit_number/(aexp*aexp*aexp); //m3/s
#else
  alpha_a=alpha_a*1e-6*unit_number; //m3/s
#endif
  return alpha_a;
}

//=========================================================
//=========================================================

__device__ float cucompute_beta(float temp, float unit_number, float aexp)
{
  // Collizional ionization rate m**3 s*-1
  // temperature in Kelvin
  float beta,T5;
  T5=temp/1e5;
  beta=5.85e-11*sqrtf(temp)/(1+sqrtf(T5))*expf(-(157809e0/temp)); //cm3/s
#ifdef COSMO
  beta=beta*1e-6*unit_number/(aexp*aexp*aexp); // !m3/s
#else
  beta=beta*1e-6*unit_number; // !m3/s
#endif
  return beta;
}

//**********************************************************************************
//**********************************************************************************


__device__ void cuCompCooling(float temp, float x, float nH, float *lambda, float *tcool, float aexp,float CLUMPF)
{

 
  float c1,c2,c3,c4,c5,c6;
  float unsurtc;
  float nh2;

  nh2=nH*1e-6;// ! m-3 ==> cm-3
  

  // Collisional Ionization Cooling

  c1=expf(-157809.1e0/temp)*1.27e-21*sqrtf(temp)/(1.f+sqrtf(temp/1e5))*x*(1.f-x)*nh2*nh2*CLUMPF;
  

  // Case A Recombination Cooling

  c2=1.778e-29*temp*powf(2e0*157807e0/temp,1.965e0)/powf(1.f+powf(2e0*157807e0/temp/0.541e0,0.502e0),2.697e0)*x*x*nh2*nh2*CLUMPF;
  
  
  // Case B Recombination Cooling

  //c6=3.435e-30*temp*powf(2e0*157807e0/temp,1.970e0)/powf(1.f+(powf(2e0*157807e0/temp/2.250e0,0.376e0)),3.720e0)*x*x*nh2*nh2*CLUMPF;
  

  // Collisional excitation cooling

  c3=expf(-118348e0/temp)*7.5e-19/(1+sqrtf(temp/1e5))*x*(1.f-x)*nh2*nh2*CLUMPF;
  
  
  // Bremmsstrahlung

  c4=1.42e-27*1.5e0*sqrtf(temp)*x*x*nh2*nh2*CLUMPF;
  
  // Compton Cooling
  
  c5=1.017e-37*powf(2.727/aexp,4)*(temp-2.727/aexp)*nh2*x;
  
  // Overall Cooling
  
  *lambda=c1+c2+c3+c4+c5;//+c6;// ! erg*cm-3*s-1
  

  // Unit Conversion

  *lambda=(*lambda)*1e-7*1e6;// ! J*m-3*s-1

  // cooling times

  unsurtc=fmaxf(c1,c2);
  unsurtc=fmaxf(unsurtc,c3);
  unsurtc=fmaxf(unsurtc,c4);
  unsurtc=fmaxf(unsurtc,fabs(c5));
  //unsurtc=fmaxf(unsurtc,c6)*1e-7;// ==> J/cm3/s
  unsurtc=unsurtc*1e-7;   // ==> J/cm3/s

  *tcool=1.5e0*nh2*(1+x)*1.3806e-23*temp/unsurtc; //Myr
}

__global__ void cuComputeTemp(float *cuxHII, float *cudensity, float *cutemperature, float *cuegy_new, float fudgecool, float c, float dt,float unit_number, int ncvgcool, float aexp, float hubblet, float *cuflx_new, float CLUMPF, float egy_min, float fesc, float boost, float *cusrc)
{
  int 	tx=threadIdx.x,
	bx=blockIdx.x,
	by=blockIdx.y,
	idx1=tx+bx*blockDim.x+by*gridDim.x*blockDim.x,
	k=idx1/(NCELLX*NCELLY),
	j=(idx1-k*(NCELLX*NCELLY))/NCELLX,
	i=idx1-k*(NCELLX*NCELLY)-j*(NCELLX),
	idx=(i+NBOUND)+(j+NBOUND)*(NCELLX+NBOUND2)+(k+NBOUND)*(NCELLX+NBOUND2)*(NCELLY+NBOUND2), // following a convention a[k,j,i] where i varies the first,
	idloc=tx,
	idloc3=3*idloc,
	igrp,
 	nitcool=0;

  float hnu0=13.6*1.6022e-19,
	Cool,
	tcool,
	dtcool,
	tcool1,
	currentcool_t=0.f,
	alpha,
	alphab,
	beta,
	eint,
	xt,
	ai_tmp1=0.,
	hnu[NGRP],		// ! Average Photon Energy (J)
	factgrp[NGRP],		
	alphae[NGRP],
	alphai[NGRP],		
	et[NGRP],
	p[NGRP];

  __shared__ float
	egyloc[BLOCKCOOL*NGRP],
    floc[3*BLOCKCOOL*NGRP],
	x0[BLOCKCOOL],
	nH[BLOCKCOOL],
    tloc[BLOCKCOOL],
	srcloc[BLOCKCOOL];

#ifdef S_X
  float N2[NGRP];
  float F2[NGRP];
  float E0overI[NGRP];
#endif

  c=c*aexp; 			// switch back to physical velocity
  SECTION_EFFICACE;
  FACTGRP;

  x0[idloc]=cuxHII[idx];
  nH[idloc]=cudensity[idx]*unit_number/(aexp*aexp*aexp);
  tloc[idloc]=cutemperature[idx]; 
  srcloc[idloc]=cusrc[idx]*unit_number/(aexp*aexp*aexp); 




  for (igrp=0;igrp<NGRP;igrp++)
	{			// switch to physical units, chemistry remains unchanged with and without cosmo
	  //egyloc[idloc+igrp*BLOCKCOOL]=cuegy_new[idx+igrp*NCELL4];
	  egyloc[idloc+igrp*BLOCKCOOL]=cuegy_new[idx+igrp*NCELL4]*unit_number/(aexp*aexp*aexp); 
	  floc[0+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[0*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
	  floc[1+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[1*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
	  floc[2+idloc3+igrp*BLOCKCOOL*3]=cuflx_new[2*NCELL4+idx+igrp*NCELL4*3]/(aexp*aexp);
 	 }

  __syncthreads();

//  float ncomov=nH[idloc]*(aexp*aexp*aexp);
//  float CLUMPF2=fminf(powf(ncomov/0.25,1.)*(ncomov>0.25)+(ncomov<=0.25)*1.,100.);
//  float CLUMPI=fminf(powf(ncomov/0.5,1.2)*(ncomov>0.5)+(ncomov<=0.5)*1.,10.);
//  float CLUMPF2=100.*powf(ncomov/(ncomov+5.),2)+1.f;
//  float CLUMPI=100.*powf(ncomov/(ncomov+8.),2)+1.f; 
  
//  float CLUMPF2=fmaxf(fminf(1000.f*powf(ncomov,2.5),70.),1.); 
//  float CLUMPI=fmaxf(fminf(20.f*powf(ncomov,2.5),20.),1.); 

//  float CLUMPF2=fmaxf(fminf(1000.f*powf(ncomov,2.5),200.),1.); 
//  float CLUMPI=fmaxf(fminf(20.f*powf(ncomov,2.5),100.),1.); 

//  float CLUMPI=fminf(expf(ncomov/0.5),20.);
//  float CLUMPF2=fminf(expf(ncomov/0.5),20.);
//  float CLUMPF2=fminf(expf(ncomov/0.25),120.);

#ifdef WCLUMP
  float CLUMPF2=fminf(fmaxf(powf(nH[idloc]/6.,0.7),1.),40.);
  float CLUMPI=1.;
#else
  float CLUMPF2=1.;
  float CLUMPI=1.;
#endif

//  float CLUMPF2=fmaxf(fminf(powf(nH[idloc]/400.)^2.5+powf(nH[idloc]/8.)^0.7*(1-x0[idloc]),100.),1.); 
//  float CLUMPI==fmaxf(fminf(powf(nH[idloc]/400.)^2.5+powf(nH[idloc]/8.)^0.2*(1-x0[idloc]),100.),1.); 

  for(igrp=0;igrp<NGRP;igrp++)
	{alphai[igrp] *= CLUMPI;
	 alphae[igrp] *= CLUMPI;}

  while(currentcool_t<dt)
    {
      nitcool++;

      eint=1.5*nH[idloc]*KBOLTZ*(1.f+x0[idloc])*tloc[idloc];

      //== Getting a timestep
      cuCompCooling(tloc[idloc],x0[idloc],nH[idloc],&Cool,&tcool1,aexp,CLUMPF2);
      
      ai_tmp1=0.;
      for (igrp=0;igrp<NGRP;igrp++) ai_tmp1 += ((alphae[igrp])*hnu[igrp]-(alphai[igrp])*hnu0)*egyloc[idloc+igrp*BLOCKCOOL];

      tcool=fabsf(eint/(nH[idloc]*(1.0f-x0[idloc])*ai_tmp1-Cool));
      ai_tmp1=0.;
      dtcool=fminf(fudgecool*tcool,dt-currentcool_t);
      
      alpha=cucompute_alpha_a(tloc[idloc],1.,1.)*CLUMPF2;
      alphab=cucompute_alpha_b(tloc[idloc],1.,1.)*CLUMPF2;
      beta=cucompute_beta(tloc[idloc],1.,1.)*CLUMPF2;
      
      //== Update
      //      egyloc[idloc]=((alpha-alphab)*x0[idloc]*x0[idloc]*nH[idloc]*nH[idloc]*dtcool+egyloc[idloc])/(1.f+dtcool*(alphai*(1.f-x0[idloc])*nH[idloc]+3*hubblet));

      // ABSORPTION
  bool test = 0;
  for(igrp=0;igrp<NGRP;igrp++)
    {
      ai_tmp1 = alphai[igrp];
      et[igrp]=((alpha-alphab)*x0[idloc]*x0[idloc]*nH[idloc]*nH[idloc]*dtcool*factgrp[igrp]+egyloc[idloc+igrp*BLOCKCOOL]+srcloc[idloc]*dtcool*fesc*boost*factgrp[igrp])/(1.f+dtcool*(ai_tmp1*(1.f-x0[idloc])*nH[idloc]+3*hubblet));
      //et[igrp]=egyloc[idloc+igrp*BLOCKCOOL];
      
      if(et[igrp]<0) 	{test=1;}
      
      p[igrp]=(1.f+(alphai[igrp]*nH[idloc]*(1-x0[idloc])+2*hubblet)*dtcool);
    }
  ai_tmp1=0.;

  if (test) 
    {fudgecool/=10.f; 
      continue;	} 

  // IONISATION
#ifndef S_X
  for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += alphai[igrp]*et[igrp];}
#else
  N2[0]=1.0f;
  float pp=(1.f-powf(x0[idloc],0.4092f)); 
  if(pp<0.f) pp=0.f; 
    
    N2[1]=1.0f+0.3908f*powf(pp,1.7592f)*E0overI[1]; 
    if(N2[1]<1.0f) N2[1]=1.0f; 
    
    //N2[1]=1.0f;

  for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += alphai[igrp]*et[igrp]*N2[igrp];}
#endif

  xt=1.f-(alpha*x0[idloc]*x0[idloc]*nH[idloc]*dtcool+(1.f -x0[idloc]))/(1.f+dtcool*(beta*x0[idloc]*nH[idloc]+ai_tmp1));
  ai_tmp1=0.;

  if((xt>1.f)||(xt<0.f)) 
    {fudgecool/=10.f; 
      continue;	} 

  cuCompCooling(tloc[idloc],xt,nH[idloc],&Cool,&tcool1,aexp,CLUMPF2);

#ifdef COOLING
  // HEATING
#ifndef S_X
  for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += et[igrp]*(alphae[igrp]*hnu[igrp]-(alphai[igrp]*hnu0));}
#else
  float pp2;
  F2[0]=1.0f;
  F2[1]=1.0f;
  pp2=1.0f-powf(xt,0.2663f); 
  if(pp2<0.f) pp2=0.f; 
  F2[1]=0.9971f*(1.0f-powf(pp2,1.3163f)); 
  
  if(F2[1]>1.0f) F2[1]=1.0f; 
  if(F2[1]<0.0f) F2[1]=0.0f; 
  
  for(igrp=0;igrp<NGRP;igrp++) {ai_tmp1 += et[igrp]*(alphae[igrp]*hnu[igrp]-(alphai[igrp]*hnu0))*F2[igrp];}
#endif

  eint=(eint+dtcool*(nH[idloc]*(1.f-xt)*(ai_tmp1)-Cool))/(1.f+2*hubblet*dtcool);
  //if(eint==0.) printf("NULL TEMP dtcool=%e Cool=%e ai=%e\n",dtcool,Cool,ai_tmp1);
  ai_tmp1=0;

  if(eint<0.f) 
    {
      fudgecool/=10.f; 
      continue;
    } 
#endif

  for(igrp =0;igrp<NGRP;igrp++)
	{egyloc[idloc+igrp*BLOCKCOOL]=et[igrp];
	 floc[0+idloc3+igrp*BLOCKCOOL*3]=floc[0+idloc3+igrp*BLOCKCOOL*3]/p[igrp];
	 floc[1+idloc3+igrp*BLOCKCOOL*3]=floc[1+idloc3+igrp*BLOCKCOOL*3]/p[igrp];
	 floc[2+idloc3+igrp*BLOCKCOOL*3]=floc[2+idloc3+igrp*BLOCKCOOL*3]/p[igrp];	
	}
  
  x0[idloc]=xt;

#ifdef COOLING
      tloc[idloc]=eint/(1.5f*nH[idloc]*KBOLTZ*(1.f+x0[idloc]));
#endif
      currentcool_t+=dtcool;
      if((nitcool==ncvgcool)&&(ncvgcool!=0)) break;
    }

for(igrp=0;igrp<NGRP;igrp++)
  {
    cuegy_new[idx+igrp*NCELL4]=fmax(egyloc[idloc+igrp*BLOCKCOOL]*aexp*aexp*aexp,egy_min*factgrp[igrp]);
    cuflx_new[0*NCELL4+idx+igrp*NCELL4*3]=floc[0+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
    cuflx_new[1*NCELL4+idx+igrp*NCELL4*3]=floc[1+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
    cuflx_new[2*NCELL4+idx+igrp*NCELL4*3]=floc[2+idloc3+igrp*BLOCKCOOL*3]*aexp*aexp;
  }
 

  cutemperature[idx]=tloc[idloc];
  cuxHII[idx]=x0[idloc];
  __syncthreads();
}


#endif
