#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "params.h"
#include "common.h"
#include "bnd.h"
#include "GPU.h"

#include "cosmo.h"
#include "Atomic.h"
#include "communication.h"


#include "Allocation.h"
#include "Io.h"
#include "Explicit.h"
#include "Interface.h"
#include "Boundary.h"

//**********************************************************
//**********************************************************

extern "C" int Mainloop(int rank, int *pos, int *neigh, int ic_rank);

//**********************************************************
//**********************************************************


#define CUERR() //printf("\n %s on %d \n",cudaGetErrorString(cudaGetLastError()),ic_rank)

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define NCELLS3 (NCELLX+NBOUND2)*(NCELLY+NBOUND2)*(NCELLZ+NBOUND2)

#define N_INT 2048
#define A_INT_MAX 0.166667


//**********************************************************
//**********************************************************


int Mainloop(int rank, int *pos, int *neigh, int ic_rank)
{

	#ifdef TIMINGS
		FILE *timefile;
		char ftime[256];
		sprintf(ftime,"time.out.%05d",rank);
		if(rank==0) timefile=fopen(ftime,"w");
	#endif

	if(rank==0) printf("Mainloop entered by proc %d\n",rank);

	float tnext;


	dim3 bcool(BLOCKCOOL);           // USED BY COOLING
	dim3 gcool(GRIDCOOLX,GRIDCOOLY);

	dim3 blocksimple(NCELLX);        // USED BY ADVECTION THREADS
	dim3 gridsimple(NCELLY,NCELLZ);



	#ifndef WMPI
	#else
	
		CUDA_CHECK_ERROR("avt bnd init\n");
		mpisynch();
		if(neigh[5]!=rank)  
		{  
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_zp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[2]%2);
			}
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_zm(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[2]%2);
			}
		}


		mpisynch();
		if(neigh[3]!=rank)
		{
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_yp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[1]%2);
			}
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_ym(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[1]%2);
			}
		}

		mpisynch();
		if(neigh[1]!=rank)
		{
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_xp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[0]%2);
			}
			for (int igrp=0;igrp<NGRP;igrp++)
			{
				cudaMemset(cubuff,0,sizeof(float)*NBUFF);
				memset(buff,0,NBUFF*sizeof(float));
				exchange_xm(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[0]%2);
			}

		}

		if(boundary==0)
		{/*
			if(pos[0]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
			if(pos[1]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_ym<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
			if(pos[2]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_zm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);

			if(pos[0]==(NGPUX-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
			if(pos[1]==(NGPUY-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_yp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
			if(pos[2]==(NGPUZ-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_zp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3) ;

	*/	}

		mpisynch();
		CUDA_CHECK_ERROR("apres bndinit\n");
	#endif


	#ifndef COSMO
		dt=courantnumber*dx/3./c;
		if(rank==0) printf("dx=%e cfl=%e dt=%e\n",dx,courantnumber,dt);
		tnext=t;//+ndumps*dt;
	#else
		aexp=astart;
		#ifndef FLAT_COSMO
			t=a2tgen(aexp,omegam,omegav,Hubble0);// Hubble0 in sec-1
		#else
			t=a2t(aexp,omegav,Hubble0);// Hubble0 in sec-1
		#endif

		tnext=t;
		float tstart=t;
		if(rank==0) printf("aexp= %f tstart=%f tmax=%f, Hubble0=%e\n",aexp,t/unit_time,tmax/unit_time,Hubble0);

		#ifndef FLAT_COSMO
			if(rank==0) printf("Building Expansion factor table");

			float da=(A_INT_MAX-aexp)/N_INT;
			float a_int[N_INT],t_int[N_INT];
			for(int i_int=0;i_int<N_INT;i_int++)
			{
				a_int[i]=aexp+i_int*da;
				t_int[i]=a2tgen(a_int[i],omegam,omegav,Hubble0); // Hubble0 in sec-1
			}

			int n_int=0;

		#endif

	#endif

	// some variables for field update
	int changefield=0;
	int forcedump;
	int ifield=0; // 2 because tfield stores the NEXT field
	float tfield;
	
	if(fieldlist){
		while(aexp>alist[ifield])
		{
			ifield++;
			if(rank==0) printf("a=%e afield=%e ifield=%d\n",t,alist[ifield],ifield);
		}
		tfield=tlist[ifield];
		if(rank==0) printf("ICs (astart=%f) between field #%d (a=%f) and field #%d (a=%f)\n",aexp,ifield-1,alist[ifield-1],ifield,alist[ifield]);
		if(rank==0) printf("starting with NEXT field #%d @ afield =%f with astart=%f\n",ifield,alist[ifield],aexp);// -1 because tfield stores the NEXT field
	}

	float ft=1.;
	#ifdef COSMO
		float factfesc=1.;
	#endif

	#ifdef HELIUM
	float *factgrp_HI,*factgrp_HeI,*factgrp_HeII;
		factgrp_HI=(float*)malloc(NGRP*sizeof(float));
		factgrp_HeI=(float*)malloc(NGRP*sizeof(float));
		factgrp_HeII=(float*)malloc(NGRP*sizeof(float));
		FACTGRP;
	#else
		float *factgrp;
		factgrp=(float*)malloc(NGRP*sizeof(float));
		FACTGRP;
	#endif
	double q0=0.,q1=0.,q3;
	double q4,q7,q8,q9,q10,q11;
	double time_old,time_new;


	// MAIN LOOP STARTS HERE ======================================================>>>>
	// ============================================================================>>>>
	// ============================================================================>>>>
	// ============================================================================>>>>
	// ============================================================================>>>>
	// ============================================================================>>>>

	cudaDeviceSynchronize();
	#ifdef WMPI	  
		mpisynch();
	#endif

	CUDA_CHECK_ERROR("avt loop\n");

	while(t<=tmax)
	{  


		cudaDeviceSynchronize();

		#ifdef WMPI	  
			mpisynch();
			get_elapsed(&time_old);
			q3=q1-q0;
			q0=time_old;
		#endif


		#ifndef COSMO
			dt=courantnumber*dx/3./c*ft;
			if(((nstep%ndisp)==0)&&(rank==0))
			{/*
				printf(" ------------------ \n");
				printf(" Step= %d Time= %f dt=%f tnext=%f cgpu (msec)=%lf\n",nstep,t/unit_time,dt/unit_time,tnext/unit_time,q3);
				printf(" ------------------ \n");
			 */	}
		#else
			dt=courantnumber*dx/3./c*ft;

			// computing boost factor
			float boost;
			if(aboost==0.){
				boost=1.;
			}
			else{
				boost=max(1.,aboost*exp(kboost/(t/unit_time)));
			}

			if(((nstep%ndisp)==0)&&(rank==0))
			{
				printf(" ------------------------------\n");
				printf(" Step= %d Time= %f Elapsed= %f dt= %f aexp=%f z=%f fesc=%f clump= %f boost=%e Next tfield=%f cgpu=%f\n",nstep,t/unit_time,(t-tstart)/unit_time,dt/unit_time,aexp,1./aexp-1.,factfesc*fesc,clump,boost,tfield/unit_time,q3);
				printf(" ----------------------------- \n");
			}
		#endif


		if(fieldlist)
		{
			// we must not go further than the next field
			if(dt>=tfield-t)
			{
				#ifdef WMPI
					if(rank==0) printf("last timestep with field #%d : next field= %f t=%f t+dt=%f\n",ifield,tfield/unit_time,t/unit_time,(t+dt)/unit_time);

					if(((tfield-t)/unit_time)==0.)
					{
						if(rank==0) printf("WARNING FIELD DT=O -> switch immediatly to next field\n"); 
						cuGetField(ifield,ic_rank);

						changefield=0;
						ifield++;
						tfield=tlist[ifield];
						ft=1./powf(2.,20);
					}
					
					else
					{
						changefield=1;
						dt=tfield-t;
						if(rank==0) printf("dt set to %f\n",dt/unit_time);
					}
				#endif
			}
		}

	//================================== UNSPLIT 3D SCHEME=============================

		for (int igrp=0;igrp<NGRP;igrp++)
		{

			#ifdef COSMO
				cuComputeELF<<<gridsimple,blocksimple>>>(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cuegy_new+igrp*NCELLS3, c, dx, dt, nstep,aexp,egy_min);
			#endif

			cudaDeviceSynchronize();
			CUERR();
			if(verbose) puts("Hyperbolic Egy ok");
			CUDA_CHECK_ERROR("Egy");

			#ifdef COSMO
			       cuComputeF_TOTAL_LF<<<gridsimple,blocksimple>>>(cuflx+igrp*NCELLS3*3,cuflx_new+igrp*NCELLS3*3,c,dx,dt,nstep,cuegy+igrp*NCELLS3, aexp);
			#endif

			cudaDeviceSynchronize();
			CUERR();

			if(verbose) puts("Hyperbolic Flux ok");
			CUDA_CHECK_ERROR("FLux");

			cudaDeviceSynchronize();

		}

		#ifdef TIMINGS     
			#ifdef WMPI	  
				mpisynch();
			#endif

			get_elapsed(&q11);
		#endif


		#ifdef TESTCOOL  
			#ifdef COSMO
				cuComputeIon<<<gridion,blockion>>>(cuegy_new, cuflx_new, cuxion, cudensity, cutemperature, dt/cooling, c, egy_min,unit_number,aexp);
			#endif

		#endif
		
		CUERR();
		if(verbose) puts("Chemistry     ok");
		CUDA_CHECK_ERROR("Chem");
		cudaDeviceSynchronize();
		#ifdef WMPI
			mpisynch();
		#endif

		#ifdef TIMINGS
			get_elapsed(&q4);
		#endif


		// Here cuegy is used to store the temperature
		//***********************
		#ifdef COSMO
			float hubblet=Hubble0*sqrtf(omegam/aexp+omegav*(aexp*aexp))/aexp;
			#ifdef HELIUM
			       cuComputeTemp<<<gcool,bcool>>>( cuxHII, cuxHeII, cuxHeIII, cudensity, cutemperature, cuegy_new, fudgecool, c, dt/cooling, unit_number, ncvgcool, aexp, hubblet, cuflx_new, clump,egy_min,fesc,boost,cusrc0);
			#else
			       cuComputeTemp<<<gcool,bcool>>>( cuxHII, cudensity, cutemperature, cuegy_new, fudgecool, c, dt/cooling, unit_number, ncvgcool, aexp, hubblet, cuflx_new, clump,egy_min,fesc,boost,cusrc0);
			#endif
		#endif

		CUERR();
		cudaDeviceSynchronize();

		if(verbose) puts("Cooling  ok");
		CUDA_CHECK_ERROR("Cooling");

		#ifdef TIMINGS
			cudaDeviceSynchronize();
			#ifdef WMPI
				mpisynch();
			#endif
			get_elapsed(&q8);
		#endif

		if(verbose) printf("update on rakn %d\n",rank);
		//****************************************
		cudaMemcpy(cuegy,cuegy_new,NCELLS3*sizeof(float)*NGRP,cudaMemcpyDeviceToDevice);
		cudaMemcpy(cuflx,cuflx_new,NCELLS3*sizeof(float)*3*NGRP,cudaMemcpyDeviceToDevice);

		if(verbose) printf("update done rakn %d\n",rank);
		CUDA_CHECK_ERROR("avt bound 0\n");


		#ifdef TIMINGS
			cudaDeviceSynchronize();
			#ifdef WMPI
				mpisynch();
			#endif
			get_elapsed(&q10);
		#endif


		if(verbose) printf("Dealing with boundaries on rakn %d",rank);

		CUDA_CHECK_ERROR("avt bound\n");

		#ifndef WMPI
			
		#else
			mpisynch();
			if(neigh[5]!=rank)  
			{  
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_zp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[2]%2);
				}
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_zm(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[2]%2);
				}
			}

			mpisynch();
			if(neigh[3]!=rank)
			{
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_yp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[1]%2);
				}
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_ym(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[1]%2);
				}
			}

			mpisynch();
			if(neigh[1]!=rank)
			{
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_xp(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[0]%2);
				}
				for (int igrp=0;igrp<NGRP;igrp++){
					cudaMemset(cubuff,0,sizeof(float)*NBUFF);
					memset(buff,0,NBUFF*sizeof(float));
					exchange_xm(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cubuff, buff, neigh, pos[0]%2);
				}

			}

			if(boundary==0)
			{
/*				if(pos[0]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
				if(pos[1]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_ym<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
				if(pos[2]==0)   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_zm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);

				if(pos[0]==(NGPUX-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
				if(pos[1]==(NGPUY-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_yp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
				if(pos[2]==(NGPUZ-1))   for (int igrp=0;igrp<NGRP;igrp++) cusetboundarytrans_zp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3) ;
*/
			}
		#endif

		cudaDeviceSynchronize(); 
		#ifdef WMPI
			mpisynch();
		#endif

		#ifdef TIMINGS
			get_elapsed(&q7);
		#endif

		CUDA_CHECK_ERROR("aPR bound\n");


		#ifdef SYNCHDUMPFIELD
			if((((nstep%ndumps)==0)||(forcedump))||(changefield))
		#else
			if(((nstep%ndumps)==0)||(forcedump))
		#endif
		{
			ntsteps=ntsteps+1;
			forcedump=0;
			#ifdef COSMO
				#ifdef FLAT_COSMO
					float aexpdump=t2a(t+dt,omegav,Hubble0);
				#else
					if(t+dt>t_int_max)
					{
						aexpdump=(a_int[int_step+2]-a_int[int_step+1])/(t_int[int_step+2]-t_int[int_step+1])*(t+dt-t_int[int_step+1]);
					}
					else
					{
						aexpdump=(a_int[int_step+1]-a_int[int_step])/(t_int[int_step+1]-t_int[int_step])*(t+dt-t_int[int_step]);
					}
				#endif
			
				printf("proc %d dumping file #%d\n",rank,ic_rank);

				cuDumpResults(ntsteps,t+dt,aexpdump,ic_rank);

			#endif
			
			tnext=tnext+ndumps*dt/ft;
			if(rank==0) printf("tnext=%f\n",tnext/unit_time);
			#ifdef WMPI
				mpisynch();
			#endif
		}

		//--------------------------------------------------------------------
		// Dealing with fieldlists
		//--------------------------------------------------------------------

		ft=fminf(ft*2.,1.);

		if(fieldlist)
		{
			if(changefield)
			{
				int ercode;
				#ifdef WMPI
					ercode=cuGetField(ifield,ic_rank);
				#endif
				if(ercode==38)
				{
					//fclose(logfile);
					if(rank==0) fclose(timefile);
					abort();
				}
				forcedump=0;
				changefield=0;
				ifield++;
				tfield=tlist[ifield];
				ft=1./powf(2.,20);
				#ifdef WMPI	  
					mpisynch();
				#endif
			}
		}


		// UPDATING VARIABLES

		t=t+dt;
		if(t>tmax)
		{
			puts("t > tmax -----> run will be terminated");
		}
		#ifdef COSMO

			#ifdef FLAT_COSMO
				aexp=t2a(t,omegav,Hubble0); // A CHANGER PAR INTERPOLATION
			#else
				if(t>t_int_max)
				{
					int_step++;
				}
				aexp=(a_int[int_step+1]-a_int[int_step])/(t_int[int_step+1]-t_int[int_step])*(t-t_int[int_step]);
			#endif
			c=c_r/aexp;
		#endif       

		cudaDeviceSynchronize();
		#ifdef WMPI
			mpisynch();
		#endif

		get_elapsed(&q1);

		nstep++;
		if(nstep==nmax) {
			if(rank==0) puts("Max number of steps achieved: STOP");
			break;
		}

		cudaDeviceSynchronize();
		#ifdef WMPI
			get_elapsed(&time_new);
			time_new=time_new-time_old;
			mpireducemax(&time_new);
			mpisynch();
		#endif

		#ifdef TIMINGS
			if(rank==0){
				get_elapsed(&q9);
				//	printf("transport=%lf chem=%lf cool=%lf update=%lf bound=%lf IO=%lf,grand total=%lf time_new=%lf\n",q11-q0,q4-q11,q8-q4,q10-q8,q7-q10,q9-q7,q9-q0,time_new);
				//fprintf(timefile,"%d %lf %lf %lf %lf %lf %lf %lf\n",nstep-1,q11-q0,q4-q11,q8-q4,q10-q8,q7-q10,q9-q7,q9-q0,time_new);
			}
		#endif

		cudaDeviceSynchronize();
		#ifdef WMPI	  
			mpisynch();
		#endif

	}

	#ifdef TIMINGS
		if(rank==0) fclose(timefile);
	#endif
	return 0;
}
