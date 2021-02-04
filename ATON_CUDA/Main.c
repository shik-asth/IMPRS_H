#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef WMPI
	#include <mpi.h>
#endif

#include "timestep.h"
#include "GPU.h"
#include "check.h"
#include "Io.h"
#include "Interface.h"
#include "Allocation.h"

#ifdef TEST
#include "bnd.h"
#endif
extern void topo_cartesian(int rank, int *dims, int *coords, int *neigh);
//**********************************************************
//**********************************************************
float *egy,*egy_new;
float *flx,*flx_new;

float *dedd;
float *src0;
int *src0pos;


float *temperature,*density,*xHII;
float *buff;
#ifdef HELIUM
	float *xHeII,*xHeIII;
#endif


//**********************************************************
//**********************************************************

float *cuegy,*cuegy_new;
float *cuflx,*cuflx_new;

float *cudedd;
float *cusrc0;
int *cusrc0pos;

float *cutemperature,*cuxHII;
float *cudensity;
#ifdef HELIUM
	float *cuxHeII, *cuxHeIII;
#endif
float *cubuff;

float t=0.;
float dt;
int nstep,ntsteps;
float aexp;
float zexp;

int nalist;
float *alist;
float *tlist;

int ncells;
float dx;
float tmax;
float courantnumber;
int ndumps;
int ndisp;
int nmax;
char rootname[512];
int verbose;
int interactive;
int nrestart;

float c;
float c_r=2.99792458e8;
float kb=1.3806e-23;
float Tcmb=2.726;
float effective_speed_of_light;
float unit_length;
float unit_time;
float unit_number;

float egy_min;
float defdens;
float deftemp;
int boundary;
int cooling;
float fudgecool;

int ncvgcool;

int nsource;
float fesc;
float aboost;
float kboost;
float a0;
float clump;

float astart=0.;
float Hubble0;
float omegam;
float omegav;

int fieldlist;
char fieldname[256];
char filealist[256];


//**********************************************************
//**********************************************************

void logo(void)
{
	puts("");
	puts("******************************************");
	puts("               CUDaton V.0.2               *");
	puts("******************************************");
	puts("");
}

//**********************************************************
//**********************************************************

int GetParameters(char *fparam)
{
	FILE *buf;
	char stream[512];
	buf=fopen(fparam,"r");
	if(buf==NULL)
	{
		printf("ERROR : cannot open %s, please check\n",fparam);
		return 1;
	}

	else
	{
		fscanf(buf,"%s %d",stream,&verbose);
		fscanf(buf,"%s %s",stream,rootname);
		fscanf(buf,"%s %d",stream,&nrestart);
		fscanf(buf,"%s %d",stream,&ncells);
		if(ncells!=NCELLS)
		{
			printf("file ncells different from the Hard coded value ncells=%d NCELLS=%d\n",ncells,NCELLS);
			abort();
		}

		fscanf(buf,"%s %f",stream,&dx);

		fscanf(buf,"%s %f",stream,&tmax);
		fscanf(buf,"%s %f",stream,&courantnumber);
		fscanf(buf,"%s %d",stream,&ndumps);
		fscanf(buf,"%s %d",stream,&ndisp);
		fscanf(buf,"%s %d",stream,&nmax);
	
		fscanf(buf,"%s %f",stream,&unit_length);
		fscanf(buf,"%s %f",stream,&unit_time);
		fscanf(buf,"%s %f",stream,&unit_number);
	
		fscanf(buf,"%s %f",stream,&effective_speed_of_light);
	
		fscanf(buf,"%s %f",stream,&egy_min);
		fscanf(buf,"%s %f",stream,&defdens);
		fscanf(buf,"%s %f",stream,&deftemp);
	
		fscanf(buf,"%s %d",stream,&boundary);
	
		fscanf(buf,"%s %d",stream,&cooling);
		fscanf(buf,"%s %f",stream,&fudgecool);

		fscanf(buf,"%s %d",stream,&ncvgcool);

		fscanf(buf,"%s %d",stream,&nsource);
		fscanf(buf,"%s %f",stream,&fesc);
		fscanf(buf,"%s %f",stream,&aboost);
		fscanf(buf,"%s %f",stream,&kboost);

		fscanf(buf,"%s %f",stream,&Hubble0);
		fscanf(buf,"%s %f",stream,&omegam);
		fscanf(buf,"%s %f",stream,&omegav);

		fscanf(buf,"%s %d",stream,&fieldlist);
		if(fieldlist)
		{
			fscanf(buf,"%s %s",stream,fieldname);
			fscanf(buf,"%s %s",stream,filealist);
		}

		fclose(buf);

		#ifdef COSMO
			if((omegam+omegav)!=1.)
			{
				printf(" Error omegam+omegav= %f !=1 . Only flat models please !\n",omegam+omegav);
			}
		#endif

		dx=dx*unit_length;
		tmax=tmax*unit_time;

	}
	return 0;
}

//**********************************************************

int main(int argc, char *argv[])
{
	int code;

	#ifdef WMPI
		int mpi_rank,mpi_size,ic_rank;
		int ierr;
		int ndevice;
		int mpi_coords[3];
		int mpi_neighbors[6];
		int dims[3];

		// Number of proc per dimension
		dims[0]=NGPUX;
		dims[1]=NGPUY;
		dims[2]=NGPUZ;

		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
		MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);

		ndevice=countdevices(mpi_rank);


		if(mpi_rank==0) printf("Starting with %d GPUs\n",mpi_size);

		initlocaldevice(mpi_rank,1);

		topo_cartesian(mpi_rank,dims,mpi_coords,mpi_neighbors);
		ic_rank=mpi_coords[0]+mpi_coords[1]*dims[0]+mpi_coords[2]*dims[0]*dims[1];

		if(mpi_rank==0)
		{
			#else 
				//initlocaldevice(1,1);
			#endif

			if(argc<2)
			{
				puts("usage: ./caton file.param");
				return 1;
			}

			logo();

			#ifdef WMPI
		}
	#endif



	code=GetParameters(argv[1]);
	if(code!=0) return 1;
	if(verbose) puts("Parameters ok");



	#ifdef WMPI
		if(mpi_rank==0) printf("proc %d : astart=%f\n",mpi_rank,astart);
	#endif

	
	Allocation();
	


	mpi_task_rank = mpi_rank;
	MPI_Get_processor_name(host_name, &host_name_length);
	
	cuAllocation();
	
	if(verbose) puts("Allocation ok");


	#ifdef WMPI
		#ifndef TEST_STROMGREN
			printf("proc %d reading ic #%d\n",mpi_rank,ic_rank);
			cuGetIC(nrestart,ic_rank);
			#ifdef TEST
				printf("Checking the variable information \n");
				for (int abc=0; abc<(NCELLX+NBOUND2)*(NCELLY+NBOUND2)*(NCELLZ+NBOUND2);abc++)
				{
					if(xHII[abc]!=0)
					{      
						printf("xHII[%d,%d]=%f, xHeII=%f, xHeIII=%f\n",mpi_rank,abc,xHII[abc],xHeII[abc],xHeIII[abc]);
					}
			  	}
			#endif
                #else
			int middle=0;
			middle=(mpi_coords[0]==dims[0]/2)*(mpi_coords[1]==dims[1]/2)*(mpi_coords[2]==dims[2]/2);
			cuGetIC_TEST(nrestart,ic_rank,middle);
		#endif
	#endif

	#ifndef COSMO
		c=effective_speed_of_light*c_r;
	#else
		c=effective_speed_of_light*c_r/astart;
		Hubble0=Hubble0/(9.7776e9*3.155815e7); // H0 in sec-1
		if(mpi_rank==0) printf("Hubble0=%e\n",Hubble0);
	#endif

	nstep=0;
	ntsteps=nrestart;
	if(mpi_rank==0) printf("astart=%e\n",astart);



	if(fieldlist)
	{
		#ifdef WMPI
			if(mpi_rank==0) puts("Start field setup");
			getalist(mpi_rank);

			if(mpi_rank==0) printf("astart in list=%e\n",alist[0]);
			if(mpi_rank==0) puts("field setup ok");
		#else
			puts("Start field setup");
			getalist(0);
			if(mpi_rank==0) printf("astart in list=%e\n",alist[0]);
			puts("field setup ok");
		#endif
	}
	else
	{
		//puts("no fieldlist");
	}



	#ifdef WMPI
		MPI_Barrier(MPI_COMM_WORLD);
		Mainloop(mpi_rank,mpi_coords,mpi_neighbors,ic_rank);
		MPI_Finalize();
	#endif

	return 0;

}

