#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <mpi.h>
#include<math.h>
#include <omp.h>
coucou
int main(int argc, char **argv){	
/********************************************************************************
		Declaration of variables and initialisation.
********************************************************************************/

	// Declaration for MPI.
	MPI_Init(&argc,&argv);
	int nbproc;
	int myrank;
	MPI_Status mystatus ;
	MPI_Comm_size( MPI_COMM_WORLD, &nbproc);	
	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );

	// Variables used to open files and to write in files.
	FILE *FileR; 
	FILE *FileW;
	FileR = fopen(argv[1],"r");
	if(FileR == NULL){ 
		printf("Impossible to open the data file. \n");
		return 1; 
	}
	char DEST_Ez[50] = "Ez";
	char DEST_Hx[50] = "Hx";
	char DEST_Hy[50] = "Hy";

	// Loop variables.
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 
	int step = 1;


	// Importation of param.dat and initialisation of other parameters.





	// Physical constants
	double mu = 4*3.141692*0.0000001;
	double epsilon = 8.854*0.000000000001;
	double Z = sqrt(mu/epsilon);
	double c = 3*100000000;

	// Calculation of the position of the antenna.



	// Variables used to impose the value of E at the antenna.




	/* Division of the domain along x, y and z depending on the number of process.
	   Each process will process a portion of the domain.*/






	/*Calculation of the number of nodes along x, y and z for each process and determination of the minimum 
	and the maximum value of i and j for each process in the global axis.*/








	/* Variables that will contain the value of the fields on the whole domain on a given time.
	   Only the root process will have this information.*/

	double***Ex_tot;
	Ex_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Ex_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Ex_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Ex_tot[i][j][k] = 0;			
			}		
		}
	}

	double***Ey_tot;
	Ey_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Ey_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Ey_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Ey_tot[i][j][k] = 0;			
			}		
		}
	}

	double***Ez_tot;
	Ez_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Ez_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Ez_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Ez_tot[i][j][k] = 0;			
			}		
		}
	}

	double***Hx_tot;
	Hx_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Hx_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Hx_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Hx_tot[i][j][k] = 0;			
			}		
		}
	}

	double***Hy_tot;
	Hy_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Hy_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Hy_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Hy_tot[i][j][k] = 0;			
			}		
		}
	}
	
	double***Hz_tot;
	Hz_tot = (double***)malloc(N*sizeof(double**));
	for(i=0;i<N;i++){
		Hz_tot[i] = (double**)malloc(N*sizeof(double*));
		for(j=0;j<N;j++){
			Hz_tot[i][j]=(double*)malloc(N*sizeof(double));
			for(k=0;k<N;k++){
				Hz_tot[i][j][k] = 0;			
			}		
		}
	}

	// Variables that will contain the previous and the updated value of the fields only on a division of the domain.

	double***Ex_prev;
	double***Ex_new;
	Ex_prev =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	Ex_new =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ex_prev[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
		Ex_new[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
	}
	for(i=0;i<point_per_proc_x[myrank];i++){
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ex_prev[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			Ex_new[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Ex_prev[i][j][k] = 0;
				Ex_new[i][j][k] = 0;	
			}
		}
	}


	double***Ey_prev;
	double***Ey_new;
	Ey_prev =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	Ey_new =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ey_prev[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
		Ey_new[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
	}
	for(i=0;i<point_per_proc_x[myrank];i++){
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ey_prev[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			Ey_new[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Ey_prev[i][j][k] = 0;
				Ey_new[i][j][k] = 0;	
			}
		}
	}


	double***Ez_prev;
	double***Ez_new;
	Ez_prev =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	Ez_new =(double***)malloc(point_per_proc_y[myrank]*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ez_prev[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
		Ez_new[i] =(double**)malloc(point_per_proc_x[myrank]*sizeof(double*));
	}
	for(i=0;i<point_per_proc_x[myrank];i++){
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ez_prev[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			Ez_new[i][j] = (double*)malloc(point_per_proc_x[myrank]*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Ez_prev[i][j][k] = 0;
				Ez_new[i][j][k] = 0;	
			}
		}
	}


	// Variables used to transfer the value of the field at certain places between the different process.	





/********************************************************************************
			Begining of the algorithm.
********************************************************************************/

}
