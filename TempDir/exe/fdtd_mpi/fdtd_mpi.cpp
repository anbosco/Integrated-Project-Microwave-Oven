// example of hybrid MPI/OpenMP program
//     run with 2 processes and 6 threads per process (ubuntu)
//         export OMP_NUM_THREADS=6
//         [ubuntu - openmpi]
//         mpirun -np 2 -cpus-per-rank 6 --bind-to core:overload-allowed  bin/fdtd_mpi
//         [windows - microsoft mpi]
//         mpiexec -np 2 bin\fdtd_mpi

#include "vtl.h"
#include "vtlSPoints.h"

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <cassert>

#include <mpi.h>
#include <omp.h>
#include <stdlib.h>

#include<stdio.h>
#include<string.h>
#include<math.h>

using namespace vtl;

void divisor(int np,int*r);
void compute_proc_ind_point(int nbproc,int Nx,int Ny,int Nz,int divx,int divy,int divz,int*i_min_proc,int*j_min_proc,int*k_min_proc,int*i_max_proc,int*j_max_proc,int*k_max_proc,int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z);
//void Update_prev_in_send(int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z,int lastx,int lasty,int lastz,double*V1,double***M1,double*V2,double***M2,int myrank,int Case);
void Update_E_inside(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double dt,double dx,double e_0,double***e_r,int Case);
void Update_E_boundary(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double**H_boundary,double dt,double dx,double e_0,double***e_r,int myrank,int Case);
void Boundary_condition_imosition(int ip,int jp,int kp,int lastx,int lasty,int lastz,int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z,double***Ex_new,double***Ey_new,double***Ez_new,int myrank);
void Update_H_inside(int i_max,int j_max,int k_max,int last1, int last2,double***H_new,double***H_prev,double***E1_prev,double***E2_prev,double dt,double dx,double mu_0,double***mu_r,int Case);
void Update_H_boundary(int i_max,int j_max,int k_max,int last,double***H_new,double***H_prev,double***E1_prev,double***E2_prev,double**E_boundary,double dt,double dx,double mu_0,double***mu_r,int myrank,int Case);
void New_in_old(int i_max,int j_max,int k_max,double***New,double***Old);
void insert_obj(double***e_r,int nb_obj,double*prop_obj,double dx,double point_per_proc_x,double point_per_proc_y,double point_per_proc_z,int lastx,int lasty,int lastz,int i_min_proc,int j_min_proc,int k_min_proc,int i_max_proc,int j_max_proc,int k_max_proc);
int compare(int temp);
void Update_send_in_mat(int i_max1,int j_max1, int i_max2,int j_max2,double**M1,double*V1,double**M2,double*V2);
void Update_prev_in_send(int i_max1,int jmax1,int k_max1,int i_max2,int j_max2,int k_max2,double*V1,double***M1,double*V2,double***M2, int Case, int point_max);
void Hom_BC(int i_min, int j_min, int k_min, int i_max, int j_max, int k_max,double*** M);
void init_geometry(std::vector<double> &geometry_init,std::vector<double> &vec_e_r,std::vector<double> &vec_mu_r,std::vector<double> &e_r_tot, std::vector<double> &mu_r_tot, int Nx, int Ny, int Nz);
void init_geom_one_proc(double***e_r,double***mu_r,std::vector<double> &e_r_tot,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz);
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &geometry_new,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta);
void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx, double val);


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
	int l = 0;
	int step = 1;


	// Importation of param.dat and initialisation of other parameters.
	double data[18];	
	char chain[150];
	for (i=0 ; i<18; i++){
		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the data file. \n");
			return 1; 
		}
		else{
		data[i] = atof(chain);
		}
	}
	fclose(FileR);
	double Lx = data[0];
	double Ly = data[1];
	double Lz = data[2];
 	double f = data[3];
  	double omega = f*2*3.141692;
	double dx = data[4];
 	int Nx = (int) (Lx/dx)+1;
	int Ny = (int) (Ly/dx)+1;
	int Nz = (int) (Lz/dx)+1;
	double dt = data[5]; 
	double Tf = data[6];
	double temp = Tf/dt;
	int step_max = (int) temp;
	int P = (int) data[7];
	double S = data[8];
	int SR = (int) 1/S;
	double l_ax = data[9];
	double l_ay = data[10];
	double l_az = data[11];
	double pos_ax = data[12];
	double pos_ay = data[13];
	double pos_az = data[14];
	int n_sphere = (int) data[15];
	int n_cylinder = (int) data[16];
	int n_cube = (int) data[17];
	double theta = 0;

	std::vector<double> prop_sphere;
	FileR = fopen(argv[2],"r");
	if(FileR == NULL){ 
		printf("Impossible to open the sphere file (Object property). \n");
		return 1; 
	}		
	for (i=0 ; i<5*n_sphere; i++){
		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the sphere file. \n");
			return 1; 
		}
		else{
			prop_sphere.push_back(atof(chain));
		}
	}
	fclose(FileR);

	std::vector<double> prop_cylinder;
	FileR = fopen(argv[3],"r");
	if(FileR == NULL){ 
		printf("Impossible to open the sphere file (Object property). \n");
		return 1; 
	}		
	for (i=0 ; i<7*n_cylinder; i++){
		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the sphere file. \n");
			return 1; 
		}
		else{
			prop_cylinder.push_back(atof(chain));
		}
	}
	fclose(FileR);

	std::vector<double> prop_cube;
	FileR = fopen(argv[4],"r");
	if(FileR == NULL){ 
		printf("Impossible to open the sphere file (Object property). \n");
		return 1; 
	}		
	for (i=0 ; i<7*n_cube; i++){
		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the sphere file. \n");
			return 1; 
		}
		else{
			prop_cube.push_back(atof(chain));
		}
	}
	fclose(FileR);

	std::vector<double> vec_e_r;
	std::vector<double> vec_mu_r;
 
	vec_e_r.push_back(1);
	vec_e_r.push_back(5);
	vec_e_r.push_back(10);
	vec_e_r.push_back(15);
	
	vec_mu_r.push_back(1);
	vec_mu_r.push_back(1);
	vec_mu_r.push_back(1);
	vec_mu_r.push_back(1);

	/* Division of the domain along x, y and z depending on the number of process.
	   Each process will process a portion of the domain.*/
	int divx = 1;
	int divy = 1;
	int divz = 1;
	int r[3];
	divisor(nbproc,r);
	divx = r[0];
	divy = r[1];
	divz = r[2];

	/*Calculation of the number of nodes along x, y and z for each process and determination of the minimum 
	and the maximum value of i and j for each process in the global axis.*/	
	int*i_min_proc = (int*)malloc(nbproc*sizeof(int));
	int*j_min_proc = (int*)malloc(nbproc*sizeof(int));
	int*k_min_proc = (int*)malloc(nbproc*sizeof(int));
	int*i_max_proc = (int*)malloc(nbproc*sizeof(int));
	int*j_max_proc = (int*)malloc(nbproc*sizeof(int));
	int*k_max_proc = (int*)malloc(nbproc*sizeof(int));
	int*point_per_proc_x = (int*)malloc(nbproc*sizeof(int));
	int*point_per_proc_y = (int*)malloc(nbproc*sizeof(int));
	int*point_per_proc_z = (int*)malloc(nbproc*sizeof(int)); 
	compute_proc_ind_point(nbproc,Nx,Ny,Nz,divx,divy,divz,i_min_proc,j_min_proc,k_min_proc,i_max_proc,j_max_proc,k_max_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z);
   	
			
	/* Variables useful to define the size of the fields on each process */
	int ip = (myrank/(divy*divz));
	int jp = (myrank%(divy*divz))%divy;
	int kp = (myrank%(divy*divz))/divy;
	int lastx = 0;
	int lasty = 0;
	int lastz = 0;
	if(divx == 1){	
		lastx = 1;
	}
	else{
		lastx = ip/(divx-1);
	}	
	if(divy == 1){	
		lasty = 1;
	}
	else{
		lasty = jp/(divy-1);
	}	
	if(divz == 1){	
		lastz = 1;
	}
	else{
		lastz = kp/(divz-1);
	}

	// Physical constants
	double mu_0 = 4*3.141692*0.0000001;
	double e_0 = 8.854*0.000000000001;
	double Z = sqrt(mu_0/e_0);
	double c = 3*100000000;
	double***e_r;
	double***mu_r;
	e_r =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	mu_r =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	std::vector<double> e_r_tot(Nx*Ny*Nz);
	std::vector<double> mu_r_tot(Nx*Ny*Nz);
	std::vector<double> geometry_init(Nx*Ny*Nz);
	std::vector<double> geometry_current(Nx*Ny*Nz);
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		e_r[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		mu_r[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			e_r[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));			
			mu_r[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
						e_r[i][j][k] = 1;
						mu_r[i][j][k] = 1;
			}
		}
	}

	/* Insertion of object*/
	std::vector<double> prop_temp(7);
	// Sphere	
	for(i=0;i<n_sphere;i++){		
		prop_temp[0] = prop_sphere[5*i];
		prop_temp[1] = prop_sphere[5*i+1];
		prop_temp[2] = prop_sphere[5*i+2];
		prop_temp[3] = prop_sphere[5*i+3];
		prop_temp[4] = prop_sphere[5*i+4];
		int Config = 0;
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[4]);
	}
	//Cylinder
	for(i=0;i<n_cylinder;i++){
		prop_temp[0] = prop_cylinder[7*i];
		prop_temp[1] = prop_cylinder[7*i+1];
		prop_temp[2] = prop_cylinder[7*i+2];
		prop_temp[3] = prop_cylinder[7*i+3];
		prop_temp[4] = prop_cylinder[7*i+4];
		prop_temp[5] = prop_cylinder[7*i+5];
		prop_temp[6] = prop_cylinder[7*i+6];
		int Config = 1;
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[5]);
	}
	// Cube
	for(i=0;i<n_cube;i++){
		prop_temp[0] = prop_cube[7*i];
		prop_temp[1] = prop_cube[7*i+1];
		prop_temp[2] = prop_cube[7*i+2];
		prop_temp[3] = prop_cube[7*i+3];
		prop_temp[4] = prop_cube[7*i+4];
		prop_temp[5] = prop_cube[7*i+5];
		prop_temp[6] = prop_cube[7*i+6];
		int Config = 2;
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[6]);
	}

	init_geometry(geometry_init,vec_e_r,vec_mu_r,e_r_tot, mu_r_tot,Nx, Ny, Nz);
	init_geom_one_proc(e_r,mu_r,e_r_tot,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz, Nx, Ny, Nz);
	// Insertion of the objects inside the domain
	/*double*prop_obj = (double*)malloc(7*nb_obj*sizeof(double));
	FileR = fopen(argv[2],"r");
	if(FileR == NULL){ 
		printf("Impossible to open the data file (Object property). \n");
		return 1; 
	}		
	for (i=0 ; i<7*nb_obj; i++){
		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the data file. \n");
			return 1; 
		}
		else{
			prop_obj[i] = atof(chain);
		}
	}
	fclose(FileR);*/
	/*if (nb_obj!=0){
		insert_obj(e_r,nb_obj,prop_obj,dx,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz,i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],i_max_proc[myrank],j_max_proc[myrank],k_max_proc[myrank]);
	}*/

	// Calculation of the position of the antenna.
	double n_ax_double = (l_ax/dx)+1;
	int n_ax = (int) n_ax_double;
	pos_ax = (pos_ax/dx);	
	int i_min_a = (int)pos_ax;
	i_min_a = i_min_a - (n_ax/2);
	int i_max_a = i_min_a + n_ax-1;

	double n_ay_double = (l_ay/dx)+1;
	int n_ay = (int) n_ay_double;
	pos_ay = (pos_ay/dx);	
	int j_min_a = (int)pos_ay;
	j_min_a = j_min_a - (n_ay/2);
	int j_max_a = j_min_a + n_ay-1;

	double n_az_double = (l_az/dx)+1;
	int n_az = (int) n_az_double;
	pos_az = (pos_az/dx);	
	int k_min_a = (int)pos_az;
	k_min_a = k_min_a - (n_az/2);
	int k_max_a = k_min_a + n_az-1;

	// Variables used to impose the value of E at the antenna.
	int b_inf_x =0;
	int b_inf_y =0;
	int b_inf_z =0;
	int b_sup_x = 0;
	int b_sup_y = 0;
	int b_sup_z = 0;
	
	/* Variables used to export the results  under the proper fomat. */

	//Electric field

	// X component
	SPoints grid_Ex;
	grid_Ex.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Ex.np1 = Vec3i(0, 0, 0);       
	grid_Ex.np2 = Vec3i(Nx, Ny-1, Nz-1); 
	grid_Ex.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Ex;
	mygrid_Ex.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Ex.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Ex.np2 = Vec3i(i_max_proc[myrank]+lastx, j_max_proc[myrank], k_max_proc[myrank]);
	mygrid_Ex.dx = Vec3d(dx, dx, dx);
  	mygrid_Ex.id = myrank;

	std::vector<SPoints> sgrids_Ex; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Ex.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Ex[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Ex[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divx!=1){
				sgrids_Ex[l].np2 = Vec3i(i_max_proc[l]+(l/(divy*divz))/(divx-1), j_max_proc[l], k_max_proc[l]);
			}
			else{
				sgrids_Ex[l].np2 = Vec3i(i_max_proc[l]+1, j_max_proc[l], k_max_proc[l]);
			}
			sgrids_Ex[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Ex[l].id = l;		 		
		}
	}

	 // creation of the fields (over my subdomain)
	int mynbp_Ex = mygrid_Ex.nbp();
	std::vector<double> Ex_vec(mynbp_Ex);
	for(i=0;i<mynbp_Ex;i++){
		Ex_vec[i] = 0;
	}
 	mygrid_Ex.scalars["E x"] = &Ex_vec;


	// Y component
	
	SPoints grid_Ey;
	grid_Ey.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Ey.np1 = Vec3i(0, 0, 0);      
	grid_Ey.np2 = Vec3i(Nx-1, Ny, Nz-1); 
	grid_Ey.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Ey;
	mygrid_Ey.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Ey.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Ey.np2 = Vec3i(i_max_proc[myrank], j_max_proc[myrank]+lasty, k_max_proc[myrank]);
	mygrid_Ey.dx = Vec3d(dx, dx, dx);
  	mygrid_Ey.id = myrank;

	std::vector<SPoints> sgrids_Ey; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Ey.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Ey[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Ey[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divy!=1){
				sgrids_Ey[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+((l%(divy*divz))%divy)/(divy-1), k_max_proc[l]);
			}
			else{
				sgrids_Ey[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+1, k_max_proc[l]);
			}
			sgrids_Ey[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Ey[l].id = l;		 		
		}
	}

	 // creation of the fields (over my subdomain)
	int mynbp_Ey = mygrid_Ey.nbp();
	std::vector<double> Ey_vec(mynbp_Ey);
  std::vector<double> er_vec(mynbp_Ey);
	for(i=0;i<mynbp_Ey;i++){
		Ey_vec[i] = 0;
	}
 	mygrid_Ey.scalars["E Y"] = &Ey_vec;
  mygrid_Ey.scalars["Geometry"] = &er_vec;

	// Z component
	SPoints grid_Ez;
	grid_Ez.o = Vec3d(10.0, 10.0, 10.0);
	grid_Ez.np1 = Vec3i(0, 0, 0);       
	grid_Ez.np2 = Vec3i(Nx-1, Ny-1, Nz); 
	grid_Ez.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Ez;
	mygrid_Ez.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Ez.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Ez.np2 = Vec3i(i_max_proc[myrank], j_max_proc[myrank], k_max_proc[myrank]+lastz);
	mygrid_Ez.dx = Vec3d(dx, dx, dx);
  	mygrid_Ez.id = myrank;

	std::vector<SPoints> sgrids_Ez; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Ez.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Ez[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Ez[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divz!=1){
				sgrids_Ez[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l], k_max_proc[l]+((l%(divy*divz))/divy)/(divz-1));
			}
			else{
				sgrids_Ez[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l], k_max_proc[l]+1);
			}
			sgrids_Ez[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Ez[l].id = l;		 		
		}
	}

	 // creation of the fields (over my subdomain)
	int mynbp_Ez = mygrid_Ez.nbp();
	std::vector<double> Ez_vec(mynbp_Ez);
	for(i=0;i<mynbp_Ez;i++){
		Ez_vec[i] = 0;
	}
 	mygrid_Ez.scalars["E z"] = &Ez_vec;

	// Magnetic field

	// X component
	SPoints grid_Hx;
	grid_Hx.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Hx.np1 = Vec3i(0, 0, 0);       
	grid_Hx.np2 = Vec3i(Nx-1, Ny, Nz); 
	grid_Hx.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Hx;
	mygrid_Hx.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Hx.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Hx.np2 = Vec3i(i_max_proc[myrank], j_max_proc[myrank]+lasty, k_max_proc[myrank]+lastz);
	mygrid_Hx.dx = Vec3d(dx, dx, dx);
  	mygrid_Hx.id = myrank;

	std::vector<SPoints> sgrids_Hx; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Hx.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Hx[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Hx[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divy!=1 && divz !=1){
				sgrids_Hx[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+((l%(divy*divz))%divy)/(divy-1), k_max_proc[l]+((l%(divy*divz))/divy)/(divz-1));
			}
			else if(divy!=1){
				sgrids_Hx[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+((l%(divy*divz))%divy)/(divy-1), k_max_proc[l]+1);
			}
			else if(divz!=1){
				sgrids_Hx[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+1, k_max_proc[l]+((l%(divy*divz))/divy)/(divz-1));
			}
			else{
				sgrids_Hx[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l]+1, k_max_proc[l]+1);
			}
			sgrids_Hx[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Hx[l].id = l;		 		
		}
	}
	 // creation of the fields (over my subdomain)
	int mynbp_Hx = mygrid_Hx.nbp();
	std::vector<double> Hx_vec(mynbp_Hx);
	for(i=0;i<mynbp_Hx;i++){
		Hx_vec[i] = 0;
	}
 	mygrid_Hx.scalars["H x"] = &Hx_vec;

	// Y component
	SPoints grid_Hy;
	grid_Hy.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Hy.np1 = Vec3i(0, 0, 0);       
	grid_Hy.np2 = Vec3i(Nx, Ny-1, Nz); 
	grid_Hy.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Hy;
	mygrid_Hy.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Hy.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Hy.np2 = Vec3i(i_max_proc[myrank]+lastx, j_max_proc[myrank], k_max_proc[myrank]+lastz);
	mygrid_Hy.dx = Vec3d(dx, dx, dx);
  	mygrid_Hy.id = myrank;

	std::vector<SPoints> sgrids_Hy; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Hy.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Hy[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Hy[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divx!=1 && divz !=1){
				sgrids_Hy[l].np2 = Vec3i(i_max_proc[l]+(l/(divy*divz))/(divx-1), j_max_proc[l], k_max_proc[l]+((l%(divy*divz))/divy)/(divz-1));
			}
			else if(divx!=1){
				sgrids_Hy[l].np2 = Vec3i(i_max_proc[l]+(l/(divy*divz))/(divx-1), j_max_proc[l], k_max_proc[l]+1);
			}
			else if(divz!=1){
				sgrids_Hy[l].np2 = Vec3i(i_max_proc[l]+1, j_max_proc[l], k_max_proc[l]+((l%(divy*divz))/divy)/(divz-1));
			}
			else{
				sgrids_Hy[l].np2 = Vec3i(i_max_proc[l]+1, j_max_proc[l], k_max_proc[l]+1);
			}
			sgrids_Hy[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Hy[l].id = l;		 		
		}
	}

	 // creation of the fields (over my subdomain)
	int mynbp_Hy = mygrid_Hy.nbp();
	std::vector<double> Hy_vec(mynbp_Hy);
	for(i=0;i<mynbp_Hy;i++){
		Hy_vec[i] = 0;
	}
 	mygrid_Hy.scalars["H y"] = &Hy_vec;

	// Z component
	SPoints grid_Hz;
	grid_Hz.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Hz.np1 = Vec3i(0, 0, 0);       
	grid_Hz.np2 = Vec3i(Nx, Ny, Nz-1); 
	grid_Hz.dx = Vec3d(dx, dx, dx); 

	SPoints mygrid_Hz;
	mygrid_Hz.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Hz.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Hz.np2 = Vec3i(i_max_proc[myrank]+lastx, j_max_proc[myrank]+lasty, k_max_proc[myrank]);
	mygrid_Hz.dx = Vec3d(dx, dx, dx);
  	mygrid_Hz.id = myrank;

	std::vector<SPoints> sgrids_Hz; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Hz.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Hz[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Hz[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divx!=1 && divy !=1){
				sgrids_Hz[l].np2 = Vec3i(i_max_proc[l]+(l/(divy*divz))/(divx-1), j_max_proc[l]+((l%(divy*divz))%divy)/(divy-1), k_max_proc[l]);
			}
			else if(divx!=1){
				sgrids_Hz[l].np2 = Vec3i(i_max_proc[l]+(l/(divy*divz))/(divx-1), j_max_proc[l]+1, k_max_proc[l]);
			}
			else if(divy!=1){
				sgrids_Hz[l].np2 = Vec3i(i_max_proc[l]+1, j_max_proc[l]+((l%(divy*divz))%divy)/(divy-1), k_max_proc[l]);
			}
			else{
				sgrids_Hz[l].np2 = Vec3i(i_max_proc[l]+1, j_max_proc[l]+1, k_max_proc[l]);
			}
			sgrids_Hz[l].dx = Vec3d(dx, dx, dx);		
		}
	}

	 // creation of the fields (over my subdomain)
	int mynbp_Hz = mygrid_Hz.nbp();
	std::vector<double> Hz_vec(mynbp_Hz);
	for(i=0;i<mynbp_Hz;i++){
		Hz_vec[i] = 0;
	}
 	mygrid_Hz.scalars["H z"] = &Hz_vec;		
	

	/* Variables that will contain the previous and the updated value of the fields only on a division of the domain.*/
	double***Ex_prev;
	double***Ex_new;
	Ex_prev =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	Ex_new =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Ex_prev[i] =(double**)malloc(point_per_proc_y[myrank]*sizeof(double*));
		Ex_new[i] =(double**)malloc(point_per_proc_y[myrank]*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ex_prev[i][j] = (double*)malloc(point_per_proc_z[myrank]*sizeof(double));
			Ex_new[i][j] = (double*)malloc(point_per_proc_z[myrank]*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Ex_prev[i][j][k] = 0;
				Ex_new[i][j][k] = 0;	
			}
		}
	}

	double***Ey_prev;
	double***Ey_new;
	Ey_prev =(double***)malloc(point_per_proc_x[myrank]*sizeof(double**));
	Ey_new =(double***)malloc(point_per_proc_x[myrank]*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ey_prev[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		Ey_new[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			Ey_prev[i][j] = (double*)malloc(point_per_proc_z[myrank]*sizeof(double));
			Ey_new[i][j] = (double*)malloc(point_per_proc_z[myrank]*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Ey_prev[i][j][k] = 0;
				Ey_new[i][j][k] = 0;	
			}
		}
	}

	double***Ez_prev;
	double***Ez_new;
	Ez_prev =(double***)malloc(point_per_proc_x[myrank]*sizeof(double**));
	Ez_new =(double***)malloc(point_per_proc_x[myrank]*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ez_prev[i] =(double**)malloc(point_per_proc_y[myrank]*sizeof(double*));
		Ez_new[i] =(double**)malloc(point_per_proc_y[myrank]*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ez_prev[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			Ez_new[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				Ez_prev[i][j][k] = 0;
				Ez_new[i][j][k] = 0;	
			}
		}
	}

	double temp1 = 0;
	double temp2 = 0;	

	double***Hx_prev;
	double***Hx_new;
	Hx_prev =(double***)malloc((point_per_proc_x[myrank])*sizeof(double**));
	Hx_new =(double***)malloc((point_per_proc_x[myrank])*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Hx_prev[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		Hx_new[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			Hx_prev[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			Hx_new[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				Hx_prev[i][j][k] = 0;
				Hx_new[i][j][k] = 0;	
			}
		}
	}

	double***Hy_prev;
	double***Hy_new;
	Hy_prev =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	Hy_new =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Hy_prev[i] =(double**)malloc((point_per_proc_y[myrank])*sizeof(double*));
		Hy_new[i] =(double**)malloc((point_per_proc_y[myrank])*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank];j++){
			Hy_prev[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			Hy_new[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				Hy_prev[i][j][k] = 0;
				Hy_new[i][j][k] = 0;	
			}
		}
	}

	double***Hz_prev;
	double***Hz_new;
	Hz_prev =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	Hz_new =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Hz_prev[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		Hz_new[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			Hz_prev[i][j] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
			Hz_new[i][j] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank];k++){
				Hz_prev[i][j][k] = 0;
				Hz_new[i][j][k] = 0;	
			}
		}
	}

	/*Variables used to transfer the value of the field at certain places between the different process.*/
	double**Ex_left;
	double**Ex_bottom;
	Ex_left =(double**)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Ex_left[i] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank];k++){
			Ex_left[i][k] = 0;
		}
	}
	Ex_bottom =(double**)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Ex_bottom[i] = (double*)malloc((point_per_proc_y[myrank])*sizeof(double));
		for(j=0;j<point_per_proc_y[myrank];j++){
			Ex_bottom[i][j] = 0;
		}
	}
	

	double*Ex_left_send;
	double*Ex_bottom_send;;
	Ex_left_send =(double*)malloc((point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank]*sizeof(double*));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Ex_left_send[i] = 0;
	}
	Ex_bottom_send =(double*)malloc((point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank]*sizeof(double*));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Ex_bottom_send[i] = 0;
	}

	double**Ey_bottom;
	double**Ey_back;
	Ey_bottom =(double**)malloc((point_per_proc_x[myrank])*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ey_bottom[i] = (double*)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			Ey_bottom[i][j] = 0;
		}
	}
	Ey_back =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double**));
	for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
		Ey_back[j] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank];k++){
			Ey_back[j][k] = 0;
		}
	}



	double*Ey_bottom_send;
	double*Ey_back_send;
	Ey_bottom_send =(double*)malloc((point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty)*sizeof(double*));
	for(i=0;i<(point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty);i++){
		Ey_bottom_send[i] = 0;
	}
	Ey_back_send =(double*)malloc((point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank]*sizeof(double*));
	for(j=0;j<(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank];j++){
		Ey_back_send[j] = 0;
	}
	
	double**Ez_left;
	double**Ez_back;
	Ez_left =(double**)malloc((point_per_proc_x[myrank])*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Ez_left[i] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
			Ez_left[i][k] = 0;
		}
	}
	Ez_back =(double**)malloc((point_per_proc_y[myrank])*sizeof(double**));
	for(j=0;j<point_per_proc_y[myrank];j++){
		Ez_back[j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
			Ez_back[j][k] = 0;
		}
	}


	double*Ez_left_send;
	double*Ez_back_send;
	Ez_left_send =(double*)malloc((point_per_proc_x[myrank])*(point_per_proc_z[myrank]+lastz)*sizeof(double*));
	for(i=0;i<(point_per_proc_x[myrank])*(point_per_proc_z[myrank]+lastz);i++){
		Ez_left_send[i] = 0;
	}
	Ez_back_send =(double*)malloc((point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz)*sizeof(double*));
	for(j=0;j<point_per_proc_y[myrank];j++){
		Ez_back_send[j] = 0;
	}

	double**Hx_right;
	double**Hx_up;
	Hx_right =(double**)malloc((point_per_proc_x[myrank])*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Hx_right[i] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
			Hx_right[i][k] = 0;
		}
	}
	Hx_up =(double**)malloc((point_per_proc_x[myrank])*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank];i++){
		Hx_up[i] = (double*)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			Hx_up[i][j] = 0;
		}
	}

	double*Hx_right_send;
	double*Hx_up_send;
	Hx_right_send =(double*)malloc((point_per_proc_x[myrank])*(point_per_proc_z[myrank]+lastz)*sizeof(double*));
	for(i=0;i<point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz);i++){
		Hx_right_send[i] = 0;
	}
	Hx_up_send =(double*)malloc((point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty)*sizeof(double*));
	for(i=0;i<point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty);i++){
		Hx_up_send[i] = 0;

	}	

	double**Hy_up;
	double**Hy_front;
	Hy_up =(double**)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Hy_up[i] = (double*)malloc((point_per_proc_y[myrank])*sizeof(double));
		for(j=0;j<point_per_proc_y[myrank];j++){
			Hy_up[i][j] = 0;
		}
	}
	Hy_front =(double**)malloc((point_per_proc_y[myrank])*sizeof(double**));
	for(j=0;j<point_per_proc_y[myrank];j++){
		Hy_front[j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
			Hy_front[j][k] = 0;
		}
	}


	double*Hy_up_send;
	double*Hy_front_send;
	Hy_up_send =(double*)malloc((point_per_proc_x[myrank]+lastx)*(point_per_proc_y[myrank])*sizeof(double*));
	for(i=0;i<(point_per_proc_x[myrank]+lastx)*(point_per_proc_y[myrank]);i++){
		Hy_up_send[i] = 0;
	}
	Hy_front_send =(double*)malloc((point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz)*sizeof(double*));
	for(j=0;j<point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz);j++){
		Hy_front_send[j] = 0;
	}

	double**Hz_right;
	double**Hz_front;
	Hz_right =(double**)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		Hz_right[i] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank];k++){
			Hz_right[i][k] = 0;
		}
	}
	Hz_front =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double**));
	for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
		Hz_front[j] = (double*)malloc((point_per_proc_z[myrank])*sizeof(double));
		for(k=0;k<point_per_proc_z[myrank];k++){
			Hz_front[j][k] = 0;
		}
	}

	double*Hz_right_send;
	double*Hz_front_send;
	Hz_right_send =(double*)malloc((point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank]*sizeof(double*));
	for(i=0;i<(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank];i++){
		Hz_right_send[i] = 0;
	}
	Hz_front_send =(double*)malloc((point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank])*sizeof(double*));
	for(j=0;j<(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]);j++){
		Hz_front_send[j] = 0;
	}	
 

/********************************************************************************
			Begining of the algorithm.
********************************************************************************/

	while(step<=step_max){
		if(step!=1){
		   /* Certain processes needs an information on Hx, Hy and Hz at places 
		   attributed to another process to update the electric field.*/
			if(divx!=1){ // Communication in the x direction

				if(ip==0){ // I receive only
					MPI_Recv(Hy_front_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Hz_front_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);    
					Update_send_in_mat(point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front,Hy_front_send,Hz_front,Hz_front_send);                                                                       
				}


				else{
					if(ip%2==1){ // I send then I receive
					 	Update_prev_in_send(1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front_send,Hy_prev,Hz_front_send,Hz_prev, 1, point_per_proc_x[myrank]);
						MPI_Send(Hy_front_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-(divy*divz),myrank,MPI_COMM_WORLD);
						MPI_Send(Hz_front_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-(divy*divz),myrank,MPI_COMM_WORLD);						

						if(lastx!=1){
							MPI_Recv(Hy_front_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hz_front_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);
							Update_send_in_mat(point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front,Hy_front_send,Hz_front,Hz_front_send);
						}				
					}
					else{ // I receive then I send
						if(lastx!=1){
							MPI_Recv(Hy_front_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hz_front_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+(divy*divz),myrank+(divy*divz),MPI_COMM_WORLD, &mystatus);							
							Update_send_in_mat(point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front,Hy_front_send,Hz_front,Hz_front_send);
						}
						Update_prev_in_send(1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front_send,Hy_prev,Hz_front_send,Hz_prev, 1, point_per_proc_x[myrank]);
						MPI_Send(Hy_front_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-(divy*divz),myrank,MPI_COMM_WORLD);
						MPI_Send(Hz_front_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-(divy*divz),myrank,MPI_COMM_WORLD);

					}
				}
			}
			if (divy != 1){// Communication in the y direction

				if (jp == 0){ //I receive only
					MPI_Recv(Hx_right_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Hz_right_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);      
					Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_x[myrank]+lastx,point_per_proc_z[myrank],Hx_right,Hx_right_send,Hz_right,Hz_right_send);                                                                       
				}
				else{
					if(jp%2==1){//I send then I receive	
						Update_prev_in_send(point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],Hx_right_send,Hx_prev,Hz_right_send,Hz_prev, 2, point_per_proc_y[myrank]);					
						MPI_Send(Hx_right_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD);
						MPI_Send(Hz_right_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD);
						if (lasty!=1){
							MPI_Recv(Hx_right_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hz_right_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);
							Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_x[myrank]+lastx,point_per_proc_z[myrank],Hx_right,Hx_right_send,Hz_right,Hz_right_send);	
						}
					}
					else{//I receive then I send
						if (lasty!=1){
							MPI_Recv(Hx_right_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hz_right_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD, &mystatus);
							Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_z[myrank]+lastz, point_per_proc_x[myrank]+lastx,point_per_proc_z[myrank],Hx_right,Hx_right_send,Hz_right,Hz_right_send);	
						}
						Update_prev_in_send(point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],Hx_right_send,Hx_prev,Hz_right_send,Hz_prev, 2, point_per_proc_y[myrank]);
						MPI_Send(Hx_right_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD);
						MPI_Send(Hz_right_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD);
					}
				}
			}
			if (divz!=1){// Communication in the z direction

				if (kp==0){//I receive only
					MPI_Recv(Hx_up_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Hy_up_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);  
					Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty, point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],Hx_up,Hx_up_send,Hy_up,Hy_up_send);                                                              
				}
				else{
					if(kp%2==1){//I send then I receive
						Update_prev_in_send(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,Hx_up_send,Hx_prev,Hy_up_send,Hy_prev, 3, point_per_proc_z[myrank]);
						MPI_Send(Hx_up_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank-divy,myrank,MPI_COMM_WORLD);
						MPI_Send(Hy_up_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank-divy,myrank,MPI_COMM_WORLD);
						if(lastz!=1){
							MPI_Recv(Hx_up_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hy_up_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);
							Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty, point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],Hx_up,Hx_up_send,Hy_up,Hy_up_send);
						}
					}
					else{//I receive then I send
						if(lastz!=1){
							MPI_Recv(Hx_up_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);
							MPI_Recv(Hy_up_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank+divy,MPI_COMM_WORLD, &mystatus);
							Update_send_in_mat(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty, point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],Hx_up,Hx_up_send,Hy_up,Hy_up_send);
						}
						Update_prev_in_send(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,Hx_up_send,Hx_prev,Hy_up_send,Hy_prev, 3, point_per_proc_z[myrank]);
						MPI_Send(Hx_up_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank-divy,myrank,MPI_COMM_WORLD);
						MPI_Send(Hy_up_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank-divy,myrank,MPI_COMM_WORLD);
					}
				}
			}
		}
   
		//Update of the electric field.
		//	X component		
		if(lasty==1 && lastz==1){
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_r,1);
		}
		else if(lasty==1){
    			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_r,1);
     			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_r,myrank,5);
		}
		else if (lastz == 1){
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_r,1);
     			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_r,myrank,3);
		}
		else{
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_r,1);
     			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_r,myrank,5);
     			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_r,myrank,3);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
				k = point_per_proc_z[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ex_new[i][j][k] = Ex_prev[i][j][k] +(dt/(e_0*e_r[i][j][k]*dx))*((Hz_right[i][k]-Hz_prev[i][j][k])-(Hy_up[i][j]-Hy_prev[i][j][k]));
			}
		}

		//	Y component
		if(lastx==1 && lastz==1){
     			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_r,2);			
		}
		else if(lastx==1){
     			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_r,2);
     			 Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_r,myrank,6);
		}
		else if(lastz==1){
     			 Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_r,2);
     			 Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_r,myrank,1);
		}
		else{
     			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_r,2);
     			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_r,myrank,6);
     			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_r,myrank,1);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
				i = point_per_proc_x[myrank]-1;
				k = point_per_proc_z[myrank]-1;
				Ey_new[i][j][k] = Ey_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((Hx_up[i][j]-Hx_prev[i][j][k])-(Hz_front[j][k]-Hz_prev[i][j][k]));
			}
		}
		//	Z component
		if(lastx==1 && lasty==1){
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_r,3);			
		}
		else if(lastx==1){
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_r,3);
      			Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_r,myrank,4);	
		}
		else if(lasty==1){
      			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_r,3);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_r,myrank,2);
		}
		else{
    			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_r,3);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_r,myrank,4);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_r,myrank,2);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				i = point_per_proc_x[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ez_new[i][j][k] = Ez_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((Hy_front[j][k]-Hy_prev[i][j][k])-(Hx_right[i][k]-Hx_prev[i][j][k]));
			}
		}

		// Boundary condition
   
   		Boundary_condition_imosition(ip,jp,kp,lastx,lasty,lastz,point_per_proc_x,point_per_proc_y,point_per_proc_z,Ex_new,Ey_new,Ez_new,myrank);		

		// Imposition of the value of the electric field for the node corresponding to the antenna.
		if(i_min_proc[myrank]<= i_max_a && j_min_proc[myrank]<= j_max_a && k_min_proc[myrank]<= k_max_a && i_max_proc[myrank]>= i_min_a && j_max_proc[myrank]>= j_min_a && k_max_proc[myrank]>= k_min_a){
			b_inf_x = i_min_a - i_min_proc[myrank];				
			b_inf_y = j_min_a - j_min_proc[myrank];
			b_inf_z = k_min_a - k_min_proc[myrank];
			b_sup_x = i_max_a - i_min_proc[myrank];
			b_sup_y = j_max_a - j_min_proc[myrank];
			b_sup_z = k_max_a - k_min_proc[myrank];
			if(b_inf_x<0){
				b_inf_x = 0;		
			}
			if(b_inf_y<0){
				b_inf_y = 0;		
			}
			if(b_inf_z<0){
				b_inf_z = 0;		
			}
			if(point_per_proc_x[myrank]-1+lastx<b_sup_x){
				b_sup_x = point_per_proc_x[myrank]-1+lastx;
			}
			if(point_per_proc_y[myrank]-1+lasty<b_sup_y){
				b_sup_y = point_per_proc_y[myrank]-1+lasty;
			}
			if(point_per_proc_z[myrank]-1+lastz<b_sup_z){
				b_sup_z = point_per_proc_z[myrank]-1+lastz;
			}
			#pragma omp parallel for default(shared) private(i,j,k)
			for(i=b_inf_x;i<=b_sup_x;i++){
				for(j=b_inf_y;j<=b_sup_y+lasty;j++){
					for(k=b_inf_z;k<=b_sup_z;k++){
            if(P==1){
						  Ey_new[i][j][k]= sin(omega*step*dt);		
            }
            else if(P==2){
              Ey_new[i][j][k]= 1;
            }			
					}				
				}
			}
		}
		
		//Storage of the new value of the electric field in E_prev.

		New_in_old(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank],Ex_new,Ex_prev);
		New_in_old(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev);
		New_in_old(point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev);		
		
		/* Certain processes needs an information on the updated electric field at places attributed to another process to update the magnetic field.*/
		if (divx!=1){// Communication in the x direction

			if(ip==0){//I only send
				Update_prev_in_send(1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ey_back_send,Ey_prev,Ez_back_send,Ez_prev, 4, point_per_proc_x[myrank]);				
				MPI_Send(Ey_back_send,(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);
				MPI_Send(Ez_back_send,(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);
			}			
			else{
				if(ip%2==1){//I receive then I send
					MPI_Recv(Ey_back_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-(divy*divz),myrank-(divy*divz),MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ez_back_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-(divy*divz),myrank-(divy*divz),MPI_COMM_WORLD, &mystatus);	
					Update_send_in_mat(point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank], point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ey_back,Ey_back_send,Ez_back,Ez_back_send);				
					if(lastx!=1){
						Update_prev_in_send(1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ey_back_send,Ey_prev,Ez_back_send,Ez_prev, 4, point_per_proc_x[myrank]);
						MPI_Send(Ey_back_send,(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);
						MPI_Send(Ez_back_send,(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);					
					}
				}
				else{//I send then I receive
					if(lastx!=1){
						Update_prev_in_send(1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ey_back_send,Ey_prev,Ez_back_send,Ez_prev, 4, point_per_proc_x[myrank]);
						MPI_Send(Ey_back_send,(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);
						MPI_Send(Ez_back_send,(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+(divy*divz),myrank,MPI_COMM_WORLD);						
					}
					MPI_Recv(Ey_back_send,(point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-(divy*divz),myrank-(divy*divz),MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ez_back_send,point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-(divy*divz),myrank-(divy*divz),MPI_COMM_WORLD, &mystatus);
					Update_send_in_mat(point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank], point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ey_back,Ey_back_send,Ez_back,Ez_back_send);
				}
			}
		}

		if(divy!=1){// Communication in the y direction

			if(jp==0){//I only send
				Update_prev_in_send(point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,Ex_left_send,Ex_prev,Ez_left_send,Ez_prev, 5, point_per_proc_y[myrank]);				
				MPI_Send(Ex_left_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
				MPI_Send(Ez_left_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
			}			
			else{
				if(jp%2==1){//I receive then I send
					MPI_Recv(Ex_left_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ez_left_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD, &mystatus);
					Update_send_in_mat(point_per_proc_x[myrank]+lastx,point_per_proc_z[myrank], point_per_proc_x[myrank],point_per_proc_z[myrank]+lastz,Ex_left,Ex_left_send,Ez_left,Ez_left_send);					
					if(lasty!=1){
						Update_prev_in_send(point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,Ex_left_send,Ex_prev,Ez_left_send,Ez_prev, 5, point_per_proc_y[myrank]);
						MPI_Send(Ex_left_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
						MPI_Send(Ez_left_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
					}
				}
				else{//I send then I receive
					if(lasty!=1){
						Update_prev_in_send(point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,Ex_left_send,Ex_prev,Ez_left_send,Ez_prev, 5, point_per_proc_y[myrank]);
						MPI_Send(Ex_left_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
						MPI_Send(Ez_left_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD);
					}

					MPI_Recv(Ex_left_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_z[myrank],MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ez_left_send,point_per_proc_x[myrank]*(point_per_proc_z[myrank]+lastz),MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD, &mystatus);
					Update_send_in_mat(point_per_proc_x[myrank]+lastx,point_per_proc_z[myrank], point_per_proc_x[myrank],point_per_proc_z[myrank]+lastz,Ex_left,Ex_left_send,Ez_left,Ez_left_send);
				}
			}
		}
		if(divz!=1){// Communication in the z direction

			if(kp==0){//I only send
				Update_prev_in_send(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,Ex_bottom_send,Ex_prev,Ey_bottom_send,Ey_prev, 6, point_per_proc_z[myrank]);				
				MPI_Send(Ex_bottom_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);
				MPI_Send(Ey_bottom_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);			
}
			else{
				if(kp%2==1){//I receive then I send
					MPI_Recv(Ex_bottom_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank-divy,myrank-divy,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ey_bottom_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank-divy,myrank-divy,MPI_COMM_WORLD, &mystatus);
					Update_send_in_mat(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank], point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,Ex_bottom,Ex_bottom_send,Ey_bottom,Ey_bottom_send);					
					if(lastz!=1){
						Update_prev_in_send(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,Ex_bottom_send,Ex_prev,Ey_bottom_send,Ey_prev, 6, point_per_proc_z[myrank]);
						MPI_Send(Ex_bottom_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);
						MPI_Send(Ey_bottom_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);
					}
				}
				else{// I send then I receive
					if(lastz!=1){
						Update_prev_in_send(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,Ex_bottom_send,Ex_prev,Ey_bottom_send,Ey_prev, 6, point_per_proc_z[myrank]);
						MPI_Send(Ex_bottom_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);
						MPI_Send(Ey_bottom_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank+divy,myrank,MPI_COMM_WORLD);
					}
					MPI_Recv(Ex_bottom_send,(point_per_proc_x[myrank]+lastx)*point_per_proc_y[myrank],MPI_DOUBLE,myrank-divy,myrank-divy,MPI_COMM_WORLD, &mystatus);
					MPI_Recv(Ey_bottom_send,point_per_proc_x[myrank]*(point_per_proc_y[myrank]+lasty),MPI_DOUBLE,myrank-divy,myrank-divy,MPI_COMM_WORLD, &mystatus);
					Update_send_in_mat(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank], point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,Ex_bottom,Ex_bottom_send,Ey_bottom,Ey_bottom_send);
				}
			}
		}			

		//Update of the magnetic field
		//	X Component
		Update_H_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]+lastz, lastz, lasty,Hx_new,Hx_prev,Ey_prev,Ez_prev,dt,dx,mu_0,mu_r,1);	
		Update_H_boundary(point_per_proc_x[myrank],0,point_per_proc_z[myrank]+lastz,lastz,Hx_new,Hx_prev,Ey_prev,Ez_prev,Ez_left,dt,dx,mu_0,mu_r,myrank,3);
		Update_H_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,0,lasty,Hx_new,Hx_prev,Ey_prev,Ez_prev,Ey_bottom,dt,dx,mu_0,mu_r,myrank,5);		
		#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)
		for(i=0;i<point_per_proc_x[myrank];i++){
			j = 0;
			k = 0;
			temp1 = Ey_prev[i][j][k];			
			temp2 = Ez_prev[i][j][k];
			Hx_new[i][j][k] = Hx_prev[i][j][k] + (dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-Ey_bottom[i][j])-(temp2-Ez_left[i][k]));
		}
		
		//	Y Component
		Update_H_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz, lastx, lastz,Hy_new,Hy_prev,Ez_prev,Ex_prev,dt,dx,mu_0,mu_r,2);	
		Update_H_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],0,lastx,Hy_new,Hy_prev,Ez_prev,Ex_prev,Ex_bottom,dt,dx,mu_0,mu_r,myrank,6);		
		Update_H_boundary(0,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,lastz,Hy_new,Hy_prev,Ez_prev,Ex_prev,Ez_back,dt,dx,mu_0,mu_r,myrank,1);		
		#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)
		for(j=0;j<point_per_proc_y[myrank];j++){
			i = 0;
			k = 0;
			temp1 = Ez_prev[i][j][k];
			temp2 = Ex_prev[i][j][k];
			Hy_new[i][j][k] = Hy_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-Ez_back[j][k])-(temp2-Ex_bottom[i][j]));	
		}
			
		//	Z Component
		Update_H_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank], lasty, lastx,Hz_new,Hz_prev,Ex_prev,Ey_prev,dt,dx,mu_0,mu_r,3);	
		Update_H_boundary(point_per_proc_x[myrank]+lastx,0,point_per_proc_z[myrank],lastx,Hz_new,Hz_prev,Ex_prev,Ey_prev,Ex_left,dt,dx,mu_0,mu_r,myrank,4);	
		Update_H_boundary(0,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],lasty,Hz_new,Hz_prev,Ex_prev,Ey_prev,Ey_back,dt,dx,mu_0,mu_r,myrank,2);		
		#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)
		for(k=0;k<point_per_proc_z[myrank];k++){
			i = 0;
			j = 0;
			temp1 = Ex_prev[i][j][k];			
			temp2 = Ey_prev[i][j][k];
			Hz_new[i][j][k] = Hz_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1 - Ex_left[i][k])-(temp2 - Ey_back[j][k]));
		}		

		// Storage of the updated value of the magnetic field in H_prev

		New_in_old(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]+lastz,Hx_new,Hx_prev);
		New_in_old(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Hy_new,Hy_prev);		
		New_in_old(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hz_new,Hz_prev);
				
		// Storage of the matrices in vectors
		
		// E_X
		int npz1 = mygrid_Ex.np1[2];
		int npz2 = mygrid_Ex.np2[2];
		Vec3i np = mygrid_Ex.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Ex.np1[1];
			int npy2 = mygrid_Ex.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Ex.np1[0];
                		int npx2 = mygrid_Ex.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Ex_vec[idx] = Ex_new[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}

		// E_Y
		npz1 = mygrid_Ey.np1[2];
		npz2 = mygrid_Ey.np2[2];
		np = mygrid_Ey.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Ey.np1[1];
			int npy2 = mygrid_Ey.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Ey.np1[0];
                		int npx2 = mygrid_Ey.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Ey_vec[idx] = Ey_new[i-npx1][j-npy1][k-npz1];
					                er_vec[idx] = e_r[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}
		
		// E_Z
		npz1 = mygrid_Ez.np1[2];
		npz2 = mygrid_Ez.np2[2];
		np = mygrid_Ez.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Ez.np1[1];
			int npy2 = mygrid_Ez.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Ez.np1[0];
                		int npx2 = mygrid_Ez.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Ez_vec[idx] = Ez_new[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}

		// H_X
		npz1 = mygrid_Hx.np1[2];
		npz2 = mygrid_Hx.np2[2];
		np = mygrid_Hx.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Hx.np1[1];
			int npy2 = mygrid_Hx.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Hx.np1[0];
                		int npx2 = mygrid_Hx.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Hx_vec[idx] = Hx_new[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}

		// H_Y
		npz1 = mygrid_Hy.np1[2];
		npz2 = mygrid_Hy.np2[2];
		np = mygrid_Hy.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Hy.np1[1];
			int npy2 = mygrid_Hy.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Hy.np1[0];
                		int npx2 = mygrid_Hy.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Hy_vec[idx] = Hy_new[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}

		
		// H_Z
		npz1 = mygrid_Hz.np1[2];
		npz2 = mygrid_Hz.np2[2];
		np = mygrid_Hz.np();
		#pragma omp parallel for default(shared) private(i,j,k)
		for (int k = npz1; k <= npz2; ++k){
			int npy1 = mygrid_Hz.np1[1];
			int npy2 = mygrid_Hz.np2[1];            
			for (int j = npy1; j <= npy2; ++j){
               			int npx1 = mygrid_Hz.np1[0];
                		int npx2 = mygrid_Hz.np2[0];                 
                		for (int i = npx1; i <= npx2; ++i){
                    			int idx = (k-npz1) * (np[1] * np[0]) + (j-npy1) * np[0] + (i-npx1);
                    			Hz_vec[idx] = Hz_new[i-npx1][j-npy1][k-npz1];
                		}
            		}
        	}


		/*Storage of the Results*/

		if(step%SR==0){//save results of the mpi process to disk
			//export_spoints_XML("Ex", step, grid_Ex, mygrid_Ex, ZIPPED, Nx, Ny, Nz, 0);
		  	export_spoints_XML("Ey", step, grid_Ey, mygrid_Ey, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Ez", step, grid_Ez, mygrid_Ez, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hx", step, grid_Hx, mygrid_Hx, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hy", step, grid_Hy, mygrid_Hy, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hz", step, grid_Hz, mygrid_Hz, ZIPPED, Nx, Ny, Nz, 0);

            		if (myrank == 0){	// save main pvti file by rank0
				//export_spoints_XMLP("Ex", step, grid_Ex, mygrid_Ex, sgrids_Ex, ZIPPED);
                		export_spoints_XMLP("Ey", step, grid_Ey, mygrid_Ey, sgrids_Ey, ZIPPED);
				//export_spoints_XMLP("Ez", step, grid_Ez, mygrid_Ez, sgrids_Ez, ZIPPED);
				//export_spoints_XMLP("Hx", step, grid_Hx, mygrid_Hx, sgrids_Hx, ZIPPED);
				//export_spoints_XMLP("Hy", step, grid_Hy, mygrid_Hy, sgrids_Hy, ZIPPED);
				//export_spoints_XMLP("Hz", step, grid_Hz, mygrid_Hz, sgrids_Hz, ZIPPED);
            		}
        	}
		step++;
   
   // Make the geometry rotate, to be suprressed once the coupling is done
   /*
		theta = step*(3.141692/4);
		rotate_geometry(geometry_init,geometry_init, Nx, Ny, Nz, Lx, Ly, Lz, dx,n_sphere, prop_sphere, n_cylinder,prop_cylinder, n_cube,prop_cube, theta);

		init_geometry(geometry_init,vec_e_r,vec_mu_r,e_r_tot, mu_r_tot,Nx, Ny, Nz);
		init_geom_one_proc(e_r,mu_r,e_r_tot,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz, Nx, Ny, Nz);*/
	}
/********************************************************************************
			Liberation of the memory.
********************************************************************************/

	//Electric field
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){		
		for(j=0;j<point_per_proc_y[myrank]-1;j++){		
			free(Ex_new[i][j]);
			free(Ex_prev[i][j]);		
		}
		free(Ex_new[i]);
		free(Ex_prev[i]);
	}
	free(Ex_new);
	free(Ex_prev);

	for(i=0;i<point_per_proc_x[myrank];i++){		
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){		
			free(Ey_new[i][j]);
			free(Ey_prev[i][j]);		
		}
		free(Ey_new[i]);
		free(Ey_prev[i]);
	}
	free(Ey_new);
	free(Ey_prev);

	for(i=0;i<point_per_proc_x[myrank];i++){		
		for(j=0;j<point_per_proc_y[myrank];j++){		
			free(Ez_new[i][j]);
			free(Ez_prev[i][j]);		
		}
		free(Ez_new[i]);
		free(Ez_prev[i]);
	}
	free(Ez_new);
	free(Ez_prev);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		free(Ex_left[i]);
	}
	free(Ex_left);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
			free(Ex_bottom[i]);
	}
	free(Ex_bottom);

	free(Ex_left_send);
	free(Ex_bottom_send);

	for(i=0;i<point_per_proc_x[myrank];i++){
		free(Ey_bottom[i]);
	}
	free(Ey_bottom);

	for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
		free(Ey_back[j]);
	}
	free(Ey_back);

	free(Ey_bottom_send);
	free(Ey_back_send);

	for(i=0;i<point_per_proc_x[myrank];i++){
		free(Ez_left[i]);
	}
	free(Ez_left);

	for(j=0;j<point_per_proc_y[myrank];j++){
		free(Ez_back[j]);
	}
	free(Ez_back);

	free(Ez_left_send);
	free(Ez_back_send);


	//Magnetic field
	for(i=0;i<point_per_proc_x[myrank];i++){		
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){		
			free(Hx_new[i][j]);
			free(Hx_prev[i][j]);		
		}
		free(Hx_new[i]);
		free(Hx_prev[i]);
	}
	free(Hx_new);
	free(Hx_prev);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){		
		for(j=0;j<point_per_proc_y[myrank];j++){		
			free(Hy_new[i][j]);
			free(Hy_prev[i][j]);		
		}
		free(Hy_new[i]);
		free(Hy_prev[i]);
	}
	free(Hy_new);
	free(Hy_prev);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){		
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){		
			free(Hz_new[i][j]);
			free(Hz_prev[i][j]);		
		}
		free(Hz_new[i]);
		free(Hz_prev[i]);
	}
	free(Hz_new);
	free(Hz_prev);

	for(i=0;i<point_per_proc_x[myrank];i++){
		free(Hx_right[i]);
	}
	free(Hx_right);
	
	for(i=0;i<point_per_proc_x[myrank];i++){
		free(Hx_up[i]);
	}
	free(Hx_up);

	free(Hx_right_send);
	free(Hx_up_send);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		free(Hy_up[i]);
	}
	free(Hy_up);

	for(j=0;j<point_per_proc_y[myrank];j++){
		free(Hy_front[j]);
	}
	free(Hy_front);

	free(Hy_up_send);
	free(Hy_front_send);

	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		free(Hz_right[i]);
	}
	free(Hz_right);

	for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
		free(Hz_front[j]);
	}
	free(Hz_front);

	free(Hz_right_send);
	free(Hz_front_send);


	// Physical variables
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
			free(e_r[i][j]);			
			free(mu_r[i][j]);
		}
		free(e_r[i]);
		free(mu_r[i]);
	}
	free(e_r);
	free(mu_r);

	//Point per proc and global indices
	free(i_min_proc);
	free(i_max_proc);
	free(j_min_proc);
	free(j_max_proc);
	free(k_min_proc);
	free(k_max_proc);
	free(point_per_proc_x);
	free(point_per_proc_y);
	free(point_per_proc_z);

	MPI_Finalize();
}



// This function split the spatial domain over the different processes
void divisor(int n_p,int*r){
	int prime_nb[10]={2,3,5,7,11,13,17,19,23,29};
	int divx = 1;
	int divy = 1;
	int divz = 1;
	int k =9;	

	while(n_p!=1){
		if(n_p % prime_nb[k]==0){			
			if(divx<=divy && divx<=divz){
				divx = divx*prime_nb[k];
			}
			else{
				if(divy<=divx && divy<=divz){
					divy = divy*prime_nb[k];
				}
				else{
					divz = divz*prime_nb[k];			
				}
			}
			n_p = n_p/prime_nb[k];
		}
		else{
			if(k==0){
				if(divx<=divy && divx<=divz){
					divx = divx*prime_nb[k];
				}
				else{
					if(divy<=divx && divy<=divz){
						divy = divy*prime_nb[k];
					}
					else{
						divz = divz*prime_nb[k];			
					}
				}
				break;
			}
			k--;
		}
	}
	r[0] = divx;
	r[1] = divy;
	r[2] = divz;
}

// This function compute the number of point per process as well as the minimum and maximum indices of the processes in the global axes
void compute_proc_ind_point(int nbproc,int Nx,int Ny,int Nz,int divx,int divy,int divz,int*i_min_proc,int*j_min_proc,int*k_min_proc,int*i_max_proc,int*j_max_proc,int*k_max_proc,int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z){
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;
	for(l=0;l<nbproc;l++){
		point_per_proc_x[l] = Nx/divx;
		point_per_proc_y[l] = Ny/divy;
		point_per_proc_z[l] = Nz/divz;		
		if(((l/(divy*divz))+1)<=(Nx % divx)){
			point_per_proc_x[l]++;
		}
		if(((l%(divy*divz))%divy+1)<=(Ny % divy)){
			point_per_proc_y[l]++;			
		}
		if(((l%(divy*divz))/divy+1)<=(Nz % divz)){
			point_per_proc_z[l]++;			
		}
	}
	for(l=0;l<nbproc;l++){
		i_min_proc[l] = 0;
		j_min_proc[l] = 0;
		k_min_proc[l] = 0;
		for(i=0;i<l/(divy*divz);i++){
			i_min_proc[l] += point_per_proc_x[i*divy*divz];
		}
		for(j=0;j<(l%(divy*divz))%divy;j++){
			j_min_proc[l] += point_per_proc_y[j];
		}
		for(k=0;k<(l%(divy*divz))/divy;k++){
			k_min_proc[l] += point_per_proc_z[divy*k];
		}
		i_max_proc[l] = i_min_proc[l] + point_per_proc_x[l]-1;
		j_max_proc[l] = j_min_proc[l] + point_per_proc_y[l]-1;
		k_max_proc[l] = k_min_proc[l] + point_per_proc_z[l]-1;
	}	
}

// This function store the vector receive by MPI in a matrix
void Update_send_in_mat(int i_max1,int j_max1, int i_max2,int j_max2,double**M1,double*V1,double**M2,double*V2){
	int i =0;
 	int j =0;
    #pragma omp parallel for default(shared) private(i,j)		
	for(i=0;i<i_max1;i++){
		for(j=0;j<j_max1;j++){
 			M1[i][j] = V1[i*(j_max1) + j];			
	  	}	
	}
    #pragma omp parallel for default(shared) private(i,j)					
	for(i=0;i<i_max2;i++){
		for(j=0;j<j_max2;j++){
	  		M2[i][j] = V2[i*(j_max2)+ j];
		  }
	}
}

// This function place the content of a matrix in a vector in order to send it with MPI
void Update_prev_in_send(int i_max1,int j_max1,int k_max1,int i_max2,int j_max2,int k_max2,double*V1,double***M1,double*V2,double***M2, int Case, int point_max){
	int i =0;
 	int j =0;
	int k =0;
    #pragma omp parallel for default(shared) private(i,j,k)
		for(i=0;i<i_max1;i++){		
			for(j=0;j<j_max1;j++){
				for(k=0;k<k_max1;k++){
					if(Case==1){
						V1[j*(k_max1)+k] = M1[0][j][k];
					}
					else if(Case==2){
						V1[i*(k_max1)+k] = M1[i][0][k] ;
					}
					else if(Case==3){
						V1[i*(j_max1)+j]=M1[i][j][0];
					}
					else if(Case==4){
						V1[j*(k_max1)+k] = M1[point_max-1][j][k];
					}
					else if(Case==5){
						V1[i*(k_max1)+k] = M1[i][point_max-1][k];
					}
					else if(Case==6){
						V1[i*(j_max1)+ j] = M1[i][j][point_max-1];
					}
				}	
			}
		}
		#pragma omp parallel for default(shared) private(i,j,k)
		for(i=0;i<i_max2;i++){		
			for(j=0;j<j_max2;j++){
				for(k=0;k<k_max2;k++){
					if(Case==1){
						V2[j*(k_max2)+k] = M2[0][j][k];
					}
					else if(Case==2){
						V2[i*(k_max2)+k] = M2[i][0][k] ;
					}
					else if(Case==3){
						V2[i*(j_max2)+j]=M2[i][j][0];
					}
					else if(Case==4){
						V2[j*(k_max2)+k] = M2[point_max-1][j][k];
					}
					else if(Case==5){
						V2[i*(k_max2)+k] = M2[i][point_max-1][k];
					}
					else if(Case==6){
						V2[i*(j_max2)+ j] = M2[i][j][point_max-1];
					}
				}	
			}
		}
}

// This function update the electric field inside the domain attributed to a given process
void Update_E_inside(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double dt,double dx,double e_0,double***e_r,int Case){
  int i =0;
  int j =0;
  int k =0;  
    #pragma omp parallel for default(shared) private(i,j,k) 
  for(i=0;i<i_max;i++){
	for(j=0;j<j_max;j++){
		for(k=0;k<k_max;k++){	
			if(Case==1){				
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i][j+1][k]-H1_prev[i][j][k])-(H2_prev[i][j][k+1]-H2_prev[i][j][k]));
			}
			else if(Case==2){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i][j][k+1]-H1_prev[i][j][k])-(H2_prev[i+1][j][k]-H2_prev[i][j][k]));
			}
			else if(Case==3){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i+1][j][k]-H1_prev[i][j][k])-(H2_prev[i][j+1][k]-H2_prev[i][j][k]));
			}	
		}
	}
  }
}

// This function update the electric field at the boundary of the domain attributed to a given process
void Update_E_boundary(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double**H_boundary,double dt,double dx,double e_0,double***e_r,int myrank,int Case){
  int i =0;
  int j =0;
  int k =0;
  int i_min=0;
  int j_min=0;
  int k_min=0;
  if(Case==1 || Case==2){
  	i_min=i_max;
  	j_max-=1;
	k_max-=1;
  }
  else if(Case==3 || Case==4){
        j_min=j_max;
  	i_max-=1;
	k_max-=1;
  }
  else if(Case==5 || Case==6){
        k_min=k_max;
  	i_max-=1;
	j_max-=1;
  }
  #pragma omp parallel for default(shared) private(i,j,k)
  for(i=i_min;i<=i_max;i++){
	  for(j=j_min;j<=j_max;j++){
		for(k=k_min;k<=k_max;k++){
			if(Case==1){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i][j][k+1]-H1_prev[i][j][k])-(H_boundary[j][k]-H2_prev[i][j][k]));
			}
			else if(Case==2){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H_boundary[j][k]-H1_prev[i][j][k])-(H2_prev[i][j+1][k]-H2_prev[i][j][k]));
			}
			else if(Case==3){
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i][j][k]*dx))*((H_boundary[i][k]-H1_prev[i][j][k])-(H2_prev[i][j][k+1]-H2_prev[i][j][k]));
			}
			else if(Case==4){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i+1][j][k]-H1_prev[i][j][k])-(H_boundary[i][k]-H2_prev[i][j][k]));
			}
			else if(Case==5){
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i][j][k]*dx))*((H1_prev[i][j+1][k]-H1_prev[i][j][k])-(H_boundary[i][j]-H2_prev[i][j][k]));
			}
			else if(Case==6){
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i][j][k]*dx))*((H_boundary[i][j]-H1_prev[i][j][k])-(H2_prev[i+1][j][k]-H2_prev[i][j][k]));
			}
		}
	}
  }
}

// This function update the magnetic field inside the domain attributed to a given process.
void Update_H_inside(int i_max,int j_max,int k_max,int last1, int last2,double***H_new,double***H_prev,double***E1_prev,double***E2_prev,double dt,double dx,double mu_0,double***mu_r,int Case){
	int i = 0;
 	int j = 0;
  	int k = 0;
	double temp1=0;
	double temp2=0;
	int i_min=1;
 	int j_min=1;
  	int k_min=1;
	if(Case==1){
		i_min=0;
 	}
  	else if(Case==2){
       		j_min=0;
  	}
 	 else if(Case==3){
      	 	 k_min=0;
  	}
	#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)	
		for(i=i_min;i<i_max;i++){
			for(j=j_min;j<j_max;j++){
				for(k=i_min;k<k_max;k++){
					if( last1==1&&((Case==1 && k == k_max-last1)||(Case==2 && i == i_max-last1)||(Case==3&&j == j_max-last1))){
						temp1=0;
					}
					else{
						temp1 = E1_prev[i][j][k];
					}
					if(last2==1&&((Case==1&&j == j_max-last2)||(Case==2&&k == k_max-last2)||(Case==3&&i == i_max-last2))){
						temp2 = 0;
					}
					else{
						temp2 = E2_prev[i][j][k];
					}
					if(Case==1){				
						H_new[i][j][k] = H_prev[i][j][k] + (dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E1_prev[i][j][k-1])-(temp2-E2_prev[i][j-1][k]));	
					}
					else if(Case==2){
						H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E1_prev[i-1][j][k])-(temp2-E2_prev[i][j][k-1]));
					}
					else if(Case==3){
						H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1 - E1_prev[i][j-1][k])-(temp2 - E2_prev[i-1][j][k]));
					}
				}
			}
		}
}

// This function update the magnetic field at the boundary of the domain attributed to a given process
void Update_H_boundary(int i_max,int j_max,int k_max,int last,double***H_new,double***H_prev,double***E1_prev,double***E2_prev,double**E_boundary,double dt,double dx,double mu_0,double***mu_r,int myrank,int Case){
	int i = 0;
 	int j = 0;
  	int k = 0;
	double temp1=0;
	double temp2=0;	
	int i_min = 0;
	int j_min = 0;
	int k_min = 0;	
	if(Case==1){
		i_max = 0;
		k_min = 1;
		j_max--;
		k_max--;
	}
	if(Case==2){
		i_max = 0;
		j_min = 1;
		j_max--;
		k_max--;
	}
	else if(Case==3){
		j_max = 0;
		k_min = 1;
		i_max--;
		k_max--;
	}
	else if(Case==4){
		j_max = 0;
		i_min = 1;
		i_max--;
		k_max--;
	}
	else if(Case==5){
		k_max = 0;
		j_min = 1;
		i_max--;
		j_max--;
	}
	else if(Case==6){
		k_max = 0;
		i_min = 1;
		i_max--;
		j_max--;
	}
 
	#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)
	for(i=i_min;i<=i_max;i++){
		for(j=j_min;j<=j_max;j++){
			for(k=k_min;k<=k_max;k++){
				if(last==1&&((Case==1&&k == k_max-last+1)||(Case==4&&i == i_max-last+1)||(Case==5&&j == j_max-last+1))){
					temp2 = 0;
				}
				else{
					temp2 = E2_prev[i][j][k];				
				}
				if(last==1&&((Case==2 && j == j_max-last+1)||(Case==3 && k == k_max-last+1)||(Case==6 && i == i_max-last+1))){
					temp1 =0;
				}
				else{
					temp1 = E1_prev[i][j][k];
				}
				if(Case==1){
					H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E_boundary[j][k])-(temp2-E2_prev[i][j][k-1]));
				}
				else if(Case==2){
					H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1 - E1_prev[i][j-1][k])-(temp2 - E_boundary[j][k]));
				}
				else if(Case==3){
					H_new[i][j][k] = H_prev[i][j][k] + (dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E1_prev[i][j][k-1])-(temp2-E_boundary[i][k]));
				}
				else if(Case==4){
					H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1 - E_boundary[i][k])-(temp2 - E2_prev[i-1][j][k]));
				}
				else if(Case==5){
					H_new[i][j][k] = H_prev[i][j][k] + (dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E_boundary[i][j])-(temp2-E2_prev[i][j-1][k]));
				}
				else if(Case==6){
					H_new[i][j][k] = H_prev[i][j][k]+(dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-E1_prev[i-1][j][k])-(temp2-E_boundary[i][j]));
				}
			}
	  }
	}
}

// This function imposes the boundary condition of the electromagnetic problem
void Boundary_condition_imosition(int ip,int jp,int kp,int lastx,int lasty,int lastz,int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z,double***Ex_new,double***Ey_new,double***Ez_new,int myrank){
  if(ip==0){
   Hom_BC(0, 0, 0, 0, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Ey_new);
   Hom_BC(0, 0, 0, 0, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Ez_new);
  }
  if(lastx==1){
    Hom_BC(point_per_proc_x[myrank]-1, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Ey_new);
    Hom_BC(point_per_proc_x[myrank]-1, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Ez_new);
  }
  if(jp==0){
    Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, 0, point_per_proc_z[myrank]-1,Ex_new);
    Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, 0, point_per_proc_z[myrank]+lastz-1,Ez_new);			
  }
  if(lasty==1){
    Hom_BC(0, point_per_proc_y[myrank]-1, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]-1,Ex_new);
    Hom_BC(0, point_per_proc_y[myrank]-1, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Ez_new);
  }
  if(kp==0){
    Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, 0,Ex_new);
    Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, 0,Ey_new);
  }
  if(lastz==1){
    Hom_BC(0, 0, point_per_proc_z[myrank]-1, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]-1,Ex_new);
    Hom_BC(0, 0, point_per_proc_z[myrank]-1, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Ey_new);
  }
}

// This function place the content of a matrix in an other matrix
void New_in_old(int i_max,int j_max,int k_max,double***New,double***Old){
	int i = 0;
  	int j = 0;
  	int k = 0;
	#pragma omp parallel for default(shared) private(i,j,k)
	for(i=0;i<i_max;i++){
		for(j=0;j<j_max;j++){
			for(k=0;k<k_max;k++){
				Old[i][j][k] = New[i][j][k];
			}
		}
	}
}

//This function place one or more objects inside the domain
void insert_obj(double***e_r,int nb_obj,double*prop_obj,double dx,double point_per_proc_x,double point_per_proc_y,double point_per_proc_z,int lastx,int lasty,int lastz,int i_min_proc,int j_min_proc,int k_min_proc,int i_max_proc,int j_max_proc,int k_max_proc){
	int i = 0;
  int j = 0;
  int k = 0;
	int l = 0;
  int prop_per_obj = 7;
  
	for(l=0;l<nb_obj;l++){
		double n_x_double = (prop_obj[prop_per_obj*l]/dx)+1;
		int n_x = (int) n_x_double;
		double pos_x = (prop_obj[prop_per_obj*l+3]/dx);	
		int i_min = (int)pos_x;
		i_min = i_min - (n_x/2);
		int i_max = i_min + n_x-1;
	
		double n_y_double = (prop_obj[prop_per_obj*l+1]/dx)+1;
		int n_y = (int) n_y_double;
		double pos_y = (prop_obj[prop_per_obj*l+4]/dx);	
		int j_min = (int)pos_y;
		j_min = j_min - (n_y/2);
		int j_max = j_min + n_y-1;
	
		double n_z_double = (prop_obj[prop_per_obj*l+2]/dx)+1;
		int n_z = (int) n_z_double;
		double pos_z = (prop_obj[prop_per_obj*l+5]/dx);	
		int k_min = (int)pos_z;
		k_min = k_min - (n_z/2);
		int k_max = k_min + n_z-1;
	
		int b_inf_x =0;
		int b_inf_y =0;
		int b_inf_z =0;
		int b_sup_x = 0;
		int b_sup_y = 0;
		int b_sup_z = 0;

		if(i_min_proc<= i_max && j_min_proc<= j_max && k_min_proc<= k_max && i_max_proc>= i_min && j_max_proc>= j_min && k_max_proc>= k_min){
				b_inf_x = i_min - i_min_proc;				
				b_inf_y = j_min - j_min_proc;
				b_inf_z = k_min - k_min_proc;
				b_sup_x = i_max - i_min_proc;
				b_sup_y = j_max - j_min_proc;
				b_sup_z = k_max - k_min_proc;
				if(b_inf_x<0){
					b_inf_x = 0;		
				}
				if(b_inf_y<0){
					b_inf_y = 0;		
				}
				if(b_inf_z<0){
					b_inf_z = 0;		
				}
				if(point_per_proc_x-1+lastx<b_sup_x){
					b_sup_x = point_per_proc_x-1+lastx;
				}
				if(point_per_proc_y-1+lasty<b_sup_y){
					b_sup_y = point_per_proc_y-1+lasty;
				}
				if(point_per_proc_z-1+lastz<b_sup_z){
					b_sup_z = point_per_proc_z-1+lastz;
				}
				#pragma omp parallel for default(shared) private(i,j,k)
				for(i=b_inf_x;i<=b_sup_x;i++){
					for(j=b_inf_y;j<=b_sup_y+lasty;j++){
						for(k=b_inf_z;k<=b_sup_z;k++){
							e_r[i][j][k]= prop_obj[l+6];					
						}				
					}
				}
			}

	}
}

// This function imposes some zero values in a matrix
void Hom_BC(int i_min, int j_min, int k_min, int i_max, int j_max, int k_max,double*** M){
  int i= 0 ;
  int j = 0;
  int k = 0;
  #pragma omp parallel for default(shared) private(i,j,k)
  for(i=i_min;i<=i_max;i++){
    for(j=j_min;j<=j_max;j++){
      for(k=k_min;k<=k_max;k++){
        M[i][j][k] = 0;
      }
    }
  }
}

// This function initialize the value of the permitivity on the whole domain 
void init_geometry(std::vector<double> &geometry_init,std::vector<double> &vec_e_r,std::vector<double> &vec_mu_r,std::vector<double> &e_r_tot, std::vector<double> &mu_r_tot, int Nx, int Ny, int Nz){
	int i=0;
	int j=0;
	int k=0;
	int val=0;
	#pragma omp parallel for default(shared) private(i,j,k)
	for(i=0;i<Nx;i++){
		for(j=0;j<Ny;j++){
			for(k=0;k<Nz;k++){
				e_r_tot[i*Ny*Nz+k*Ny+j] = vec_e_r[(int) geometry_init[i*Ny*Nz+k*Ny+j]];
				mu_r_tot[i*Ny*Nz+k*Ny+j] = vec_mu_r[(int) geometry_init[i*Ny*Nz+k*Ny+j]];
			}
		}
	}
}

// This function initialize the value of the permitivity on the domain associated to the current process
void init_geom_one_proc(double***e_r,double***mu_r,std::vector<double> &e_r_tot,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz){
	int i=0;
	int j=0;
	int k=0;
	#pragma omp parallel for default(shared) private(i,j,k)
	for(i=0;i<point_per_proc_x+lastx;i++){
		for(j=0;j<point_per_proc_y+lasty;j++){
			for(k=0;k<point_per_proc_z+lastz;k++){
				e_r[i][j][k] = e_r_tot[(i+i_min_proc)*Ny*Nz+(k+k_min_proc)*Ny+(j+j_min_proc)];
			}
		}
	}
}

// This function make the objects rotate inside the domain
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &geometry_new,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta){
	int i = 0;
	int j = 0;
	int k = 0;
	int object = 0;
	double xrot = Lx/2;
	double yrot = Ly/2;	
	double x_after;
	double y_after;
	double z_after;
	double x_before;
	double y_before;
	double xc;
	double yc;
	double zc;
	double r;
	double l;

	#pragma omp parallel for default(shared) private(i,j,k,x_after,y_after,z_after,x_before,y_before,xc,yc,zc,r,l)
	for(i=0;i<Nx;i++){
		for(j=0;j<Ny;j++){
			for(k=0;k<Nz;k++){
				geometry_init[i*Ny*Nz+k*Ny+j] = 0;
				// Coordinate after rotation
				x_after = i*dx;
				y_after = j*dx;
				z_after = k*dx;
				// Coordinate before rotation
				x_before = xrot + (x_after-xrot)*cos(theta) + (y_after-yrot)*sin(theta);
				y_before = yrot - (x_after-xrot)*sin(theta)+(y_after-yrot)*cos(theta);

				for(object=0;object<ncube;object++){//	Cube
					if(((x_before)<=info_cube[7*object+3]+info_cube[7*object+0]/2)&&((x_before)>=info_cube[7*object+3]-info_cube[7*object+0]/2)&&((y_before)<=info_cube[7*object+4]+info_cube[7*object+1]/2)&&((y_before)>=info_cube[7*object+4]-info_cube[7*object+1]/2)&&((z_after)<=info_cube[7*object+5]+info_cube[7*object+2]/2)&&((z_after)>=info_cube[7*object+5]-info_cube[7*object+2]/2)){
						geometry_init[i*Ny*Nz+k*Ny+j] = info_cube[7*object+6];
					}
				}

				for(object=0;object<ncylinder;object++){// 	Cylinder
					xc = info_cylinder[7*object+1];
					yc = info_cylinder[7*object+2];
					zc = info_cylinder[7*object+3];
					r = info_cylinder[7*object+4];
					l = info_cylinder[7*object+6];

					if(info_cylinder[7*object]==0){
						if(((y_before-yc)*(y_before-yc)+(z_after-zc)*(z_after-zc)<=r*r) && x_before<=xc+l/2 && x_before>= xc-l/2){
							geometry_init[i*Ny*Nz+k*Ny+j]=info_cylinder[7*object+5];
						}
					}
					else if(info_cylinder[7*object]==1){
						if(((x_before-xc)*(x_before-xc)+(z_after-zc)*(z_after-zc)<=r*r) && y_before<=yc+l/2 && y_before>= yc-l/2){
							geometry_init[i*Ny*Nz+k*Ny+j]=info_cylinder[7*object+5];
						}
					}
					else if(info_cylinder[7*object]==2){						
						if(((x_before-xc)*(x_before-xc)+(y_before-yc)*(y_before-yc)<=r*r) && z_after<=zc+l/2 && z_after>= zc-l/2){
							geometry_init[i*Ny*Nz+k*Ny+j]=info_cylinder[7*object+5];
						}
					}
				}

				for(object=0;object<nsphere;object++){//	Sphere
					if(((info_sphere[5*object+0]-x_before)*(info_sphere[5*object+0]-x_before)+(info_sphere[5*object+1]-y_before)*(info_sphere[5*object+1]-y_before)+(info_sphere[5*object+2]-z_after)*(info_sphere[5*object+2]-z_after))<=info_sphere[5*object+3]*info_sphere[5*object+3]){
						geometry_init[i*Ny*Nz+k*Ny+j] = info_sphere[5*object+4];
					}
				}				
			}
		}
	}
}

void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val){
	
	int i=0;
	int j=0;
	int k=0;
	
	if(P==2){	// Cube
		#pragma omp parallel for default(shared) private(i,j,k)
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z;k++){
					if(((i*dx)<=properties[3]+properties[0]/2)&&((i*dx)>=properties[3]-properties[0]/2)&&((j*dx)<=properties[4]+properties[1]/2)&&((j*dx)>=properties[4]-properties[1]/2)&&((k*dx)<=properties[5]+properties[2]/2)&&((k*dx)>=properties[5]-properties[2]/2)){
						geometry[i*Y*Z+k*Y+j]=val;
					}
				}
			}
		}
	}
	else if(P==1){	//Cylinder
		double xc = properties[1];
		double yc = properties[2];
		double zc = properties[3];
		double r = properties[4];
		double l = properties[6];
		double xp;
		double yp;
		double zp;	
		#pragma omp parallel for default(shared) private(i,j,k,xp,yp,zp)	
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					xp = i*dx;
					yp = j*dx;
					zp = k*dx;
					if(properties[0]==0){
						if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
						}
					}
					else if(properties[0]==1){
						if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
						}
					}
					else if(properties[0]==2){						
						if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
						}
					}
				}
			}
		}
	}
	else if(P==0){	//Circle
		#pragma omp parallel for default(shared) private(i,j,k)
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-i*dx)*(properties[0]-i*dx)+(properties[1]-j*dx)*(properties[1]-j*dx)+(properties[2]-k*dx)*(properties[2]-k*dx))<=properties[3]*properties[3]){
						geometry[i*Y*Z+k*Y+j]=val;
					}
				}
			}
		}
	}
}

// This function rotate the power grid to make it compatible with the thermic part
void rotate_Power_grid(std::vector<double> &Power_electro,std::vector<double> &Power_thermo,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta){
	int i = 0;
	int j = 0;
	int k = 0;
	double xc = Lx/2;
	double yc = Ly/2; 
	double x_before ;
	double y_before;
	double z_before;
	double x_after;
	double y_after;
	double z_after;
	double x1;
	double y1;
	int i1;
	int j1;
	double xi;
	double eta;
	#pragma omp parallel for default(shared) private(i,j,k,x_before,y_before,z_before,x_after,y_after,z_after,x1,y1,i1,j1,xi,eta)
	for(k=0;k<Nz;k++){
		for(j=0;j<Ny;j++){
			for(i=0;i<Nx;i++){
				// Coordinate before rotation
				x_before = i*dx;
				y_before = j*dx;
				z_before = k*dx;
				// Coordinate after rotation
				x_after = xc + (x_before-xc)*cos(theta) - (y_before-yc)*sin(theta);
				y_after = yc + (x_before-xc)*sin(theta)+(y_before-yc)*cos(theta);
				z_after = z_before;

				if(x_after<=0||y_after<=0||Lx<=x_after||Ly<=y_after){ // The dissipated power is set to 0 for points leaving the domain after rotation
					Power_thermo[i*Ny*Nz+k*Ny+j] = 0;
				}
				else{	// Bilinear approximation of the power
					x1 = x_after/dx;
					y1 = y_after/dx;
					i1 = (int) x1;
					j1 = (int) y1;
					x1 = i1*dx;
					y1 = j1*dx;
					xi = (x_after-x1)/dx;
					eta = (y_after-y1)/dx;
					Power_thermo[i*Ny*Nz+k*Ny+j] = (1-xi)*(1-eta)*Power_electro[(i1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1-eta)*Power_electro[(i1+1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1+eta)*Power_electro[(i1+1)*Ny*Nz+(k)*Ny+(j1+1)]+(1-xi)*(1+eta)*Power_electro[(i1)*Ny*Nz+(k)*Ny+(j1+1)];
					Power_thermo[i*Ny*Nz+k*Ny+j] = Power_thermo[i*Ny*Nz+k*Ny+j]/4;
 
				}
			}
		}
	}

}





