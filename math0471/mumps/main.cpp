// example of hybrid MPI/OpenMP program
//     run with 2 processes and 6 threads per process (ubuntu)
//         export OMP_NUM_THREADS=6
//         [ubuntu - openmpi]
//         mpirun -np 2 -cpus-per-rank 6 --bind-to core:overload-allowed  bin/fdtd_mpi
//         [windows - microsoft mpi]
//         mpiexec -np 2 bin\fdtd_mpi

#include "vtl.h"
#include "vtlSPoints.h"
#include "laplace.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
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
// from c_example.c ------
#include "mpi.h"
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1]
// -----------------------

/***************************************************************************************
				Electro functions
***************************************************************************************/
void divisor(int np,int*r);
void compute_proc_ind_point(int nbproc,int Nx,int Ny,int Nz,int divx,int divy,int divz,int*i_min_proc,int*j_min_proc,int*k_min_proc,int*i_max_proc,int*j_max_proc,int*k_max_proc,int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z);
//void Update_prev_in_send(int*point_per_proc_x,int*point_per_proc_y,int*point_per_proc_z,int lastx,int lasty,int lastz,double*V1,double***M1,double*V2,double***M2,int myrank,int Case);
void Update_E_inside(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double dt,double dx,double e_0,std::vector<double> &e_r,int Case);
void Update_E_boundary(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double**H_boundary,double dt,double dx,double e_0,std::vector<double> &e_r,int myrank,int Case,int Nx, int Ny, int Nz);
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
void init_geom_one_proc(std::vector<double> &e_rx,double***mu_r,std::vector<double> &e_ry,std::vector<double> &e_rz,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz);
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta);
void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx, double val,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz , std::vector<double> &vec_er);
void place_cube(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er);
void place_cylinder(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er);
void place_sphere(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er);
void set_rel_perm_one_proc(std::vector<double> &e_r,std::vector<double> &e_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz, int comp);
void rotate_rel_perm(std::vector<double> &e_r_tot,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,int component);
void set_vec(std::vector<double> &vec, int nbp, double val);
void export_coupe(int direction, int component, double pos1,double pos2,int Nx_tot,int Ny_tot,int Nz_tot,double***M,double dx,int step,int myrank, int i_min,int i_max,int j_min ,int j_max ,int k_min ,int k_max ,int Nx,int Ny,int Nz,int lastx,int lasty,int lastz);
void export_power_thermo(std::vector<double> &Power_tot,int Nx,int Ny,int Nz);
void rotate_Power_grid(std::vector<double> &Power_electro,std::vector<double> &Power_thermo,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta);


/***************************************************************************************
				Thermo functions
***************************************************************************************/

void main_th(std::vector<double> &Source_elec, std::vector<double> &Temperature, std::vector<int> &BC, std::vector<double> &T_Dir, double T_0,double dx,double h_air, double Lx, double Ly, double Lz, double dt, int step_max, int nb_source, int SR, double theta, int n_sphere, int n_cylinder, int n_cube, std::vector<double> &prop_sphere, std::vector<double> &prop_cylinder, std::vector<double> &prop_cube, double T_food_init, double x_min_th, double y_min_th, double z_min_th, double dx_electro, double Lx_electro, double Ly_electro, double Lz_electro, int prop_per_source, std::vector<double> &prop_source, std::vector<double> &Cut, std::vector<double> &Pos_cut, int N_cut,std::vector<double> &step_cut, int nb_probe, std::vector<double> &Pos_probe, DMUMPS_STRUC_C &id,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &constant,std::vector<double> &geometry,int step_pos,int thermo_domain);
void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain);
void Compute_a_T0(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z);
void insert_obj_th(std::vector<double> &temp, std::vector<double> &k_heat_x, std::vector<double> &k_heat_y, std::vector<double> &k_heat_z, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &prop_obj, int X,int Y,int Z, double dx,std::vector<double> &geometry);
void insert_Source_th(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp,double x_min_th,double y_min_th,double z_min_th);
void export_coupe_th(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,double dx, int step, double x_min_th, double y_min_th, double z_min_th);
void export_probe_th(double nb_probe , std::vector<double> &probe,int step_max,int step_pos);
void set_kheat(int Case,int X,int Y,int Z, std::vector<double> &properties,int l,double dx,std::vector<double> &k_heat);
void Compute_a_T0_2(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h);
void place_geometry_th(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &k_heatx,std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,std::vector<double> &cp, double x_min_th, double y_min_th, double z_min_th);
void set_T0(std::vector<double> &Temp,std::vector<double> &geometry,double T_0,double T_init_food,int  X,int  Y,int  Z );
void rotate_Power_grid_th(std::vector<double> &Source_init,std::vector<double> &Source_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta);
void rotate_T_grid(std::vector<double> &T_init,std::vector<double> &T_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta, double T_air);
void set_source_from_elec(std::vector<double> &Source,std::vector<double> &Source_elec,double x_min_th,double y_min_th,double z_min_th,double dx,double dx_electro,int X_th,int Y_th,int Z_th,int X_elec,int Y_elec,int Z_elec);
int get_my_rank();
void check_MUMPS(DMUMPS_STRUC_C &id);
void init_MUMPS(DMUMPS_STRUC_C &id);
void end_MUMPS(DMUMPS_STRUC_C &id);
void solve_MUMPS(DMUMPS_STRUC_C &id, int step);
void host_work(DMUMPS_STRUC_C &id,double Lx,double Ly,double Lz,double delta_x,double delta_t,int step_max,double theta,int nb_source, std::vector<double> &prop_source,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0,int SR,std::vector<double> &Cut, std::vector<double> &Pos_cut, std::vector<double> &step_cut, double nb_probe, std::vector<double> &Pos_probe,int n_sphere,std::vector<double> &prop_sphere,int n_cylinder,std::vector<double> &prop_cylinder,int n_cube,std::vector<double> &prop_cube, double T_init_food,double h_air,double x_min_th,double y_min_th, double z_min_th,double dx_electro,int X_elec,int Y_elec,int Z_elec,std::vector<double> &Source_elec,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &constant,std::vector<double> &geometry,int step_pos,std::vector<double> &Temp,int thermo_domain);
void slave_work(DMUMPS_STRUC_C &id, int step_max);






int main(int argc, char **argv){	

/********************************************************************************
			Importation (ELECTRO).
********************************************************************************/
	// Declaration for MPI.
	MPI_Init(&argc,&argv);
	int nbproc;
	int myrank;
	MPI_Status mystatus ;
	MPI_Comm_size( MPI_COMM_WORLD, &nbproc);	
	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );

 	// initialise MUMPS/MPI
    	DMUMPS_STRUC_C id;
    	init_MUMPS(id);
    


	// Variables used to open files and to write in files.
  	int next_cut = 0;
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
	int step_pos = 0;
	int step_prec = 0;


	// Importation of param.dat and initialisation of other parameters.
	double data[21];	
	char chain[150];
	for (i=0 ; i<21; i++){
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
  	double E_amp = data[18];
	double delta_theta = data[19];
	int solve_electro = (int) data[20];
	double theta = 0;
	int nx;
	int ny;
	int nz;
	int nxp;
	int nyp;
	int nzp;
	double T_mean = 20/(f*dt);
	int step_mean = (int) T_mean;
	double Residual = 0;
  double Residual_0 = 0;
	int steady_state_reached = 0;
	int test_steady = 0;

  	// Prop spheres
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

  	// Prop cylinders
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

  	// Prop cubes
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
 
    // Property of spatial cuts
    std::vector<double> Cut(3);
    std::vector<double> Pos_cut(6);
    int N_cut;
    std::vector<double> step_cut;
    double data_cut[10];
    FileR = fopen(argv[5],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Cuts file. \n");
    	return 1; 
    }
    for (i=0 ; i<10; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the Cuts file. \n");
		return 1; 
  	  }	
   	 else{
		data_cut[i] = atof(chain);
   	 }
    }
    Cut[0] = data_cut[0];		// Cut along x
    Pos_cut[0] = data_cut[1];
    Pos_cut[1] = data_cut[2];
    Cut[1] = data_cut[3];		// Cut along y
    Pos_cut[2] = data_cut[4];
    Pos_cut[3] = data_cut[5];
    Cut[2] = data_cut[6];		// Cut along z
    Pos_cut[4] = data_cut[7];
    Pos_cut[5] = data_cut[8];
    N_cut = (int) data_cut[9];		// Number of cuts
    if(N_cut !=0){
    	for(i=0;i<N_cut;i++){
 		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Cuts file. \n");
			return 1; 
  	  	}
		step_cut.push_back(atof(chain)/dt);
    	}
    }
    fclose(FileR);
    
    
/********************************************************************************
			Importation (THERMO).
********************************************************************************/

    // Parameter of the simulation and boundary conditions
    std::vector<int>  BC(6);
    std::vector<double> T_Dir(6);
    double T_0;
    

    FileR = fopen(argv[6],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Heat data file. \n");
    	return 1; 
    }

    double data_th[40];	
    for (i=0 ; i<40; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the Heat data file2. \n");
		return 1; 
  	  }	
   	 else{
		data_th[i] = atof(chain);
   	 }
    }
    fclose(FileR);
    double Lx_th = data_th[0];    // Length of the domain
    double Ly_th = data_th[1];
    double Lz_th = data_th[2];
    double dx_th = data_th[3];    // Grid spacing
    double dt_th = data_th[4];    // Time step
    double Tf_th = data_th[5];    // Final time
    double temp_th = Tf_th/dt_th;
    int step_max_th = (int) temp_th;
    temp = Tf/Tf_th;
    int step_pos_max = (int) temp;
    int nb_source_th = (int) data_th[6];   // Number of power source
    int nb_obj_th = (int) data_th[7];      // Number of objects inside the domain
    for(i=0;i<6;i++){
    	BC[i] = data_th[8+i];
    	T_Dir[i] = data_th[14+i];
    }
    T_0 = data_th[20]; 			// Air Temperature
    double kheat_param_th = data_th[21];   
    double rho_param_th = data_th[22];
    double cp_param_th = data_th[23];
    double S_th = data_th[24];    // Sampling rate
    int SR_th = (int) 1/S_th;
    double theta_th = data_th[25];
    int n_sphere_th = (int) data_th[26];	// Nb of sphere
    int n_cylinder_th = (int) data_th[27];	// Nb of cylinder
    int n_cube_th = (int) data_th[28];	// Nb of cube
    double T_food_init_th = data_th[29];	// Initial temperature of the food
    double h_air = data_th[30];		// h_air
    double x_min_th = data_th[31];		// x_min
    double y_min_th = data_th[32];		// y_min
    double z_min_th = data_th[33];		// z_min
    double dx_electro = data_th[34];	// dx_electro
    double Lx_electro = data_th[35];	// dx_electro
    double Ly_electro = data_th[36];	// dx_electro
    double Lz_electro = data_th[37];	// dx_electro
    int solve_thermo = (int) data_th[38];
    int thermo_domain = (int) data_th[39];	

    int X_th = (int) (Lx_th/dx_th)+1;
    int Y_th = (int) (Ly_th/dx_th)+1;
    int Z_th = (int) (Lz_th/dx_th)+1;
    std::vector<double> Temperature(X_th*Y_th*Z_th);
    set_vec(Temperature, X_th*Y_th*Z_th, T_food_init_th);
    int prop_per_source_th = 7;       // Number of properties per power source
    std::vector<double> prop_source_th(prop_per_source_th*nb_source_th);
    
    
    // Properties of the power sources(heat)
    FileR = fopen(argv[7],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Source file. \n");
    	return 1; 
    }
    for (i=0 ; i<prop_per_source_th*nb_source_th; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the source file. \n");
		return 1; 
  	  }	
   	 else{
		prop_source_th[i] = atof(chain);
   	 }
    }
    fclose(FileR);
    
    // Property of cuts
    std::vector<double> Cut_th(3);
    std::vector<double> Pos_cut_th(6);
    int N_cut_th;
    std::vector<double> step_cut_th;
    double data_cut_th[10];
    FileR = fopen(argv[8],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Cuts file. \n");
    	return 1; 
    }
    for (i=0 ; i<10; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the Cuts file. \n");
		return 1; 
  	  }	
   	 else{
		data_cut_th[i] = atof(chain);
   	 }
    }
    Cut_th[0] = data_cut_th[0];		// Cut along x
    Pos_cut_th[0] = data_cut_th[1];
    Pos_cut_th[1] = data_cut_th[2];
    Cut_th[1] = data_cut_th[3];		// Cut along y
    Pos_cut_th[2] = data_cut_th[4];
    Pos_cut_th[3] = data_cut_th[5];
    Cut_th[2] = data_cut_th[6];		// Cut along z
    Pos_cut_th[4] = data_cut_th[7];
    Pos_cut_th[5] = data_cut_th[8];
    N_cut_th = (int) data_cut_th[9];		// Number of cuts
    if(N_cut_th !=0){
    	for(i=0;i<N_cut_th;i++){
 		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Cuts file. \n");
			return 1; 
  	  	}
		step_cut_th.push_back(atof(chain)/dt_th);
    	}
    }
    fclose(FileR);

    
    // Property of temporal probes
    double nb_probe_th;
    std::vector<double> Pos_probe_th;
    FileR = fopen(argv[9],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Probe file. \n");
    	return 1; 
    }
    if (fgets(chain, 150, FileR) == NULL){
	printf("Impossible to read the Probe file. \n");
	return 1; 
   }	
   else{
	nb_probe_th = atof(chain);
   }
   if(nb_probe_th!=0){
   	for(i=0;i<3*nb_probe_th;i++){
    		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Cuts file. \n");
			return 1; 
  		}	
 		else{
			Pos_probe_th.push_back(atof(chain));
  		}
    	}	 
   }
    
    for(i=0;i<nb_probe_th;i++){
		Pos_probe_th[i*3] = (Pos_probe_th[i*3]-x_min_th)/dx_th;
		Pos_probe_th[i*3+1] = (Pos_probe_th[i*3+1]-y_min_th)/dx_th;
		Pos_probe_th[i*3+2] = (Pos_probe_th[i*3+2]-z_min_th)/dx_th;
	}
    
    
    
    
/********************************************************************************
		Declaration of variables. (ELECTRO)
********************************************************************************/     
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
  int firstx = 0;
	int firsty = 0;
	int firstz = 0; 
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
 	if(ip==0){
    		firstx=1;
  	}
  	if(jp==0){
    		firsty=1;
  	}
  	if(kp==0){
    		firstz=1;
  	}

	// Physical constants
	double mu_0 = 4*3.141692*0.0000001;
	double e_0 = 8.854*0.000000000001;
	double Z = sqrt(mu_0/e_0);
	double c = 3*100000000;
	double***mu_r;
	mu_r =(double***)malloc((point_per_proc_x[myrank]+lastx)*sizeof(double**));

	std::vector<double> e_rx((point_per_proc_x[myrank]+lastx)*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]));
	std::vector<double> e_ry((point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]));
	std::vector<double> e_rz((point_per_proc_x[myrank])*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz));
  	set_vec(e_rx, (point_per_proc_x[myrank]+lastx)*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]), 1);
 	set_vec(e_ry, (point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]), 1);
 	set_vec(e_rz, (point_per_proc_x[myrank])*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz), 1);	
  
	std::vector<double> e_r_totx(Nx*Ny*Nz+(Ny*Nz));
	std::vector<double> e_r_toty(Nx*Ny*Nz+(Nx*Nz));
	std::vector<double> e_r_totz(Nx*Ny*Nz+(Nx*Ny));
	std::vector<double> mu_r_tot(Nx*Ny*Nz);
  	set_vec(e_r_totx, Nx*Ny*Nz+(Ny*Nz), 1);
  	set_vec(e_r_toty, Nx*Ny*Nz+(Nx*Nz), 1);
  	set_vec(e_r_totz, Nx*Ny*Nz+(Nx*Ny), 1);



	std::vector<double> geometry_init(Nx*Ny*Nz);
	std::vector<double> geometry_current(Nx*Ny*Nz);
	for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
		mu_r[i] =(double**)malloc((point_per_proc_y[myrank]+lasty)*sizeof(double*));
		for(j=0;j<point_per_proc_y[myrank]+lasty;j++){		
			mu_r[i][j] = (double*)malloc((point_per_proc_z[myrank]+lastz)*sizeof(double));
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
						mu_r[i][j][k] = 1;
			}
		}
	}


	/*Insertion of object*/
	std::vector<double> prop_temp(7);
	// Sphere	
	for(i=0;i<n_sphere;i++){		
		prop_temp[0] = prop_sphere[5*i];
		prop_temp[1] = prop_sphere[5*i+1];
		prop_temp[2] = prop_sphere[5*i+2];
		prop_temp[3] = prop_sphere[5*i+3];
		prop_temp[4] = prop_sphere[5*i+4];
		int Config = 0;
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[4],e_r_totx,e_r_toty,e_r_totz ,vec_e_r);
	}
	// Cylinder
	for(i=0;i<n_cylinder;i++){
		prop_temp[0] = prop_cylinder[7*i];
		prop_temp[1] = prop_cylinder[7*i+1];
		prop_temp[2] = prop_cylinder[7*i+2];
		prop_temp[3] = prop_cylinder[7*i+3];
		prop_temp[4] = prop_cylinder[7*i+4];
		prop_temp[5] = prop_cylinder[7*i+5];
		prop_temp[6] = prop_cylinder[7*i+6];
		int Config = 1;
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[5],e_r_totx,e_r_toty,e_r_totz ,vec_e_r);
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
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[6],e_r_totx,e_r_toty,e_r_totz ,vec_e_r);
	}
 	// Set the geometry on the current proc
	init_geom_one_proc(e_rx,mu_r,e_ry,e_rz,e_r_totx,e_r_toty,e_r_totz,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz, Nx, Ny, Nz);


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
	

/********************************************************************************
		Declaration of the grid used to export the results.
********************************************************************************/

	/********** Power Grid************/
	SPoints grid_Power;
	grid_Power.o = Vec3d(10.0, 10.0, 10.0); 
	grid_Power.np1 = Vec3i(0, 0, 0);       
	grid_Power.np2 = Vec3i(Nx-1, Ny-1, Nz-1); 
	grid_Power.dx = Vec3d(dx, dx, dx);  

	SPoints mygrid_Power;
	mygrid_Power.o = Vec3d(10.0, 10.0, 10.0);
	mygrid_Power.np1 = Vec3i(i_min_proc[myrank], j_min_proc[myrank], k_min_proc[myrank]);
	mygrid_Power.np2 = Vec3i(i_max_proc[myrank], j_max_proc[myrank], k_max_proc[myrank]);
	mygrid_Power.dx = Vec3d(dx, dx, dx);
	mygrid_Power.id = myrank;

	std::vector<SPoints> sgrids_Power; // list of subgrids (used by rank0 for pvti file)

	if (myrank == 0){
		sgrids_Power.resize(nbproc);
		for (int l = 0; l < nbproc; ++l){
			sgrids_Power[l].o = Vec3d(10.0, 10.0, 10.0);
			sgrids_Power[l].np1 = Vec3i(i_min_proc[l], j_min_proc[l], k_min_proc[l]);
			if(divx!=1){
				sgrids_Power[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l], k_max_proc[l]);
			}
			else{
				sgrids_Power[l].np2 = Vec3i(i_max_proc[l], j_max_proc[l], k_max_proc[l]);
			}
			sgrids_Power[l].dx = Vec3d(dx, dx, dx);	
      			sgrids_Power[l].id = l;		 		
		}
	}
	 // creation of the fields (over my subdomain)
	int mynbp_Power = mygrid_Power.nbp();	
	std::vector<double> Power_new(mynbp_Power);
	set_vec(Power_new,mynbp_Power, 0);
	std::vector<double> Power_old(mynbp_Power);
	set_vec(Power_old, mynbp_Power , 0);
	mygrid_Power.scalars["Power"] = &Power_new;
  std::vector<double> Power_tot(Nx*Ny*Nz);
  std::vector<double> Power_tot_rotated_back(Nx*Ny*Nz);
	set_vec(Power_tot, Nx*Ny*Nz , 0);
	set_vec(Power_tot_rotated_back, Nx*Ny*Nz , 0);
  std::vector<double> Power_send(mynbp_Power);
	set_vec(Power_send, mynbp_Power , 0);


	/************ Electric field ****************/

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
  std::vector<double> er_export(mynbp_Ey);
	for(i=0;i<mynbp_Ey;i++){
		Ey_vec[i] = 0;
	}
 	mygrid_Ey.scalars["E Y"] = &Ey_vec;
  mygrid_Ey.scalars["Geometry"] = &er_export;

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


	/*********** Magnetic field ****************/

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
		Declaration of variables. (THERMO)
********************************************************************************/ 

/*Declaration and Initialization of physical characteristics.*/	
    int n_th = X_th*Y_th*Z_th;   
    std::vector<double> geometry_th(n_th);
    std::vector<double> k_heat_x(n_th+Y_th*Z_th);	// One more point in the x direction
    std::vector<double> k_heat_y(n_th+X_th*Z_th);	// One more point in the y direction
    std::vector<double> k_heat_z(n_th+X_th*Y_th);	// One more point in the z direction
    std::vector<double> rho(n_th);    
    std::vector<double> cp(n_th);
    std::vector<double> vec_k;
    std::vector<double> vec_rho;
    std::vector<double> vec_cp;
 
    // Air
    vec_k.push_back(0.025);
    vec_rho.push_back(1.2);
    vec_cp.push_back(1004);

    // Chicken
    vec_k.push_back(0.5);
    vec_rho.push_back(1080);
    vec_cp.push_back(3132);

    // Not physical
    vec_k.push_back(1);
    vec_rho.push_back(1);
    vec_cp.push_back(1000000);
    
    std::vector<double> constant(n_th);
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<X_th*Y_th*Z_th;i++){  	
   	rho[i] = vec_rho[0];
	cp[i] = vec_cp[0];                            /*************** UTILISER SET VEC PLUTOT ***********************/
    }  
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n_th+Y_th*Z_th;i++){ 
	k_heat_x[i] = vec_k[0];
    }
   #pragma omp parallel for default(shared) private(i)
   for(i=0;i<n_th+X_th*Z_th;i++){ 
	k_heat_y[i] = vec_k[0];
    }
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n_th+X_th*Y_th;i++){ 
	k_heat_z[i] = vec_k[0];
    }


    // Placement of the geometry
	// Sphere	
	for(i=0;i<n_sphere;i++){
		int j=0;
		int prop_per_obj = 5;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_sphere[prop_per_obj*i+j];
		}		
		int Config = 0;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[4],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th);
	}
	//Cylinder
	for(i=0;i<n_cylinder;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cylinder[prop_per_obj*i+j];
		}
		int Config = 1;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[5],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th);
	}
	// Cube
	for(i=0;i<n_cube;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cube[prop_per_obj*i+j];
		}
		int Config = 2;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[6],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th);
	}    
        #pragma omp parallel for default(shared) private(i)
  for(i=0;i<X_th*Y_th*Z_th;i++){
  	constant[i] = (dt_th)/(rho[i]*cp[i]*dx_th*dx_th);
  }


/********************************************************************************
			Begining of the algorithm.
********************************************************************************/


step_pos_max = 3;    // To be suppressed afterwards


if(solve_electro==0){	// Desactivation of the electro magnetic solver
	step_pos_max = 0;
	step_max = -1;
}

while(step_pos<=step_pos_max){

	// Reinitialisation of the calculation of the fields and of the power grid
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]-1,Ex_new);
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Ey_new);
 	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Ez_new);
	
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]-1,Ex_prev);
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Ey_prev);
 	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Ez_prev);

	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]+lastz-1,Hx_new);
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Hy_new);
 	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Hz_new);

	Hom_BC(0, 0, 0, point_per_proc_x[myrank]-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]+lastz-1,Hx_prev);
	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]-1, point_per_proc_z[myrank]+lastz-1,Hy_prev);
 	Hom_BC(0, 0, 0, point_per_proc_x[myrank]+lastx-1, point_per_proc_y[myrank]+lasty-1, point_per_proc_z[myrank]-1,Hz_prev);

	set_vec(Power_tot, Nx*Ny*Nz, 0);
	set_vec(Power_new, mynbp_Power, 0);
	set_vec(Power_old, mynbp_Power, 0);

	if(myrank==0){
		printf("\n*********************************************************\n          POSITION DE LA NOURRITURE : %d sur %d \n*********************************************************\n",step_pos,step_pos_max);
		if(solve_electro==1){
			printf("      SOLVING OF THE ELECTRO MAGNETIC EQUATION... \n",step_pos,step_pos_max);
		}
	}

	while(step<=step_max){
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
   
		//Update of the electric field.
		//	X component		
		if(lasty==1 && lastz==1){
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
		}
		else if(lasty==1){
    			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
     			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_rx,myrank,5,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
		}
		else if (lastz == 1){
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
     		  	Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_rx,myrank,3,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
		}
		else{
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
     			  Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_rx,myrank,5,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
     		  	Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_rx,myrank,3,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
				k = point_per_proc_z[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ex_new[i][j][k] = Ex_prev[i][j][k] +(dt/(e_0*e_rx[i*(point_per_proc_y[myrank]*point_per_proc_z[myrank])+k*(point_per_proc_y[myrank])+j]*dx))*((Hz_right[i][k]-Hz_prev[i][j][k])-(Hy_up[i][j]-Hy_prev[i][j][k]));
			}
		}

		//	Y component
		if(lastx==1 && lastz==1){
     			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);			
		}
		else if(lastx==1){
     			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);
     			 Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_ry,myrank,6,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
		}
		else if(lastz==1){
     			 Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);
     			 Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_ry,myrank,1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
		}
		else{
     			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);
     			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_ry,myrank,6,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
     			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_ry,myrank,1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
				i = point_per_proc_x[myrank]-1;
				k = point_per_proc_z[myrank]-1;
				Ey_new[i][j][k] = Ey_prev[i][j][k] + (dt/(e_0*e_ry[i*((point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank])+k*(point_per_proc_y[myrank]+lasty)+j]*dx))*((Hx_up[i][j]-Hx_prev[i][j][k])-(Hz_front[j][k]-Hz_prev[i][j][k]));
			}
		}
		//	Z component
		if(lastx==1 && lasty==1){
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);			
		}
		else if(lastx==1){
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
      			Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_rz,myrank,4,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);	
		}
		else if(lasty==1){
      			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
    			  Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_rz,myrank,2,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
		}
		else{
    			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_rz,myrank,4,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_rz,myrank,2,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				i = point_per_proc_x[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ez_new[i][j][k] = Ez_prev[i][j][k] + (dt/(e_0*e_rz[i*(point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz))+k*(point_per_proc_y[myrank])+j]*dx))*((Hy_front[j][k]-Hy_prev[i][j][k])-(Hx_right[i][k]-Hx_prev[i][j][k]));
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
			if(P==1){
				#pragma omp parallel for default(shared) private(i,j,k)
				for(i=b_inf_x;i<=b_sup_x;i++){
					for(j=b_inf_y;j<=b_sup_y+lasty;j++){
						for(k=b_inf_z;k<=b_sup_z;k++){
							double zrel = (k+k_min_proc[myrank]-k_min_a)*dx;
							Ey_new[i][j][k]= E_amp*sin((3.141692*zrel)/l_az)*sin(omega*step*dt);	
          				  	}		
					}				
				}
				#pragma omp parallel for default(shared) private(i,j,k)
				for(i=b_inf_x;i<=b_sup_x;i++){
					for(j=b_inf_y;j<=b_sup_y;j++){
						for(k=b_inf_z;k<=b_sup_z+lastz;k++){           				 
							double yrel = (j+j_min_proc[myrank]-j_min_a)*dx;	
							Ez_new[i][j][k]= E_amp*sin((3.141692*yrel)/l_ay)*sin(omega*step*dt);	
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
                         er_export[idx] = e_ry[(i-npx1)*np[1]*np[2]+(k-npz1)*np[1]+(j-npy1)];
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
		/***********************************************
				Storage of the Results
		***********************************************/

		if(step%SR==0){//save results of the mpi process to disk
			//export_spoints_XML("Ex", step, grid_Ex, mygrid_Ex, ZIPPED, Nx, Ny, Nz, 0);
		  	//export_spoints_XML("Ey", step, grid_Ey, mygrid_Ey, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Ez", step, grid_Ez, mygrid_Ez, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hx", step, grid_Hx, mygrid_Hx, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hy", step, grid_Hy, mygrid_Hy, ZIPPED, Nx, Ny, Nz, 0);
			//export_spoints_XML("Hz", step, grid_Hz, mygrid_Hz, ZIPPED, Nx, Ny, Nz, 0);

            		if (myrank == 0){	// save main pvti file by rank0
				//export_spoints_XMLP("Ex", step, grid_Ex, mygrid_Ex, sgrids_Ex, ZIPPED);
                		//export_spoints_XMLP("Ey", step, grid_Ey, mygrid_Ey, sgrids_Ey, ZIPPED);
				//export_spoints_XMLP("Ez", step, grid_Ez, mygrid_Ez, sgrids_Ez, ZIPPED);
				//export_spoints_XMLP("Hx", step, grid_Hx, mygrid_Hx, sgrids_Hx, ZIPPED);
				//export_spoints_XMLP("Hy", step, grid_Hy, mygrid_Hy, sgrids_Hy, ZIPPED);
				//export_spoints_XMLP("Hz", step, grid_Hz, mygrid_Hz, sgrids_Hz, ZIPPED);
            		}        
        	}  
     // Extraction of a cut if needed
	if(step == step_cut[next_cut]){// To extract a cut
		next_cut++;
		if(Cut[0]==1){// Cut along x
			export_coupe(1, 1, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Ex_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(1, 2, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Ey_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(1, 3, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Ez_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(1, 4, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Hx_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(1, 5, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Hy_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(1, 6, Pos_cut[0], Pos_cut[1], Nx, Ny, Nz, Hz_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
		}
		if(Cut[1]==1){// Cut along y
			export_coupe(2, 1, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Ex_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(2, 2, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Ey_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(2, 3, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Ez_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(2, 4, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Hx_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(2, 5, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Hy_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(2, 6, Pos_cut[2], Pos_cut[3], Nx, Ny, Nz, Hz_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
		}
		if(Cut[2]==1){// Cut along z
			export_coupe(3, 1, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Ex_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(3, 2, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Ey_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(3, 3, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Ez_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(3, 4, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Hx_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(3, 5, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Hy_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
      export_coupe(3, 6, Pos_cut[4], Pos_cut[5], Nx, Ny, Nz, Hz_new, dx, step, myrank,i_min_proc[myrank],i_max_proc[myrank],j_min_proc[myrank],j_max_proc[myrank],k_min_proc[myrank],k_max_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz);
		}				
	}

	// Computation of the power grid (TO BE PARAMETRIZED)
		nx = point_per_proc_x[myrank];
		ny = point_per_proc_y[myrank];
		nz = point_per_proc_z[myrank];

	for(i=1;i<(nx-lastx);i++){
			for(j=1;j<(ny-lasty);j++){
				for(k=1;k<(nz-lastz);k++) {
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+(j-firsty)]*Ex_new[i][j-firsty][k-firstz]*Ex_new[i][j-firsty][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+j+1-firsty]*Ex_new[i][j-firsty+1][k-firstz]*Ex_new[i][j-firsty+1][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j-firsty]*Ex_new[i][j-firsty][k+1-firstz]*Ex_new[i][j-firsty][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/12;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k-firstz]*Ey_new[i-firstx][j][k-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k-firstz]*Ey_new[i+1-firstx][j][k-firstz])+(e_ry[(i-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k+1-firstz]*Ey_new[i-firstx][j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/12;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i-firstx][j-firsty][k]*Ez_new[i-firstx][j-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i+1-firstx][j-firsty][k]*Ez_new[i+1-firstx][j-firsty][k])+(e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i-firstx][j+1-firsty][k]*Ez_new[i-firstx][j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/12;		        
        }
			}
		}
   
   if(firstx!=1){
       for(i=0;i<=0;i++){
		  	for(j=1;j<(ny-lasty);j++){
				for(k=1;k<(nz-lastz);k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+(j-firsty)]*Ex_new[i][j-firsty][k-firstz]*Ex_new[i][j-firsty][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+j+1-firsty]*Ex_new[i][j-firsty+1][k-firstz]*Ex_new[i][j-firsty+1][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j-firsty]*Ex_new[i][j-firsty][k+1-firstz]*Ex_new[i][j-firsty][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/12;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_back[j][k-firstz]*Ey_back[j][k-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k-firstz]*Ey_new[i+1-firstx][j][k-firstz])+(e_ry[(i)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_back[j][k+1-firstz]*Ey_back[j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/12;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_back[j-firsty][k]*Ez_back[j-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i+1-firstx][j-firsty][k]*Ez_new[i+1-firstx][j-firsty][k])+(e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_back[j+1-firsty][k]*Ez_back[j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/12;		        
        }
			}
		}
   }   
   if(firsty!=1){
   for(i=1;i<(nx-lastx);i++){
			for(j=0;j<=0;j++){
				for(k=1;k<(nz-lastz);k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+(j)]*Ex_left[i][k-firstz]*Ex_left[i][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+j+1-firsty]*Ex_new[i][j-firsty+1][k-firstz]*Ex_new[i][j-firsty+1][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j]*Ex_left[i][k+1-firstz]*Ex_left[i][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/12;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k-firstz]*Ey_new[i-firstx][j][k-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k-firstz]*Ey_new[i+1-firstx][j][k-firstz])+(e_ry[(i-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k+1-firstz]*Ey_new[i-firstx][j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/12;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i-firstx][k]*Ez_left[i-firstx][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i+1-firstx][k]*Ez_left[i+1-firstx][k])+(e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i-firstx][j+1-firsty][k]*Ez_new[i-firstx][j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/12;		        
        }
			}
		}   
   }      
   if(firstz!=1){
   for(i=1;i<(nx-lastx);i++){
			for(j=1;j<(ny-lasty);j++){
				for(k=0;k<=0;k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k)*(ny)+(j-firsty)]*Ex_bottom[i][j-firsty]*Ex_bottom[i][j-firsty])+(e_rx[(i)*(ny)*(nz)+(k)*(ny)+j+1-firsty]*Ex_bottom[i][j-firsty+1]*Ex_bottom[i][j-firsty+1])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j-firsty]*Ex_new[i][j-firsty][k+1-firstz]*Ex_new[i][j-firsty][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/12;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i-firstx][j]*Ey_bottom[i-firstx][j])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i+1-firstx][j]*Ey_bottom[i+1-firstx][j])+(e_ry[(i-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k+1-firstz]*Ey_new[i-firstx][j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/12;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i-firstx][j-firsty][k]*Ez_new[i-firstx][j-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i+1-firstx][j-firsty][k]*Ez_new[i+1-firstx][j-firsty][k])+(e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i-firstx][j+1-firsty][k]*Ez_new[i-firstx][j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/12;		        
        }
			}
		}  
   }
   
   if(firstx!=1 && firsty!=1){
     for(i=0;i<=0;i++){
		  	for(j=0;j<=0;j++){
				for(k=1;k<(nz-lastz);k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+(j)]*Ex_left[i][k-firstz]*Ex_left[i][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz)*(ny)+j+1-firsty]*Ex_new[i][j-firsty+1][k-firstz]*Ex_new[i][j-firsty+1][k-firstz])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j]*Ex_left[i][k+1-firstz]*Ex_left[i][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/11;
					
          
          //y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_back[j][k-firstz]*Ey_back[j][k-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k-firstz]*Ey_new[i+1-firstx][j][k-firstz])+(e_ry[(i)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_back[j][k+1-firstz]*Ey_back[j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/11;
				
        
        	//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i+1-firstx][k]*Ex_left[i+1-firstx][k])+(e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_back[j+1-firsty][k]*Ez_back[j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/11;		        
        }
			}
		}
   }
   
   if(firstx!=1 && firstz!=1){
     for(i=0;i<=0;i++){
		  	for(j=1;j<(ny-lasty);j++){
				for(k=0;k<=0;k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k)*(ny)+(j-firsty)]*Ex_bottom[i][j-firsty]*Ex_bottom[i][j-firsty])+(e_rx[(i)*(ny)*(nz)+(k)*(ny)+j+1-firsty]*Ex_bottom[i][j-firsty+1]*Ex_bottom[i][j-firsty+1])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j-firsty]*Ex_new[i][j-firsty][k+1-firstz]*Ex_new[i][j-firsty][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/11;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i+1-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i+1-firstx][j]*Ey_bottom[i+1-firstx][j])+(e_ry[(i)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_back[j][k+1-firstz]*Ey_back[j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/11;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_back[j-firsty][k]*Ez_back[j-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j-firsty]*Ez_new[i+1-firstx][j-firsty][k]*Ez_new[i+1-firstx][j-firsty][k])+(e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_back[j+1-firsty][k]*Ez_back[j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/11;		        
        }
			}
		}  
   }
   
   if(firsty!=1 && firstz!=1){
     for(i=1;i<(nx-lastx);i++){
			for(j=0;j<=0;j++){
				for(k=0;k<=0;k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k)*(ny)+j+1-firsty]*Ex_bottom[i][j-firsty+1]*Ex_bottom[i][j-firsty+1])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j]*Ex_left[i][k+1-firstz]*Ex_left[i][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/11;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i-firstx][j]*Ey_bottom[i-firstx][j])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i+1-firstx][j]*Ey_bottom[i+1-firstx][j])+(e_ry[(i-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i-firstx][j][k+1-firstz]*Ey_new[i-firstx][j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/11;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i-firstx][k]*Ez_left[i-firstx][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i+1-firstx][k]*Ez_left[i+1-firstx][k])+(e_rz[(i-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i-firstx][j+1-firsty][k]*Ez_new[i-firstx][j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/11;		        
        }
			}
		} 
   }
   if(firstx!=1 && firsty!=1 && firstz!=1){
     for(i=1;i<(nx-lastx);i++){
			for(j=0;j<=0;j++){
				for(k=0;k<=0;k++) {    
           //x    
           Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((e_rx[(i)*(ny)*(nz)+(k)*(ny)+j+1-firsty]*Ex_bottom[i][j-firsty+1]*Ex_bottom[i][j-firsty+1])+(e_rx[(i)*(ny)*(nz)+(k+1-firstz)*(ny)+j]*Ex_left[i][k+1-firstz]*Ex_left[i][k+1-firstz])+(e_rx[(i)*(ny)*(nz)+(k-firstz+1)*(ny)+j+1-firsty]*Ex_new[i][j+1-firsty][k+1-firstz]*Ex_new[i][j+1-firsty][k+1-firstz]))/9;
					//y
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_ry[(i+1-firstx)*(ny+lasty)*nz+(k)*(ny+lasty)+j]*Ey_bottom[i+1-firstx][j]*Ey_bottom[i+1-firstx][j])+(e_ry[(i)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_back[j][k+1-firstz]*Ey_back[j][k+1-firstz])+(e_ry[(i+1-firstx)*(ny+lasty)*nz+(k+1-firstz)*(ny+lasty)+j]*Ey_new[i+1-firstx][j][k+1-firstz]*Ey_new[i+1-firstx][j][k+1-firstz]))/9;
					//z
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j]*Ez_left[i+1-firstx][k]*Ez_left[i+1-firstx][k])+(e_rz[(i)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_back[j+1-firsty][k]*Ez_back[j+1-firsty][k])+(e_rz[(i+1-firstx)*(ny)*(nz+lastz)+(k)*ny+j+1-firsty]*Ez_new[i+1-firstx][j+1-firsty][k]*Ez_new[i+1-firstx][j+1-firsty][k]))/9;		        
        }
			}
		}
   } 
 

		if(step%step_mean==0){ 
			/**************************************
				Steady state verification
			**************************************/

			// Check if steady state is reached on the current process
			Residual = 0;
			for(i=0;i<nx;i++){
				for(j=0;j<ny;j++){
					for(k=0;k<(nz);k++) {                         
						Power_new[i+j*nx+k*(ny)*nx] = (f*Power_new[i+j*nx+k*(ny)*nx])/(step_mean);						
						Residual = Residual + (Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx])*(Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx]);
						Power_old[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx];
					}
				}
			}		
     			 if((step/step_mean)==1||(step/step_mean)%5==0){
				if(Residual==0){
					Residual = 1;
					steady_state_reached=1;
				}		
        			Residual_0 = Residual;
      			}		

      			if(Residual<Residual_0){
				steady_state_reached = 1;
			}
			else{
				steady_state_reached = 0;
			}

			// Communication between the process in order to determine if the algorithm must continue or not.
			if(myrank==0){
				for(k=1;k<nbproc;k++){
					MPI_Recv(&test_steady,1,MPI_INT,k,k, MPI_COMM_WORLD, &mystatus );
					if(test_steady!=1){
						steady_state_reached = 0;
					}
				}
				for(k=1;k<nbproc;k++){
					MPI_Send(&steady_state_reached,1,MPI_INT, k, myrank, MPI_COMM_WORLD );
				}
			}
			else{
				MPI_Send(&steady_state_reached,1,MPI_INT, 0, myrank, MPI_COMM_WORLD );	
				MPI_Recv(&steady_state_reached,1,MPI_INT,0,0, MPI_COMM_WORLD, &mystatus );
			}    
       			printf("Step:  %d Rank : %d Residual : %lf\n",step, myrank, Residual/Residual_0);

   			steady_state_reached=1;		/************** To be suppressed if we want to reach the steady state *********************/

			if(steady_state_reached==1){
				break;
 			}
			else{
				set_vec(Power_new, nx*ny*nz, 0);
			}
		}

 	step++;	
	}
if(solve_electro==1){
	if(myrank==0){
	     for(i=i_min_proc[myrank];i<=i_max_proc[myrank];i++){
	       for(j=j_min_proc[myrank];j<=j_max_proc[myrank];j++){
		 for(k=k_min_proc[myrank];k<=k_max_proc[myrank];k++){
		   Power_tot[i*Ny*Nz+k*Ny+j] = Power_new[i+j*point_per_proc_x[myrank]+k*point_per_proc_x[myrank]*point_per_proc_y[myrank]];
		 }
	       }
	     }
	     for(l=1;l<nbproc;l++){
	       Power_send.resize(point_per_proc_x[l]*(point_per_proc_y[l])*(point_per_proc_z[l]));
	       MPI_Recv(&Power_send[0],point_per_proc_x[l]*(point_per_proc_y[l])*(point_per_proc_z[l]),MPI_DOUBLE,l,l,MPI_COMM_WORLD, &mystatus);
	       for(i=0;i<point_per_proc_x[l];i++){
		 for(j=0;j<point_per_proc_y[l];j++){
		   for(k=0;k<point_per_proc_z[l];k++){    
		       Power_tot[(i+i_min_proc[l])*Ny*Nz+(k+k_min_proc[l])*Ny+j+j_min_proc[l]] =  Power_send[i+j*point_per_proc_x[l]+k*point_per_proc_x[l]*point_per_proc_y[l]];  
		   }
		 }
	       }
	     }     
	   }
	   else{
	     MPI_Send(&Power_new[0],point_per_proc_x[myrank]*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]),MPI_DOUBLE,0,myrank,MPI_COMM_WORLD);     
	   }
	   //export_power_thermo(Power_tot,Nx,Ny,Nz);   // Suppress after coupling
	   
		// Saving of the last step

		//export_spoints_XML("Ex", step, grid_Ex, mygrid_Ex, ZIPPED, Nx, Ny, Nz, 0);
		export_spoints_XML("Ey", step+step_prec, grid_Ey, mygrid_Ey, ZIPPED, Nx, Ny, Nz, 0);
		export_spoints_XML("Ez", step+step_prec, grid_Ez, mygrid_Ez, ZIPPED, Nx, Ny, Nz, 0);
		//export_spoints_XML("Hx", step+step_prec, grid_Hx, mygrid_Hx, ZIPPED, Nx, Ny, Nz, 0);
		//export_spoints_XML("Hy", step+step_prec, grid_Hy, mygrid_Hy, ZIPPED, Nx, Ny, Nz, 0);
		//export_spoints_XML("Hz", step+step_prec, grid_Hz, mygrid_Hz, ZIPPED, Nx, Ny, Nz, 0);
		export_spoints_XML("Power", step+step_prec, grid_Power, mygrid_Power, ZIPPED, Nx, Ny, Nz, 0);
		if (myrank == 0){	// save main pvti file by rank0
			//export_spoints_XMLP("Ex", step+step_prec, grid_Ex, mygrid_Ex, sgrids_Ex, ZIPPED);
	      		export_spoints_XMLP("Ey", step+step_prec, grid_Ey, mygrid_Ey, sgrids_Ey, ZIPPED);
			export_spoints_XMLP("Ez", step+step_prec, grid_Ez, mygrid_Ez, sgrids_Ez, ZIPPED);
			//export_spoints_XMLP("Hx", step+step_prec, grid_Hx, mygrid_Hx, sgrids_Hx, ZIPPED);
			//export_spoints_XMLP("Hy", step+step_prec, grid_Hy, mygrid_Hy, sgrids_Hy, ZIPPED);
			//export_spoints_XMLP("Hz", step+step_prec, grid_Hz, mygrid_Hz, sgrids_Hz, ZIPPED);
			export_spoints_XMLP("Power", step+step_prec, grid_Power, mygrid_Power, sgrids_Power, ZIPPED);
		}
		step_prec += step;
		step = 1;
}	
/********************************************************************************
			Thermic calculation
*******************************************************************************/
if(solve_thermo){
	if(myrank==0){
	printf("\nSOLVING OF THE HEAT EQUATION...\n");
	}
	rotate_Power_grid(Power_tot,Power_tot_rotated_back,Nx,Ny,Nz,Lx,Ly,Lz,dx,theta);
	main_th(Power_tot_rotated_back, Temperature,BC, T_Dir,  T_0,dx_th,h_air,Lx_th,Ly_th,Lz_th,dt_th,step_max_th,nb_source_th,SR_th,theta_th,n_sphere,n_cylinder,n_cube,prop_sphere,prop_cylinder,prop_cube,T_food_init_th,x_min_th,y_min_th,z_min_th,dx,Lx,Ly_electro,Lz_electro,prop_per_source_th, prop_source_th, Cut_th,Pos_cut_th,N_cut_th,step_cut_th,nb_probe_th,Pos_probe_th, id,k_heat_x,k_heat_y,k_heat_z,rho,cp,vec_k,vec_rho,vec_cp,constant,geometry_th,step_pos,thermo_domain);
}

/*******************************************************************************
			   Rotation
*******************************************************************************/
   
      	theta += delta_theta;
      	rotate_geometry(geometry_init,e_r_totx,e_r_toty,e_r_totz ,vec_e_r, Nx, Ny, Nz, Lx, Ly, Lz, dx,n_sphere, prop_sphere, n_cylinder,prop_cylinder, n_cube,prop_cube, theta);                    
	init_geom_one_proc(e_rx,mu_r,e_ry,e_rz,e_r_totx,e_r_toty,e_r_totz,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz      , Nx, Ny, Nz);

step_pos++;	  
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
			free(mu_r[i][j]);
		}
		free(mu_r[i]);
	}
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

  // finalise MUMPS/MPI
  end_MUMPS(id);
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
void Update_E_inside(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double dt,double dx,double e_0,std::vector<double> &e_r,int Case){
  int i =0;
  int j =0;
  int k =0;  
    #pragma omp parallel for default(shared) private(i,j,k) 
  for(i=0;i<i_max;i++){
	for(j=0;j<j_max;j++){
		for(k=0;k<k_max;k++){	
			if(Case==1){// Comp x				
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i*j_max*k_max+k*j_max+j]*dx))*((H1_prev[i][j+1][k]-H1_prev[i][j][k])-(H2_prev[i][j][k+1]-H2_prev[i][j][k]));
			}
			else if(Case==2){// Comp y 
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*j_max*k_max+k*j_max+j]*dx))*((H1_prev[i][j][k+1]-H1_prev[i][j][k])-(H2_prev[i+1][j][k]-H2_prev[i][j][k]));
			}
			else if(Case==3){// Comp z 
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*j_max*k_max+k*j_max+j]*dx))*((H1_prev[i+1][j][k]-H1_prev[i][j][k])-(H2_prev[i][j+1][k]-H2_prev[i][j][k]));
			}	
		}
	}
  }
}

// This function update the electric field at the boundary of the domain attributed to a given process
void Update_E_boundary(int i_max,int j_max,int k_max,double***E_new,double***E_prev,double***H1_prev,double***H2_prev,double**H_boundary,double dt,double dx,double e_0,std::vector<double> &e_r,int myrank,int Case,int Nx, int Ny, int Nz){
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
			if(Case==1){ //Comp y 
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H1_prev[i][j][k+1]-H1_prev[i][j][k])-(H_boundary[j][k]-H2_prev[i][j][k]));
			}
			else if(Case==2){// Comp z
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H_boundary[j][k]-H1_prev[i][j][k])-(H2_prev[i][j+1][k]-H2_prev[i][j][k]));
			}
			else if(Case==3){// Comp x
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H_boundary[i][k]-H1_prev[i][j][k])-(H2_prev[i][j][k+1]-H2_prev[i][j][k]));
			}
			else if(Case==4){// Comp z
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H1_prev[i+1][j][k]-H1_prev[i][j][k])-(H_boundary[i][k]-H2_prev[i][j][k]));
			}
			else if(Case==5){// Comp x
				E_new[i][j][k] = E_prev[i][j][k] +(dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H1_prev[i][j+1][k]-H1_prev[i][j][k])-(H_boundary[i][j]-H2_prev[i][j][k]));
			}
			else if(Case==6){// Comp y 
				E_new[i][j][k] = E_prev[i][j][k] + (dt/(e_0*e_r[i*Ny*Nz+k*Ny+j]*dx))*((H_boundary[i][j]-H1_prev[i][j][k])-(H2_prev[i+1][j][k]-H2_prev[i][j][k]));
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
void init_geom_one_proc(std::vector<double> &e_rx,double***mu_r,std::vector<double> &e_ry,std::vector<double> &e_rz,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz){
	set_rel_perm_one_proc(e_rx,e_r_totx,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,0);
	set_rel_perm_one_proc(e_ry,e_r_toty,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,1);
	set_rel_perm_one_proc(e_rz,e_r_totz,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,2);	
}

// This function make the objects rotate inside the domain
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta){
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
 
 rotate_rel_perm(e_r_totx,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,0);
 rotate_rel_perm(e_r_toty,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,1);
 rotate_rel_perm(e_r_totz,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,2);
}

void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz , std::vector<double> &vec_er){
	
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
		place_cube(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er);
		place_cube(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er);
		place_cube(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er);
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
		place_cylinder(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er);
		place_cylinder(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er);
		place_cylinder(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er);
	}
	else if(P==0){	//Sphere
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
		place_sphere(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er);
		place_sphere(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er);
		place_sphere(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er);                                                     
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
					xi =-1+2*(x_after-x1)/dx;
					eta =-1+2*(y_after-y1)/dx;
					Power_thermo[i*Ny*Nz+k*Ny+j] = (1-xi)*(1-eta)*Power_electro[(i1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1-eta)*Power_electro[(i1+1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1+eta)*Power_electro[(i1+1)*Ny*Nz+(k)*Ny+(j1+1)]+(1-xi)*(1+eta)*Power_electro[(i1)*Ny*Nz+(k)*Ny+(j1+1)];
					Power_thermo[i*Ny*Nz+k*Ny+j] = Power_thermo[i*Ny*Nz+k*Ny+j]/4;
 
				}
			}
		}
	}
}

void place_cube(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er) {
	int i = 0;
	int j = 0;
	int k = 0;	
	double xx = 0;
	double yy = 0;
	double zz = 0;	
	if(component==0){
		xx = 1;
	}
	else if(component==1){
		yy = 1;
	}
	else if(component==2){
		zz = 1;
	}
		//#pragma omp parallel for default(shared) private(i,j,k)
		for(i=0;i<X+xx;i++){
			for(j=0;j<Y+yy;j++){
				for(k=0;k<Z+zz;k++){
					if(((i*dx-0.5*xx*dx)<=properties[3]+properties[0]/2)&&((i*dx-0.5*xx*dx)>=properties[3]-properties[0]/2)&&((j*dx-0.5*yy*dx)<=properties[4]+properties[1]/2)&&((j*dx-0.5*yy*dx)>=properties[4]-properties[1]/2)&&((k*dx-0.5*zz*dx)<=properties[5]+properties[2]/2)&&((k*dx-0.5*zz*dx)>=properties[5]-properties[2]/2)){
						e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
					}
				}
			}
		}
}

void place_cylinder(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er){
	int i = 0;
	int j = 0;
	int k = 0;	
	double xx = 0;
	double yy = 0;
	double zz = 0;	
	if(component==0){
		xx = 1;
	}
	else if(component==1){
		yy = 1;
	}
	else if(component==2){
		zz = 1;
	}
	double xc = properties[1];
	double yc = properties[2];
	double zc = properties[3];
	double r = properties[4];
	double l = properties[6];
	double xp;
	double yp;
	double zp;	
	//#pragma omp parallel for default(shared) private(i,j,k,xp,yp,zp)	
	for(k=0;k<Z+zz;k++){
		for(j=0;j<Y+yy;j++){
			for(i=0;i<X+xx;i++){
				xp = i*dx-0.5*xx*dx;
				yp = j*dx-0.5*yy*dx;
				zp = k*dx-0.5*zz*dx;
				if(properties[0]==0){
					if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
						e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
					}
				}
				else if(properties[0]==1){
					if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
						e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
					}
				}
				else if(properties[0]==2){						
					if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
						e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
					}
				}
			}
		}
	}
}

void place_sphere(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er){
	double i = 0;
	double j = 0;
	double k = 0;	
	double xx = 0;
	double yy = 0;
	double zz = 0;	
	if(component==0){
		xx = 1;
	}
	else if(component==1){
		yy = 1;
	}
	else if(component==2){
		zz = 1;
	}
		for(k=0;k<Z+zz;k++){
			for(j=0;j<Y+yy;j++){
				for(i=0;i<X+xx;i++){
          double xp = (i*dx)-0.5*xx*dx;
          double yp = (j*dx)-0.5*yy*dx;
          double zp = (k*dx)-0.5*zz*dx;
					if(((properties[0]-(xp))*(properties[0]-(xp))+(properties[1]-(yp))*(properties[1]-(yp))+(properties[2]-(zp))*(properties[2]-(zp)))<=properties[3]*properties[3]){
						e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
					}
				}
			}
		}
}

void set_rel_perm_one_proc(std::vector<double> &e_r,std::vector<double> &e_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz, int component){
	int i=0;
	int j=0;
	int k=0;
	int xx = 0;
	int yy = 0;
	int zz = 0;	
	int xxb = 0;
	int yyb = 0;
	int zzb = 0;	
	if(component==0){
		xx = lastx;
 	  xxb = 1;
	}
	else if(component==1){
		yy = lasty;
    yyb = 1;
	}
	else if(component==2){
		zz = lastz;
    zzb = 1;
	}
	#pragma omp parallel for default(shared) private(i,j,k)
	for(i=0;i<point_per_proc_x+xx;i++){
		for(j=0;j<point_per_proc_y+yy;j++){
			for(k=0;k<point_per_proc_z+zz;k++){
				e_r[(i)*(point_per_proc_y+yy)*(point_per_proc_z+zz)+(k)*(point_per_proc_y+yy)+(j)] = e_r_tot[(i+i_min_proc)*(Ny+yyb)*(Nz+zzb)+(k+k_min_proc)*(Ny+yyb)+(j+j_min_proc)];
			}
		}
	}
}

void rotate_rel_perm(std::vector<double> &e_r_tot,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,int component){
	double xx = 0;
	double yy = 0;
	double zz = 0;	
	if(component==0){
		xx = 1;
	}
	else if(component==1){
		yy = 1;
	}
	else if(component==2){
		zz = 1;
	}
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

	//#pragma omp parallel for default(shared) private(i,j,k,x_after,y_after,z_after,x_before,y_before,xc,yc,zc,r,l)
	for(i=0;i<Nx+xx;i++){
		for(j=0;j<Ny+yy;j++){
			for(k=0;k<Nz+zz;k++){
				e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = 1;
				// Coordinate after rotation
				x_after = i*dx-0.5*xx*dx;
				y_after = j*dx-0.5*yy*dx;
				z_after = k*dx-0.5*zz*dx;
				// Coordinate before rotation
				x_before = xrot + (x_after-xrot)*cos(theta) + (y_after-yrot)*sin(theta);
				y_before = yrot - (x_after-xrot)*sin(theta)+(y_after-yrot)*cos(theta);

				for(object=0;object<ncube;object++){//	Cube
					if(((x_before)<=info_cube[7*object+3]+info_cube[7*object+0]/2)&&((x_before)>=info_cube[7*object+3]-info_cube[7*object+0]/2)&&((y_before)<=info_cube[7*object+4]+info_cube[7*object+1]/2)&&((y_before)>=info_cube[7*object+4]-info_cube[7*object+1]/2)&&((z_after)<=info_cube[7*object+5]+info_cube[7*object+2]/2)&&((z_after)>=info_cube[7*object+5]-info_cube[7*object+2]/2)){
						e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cube[7*object+6]];
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
              e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
						}
					}
					else if(info_cylinder[7*object]==1){
						if(((x_before-xc)*(x_before-xc)+(z_after-zc)*(z_after-zc)<=r*r) && y_before<=yc+l/2 && y_before>= yc-l/2){
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
						}
					}
					else if(info_cylinder[7*object]==2){						
						if(((x_before-xc)*(x_before-xc)+(y_before-yc)*(y_before-yc)<=r*r) && z_after<=zc+l/2 && z_after>= zc-l/2){
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
						}
					}
				}

				for(object=0;object<nsphere;object++){//	Sphere
					if(((info_sphere[5*object+0]-x_before)*(info_sphere[5*object+0]-x_before)+(info_sphere[5*object+1]-y_before)*(info_sphere[5*object+1]-y_before)+(info_sphere[5*object+2]-z_after)*(info_sphere[5*object+2]-z_after))<=info_sphere[5*object+3]*info_sphere[5*object+3]){
            e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_sphere[5*object+4]];
					}
				}				
			}
		}
	}
}
void set_vec(std::vector<double> &vec, int nbp, double val){
  int i;
  for(i=0;i<nbp;i++){
    vec[i] = val;
  }
}

void export_coupe(int direction, int component, double pos1,double pos2,int Nx_tot,int Ny_tot,int Nz_tot,double***M,double dx,int step,int myrank, int i_min,int i_max,int j_min ,int j_max ,int k_min ,int k_max ,int Nx,int Ny,int Nz,int lastx,int lasty,int lastz){
  double xx = 0;
	double yy = 0;
	double zz = 0;	
	char file_name[50] = "Cut_";  
  if(component==1){
    strcat(file_name,"Ex");
  }
  else if(component==2){
    strcat(file_name,"Ey");		
  }
  else if(component==3){
    strcat(file_name,"Ez");
  }
  else if(component==4){
    strcat(file_name,"Hx");
  }
  else if(component==5){
    strcat(file_name,"Hy");
  }
  else if(component==6){
    strcat(file_name,"Hz");
  }  
	if(component==1||component==5||component==6){
		xx = 1;
	}
	else if(component==2||component==4||component==6){
		yy = 1;
	}
	else if(component==3||component==4||component==5){
		zz = 1;
	}
  double x_min = i_min*dx - 0.5*xx*dx;
  double y_min = j_min*dx - 0.5*yy*dx;
  double z_min = k_min*dx - 0.5*yy*dx;
  double x_max = (i_max+xx*lastx)*dx - 0.5*xx*dx;
  double y_max = (j_max+yy*lasty)*dx - 0.5*yy*dx;
  double z_max = (k_max+zz*lastz)*dx - 0.5*yy*dx;
  
  double pos1_temp = pos1/dx ;
  double pos2_temp = pos2/dx ;
  int pos1_int = (int) pos1_temp;
  int pos2_int = (int) pos2_temp;
  int i =0;
  char stepnumber[20];
  char rank[20];
  snprintf(stepnumber,sizeof stepnumber,"%d", step);
  snprintf(rank,sizeof rank,"%d", myrank);
	FILE *FileW;    	
	if(direction==1){// Cut along x	
  	  if(pos1>=y_min && pos1<=y_max && pos2>=z_min && pos2<=z_max){
  		strcat(file_name,"_alongX_step");
	  	strcat(file_name,stepnumber);
      		strcat(file_name,"_rank");
		strcat(file_name,rank);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");		
      		pos1_int = pos1_int - y_min;
      		pos2_int = pos2_int - z_min;
		  for(i=0;i<Nx+xx*lastx;i++){
		  	fprintf(FileW," %lf \n ",M[i][pos1_int][pos2_int]);
		  }
      		fclose(FileW);  
    	  }
	}
	else if(direction==2){// Cut along y
     if(pos1>=x_min && pos1<=x_max && pos2>=z_min && pos2<=z_max){
  		strcat(file_name,"_alongY_step");
	  	strcat(file_name,stepnumber);
      		strcat(file_name,"_rank");
		strcat(file_name,rank);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");
  		pos1_int = pos1_int - x_min;
    		pos2_int = pos2_int - z_min;
		for(i=0;i<Ny+yy*lasty;i++){
			 fprintf(FileW," %lf \n ",M[pos1_int][i][pos2_int]);
		}
    		fclose(FileW);
    }
	}
	else if(direction==3){// Cut along z	
     if(pos1>=x_min && pos1<=x_max && pos2>=y_min && pos2<=y_max){
  		strcat(file_name,"_alongZ_step");
	  	strcat(file_name,stepnumber);
     		strcat(file_name,"_rank");
		strcat(file_name,rank);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");
     		pos1_int = pos1_int - x_min;
      		pos2_int = pos2_int - y_min;
		  for(i=0;i<Nz+zz*lastz;i++){
		  	fprintf(FileW," %lf \n ",M[pos1_int][pos2_int][i]);
		  }
     		fclose(FileW);
    }
	}
}
void export_power_thermo(std::vector<double> &Power_tot,int Nx,int Ny,int Nz){
   int i = 0;
   int j = 0;
   int k = 0;
   FILE *FileW;
   FileW = fopen("Power_thermo.dat","w");
   for(i=0;i<Nx;i++){
     for(j=0;j<Ny;j++){
       for(k=0;k<Nz;k++){
         fprintf(FileW," %lf \n ",Power_tot[i+j*Nx+k*Nx*Ny]);
       }
     }
   }
   fclose(FileW);
 }
 
 
 
 
 
 
 
 
 
 
 
 
 int get_my_rank()
{
    int myid;
    int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    return myid;
}

void check_MUMPS(DMUMPS_STRUC_C &id)
{
    if (id.infog[0] < 0)
    {
        std::cout << "[" << get_my_rank() << "] MUMPS Error:\n";
        std::cout << "\tINFOG(1)=" << id.infog[0] << '\n';
        std::cout << "\tINFOG(2)=" << id.infog[1] << std::endl;
    }
}

void init_MUMPS(DMUMPS_STRUC_C &id)
{
    id.comm_fortran = -987654; //USE_COMM_WORLD;
    id.par = 1;                // 1=host involved in factorization phase
    id.sym = 0;                // 0=unsymmetric
    id.job = -1;
    std::cout << "[" << get_my_rank() << "] Init MUMPS package." << std::endl;
    dmumps_c(&id);
    check_MUMPS(id);
}

void end_MUMPS(DMUMPS_STRUC_C &id)
{
    id.job = -2;
    std::cout << "[" << get_my_rank() << "] Terminate MUMPS instance." << std::endl;
    dmumps_c(&id);
    check_MUMPS(id);
}

void solve_MUMPS(DMUMPS_STRUC_C &id, int step)
{
    
    id.ICNTL(1) = -1; // stream for error messages [def=6]
    id.ICNTL(2) = -1; // stream for diag printing, statistics, warnings [def=0]
    id.ICNTL(3) = -1; // stream for global information [def=6]
    id.ICNTL(4) = 0;  // level of printing [def=2]
    
    // id.ICNTL(5)   // matrix input format
    // id.ICNTL(6)   // permutation/scaling
    // id.ICNTL(7)   // ordering
    // id.ICNTL(8)   // scaling strategy [def=auto]
    // id.ICNTL(9)   // use A or A^T [def=A]
    // id.ICNTL(10)  // iterative refinement [def=0=disabled]
    // id.ICNTL(11)  // compute statistics on error [def=0=disabled]
    // id.ICNTL(12)  // ordering strategy for sym matrices [def=0=auto]
    // id.ICNTL(13)  // parallelism of root node (scalapack) [def=0=parallel with scalapack]
    // id.ICNTL(14)  // % incr of working space [def=20=20%]
    // id.ICNTL(15-17)  // NOT USED
    // id.ICNTL(18)  // distributed input matrix [def=0=centralized]
    // id.ICNTL(19)  // Schur complement [def=0=no schur cplt]
    // id.ICNTL(20)  // format of rhs [def=0=dense]
    // id.ICNTL(21)  // distribution of solution vectors [def=0=centralized]
    // id.ICNTL(22)  // out-of-core [def=0=in-core]
    // id.ICNTL(23)  // max memory [def=0=estimate]
    // id.ICNTL(24)  // null pivot detectio [def=0=disabled]
    // id.ICNTL(25)  // solution for deficiant matrix [def=0=1 sol is returned]
    // id.ICNTL(26)  // [see schur cplt]
    // id.ICNTL(27)  // blocking size for multiple rhs [def=-32=auto]
    // id.ICNTL(28)  // parallel ordering [def=0=auto]
    // id.ICNTL(29)  // parallel ordering method (if scotch/parmetis) [def=0=auto]
    // id.ICNTL(30)  // compute some A^-1 entries
    // id.ICNTL(31)  // keep factors [def=0=yes]
    // id.ICNTL(32)  // forward elimination during factorization [def=0=disabled]
    // id.ICNTL(33)  // compute det(A)
    // id.ICNTL(34)  // NOT USED
    // id.ICNTL(35)  // BLR factorization (def=0=disabled)

   // std::cout << "[" << get_my_rank()
   //           << "] Call the MUMPS package (analyse, factorization and solve)." << std::endl;
    if(step==1){
  	  id.job = 6;
     }
     else {
       id.job = 3;
     }
    dmumps_c(&id);
    check_MUMPS(id);
}
/**********************************************************************
 			HOST WORK	
 *********************************************************************/    
void host_work(DMUMPS_STRUC_C &id,double Lx,double Ly,double Lz,double delta_x,double delta_t,int step_max,double theta,int nb_source, std::vector<double> &prop_source,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0,int SR,std::vector<double> &Cut, std::vector<double> &Pos_cut, std::vector<double> &step_cut, double nb_probe, std::vector<double> &Pos_probe,int n_sphere,std::vector<double> &prop_sphere,int n_cylinder,std::vector<double> &prop_cylinder,int n_cube,std::vector<double> &prop_cube, double T_init_food,double h_air,double x_min_th,double y_min_th, double z_min_th,double dx_electro,int X_elec,int Y_elec,int Z_elec,std::vector<double> &Source_elec,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &constant,std::vector<double> &geometry,int step_pos,std::vector<double> &Temp,int thermo_domain){   
	 SPoints grid2;

    // setup grids

    grid2.o = Vec3d(10.0, 10.0, 10.0); // origin
    Vec3d L2(0.3, 0.3, 0.3);        // box dimensions

    
    grid2.np1 = Vec3i(0, 0, 0);    // first index
    grid2.np2 = Vec3i(X_elec-1, Y_elec-1, Z_elec-1); // last index

    grid2.dx = L2 / (grid2.np() - 1); // compute spacing

    grid2.scalars["Power"] = &Source_elec;


    SPoints grid;

    // setup grids

    grid.o = Vec3d(x_min_th+10.0, y_min_th+10.0, z_min_th+10.0); // origin
    Vec3d L(Lx, Ly, Lz);        // box dimensions
    int X = (int) (Lx/delta_x)+1;
    int Y = (int) (Ly/delta_x)+1;
    int Z = (int) (Lz/delta_x)+1;

    grid.np1 = Vec3i(0, 0, 0);    // first index
    grid.np2 = Vec3i(X-1, Y-1, Z-1); // last index

    int i = 0;
    int i_vec=0;
    int count=0;
    int col=0;

    grid.dx = L / (grid.np() - 1); // compute spacing

    int nbp = grid.nbp();

    double theta_angle = 0;
    // Declaration of MUMPS variable
    MUMPS_INT n = X*Y*Z;
    std::vector<MUMPS_INT> irn;
    std::vector<MUMPS_INT> jcn;
    std::vector<double> a;
    std::vector<double> b;
    
    // Variables used to solve the system
    std::vector<double> T_init(n);// Suppress this when we have shown numerical diffisivity    
    std::vector<double> Temp2(n);    
    std::vector<double> Source(n);
    std::vector<double> Source_init(n);
    for(i=0;i<n;i++){
    	Source[i] = 0;
    }
    grid.scalars["Geometry"] = &geometry;
    grid.scalars["Temp"] = &Temp;
    grid.scalars["Source"] = &Source;
    grid.scalars["Rho"] = &rho;
    grid.scalars["cp"] = &cp;
    

    //Declaration and Initialization of variables managing the boundaries. 
    std::vector<int> ip_h(n);
    std::vector<int> jp_h(n);
    std::vector<int> kp_h(n);
    std::vector<int> lastx_h(n);
    std::vector<int> lasty_h(n);
    std::vector<int> lastz_h(n);  

    // Variable use for probing the temperature
    if(nb_probe!=0){
    	std::vector<double> probe(nb_probe*step_max);
    }
    std::vector<double> probe(nb_probe*step_max);
	#pragma omp parallel for default(shared) private(i_vec)
	for(i_vec=0;i_vec<X*Y*Z;i_vec++){
		ip_h[i_vec] = (i_vec/(Y*Z));
		jp_h[i_vec] = (i_vec%(Y*Z))%Y;
		kp_h[i_vec] = (i_vec%(Y*Z))/Y;
		lastx_h[i_vec] = 0;
		lasty_h[i_vec]= 0;
		lastz_h[i_vec] = 0;
		if(X == 1){	
			lastx_h[i_vec] = 1;
		}
		else{
			lastx_h[i_vec] = ip_h[i_vec]/(X-1);
		}	
		if(Y == 1){	
			lasty_h[i_vec] = 1;
		}
		else{
			lasty_h[i_vec] = jp_h[i_vec]/(Y-1);
		}	
		if(Z == 1){	
			lastz_h[i_vec] = 1;
		}
		else{
			lastz_h[i_vec] = kp_h[i_vec]/(Z-1);
		}
	}

     // Source from electro calculation
  set_source_from_elec(Source,Source_elec,x_min_th,y_min_th,z_min_th,delta_x,dx_electro,X,Y,Z,X_elec,Y_elec,Z_elec);
  
  // Insertion of one or more power source inside the domain (Will disappear when coupling is done)
  if(nb_source!=0){
  	insert_Source_th(Source,nb_source, prop_source, X,Y,Z, delta_x, rho, cp,x_min_th,y_min_th,z_min_th);
  }

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<n;i++){
    	Source_init[i] = Source[i];
    }
	

  // Computation of the matrix and of the initial temperature
  if(thermo_domain==0){
  Compute_a_T0(irn ,jcn, X, Y, Z, ip_h, jp_h, kp_h, lastx_h, lasty_h, lastz_h, a, b,Temp,constant,BC,T_Dir,T_0,theta,k_heat_x,k_heat_y,k_heat_z);		// Old Boundary Conditions
  }
  else{
  Compute_a_T0_2(irn ,jcn,  X,  Y,  Z,ip_h,jp_h,kp_h,lastx_h,lasty_h,lastz_h, a, b,Temp,constant,BC,T_Dir, T_0,theta, k_heat_x,k_heat_y,k_heat_z,geometry,delta_x,h_air);		// New boundary conditions  
  }
  MUMPS_INT8 nnz =  irn.size();
  //set_T0(Temp,geometry,T_0,T_init_food,  X,  Y,  Z );
    
    // Preparation of MUMPS job
    id.n = n;
    id.nnz = a.size();
    id.irn = &irn[0];
    id.jcn = &jcn[0];
    id.a = &a[0];
    int step = 1;

    // save results to disk
    export_spoints_XML("Temperature_field", step+step_pos*step_max, grid, grid, Zip::ZIPPED, X, Y, Z, 1);
    // The value at the probe is registered
	for(i=0;i<nb_probe;i++){	
		int nx = (int) Pos_probe[i*3];
		int ny = (int) Pos_probe[i*3+1];
		int nz = (int) Pos_probe[i*3+2];
		probe[step_max*i] = Temp[ny+nz*Y+nx*Y*Z];
	}

    int next_cut=0;
    while(step<step_max){
	//Computation of the right hand side
	Compute_RHS(b,irn,jcn,Temp,Source,Temp2,X,Y,Z, nnz, rho, cp,geometry,delta_t,thermo_domain); 
    	
	// Resolution of the system
	id.rhs = &Temp[0];
    	solve_MUMPS(id,step);
	
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i] = id.rhs[i];
	}

	// Extraction of a cut if needed
	if(step == step_cut[next_cut]){// To extract a cut
		next_cut++;
		if(Cut[0]==1){
			export_coupe_th(1, Pos_cut[0], Pos_cut[1], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
		}
		if(Cut[1]==1){
			export_coupe_th(2, Pos_cut[2], Pos_cut[3], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
		}
		if(Cut[2]==1){
			export_coupe_th(3, Pos_cut[4], Pos_cut[5], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
		}
				
	}
	

       // The value at the probe is registered
	for(i=0;i<nb_probe;i++){
		int nx = (int) Pos_probe[i*3];
		int ny = (int) Pos_probe[i*3+1];
		int nz = (int) Pos_probe[i*3+2];
		probe[step+step_max*i] = Temp[ny+nz*Y+nx*Y*Z];
	}
  	step++;
        // Save results to disk if needed	
    	if(step%SR==0){
    		export_spoints_XML("Temperature_field", step+step_pos*step_max, grid, grid, Zip::ZIPPED, X, Y,  Z, 1);
		export_spoints_XML("Test_power", step+step_pos*step_max, grid2, grid2, Zip::ZIPPED, X_elec, Y_elec,  Z_elec, 1);

		/************* To be suppress when coupling is done ****************/

		// Power rotation		
		/*theta_angle = theta_angle + 3.141692/8;
		rotate_Power_grid_th(Source_init,Source,X, Y, Z, Lx, Ly, Lz, delta_x, theta_angle);*/
		/******************************************************************/
				
  	  }		
    }
   /***************************  To be suppress when coupling is done **************/
   /* while(step<60){step++;			// Temperature Rotation
    	// Temperature rotation
		for(i=0;i<X*Y*Z;i++){
			T_init[i] = Temp[i];
		}
		export_spoints_XML("laplace", step, grid, grid, Zip::ZIPPED, X, Y,  Z, 1);
		theta_angle = 3.141692/8;
		rotate_T_grid(T_init,Temp,X,Y, Z,  Lx,  Ly,  Lz, delta_x, theta_angle, T_0);		
		step++;
    }*/
   /********************************************************************************/
    export_probe_th(nb_probe , probe, step_max,step_pos);      
}

void slave_work(DMUMPS_STRUC_C &id, int step_max){
 int step = 1;
    while(step<step_max){
	//if(step == 1){
   	 solve_MUMPS(id,step);  
	 step++;
	//}
    }
}


/**********************************************************************
 			Main	
 *********************************************************************/

void main_th(std::vector<double> &Source_elec, std::vector<double> &Temperature, std::vector<int> &BC, std::vector<double> &T_Dir, double T_0,double dx,double h_air, double Lx, double Ly, double Lz, double dt, int step_max, int nb_source, int SR, double theta, int n_sphere, int n_cylinder, int n_cube, std::vector<double> &prop_sphere, std::vector<double> &prop_cylinder, std::vector<double> &prop_cube, double T_food_init, double x_min_th, double y_min_th, double z_min_th, double dx_electro, double Lx_electro, double Ly_electro, double Lz_electro, int prop_per_source, std::vector<double> &prop_source, std::vector<double> &Cut, std::vector<double> &Pos_cut, int N_cut,std::vector<double> &step_cut, int nb_probe, std::vector<double> &Pos_probe, DMUMPS_STRUC_C &id,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &constant,std::vector<double> &geometry,int step_pos,int thermo_domain){
	int X_electro = (int) (Lx_electro/dx_electro)+1;
    	int Y_electro = (int) (Ly_electro/dx_electro)+1;
    	int Z_electro = (int) (Lz_electro/dx_electro)+1;
    // split work among processes
    if (get_my_rank() == 0)
        host_work(id, Lx, Ly, Lz, dx, dt, step_max,theta, nb_source, prop_source,BC,T_Dir,T_0,SR,Cut,Pos_cut,step_cut,nb_probe,Pos_probe,n_sphere,prop_sphere,n_cylinder,prop_cylinder,n_cube,prop_cube,  T_food_init,h_air, x_min_th, y_min_th, z_min_th,dx_electro, X_electro, Y_electro, Z_electro,Source_elec,k_heat_x,k_heat_y,k_heat_z,rho,cp,vec_k,vec_rho,vec_cp,constant,geometry,step_pos,Temperature,thermo_domain);
    else
        slave_work(id,step_max);
}

// This function computes the right hand side of the system to be solved
void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain){
	int i = 0;
  double T_inf = 20;
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp2[i]=0;
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<nnz;i++){
		Temp2[irn[i]-1]+=pre_mat[i]*Temp[jcn[i]-1];
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
    if(geometry[i]!=0&&(geometry[i-Y*Z]==0||geometry[i+Y*Z]==0||geometry[i-1]==0||geometry[i+1]==0||geometry[i-Y]==0||geometry[i+Y]==0)&& (thermo_domain==1)){// Neuman
      Temp2[i] = -T_inf;
    }
    else if(geometry[i]!=0 || thermo_domain==0){
		  Temp2[i]+=(dt*Source[i])/(rho[i]*cp[i]);		
    }
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i]=Temp2[i];
	}
}

// This function imposes the boundary conditions, computes the A matrix and set an initial temperature over the domain (Old version)
void Compute_a_T0(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z){
	int i_vec = 0;
	int i=0;
	int j = 0;
	int k =0;
	for(i_vec=0;i_vec<X*Y*Z;i_vec++){
  		Temp[i_vec] = T_0;	
 	 }
      for(j=0;j<Y;j++){		
		for(k=0;k<Z;k++){
			for(i=0;i<X;i++){	
				i_vec = i*Y*Z+k*Y+j;
		      		if(jp_h[i_vec]==0){
	  		if(BC[0]==1){ //Dirichlet
			  	Temp[i_vec] = T_Dir[0];
	  		}
		}

		if(lasty_h[i_vec]==1){
			if(BC[1]==1){ //Dirichlet
				Temp[i_vec] = T_Dir[1];
			}
		}
		if(kp_h[i_vec]==0){
			if(BC[2]==1){ //Dirichlet
				Temp[i_vec] = T_Dir[2];
			}
		}
		if(lastz_h[i_vec]==1){
			if(BC[3]==1){ //Dirichlet
				Temp[i_vec] = T_Dir[3];
			}
		}
		if(ip_h[i_vec]==0){
			if(BC[4]==1){ //Dirichlet
				Temp[i_vec] = T_Dir[4];
			}
		}
		if(lastx_h[i_vec]==1){
			if(BC[5]==1){ //Dirichlet
				Temp[i_vec] = T_Dir[5];
			}

		}
      
		if(jp_h[i_vec]==0){
			if(BC[0]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1+1);
				a.push_back(1);
				b.push_back(1);
			}
		}

		else if(lasty_h[i_vec]==1){
			if(BC[1]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1-1);
				a.push_back(1);
				b.push_back(1);        
			}

		}
		else if(kp_h[i_vec]==0){
			if(BC[2]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1+Y);
				a.push_back(1);
				b.push_back(1);  

			}
		}
		else if(lastz_h[i_vec]==1){
			if(BC[3]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1-Y);
				a.push_back(1);
				b.push_back(1);        
			}
		}
		else if(ip_h[i_vec]==0){
			if(BC[4]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1+Y*Z);
				a.push_back(1);
				b.push_back(1);        
			}
		}
		else if(lastx_h[i_vec]==1){
			if(BC[5]==1){ //Dirichlet
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(1);
				b.push_back(1);
			}
			else{  // Neuman
				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1);
				a.push_back(-1);
				b.push_back(-1);

				irn.push_back(i_vec+1);
				jcn.push_back(i_vec+1-Y*Z);
				a.push_back(1);
				b.push_back(1);        
			}
		}
				else{   // Inside of the domain
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(1+theta*(constant[i_vec]*(k_heat_x[i*Y*Z+k*Y+j]+k_heat_x[(i+1)*Y*Z+k*Y+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]+k_heat_z[i*Y*(Z+1)+k*Y+j]+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j])));	// I- A(theta)
					b.push_back(1-(1-theta)*(constant[i_vec]*(k_heat_x[i*Y*Z+k*Y+j]+k_heat_x[(i+1)*Y*Z+k*Y+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]+k_heat_z[i*Y*(Z+1)+k*Y+j]+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j])));	// I + A(1-theta)				

					irn.push_back(i_vec+1);
					jcn.push_back(i_vec);
					a.push_back(-theta*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]);		// I+ A(1-theta)
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+2);
					a.push_back(-theta*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]);		// I+ A(1-theta)
					
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y+1);
					a.push_back(-theta*constant[i_vec]*k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]);		// I+ A(1-theta)
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y+1);
					a.push_back(-theta*constant[i_vec]*k_heat_z[i*Y*(Z+1)+k*Y+j]);		// I- A(theta)		
					b.push_back((1-theta)*constant[i_vec]*k_heat_z[i*Y*(Z+1)+k*Y+j]);		// I+ A(1-theta)
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y*Z+1);
					a.push_back(-theta*constant[i_vec]*k_heat_x[(i+1)*Y*Z+k*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_x[(i+1)*Y*Z+k*Y+j]);		// I+ A(1-theta)
					
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y*Z+1);
					a.push_back(-theta*constant[i_vec]*k_heat_x[i*Y*Z+k*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_x[i*Y*Z+k*Y+j]);      	// I+ A(1-theta)
					
				}									

			} 
		}
	}
}


// This function inserts one or more objects inside the domain (No longer used) 
void insert_obj_th(std::vector<double> &temp, std::vector<double> &k_heat_x, std::vector<double> &k_heat_y, std::vector<double> &k_heat_z, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &properties, int X,int Y,int Z, double dx,std::vector<double> &geometry){
  	int i = 0;
  	int j = 0;
  	int k = 0;
  	int l = 0;
	int prop_per_obj = 10;
	for(l=0;l<nb_obj;l++){
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z;k++){
					if(((i*dx)<=properties[prop_per_obj*l+3]+properties[prop_per_obj*l+0]/2)&&((i*dx)>=properties[prop_per_obj*l+3]-properties[prop_per_obj*l+0]/2)&&((j*dx)<=properties[prop_per_obj*l+4]+properties[prop_per_obj*l+1]/2)&&((j*dx)>=properties[prop_per_obj*l+4]-properties[prop_per_obj*l+1]/2)&&((k*dx)<=properties[prop_per_obj*l+5]+properties[prop_per_obj*l+2]/2)&&((k*dx)>=properties[prop_per_obj*l+5]-properties[prop_per_obj*l+2]/2)){
						temp[i*Y*Z+j+k*Y]= properties[prop_per_obj*l+6];
						rho[i*Y*Z+j+k*Y]= properties[prop_per_obj*l+8];
						cp[i*Y*Z+j+k*Y]= properties[prop_per_obj*l+9];
						geometry[i*Y*Z+j+k*Y]= 1;
					}
					else{
						geometry[i*Y*Z+j+k*Y]= 0;
					}						
				}
			}		
		}	
  set_kheat(0, X, Y, Z, properties, l, dx, k_heat_x);
  set_kheat(1, X, Y, Z, properties, l, dx, k_heat_y);
  set_kheat(2, X, Y, Z, properties, l, dx, k_heat_z);
	}	
}

// This function inserts one or more sources inside the domain
void insert_Source_th(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp,double x_min_th,double y_min_th,double z_min_th){
	int i = 0;
  	int j = 0;
  	int k = 0;
	int l = 0;
	int prop_per_source = 7;
	
	for(l=0;l<nb_source;l++){
		double n_x_double = ((prop_source[prop_per_source*l])/dx)+1;
		int n_x = (int) n_x_double;
		double pos_x = ((prop_source[prop_per_source*l+3]-x_min_th)/dx);	
		int i_min = (int)pos_x;
		i_min = i_min - (n_x/2);
		int i_max = i_min + n_x-1;
	
		double n_y_double = ((prop_source[prop_per_source*l+1])/dx)+1;
		int n_y = (int) n_y_double;
		double pos_y = ((prop_source[prop_per_source*l+4]-y_min_th)/dx);	
		int j_min = (int)pos_y;
		j_min = j_min - (n_y/2);
		int j_max = j_min + n_y-1;
	
		double n_z_double = ((prop_source[prop_per_source*l+2])/dx)+1;
		int n_z = (int) n_z_double;
		double pos_z = ((prop_source[prop_per_source*l+5]-z_min_th)/dx);	
		int k_min = (int)pos_z;
		k_min = k_min - (n_z/2);
		int k_max = k_min + n_z-1;
	
		int b_inf_x =0;
		int b_inf_y =0;
		int b_inf_z =0;
		int b_sup_x = 0;
		int b_sup_y = 0;
		int b_sup_z = 0;

		b_inf_x = i_min;				
		b_inf_y = j_min;
		b_inf_z = k_min;
		b_sup_x = i_max;
		b_sup_y = j_max;
		b_sup_z = k_max;
		if(b_inf_x<0){
			b_inf_x = 0;		
		}
		if(b_inf_y<0){
			b_inf_y = 0;		
		}
		if(b_inf_z<0){
			b_inf_z = 0;		
		}
		if(X-1<b_sup_x){
			b_sup_x = X-1;
		}
		if(Y-1<b_sup_y){
			b_sup_y = Y-1;
		}
		if(Z-1<b_sup_z){
			b_sup_z = Z-1;
		}
		#pragma omp parallel for default(shared) private(i,j,k)
		for(i=b_inf_x;i<=b_sup_x;i++){
			for(j=b_inf_y;j<=b_sup_y;j++){
				for(k=b_inf_z;k<=b_sup_z;k++){
					Source[i*Y*Z+j+k*Y]= prop_source[prop_per_source*l+6];					
				}				
			}
		}
	}
}

// This function export the value of the temperature on a line inside the domain directed along a given direction 
void export_coupe_th(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,	double dx, int step, double x_min_th, double y_min_th, double z_min_th){
	int i =0;
	char file_name[50] = "Cut";
	char stepnumber[20];
	snprintf(stepnumber,sizeof stepnumber,"%d", step);
	FILE *FileW;    	
	if(direction==1){// Cut along x	
		pos1 = (pos1-y_min_th)/dx ;
		pos2 = (pos2-z_min_th)/dx ;
		int pos1_int = (int) pos1;
		int pos2_int = (int) pos2;
		strcat(file_name,"_alongX_step");
		strcat(file_name,stepnumber);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");		
		for(i=0;i<Nx;i++){
			fprintf(FileW," %lf \n ",temp[pos1_int+pos2_int*Ny+i*Ny*Nz]);
		}
	}
	else if(direction==2){// Cut along y
		pos1 = (pos1-x_min_th)/dx ;
		pos2 = (pos2-z_min_th)/dx ;
		int pos1_int = (int) pos1;
		int pos2_int = (int) pos2;
		strcat(file_name,"_alongY_step");
		strcat(file_name,stepnumber);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");
		for(i=0;i<Ny;i++){
			fprintf(FileW," %lf \n ",temp[i+pos2_int*Ny+pos1_int*Ny*Nz]);
		}
	}
	else if(direction==3){// Cut along z	
		pos1 = (pos1-x_min_th)/dx ;
		pos2 = (pos2-y_min_th)/dx ;
		int pos1_int = (int) pos1;
		int pos2_int = (int) pos2;
		strcat(file_name,"_alongZ_step");
		strcat(file_name,stepnumber);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");
		for(i=0;i<Nz;i++){
			fprintf(FileW," %lf \n ",temp[pos2_int+i*Ny+pos1_int*Ny*Nz]);
		}
	}
	fclose(FileW);
}

// This function export the value of the temperature as a function of time for a given position
void export_probe_th(double nb_probe , std::vector<double> &probe,int step_max,int step_pos){
  int i = 0;
  int j = 0;
  FILE *FileW;
  for(i=0;i<nb_probe;i++){
    	  char file_name[50] = "Probe";
	  char probenumber[20];
	  char step_minc[20];
	  char step_maxc[20];
	  snprintf(step_minc,sizeof step_minc,"%d", step_pos*step_max);
	  snprintf(step_maxc,sizeof step_maxc,"%d", (step_pos+1)*step_max);
	  snprintf(probenumber,sizeof probenumber,"%d", (i+1));
    	  strcat(file_name,probenumber);
	  strcat(file_name,"_step");
	  strcat(file_name,step_minc);	
	  strcat(file_name,"_to_step");
	  strcat(file_name,step_maxc);
	  strcat(file_name,".txt");
    FileW = fopen(file_name,"w");
    for(j=0;j<step_max;j++){
      fprintf(FileW," %lf \n ",probe[step_max*i+j]);
    }
    fclose(FileW);
  }
}

// This function set the value of the heat conductivity of a given object inside the domain (No longer used)
void set_kheat(int Case,int X,int Y,int Z, std::vector<double> &properties,int l,double dx,std::vector<double> &k_heat){
  int i = 0;
  int j = 0;
  int k = 0;
  int xx=0;
  int yy=0;
  int zz=0; 
  int prop_per_obj = 10;
  if(Case==0){
    xx++;
  }
  else if(Case==1){
    yy++;
  }
  else if(Case==2){
    zz++;
  }
  for(i=0;i<X+xx;i++){
		for(j=0;j<Y+yy;j++){
			for(k=0;k<Z+zz;k++){
				if(((i*dx-0.5*xx)<=properties[prop_per_obj*l+3]+properties[prop_per_obj*l+0]/2)&&((i*dx-0.5*xx)>=properties[prop_per_obj*l+3]-properties[prop_per_obj*l+0]/2)&&((j*dx-0.5*yy)<=properties[prop_per_obj*l+4]+properties[prop_per_obj*l+1]/2)&&((j*dx-0.5*yy)>=properties[prop_per_obj*l+4]-properties[prop_per_obj*l+1]/2)&&((k*dx-0.5*zz)<=properties[prop_per_obj*l+5]+properties[prop_per_obj*l+2]/2)&&((k*dx-0.5*zz)>=properties[prop_per_obj*l+5]-properties[prop_per_obj*l+2]/2)){
					k_heat[i*(Y+yy)*(Z+zz)+j+k*(Y+yy)]= properties[prop_per_obj*l+7];
				}
				}
			}
		}
}

// This function compute the A matrix and set the temperature everywhere at the temperature of the air
void Compute_a_T0_2(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h){
	int i_vec = 0;
	int i=0;
	int j = 0;
	int k =0;
	double T_inf = 20;

	for(j=0;j<Y;j++){		
		for(k=0;k<Z;k++){
			for(i=0;i<X;i++){	
				i_vec = i*Y*Z+k*Y+j;
				if(geometry[i_vec]==0){
					Temp[i_vec] = T_inf;
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(1);
					b.push_back(1);
				}
				else if(geometry[i_vec-Y*Z]==0&&geometry[i_vec+Y*Z]!=0){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_x[i_vec+Y*Z]/(h*dx));
					b.push_back(-1-k_heat_x[i_vec+Y*Z]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1+Y*Z);
					a.push_back(+k_heat_x[i_vec+Y*Z]/(h*dx));
					b.push_back(+k_heat_x[i_vec+Y*Z]/(h*dx));
				}
				else if((geometry[i_vec+Y*Z]==0&&geometry[i_vec-Y*Z]!=0)){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_x[i_vec]/(h*dx));
					b.push_back(-1-k_heat_x[i_vec]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1-Y*Z);
					a.push_back(+k_heat_x[i_vec]/(h*dx));
					b.push_back(+k_heat_x[i_vec]/(h*dx));
				}
				else if(geometry[i_vec-1]==0&&geometry[i_vec+1]!=0){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]/(h*dx));
					b.push_back(-1-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1+1);
					a.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]/(h*dx));
					b.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]/(h*dx));
					}
				else if((geometry[i_vec+1]==0 && geometry[i_vec-1]!=0)){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]/(h*dx));
					b.push_back(-1-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1-1);
					a.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]/(h*dx));
					b.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]/(h*dx));
				}
				else if(geometry[i_vec-Y]==0&&geometry[i_vec+Y]!=0){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]/(h*dx));
					b.push_back(-1-k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1+Y);
					a.push_back(+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]/(h*dx));
					b.push_back(+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]/(h*dx));
				}
				else if((geometry[i_vec+Y]==0&&geometry[i_vec-Y]!=0)){
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(-1-k_heat_z[i*Y*(Z+1)+(k)*Y+j]/(h*dx));
					b.push_back(-1-k_heat_z[i*Y*(Z+1)+(k)*Y+j]/(h*dx));
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1-Y);
					a.push_back(+k_heat_z[i*Y*(Z+1)+(k)*Y+j]/(h*dx));
					b.push_back(+k_heat_z[i*Y*(Z+1)+(k)*Y+j]/(h*dx));
				}
					else{// Inside of the domain
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back(1+theta*(constant[i_vec]*(k_heat_x[i*Y*Z+k*Y+j]+k_heat_x[(i+1)*Y*Z+k*Y+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]+k_heat_z[i*Y*(Z+1)+k*Y+j]+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j])));	// I- A(theta)
					b.push_back(1-(1-theta)*(constant[i_vec]*(k_heat_x[i*Y*Z+k*Y+j]+k_heat_x[(i+1)*Y*Z+k*Y+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]+k_heat_z[i*Y*(Z+1)+k*Y+j]+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j])));	// I + A(1-theta)				

					irn.push_back(i_vec+1);
					jcn.push_back(i_vec);
					a.push_back(-theta*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]);		// I+ A(1-theta)
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+2);
					a.push_back(-theta*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]);		// I+ A(1-theta)
					
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y+1);
					a.push_back(-theta*constant[i_vec]*k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]);		// I+ A(1-theta)
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y+1);
					a.push_back(-theta*constant[i_vec]*k_heat_z[i*Y*(Z+1)+k*Y+j]);		// I- A(theta)		
					b.push_back((1-theta)*constant[i_vec]*k_heat_z[i*Y*(Z+1)+k*Y+j]);		// I+ A(1-theta)
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y*Z+1);
					a.push_back(-theta*constant[i_vec]*k_heat_x[(i+1)*Y*Z+k*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_x[(i+1)*Y*Z+k*Y+j]);		// I+ A(1-theta)
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y*Z+1);
					a.push_back(-theta*constant[i_vec]*k_heat_x[i*Y*Z+k*Y+j]);		// I- A(theta)
					b.push_back((1-theta)*constant[i_vec]*k_heat_x[i*Y*Z+k*Y+j]);      	// I+ A(1-theta)
				}
			}
		}
	}
}

// This function can place different geometry of objects inside the domain (Need to be parametrized!!)
void place_geometry_th(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &k_heatx,std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,std::vector<double> &cp, double x_min_th, double y_min_th, double z_min_th){	
	int i=0;
	int j=0;
	int k=0;

	if(P==2){
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z;k++){
					if(((x_min_th+(i*dx))<=properties[3]+properties[0]/2)&&((x_min_th+i*dx)>=properties[3]-properties[0]/2)&&((y_min_th+j*dx)<=properties[4]+properties[1]/2)&&((y_min_th+j*dx)>=properties[4]-properties[1]/2)&&((z_min_th+k*dx)<=properties[5]+properties[2]/2)&&((z_min_th+k*dx)>=properties[5]-properties[2]/2)){
						geometry[i*Y*Z+k*Y+j]=val;
						rho[i*Y*Z+k*Y+j] = vec_rho[val];
						cp[i*Y*Z+k*Y+j] = vec_cp[val];
					}
				}
			}
		}
		for(i=0;i<X+1;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z;k++){
					if(((x_min_th+i*dx)-0.5<=properties[3]+properties[0]/2)&&((x_min_th+i*dx)-0.5>=properties[3]-properties[0]/2)&&((y_min_th+j*dx)<=properties[4]+properties[1]/2)&&((y_min_th+j*dx)>=properties[4]-properties[1]/2)&&((z_min_th+k*dx)<=properties[5]+properties[2]/2)&&((z_min_th+k*dx)>=properties[5]-properties[2]/2)){
						k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
					}
				}
			}
		}
		for(i=0;i<X;i++){
			for(j=0;j<Y+1;j++){
				for(k=0;k<Z;k++){
					if(((x_min_th+i*dx)<=properties[3]+properties[0]/2)&&((x_min_th+i*dx)>=properties[3]-properties[0]/2)&&((y_min_th+j*dx)-0.5<=properties[4]+properties[1]/2)&&((y_min_th+j*dx)-0.5>=properties[4]-properties[1]/2)&&((z_min_th+k*dx)<=properties[5]+properties[2]/2)&&((z_min_th+k*dx)>=properties[5]-properties[2]/2)){
						k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
					}
				}
			}
		}
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z+1;k++){
					if(((x_min_th+i*dx)<=properties[3]+properties[0]/2)&&((x_min_th+i*dx)>=properties[3]-properties[0]/2)&&((y_min_th+j*dx)<=properties[4]+properties[1]/2)&&((y_min_th+j*dx)>=properties[4]-properties[1]/2)&&((z_min_th+k*dx)-0.5<=properties[5]+properties[2]/2)&&((z_min_th+k*dx)-0.5>=properties[5]-properties[2]/2)){
						k_heatz[i*(Y)*(Z+1)+k*(Y)+j] = vec_k[val];
					}
				}
			}
		}





	}
	else if(P==1){
		double xc = properties[1];
		double yc = properties[2];
		double zc = properties[3];
		double r = properties[4];
		double l = properties[6];
		double xp;
		double yp;
		double zp;		
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					xp = x_min_th+i*dx;
					yp = y_min_th+j*dx;
					zp = z_min_th+k*dx;
					if(properties[0]==0){
						if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
							rho[i*Y*Z+k*Y+j] = vec_rho[val];
							cp[i*Y*Z+k*Y+j] = vec_cp[val];
						}
					}
					else if(properties[0]==1){
						if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
							rho[i*Y*Z+k*Y+j] = vec_rho[val];
							cp[i*Y*Z+k*Y+j] = vec_cp[val];
						}
					}
					else if(properties[0]==2){						
						if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
							geometry[i*Y*Z+k*Y+j]=val;
							rho[i*Y*Z+k*Y+j] = vec_rho[val];
							cp[i*Y*Z+k*Y+j] = vec_cp[val];
						}
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X+1;i++){
					xp = x_min_th+i*dx-0.5;
					yp = y_min_th+j*dx;
					zp = z_min_th+k*dx;
					if(properties[0]==0){
						if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
							k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
						}
					}
					else if(properties[0]==1){
						if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
							k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
						}
					}
					else if(properties[0]==2){						
						if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
							k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
						}
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y+1;j++){
				for(i=0;i<X;i++){
					xp = x_min_th+i*dx;
					yp = y_min_th+j*dx-0.5;
					zp = z_min_th+k*dx;
					if(properties[0]==0){
						if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
							k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
						}
					}
					else if(properties[0]==1){
						if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
							k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
						}
					}
					else if(properties[0]==2){						
						if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
							k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
						}
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X+1;i++){
					xp = x_min_th+i*dx;
					yp = x_min_th+j*dx;
					zp = z_min_th+k*dx+0.5;
					if(properties[0]==0){
						if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
							k_heatz[i*(Y)*(Z+1)+k*(Y)+j] = vec_k[val];
						}
					}
					else if(properties[0]==1){
						if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
							k_heatz[i*(Y)*(Z+1)+k*(Y)+j] = vec_k[val];
						}
					}
					else if(properties[0]==2){						
						if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
							k_heatz[i*(Y)*(Z+1)+k*(Y)+j] = vec_k[val];
						}
					}
				}
			}
		}

	}
	else if(P==0){
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-(x_min_th+i*dx))*(properties[0]-(x_min_th+i*dx))+(properties[1]-(y_min_th+j*dx))*(properties[1]-(y_min_th+j*dx))+(properties[2]-(z_min_th+k*dx))*(properties[2]-(z_min_th+k*dx)))<=properties[3]*properties[3]){
						geometry[i*Y*Z+k*Y+j]=val;
						rho[i*Y*Z+k*Y+j] = vec_rho[val];
						cp[i*Y*Z+k*Y+j] = vec_cp[val];
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X+1;i++){
					if(((properties[0]-(x_min_th+i*dx+0.5))*(properties[0]-(x_min_th+i*dx+0.5))+(properties[1]-(y_min_th+j*dx))*(properties[1]-(y_min_th+j*dx))+(properties[2]-(z_min_th+k*dx))*(properties[2]-(z_min_th+k*dx)))<=properties[3]*properties[3]){
						k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y+1;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-(x_min_th+i*dx))*(properties[0]-(x_min_th+i*dx))+(properties[1]-(y_min_th+j*dx+0.5))*(properties[1]-(y_min_th+j*dx+0.5))+(properties[2]-(z_min_th+k*dx))*(properties[2]-(z_min_th+k*dx)))<=properties[3]*properties[3]){
						k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
					}
				}
			}
		}
		for(k=0;k<Z+1;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-(x_min_th+i*dx))*(properties[0]-(x_min_th+i*dx))+(properties[1]-(y_min_th+j*dx))*(properties[1]-(y_min_th+j*dx))+(properties[2]-(z_min_th+k*dx+0.5))*(properties[2]-(z_min_th+k*dx+0.5)))<=properties[3]*properties[3]){
						k_heatz[i*(Y)*(Z+1)+k*(Y)+j] = vec_k[val];
					}
				}
			}
		}

	}
}

// This function set the initial temperature of the object at the one specified in the input file
void set_T0(std::vector<double> &Temp,std::vector<double> &geometry,double T_0,double T_init_food,int  X,int  Y,int  Z ){
	int i = 0;
	int j = 0;
 	int k = 0;
	for(i=0;i<X;i++){
		for(k=0;k<Z;k++){
			for(j=0;j<Y;j++){			
				if(geometry[i*(Y)*(Z)+k*(Y)+j]==0){
					Temp[i*(Y)*(Z)+k*(Y)+j]=T_0;
				}
				else{
					Temp[i*(Y)*(Z)+k*(Y)+j]=T_init_food;				
				}
			}
		}
	}
}

// This function rotate the power grid (won't be used anymore once the coupling is done)
void rotate_Power_grid_th(std::vector<double> &Source_init,std::vector<double> &Source_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta){
	int i = 0;
	int j = 0;
	int k = 0;
	double xc = Lx/2;
	double yc = Ly/2;
	for(i=0;i<Nx;i++){ 
		for(k=0;k<Nz;k++){
			for(j=0;j<Ny;j++){			
				double x_before = i*dx;
				double y_before = j*dx;
				double z_before = k*dx;
				double x_after = xc + (x_before-xc)*cos(theta) - (y_before-yc)*sin(theta);
				double y_after = yc + (x_before-xc)*sin(theta)+(y_before-yc)*cos(theta);
				double z_after = z_before;

				double point1 = 1;
				double point2 = 1;
				double point3 = 1;
				double point4 = 1;

				if(x_after<=0||y_after<=0||Lx<=x_after||Ly<=y_after){
					Source_curr[i*Ny*Nz+k*Ny+j] = 0;
				}
				else{
					double x1 = x_after/dx;
					double y1 = y_after/dx;
					int i1 = (int) x1;
					int j1 = (int) y1;
					x1 = i1*dx;
					y1 = j1*dx;
					double xi = (x_after-x1)/dx;
					double eta = (y_after-y1)/dx;
					Source_curr[i*Ny*Nz+k*Ny+j] = (1-xi)*(1-eta)*Source_init[(i1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1-eta)*Source_init[(i1+1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1+eta)*Source_init[(i1+1)*Ny*Nz+(k)*Ny+(j1+1)]+(1-xi)*(1+eta)*Source_init[(i1)*Ny*Nz+(k)*Ny+(j1+1)];
					Source_curr[i*Ny*Nz+k*Ny+j] = Source_curr[i*Ny*Nz+k*Ny+j]/4;
 
				}
			}
		}
	}

}

// This function rotate the temperature grid (will not be used because of different problems, see report)
void rotate_T_grid(std::vector<double> &T_init,std::vector<double> &T_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta, double T_air){
	int i = 0;
	int j = 0;
	int k = 0;
	double xc = Lx/2;
	double yc = Ly/2; 
	for(i=0;i<Nx;i++){
		for(k=0;k<Nz;k++){
			for(j=0;j<Ny;j++){			
				double x_before = i*dx;
				double y_before = j*dx;
				double z_before = k*dx;
				double x_after = xc + (x_before-xc)*cos(theta) - (y_before-yc)*sin(theta);
				double y_after = yc + (x_before-xc)*sin(theta)+(y_before-yc)*cos(theta);
				double z_after = z_before;

				double point1 = 1;
				double point2 = 1;
				double point3 = 1;
				double point4 = 1;

				if(x_after<=0||y_after<=0||Lx<=x_after||Ly<=y_after){
					T_curr[i*Ny*Nz+k*Ny+j] = 20;
				}
				else{
					double x1 = x_after/dx;
					double y1 = y_after/dx;
					int i1 = (int) x1;
					int j1 = (int) y1;
					x1 = i1*dx;
					y1 = j1*dx;
					double xi = (x_after-x1)/dx;
					double eta = (y_after-y1)/dx;
					T_curr[i*Ny*Nz+k*Ny+j] = (1-xi)*(1-eta)*T_init[(i1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1-eta)*T_init[(i1+1)*Ny*Nz+(k)*Ny+(j1)] + (1+xi)*(1+eta)*T_init[(i1+1)*Ny*Nz+(k)*Ny+(j1+1)]+(1-xi)*(1+eta)*T_init[(i1)*Ny*Nz+(k)*Ny+(j1+1)]; 
					T_curr[i*Ny*Nz+k*Ny+j] = T_curr[i*Ny*Nz+k*Ny+j]/4;
				}
			}
		}
	}

}

void set_source_from_elec(std::vector<double> &Source,std::vector<double> &Source_elec,double x_min_th,double y_min_th,double z_min_th,double dx,double dx_electro,int X_th,int Y_th,int Z_th,int X_elec,int Y_elec,int Z_elec){
	double x_elec;
	double y_elec;
	double z_elec;
	int i = 0;
	int j = 0;
	int k = 0;
	int i_th = 0;
	int j_th = 0;
	int k_th = 0;
	double x1;
	double y1;
	double z1;
	int i1;
	int j1;
	int k1;
	double xi = 0;
	double eta = 0;
	double zeta = 0;
	for(i=0;i<X_th;i++){
		for(k=0;k<Z_th;k++){
			for(j=0;j<Y_th;j++){
				x_elec = x_min_th + (i*dx);
				y_elec = y_min_th + (j*dx);
				z_elec = z_min_th + (k*dx);
					
				x1 = x_elec/dx_electro;
				y1 = y_elec/dx_electro;
				z1 = z_elec/dx_electro;

				i1 = (int) x1;
				j1 = (int) y1;
				k1 = (int) z1;

				x1 = i1*dx_electro;
				y1 = j1*dx_electro;
				z1 = k1*dx_electro;

				xi =-1+2*(x_elec-x1)/dx_electro;
				eta =-1+2* (y_elec-y1)/dx_electro;
				zeta =-1+2* (z_elec-z1)/dx_electro;

				Source[i*Y_th*Z_th+k*Y_th+j] = 0;
				
				// z_inf
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j] +(1-xi)*(1-eta)*(1-zeta)*Source_elec[(i1)*Z_elec*Y_elec+(j1)+(k1)*Y_elec] + (1+xi)*(1-eta)*(1-zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1)+(k1)*Y_elec] + (1-xi)*(1+eta)*(1-zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1+1)+(k1)*Y_elec] +( 1+xi)*(1+eta)*(1-zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1+1)+(k1)*Y_elec];
				// z_sup
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j] +(1-xi)*(1-eta)*(1+zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1)+(k1+1)*Y_elec] + (1+xi)*(1-eta)*(1+zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1)+(k1+1)*Y_elec] + (1-xi)*(1+eta)*(1+zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1+1)+(k1+1)*Y_elec] +( 1+xi)*(1+eta)*(1+zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1+1)+(k1+1)*Y_elec];

				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j]/8;
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j];
			}
		}
	}
}
 
 
 
 
 
 




