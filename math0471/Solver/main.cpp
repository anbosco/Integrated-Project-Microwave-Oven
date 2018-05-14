#include "vtl.h"
#include "vtlSPoints.h"
#include "laplace.h"
#include "elm.h"
#include "thermal.h"

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



	/* Handling of input files : The directory that contains
	every input files. */
	// 1) Only the directory.


	if(argc != 2){
		std::cerr << "Error locating the input files" << std::endl;
		return 1;
	}

	std::string TestDir = argv[1] + std::string("/");
	std::string Elm = TestDir + std::string("Interface.dat");				// General parameters for the electro simulation
	std::string Heat =  TestDir + std::string("param_heat.dat");				// General parameters for the thermal simulation
	std::string Cube = TestDir + std::string("Prop_cube.dat");				// Properties of cubic objects
	std::string Cyl = TestDir + std::string("Prop_cylinder.dat");				// Properties of cylindrical objects
	std::string Sphere = TestDir + std::string("Prop_sphere.dat");				// Properties of spherical objects
	std::string SpatialCutElm = TestDir + std::string("Spatial_cut.dat");			// Properties of Spatial cuts for electro simulation
	std::string SpatialCutTemp = TestDir + std::string("Spatial_cut_temp.dat");		// Properties of Spatial cuts for thermal simulation
	std::string TemporalProbeTemp = TestDir + std::string("Temporal_probe.dat");		// Properties of temporal probe for thermal simulation
	std::string TemporalProbeElm =  TestDir + std::string("temp_probe_electro.dat");	// Properties of temporal probe for electro simulation
	std::string HeatSource = TestDir + std::string("prop_source_heat.dat");			// Properties of heat sources placed manually
	std::string Save_Data = TestDir + std::string("Save_Data.dat");

	// Variables used to open files and to write in files.
  	int next_cut = 0;
	FILE *FileR;
	FILE *FileW;
	FileR = fopen(Elm.c_str(),"r");
	if(FileR == NULL){
		std::cerr << "Impossible to open the data file." << std::endl;
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
			std::cerr << "Impossible to read the data file." << std::endl;
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
	double T_mean = 120/(f*dt); 
	int step_mean = (int) T_mean;
	double Residual = 0;
  	double Residual_0 = 0;
	int steady_state_reached = 0;
	int test_steady = 0;
	double total_power_diss = 0;

  	// Prop spheres
	std::vector<double> prop_sphere;
	FileR = fopen(Sphere.c_str(),"r");
	if(FileR == NULL){
		std::cerr << "Impossible to open the sphere file (Object property)." << std::endl;
		return 1;
	}
	for (i=0 ; i<5*n_sphere; i++){
		if (fgets(chain, 150, FileR) == NULL){
			std::cerr << "Impossible to read the sphere file." << std::endl;
			return 1;
		}
		else{
			prop_sphere.push_back(atof(chain));
		}
	}
	fclose(FileR);

  	// Prop cylinders
	std::vector<double> prop_cylinder;
	FileR = fopen(Cyl.c_str(),"r");
	if(FileR == NULL){
		std::cerr << "Impossible to open the cylinder file (Object property)." << std::endl;
		return 1;
	}
	for (i=0 ; i<7*n_cylinder; i++){
		if (fgets(chain, 150, FileR) == NULL){
			std::cerr << "Impossible to read the cylinder file." << std::endl;
			return 1;
		}
		else{
			prop_cylinder.push_back(atof(chain));
		}
	}
	fclose(FileR);

  	// Prop cubes
	std::vector<double> prop_cube;
	FileR = fopen(Cube.c_str(),"r");
	if(FileR == NULL){
		std::cerr << "Impossible to open the cube file (Object property)." << std::endl;
		return 1;
	}
	for (i=0 ; i<7*n_cube; i++){
		if (fgets(chain, 150, FileR) == NULL){
			std::cerr << "Impossible to read the cube file." << std::endl;
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
    FileR = fopen(SpatialCutElm.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Cuts file." << std::endl;
    	return 1;
    }
    for (i=0 ; i<10; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		std::cerr << "Impossible to read the Cuts file." << std::endl;
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
			std::cerr << "Impossible to read the Cuts file." << std::endl;
			return 1;
  	  	}
		step_cut.push_back(atof(chain)/dt);
    	}
    }
    fclose(FileR);

    // Property of temporal probes
    double probe_electro;
    std::vector<double> Pos_probe_electro;
    FileR = fopen(TemporalProbeElm.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Electro Probe file." << std::endl;
    	return 1;
    }
    if (fgets(chain, 150, FileR) == NULL){
	std::cerr << "Impossible to read the Electro Probe file." << std::endl;
	return 1;
   }
   else{
	probe_electro = atof(chain);
   }
   if(probe_electro!=0){
   	for(i=0;i<3;i++){
    		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Electro Probe file. \n");
			return 1;
  		}
 		else{
			Pos_probe_electro.push_back(atof(chain));
  		}
    	}
   }
	Pos_probe_electro[0] = (Pos_probe_electro[0])/dx;
	Pos_probe_electro[1] = (Pos_probe_electro[1])/dx;
	Pos_probe_electro[2] = (Pos_probe_electro[2])/dx;

	std::vector<double> Ex_probe;
	std::vector<double> Ey_probe;
	std::vector<double> Ez_probe;
	std::vector<double> Hx_probe;
	std::vector<double> Hy_probe;
	std::vector<double> Hz_probe;


    // Property of temporal probes
    std::vector<double> Save_field;
    FileR = fopen(Save_Data.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Save Data file." << std::endl;
    	return 1;
    }
   	for(i=0;i<6;i++){
    		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Save Data file. \n");
			return 1;
  		}
 		else{
			Save_field.push_back(atof(chain));
  		}
    	}


/********************************************************************************
			Importation (THERMO).
********************************************************************************/
  
    // Parameter of the simulation and boundary conditions
    std::vector<int>  BC(6);
    std::vector<double> T_Dir(6);
    double T_0;


    FileR = fopen(Heat.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Heat data file." << std::endl;
    	return 1;
    }

    double data_th[40];
    for (i=0 ; i<40; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		std::cerr << "Impossible to read the Heat data file." << std::endl;
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
    FileR = fopen(HeatSource.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Source file." << std::endl;
    	return 1;
    }
    for (i=0 ; i<prop_per_source_th*nb_source_th; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
			 std::cerr << "Impossible to read the Source file." << std::endl;
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
    FileR = fopen(SpatialCutTemp.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Cuts file." << std::endl;
    	return 1;
    }
    for (i=0 ; i<10; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
			 std::cerr << "Impossible to open the Cuts file." << std::endl;
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
    FileR = fopen(TemporalProbeTemp.c_str(),"r");
    if(FileR == NULL){
    	std::cerr << "Impossible to open the Probe file." << std::endl;
    	return 1;
    }
    if (fgets(chain, 150, FileR) == NULL){
			std::cerr << "Impossible to read the Probe file." << std::endl;
			return 1;
   }
   else{
	nb_probe_th = atof(chain);
   }
   if(nb_probe_th!=0){
   	for(i=0;i<3*nb_probe_th;i++){
    		if (fgets(chain, 150, FileR) == NULL){
			std::cerr << "Impossible to read the Probe file." << std::endl;
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

fclose(FileR);




/********************************************************************************
		Declaration of variables. (ELECTRO)
********************************************************************************/
	std::vector<double> vec_e_r;
	std::vector<double> vec_e_r_hot;
	std::vector<double> vec_mu_r;
	std::vector<double> vec_e_diel;
	std::vector<double> vec_e_diel_hot;
	std::vector<double> Temp_phase_change;

	vec_e_r.push_back(1);			// air
	vec_e_r.push_back(51.5);		// potato
	vec_e_r.push_back(4);			// Food from ref
	vec_e_r.push_back(53.5);		// Water
	vec_e_r.push_back(49);			// Chicken
	vec_e_r.push_back(53.5);		// Not physical	
	vec_e_r.push_back(11.345);		// Chicken Bone
	vec_e_r.push_back(5.56);		// Chicken Fat	
	vec_e_r.push_back(59);			// Chicken Muscle
   

	vec_e_r_hot.push_back(1);		// air
	vec_e_r_hot.push_back(1000);		// potato
	vec_e_r_hot.push_back(1000);		// Food from ref
	vec_e_r_hot.push_back(75.8);		// Water
	vec_e_r_hot.push_back(49);		// Chicken
	vec_e_r_hot.push_back(1000);		// Not physical
	vec_e_r_hot.push_back(11.345));		// Chicken Bone
	vec_e_r_hot.push_back(5.56);		// Chicken Fat
	vec_e_r_hot.push_back(59);		// Chicken Muscle

	vec_mu_r.push_back(1);			// air
	vec_mu_r.push_back(1);			// potato
	vec_mu_r.push_back(1);			// Food from ref
	vec_mu_r.push_back(1);			// Water
	vec_mu_r.push_back(1);			// Chicken
	vec_mu_r.push_back(1);			// Not physical
	vec_mu_r.push_back(1);			// Chicken Bone
	vec_mu_r.push_back(1);			// Chicken Fat
	vec_mu_r.push_back(1);			// Chicken Muscle

	vec_e_diel.push_back(0);		// air
	vec_e_diel.push_back(16.3);		// potato
	vec_e_diel.push_back(17.3);		// Food from ref
	vec_e_diel.push_back(1);		// Water
	vec_e_diel.push_back(16.1);		// Chicken
	vec_e_diel.push_back(18.3);		// Not physical
	vec_e_diel.push_back(2.904);		// Chicken Bone
	vec_e_diel.push_back(2.29);		// Chicken Fat
	vec_e_diel.push_back(54.28);		// Chicken Muscle

	vec_e_diel_hot.push_back(0);		// air
	vec_e_diel_hot.push_back(100);		// potato
	vec_e_diel_hot.push_back(100);		// Food from ref
	vec_e_diel_hot.push_back(5);		// Water
	vec_e_diel_hot.push_back(16.1);		// Chicken
	vec_e_diel_hot.push_back(100);		// Not physical
	vec_e_diel_hot.push_back(2.904);	// Chicken Bone
	vec_e_diel_hot.push_back(2.29);		// Chicken Fat
	vec_e_diel_hot.push_back(54.28);	// Chicken Musclel

	Temp_phase_change.push_back(0);		// air
	Temp_phase_change.push_back(20);	// potato
	Temp_phase_change.push_back(20);	// Food from ref
	Temp_phase_change.push_back(0);		// Water
	Temp_phase_change.push_back(0);		// Chicken
	Temp_phase_change.push_back(20);	// Not physical
	Temp_phase_change.push_back(0);		// Chicken Bone
	Temp_phase_change.push_back(0);		// Chicken Fat
	Temp_phase_change.push_back(0);		// Chicken Musclel

	// Used to check the steady state
	double E_max_new = 0;
	double E_max_old = 1;

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
	std::vector<double> e_diel((point_per_proc_x[myrank])*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]));
	set_vec(e_diel, (point_per_proc_x[myrank])*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]) , 0);
  	set_vec(e_rx, (point_per_proc_x[myrank]+lastx)*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]), 1);
 	set_vec(e_ry, (point_per_proc_x[myrank])*(point_per_proc_y[myrank]+lasty)*(point_per_proc_z[myrank]), 1);
 	set_vec(e_rz, (point_per_proc_x[myrank])*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]+lastz), 1);

	std::vector<double> e_r_totx(Nx*Ny*Nz+(Ny*Nz));
	std::vector<double> e_r_toty(Nx*Ny*Nz+(Nx*Nz));
	std::vector<double> e_r_totz(Nx*Ny*Nz+(Nx*Ny));
	std::vector<double> e_diel_tot(Nx*Ny*Nz);
	std::vector<double> mu_r_tot(Nx*Ny*Nz);
  	set_vec(e_r_totx, Nx*Ny*Nz+(Ny*Nz), 1);
  	set_vec(e_r_toty, Nx*Ny*Nz+(Nx*Nz), 1);
  	set_vec(e_r_totz, Nx*Ny*Nz+(Nx*Ny), 1);
	set_vec(e_diel_tot, Nx*Ny*Nz, 0);



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
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[4],e_r_totx,e_r_toty,e_r_totz ,vec_e_r ,e_diel_tot, vec_e_diel,T_food_init_th,vec_e_r_hot, vec_e_diel_hot,Temp_phase_change);
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
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[5],e_r_totx,e_r_toty,e_r_totz ,vec_e_r,e_diel_tot, vec_e_diel,T_food_init_th,vec_e_r_hot, vec_e_diel_hot,Temp_phase_change);
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
		place_geometry(Nx,Ny, Nz, prop_temp, Config, geometry_init , dx, prop_temp[6],e_r_totx,e_r_toty,e_r_totz ,vec_e_r,e_diel_tot, vec_e_diel,T_food_init_th,vec_e_r_hot, vec_e_diel_hot,Temp_phase_change);
	}
 	// Set the geometry on the current proc
	init_geom_one_proc(e_rx,mu_r,e_ry,e_rz,e_r_totx,e_r_toty,e_r_totz,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz, Nx, Ny, Nz,e_diel,e_diel_tot);


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
	mygrid_Power.scalars["diel_perm"] = &e_diel;
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
			sgrids_Hz[l].id = l;
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
    std::vector<double> vec_rho_hot;
    std::vector<double> vec_cp_hot;
    std::vector<double> vec_k_hot;

    // Air
    vec_k.push_back(0.025);
    vec_rho.push_back(1.2);
    vec_cp.push_back(1004);
    vec_rho_hot.push_back(1.2);
    vec_cp_hot.push_back(1004);
    vec_k_hot.push_back(0.025);

    // potato
    vec_k.push_back(0.56);
    vec_rho.push_back(1130);
    vec_cp.push_back(3530);
    vec_rho_hot.push_back(100);
    vec_cp_hot.push_back(100);
    vec_k_hot.push_back(2);

    // SomeFood from ref
    vec_k.push_back(1.4);
    vec_rho.push_back(725);
    vec_cp.push_back(1450);
    vec_rho_hot.push_back(770);
    vec_cp_hot.push_back(2770);
    vec_k_hot.push_back(0.4);

    // Water
    vec_k.push_back(2.1);
    vec_rho.push_back(917);
    vec_cp.push_back(2060);
    vec_rho_hot.push_back(1000);
    vec_cp_hot.push_back(4200);
    vec_k_hot.push_back(0.6);

    // Chicken
    vec_k.push_back(0.7);
    vec_rho.push_back(1080);
    vec_cp.push_back(3132);
    vec_rho_hot.push_back(1080);
    vec_cp_hot.push_back(3132);
    vec_k_hot.push_back(0.7);

    // Not physical
    vec_k.push_back(1);
    vec_rho.push_back(1);
    vec_cp.push_back(1);
    vec_rho_hot.push_back(1);
    vec_cp_hot.push_back(1);
    vec_k_hot.push_back(1);

    // Chicken bones
    vec_k.push_back(0.58);
    vec_rho.push_back(1750);
    vec_cp.push_back(440);
    vec_rho_hot.push_back(1750);
    vec_cp_hot.push_back(440);
    vec_k_hot.push_back(0.58);

    // Chicken fat
    vec_k.push_back(0.201);
    vec_rho.push_back(909.4);
    vec_cp.push_back(2348);
    vec_rho_hot.push_back(909.4);
    vec_cp_hot.push_back(2348);
    vec_k_hot.push_back(0.201);

    // Chicken muscle
    vec_k.push_back(0.478);
    vec_rho.push_back(1059.9);
    vec_cp.push_back(3421);
    vec_rho_hot.push_back(1059.9);
    vec_cp_hot.push_back(3421);
    vec_k_hot.push_back(0.478);


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


    	// Placement of the geometry(Thermo)
	// Sphere
	for(i=0;i<n_sphere;i++){
		int j=0;
		int prop_per_obj = 5;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_sphere[prop_per_obj*i+j];
		}
		int Config = 0;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[4],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th,vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
	}
	//Cylinder
	for(i=0;i<n_cylinder;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cylinder[prop_per_obj*i+j];
		}
		int Config = 1;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[5],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th, vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
	}
	// Cube
	for(i=0;i<n_cube;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cube[prop_per_obj*i+j];
		}
		int Config = 2;
		place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[6],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th, vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
	}
        #pragma omp parallel for default(shared) private(i)
  for(i=0;i<X_th*Y_th*Z_th;i++){
  	constant[i] = (dt_th)/(rho[i]*cp[i]*dx_th*dx_th);
  }


/********************************************************************************
			Begining of the algorithm.
********************************************************************************/

if(solve_electro==0){	// Desactivation of the electro magnetic solver
	step_pos_max = 0;
	step = -1;
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
		printf("\n*********************************************************\n          POSITION DE LA NOURRITURE : %d sur %d \n*********************************************************\n",step_pos ,step_pos_max);
		if(solve_electro==1){
			printf("      SOLVING OF THE ELECTRO MAGNETIC EQUATION... \n");
		}
	}

	while(step!=-1){
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
					 	Update_prev_in_send(1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front_send,Hy_prev,Hz_front_send,Hz_prev, 1, point_per_proc_x[myrank]+lastx);
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
						Update_prev_in_send(1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Hy_front_send,Hy_prev,Hz_front_send,Hz_prev, 1, point_per_proc_x[myrank]+lastx);
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
						Update_prev_in_send(point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],Hx_right_send,Hx_prev,Hz_right_send,Hz_prev, 2, point_per_proc_y[myrank]+lasty);
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
						Update_prev_in_send(point_per_proc_x[myrank],1,point_per_proc_z[myrank]+lastz,point_per_proc_x[myrank]+lastx,1,point_per_proc_z[myrank],Hx_right_send,Hx_prev,Hz_right_send,Hz_prev, 2, point_per_proc_y[myrank]+lasty);
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
						Update_prev_in_send(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,Hx_up_send,Hx_prev,Hy_up_send,Hy_prev, 3, point_per_proc_z[myrank]+lastz);
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
						Update_prev_in_send(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,1,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],1,Hx_up_send,Hx_prev,Hy_up_send,Hy_prev, 3, point_per_proc_z[myrank]+lastz);
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
			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_rx,myrank,5,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
    			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
		}
		else if (lastz == 1){
			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_rx,myrank,3,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank],Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);
		}
		else{
			Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hy_up,dt,dx,e_0,e_rx,myrank,5,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
     		  	Update_E_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,Hz_right,dt,dx,e_0,e_rx,myrank,3,point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(i=0;i<point_per_proc_x[myrank]+lastx;i++){
				k = point_per_proc_z[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ex_new[i][j][k] = Ex_prev[i][j][k] +(dt/(e_0*e_rx[i*(point_per_proc_y[myrank]*point_per_proc_z[myrank])+k*(point_per_proc_y[myrank])+j]*dx))*((Hz_right[i][k]-Hz_prev[i][j][k])-(Hy_up[i][j]-Hy_prev[i][j][k]));
			}
      			Update_E_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]-1,Ex_new,Ex_prev,Hz_prev,Hy_prev,dt,dx,e_0,e_rx,1);

		}

		//	Y component
		if(lastx==1 && lastz==1){
     			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);
		}
		else if(lastx==1){
     			 Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_ry,myrank,6,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
			 Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);
		}
		else if(lastz==1){
			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_ry,myrank,1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
     			 Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank],Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);

		}
		else{
			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hx_up,dt,dx,e_0,e_ry,myrank,6,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
     			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,Hz_front,dt,dx,e_0,e_ry,myrank,1,point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(j=0;j<point_per_proc_y[myrank]+lasty;j++){
				i = point_per_proc_x[myrank]-1;
				k = point_per_proc_z[myrank]-1;
				Ey_new[i][j][k] = Ey_prev[i][j][k] + (dt/(e_0*e_ry[i*((point_per_proc_y[myrank]+lasty)*point_per_proc_z[myrank])+k*(point_per_proc_y[myrank]+lasty)+j]*dx))*((Hx_up[i][j]-Hx_prev[i][j][k])-(Hz_front[j][k]-Hz_prev[i][j][k]));
			}
     			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]-1,Ey_new,Ey_prev,Hx_prev,Hz_prev,dt,dx,e_0,e_ry,2);

		}
		//	Z component
		if(lastx==1 && lasty==1){
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
		}
		else if(lastx==1){
			Update_E_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_rz,myrank,4,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
      			Update_E_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
		}
		else if(lasty==1){
			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_rz,myrank,2,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
      			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
		}
		else{
			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hx_right,dt,dx,e_0,e_rz,myrank,4,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
    			Update_E_boundary(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,Hy_front,dt,dx,e_0,e_rz,myrank,2,point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz);
			#pragma omp parallel for default(shared) private(i,j,k)
			for(k=0;k<point_per_proc_z[myrank]+lastz;k++){
				i = point_per_proc_x[myrank]-1;
				j = point_per_proc_y[myrank]-1;
				Ez_new[i][j][k] = Ez_prev[i][j][k] + (dt/(e_0*e_rz[i*(point_per_proc_y[myrank]*(point_per_proc_z[myrank]+lastz))+k*(point_per_proc_y[myrank])+j]*dx))*((Hy_front[j][k]-Hy_prev[i][j][k])-(Hx_right[i][k]-Hx_prev[i][j][k]));
			}
    			Update_E_inside(point_per_proc_x[myrank]-1,point_per_proc_y[myrank]-1,point_per_proc_z[myrank]+lastz,Ez_new,Ez_prev,Hy_prev,Hx_prev,dt,dx,e_0,e_rz,3);
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
							Ey_new[i][j][k]= E_amp*sin((3.141692*yrel)/l_ay)*sin(omega*step*dt);
          				  	}
					}
				}
			}
			if(P==2){
				#pragma omp parallel for default(shared) private(i,j,k)
				for(i=b_inf_x;i<=b_sup_x;i++){
					for(j=b_inf_y;j<=b_sup_y+lasty;j++){
						for(k=b_inf_z;k<=b_sup_z;k++){
							Ey_new[i][j][k]= E_amp*sin(omega*step*dt);
          				  	}
					}
				}
			}
			if(P==3){
				#pragma omp parallel for default(shared) private(i,j,k)
				for(i=b_inf_x;i<=b_sup_x;i++){
					for(j=b_inf_y;j<=b_sup_y+lasty;j++){
						for(k=b_inf_z;k<=b_sup_z;k++){
							Ey_new[i][j][k]= E_amp;
           					}
					}
				}
			}
		}
		/*nx = point_per_proc_x[myrank];
		ny = point_per_proc_y[myrank];
		nz = point_per_proc_z[myrank];
		i = nx/2;
		j = ny/2;
		k = nz/2;
		if(E_max_new<sqrt((Ex_new[i][j][k]*Ex_new[i][j][k])+(Ey_new[i][j][k]*Ey_new[i][j][k])+(Ez_new[i][j][k]*Ez_new[i][j][k]))){
			E_max_new=sqrt((Ex_new[i][j][k]*Ex_new[i][j][k])+(Ey_new[i][j][k]*Ey_new[i][j][k])+(Ez_new[i][j][k]*Ez_new[i][j][k]));
		}
		// Check for steady state
		if(step%step_mean==0){
			Residual = sqrt((E_max_new-E_max_old)*(E_max_new-E_max_old));			
			//printf("Step:  %d Rank : %lf Residual : %lf\n",step, E_max_new, E_max_old);
			Residual = Residual/E_max_old;
			printf("Step:  %d Rank : %d Residual : %lf\n",step, myrank, Residual);
			E_max_old = E_max_new;
			E_max_new = 0;
		}*/	


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
		Update_H_boundary(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,0,lasty,Hx_new,Hx_prev,Ey_prev,Ez_prev,Ey_bottom,dt,dx,mu_0,mu_r,myrank,5);
		Update_H_inside(point_per_proc_x[myrank],point_per_proc_y[myrank]+lasty,point_per_proc_z[myrank]+lastz, lastz, lasty,Hx_new,Hx_prev,Ey_prev,Ez_prev,dt,dx,mu_0,mu_r,1);
		Update_H_boundary(point_per_proc_x[myrank],0,point_per_proc_z[myrank]+lastz,lastz,Hx_new,Hx_prev,Ey_prev,Ez_prev,Ez_left,dt,dx,mu_0,mu_r,myrank,3);
		#pragma omp parallel for default(shared) private(i,j,k,temp1,temp2)
		for(i=0;i<point_per_proc_x[myrank];i++){
			j = 0;
			k = 0;
			temp1 = Ey_prev[i][j][k];
			temp2 = Ez_prev[i][j][k];
			Hx_new[i][j][k] = Hx_prev[i][j][k] + (dt/(mu_0*mu_r[i][j][k]*dx))*((temp1-Ey_bottom[i][j])-(temp2-Ez_left[i][k]));
		}

		//	Y Component
		Update_H_boundary(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],0,lastx,Hy_new,Hy_prev,Ez_prev,Ex_prev,Ex_bottom,dt,dx,mu_0,mu_r,myrank,6);
		Update_H_inside(point_per_proc_x[myrank]+lastx,point_per_proc_y[myrank],point_per_proc_z[myrank]+lastz, lastx, lastz,Hy_new,Hy_prev,Ez_prev,Ex_prev,dt,dx,mu_0,mu_r,2);
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

		/*************** Storage of the matrices in vectors for exportation ***************/
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
		/***********************************************************************************/

		/*************************** Storage of the Results ************** *****************/

		if(step%SR==0){//save results of the mpi process to disk
			if(Save_field[0]==1)
				export_spoints_XML("Ex", step, grid_Ex, mygrid_Ex, ZIPPED, Nx, Ny, Nz, 0);
			if(Save_field[1]==1)
			  	export_spoints_XML("Ey", step, grid_Ey, mygrid_Ey, ZIPPED, Nx, Ny+lasty, Nz, 0);
			if(Save_field[2]==1)
				export_spoints_XML("Ez", step, grid_Ez, mygrid_Ez, ZIPPED, Nx, Ny, Nz, 0);
			if(Save_field[3]==1)
				export_spoints_XML("Hx", step, grid_Hx, mygrid_Hx, ZIPPED, Nx, Ny, Nz, 0);
			if(Save_field[4]==1)
				export_spoints_XML("Hy", step, grid_Hy, mygrid_Hy, ZIPPED, Nx, Ny, Nz, 0);
			if(Save_field[5]==1)
				export_spoints_XML("Hz", step, grid_Hz, mygrid_Hz, ZIPPED, Nx, Ny, Nz, 0);

            		if (myrank == 0){	// save main pvti file by rank0
				if(Save_field[0]==1)
					export_spoints_XMLP("Ex", step, grid_Ex, mygrid_Ex, sgrids_Ex, ZIPPED);
				if(Save_field[1]==1)
		        		export_spoints_XMLP("Ey", step, grid_Ey, mygrid_Ey, sgrids_Ey, ZIPPED);
				if(Save_field[2]==1)
					export_spoints_XMLP("Ez", step, grid_Ez, mygrid_Ez, sgrids_Ez, ZIPPED);
				if(Save_field[3]==1)
					export_spoints_XMLP("Hx", step, grid_Hx, mygrid_Hx, sgrids_Hx, ZIPPED);
				if(Save_field[4]==1)
					export_spoints_XMLP("Hy", step, grid_Hy, mygrid_Hy, sgrids_Hy, ZIPPED);
				if(Save_field[5]==1)
					export_spoints_XMLP("Hz", step, grid_Hz, mygrid_Hz, sgrids_Hz, ZIPPED);
            		}
        	}
		// Storage of the value of the fields at the probes
		if((probe_electro==1)&&(Pos_probe_electro[0]<i_max_proc[myrank])&&(Pos_probe_electro[0]>i_min_proc[myrank])&&(Pos_probe_electro[1]<j_max_proc[myrank])&&(Pos_probe_electro[1]>j_min_proc[myrank])&&(Pos_probe_electro[2]<k_max_proc[myrank])&&(Pos_probe_electro[2]>k_min_proc[myrank])){
			i = Pos_probe_electro[0];
			j = Pos_probe_electro[1];
			k = Pos_probe_electro[2];
			Ex_probe.push_back(Ex_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
			Ey_probe.push_back(Ey_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
			Ez_probe.push_back(Ez_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
			Hx_probe.push_back(Hx_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
			Hy_probe.push_back(Hy_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
			Hz_probe.push_back(Hz_new[i-i_min_proc[myrank]][j-j_min_proc[myrank]][k-k_min_proc[myrank]]);
		}
     		//Extraction of a cut if needed 

	if(step == (int) step_cut[next_cut]){// To extract a cut
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
/*****************************************************************************************/


	// Computation of the power grid (TO BE PARAMETRIZED)
		nx = point_per_proc_x[myrank];
		ny = point_per_proc_y[myrank];
		nz = point_per_proc_z[myrank];

		int iii = 0;		// Variables used to handle the boudaries
		int jjj = 0;
		int kkk = 0;
   
		#pragma omp parallel for default(shared) private(i,j,k)
		for(i=1;i<(nx-lastx);i++){
			for(j=1;j<(ny-lasty);j++){
				for(k=1;k<(nz-lastz);k++) {
					iii = 0;
					jjj = 0;
					kkk = 0;
					if(firstx!=1 && lastx!=1 && i == nx-1){
						iii = 1;
					}
					if(firsty!=1 && lasty!=1 && j == ny-1){
						jjj = 1;
					}
					if(firstz!=1 && lastz!=1 && k == nz-1){
						kkk = 1;
					}
           				//x
           				Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_new[i][j-firsty][k-firstz]*Ex_new[i][j-firsty][k-firstz])+(Ex_new[i][j-firsty+1-jjj][k-firstz]*Ex_new[i][j-firsty+1-jjj][k-firstz])+(Ex_new[i][j-firsty][k+1-firstz-kkk]*Ex_new[i][j-firsty][k+1-firstz-kkk])+(Ex_new[i][j+1-firsty-jjj][k+1-firstz-kkk]*Ex_new[i][j+1-firsty-jjj][k+1-firstz-kkk]))/12;					
					//y					
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx]+  e_0*((Ey_new[i-firstx][j][k-firstz]*Ey_new[i-firstx][j][k-firstz])+(Ey_new[i+1-firstx-iii][j][k-firstz]*Ey_new[i+1-firstx-iii][j][k-firstz])+(Ey_new[i-firstx][j][k+1-firstz-kkk]*Ey_new[i-firstx][j][k+1-firstz-kkk])+(Ey_new[i+1-firstx-iii][j][k+1-firstz-kkk]*Ey_new[i+1-firstx-iii][j][k+1-firstz-kkk]))/12;          
					//z					
					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_new[i-firstx][j-firsty][k]*Ez_new[i-firstx][j-firsty][k])+(Ez_new[i+1-firstx-iii][j-firsty][k]*Ez_new[i+1-firstx-iii][j-firsty][k])+(Ez_new[i-firstx][j+1-firsty-jjj][k]*Ez_new[i-firstx][j+1-firsty-jjj][k])+(Ez_new[i+1-firstx-iii][j+1-firsty-jjj][k]*Ez_new[i+1-firstx-iii][j+1-firsty-jjj][k]))/12;
					
      				}
			}
		}

  		if(firstx!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
		        for(i=0;i<=0;i++){
				  for(j=1;j<(ny-lasty);j++){
					for(k=1;k<(nz-lastz);k++){
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k  == nz-1){
							kkk = 1;
						}			   			
						//x
			   			Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_new[i][j-firsty][k-firstz]*Ex_new[i][j-firsty][k-firstz])+(Ex_new[i][j-firsty+1-jjj][k-firstz]*Ex_new[i][j-firsty+1-jjj][k-firstz])+(Ex_new[i][j-firsty][k+1-firstz-kkk]*Ex_new[i][j-firsty][k+1-firstz-kkk])+(Ex_new[i][j+1-firsty-jjj][k+1-firstz-kkk]*Ex_new[i][j+1-firsty-jjj][k+1-firstz-kkk]))/12;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_back[j][k-firstz]*Ey_back[j][k-firstz])+(Ey_new[i+1-iii-firstx][j][k-firstz]*Ey_new[i+1-iii-firstx][j][k-firstz])+(Ey_back[j][k+1-kkk-firstz]*Ey_back[j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/12;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_back[j-firsty][k]*Ez_back[j-firsty][k])+(Ez_new[i+1-iii-firstx][j-firsty][k]*Ez_new[i+1-iii-firstx][j-firsty][k])+(Ez_back[j+1-jjj-firsty][k]*Ez_back[j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/12;
					}
				}
			}
 	 	}
   		if(firsty!=1){			
			#pragma omp parallel for default(shared) private(i,j,k)
   			for(i=1;i<(nx-lastx);i++){
				for(j=0;j<=0;j++){
					for(k=1;k<(nz-lastz);k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
           					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_left[i][k-firstz]*Ex_left[i][k-firstz])+(Ex_new[i][j-firsty+1-jjj][k-firstz]*Ex_new[i][j-firsty+1-jjj][k-firstz])+(Ex_left[i][k+1-kkk-firstz]*Ex_left[i][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/12;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_new[i-firstx][j][k-firstz]*Ey_new[i-firstx][j][k-firstz])+(Ey_new[i+1-iii-firstx][j][k-firstz]*Ey_new[i+1-iii-firstx][j][k-firstz])+(Ey_new[i-firstx][j][k+1-kkk-firstz]*Ey_new[i-firstx][j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/12;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_left[i-firstx][k]*Ez_left[i-firstx][k])+(Ez_left[i+1-iii-firstx][k]*Ez_left[i+1-iii-firstx][k])+(Ez_new[i-firstx][j+1-jjj-firsty][k]*Ez_new[i-firstx][j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/12;
        				}
				}
			}
   		}
  		 if(firstz!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
   			for(i=1;i<(nx-lastx);i++){
				for(j=1;j<(ny-lasty);j++){
					for(k=0;k<=0;k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
           					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_bottom[i][j-firsty]*Ex_bottom[i][j-firsty])+(Ex_bottom[i][j-firsty+1-jjj]*Ex_bottom[i][j-firsty+1-jjj])+(Ex_new[i][j-firsty][k+1-kkk-firstz]*Ex_new[i][j-firsty][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/12;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_bottom[i-firstx][j]*Ey_bottom[i-firstx][j])+(Ey_bottom[i+1-iii-firstx][j]*Ey_bottom[i+1-iii-firstx][j])+(Ey_new[i-firstx][j][k+1-kkk-firstz]*Ey_new[i-firstx][j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+lasty-firstz]*Ey_new[i+1-iii-firstx][j][k+lasty-firstz]))/12;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_new[i-firstx][j-firsty][k]*Ez_new[i-firstx][j-firsty][k])+(Ez_new[i+1-iii-firstx][j-firsty][k]*Ez_new[i+1-iii-firstx][j-firsty][k])+(Ez_new[i-firstx][j+1-jjj-firsty][k]*Ez_new[i-firstx][j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/12;
        				}
				}
			}
  		}

   		if(firstx!=1 && firsty!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
     			for(i=0;i<=0;i++){
		  		for(j=0;j<=0;j++){
					for(k=1;k<(nz-lastz);k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
          					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_left[i][k-firstz]*Ex_left[i][k-firstz])+(Ex_new[i][j-firsty+1-jjj][k-firstz]*Ex_new[i][j-firsty+1-jjj][k-firstz])+(Ex_left[i][k+1-kkk-firstz]*Ex_left[i][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/11;
          					//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_back[j][k-firstz]*Ey_back[j][k-firstz])+(Ey_new[i+1-iii-firstx][j][k-firstz]*Ey_new[i+1-iii-firstx][j][k-firstz])+(Ey_back[j][k+1-kkk-firstz]*Ey_back[j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/11;
        					//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_left[i+1-iii-firstx][k]*Ez_left[i+1-iii-firstx][k])+(Ez_back[j+1-jjj-firsty][k]*Ez_back[j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/11;
        				}
				}
			}
   		}

   		if(firstx!=1 && firstz!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
     			for(i=0;i<=0;i++){
		  		for(j=1;j<(ny-lasty);j++){
					for(k=0;k<=0;k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
           					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_bottom[i][j-firsty]*Ex_bottom[i][j-firsty])+(Ex_bottom[i][j-firsty+1-jjj]*Ex_bottom[i][j-firsty+1-jjj])+(Ex_new[i][j-firsty][k+1-kkk-firstz]*Ex_new[i][j-firsty][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/11;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_bottom[i+1-iii-firstx][j]*Ey_bottom[i+1-iii-firstx][j])+(Ey_back[j][k+1-kkk-firstz]*Ey_back[j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/11;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_back[j-firsty][k]*Ez_back[j-firsty][k])+(Ez_new[i+1-iii-firstx][j-firsty][k]*Ez_new[i+1-iii-firstx][j-firsty][k])+(Ez_back[j+1-jjj-firsty][k]*Ez_back[j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/11;
        				}
				}
			}
   		}

   		if(firsty!=1 && firstz!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
     			for(i=1;i<(nx-lastx);i++){
				for(j=0;j<=0;j++){
					for(k=0;k<=0;k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
           					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_bottom[i][j-firsty+1-jjj]*Ex_bottom[i][j-firsty+1-jjj])+(Ex_left[i][k+1-kkk-firstz]*Ex_left[i][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/11;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_bottom[i-firstx][j]*Ey_bottom[i-firstx][j])+(Ey_bottom[i+1-iii-firstx][j]*Ey_bottom[i+1-iii-firstx][j])+(Ey_new[i-firstx][j][k+1-kkk-firstz]*Ey_new[i-firstx][j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/11;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_left[i-firstx][k]*Ez_left[i-firstx][k])+(Ez_left[i+1-iii-firstx][k]*Ez_left[i+1-iii-firstx][k])+(Ez_new[i-firstx][j+1-jjj-firsty][k]*Ez_new[i-firstx][j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]))/11;
        				}
				}
			}
   		}
   		if(firstx!=1 && firsty!=1 && firstz!=1){
			#pragma omp parallel for default(shared) private(i,j,k)
     			for(i=0;i<=0;i++){
				for(j=0;j<=0;j++){
					for(k=0;k<=0;k++) {
						iii = 0;
						jjj = 0;
						kkk = 0;
						if(firstx!=1 && lastx!=1 && i == nx-1){
							iii = 1;
						}
						if(firsty!=1 && lasty!=1 && j == ny-1){
							jjj = 1;
						}
						if(firstz!=1 && lastz!=1 && k == nz-1){
							kkk = 1;
						}
           					//x
           					Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] + e_0*((Ex_bottom[i][j-firsty+1-jjj]*Ex_bottom[i][j-firsty+1-jjj])+(Ex_left[i][k+1-kkk-firstz]*Ex_left[i][k+1-kkk-firstz])+(Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]*Ex_new[i][j+1-jjj-firsty][k+1-kkk-firstz]))/9;
						//y
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ey_bottom[i+1-iii-firstx][j]*Ey_bottom[i+1-iii-firstx][j])+(Ey_back[j][k+1-kkk-firstz]*Ey_back[j][k+1-kkk-firstz])+(Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]*Ey_new[i+1-iii-firstx][j][k+1-kkk-firstz]))/9;
						//z
						Power_new[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx] +  e_0*((Ez_left[i+1-iii-firstx][k]*Ez_left[i+1-iii-firstx][k])+(Ez_back[j+1-jjj-firsty][k]*Ez_back[j+1-jjj-firsty][k])+(Ez_new[i+1-iii-firstx][j+1-jjj-firsty][k]*Ez_new[i+1-iii-firstx][j+1-firsty][k]))/9;
        				}
				}
			}
   		}
		if(step%step_mean==0||(solve_thermo==0 && step == step_max)){
			/**************************************
				Steady state verification
			**************************************/

			/****************** Check if steady state is reached on the current process **********************/
			Residual = 0;
			Residual_0 = 0;
			for(i=0;i<nx;i++){
				for(j=0;j<ny;j++){
					for(k=0;k<(nz);k++) {
						Power_new[i+j*nx+k*(ny)*nx] = (3.141692*f*Power_new[i+j*nx+k*(ny)*nx])/(step_mean);
						Power_new[i+j*nx+k*(ny)*nx] = e_diel[j+k*ny+i*(ny)*nz]*Power_new[i+j*nx+k*(ny)*nx];
						Residual = Residual + (Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx])*(Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx]);
						Residual_0 = Residual_0 + (Power_old[i+j*nx+k*(ny)*nx])*(Power_old[i+j*nx+k*(ny)*nx]);
						/*if(i==nx/2&&j==ny/2&&k==nz/2){
							Residual = ((Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx])*(Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx]));
							Residual = sqrt((Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx])*(Power_new[i+j*nx+k*(ny)*nx]-Power_old[i+j*nx+k*(ny)*nx]));
							Residual = Residual/Power_old[i+j*nx+k*(ny)*nx];
						}*/
						Power_old[i+j*nx+k*(ny)*nx] = Power_new[i+j*nx+k*(ny)*nx];
					}
				}
			}
			/*if(Residual<0.0025){
				steady_state_reached = 1;
			}
			else{
				steady_state_reached = 0;
			}*/
     			 if((step/step_mean)==1){
				if(Residual==0){
					Residual = 1;
					steady_state_reached=1;
				}
        			Residual_0 = Residual;
      			}

      			if((Residual<0.1*Residual_0 && (step/step_mean)>2)||Residual==0){
				steady_state_reached = 1;
			}
			else{
				steady_state_reached = 0;
			}
			/**************************************************************************************************/

			/* Communication between the process in order to determine if the algorithm must continue or not. */
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
			//printf("Step:  %d Rank : %d Residual : %lf\n",step, myrank, Residual);

			/****************************************************************************************************/

			if(step>1800000||(solve_thermo==0 && step == step_max))
   			steady_state_reached=1;		/************** To be suppressed if we want to reach the steady state *********************/

			if(steady_state_reached==1){
				break;
 			}
			else{
				set_vec(Power_new, nx*ny*nz, 0);
				Residual_0 = Residual;
			}
		}

 	step++;
	}
	if(solve_electro==1){	// We save the last step of the electro calculation if there was any
		total_power_diss = 0;
		if(myrank==0){
	     		for(i=i_min_proc[myrank];i<=i_max_proc[myrank];i++){
	       			for(j=j_min_proc[myrank];j<=j_max_proc[myrank];j++){
		 			for(k=k_min_proc[myrank];k<=k_max_proc[myrank];k++){
		   				Power_tot[i*Ny*Nz+k*Ny+j] = Power_new[i+j*point_per_proc_x[myrank]+k*point_per_proc_x[myrank]*point_per_proc_y[myrank]];
						total_power_diss = total_power_diss + Power_new[i+j*point_per_proc_x[myrank]+k*point_per_proc_x[myrank]*point_per_proc_y[myrank]]*dx*dx*dx;
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
							total_power_diss = total_power_diss + Power_send[i+j*point_per_proc_x[l]+k*point_per_proc_x[l]*point_per_proc_y[l]]*dx*dx*dx;
		   				}
		 			}
	       			}
	     		}
	   	}
	   	else{
	     		MPI_Send(&Power_new[0],point_per_proc_x[myrank]*(point_per_proc_y[myrank])*(point_per_proc_z[myrank]),MPI_DOUBLE,0,myrank,MPI_COMM_WORLD);
	   	}
		if(Save_field[0]==1)
			export_spoints_XML("Ex", step, grid_Ex, mygrid_Ex, ZIPPED, Nx, Ny, Nz, 0);
		if(Save_field[1]==1)
			export_spoints_XML("Ey", step+step_prec, grid_Ey, mygrid_Ey, ZIPPED, Nx, Ny, Nz, 0);
		if(Save_field[2]==1)
			export_spoints_XML("Ez", step+step_prec, grid_Ez, mygrid_Ez, ZIPPED, Nx, Ny, Nz, 0);
		if(Save_field[3]==1)
			export_spoints_XML("Hx", step+step_prec, grid_Hx, mygrid_Hx, ZIPPED, Nx, Ny, Nz, 0);
		if(Save_field[4]==1)
			export_spoints_XML("Hy", step+step_prec, grid_Hy, mygrid_Hy, ZIPPED, Nx, Ny, Nz, 0);
		if(Save_field[5]==1)
			export_spoints_XML("Hz", step+step_prec, grid_Hz, mygrid_Hz, ZIPPED, Nx, Ny, Nz, 0);
		export_spoints_XML("Power", step+step_prec, grid_Power, mygrid_Power, ZIPPED, Nx, Ny, Nz, 0);
		if (myrank == 0){	// save main pvti file by rank0
			if(Save_field[0]==1)
				export_spoints_XMLP("Ex", step+step_prec, grid_Ex, mygrid_Ex, sgrids_Ex, ZIPPED);
			if(Save_field[1]==1)
		      		export_spoints_XMLP("Ey", step+step_prec, grid_Ey, mygrid_Ey, sgrids_Ey, ZIPPED);
			if(Save_field[2]==1)
				export_spoints_XMLP("Ez", step+step_prec, grid_Ez, mygrid_Ez, sgrids_Ez, ZIPPED);
			if(Save_field[3]==1)
				export_spoints_XMLP("Hx", step+step_prec, grid_Hx, mygrid_Hx, sgrids_Hx, ZIPPED);
			if(Save_field[4]==1)
				export_spoints_XMLP("Hy", step+step_prec, grid_Hy, mygrid_Hy, sgrids_Hy, ZIPPED);
			if(Save_field[5]==1)
				export_spoints_XMLP("Hz", step+step_prec, grid_Hz, mygrid_Hz, sgrids_Hz, ZIPPED);
			export_spoints_XMLP("Power", step+step_prec, grid_Power, mygrid_Power, sgrids_Power, ZIPPED);
		}

		// Temporal probe
		if((probe_electro==1)&&(Pos_probe_electro[0]<i_max_proc[myrank])&&(Pos_probe_electro[0]>i_min_proc[myrank])&&(Pos_probe_electro[1]<j_max_proc[myrank])&&(Pos_probe_electro[1]>j_min_proc[myrank])&&(Pos_probe_electro[2]<k_max_proc[myrank])&&(Pos_probe_electro[2]>k_min_proc[myrank])){
			export_temp_probe_electro(Ex_probe,step,"Ex_temporal_probe");
			export_temp_probe_electro(Ey_probe,step,"Ey_temporal_probe");
			export_temp_probe_electro(Ez_probe,step,"Ez_temporal_probe");
			export_temp_probe_electro(Hx_probe,step,"Hx_temporal_probe");
			export_temp_probe_electro(Hy_probe,step,"Hy_temporal_probe");
			export_temp_probe_electro(Hz_probe,step,"Hz_temporal_probe");
		}
		step_prec += step;
		step = 1;
		if(myrank == 0){
			printf("\n\n TOTAL DISSIPATED POWER INSIDE THE FOOD : %lf [W] \n\n",total_power_diss);
		}
	}

/********************************************************************************
			Thermic calculation
*******************************************************************************/
	if(solve_thermo){
		if(myrank==0){
			printf("\n SOLVING OF THE HEAT EQUATION...\n");
		}
		rotate_Power_grid(Power_tot,Power_tot_rotated_back,Nx,Ny,Nz,Lx,Ly,Lz,dx,theta);
		main_th(Power_tot_rotated_back, Temperature,BC, T_Dir,  T_0,dx_th,h_air,Lx_th,Ly_th,Lz_th,dt_th,step_max_th,nb_source_th,SR_th,theta_th,n_sphere,n_cylinder,n_cube,prop_sphere,prop_cylinder,prop_cube,T_food_init_th,x_min_th,y_min_th,z_min_th,dx,Lx,Ly_electro,Lz_electro,prop_per_source_th, prop_source_th, Cut_th,Pos_cut_th,N_cut_th,step_cut_th,nb_probe_th,Pos_probe_th, id,k_heat_x,k_heat_y,k_heat_z,rho,cp,vec_k,vec_rho,vec_cp,constant,geometry_th,step_pos,thermo_domain);
		if(myrank==0){
			for(l=1;l<nbproc;l++){
	    			MPI_Send(&Temperature[0],X_th*Y_th*Z_th,MPI_DOUBLE,l,myrank,MPI_COMM_WORLD);
			}
		}
		else{
			MPI_Recv(&Temperature[0],X_th*Y_th*Z_th,MPI_DOUBLE,0,0,MPI_COMM_WORLD, &mystatus);
		}

		// Update of the parameter(Thermo)
		// Sphere
		for(i=0;i<n_sphere;i++){
			int j=0;
			int prop_per_obj = 5;
			for(j=0;j<prop_per_obj;j++){
				prop_temp[j] = prop_sphere[prop_per_obj*i+j];
			}
			int Config = 0;
			place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[4],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th,vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
		}
		//Cylinder
		for(i=0;i<n_cylinder;i++){
			int j=0;
			int prop_per_obj = 7;
			for(j=0;j<prop_per_obj;j++){
				prop_temp[j] = prop_cylinder[prop_per_obj*i+j];
			}
			int Config = 1;
			place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[5],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th, vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
		}
		// Cube
		for(i=0;i<n_cube;i++){
			int j=0;
			int prop_per_obj = 7;
			for(j=0;j<prop_per_obj;j++){
				prop_temp[j] = prop_cube[prop_per_obj*i+j];
			}
			int Config = 2;
			place_geometry_th(X_th,Y_th, Z_th, prop_temp, Config, geometry_th , dx_th, prop_temp[6],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp,x_min_th,y_min_th,z_min_th, vec_rho_hot,vec_cp_hot,Temp_phase_change,Temperature,vec_k_hot);
		}
        	#pragma omp parallel for default(shared) private(i)
  		for(i=0;i<X_th*Y_th*Z_th;i++){
  			constant[i] = (dt_th)/(rho[i]*cp[i]*dx_th*dx_th);
  		}
	}
/*******************************************************************************
			   Rotation
*******************************************************************************/

      	theta += delta_theta;
	if(delta_theta!=0){
	      	rotate_geometry(geometry_init,e_r_totx,e_r_toty,e_r_totz ,vec_e_r, Nx, Ny, Nz, Lx, Ly, Lz, dx,n_sphere, prop_sphere, n_cylinder,prop_cylinder, n_cube,prop_cube, theta,e_diel_tot,vec_e_diel, Temperature, vec_e_diel_hot,vec_e_r_hot,Temp_phase_change, dx_th,x_min_th,y_min_th,z_min_th,X_th,Y_th,Z_th);
		init_geom_one_proc(e_rx,mu_r,e_ry,e_rz,e_r_totx,e_r_toty,e_r_totz,mu_r_tot, i_min_proc[myrank],j_min_proc[myrank],k_min_proc[myrank],point_per_proc_x[myrank],point_per_proc_y[myrank],point_per_proc_z[myrank],lastx,lasty,lastz, Nx, Ny, Nz,e_diel,e_diel_tot);
	}
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
