// solves a Laplacian over a cube with mumps

#include "vtl.h"
#include "vtlSPoints.h"
#include "laplace.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <cmath>

using namespace vtl;

// from c_example.c ------
#include "mpi.h"
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1] // macro s.t. indices match documentation

// -----------------------

void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt);
void Compute_a_T0(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z);
void insert_obj(std::vector<double> &temp, std::vector<double> &k_heat_x, std::vector<double> &k_heat_y, std::vector<double> &k_heat_z, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &prop_obj, int X,int Y,int Z, double dx,std::vector<double> &geometry);
void insert_Source(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp);
void export_coupe(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,double dx, int step);
void export_probe(double nb_probe , std::vector<double> &probe,int step_max);
void set_kheat(int Case,int X,int Y,int Z, std::vector<double> &properties,int l,double dx,std::vector<double> &k_heat);
void Compute_a_T0_2(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h);
void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &k_heatx,std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,std::vector<double> &cp);
void set_T0(std::vector<double> &Temp,std::vector<double> &geometry,double T_0,double T_init_food,int  X,int  Y,int  Z );
void rotate_Power_grid(std::vector<double> &Source_init,std::vector<double> &Source_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta);
void rotate_T_grid(std::vector<double> &T_init,std::vector<double> &T_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta, double T_air);

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

void host_work(DMUMPS_STRUC_C &id,double Lx,double Ly,double Lz,double delta_x,double delta_t,int step_max,double theta,int nb_source, std::vector<double> &prop_source, int nb_obj,std::vector<double> &prop_obj,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0,double kheat_param,double rho_param,double cp_param,int SR,std::vector<double> &Cut, std::vector<double> &Pos_cut, std::vector<double> &step_cut, double nb_probe, std::vector<double> &Pos_probe,int n_sphere,std::vector<double> &prop_sphere,int n_cylinder,std::vector<double> &prop_cylinder,int n_cube,std::vector<double> &prop_cube, double T_init_food,double h_air){   
    SPoints grid;

    // setup grids

    grid.o = Vec3d(10.0, 10.0, 10.0); // origin
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
    std::cout << nbp << " points created\n";
    std::cout << grid;

    double theta_angle = 0;
    // Declaration of MUMPS variable
    MUMPS_INT n = X*Y*Z;
    std::vector<MUMPS_INT> irn;
    std::vector<MUMPS_INT> jcn;
    std::vector<double> a;
    std::vector<double> b;
    
    // Variables used to solve the system
    std::vector<double> Temp(n);
    std::vector<double> T_init(n);// Suppress this when we have shown numerical diffisivity
    std::vector<double> geometry(n);
    grid.scalars["Geometry"] = &geometry;
    grid.scalars["Temp"] = &Temp;
    std::vector<double> Temp2(n);    
    std::vector<double> Source(n);
    std::vector<double> Source_init(n);
    for(i=0;i<n;i++){
    	Source[i] = 0;
    }

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

    /*Declaration and Initialization of physical characteristics.*/	
   
    std::vector<double> k_heat_x(n+Y*Z);	// One more point in the x direction
    std::vector<double> k_heat_y(n+X*Z);	// One more point in the y direction
    std::vector<double> k_heat_z(n+X*Y);	// One more point in the z direction
    std::vector<double> rho(n);    
    std::vector<double> cp(n);
    grid.scalars["Rho"] = &rho;
    grid.scalars["cp"] = &cp;
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
    
    std::vector<double> constant(n);
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<X*Y*Z;i++){  	
   	rho[i] = vec_rho[0];
	cp[i] = vec_cp[0];
    }  
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n+Y*Z;i++){ 
	k_heat_x[i] = vec_k[0];
    }
   #pragma omp parallel for default(shared) private(i)
   for(i=0;i<n+X*Z;i++){ 
	k_heat_y[i] = vec_k[0];
    }
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n+X*Y;i++){ 
	k_heat_z[i] = vec_k[0];
    }
    // Placement of the geometry
    std::vector<double> prop_temp(7);
	// Sphere	
	for(i=0;i<n_sphere;i++){
		int j=0;
		int prop_per_obj = 5;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_sphere[prop_per_obj*i+j];
		}		
		int Config = 0;
		place_geometry(X,Y, Z, prop_temp, Config, geometry , delta_x, prop_temp[4],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp);
	}
	//Cylinder
	for(i=0;i<n_cylinder;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cylinder[prop_per_obj*i+j];
		}
		int Config = 1;
		place_geometry(X,Y, Z, prop_temp, Config, geometry , delta_x, prop_temp[5],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp);
	}
	// Cube
	for(i=0;i<n_cube;i++){
		int j=0;
		int prop_per_obj = 7;
		for(j=0;j<prop_per_obj;j++){
			prop_temp[j] = prop_cube[prop_per_obj*i+j];
		}
		int Config = 2;
		place_geometry(X,Y, Z, prop_temp, Config, geometry , delta_x, prop_temp[6],vec_k,vec_rho,vec_cp,k_heat_x,k_heat_y,k_heat_z,rho,cp);
	}    
  
  // Insertion of one or more power source inside the domain (Will disappear when coupling is done)
  if(nb_source!=0){
  	insert_Source(Source,nb_source, prop_source, X,Y,Z, delta_x, rho, cp);
  }

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<n;i++){
    	Source_init[i] = Source[i];
    }

 #pragma omp parallel for default(shared) private(i)
  for(i=0;i<X*Y*Z;i++){
  	constant[i] = (delta_t)/(rho[i]*cp[i]*delta_x*delta_x);
  }

  // Computation of the matrix and of the initial temperature
 // Compute_a_T0(irn ,jcn, X, Y, Z, ip_h, jp_h, kp_h, lastx_h, lasty_h, lastz_h, a, b,Temp,constant,BC,T_Dir,T_0,theta,k_heat_x,k_heat_y,k_heat_z);		// Old Boundary Conditions
  
  Compute_a_T0_2(irn ,jcn,  X,  Y,  Z,ip_h,jp_h,kp_h,lastx_h,lasty_h,lastz_h, a, b,Temp,constant,BC,T_Dir, T_0,theta, k_heat_x,k_heat_y,k_heat_z,geometry,delta_x,h_air);		// New boundary conditions  
  MUMPS_INT8 nnz =  irn.size();
  set_T0(Temp,geometry,T_0,T_init_food,  X,  Y,  Z );
    
    // Preparation of MUMPS job
    id.n = n;
    id.nnz = a.size();
    id.irn = &irn[0];
    id.jcn = &jcn[0];
    id.a = &a[0];
    int step = 1;

    // save results to disk
    export_spoints_XML("laplace", step, grid, grid, Zip::ZIPPED, X, Y, Z, 1);
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
	Compute_RHS(b,irn,jcn,Temp,Source,Temp2,X,Y,Z, nnz, rho, cp,geometry,delta_t); 
    	
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
			export_coupe(1, Pos_cut[0], Pos_cut[1], X, Y, Z, Temp, delta_x, step);
		}
		if(Cut[1]==1){
			export_coupe(2, Pos_cut[2], Pos_cut[3], X, Y, Z, Temp, delta_x, step);
		}
		if(Cut[2]==1){
			export_coupe(3, Pos_cut[4], Pos_cut[5], X, Y, Z, Temp, delta_x, step);
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
    		export_spoints_XML("laplace", step, grid, grid, Zip::ZIPPED, X, Y,  Z, 1);

		/************* To be suppress when coupling is done ****************/

		// Power rotation		
		theta_angle = theta_angle + 3.141692/8;
		rotate_Power_grid(Source_init,Source,X, Y, Z, Lx, Ly, Lz, delta_x, theta_angle);
		/******************************************************************/
				
  	  }		
    }
   /***************************  To be suppress when coupling is done **************/
  /*  while(step<60){step++;			// Temperature Rotation
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
    export_probe(nb_probe , probe, step_max);      
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

int main(int argc, char *argv[]){
    // initialise MUMPS/MPI
    MPI_Init(&argc, &argv);
    DMUMPS_STRUC_C id;
    init_MUMPS(id);

   /* Reading of the input files */

    int i = 0;
    std::vector<int>  BC(6);
    std::vector<double> T_Dir(6);
    double T_0;

    // Parameter of the simulation and boundary conditions
    FILE *FileR; 
    FileR = fopen(argv[1],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the data file. \n");
    	return 1; 
    }

    double data[31];	
    char chain[150];
    for (i=0 ; i<31; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the data file. \n");
		return 1; 
  	  }	
   	 else{
		data[i] = atof(chain);
   	 }
    }
    fclose(FileR);
    double Lx = data[0];    // Length of the domain
    double Ly = data[1];
    double Lz = data[2];
    double dx = data[3];    // Grid spacing
    double dt = data[4];    // Time step
    double Tf = data[5];    // Final time
    double temp = Tf/dt;
    int step_max = (int) temp;
    int nb_source = (int) data[6];   // Number of power source
    int nb_obj = (int) data[7];      // Number of objects inside the domain
    for(i=0;i<6;i++){
    	BC[i] = data[8+i];
    	T_Dir[i] = data[14+i];
    }
    T_0 = data[20]; 			// Air Temperature
    double kheat_param = data[21];   
    double rho_param = data[22];
    double cp_param = data[23];
    double S = data[24];    // Sampling rate
    int SR = (int) 1/S;
    double theta = data[25];
    int n_sphere = (int) data[26];	// Nb of sphere
    int n_cylinder = (int) data[27];	// Nb of cylinder
    int n_cube = (int) data[28];	// Nb of cube
    double T_food_init = data[29];	// Initial temperature of the food
    double h_air = data[30];		// h_air

    int prop_per_obj = 10;         // Number of properties per object
    int prop_per_source = 7;       // Number of properties per power source
    std::vector<double> prop_obj(prop_per_obj*nb_obj);
    std::vector<double> prop_source(prop_per_source*nb_source);


    // Properties of the power sources
    FileR = fopen(argv[2],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Source file. \n");
    	return 1; 
    }
    for (i=0 ; i<prop_per_source*nb_source; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the source file. \n");
		return 1; 
  	  }	
   	 else{
		prop_source[i] = atof(chain);
   	 }
    }
    fclose(FileR);


    // Properties of the objects
    FileR = fopen(argv[3],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Object file. \n");
    	return 1; 
    }
    for (i=0 ; i<prop_per_obj*nb_obj; i++){
 	   if (fgets(chain, 150, FileR) == NULL){
		printf("Impossible to read the source file. \n");
		return 1; 
  	  }	
   	 else{
		prop_obj[i] = atof(chain);
   	 }
    }
    fclose(FileR);

    // Property of cuts
    std::vector<double> Cut(3);
    std::vector<double> Pos_cut(6);
    int N_cut;
    std::vector<double> step_cut;
    double data_cut[10];
    FileR = fopen(argv[4],"r");
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

    // Property of temporal probes
    double nb_probe;
    std::vector<double> Pos_probe;
    FileR = fopen(argv[5],"r");
    if(FileR == NULL){ 
    	printf("Impossible to open the Probe file. \n");
    	return 1; 
    }
    if (fgets(chain, 150, FileR) == NULL){
	printf("Impossible to read the Cuts file. \n");
	return 1; 
   }	
   else{
	nb_probe = atof(chain);
   }
   if(nb_probe!=0){
   	for(i=0;i<3*nb_probe;i++){
    		if (fgets(chain, 150, FileR) == NULL){
			printf("Impossible to read the Cuts file. \n");
			return 1; 
  		}	
 		else{
			Pos_probe.push_back(atof(chain)/dx);
  		}
    	}	 
   }

   // Sphere properties (Position of the center, radius and material)
   std::vector<double> prop_sphere;
	FileR = fopen(argv[6],"r");
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

    // Cylinder properties (Position of the center, radius, axis, length and material)
    std::vector<double> prop_cylinder;
	FileR = fopen(argv[7],"r");
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

    // Cube property (Position of the center, length in each direction and material)
    std::vector<double> prop_cube;
	FileR = fopen(argv[8],"r");
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
 

    // split work among processes
    if (get_my_rank() == 0)
        host_work(id, Lx, Ly, Lz, dx, dt, step_max,theta, nb_source, prop_source, nb_obj,prop_obj,BC,T_Dir,T_0,kheat_param,rho_param,cp_param,SR,Cut,Pos_cut,step_cut,nb_probe,Pos_probe,n_sphere,prop_sphere,n_cylinder,prop_cylinder,n_cube,prop_cube,  T_food_init,h_air);
    else
        slave_work(id,step_max);

    // finalise MUMPS/MPI
    end_MUMPS(id);
    MPI_Finalize();

    return 0;
}

// This function computes the right hand side of the system to be solved
void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt){
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
    if(geometry[i]!=0&&(geometry[i-Y*Z]==0||geometry[i+Y*Z]==0||geometry[i-1]==0||geometry[i+1]==0||geometry[i-Y]==0||geometry[i+Y]==0)){// Neuman
      Temp2[i] = -T_inf;
    }
    else{
		  Temp2[i]+=(dt*Source[i])/(rho[i]*cp[i]);		
    }
    //Temp2[i]+=(Source[i])/(rho[i]*cp[i]);
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
void insert_obj(std::vector<double> &temp, std::vector<double> &k_heat_x, std::vector<double> &k_heat_y, std::vector<double> &k_heat_z, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &properties, int X,int Y,int Z, double dx,std::vector<double> &geometry){
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
void insert_Source(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp){
	int i = 0;
  	int j = 0;
  	int k = 0;
	int l = 0;
	int prop_per_source = 7;
	
	for(l=0;l<nb_source;l++){
		double n_x_double = (prop_source[prop_per_source*l]/dx)+1;
		int n_x = (int) n_x_double;
		double pos_x = (prop_source[prop_per_source*l+3]/dx);	
		int i_min = (int)pos_x;
		i_min = i_min - (n_x/2);
		int i_max = i_min + n_x-1;
	
		double n_y_double = (prop_source[prop_per_source*l+1]/dx)+1;
		int n_y = (int) n_y_double;
		double pos_y = (prop_source[prop_per_source*l+4]/dx);	
		int j_min = (int)pos_y;
		j_min = j_min - (n_y/2);
		int j_max = j_min + n_y-1;
	
		double n_z_double = (prop_source[prop_per_source*l+2]/dx)+1;
		int n_z = (int) n_z_double;
		double pos_z = (prop_source[prop_per_source*l+5]/dx);	
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
void export_coupe(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,	double dx, int step){
	pos1 = pos1/dx ;
	pos2 = pos2/dx ;
	int pos1_int = (int) pos1;
	int pos2_int = (int) pos2;
	int i =0;
	char file_name[50] = "Cut";
	char stepnumber[20];
	snprintf(stepnumber,sizeof stepnumber,"%d", step);
	FILE *FileW;    	
	if(direction==1){// Cut along x	
		strcat(file_name,"_alongX_step");
		strcat(file_name,stepnumber);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");		
		for(i=0;i<Nx;i++){
			fprintf(FileW," %lf \n ",temp[pos1_int+pos2_int*Ny+i*Ny*Nz]);
		}
	}
	else if(direction==2){// Cut along y
		strcat(file_name,"_alongY_step");
		strcat(file_name,stepnumber);
		strcat(file_name,".txt");
		FileW = fopen(file_name,"w");
		for(i=0;i<Ny;i++){
			fprintf(FileW," %lf \n ",temp[i+pos2_int*Ny+pos1_int*Ny*Nz]);
		}
	}
	else if(direction==3){// Cut along z	
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
void export_probe(double nb_probe , std::vector<double> &probe,int step_max){
  int i = 0;
  int j = 0;
  FILE *FileW;
  for(i=0;i<nb_probe;i++){
    char file_name[50] = "Probe";
	  char probenumber[20];
	  snprintf(probenumber,sizeof probenumber,"%d", (i+1));
    strcat(file_name,probenumber);
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
	for(i_vec=0;i_vec<X*Y*Z;i_vec++){
  		Temp[i_vec] = T_0;	
 	 }

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
void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &k_heatx,std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,std::vector<double> &cp){	
	int i=0;
	int j=0;
	int k=0;

	if(P==2){
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z;k++){
					if(((i*dx)<=properties[3]+properties[0]/2)&&((i*dx)>=properties[3]-properties[0]/2)&&((j*dx)<=properties[4]+properties[1]/2)&&((j*dx)>=properties[4]-properties[1]/2)&&((k*dx)<=properties[5]+properties[2]/2)&&((k*dx)>=properties[5]-properties[2]/2)){
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
					if(((i*dx)-0.5<=properties[3]+properties[0]/2)&&((i*dx)-0.5>=properties[3]-properties[0]/2)&&((j*dx)<=properties[4]+properties[1]/2)&&((j*dx)>=properties[4]-properties[1]/2)&&((k*dx)<=properties[5]+properties[2]/2)&&((k*dx)>=properties[5]-properties[2]/2)){
						k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
					}
				}
			}
		}
		for(i=0;i<X;i++){
			for(j=0;j<Y+1;j++){
				for(k=0;k<Z;k++){
					if(((i*dx)<=properties[3]+properties[0]/2)&&((i*dx)>=properties[3]-properties[0]/2)&&((j*dx)-0.5<=properties[4]+properties[1]/2)&&((j*dx)-0.5>=properties[4]-properties[1]/2)&&((k*dx)<=properties[5]+properties[2]/2)&&((k*dx)>=properties[5]-properties[2]/2)){
						k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
					}
				}
			}
		}
		for(i=0;i<X;i++){
			for(j=0;j<Y;j++){
				for(k=0;k<Z+1;k++){
					if(((i*dx)<=properties[3]+properties[0]/2)&&((i*dx)>=properties[3]-properties[0]/2)&&((j*dx)<=properties[4]+properties[1]/2)&&((j*dx)>=properties[4]-properties[1]/2)&&((k*dx)-0.5<=properties[5]+properties[2]/2)&&((k*dx)-0.5>=properties[5]-properties[2]/2)){
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
					xp = i*dx;
					yp = j*dx;
					zp = k*dx;
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
					xp = i*dx-0.5;
					yp = j*dx;
					zp = k*dx;
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
					xp = i*dx;
					yp = j*dx-0.5;
					zp = k*dx;
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
					xp = i*dx;
					yp = j*dx;
					zp = k*dx+0.5;
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
					if(((properties[0]-i*dx)*(properties[0]-i*dx)+(properties[1]-j*dx)*(properties[1]-j*dx)+(properties[2]-k*dx)*(properties[2]-k*dx))<=properties[3]*properties[3]){
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
					if(((properties[0]-i*dx+0.5)*(properties[0]-i*dx+0.5)+(properties[1]-j*dx)*(properties[1]-j*dx)+(properties[2]-k*dx)*(properties[2]-k*dx))<=properties[3]*properties[3]){
						k_heatx[i*Y*Z+k*Y+j] = vec_k[val];
					}
				}
			}
		}
		for(k=0;k<Z;k++){
			for(j=0;j<Y+1;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-i*dx)*(properties[0]-i*dx)+(properties[1]-j*dx+0.5)*(properties[1]-j*dx+0.5)+(properties[2]-k*dx)*(properties[2]-k*dx))<=properties[3]*properties[3]){
						k_heaty[i*(Y+1)*Z+k*(Y+1)+j] = vec_k[val];
					}
				}
			}
		}
		for(k=0;k<Z+1;k++){
			for(j=0;j<Y;j++){
				for(i=0;i<X;i++){
					if(((properties[0]-i*dx)*(properties[0]-i*dx)+(properties[1]-j*dx)*(properties[1]-j*dx)+(properties[2]-k*dx+0.5)*(properties[2]-k*dx+0.5))<=properties[3]*properties[3]){
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
void rotate_Power_grid(std::vector<double> &Source_init,std::vector<double> &Source_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta){
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
					T_curr[i*Ny*Nz+k*Ny+j] = T_air;
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



 
