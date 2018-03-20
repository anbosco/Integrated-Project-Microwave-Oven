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

void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int NNZ,std::vector<double> &rho, std::vector<double> &cp);
void Compute_a_T0(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0);
void insert_obj(std::vector<double> &temp, std::vector<double> &k_heat, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &prop_obj, int X,int Y,int Z, double dx);
void insert_Source(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp);
void export_coupe(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,	double dx);

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

    std::cout << "[" << get_my_rank()
              << "] Call the MUMPS package (analyse, factorization and solve)." << std::endl;
    if(step==1){
  	  id.job = 6;
     }
     else {
       id.job = 3;
     }
    dmumps_c(&id);
    check_MUMPS(id);
}

void host_work(DMUMPS_STRUC_C &id,double Lx,double Ly,double Lz,double delta_x,double delta_t,int step_max,int nb_source, std::vector<double> &prop_source, int nb_obj,std::vector<double> &prop_obj,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0,double kheat_param,double rho_param,double cp_param,int SR){   
    SPoints grid;

    // setup grid

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

    // Declaration of MUMPS variable
    MUMPS_INT n = X*Y*Z;
    std::vector<MUMPS_INT> irn;
    std::vector<MUMPS_INT> jcn;
    std::vector<double> a;
    std::vector<double> b;
    
    // Variables used to solve the system
    std::vector<double> Temp(n);
    grid.scalars["Temp"] = &Temp;
    std::vector<double> Temp2(n);    
    std::vector<double> Source(n);
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
   
    std::vector<double> k_heat(n);
    std::vector<double> rho(n);
    std::vector<double> cp(n);
    std::vector<double> constant(n);
  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<X*Y*Z;i++){ 
 	k_heat[i] = kheat_param;
	rho[i] = rho_param;
	cp[i] = cp_param;
  }  

  // Insertion of one or more objects inside the domain
  if(nb_obj!=0){
  	insert_obj(Temp, k_heat, rho, cp,nb_obj, prop_obj, X, Y, Z, delta_x);
  }  

  // Insertion of one or more power source inside the domain
  if(nb_source!=0){
  	insert_Source(Source,nb_source, prop_source, X,Y,Z, delta_x, rho, cp);
  }

 #pragma omp parallel for default(shared) private(i)
  for(i=0;i<X*Y*Z;i++){
  	constant[i] = (0.5)*(k_heat[i]*delta_t)/(rho[i]*cp[i]*delta_x*delta_x);
  }

  // Computation of the matrix and of the initial temperature
  Compute_a_T0(irn ,jcn, X, Y, Z, ip_h, jp_h, kp_h, lastx_h, lasty_h, lastz_h, a, b,Temp,constant,BC,T_Dir,T_0);
  MUMPS_INT8 nnz =  irn.size();

  // Modification of the temperature in the region corresponding to the objects
  if(nb_obj!=0){
  	insert_obj(Temp, k_heat, rho, cp,nb_obj, prop_obj, X, Y, Z, delta_x);
  }
    
    // Preparation of MUMPS job
    id.n = n;
    id.nnz = a.size();
    id.irn = &irn[0];
    id.jcn = &jcn[0];
    id.a = &a[0];

    int step = 1;
    // save results to disk
    export_spoints_XML("laplace", step, grid, grid, Zip::ZIPPED, X, Y, Z, 1);

    while(step<step_max){
	/*if(step==2){ // Used to create a pulse of power
		for(i=0;i<X*Y*Z;i++){
			Source[i] = 0;		// To be parametrized!
		}
	}*/

	//Computation of the right hand side
	Compute_RHS(b,irn,jcn,Temp,Source,Temp2,X,Y,Z, nnz, rho, cp); 
    	id.rhs = &Temp[0];

	// Resolution of the system
    	solve_MUMPS(id,step);
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i] = id.rhs[i];
	}
	if(step == 400){// To extract a cut
		export_coupe(1,0.15,0.15, X, Y, Z, Temp, delta_x);		//To be parametrized!
	}
	step++;

        // Save results to disk if needed	
    	if(step%SR==0){
    		export_spoints_XML("laplace", step, grid, grid, Zip::ZIPPED, X, Y,  Z, 1);
  	  }
    }  
}

void slave_work(DMUMPS_STRUC_C &id, int step_max)
{
 int step = 1;
    while(step<step_max){
   	 solve_MUMPS(id,step);  
	 step++;
    }
}

int main(int argc, char *argv[])
{
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

    double data[25];	
    char chain[150];
    for (i=0 ; i<25; i++){
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
    T_0 = data[20];
    double kheat_param = data[21];
    double rho_param = data[22];
    double cp_param = data[23];
    double S = data[24];    // Sampling rate
    int SR = (int) 1/S;
    printf("        SR = %d    \n", SR);


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
    	printf("Impossible to open the Source file. \n");
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

    // split work among processes
    if (get_my_rank() == 0)
        host_work(id, Lx, Ly, Lz, dx, dt, step_max, nb_source, prop_source, nb_obj,prop_obj,BC,T_Dir,T_0,kheat_param,rho_param,cp_param,SR);
    else
        slave_work(id,step_max);

    // finalise MUMPS/MPI
    end_MUMPS(id);
    MPI_Finalize();

    return 0;
}

// This function computes the right hand side of the system to be solved
void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp){
	int i = 0;
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
		Temp2[i]+=(Source[i])/(rho[i]*cp[i]);		
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i]=Temp2[i];
	}
}

// This function imposes the boundary conditions, computes the A matrix and set an initial temperature over the domain
void Compute_a_T0(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0){
int i_vec = 0;
for(i_vec=0;i_vec<X*Y*Z;i_vec++){
  	Temp[i_vec] = T_0;	
  }
  for(i_vec=0;i_vec<X*Y*Z;i_vec++){
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
			a.push_back(1+6*constant[i_vec]);
			b.push_back(1-6*constant[i_vec]);
				
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);
				
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec+2);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);
					
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec+Y+1);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);
				
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec-Y+1);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);
				
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec+Y*Z+1);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);
					
			irn.push_back(i_vec+1);
			jcn.push_back(i_vec-Y*Z+1);
			a.push_back(-constant[i_vec]);
			b.push_back(constant[i_vec]);      
		}

	} 
}

// This function inserts one or more objects inside the domain
void insert_obj(std::vector<double> &temp, std::vector<double> &k_heat, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &prop_obj, int X,int Y,int Z, double dx){
	int i = 0;
  	int j = 0;
  	int k = 0;
	int l = 0;
	int prop_per_obj = 10;
	
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
					temp[i*Y*Z+j+k*Y]= prop_obj[prop_per_obj*l+6];
					k_heat[i*Y*Z+j+k*Y]= prop_obj[prop_per_obj*l+7];
					rho[i*Y*Z+j+k*Y]= prop_obj[prop_per_obj*l+8];
					cp[i*Y*Z+j+k*Y]= prop_obj[prop_per_obj*l+9];					
				}				
			}
		}
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
					Source[i*Y*Z+j+k*Y]= prop_source[prop_per_source*l+6]/(rho[i*Y*Z+j+k*Y]*cp[i*Y*Z+j+k*Y]);					
				}				
			}
		}
	}
}

// This function export the value of the temperature on a line inside the domain directed along a given direction (This function is still in progress)
void export_coupe(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,	double dx){
	pos1 = pos1/dx ;
	pos2 = pos2/dx ;
	int pos1_int = (int) pos1;
	int pos2_int = (int) pos2;
	int i =0;
	printf("%d %d \n",pos1_int,pos2_int);
	FILE *FileW;
    	FileW = fopen("Cut.txt","w");
	if(direction==1){	
		for(i=0;i<Nx;i++){
			fprintf(FileW," %lf \n ",temp[pos1_int+pos2_int*Ny+i*Ny*Nz]);
		}
	}
	fclose(FileW);
}









