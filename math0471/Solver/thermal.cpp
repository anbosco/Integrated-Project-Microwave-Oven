#include "vtl.h"
#include "vtlSPoints.h"
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

#include "mpi.h"
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1]

using namespace vtl;



/*****************************************************************************************
				THERMIC FUNCTIONS
*****************************************************************************************/

int get_my_rank(){
    int myid;
    int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    return myid;
}

void check_MUMPS(DMUMPS_STRUC_C &id){
    if (id.infog[0] < 0){
        std::cout << "[" << get_my_rank() << "] MUMPS Error:\n";
        std::cout << "\tINFOG(1)=" << id.infog[0] << '\n';
        std::cout << "\tINFOG(2)=" << id.infog[1] << std::endl;
    }
}

void init_MUMPS(DMUMPS_STRUC_C &id){
    id.comm_fortran = -987654; //USE_COMM_WORLD;
    id.par = 1;                // 1=host involved in factorization phase
    id.sym = 0;                // 0=unsymmetric
    id.job = -1;
    std::cout << "[" << get_my_rank() << "] Init MUMPS package." << std::endl;
    dmumps_c(&id);
    check_MUMPS(id);
}

void end_MUMPS(DMUMPS_STRUC_C &id){
    id.job = -2;
    std::cout << "[" << get_my_rank() << "] Terminate MUMPS instance." << std::endl;
    dmumps_c(&id);
    check_MUMPS(id);
}

void solve_MUMPS(DMUMPS_STRUC_C &id, int step){

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
    

    // setup source grid
    SPoints grid2;
    grid2.o = Vec3d(10.0, 10.0, 10.0); // origin
    Vec3d L2(0.3, 0.3, 0.3);        // box dimensions    
    grid2.np1 = Vec3i(0, 0, 0);    // first index
    grid2.np2 = Vec3i(X_elec-1, Y_elec-1, Z_elec-1); // last index
    grid2.dx = L2 / (grid2.np() - 1); // compute spacing
    grid2.scalars["Power"] = &Source_elec;
    

    // setup thermic grid
    SPoints grid;
    grid.o = Vec3d(x_min_th+10.0, y_min_th+10.0, z_min_th+10.0); // origin
    Vec3d L(Lx, Ly, Lz);        // box dimensions
    int X = (int) (Lx/delta_x)+1;
    int Y = (int) (Ly/delta_x)+1;
    int Z = (int) (Lz/delta_x)+1;
    grid.np1 = Vec3i(0, 0, 0);    // first index
    grid.np2 = Vec3i(X-1, Y-1, Z-1); // last index
    grid.dx = L / (grid.np() - 1); // compute spacing
    int nbp = grid.nbp();
   

    // setup conductivity grid
    SPoints gridk;
    gridk.o = Vec3d(x_min_th+10.0-0.5*delta_x, y_min_th+10.0, z_min_th+10.0); // origin
    Vec3d Lk(Lx+delta_x, Ly, Lz);       // box dimensions    
    gridk.np1 = Vec3i(0, 0, 0);    // first index
    gridk.np2 = Vec3i(X, Y-1, Z-1); // last index
    gridk.dx = Lk / (gridk.np() - 1); // compute spacing
    gridk.scalars["Conductivity"] = &k_heat_x;

    // Loop variables
    int i = 0;
    int i_vec=0;
    int count=0;
    int col=0;    

    double theta_angle = 0;		// Used to make the rotation inside the thermic solver (to be suppressed after coupling is done)

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
    //if(nb_probe!=0){
    	//std::vector<double> probe(nb_probe*step_max);
 //  }
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
  
  // Insertion of one or more power source inside the domain (Different from the one coming from the electro magnetic calculation)
  if(nb_source!=0){
  	insert_Source_th(Source,nb_source, prop_source, X,Y,Z, delta_x, rho, cp,x_min_th,y_min_th,z_min_th);
  }
  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<n;i++){
    	Source_init[i] = Source[i];
  }
	

  // Computation of the matrix and of the initial temperature
  if(thermo_domain==0){
  Compute_a_T0_steady(irn ,jcn,  X,  Y,  Z,ip_h,jp_h,kp_h,lastx_h,lasty_h,lastz_h, a, b,Temp,constant,BC,T_Dir, T_0,theta, k_heat_x,k_heat_y,k_heat_z,geometry,delta_x,h_air);		// Old Boundary Conditions
  }
  else{
  Compute_a_T0_2(irn ,jcn,  X,  Y,  Z,ip_h,jp_h,kp_h,lastx_h,lasty_h,lastz_h, a, b,Temp,constant,BC,T_Dir, T_0,theta, k_heat_x,k_heat_y,k_heat_z,geometry,delta_x,h_air);		// New boundary conditions  
  }
  MUMPS_INT8 nnz =  irn.size();
    
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

   // Steady state Calculation
    if(thermo_domain==0){
      Compute_RHS_steady(b,irn,jcn,Temp,Source,Temp2,X,Y,Z, nnz, rho, cp,geometry,delta_t,thermo_domain, BC,T_Dir,h_air);
      id.rhs = &Temp[0];
     	solve_MUMPS(id,step);
      step++;
      // Export Steady state temperature field
      export_spoints_XML("Temperature_field", step+step_pos*step_max, grid, grid, Zip::ZIPPED, X, Y,  Z, 1);
      //Export Cut if needed
      if(Cut[0]==1){		// Cut along x
  		export_coupe_th(1, Pos_cut[0], Pos_cut[1], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
      }
      if(Cut[1]==1){		// Cut along y
        	export_coupe_th(2, Pos_cut[2], Pos_cut[3], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
      }
      if(Cut[2]==1){		//Cut along z
  		export_coupe_th(3, Pos_cut[4], Pos_cut[5], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
      }
    }


    else{
	// Transient Calculation
      int next_cut=0;
      while(step<step_max){
  	//Computation of the right hand side
  	Compute_RHS(b,irn,jcn,Temp,Source,Temp2,X,Y,Z, nnz, rho, cp,geometry,delta_t,thermo_domain, BC,h_air); 
      	
  	// Resolution of the system
  	id.rhs = &Temp[0];
      	solve_MUMPS(id,step);
  	
  	#pragma omp parallel for default(shared) private(i)
  	for(i=0;i<X*Y*Z;i++){
  		Temp[i] = id.rhs[i];
  	}
  
  	// Extraction of a cut if needed
  	if(step == step_cut[next_cut]){
  		next_cut++;
  		if(Cut[0]==1){	//Cut along x
  			export_coupe_th(1, Pos_cut[0], Pos_cut[1], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
  		}
  		if(Cut[1]==1){	// Cut along y 
  			export_coupe_th(2, Pos_cut[2], Pos_cut[3], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
  		}
  		if(Cut[2]==1){	// Cut along z
  			export_coupe_th(3, Pos_cut[4], Pos_cut[5], X, Y, Z, Temp, delta_x, step+step_pos*step_max,x_min_th,y_min_th,z_min_th);
  		}
  				
  	}	
  
         // The value at the temporal probe is registered
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
		export_spoints_XML("Heat conductivity field", step+step_pos*step_max, gridk, gridk, Zip::ZIPPED, X+1, Y,  Z, 1);     // To check the phase transition in conductivity
		//export_spoints_XML("Test_power", step+step_pos*step_max, grid2, grid2, Zip::ZIPPED, X_elec, Y_elec,  Z_elec, 1);  // To check if the source is correctly transferred.
  
  		/************* To be suppress when coupling is done ****************/
  
  		// Power rotation		
  		/*theta_angle = theta_angle + 3.141692/8;
  		rotate_Power_grid_th(Source_init,Source,X, Y, Z, Lx, Ly, Lz, delta_x, theta_angle);*/
  		/******************************************************************/
  				
    	  }		
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

void slave_work(DMUMPS_STRUC_C &id, int step_max, int thermo_domain){
 int step = 1;
 if(thermo_domain==0){
   solve_MUMPS(id,step); 
 }
 else{
    while(step<step_max){
   	 solve_MUMPS(id,step);  
	   step++;
    }
 }
}


/**********************************************************************
 			Main Thermic Solver	
 *********************************************************************/
void main_th(std::vector<double> &Source_elec, std::vector<double> &Temperature, std::vector<int> &BC, std::vector<double> &T_Dir, double T_0,double dx,double h_air, double Lx, double Ly, double Lz, double dt, int step_max, int nb_source, int SR, double theta, int n_sphere, int n_cylinder, int n_cube, std::vector<double> &prop_sphere, std::vector<double> &prop_cylinder, std::vector<double> &prop_cube, double T_food_init, double x_min_th, double y_min_th, double z_min_th, double dx_electro, double Lx_electro, double Ly_electro, double Lz_electro, int prop_per_source, std::vector<double> &prop_source, std::vector<double> &Cut, std::vector<double> &Pos_cut, int N_cut,std::vector<double> &step_cut, int nb_probe, std::vector<double> &Pos_probe, DMUMPS_STRUC_C &id,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &constant,std::vector<double> &geometry,int step_pos,int thermo_domain){
	int X_electro = (int) (Lx_electro/dx_electro)+1;
    	int Y_electro = (int) (Ly_electro/dx_electro)+1;
    	int Z_electro = (int) (Lz_electro/dx_electro)+1;
    // split work among processes
    if (get_my_rank() == 0)
        host_work(id, Lx, Ly, Lz, dx, dt, step_max,theta, nb_source, prop_source,BC,T_Dir,T_0,SR,Cut,Pos_cut,step_cut,nb_probe,Pos_probe,n_sphere,prop_sphere,n_cylinder,prop_cylinder,n_cube,prop_cube,  T_food_init,h_air, x_min_th, y_min_th, z_min_th,dx_electro, X_electro, Y_electro, Z_electro,Source_elec,k_heat_x,k_heat_y,k_heat_z,rho,cp,vec_k,vec_rho,vec_cp,constant,geometry,step_pos,Temperature,thermo_domain);
    else
        slave_work(id,step_max,thermo_domain);
}


// This function computes the right hand side of the system to be solved
void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain, std::vector<int> &BC, double h){
	int i = 0;
  int ii=0;
  int jj =0;
  int kk = 0;
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
	for(ii=1;ii<X-1;ii++){
   for(jj=1;jj<Y-1;jj++){
     for(kk=1;kk<Z-1;kk++){
         i = ii*X*Y+kk*Y+jj;
        if(geometry[i]==0&&((geometry[i+Y*Z]!=0&&BC[4]==0)||(geometry[i-Y*Z]!=0&&BC[5]==0)||(geometry[i+1]!=0&&BC[0]==0)||(geometry[i-1]!=0&&BC[1]==0)||(geometry[i+Y]!=0&&BC[2]==0)||(geometry[i-Y]!=0&&BC[3]==0))&& (thermo_domain==1)){// Neuman
          Temp2[i] = -h*T_inf;	// Associated to the convection condition on the surface of the food (activated only when it is specified that the heat equation has to be solved only inside the food).
        }
        else if((geometry[i]!=0&&((geometry[i+Y*Z]!=0||BC[5]==0)&&(geometry[i-Y*Z]!=0||BC[4]==0)&&(geometry[i+1]!=0||BC[1]==0)&&(geometry[i-1]!=0||BC[0]==0)&&(geometry[i+Y]!=0||BC[3]==0)&&(geometry[i-Y]!=0||BC[2]==0))) || thermo_domain==0){
    		  Temp2[i]+=(dt*Source[i])/(rho[i]*cp[i]);		
        }
      }
    }
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i]=Temp2[i];
	}
}

void Compute_RHS_steady(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain, std::vector<int> &BC,std::vector<double> &T_Dir, double h){	
  int i = 0;
  int ii=0;
  int jj =0;
  int kk = 0;
  double T_inf = 20;
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp2[i]=0;
	}
	#pragma omp parallel for default(shared) private(i,ii,jj,kk)
	 for(jj=1;jj<Y-1;jj++){
		 for(kk=1;kk<Z-1;kk++){
			for(ii=1;ii<X-1;ii++){    
				i = ii*X*Y+kk*Y+jj;
				if(geometry[i]==0&&((geometry[i+Y*Z]!=0&&BC[4]==0)||(geometry[i-Y*Z]!=0&&BC[5]==0)||(geometry[i+1]!=0&&BC[0]==0)||(geometry[i-1]!=0&&BC[1]==0)||(geometry[i+Y]!=0&&BC[2]==0)||(geometry[i-Y]!=0&&BC[3]==0))){// Neuman 
					/*if(geometry[i-1]!=0){		// Test for Neumann non homogene on only one face (Face 1)
						Temp2[i] = -1*T_inf;
					}        
					else{
						Temp2[i] = -h*T_inf;
					}*/
					Temp2[i] = -h*T_inf;	// Associated to the convection condition on the surface of the food.
				}
				else if(geometry[i]!=0&&(geometry[i+Y*Z]==0&&BC[5]==1)){
				  Temp2[i] = T_Dir[5];
				}
				else if(geometry[i]!=0&&(geometry[i-Y*Z]==0&&BC[4]==1)){
				  Temp2[i] = T_Dir[4];
				}
				else if(geometry[i]!=0&&(geometry[i+1]==0&&BC[1]==1)){
				  Temp2[i] = T_Dir[1];
				}
				else if(geometry[i]!=0&&(geometry[i-1]==0&&BC[0]==1)){
				  Temp2[i] = T_Dir[0];
				}
				else if(geometry[i]!=0&&(geometry[i+Y]==0&&BC[3]==1)){
				  Temp2[i] = T_Dir[3];
				}
				else if(geometry[i]!=0&&(geometry[i-Y]==0&&BC[2]==1)){
				  Temp2[i] = T_Dir[2];
				}
				else if((geometry[i]!=0&&((geometry[i+Y*Z]!=0||BC[5]==0)&&(geometry[i-Y*Z]!=0||BC[4]==0)&&(geometry[i+1]!=0||BC[1]==0)&&(geometry[i-1]!=0||BC[0]==0)&&(geometry[i+Y]!=0||BC[3]==0)&&(geometry[i-Y]!=0||BC[2]==0)))){ 
			    		  Temp2[i]+=(dt*Source[i])/(rho[i]*cp[i]);	
				}        
				else{        
				   Temp2[i] = T_inf;
        			}
      			}
  		}
	}
	#pragma omp parallel for default(shared) private(i)
	for(i=0;i<X*Y*Z;i++){
		Temp[i]=Temp2[i];
	}
}

// This function imposes the boundary conditions, computes the A matrix and set an initial temperature over the domain (The heat equation is solved over the whole domain)
void Compute_a_T0_steady(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h){
	int i_vec = 0;
	int i=0;
	int j = 0;
	int k =0;
	double T_inf = 20;

	for(j=0;j<Y;j++){		
		for(k=0;k<Z;k++){
			for(i=0;i<X;i++){	
				i_vec = i*Y*Z+k*Y+j;
				/**** For the point in air, we put either a 1 on the diagonal or a heat flux if the point is next to the object *****/
				if(geometry[i_vec]==0){
          				if(i==0||j==0||k==0||i==X-1||j==Y-1||k==Z-1){
		     				Temp[i_vec] = T_inf;
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(1);
	    					b.push_back(1);
          				}
          				else{ 
					// Flux in the y direction is imposed
					if(geometry[i_vec-1]!=0&&geometry[i_vec-2]!=0&&BC[1]==0){
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
	    					b.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-1);
	    					a.push_back(-h);
	    					b.push_back(-h);

						/*a.push_back(-1);		// Test for Neuman non homogene on only 1 face(Face 1)
	    					b.push_back(-1);*/

	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-2);
	    					a.push_back(k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
	    					b.push_back(k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
            				}
           				else if(geometry[i_vec+1]!=0&&geometry[i_vec+2]!=0&&BC[0]==0){
		      				irn.push_back(i_vec+1);
	   					  jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
	    					b.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+1);
	    					a.push_back(-h);
	    					b.push_back(-h);
	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+2);
	    					a.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
	    					b.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
            				}
            				// Flux in the z direction is imposed
            				else if(geometry[i_vec+Y]!=0&&geometry[i_vec+2*Y]!=0&&BC[2]==0){
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
	    					b.push_back(-k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+Y);
	    					a.push_back(-h);
	    					b.push_back(-h);
	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+2*Y);
	    					a.push_back(+k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
	    					b.push_back(+k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
            				}
           				else if(geometry[i_vec-Y]!=0&&geometry[i_vec-2*Y]!=0&&BC[3]==0){
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
	    					b.push_back(-k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-Y);
	    					a.push_back(-h);
	    					b.push_back(-h);
	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-2*Y);
	    					a.push_back(+k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
	    					b.push_back(+k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
            				}
					// Flux in the x direction is imposed
            				else if(geometry[i_vec+Y*Z]!=0&&geometry[i_vec+2*Y*Z]!=0&&BC[4]==0){
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
	    					b.push_back(-k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+Y*Z);
	    					a.push_back(-h);
	    					b.push_back(-h);
	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1+2*Y*Z);
	    					a.push_back(+k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
	    					b.push_back(+k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
            				}
            				else if(geometry[i_vec-Y*Z]!=0&&geometry[i_vec-2*Y*Z]!=0&&BC[5]==0){
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(-k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
	    					b.push_back(-k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
		    
		      				irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-Y*Z);
	    					a.push_back(-h);
	    					b.push_back(-h);
	    				
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1-2*Y*Z);
	    					a.push_back(+k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
	    					b.push_back(+k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
            				}
            				else{
	    					Temp[i_vec] = T_inf;
	    					irn.push_back(i_vec+1);
	    					jcn.push_back(i_vec+1);
	    					a.push_back(1);
	    					b.push_back(1);
            				}
           			}
			}
				/**********************************************************/

				/***** Imposition of the dirichlet boundary condition **********/
				else if(geometry[i_vec-Y*Z]==0&&geometry[i_vec+Y*Z]!=0&&BC[4]==1){ // face 4
            				// Dirichlet
  					irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				else if((geometry[i_vec+Y*Z]==0&&geometry[i_vec-Y*Z]!=0&&BC[5]==1)){ // face 5
		 			 // Dirichlet
		  			 irn.push_back(i_vec+1);
	  				 jcn.push_back(i_vec+1);
					 a.push_back(1);
	  				 b.push_back(1);
				}
				else if(geometry[i_vec-1]==0&&geometry[i_vec+1]!=0&&BC[0]==1){ // face 0
		      			// Dirichlet
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
        			}
				else if((geometry[i_vec+1]==0 && geometry[i_vec-1]!=0&&BC[1]==1)){ // face 1
         				// Dirichlet
           				irn.push_back(i_vec+1);
	  				jcn.push_back(i_vec+1);
	  				a.push_back(1);
	  				b.push_back(1); 
				}
				else if(geometry[i_vec-Y]==0&&geometry[i_vec+Y]!=0&&BC[2]==1){ // face2
            				// Dirichlet
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				else if((geometry[i_vec+Y]==0&&geometry[i_vec-Y]!=0&&BC[3]==1)){  // face 3
         				// Dirichlet
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				/***************************************************************/

				/********* Heat equation is solved inside the domain ***********/
					else{
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+1);
					a.push_back((constant[i_vec]*(k_heat_x[i*Y*Z+k*Y+j]+k_heat_x[(i+1)*Y*Z+k*Y+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]+k_heat_z[i*Y*(Z+1)+k*Y+j]+k_heat_z[i*Y*(Z+1)+(k+1)*Y+j])));			

					irn.push_back(i_vec+1);
					jcn.push_back(i_vec);
					a.push_back(-constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j]);		
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+2);
					a.push_back(-constant[i_vec]*k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+1]);		
					
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y+1);
					a.push_back(-constant[i_vec]*k_heat_z[i*Y*(Z+1)+(k+1)*Y+j]);		
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y+1);
					a.push_back(-constant[i_vec]*k_heat_z[i*Y*(Z+1)+k*Y+j]);				
			
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec+Y*Z+1);
					a.push_back(-constant[i_vec]*k_heat_x[(i+1)*Y*Z+k*Y+j]);		
				
					irn.push_back(i_vec+1);
					jcn.push_back(i_vec-Y*Z+1);
					a.push_back(-constant[i_vec]*k_heat_x[i*Y*Z+k*Y+j]);		
				}
			    /********************************************************************/
			}
		}
	}	
}

// This function inserts one or more sources inside the domain
void insert_Source_th(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp,double x_min_th,double y_min_th,double z_min_th){
	int i = 0;
  	int j = 0;
  	int k = 0;
	int l = 0;
	int prop_per_source = 7;
	double eta=0;

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
		for(j=b_inf_y;j<=b_sup_y;j++){
			for(k=b_inf_z;k<=b_sup_z;k++){
				for(i=b_inf_x;i<=b_sup_x;i++){
					eta = j*dx-(b_inf_y*dx);
					Source[i*Y*Z+j+k*Y]= prop_source[prop_per_source*l+6]*sin(eta*(3.141592/((b_sup_y-b_inf_y)*dx)));		// sinusoidal source
					//Source[i*Y*Z+j+k*Y]= prop_source[prop_per_source*l+6];		//constant source
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

// This function compute the A matrix and set the temperature everywhere at the temperature of the air (The heat equation is solved only inside the food and a convection condition is imposed on the food boundaries)
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
				/**** For the point in air, we put either a 1 on the diagonal or a heat flux if the point is next to the object *****/
				if(geometry[i_vec]==0){
          				if(i==0||j==0||k==0||i==1||j==1||k==1||i==X-1||j==Y-1||k==Z-1||i==X-2||j==Y-2||k==Z-2){
              				Temp[i_vec] = T_inf;
    					irn.push_back(i_vec+1);
    					jcn.push_back(i_vec+1);
    					a.push_back(1);
    					b.push_back(1);
          				}
          				else{ 
						//Flux imposed in the y direction
			    			if(geometry[i_vec+1]!=0&&geometry[i_vec+2]!=0&&BC[0]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
		    					b.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+1);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+2);
		    					a.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
		    					b.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j+2]/(2*dx));
			    			}			
			    			else if(geometry[i_vec-1]!=0&&geometry[i_vec-2]!=0&&BC[1]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
		    					b.push_back(-k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-1);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-2);
		    					a.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
		    					b.push_back(+k_heat_y[i*(Y+1)*Z+k*(Y+1)+j-1]/(2*dx));
			    			}
						// Fluw imposed in the z direction
			    			else if(geometry[i_vec+Y]!=0&&geometry[i_vec+2*Y]!=0&&BC[2]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
		    					b.push_back(-k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+Y);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+2*Y);
		    					a.push_back(+k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
		    					b.push_back(+k_heat_z[i*Y*(Z+1)+(k+2)*Y+j]/(2*dx));
			    			}
			    			else if(geometry[i_vec-Y]!=0&&geometry[i_vec-2*Y]!=0&&BC[3]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
		    					b.push_back(-k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-Y);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-2*Y);
		    					a.push_back(+k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
		    					b.push_back(+k_heat_z[i*Y*(Z+1)+(k-1)*Y+j]/(2*dx));
			    			}
						// Flux imposed in the x direction
			    			else if(geometry[i_vec+Y*Z]!=0&&geometry[i_vec+2*Y*Z]!=0&&BC[4]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
		    					b.push_back(-k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+Y*Z);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1+2*Y*Z);
		    					a.push_back(+k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
		    					b.push_back(+k_heat_x[(i+2)*Y*(Z)+(k)*Y+j]/(2*dx));
			    			}
			    			else if(geometry[i_vec-Y*Z]!=0&&geometry[i_vec-2*Y*Z]!=0&&BC[5]==0){
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(-k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
		    					b.push_back(-k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
			    
			      				irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-Y*Z);
		    					a.push_back(-h);
		    					b.push_back(-h);
		    				
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1-2*Y*Z);
		    					a.push_back(+k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
		    					b.push_back(+k_heat_x[(i-1)*Y*(Z)+(k)*Y+j]/(2*dx));
			    			}
			    			else{
		    					Temp[i_vec] = T_inf;
		    					irn.push_back(i_vec+1);
		    					jcn.push_back(i_vec+1);
		    					a.push_back(1);
		    					b.push_back(1);
			    			}
           				}
				}
				/**********************************************************/

				/***** Imposition of the dirichlet boundary condition **********/
				else if(geometry[i_vec-Y*Z]==0&&geometry[i_vec+Y*Z]!=0&&BC[4]==1){ // face 4
            				// Dirichlet
            				Temp[i_vec] = T_Dir[4];
  					irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				else if((geometry[i_vec+Y*Z]==0&&geometry[i_vec-Y*Z]!=0&&BC[5]==1)){ // face 5
         				// Dirichlet
           				Temp[i_vec] = T_Dir[5];
           				irn.push_back(i_vec+1);
  				 	jcn.push_back(i_vec+1);
				   	a.push_back(1);
  				 	b.push_back(1);
				}
				else if(geometry[i_vec-1]==0&&geometry[i_vec+1]!=0&&BC[0]==1){ // face 0
		      			// Dirichlet
            				Temp[i_vec] = T_Dir[0];
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
        			}
				else if((geometry[i_vec+1]==0 && geometry[i_vec-1]!=0&&BC[1]==1)){ // face 1
         				// Dirichlet
           				Temp[i_vec] = T_Dir[1];
           				irn.push_back(i_vec+1);
  				 	jcn.push_back(i_vec+1);
  				 	a.push_back(1);
  				 	b.push_back(1); 
				}
				else if(geometry[i_vec-Y]==0&&geometry[i_vec+Y]!=0&&BC[2]==1){ // face2
            				// Dirichlet
            				Temp[i_vec] = T_Dir[2];
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				else if((geometry[i_vec+Y]==0&&geometry[i_vec-Y]!=0&&BC[3]==1)){  // face 3
         				// Dirichlet
             				Temp[i_vec] = T_Dir[3];
            				irn.push_back(i_vec+1);
  					jcn.push_back(i_vec+1);
  					a.push_back(1);
  					b.push_back(1);
				}
				/***************************************************************/

				/********* Heat equation is solved inside the domain ***********/
				else{
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
			    /********************************************************************/
			}
		}
	}
}

// This function can place different geometry of objects inside the domain (Need to be parametrized!!)
void place_geometry_th(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &vec_k,std::vector<double> &vec_rho,std::vector<double> &vec_cp,std::vector<double> &k_heatx,std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,std::vector<double> &cp, double x_min_th, double y_min_th, double z_min_th, std::vector<double> &vec_rho_hot,std::vector<double> &vec_cp_hot,std::vector<double> &Temp_phase_change,std::vector<double> &Temperature,std::vector<double> &vec_k_hot){
	if(P==2){ // Cube
		place_cube_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, geometry ,vec_rho,dx,properties,val, vec_rho_hot,Temp_phase_change,Temperature);
		place_cube_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, rho ,vec_rho,dx,properties,val, vec_rho_hot,Temp_phase_change,Temperature);
		place_cube_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, cp ,vec_cp,dx,properties,val, vec_cp_hot,Temp_phase_change,Temperature);
		place_cube_th(X,Y,Z,1,0,0,x_min_th,y_min_th,z_min_th, k_heatx ,vec_k,dx,properties,val, vec_k_hot,Temp_phase_change,Temperature);
		place_cube_th(X,Y,Z,0,1,0,x_min_th,y_min_th,z_min_th, k_heaty ,vec_k,dx,properties,val, vec_k_hot,Temp_phase_change,Temperature);
		place_cube_th(X,Y,Z,0,0,1,x_min_th,y_min_th,z_min_th, k_heatz ,vec_k,dx,properties,val, vec_k_hot,Temp_phase_change,Temperature);

	}
	else if(P==1){ // Cylinder
		place_cylinder_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, geometry ,vec_rho ,dx,properties,val, vec_cp_hot,Temp_phase_change,Temperature);
		place_cylinder_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, rho ,vec_rho ,dx,properties,val, vec_rho_hot,Temp_phase_change,Temperature);
		place_cylinder_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, cp ,vec_cp ,dx,properties,val, vec_cp_hot,Temp_phase_change,Temperature);
		place_cylinder_th(X,Y,Z,1,0,0,x_min_th,y_min_th,z_min_th, k_heatx ,vec_k ,dx,properties,val, vec_k_hot,Temp_phase_change,Temperature);
		place_cylinder_th(X,Y,Z,0,1,0,x_min_th,y_min_th,z_min_th, k_heaty ,vec_k ,dx,properties,val,vec_k_hot,Temp_phase_change,Temperature);
		place_cylinder_th(X,Y,Z,0,0,1,x_min_th,y_min_th,z_min_th, k_heatz ,vec_k ,dx,properties,val, vec_k_hot,Temp_phase_change,Temperature);
	}
	else if(P==0){// Sphere
		place_sphere_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, geometry, vec_rho ,dx,properties,val, vec_cp_hot,Temp_phase_change,Temperature);
		place_sphere_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, rho, vec_rho ,dx,properties,val, vec_rho_hot,Temp_phase_change,Temperature);
		place_sphere_th(X,Y,Z,0,0,0,x_min_th,y_min_th,z_min_th, cp, vec_cp ,dx,properties,val, vec_cp_hot,Temp_phase_change,Temperature);
		place_sphere_th(X,Y,Z,1,0,0,x_min_th,y_min_th,z_min_th, k_heatx, vec_k ,dx,properties,val,vec_k_hot,Temp_phase_change,Temperature);
		place_sphere_th(X,Y,Z,0,1,0,x_min_th,y_min_th,z_min_th, k_heaty, vec_k ,dx,properties,val,vec_k_hot,Temp_phase_change,Temperature);
		place_sphere_th(X,Y,Z,0,0,1,x_min_th,y_min_th,z_min_th, k_heatz, vec_k ,dx,properties,val,vec_k_hot,Temp_phase_change,Temperature);
	}
}

// This function set the initial temperature of the object at the one specified in the input file
void set_T0(std::vector<double> &Temp,std::vector<double> &geometry,double T_0,double T_init_food,int  X,int  Y,int  Z ){
	int i = 0;
	int j = 0;
 	int k = 0;
	#pragma omp parallel for default(shared) private(i,j,k)	
	for(j=0;j<Y;j++){
		for(k=0;k<Z;k++){		
			for(i=0;i<X;i++){
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

// This function rotate the temperature grid (Just used to show how bad it is)
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

// This function interpolate the power grid coming from the electromagnetic calculation on the thermic source grid
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
	for(j=0;j<Y_th;j++){
		for(k=0;k<Z_th;k++){			
			for(i=0;i<X_th;i++){
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

				// z_inf (1-zeta)
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j] +(1-xi)*(1-eta)*(1-zeta)*Source_elec[(i1)*Z_elec*Y_elec+(j1)+(k1)*Y_elec] + (1+xi)*(1-eta)*(1-zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1)+(k1)*Y_elec] + (1-xi)*(1+eta)*(1-zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1+1)+(k1)*Y_elec] +( 1+xi)*(1+eta)*(1-zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1+1)+(k1)*Y_elec];
				// z_sup (1+zeta)
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j] +(1-xi)*(1-eta)*(1+zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1)+(k1+1)*Y_elec] + (1+xi)*(1-eta)*(1+zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1)+(k1+1)*Y_elec] + (1-xi)*(1+eta)*(1+zeta)*Source_elec[(i1)*Y_elec*Z_elec+(j1+1)+(k1+1)*Y_elec] +( 1+xi)*(1+eta)*(1+zeta)*Source_elec[(i1+1)*Y_elec*Z_elec+(j1+1)+(k1+1)*Y_elec];

				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j]/8;
				Source[i*Y_th*Z_th+k*Y_th+j] = Source[i*Y_th*Z_th+k*Y_th+j];
			}
		}
	}
}

// This function places a cubic object inside the thermal grid
void place_cube_th(int X, int Y, int Z, double xx, double yy, double zz, double x_min_th, double y_min_th, double z_min_th, std::vector<double> &M ,std::vector<double> &vec_val,double dx, std::vector<double> &properties, int val, std::vector<double> &vec_hot,std::vector<double> &Temp_phase_change,std::vector<double> &Temperature){
	int i = 0;
	int j = 0;
	int k = 0;
	double test_temp;
	for(j=0;j<Y+yy;j++){
		for(k=0;k<Z+zz;k++){
			for(i=0;i<X+xx;i++){							
				if(((x_min_th+(i*dx)-0.5*dx*xx)<=properties[3]+properties[0]/2)&&((x_min_th+i*dx-0.5*dx*xx)>=properties[3]-properties[0]/2)&&((y_min_th+j*dx-0.5*dx*yy)<=properties[4]+properties[1]/2)&&((y_min_th+j*dx-0.5*dx*yy)>=properties[4]-properties[1]/2)&&((z_min_th+k*dx-0.5*dx*zz)<=properties[5]+properties[2]/2)&&((z_min_th+k*dx-0.5*dx*zz)>=properties[5]-properties[2]/2)){
					test_temp = (Temperature[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]+Temperature[(i-xx)*(Y+yy)*(Z+zz)+(k-zz)*(Y+yy)+(j-yy)])/2;
					if(test_temp<Temp_phase_change[val]){
						M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_val[val];
					}
					else{
						M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_hot[val];
					}
				}
			}
		}
	}
}

// This function places a cylindrical object inside the thermal grid
void place_cylinder_th(int X,int Y,int Z, double xx, double yy, double zz, double x_min_th, double y_min_th, double z_min_th,std::vector<double> &M,std::vector<double> &vec_val,double dx,std::vector<double> &properties,double val, std::vector<double> &vec_hot,std::vector<double> &Temp_phase_change,std::vector<double> &Temperature){
	int i = 0;
	int j = 0;
	int k = 0;
	double test_temp;
	double xc = properties[1];
	double yc = properties[2];
	double zc = properties[3];
	double r = properties[4];
	double l = properties[6];	
	for(j=0;j<Y+yy;j++){
		for(k=0;k<Z+zz;k++){
			for(i=0;i<X+xx;i++){
				double xp = x_min_th+i*dx-xx*0.5*dx;
				double yp = y_min_th+j*dx-yy*0.5*dx;
				double zp = z_min_th+k*dx-zz*0.5*dx;
				if(properties[0]==0){
					if(((yp-yc)*(yp-yc)+(zp-zc)*(zp-zc)<=r*r) && xp<=xc+l/2 && xp>= xc-l/2){
						test_temp = (Temperature[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]+Temperature[(i-xx)*(Y+yy)*(Z+zz)+(k-zz)*(Y+yy)+(j-yy)])/2;
						if(test_temp<Temp_phase_change[val])	{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_val[val];
						}
						else{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_hot[val];
						}
					}
				}
				else if(properties[0]==1){
					if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
						test_temp = (Temperature[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]+Temperature[(i-xx)*(Y+yy)*(Z+zz)+(k-zz)*(Y+yy)+(j-yy)])/2;
						if(test_temp<Temp_phase_change[val])	{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_val[val];
						}
						else{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_hot[val];
						}
					}
				}
				else if(properties[0]==2){
					if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
						test_temp = (Temperature[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]+Temperature[(i-xx)*(Y+yy)*(Z+zz)+(k-zz)*(Y+yy)+(j-yy)])/2;
						if(test_temp<Temp_phase_change[val])	{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_val[val];
						}
						else{
							M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_hot[val];
						}
					}
				}
			}
		}
	}
}

// This function places a spherical object inside the thermal grid
void place_sphere_th(int X, int Y, int Z, double xx, double yy, double zz, double x_min_th, double y_min_th, double z_min_th,std::vector<double> &M,std::vector<double> &vec_val, double dx, std::vector<double> &properties, double val, std::vector<double> &vec_hot,std::vector<double> &Temp_phase_change,std::vector<double> &Temperature){
	int i = 0;
	int j = 0;
	int k = 0;
	double test_temp;
	for(j=0;j<Y+yy;j++){
		for(k=0;k<Z+zz;k++){		
			for(i=0;i<X+xx;i++){
				if(((properties[0]-(x_min_th+i*dx-xx*0.5*dx))*(properties[0]-(x_min_th+i*dx-xx*0.5*dx))+(properties[1]-(y_min_th+j*dx-yy*0.5*dx))*(properties[1]-(y_min_th+j*dx-yy*0.5*dx))+(properties[2]-(z_min_th+k*dx-zz*0.5*dx))*(properties[2]-(z_min_th+k*dx-zz*0.5*dx)))<=properties[3]*properties[3]){
					test_temp = (Temperature[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]+Temperature[(i-xx)*(Y+yy)*(Z+zz)+(k-zz)*(Y+yy)+(j-yy)])/2;
					if(test_temp<Temp_phase_change[val])	{
						M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_val[val];
					}
					else{
						M[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j]= vec_hot[val];
					}
				}
			}
		}
	}
}
