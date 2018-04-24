#include "vtl.h"
#include "vtlSPoints.h"
#include "elm.h"

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

#include "mpi.h"
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1]

// Tous les includes sont pas nécessaires, i'll edit later i guess

/*****************************************************************************************
			ELECTRO-MAGNETIC FUNCTIONS
*****************************************************************************************/

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
						V1[i*(j_max1)+j] = M1[i][j][0];
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
						V2[i*(j_max2)+j] = M2[i][j][0];
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
				for(k=k_min;k<k_max;k++){
					if( last1==1&&((Case==1 && k == k_max-last1)||(Case==2 && i == i_max-last1)||(Case==3&&j == j_max-last1))){
						temp1=0;
					}
					else{
						temp1 = E1_prev[i][j][k];
					}
					if(last2==1&&((Case==1 &&j  == j_max-last2)||(Case==2 && k == k_max-last2)||(Case==3 && i == i_max-last2))){
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
	else if(Case==2){
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

//This function place one or more objects inside the domain (To be supressed)
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
void init_geom_one_proc(std::vector<double> &e_rx,double***mu_r,std::vector<double> &e_ry,std::vector<double> &e_rz,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz,std::vector<double> &e_diel,std::vector<double> &e_diel_tot){int i;
int j;
int k;
	set_rel_perm_one_proc(e_rx,e_r_totx,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,0);
	set_rel_perm_one_proc(e_ry,e_r_toty,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,1);
	set_rel_perm_one_proc(e_rz,e_r_totz,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,2);
	set_rel_perm_one_proc(e_diel,e_diel_tot,i_min_proc,j_min_proc,k_min_proc,point_per_proc_x,point_per_proc_y,point_per_proc_z,lastx,lasty,lastz,Nx,Ny,Nz,3);

}

// This function make the objects rotate inside the electromagnetic grid
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,std::vector<double> &e_diel_tot,std::vector<double> &vec_e_diel, std::vector<double> &Temperature, std::vector<double> &vec_e_diel_hot,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change, double dx_th, double x_min_th, double y_min_th, double z_min_th,double X_th,double Y_th, double Z_th){
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

 rotate_rel_perm(e_r_totx,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,0, Temperature, vec_e_r_hot,vec_e_r_hot,Temp_phase_change, dx_th,x_min_th,y_min_th,z_min_th,X_th,Y_th, Z_th);
 rotate_rel_perm(e_r_toty,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,1, Temperature, vec_e_r_hot,vec_e_r_hot,Temp_phase_change, dx_th,x_min_th,y_min_th,z_min_th,X_th,Y_th, Z_th);
 rotate_rel_perm(e_r_totz,vec_er,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,2, Temperature, vec_e_r_hot,vec_e_r_hot,Temp_phase_change, dx_th,x_min_th,y_min_th,z_min_th,X_th,Y_th, Z_th);
 rotate_rel_perm(e_diel_tot,vec_e_diel,Nx, Ny,Nz,Lx,Ly,Lz,dx,nsphere,info_sphere,ncylinder,info_cylinder,ncube,info_cube,theta,3, Temperature, vec_e_diel_hot,vec_e_diel_hot,Temp_phase_change, dx_th,x_min_th,y_min_th,z_min_th,X_th,Y_th, Z_th);
}

void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx,double val,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz , std::vector<double> &vec_er,std::vector<double> &e_diel_tot, std::vector<double> &vec_e_diel,double T_food_init_th,std::vector<double> &vec_e_r_hot, std::vector<double> &vec_e_diel_hot,std::vector<double> &Temp_phase_change){

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
		place_cube(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cube(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cube(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cube(X,Y,Z,properties,geometry,dx,val,3,e_diel_tot ,vec_e_diel,T_food_init_th,vec_e_diel_hot,Temp_phase_change);
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
		place_cylinder(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cylinder(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cylinder(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_cylinder(X,Y,Z,properties,geometry,dx,val,3,e_diel_tot ,vec_e_diel,T_food_init_th,vec_e_diel_hot,Temp_phase_change);
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
		place_sphere(X,Y,Z,properties,geometry,dx,val,0,e_r_totx ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_sphere(X,Y,Z,properties,geometry,dx,val,1,e_r_toty ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_sphere(X,Y,Z,properties,geometry,dx,val,2,e_r_totz ,vec_er,T_food_init_th,vec_e_r_hot,Temp_phase_change);
		place_sphere(X,Y,Z,properties,geometry,dx,val,3,e_diel_tot ,vec_e_diel,T_food_init_th,vec_e_diel_hot,Temp_phase_change);
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
				else{	// Bilinear interpolation of the power
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

// This function places a cubic object inside the domain
void place_cube(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change) {
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
						if(T_food_init_th<Temp_phase_change[(int) val]){
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
						}
						else{
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_e_r_hot[(int) val];
						}
					}
				}
			}
		}
}

// This function places a cylindrical object inside the domain
void place_cylinder(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change){
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
						if(T_food_init_th<Temp_phase_change[(int) val]){
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
						}
						else{
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_e_r_hot[(int) val];
						}
					}
				}
				else if(properties[0]==1){
					if(((xp-xc)*(xp-xc)+(zp-zc)*(zp-zc)<=r*r) && yp<=yc+l/2 && yp>= yc-l/2){
						if(T_food_init_th<Temp_phase_change[(int) val]){
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
						}
						else{
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_e_r_hot[(int) val];
						}
					}
				}
				else if(properties[0]==2){
					if(((xp-xc)*(xp-xc)+(yp-yc)*(yp-yc)<=r*r) && zp<=zc+l/2 && zp>= zc-l/2){
						if(T_food_init_th<Temp_phase_change[(int) val]){
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
						}
						else{
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_e_r_hot[(int) val];
						}
					}
				}
			}
		}
	}
}

// This function places a spherical object inside the domain
void place_sphere(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change){
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
						if(T_food_init_th<Temp_phase_change[(int) val]){
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_er[(int) val];
						}
						else{
							e_r_tot[i*(Y+yy)*(Z+zz)+k*(Y+yy)+j] = vec_e_r_hot[(int) val];
						}
					}
				}
			}
		}
}

// This function set the grid of relative permittivity on the current process
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

// This function rotate the total grid of relative permitivitty
void rotate_rel_perm(std::vector<double> &e_r_tot,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,int component, std::vector<double> &Temperature, std::vector<double> &vec_e_diel_hot,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change, double dx_th, double x_min_th, double y_min_th, double z_min_th,double X_th,double Y_th, double Z_th){
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
	double x1;
	double y1;
	double z1;
	int i1;
	int j1;
	double xi;
	double eta;

	//#pragma omp parallel for default(shared) private(i,j,k,x_after,y_after,z_after,x_before,y_before,xc,yc,zc,r,l)
	for(i=0;i<Nx+xx;i++){
		for(j=0;j<Ny+yy;j++){
			for(k=0;k<Nz+zz;k++){
        if(component==3){
          e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = 0;
        }
        else{
				  e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = 1;
        }
				// Coordinate after rotation
				x_after = i*dx-0.5*xx*dx;
				y_after = j*dx-0.5*yy*dx;
				z_after = k*dx-0.5*zz*dx;
				// Coordinate before rotation
				x_before = xrot + (x_after-xrot)*cos(theta) + (y_after-yrot)*sin(theta);
				y_before = yrot - (x_after-xrot)*sin(theta)+(y_after-yrot)*cos(theta);

				x1 = (x_before-x_min_th)/dx_th;
				y1 = (y_before-y_min_th)/dx_th;
				i1 = (int) x1;
				j1 = (int) y1;
				x1 = i1*dx_th;
				y1 = j1*dx_th;
				xi =-1+2*((x_before-x_min_th)-x1)/dx_th;
				eta =-1+2*((y_before-y_min_th)-y1)/dx_th;
				double Temp;


				for(object=0;object<ncube;object++){//	Cube
					if(((x_before)<=info_cube[7*object+3]+info_cube[7*object+0]/2)&&((x_before)>=info_cube[7*object+3]-info_cube[7*object+0]/2)&&((y_before)<=info_cube[7*object+4]+info_cube[7*object+1]/2)&&((y_before)>=info_cube[7*object+4]-info_cube[7*object+1]/2)&&((z_after)<=info_cube[7*object+5]+info_cube[7*object+2]/2)&&((z_after)>=info_cube[7*object+5]-info_cube[7*object+2]/2)){
						Temp = (1-xi)*(1-eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1-eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1+eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1+1)]+(1-xi)*(1+eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1+1)];
						Temp = Temp/4;
						if(Temp<Temp_phase_change[(int) info_cube[7*object+6]]){
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cube[7*object+6]];
						}
						else{
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_e_r_hot[(int) info_cube[7*object+6]];
						}
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
							Temp = (1-xi)*(1-eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1-eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1+eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1+1)]+(1-xi)*(1+eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1+1)];
							Temp = Temp/4;
              						if(Temp<Temp_phase_change[(int) info_cube[7*object+6]]){
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
							}
							else{
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_e_r_hot[(int) info_cylinder[7*object+5]];
							}
						}
					}
					else if(info_cylinder[7*object]==1){
						if(((x_before-xc)*(x_before-xc)+(z_after-zc)*(z_after-zc)<=r*r) && y_before<=yc+l/2 && y_before>= yc-l/2){
							Temp = (1-xi)*(1-eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1-eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1+eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1+1)]+(1-xi)*(1+eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1+1)];
							Temp = Temp/4;
							if(Temp<Temp_phase_change[(int) info_cube[7*object+6]]){
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
							}
							else{
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_e_r_hot[(int) info_cylinder[7*object+5]];
							}
						}
					}
					else if(info_cylinder[7*object]==2){
						if(((x_before-xc)*(x_before-xc)+(y_before-yc)*(y_before-yc)<=r*r) && z_after<=zc+l/2 && z_after>= zc-l/2){
							Temp = (1-xi)*(1-eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1-eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1+eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1+1)]+(1-xi)*(1+eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1+1)];
							Temp = Temp/4;
							if(Temp<Temp_phase_change[(int) info_cube[7*object+6]]){
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_cylinder[7*object+5]];
							}
							else{
								e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_e_r_hot[(int) info_cylinder[7*object+5]];
							}
						}
					}
				}

				for(object=0;object<nsphere;object++){//	Sphere
					if(((info_sphere[5*object+0]-x_before)*(info_sphere[5*object+0]-x_before)+(info_sphere[5*object+1]-y_before)*(info_sphere[5*object+1]-y_before)+(info_sphere[5*object+2]-z_after)*(info_sphere[5*object+2]-z_after))<=info_sphere[5*object+3]*info_sphere[5*object+3]){
						Temp = (1-xi)*(1-eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1-eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1)] + (1+xi)*(1+eta)*Temperature[(i1+1)*Y_th*Z_th+(k)*Y_th+(j1+1)]+(1-xi)*(1+eta)*Temperature[(i1)*Y_th*Z_th+(k)*Y_th+(j1+1)];
						Temp = Temp/4;
           					if(Temp<Temp_phase_change[(int) info_sphere[4*object+4]]){
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_er[(int) info_sphere[5*object+4]];
						}
						else{
							e_r_tot[i*(Ny+yy)*(Nz+zz)+k*(Ny+yy)+j] = vec_e_r_hot[(int) info_sphere[5*object+4]];
						}
					}
				}
			}
		}
	}
}

// This function initialise the value of a std vector
void set_vec(std::vector<double> &vec, int nbp, double val){
  int i;
  for(i=0;i<nbp;i++){
    vec[i] = val;
  }
}

// This function extract a cut of the electro magnetic calculation
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
  double z_min = k_min*dx - 0.5*zz*dx;
  double x_max = (i_max+xx*lastx)*dx - 0.5*xx*dx;
  double y_max = (j_max+yy*lasty)*dx - 0.5*yy*dx;
  double z_max = (k_max+zz*lastz)*dx - 0.5*zz*dx;

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

// This function export the power grid in a txt file (Will not be used after coupling is done)
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

void export_temp_probe_electro(std::vector<double> &vec,int step,char *Filename){
	int i =0;
	FILE *FileW;
	FileW = fopen(Filename,"w");
	for(i=0;i<step;i++){
		fprintf(FileW," %lf \n ",vec[i]);
	}
	fclose(FileW);	
}
