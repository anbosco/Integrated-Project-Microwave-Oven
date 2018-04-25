#ifndef ELM_H
#define ELM_H

#include "vtl.h"
#include <vector>

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
void init_geom_one_proc(std::vector<double> &e_rx,double***mu_r,std::vector<double> &e_ry,std::vector<double> &e_rz,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &mu_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz,std::vector<double> &e_diel,std::vector<double> &e_diel_tot);
void rotate_geometry(std::vector<double> &geometry_init,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,std::vector<double> &e_diel_tot,std::vector<double> &vec_e_diel, std::vector<double> &Temperature, std::vector<double> &vec_e_diel_hot,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change, double dx_th, double x_min_th, double y_min_th, double z_min_th,double X_th,double Y_th, double Z_th);
void place_geometry(int X,int Y, int Z, std::vector<double> &properties, int P,std::vector<double> &geometry, double dx, double val,std::vector<double> &e_r_totx,std::vector<double> &e_r_toty,std::vector<double> &e_r_totz , std::vector<double> &vec_er,std::vector<double> &e_diel_tot, std::vector<double> &vec_e_diel,double T_food_init_th,std::vector<double> &vec_e_r_hot, std::vector<double> &vec_e_diel_hot,std::vector<double> &Temp_phase_change);
void place_cube(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change);
void place_cylinder(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change);
void place_sphere(int X,int Y, int Z, std::vector<double> &properties,std::vector<double> &geometry, double dx,double val, int component,std::vector<double> &e_r_tot , std::vector<double> &vec_er,double T_food_init_th,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change);
void set_rel_perm_one_proc(std::vector<double> &e_r,std::vector<double> &e_r_tot,int i_min_proc,int j_min_proc,int k_min_proc,int point_per_proc_x,int point_per_proc_y,int point_per_proc_z,int lastx,int lasty,int lastz,int Nx, int Ny, int Nz, int comp);
void rotate_rel_perm(std::vector<double> &e_r_tot,std::vector<double> &vec_er,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, int nsphere,std::vector<double> &info_sphere, int ncylinder,std::vector<double> &info_cylinder, int ncube,std::vector<double> &info_cube, double theta,int component, std::vector<double> &Temperature, std::vector<double> &vec_e_diel_hot,std::vector<double> &vec_e_r_hot,std::vector<double> &Temp_phase_change, double dx_th, double x_min_th, double y_min_th, double z_min_th,double X_th,double Y_th, double Z_th);
void set_vec(std::vector<double> &vec, int nbp, double val);
void export_coupe(int direction, int component, double pos1,double pos2,int Nx_tot,int Ny_tot,int Nz_tot,double***M,double dx,int step,int myrank, int i_min,int i_max,int j_min ,int j_max ,int k_min ,int k_max ,int Nx,int Ny,int Nz,int lastx,int lasty,int lastz);
void export_power_thermo(std::vector<double> &Power_tot,int Nx,int Ny,int Nz);
void rotate_Power_grid(std::vector<double> &Power_electro,std::vector<double> &Power_thermo,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta);
void export_temp_probe_electro(std::vector<double> &vec,int step,const char *Filename);




#endif
