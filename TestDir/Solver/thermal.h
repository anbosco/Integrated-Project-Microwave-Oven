#ifndef THERMAL_H
#define THERMAL_H

#include "vtl.h"
#include "dmumps_c.h"
#include <vector>
#include "material.h"

void main_th(std::vector<double> &Source_elec, std::vector<double> &Temperature,
  std::vector<int> &BC, std::vector<double> &T_Dir, double T_0,double dx,double h_air, double Lx,
  double Ly, double Lz, double dt, int step_max, int nb_source, int SR, double theta, int n_sphere,
  int n_cylinder, int n_cube, std::vector<double> &prop_sphere, std::vector<double> &prop_cylinder,
  std::vector<double> &prop_cube, double T_food_init, double x_min_th, double y_min_th,
  double z_min_th, double dx_electro, double Lx_electro, double Ly_electro, double Lz_electro,
  int prop_per_source, std::vector<double> &prop_source, std::vector<double> &Cut,
  std::vector<double> &Pos_cut, int N_cut,std::vector<double> &step_cut, int nb_probe,
  std::vector<double> &Pos_probe, DMUMPS_STRUC_C &id,std::vector<double> &k_heat_x,
  std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &rho,
  std::vector<double> &cp,std::vector<double> &constant,std::vector<double> &geometry,
  int step_pos,int thermo_domain);

void Compute_RHS(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain, std::vector<int> &BC, double h);
void Compute_RHS_steady(std::vector<double> &pre_mat, std::vector<int> &irn , std::vector<int> &jcn , std::vector<double> &Temp, std::vector<double> &Source, std::vector<double> &Temp2, int X, int Y, int Z, int nnz ,std::vector<double> &rho, std::vector<double> &cp,std::vector<double> &geometry,double dt,int thermo_domain, std::vector<int> &BC,std::vector<double> &T_Dir, double h);
void Compute_a_T0_steady(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h);
void insert_obj_th(std::vector<double> &temp, std::vector<double> &k_heat_x, std::vector<double> &k_heat_y, std::vector<double> &k_heat_z, std::vector<double> &rho, std::vector<double> &cp,int nb_obj, std::vector<double> &prop_obj, int X,int Y,int Z, double dx,std::vector<double> &geometry);
void insert_Source_th(std::vector<double> &Source,int nb_source, std::vector<double> &prop_source, int X,int Y,int Z, double dx, std::vector<double> &rho, std::vector<double> &cp,double x_min_th,double y_min_th,double z_min_th);
void export_coupe_th(int direction, double pos1, double pos2, int Nx, int Ny, int Nz, std::vector<double> &temp,double dx, int step, double x_min_th, double y_min_th, double z_min_th);
void export_probe_th(double nb_probe , std::vector<double> &probe,int step_max,int step_pos);
void set_kheat(int Case,int X,int Y,int Z, std::vector<double> &properties,int l,double dx,std::vector<double> &k_heat);
void Compute_a_T0_2(std::vector<int> &irn , std::vector<int> &jcn, int X, int Y, int Z,std::vector<int> &ip_h,std::vector<int> &jp_h,std::vector<int> &kp_h,std::vector<int> &lastx_h,std::vector<int> &lasty_h,std::vector<int> &lastz_h, std::vector<double> &a, std::vector<double> &b,std::vector<double> &Temp,std::vector<double> &constant,std::vector<int> &BC,std::vector<double> &T_Dir,double T_0, double theta,std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,std::vector<double> &geometry,double dx, double h);
void place_geometry_th(int X,int Y, int Z, std::vector<double> &properties, int P,
  std::vector<double> &geometry, double dx,double val,std::vector<double> &k_heatx,
  std::vector<double> &k_heaty,std::vector<double> &k_heatz,std::vector<double> &rho,
  std::vector<double> &cp, double x_min_th, double y_min_th, double z_min_th,
  std::vector<double> &Temperature, Material geom);
void set_T0(std::vector<double> &Temp,std::vector<double> &geometry,double T_0,double T_init_food,int  X,int  Y,int  Z,std::vector<int> &BC,std::vector<double> &T_Dir);
void rotate_Power_grid_th(std::vector<double> &Source_init,std::vector<double> &Source_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta);
void rotate_T_grid(std::vector<double> &T_init,std::vector<double> &T_curr,int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double dx, double theta, double T_air);
void set_source_from_elec(std::vector<double> &Source,std::vector<double> &Source_elec,double x_min_th,double y_min_th,double z_min_th,double dx,double dx_electro,int X_th,int Y_th,int Z_th,int X_elec,int Y_elec,int Z_elec);

void place_cube_th(int X, int Y, int Z, double xx, double yy, double zz, double x_min_th,
  double y_min_th, double z_min_th, std::vector<double> &M ,std::vector<double> &vec_val,
  double dx, std::vector<double> &properties, int val,
  std::vector<double> &Temp_phase_change, std::vector<double> &Temperature);

void place_cylinder_th(int X,int Y,int Z, double xx, double yy, double zz, double x_min_th,
    double y_min_th, double z_min_th,std::vector<double> &M, std::vector<double> &vec_val,
    double dx,std::vector<double> &properties,double val,
    std::vector<double> &Temp_phase_change,std::vector<double> &Temperature);

void place_sphere_th(int X, int Y, int Z, double xx, double yy, double zz, double x_min_th,
    double y_min_th, double z_min_th,std::vector<double> &M, std::vector<double> &vec_val,
    double dx, std::vector<double> &properties, double val,
    std::vector<double> &Temp_phase_change,std::vector<double> &Temperature);
int get_my_rank();
void check_MUMPS(DMUMPS_STRUC_C &id);
void init_MUMPS(DMUMPS_STRUC_C &id);
void end_MUMPS(DMUMPS_STRUC_C &id);
void solve_MUMPS(DMUMPS_STRUC_C &id, int step);
void host_work(DMUMPS_STRUC_C &id,double Lx,double Ly,double Lz,double delta_x,double delta_t,
  int step_max,double theta,int nb_source, std::vector<double> &prop_source,std::vector<int> &BC,
  std::vector<double> &T_Dir,double T_0,int SR,std::vector<double> &Cut, std::vector<double> &Pos_cut,
   std::vector<double> &step_cut, double nb_probe, std::vector<double> &Pos_probe,int n_sphere,
   std::vector<double> &prop_sphere,int n_cylinder,std::vector<double> &prop_cylinder,int n_cube,
   std::vector<double> &prop_cube, double T_init_food,double h_air,double x_min_th,double y_min_th,
   double z_min_th,double dx_electro,int X_elec,int Y_elec,int Z_elec,std::vector<double> &Source_elec,
   std::vector<double> &k_heat_x,std::vector<double> &k_heat_y,std::vector<double> &k_heat_z,
   std::vector<double> &rho,std::vector<double> &cp,std::vector<double> &constant,
   std::vector<double> &geometry,int step_pos,std::vector<double> &Temp,int thermo_domain);
void slave_work(DMUMPS_STRUC_C &id, int step_max, int thermo_domain);




#endif
