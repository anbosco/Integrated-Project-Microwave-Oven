#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <cassert>


typedef struct Material{
	std::string name;
	int id;
	double rho;
	double rhoHot;
	double k;
	double kHot;
  double cp;
  double cpHot;
  double TempPhaseChange;
	double er;
	double erHot;
	double ur;
	double urHot;
	double eDiel;
	double eDielHot;
}Material;

Material ChoseMaterial(int ID);


#endif // MATERIAL_H
