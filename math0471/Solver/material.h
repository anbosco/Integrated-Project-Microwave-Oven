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
	std::vector<double> rho;
	std::vector<double> k;
  std::vector<double> cp;
  std::vector<double> TempPhaseChange;
	std::vector<double> er;
	std::vector<double> ur;
	std::vector<double> eDiel;
}Material;

Material ChoseMaterial(int ID);


#endif // MATERIAL_H
