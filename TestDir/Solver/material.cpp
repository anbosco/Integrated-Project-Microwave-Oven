#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <cassert>
#include "material.h"
#include<stdlib.h>



Material ChoseMaterial(int Val){

Material ReturnMaterial;

/*##############################################
##DEFINITION OF SOME MATERIALS USING STRUCTURE##
################################################*/


switch(Val){
  case 0:
  {
  // Air
  Material air;
  air.name = "air";
  air.rho = {1, 1};
  air.id = 0;
  air.k = {1, 1};
  air.cp = {1, 1};
  air.TempPhaseChange = {0};
  air.er = {1, 1};
  air.ur = {1, 1};
  air.eDiel = {0, 0};
  ReturnMaterial = air;
}
  break;


  case 1:
  {
  // Water
  Material water;
  water.name = "water";
  water.id = 1;
  // etc
  ReturnMaterial = water;
}
  break;


  case 2:
  {
   // Some food from a reference.
  Material FoodFromRef;
  FoodFromRef.name = "FoodFromRef";
  FoodFromRef.rho = {725, 770, 800};
  FoodFromRef.id = 2;
  FoodFromRef.k = {1.4, 0.4, 0.1};
  FoodFromRef.cp = {1450,2770, 5000};
  FoodFromRef.TempPhaseChange = {0, 1};
  FoodFromRef.er = {1, 1, 1};
  FoodFromRef.ur = {1, 1, 1};
  FoodFromRef.eDiel = {1, 1, 1};
  ReturnMaterial = FoodFromRef;
}
  break;


  default: std::cerr << "Material is not valid. Please refer to the documentation." << std::endl;
                exit (EXIT_FAILURE);

  }
return ReturnMaterial;

}
