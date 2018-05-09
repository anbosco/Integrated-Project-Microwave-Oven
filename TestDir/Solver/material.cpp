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
// Air

Material air;
air.name = "air";
air.rho = 1;
air.rhoHot = 1;
air.id = 0;
air.kHot = 1;
air.k = 1;
air.cp = 1;
air.cpHot = 1;
air.TempPhaseChange = 0;
air.er = 1;
air.erHot = 1;
air.ur = 1;
air.urHot = 1;
air.eDiel = 1;
air.eDielHot = 1;

// Water
Material water;
water.name = "water";
water.id = 1;
// etc

Material FoodFromRef;
FoodFromRef.name = "FoodFromRef";
FoodFromRef.rho = 725;
FoodFromRef.rhoHot = 770;
FoodFromRef.id = 2;
FoodFromRef.kHot = 0.4;
FoodFromRef.k = 1.4;
FoodFromRef.cp = 1450;
FoodFromRef.cpHot = 2770;
FoodFromRef.TempPhaseChange = 1;
FoodFromRef.er = 1;
FoodFromRef.erHot = 1;
FoodFromRef.ur = 1;
FoodFromRef.urHot = 1;
FoodFromRef.eDiel = 1;
FoodFromRef.eDielHot = 1;

switch(Val){
  case 0: ReturnMaterial = air;
          break;
  case 1: ReturnMaterial = water;
          break;
  case 2: ReturnMaterial = FoodFromRef;
          break;


  default: std::cerr << "Material is not valid. Please refer to the documentation." << std::endl;
                exit (EXIT_FAILURE);

  }
return ReturnMaterial;

}
