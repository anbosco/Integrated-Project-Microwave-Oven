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
  air.rho = {1.2, 1.2};
  air.id = 0;
  air.k = {0.025, 0.025};
  air.cp = {1004, 1004};
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
  water.name = "Water";
  water.rho = {995, 996, 997, 998, 999 , 999.84 , 999.9, 999.9, 999.9, 999.9, 999.5, 999 ,997 };
  water.id = 1;
  water.k = {2.3, 2.2, 2.1, 2, 1.9, 0.569, 0.572, 0.574, 0.578, 0.582, 0.586, 0.595, 0.6};
  water.cp = {2000, 2005, 2010, 2015, 2020, 4217, 4214, 4210, 4205, 4200, 4195, 4185, 4180};
  water.TempPhaseChange = {-4,-3,-2,-1, 0, 1 , 2, 3, 4, 5, 9, 20};
  water.er = {3.15, 3.16, 3.17, 3.18, 3.19, 60, 65, 70, 75.8, 75.8, 75.8,75.8, 75.8};
  water.ur = {1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1};
  water.eDiel = {0.001, 0.002, 0.003, 0.004, 0.005, 3, 3.5, 4, 4.5, 5, 5, 5, 5};
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


  case 3:
  {
   // Not Phy alpha=1
  Material Alpha;
  Alpha.name = "Alpha";
  Alpha.rho = {1, 1};
  Alpha.id = 3;
  Alpha.k = {1, 1};
  Alpha.cp = {1, 1};
  Alpha.TempPhaseChange = {0};
  Alpha.er = {1, 1};
  Alpha.ur = {1, 1};
  Alpha.eDiel = {1, 1};
  ReturnMaterial = Alpha;
}
  break;

  case 4:
  {
   // Not Phy
  Material Alpha;
  Alpha.name = "Alpha";
  Alpha.rho = {1, 1};
  Alpha.id = 4;
  Alpha.k = {2, 2};
  Alpha.cp = {1, 1};
  Alpha.TempPhaseChange = {0};
  Alpha.er = {1, 1};
  Alpha.ur = {1, 1};
  Alpha.eDiel = {1, 1};
  ReturnMaterial = Alpha;
}
  break;

  case 5:
  {
   // SNOt phy.
  Material Alpha;
  Alpha.name = "Alpha";
  Alpha.rho = {1, 1};
  Alpha.id = 5;
  Alpha.k = {1, 1};
  Alpha.cp = {1, 1};
  Alpha.TempPhaseChange = {0};
  Alpha.er = {1, 1};
  Alpha.ur = {1, 1};
  Alpha.eDiel = {1, 1};
  ReturnMaterial = Alpha;
}
  break;



  case 6:
  {
   // BONES.
  Material bone;
  bone.name = "Chicken Bone";
  bone.rho = {1750, 1750};
  bone.id = 6;
  bone.k = {0.58, 0.58};
  bone.cp = {440, 440};
  bone.TempPhaseChange = {0};
  bone.er = {11.345, 11.345};
  bone.ur = {1, 1};
  bone.eDiel = {2.904, 2.904};
  ReturnMaterial = bone;
}
  break;

  case 7:
  {
   // Chicken Fat.
  Material fat;
  fat.name = "Chicken fat";
  fat.rho = {909.4, 909.4};
  fat.id = 7;
  fat.k = {0.201, 0.201};
  fat.cp = {2348, 2348};
  fat.TempPhaseChange = {0};
  fat.er = {5.56, 5.56};
  fat.ur = {1, 1};
  fat.eDiel = {54.28, 54.28};
  ReturnMaterial = fat;
}
  break;


  case 8:
  {
   // Chicken Muscle .
  Material mus;
  mus.name = "Chicken muscle";
  mus.rho = {1059.9, 1059.9};
  mus.id = 8;
  mus.k = {0.478, 0.478};
  mus.cp = {3421, 3421};
  mus.TempPhaseChange = {0};
  mus.er = {59, 59};
  mus.ur = {1, 1};
  mus.eDiel = {54.28, 54.28};
  ReturnMaterial = mus;
}
  break;

  case 9:
  {
   // Not Phy alpha = 0.0002
  Material Alpha;
  Alpha.name = "Alpha";
  Alpha.rho = {10000, 10000};
  Alpha.id = 9;
  Alpha.k = {2, 2};
  Alpha.cp = {1, 1};
  Alpha.TempPhaseChange = {0};
  Alpha.er = {1, 1};
  Alpha.ur = {1, 1};
  Alpha.eDiel = {1, 1};
  ReturnMaterial = Alpha;
}

  case 99:
  {
   // SNOt phy.
  Material Alpha;
  Alpha.name = "Alpha";
  Alpha.rho = {1, 1};
  Alpha.id = 99;
  Alpha.k = {1, 1};
  Alpha.cp = {1, 1};
  Alpha.TempPhaseChange = {0};
  Alpha.er = {9, 9};
  Alpha.ur = {1, 1};
  Alpha.eDiel = {1, 1};
  ReturnMaterial = Alpha;
}
  break;

  default: std::cerr << "Material is not valid. Please refer to the documentation." << std::endl;
                exit (EXIT_FAILURE);

  }
return ReturnMaterial;

}
