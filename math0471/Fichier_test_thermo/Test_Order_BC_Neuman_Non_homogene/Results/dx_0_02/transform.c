#include <stdio.h>

int main() {
    printf("Refactoring data: reverberation chamber");

    char buffer[8196];

    FILE *fInput;
    fInput = fopen("RVC_CAL_PAD01_SPO01.Result","rb");
    if (fInput==NULL){
	printf("Impossible to read file");
    }
    FILE *fOutput;
    fOutput = fopen("data.txt","w");
    
    //fread(buffer,2006,1,fInput);
    while(fread(buffer,sizeof(buffer),1,fInput)){
	printf("Coucou    ");
        fwrite(buffer,sizeof(buffer),1,fOutput);
    }

    fclose(fInput);
    fclose(fOutput);

    return 0;
}
