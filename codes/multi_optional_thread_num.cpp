#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <string.h>

//start*************************************************************************************************
#include <thread> //for using thread
#include <semaphore.h> //for using semaphor
//end*************************************************************************************************

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder

#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

#define NUMBER_OF_INPUT_CELLS 784   ///< use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_HIDDEN_CELLS 256   ///< use 256 hidden cells in one hidden layer
#define NUMBER_OF_OUTPUT_CELLS 10   ///< use 10 output cells to model 10 digits (0-9)

#define MNIST_MAX_TESTING_IMAGES 10000                      ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28                                  ///< image width in pixel
#define MNIST_IMG_HEIGHT 28                                 ///< image height in pixel

using namespace std;


typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;
typedef struct Hidden_Node Hidden_Node;
typedef struct Output_Node Output_Node;
vector<Hidden_Node> hidden_nodes(NUMBER_OF_HIDDEN_CELLS);
vector<Output_Node> output_nodes(NUMBER_OF_OUTPUT_CELLS);

/**
 * @brief Data block defining a hidden cell
 */

struct Hidden_Node{
    double weights[28*28];
    double bias;
    double output;
};

/**
 * @brief Data block defining an output cell
 */

struct Output_Node{
    double weights[256];
    double bias;
    double output;
};

/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_Image{
    uint8_t pixel[28*28];
};

/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

/**
 * @details Set cursor position to given coordinates in the terminal window
 */
// *******************************************************************************************8start
// sem_t p3;
void locateCursor(const int row, const int col)
{
    // sem_init(&p3 , 0 , 1);
    // sem_wait(&p3);
    printf("%c[%d;%dH",27,row,col);
    // sem_post(&p3);
}
//************************************************************************************************end

/**
 * @details Clear terminal screen by printing an escape sequence
 */

void clearScreen(){
    printf("\e[1;1H\e[2J");
}

/**
 * @details Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 */

void displayImage(MNIST_Image *img, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");

    for (int y=0; y<MNIST_IMG_HEIGHT; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
        strcat(imgStr,"|");

        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
}

/**
 * @details Outputs a 28x28 text frame at a defined screen position
 */
// *******************************************************************************************8start
// sem_t p8;
// sem_t p9;
// sem_init(p9 , 0 , 1);
void displayImageFrame(int row, int col)
{

    // sem_init(&p8 , 0 , 1);
    if (col!=0 && row!=0) locateCursor(row, col);
    // sem_wait(&p8);
    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------");
    // sem_post(&p8);

}
//*******************************************************************************************end

/**
 * @details Outputs reading progress while processing MNIST testing images
 */
// ******************************************************************************************************************start
// sem_t p1;
void displayLoadingProgressTesting(int imgCount, int y, int x){
    
    // sem_init(&p1 , 0 , 1);

    float progress = (float)(imgCount+1)/(float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x!=0 && y!=0) locateCursor(y, x);
    // sem_wait(&p1);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);
    // sem_post(&p1);
}
// *****************************************************************************************************************end

/**
 * @details Outputs image recognition progress and error count
 */

void displayProgress(int imgCount, int errCount, int y, int x)
{

    double successRate = 1 - ((double)errCount/(double)(imgCount+1));

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n",imgCount+1-errCount, errCount, successRate*100);


}

/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n)
{

    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

    return (b0 | b1 | b2 | b3);

}

/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh)
{

    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){

    lfh->magicNumber =0;
    lfh->maxImages   =0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);

}

/**
 * @details Open MNIST image file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st IMAGE
 */
// **************************************************************************************************************start
// sem_t p6;
FILE *openMNISTImageFile(char *fileName)
{
    // sem_init(&p6 , 0 , 1);

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) 
    {
        // sem_wait(&p6);
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        // sem_post(&p6);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}
//***********************************************************************************************************end


/**
 * @details Open MNIST label file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st LABEL
 */
// *************************************************************************************************** start
// sem_t p7;
FILE *openMNISTLabelFile(char *fileName)
{
    
    // sem_init(&p7 , 0 , 1);

    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) 
    {
        // sem_wait(&p7);
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        // sem_post(&p7);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}
//****************************************************************************************************end

/**
 * @details Returns the next image in the given MNIST image file
 */
// *****************************************************************************start
// sem_t p2;
MNIST_Image getImage(FILE *imageFile)
{

    // sem_init(&p2 , 0 , 1);
    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        // sem_wait(&p2);
        printf("\nError when reading IMAGE file! Abort!\n");
        // sem_post(&p2);
        exit(1);
    }

    return img;
}
// *****************************************************************************************end

/**
 * @details Returns the next label in the given MNIST label file
 */
// ********************************************************************************88start
// sem_t p4;
MNIST_Label getLabel(FILE *labelFile)
{

    // sem_init(&p4 , 0 , 1);
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1)
    {
        // sem_wait(&p4);
        printf("\nError when reading LABEL file! Abort!\n");
        // sem_post(&p4);
        exit(1);
    }

    return lbl;
}
//*********************************************************************************end

/**
 * @brief allocate weights and bias to respective hidden cells
 */

void allocateHiddenParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(HIDDEN_WEIGHTS_FILE);
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 28*28; ++i){
            in >> hidden_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> hidden_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

/**
 * @brief allocate weights and bias to respective output cells
 */

void allocateOutputParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 256; ++i){
            in >> output_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> output_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

/**
 * @details The output prediction is derived by finding the maxmimum output value
 * and returning its index (=0-9 number) as the prediction.
 */

int getNNPrediction(){

    double maxOut = 0;
    int maxInd = 0;

    for (int i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){

        if (output_nodes[i].output > maxOut){
            maxOut = output_nodes[i].output;
            maxInd = i;
        }
    }

    return maxInd;

}

/**
 * @details test the neural networks to obtain its accuracy when classifying
 * 10k images.
 */
//****************************************************************************************************************start
MNIST_Image img;
FILE *imageFile, *labelFile;
int errCount = 0;
int num_hidden_thread;
int neuron_each_thread;

vector< thread > hidden_threads;
vector< thread > output_threads;




sem_t producer_1_to_2; 
sem_t *consumer_1_to_2; //[num_hidden_thread]; //full semaphore((((()))))
// consumer_1_to_2 = (*sem_t)malloc(num_hidden_thread * sizeof(sem_t));
// vector < sem_t> consumer_1_to_2;
sem_t *producer_2_to_3; 
// producer_2_to_3 = (*sem_t)malloc(num_hidden_thread * sizeof(sem_t));
// vector <sem_t> producer_2_to_3;
sem_t consumer_2_to_3[10];

sem_t producer_3_to_4[10]; //empty semaphore
sem_t consumer_3_to_4;

sem_t pro_print;
sem_t cons_print;


void wait_semaphore(sem_t *s , int number)
{
    for(int i = 0; i < number; i ++)
    {
        sem_wait(s);
    }
}

void signal_semaphore(sem_t *s , int number)
{
    for(int i = 0; i < number; i ++)
    {
        sem_post(&s[i]);
    }
}



void read_each_picture()
{
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        for(int i = 0; i < num_hidden_thread; i++) 
        {
            sem_wait(&producer_1_to_2);
        }
        // wait_semaphore(&producer_1_to_2 , num_hidden_thread);
        img = getImage(imageFile);
        sem_wait(&pro_print);
        displayLoadingProgressTesting(imgCount,5,5);
        displayImage(&img, 8,6);
        sem_post(&cons_print);
        for(int i =0; i < num_hidden_thread; i ++)
        {
            sem_post(&consumer_1_to_2[i]);
        }
        // signal_semaphore(consumer_1_to_2 , num_hidden_thread);
    }


}


void calculate_in_hidden_thread(int iB , int iE , int index)
{
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        // for(int k1=0; k1 < 10; k1++)
        // {
        //     sem_wait(&producer_2_to_3[index]);
            
        // }
        wait_semaphore(&producer_2_to_3[index] , 10);
        sem_wait(&consumer_1_to_2[index]); //avaz kardam
        for (int j = iB; j < iE; j++) 
        {
            hidden_nodes[j].output = 0;
            for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
                hidden_nodes[j].output += img.pixel[z] * hidden_nodes[j].weights[z];
            }
            hidden_nodes[j].output += hidden_nodes[j].bias;
            hidden_nodes[j].output = (hidden_nodes[j].output >= 0) ?  hidden_nodes[j].output : 0;
        }
        // sem_wait(&c1);
        // counter1 ++;
        // if(counter1 == 8)
        // {
        for(int i = 0; i < 10; i++)
        {
            sem_post(&consumer_2_to_3[i]);
        }
        // signal_semaphore(consumer_2_to_3 , 10);
        sem_post(&producer_1_to_2);
        //     counter1 = 0;
        // }
        // sem_post(&c1);

    }


}

void calculate_in_output_thread(int output_index)
{
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        // for(int a = 0; a < 8; a++)
        // {
        //     sem_wait(&consumer_2_to_3[output_index]);

        // }
        wait_semaphore(&consumer_2_to_3[output_index] , num_hidden_thread);
        sem_wait(&producer_3_to_4[output_index]);
        

        // for (int i= 0; i < NUMBER_OF_OUTPUT_CELLS; i++)
        // {
            output_nodes[output_index].output = 0;
            for (int j = 0; j < NUMBER_OF_HIDDEN_CELLS; j++) 
            {
                output_nodes[output_index].output += hidden_nodes[j].output * output_nodes[output_index].weights[j];
            }
            output_nodes[output_index].output += 1/(1+ exp(-1* output_nodes[output_index].output));
            output_nodes[output_index].output += output_nodes[output_index].bias;
        // }
        // sem_wait(&c2);
        // counter2 ++;
        // if(counter2 == 10)
        // {
        for(int i = 0; i < num_hidden_thread; i++)
        {
            sem_post(&producer_2_to_3[i]);
        }
        // signal_semaphore(producer_2_to_3 , num_hidden_thread);
        sem_post(&consumer_3_to_4);
        //     counter2 = 0;
        // }
        // sem_post(&c2);
    }


}

void predict_output_in_final_thread()
{
    // sem_init(&p5 , 0 , 1);
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        // for(int h1 = 0; h1 < 10; h1++)
        // {
        //     sem_wait(&consumer_3_to_4);
        // }
        // wait_semaphore()
        wait_semaphore(&consumer_3_to_4 , 10);
        MNIST_Label lbl = getLabel(labelFile);
        int predictedNum = getNNPrediction();
        if (predictedNum!=lbl) 
            errCount++;
        sem_wait(&cons_print);
        printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);
        displayProgress(imgCount, errCount, 5, 66);
        sem_post(&pro_print);
        // for(int o=0; o < 10; o++)
        // {
        //     sem_post(&producer_3_to_4[o]);
        // }
        signal_semaphore(producer_3_to_4 , 10);
    }

    


}

void testNN(){
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    // sem_init(&s1 , 0 , 8);
    // sem_init(&s2 , 0 , 10);
    // sem_init(&s3 , 0 , 1);
    // sem_init(&c1 , 0 , 1);
    // sem_init(&c2 , 0 , 1);
    // sem_init(&p1 , 0 , 1);
    sem_init(&pro_print , 0 , 1);
    sem_init(&cons_print , 0 , 0);
    // sem_t cons_print;
    sem_init(&producer_1_to_2 , 0 , num_hidden_thread);
    for(int i = 0; i < num_hidden_thread; i++)
    {
        sem_init(&consumer_1_to_2[i] , 0 , 0);
        sem_init(&producer_2_to_3[i] , 0 , 10); 
        
    }
    // for(int i = 0; i < num_hidden_thread; i++)
    // {
    //     sem_init(&producer_2_to_3[i] , 0 , 10); 
    // }
    for(int i = 0; i < 10; i++)
    {
        sem_init(&consumer_2_to_3[i] , 0 , 0);
        sem_init(&producer_3_to_4[i] , 0 , 1);    

    }
    sem_init(&consumer_3_to_4 , 0 , 0);

    // for(int i=0; i < 10; i++)
    // {
    //     sem_init(&producer_3_to_4[i] , 0 , 1);    
    // }
    // sem_init(&producer_4_to_0 , 0 , 0);
    // sem_init(&consumer_4_to_0 , 0 , 0);
    // open MNIST files
    // screen output for monitoring progress
    displayImageFrame(7,5);
    thread get_input(read_each_picture);
    // vector< thread > hidden_threads;
    // vector< thread > output_threads;
    neuron_each_thread = 256 / num_hidden_thread;
    // consumer_1_to_2 = vector <sem_t> a1(num_hidden_thread);
    // producer_2_to_3 = vector <sem_t> a2(num_hidden_thread);
    // consumer_1_to_2.resize(num_hidden_thread);
    // producer_2_to_3.resize(num_hidden_thread);

    for(int i =0; i < num_hidden_thread; i++)
    {
        hidden_threads.push_back(thread(calculate_in_hidden_thread , i*neuron_each_thread , 
            (i*neuron_each_thread) + neuron_each_thread, i));
    }
    for(int i =0; i < 10; i ++)
    {
        output_threads.push_back(thread(calculate_in_output_thread , i));
    }
    thread predict_output(predict_output_in_final_thread);
    // thread hidden_1(calculate_in_hidden_thread , 0 , 32 , 0);
    // thread hidden_2(calculate_in_hidden_thread , 32 , 64 , 1);
    // thread hidden_3(calculate_in_hidden_thread , 64 , 96 , 2);
    // thread hidden_4(calculate_in_hidden_thread , 96 , 128 , 3);
    // thread hidden_5(calculate_in_hidden_thread , 128 , 160 , 4);
    // thread hidden_6(calculate_in_hidden_thread , 160 , 192 , 5);
    // thread hidden_7(calculate_in_hidden_thread , 192 , 224 , 6);
    // thread hidden_8(calculate_in_hidden_thread , 224 , 256 , 7);

    // thread output_1(calculate_in_output_thread , 0);
    // thread output_2(calculate_in_output_thread , 1);
    // thread output_3(calculate_in_output_thread , 2);
    // thread output_4(calculate_in_output_thread , 3);
    // thread output_5(calculate_in_output_thread , 4);
    // thread output_6(calculate_in_output_thread , 5);
    // thread output_7(calculate_in_output_thread , 6);
    // thread output_8(calculate_in_output_thread , 7);
    // thread output_9(calculate_in_output_thread , 8);
    // thread output_10(calculate_in_output_thread , 9);

    get_input.join();
    for(int i =0; i < num_hidden_thread; i++)
    {
        hidden_threads[i].join();
    }
    for(int i =0; i < 10; i++)
    {
        output_threads[i].join();
    }
    predict_output.join();
    // hidden_1.join();
    // hidden_2.join();
    // hidden_3.join();
    // hidden_4.join();
    // hidden_5.join();
    // hidden_6.join();
    // hidden_7.join();
    // hidden_8.join();

    // output_1.join();
    // output_2.join();
    // output_3.join();
    // output_4.join();
    // output_5.join();
    // output_6.join();
    // output_7.join();
    // output_8.join();
    // output_9.join();
    // output_10.join();

    fclose(imageFile);
    fclose(labelFile);

}
int main(int argc, const char * argv[]) {
    printf("choose a number for threads in hidden layer :)) \n");

    // remember the time in order to calculate processing time at the end
    scanf("%d" , &num_hidden_thread);
    consumer_1_to_2 = (sem_t *)malloc(num_hidden_thread * sizeof(sem_t));
    producer_2_to_3 = (sem_t *)malloc(num_hidden_thread * sizeof(sem_t));


    time_t startTime = time(NULL);

    // clear screen of terminal window
    clearScreen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");

    // alocating respective parameters to hidden and output layer cells
    allocateHiddenParameters();
    allocateOutputParameters();

    //test the neural network
    testNN();

    locateCursor(38, 5);

    // calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);

    return 0;
}
