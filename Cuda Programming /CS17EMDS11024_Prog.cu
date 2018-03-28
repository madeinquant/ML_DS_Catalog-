//Submission should be named as  <RollNumber>_Prog.cu
//Upload just this cu file and nothing else. If you upload it as a zip, it will not be evaluated. 


#include <stdio.h>
#define M 514 
//Input has 514 rows and columns 

#define N 512 
//For output, only 512 rows and columns need to be computed. 


//TODO: WRITE GPU KERNEL. It should not be called repeatedly from the host, but just once. Each time it is called, it may process more than pixel or not process any pixel at all. 

__global__ void avg_4(int *a,int *b) {
  // = float[N][N];
   int num_blocks = 9;
   int num_threads =48;   
   int num_iter = M*M/(num_blocks*num_threads);
 
   for (int k=0;k <= num_iter ;k++)  {
        int blk  = blockIdx.x*1;int thd = threadIdx.x*1;

        int thread_num = ( (k*num_blocks + blk)*num_threads + thd) ;
        int x_ =  thread_num/M;
        int y_ =  thread_num % M;
        if ((x_ > 0) && (x_ < M-1) && (y_ > 0) && ( y_  < M-1)) {
           int x_p_1 = x_ + 1 ;
           int y_p_1 = y_ + 1;
           int x_m_1 = x_ - 1;
           int y_m_1 = y_ - 1;
           int x_p_1_a  = x_p_1*M + y_;
           int x_m_1_a  = x_m_1*M + y_;
           int y_p_1_a  = x_*M +  y_p_1;
           int y_m_1_a  = x_*M + y_m_1;
           int avg =  (a[x_p_1_a] + a[x_m_1_a] +  a[y_p_1_a] + a[y_m_1_a])/4;
           b[x_*M + y_] = avg; }
               }
    
   }



main (int argc, char **argv) {
  int A[M][M], B[M][M];
  int *d_A, *d_B; // These are the copies of A and B on the GPU
  int *h_B;       // This is a host copy of the output of B from the GPU
  int i, j;

  // Input is randomly generated
  for(i=0;i<M;i++) {
    for(j=0;j<M;j++) {
      A[i][j] = rand()/1795831;
      //printf("%d\n",A[i][j]);
    }
  }

  // sequential implementation of main computation
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
      B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4;
    }
  }


  // TODO: ALLOCATE MEMORY FOR GPU COPIES OF d_A AND d_B
  cudaMalloc(&d_A,M*M*sizeof(int));
  cudaMalloc(&d_B,M*M*sizeof(int));

  // TODO: COPY A TO d_A
  cudaMemcpy(d_A,A,sizeof(int)*M*M, cudaMemcpyHostToDevice);

  // TODO: CREATE BLOCKS with THREADS AND INVOKE GPU KERNEL
   //Use 9 blocks, each with 48 threads
  avg_4<<<9,48>>>(d_A,d_B);

  // TODO: COPY d_B BACK FROM GPU to CPU in variable h_B
  h_B = (int *)malloc(M*M*sizeof(int));   
  cudaMemcpy(h_B,d_B,sizeof(int)*M*M,cudaMemcpyDeviceToHost); 
  

  // TODO: Verify result is correct by comparing
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
    //TODO: compare each element of h_B and B by subtracting them
        //print only those elements for which the above subtraction is non-zero
      if ( B[i][j] - h_B[i*M + j] != 0)  
         { printf("Elements at location (i,j) = (%d,%d) does not match\n",i,j);
           printf("Expected element B[i,j] = %d\n",B[i][j]);
           printf("Actual element h_B[i,j] = %d\n",h_B[i*M + j]);
                
         } 
    }
   }
    //IF even one element of h_B and B differ, report an error.
    //Otherwise, there is no error.
    //If your program is correct, no error should occur.
}

/*Remember the following guidelines to avoid losing marks
Index of an array should not exceed the array size. 
Do not ignore the fact that boundary rows and columns need not be computed (in fact, they cannot be computed since they don't have four neighbors)
No output array-element should be computed more than once
No marks will be given if the program does not compile or run (TAs will not debug your program at all)
*/
