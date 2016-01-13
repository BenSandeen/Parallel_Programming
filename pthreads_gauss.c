/* Gaussian elimination without pivoting.
 * Compile with "gcc4 -o gauss gauss.c" 
 */

/* Juan Dominguez and Ben Sandeen*/

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <pthread.h>
#include <omp.h>

/*#include <ulocks.h>
#include <task.h>
*/

char *ID;

/* Program Parameters */
#define MAXN 9001  /* Max value of N */
int N;  /* Matrix size */
int procs;  /* Number of processors to use */
int norm, row; // initializing global vars
float multiplier;
int chunk;// = 200;
int max_row;

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
pthread_mutex_t row_lock;
pthread_barrier_t barrier;
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
    * It is this routine that is timed.
    * It is called only on the parent.
    */

void *gauss2(void*); // prototype of parallelized function 

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int submit = 0;  /* = 1 if submission parameters should be used */
  int seed = 0;  /* Random seed */
  char uid[L_cuserid + 2]; /*User name */

  /* Read command-line arguments */
  //  if (argc != 3) {
  if ( argc == 1 && !strcmp(argv[1], "submit") ) {
    /* Use submission parameters */
    submit = 1;
    N = 4;
    procs = 2;
    printf("\nSubmission run for \"%s\".\n", cuserid(uid));
      /*uid = ID;*/
    strcpy(uid,ID);
    srand(randm());
  }
  else {
    if (argc == 3) {
      seed = atoi(argv[3]);
      srand(seed);
      printf("Random seed = %i\n", seed);
    }
    else {
      printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
       argv[0]);
      printf("       %s submit\n", argv[0]);
      exit(0);
    }
  }
    //  }
  /* Interpret command-line args */
  if (!submit) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
    procs = atoi(argv[2]);
    if (procs < 1) {
      printf("Warning: Invalid number of processors = %i.  Using 1.\n", procs);
      procs = 1;
    }
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
  printf("Number of processors = %i.\n", procs);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    // if (col%12==0) printf(".");
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 501) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
  printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 501) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  ID = argv[argc-1];
  argc--;

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  // initializes chunk sizes to a value proportional to N/procs
  chunk = N / (int)(1.3*procs);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
   (float)(usecstop - usecstart)/(float)1000);

}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, procs, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
#define MAX_THREADS 1024

// this function calls the parallelized function gauss2, initializes 
// lock, barrier, and other vars
void gauss() {
  int index; // var for for loop that creates threads
  max_row = N;
  pthread_t threads[procs]; // initializes array for the threads
  pthread_mutex_init(&row_lock,NULL);
  pthread_barrier_init(&barrier, NULL, procs);
  norm = 0;
  row = norm + 1;
  
  // printf(sizeof(pthread[0]));
  // printf("%i\n", sizeof(threads[0]));

  for (index = 0; index < procs; index++) {
    pthread_create(&threads[index],NULL, &gauss2,(void*) index);
  }

  for (index = 0; index < procs; index++) {
    pthread_join(threads[index], NULL);
  }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  // Don't need to parallelize
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (int col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
  
  // destroy the lock and barrier
  pthread_mutex_destroy(&row_lock);
  pthread_barrier_destroy(&barrier);

  // print if matrix is small enough
  print_inputs();
  print_X();
}

void *gauss2(void *threadid) {
  // pthread_t self;
  long tid;
  tid = (long)threadid;
  float temp_multiplier;
  int temp_row = 0; // initializes local vars for the row index...
  int temp_max = 0; // ...and index of last row to go to

  // loops through diagonals
  while(norm < N-1){
    temp_row = 0;
      while (temp_row < max_row){
        pthread_mutex_lock(&row_lock);
        temp_row = row;
        row += chunk;
        // used for dynamic scheduling to decrease chunk size as remaining
        // work decreases
        chunk = (chunk>>1)+1;
        temp_max = temp_row + chunk;
        pthread_mutex_unlock(&row_lock);

        /* Gaussian elimination */
        for(; temp_row < temp_max && temp_row < max_row; temp_row++){
          temp_multiplier = A[temp_row][norm] / A[norm][norm];//get normalizing factor
          B[temp_row] -= B[norm] * temp_multiplier;
          for (int col = norm; col < N; col++) {
            A[temp_row][col] -= A[norm][col] * temp_multiplier;
          }
        }
    }

    pthread_barrier_wait(&barrier);
    if (tid == 0){
      norm++;
      row = norm + 1;
    }
    pthread_barrier_wait(&barrier);
  }


  // print_inputs();
  // print_X();
  pthread_exit(NULL);
}
