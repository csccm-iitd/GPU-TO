#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

void mgcg1(double Ae[][24],double *F,double *U, double *CX, double *FDofs, int nFD,  int nelx, int nely, int nelz, int nl, double tol, int maxit, double *res, int *nit);
void initVcycle(double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0);
void vcycle(double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0);
void cg0(double Ae[][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void cg(double Ae[][24][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *FDofs, int nFD, int nelx, int nely, int nelz);
void printV(char *fname, double *V, int n, int ncol);
void set0(double *V, int n);
void defect0(double Ae[][24],double *F,double *U, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void defect(double A[][24][24],double *F,double *U, double *R, double *FDofs, int nFD, int nelx, int nely, int nelz);
void multAV0(double Ae[][24],double *U,double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void multAV(double A[][24][24],double *U,double *R, double *FDofs, int nFD, int nelx, int nely, int nelz);
double scalVV(double *V1, double *V2, int ndof);
void copyVV(double *V1, double *V2, int ndof);
void addVsV(double *V1,double *V2,double *V3, double s, int ndof);
int readCX(double *CX);
void CorrCsA(double A[][24][24], double *FDofs, int nFD, int nelx, int nely, int nelz);
void printCsAe(double A[][24][24], int nelx, int nely, int nelz);
void CmpCsAe(double A[][24][24],double A1[][24][24], int nelx, int nely, int nelz);
void defD(double D[6][6],double v);
void Bem3(double B[6][24],double s,double e,double t,double hx,double hy,double hz);
void choldc(int n, double aq[], double p[]);
void cholsl(int n, double aq[], double p[], double b[], double x[]);
int estWorkSpace(int nFD0, int nl, int nelx0, int nely0, int nelz0);
void prepAelm(double v, double hx,double hy,double hz);
void fprintV(char *format);
void fprintI(char *format, int v);
void fprintII(char *format, int v1, int v2);
void fprintIII(char *format, int v1, int v2, int v3);
void fprintD(char *format, double v);
void fprintDD(char *format, double v1, double v2);
void fprintID(char *format, int v1, double v2);
void fprintIDD(char *format, int v1, double v2, double v3);
void fprintVc(char *fname, double *V0, int n, int ncol);

FILE *flog;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
                            
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      flog=fopen("mgcg1.log", "a");
      fprintf(flog,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      fclose(flog);
      if (abort) exit(code);
   }
}

#define check() gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize())

#define maxThreads 1024

__global__ void coarseAe2c(double *A, double *CXP, int nelx, int nely, int nelz);
__global__ void coarseAe3c(double AP[],double A[], int nelx, int nely, int nelz);
__global__ void CorrCsAc(double A[], double *FDofs, int nFD, int nelx, int nely, int nelz);
__global__ void assembAc(int nA, double Ae[], double *AQ, int nelx, int nely, int nelz);
__global__ void prepADc(double A[], double *AD, int nelx, int nely, int nelz);
__global__ void prepAD0c(double *AD, double *CX, int nelx, int nely, int nelz);
__global__ void multAV0c(double *U,double *R,double *CX, int nelx, int nely, int nelz);
__global__ void defect0c(double *F,double *U,double *R, double *CX, int nelx, int nely, int nelz);
__global__ void defectc(double A[],double *F,double *U, double *R, int nelx, int nely, int nelz);
__global__ void copyVVc(double *V1, double *V2, int n);
__global__ void set0c(double *V, int n);
__global__ void scalVVc(double *V1, double *V2, int n, double *rnorm);

void choldc1(int n, double aq[], double p[]);
__global__ void cholRow(double aq[], double p[], int i, int n);
__global__ void choldcc(int n, double aq[], double p[]);
__global__ void cholslc(int n, double aq[], double p[], double b[], double x[]);
void cholsl1(int n, double aq[], double p[], double b[], double x[]);
__global__ void cholslCol1(int n, int i, double aq[], double p[], double x[]);
__global__ void cholslCol2(int n, int i, double aq[], double p[], double x[]);

void rstrFDcenv(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD);
__global__ void rstrFDc(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD);
__global__ void preRelc(double *AD, double *F,double *U, int nelx, int nely, int nelz);
__global__ void rstrFc(double *RP, double *F, double *FDofs, int nelx, int nely, int nelz, int nFD);
__global__ void interpc(double *UP, double *U, int nelx, int nely, int nelz);
__global__ void addVsVc(double *V1,double *V2,double *V3, double s, int n);
__global__ void ZeroBCc(double *U, double *FDofs, int nFD, int nelx, int nely, int nelz);
__global__ void Jacobc(double *AD, double *U, double *R, int nelx, int nely, int nelz);

__device__ void fprintIDc(char *format, int v1, double v2);

void printVc(char *fname, double *V0, int n, int ncol);

static int printLev=1;

double Aelm[300][8];

__constant__ double Aelc[300][8];

__constant__ double Ae[24][24];

/*-----------------------------------------------------------------------*/
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *Aep, *F, *U, *CX, *FDofs, *W;
    double Ae[24][24];
    double dnelx, dnely, dnelz, dnl, dnswp, tol, dmaxit, res;
    int nelx, nely, nelz, nl, nswp=2, maxit;
    int nit;
    
    Aep = mxGetDoubles(prhs[0]);
    int mrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);
    for (int i=0; i<24; i++)
    for (int j=0; j<24; j++)
        Ae[i][j]=Aep[24*i+j];
    
    //mexPrintf("\n%d %d input dimensions.\n", mrows, ncols);
    F = mxGetDoubles(prhs[1]);
    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    //mexPrintf("\n%d %d F dimensions.\n", mrows, ncols);
    U = mxGetDoubles(prhs[2]);
    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    //mexPrintf("\n%d %d U dimensions.\n", mrows, ncols);
    CX = mxGetDoubles(prhs[3]);
    mrows = mxGetM(prhs[3]);
    ncols = mxGetN(prhs[3]);
    //mexPrintf("\n%d %d CX dimensions.\n", mrows, ncols);
    FDofs = mxGetDoubles(prhs[4]);
    mrows = mxGetM(prhs[4]);
    ncols = mxGetN(prhs[4]);
    //mexPrintf("\n%d %d FDofs dimensions.\n", mrows, ncols);
    //W = mxGetDoubles(prhs[5]);
    //int mrowsW = mxGetM(prhs[5]);
    //int ncolsW = mxGetN(prhs[5]);
    //mexPrintf("\n%d %d WRK dimensions.\n", mrowsW, ncolsW);
    //int nFD = mxGetN(prhs[4]);
    int nFD = (int) mxGetScalar(prhs[5]); 
    //mexPrintf("\n%d nFD\n", nFD);
    nelx = (int) mxGetScalar(prhs[6]); 
    nely = (int) mxGetScalar(prhs[7]); 
    nelz = (int) mxGetScalar(prhs[8]); 
    nl   = (int) mxGetScalar(prhs[9]); 
    tol  =       mxGetScalar(prhs[10]); 
    maxit= (int) mxGetScalar(prhs[11]); 
    printLev= (int) mxGetScalar(prhs[12]); 
    int lenW=estWorkSpace(nFD, nl, nelx, nely, nelz);
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    if (lenW>f)
    {
        mexPrintf("\nMGCG - Cuda device has not %d free memory, %d is required.\n", f, lenW);
        plhs[0] = mxCreateDoubleScalar(0.0);
        plhs[1] = mxCreateDoubleScalar(-1.0);
    }
    else
    {
        //printV("mgcg1F",F, nelx, nely, nelz);
        mgcg1(Ae, F, U, CX, FDofs, nFD, nelx, nely, nelz, nl, tol, maxit, &res, &nit);
        plhs[0] = mxCreateDoubleScalar(res);
        plhs[1] = mxCreateDoubleScalar((double)nit);
    }
}

void mgcg1(double Ae0[][24],double *F0,double *U0, double *CX0, double *FDofs0, int nFD, int nelx, int nely, int nelz, int nl, double tol, int maxit, double *res, int *nit)
{
    flog=fopen("mgcg1.log", "w");
    fclose(flog);
    int nswp=3;
    int nnel=nelx*nely*nelz;
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int nBlocksY = nelx+1;
    int nBlocksX = nelz+1;
    int nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
    int nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
    int nThreadsY=1;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    dim3 threads(nThreadsX,nThreadsY);

    clock_t tmgcg0=clock();

    cudaDeviceReset();
    cudaSetDevice(0);
    cudaFree(0); // This will establish a context on the device
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    HANDLE_ERROR( cudaMemcpyToSymbol( Ae, Ae0, sizeof(double) * 24*24) );

    int lenW=estWorkSpace(nFD, nl, nelx, nely, nelz);
    if (printLev>=1) {
          fprintII("Reported GPU memory. Total: %d kb Free: %d kb\n",t/1024,f/1024);
          fprintII("Number of dofs %d, Device memory required %d (Mb)\n", ndof, (lenW * sizeof(double))/1048576+1);
    }
    double *devW;
    HANDLE_ERROR( cudaMalloc( (void**)&devW, lenW * sizeof(double) ) );
  
    double *W=&devW[0];
    int lW=0;
    double *F=&W[lW]; lW += ndof;
    HANDLE_ERROR( cudaMemcpy( F, F0, ndof * sizeof(double), cudaMemcpyHostToDevice ) );
    double *U=&W[lW]; lW += ndof;
    HANDLE_ERROR( cudaMemcpy( U, U0, ndof * sizeof(double), cudaMemcpyHostToDevice ) );
    double *CX=&W[lW]; lW += nnel;
    HANDLE_ERROR( cudaMemcpy( CX, CX0, nnel * sizeof(double), cudaMemcpyHostToDevice ) );
    double *FDofs=&W[lW]; lW += nFD;
    HANDLE_ERROR( cudaMemcpy( FDofs, FDofs0, nFD * sizeof(double), cudaMemcpyHostToDevice ) );
    double *R=&W[lW]; lW += ndof;
    double *P=&W[lW]; lW += ndof;
    double *Q=&W[lW]; lW += ndof;
    double *Z=&W[lW]; lW += ndof;
    double *W1=&W[lW];
    int n, itcg;
    double rnorm, res0, rho, rho_p, beta, dpr, alpha, relres;
    
    set0c<<<1,maxThreads>>>(U, ndof); if(printLev>3)check();
    copyVVc<<<1,maxThreads>>>(F,R,ndof); if(printLev>3)check();
    res0=sqrt(scalVV(F,F,ndof));
    if (printLev>=1)
       fprintD("MGCG Initial residual=%12.6g\n",res0);
    
    initVcycle(CX, FDofs, W1, nFD, nl, nelx, nely, nelz);
    
    vcycle(R, P, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
    
    if (printLev>1)
    {
        defect0c<<<grids,threads>>>(R, P, Z, CX, nelx, nely, nelz); if(printLev>3)check();
        fprintD("CG  res0=%12.6g\n",res0);
        fprintD("CG  res1=%12.6g\n",sqrt(scalVV(Z,Z,ndof)));
    }
    copyVVc<<<1,maxThreads>>>(P,Z,ndof); if(printLev>3)check();
    rho=scalVV(R,Z,ndof);
    rho_p=rho;
    for (itcg = 0; itcg < maxit; ++itcg)
    {
        clock_t t0=clock();
        multAV0c<<<grids,threads>>>(P, Q, CX, nelx, nely, nelz); if(printLev>3)check();
        ZeroBCc<<<grids,threads>>>(Q,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
        alpha=rho_p/scalVV(P,Q,ndof);
        addVsVc<<<1,maxThreads>>>(U,P,U,alpha,ndof); if(printLev>3)check();
        addVsVc<<<1,maxThreads>>>(R,Q,R,-alpha,ndof); if(printLev>3)check();
        //set0(Z, ndof);
        vcycle(R, Z, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
        rho=scalVV(R,Z,ndof);
        beta = rho/rho_p;
        addVsVc<<<1,maxThreads>>>(Z,P,P,beta,ndof); if(printLev>3)check();
        rho_p = rho;
        *res = sqrt(scalVV(R,R,ndof));
        relres = *res/res0;
        clock_t t1=clock();
        if (printLev>=2)
            fprintIDD("CG It %d Time %8.3g  %12.6g\n",itcg, (double)(t1-t0)/CLOCKS_PER_SEC,relres);
        if (relres < tol || itcg>=maxit)
            break;
    }
    //multAV0(Ae, P, Q, CX, FDofs, nFD, nelx, nely, nelz);
    //alpha=rho_p/scalVV(P,Q,ndof);
    //addVsV(U,P,U,alpha,ndof);
    //defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
    //*res = sqrt(scalVV(R,R,ndof))/res0;
    HANDLE_ERROR( cudaMemcpy( U0, U, ndof * sizeof(double), cudaMemcpyDeviceToHost ) );
    for (n = 1; n < ndof; n +=3)
        U0[n]=-U0[n];

    *res=relres;
    *nit=itcg;
    if (printLev>=1) 
       fprintIDD("MGCG Nit=%d  Time %8.3g  Final residual=%12.6g\n",itcg,(double)(clock()-tmgcg0)/CLOCKS_PER_SEC,relres);
    HANDLE_ERROR( cudaFree( devW ) );
    fclose(flog);
}

int estWorkSpace(int nFD0, int nl, int nelx0, int nely0, int nelz0)
{
    int nnel, nnod, ndof, nelx, nely, nelz;
    int nFD, lW;
    int nnel0=nelx0*nely0*nelz0;
    int nnod0=(nelx0+1)*(nely0+1)*(nelz0+1);
    int ndof0=3*nnod0;
    ndof=ndof0;
    nFD=nFD0;
        
    lW=ndof0*6 + nnel0 + nFD0;   
    for (int lev = 0; lev <nl; ++lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnel=nelx*nely*nelz;
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        
        if (lev==0)
            lW += 2*ndof;
        else
            lW += 4*ndof + 300*nnel + nFD;
            
        if (lev==nl-1)
            lW += 3*ndof+ndof*ndof;
    }
    return lW;
}

static int levW[20], lnFD[20];

void initVcycle(double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0)
{
    int n, nnel, nnod, ndof, nelx, nely, nelz;
    double *U, *F, *R, *CX, *A, *AP, *AD, *AQ, *FDofs, *UP, *RP, *CXP, *FDofsP, *BC;
    double *P, *Q, *Z, *WRK;
    double nu=0.3, hx0=1, hy0=1, hz0=1;
    double hx, hy, hz;
    int nFD, lW;
    int nnel0=nelx0*nely0*nelz0;
    int nnod0=(nelx0+1)*(nely0+1)*(nelz0+1);
    int ndof0=3*nnod0;
    ndof=ndof0;
    nFD=nFD0;
    lnFD[0]=nFD0;
    int nBlocksY = nelx+1;
    int nBlocksX = nelz+1;
    int nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
    int nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
    int nThreadsY=1;
    int nThreadsZ=1;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    dim3 threads(nThreadsX,nThreadsY);

    lW=0;   
    for (int lev = 0; lev <nl; ++lev)
    {
        if (printLev>5)
           fprintI("MGCG initVcycle lev=%d\n",lev);
        levW[lev]=lW;
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnel=nelx*nely*nelz;
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        hx=hx0*pow(2,lev);
        hy=hy0*pow(2,lev);
        hz=hz0*pow(2,lev);
        nBlocksY = nelx+1;
        nBlocksX = nelz+1;
        nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
        nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
        nThreadsY=1;
        dim3 grids1(nBlocksX,nBlocksY,nBlocksZ);
        dim3 threads1(nThreadsX,nThreadsY);
        if (lev==0)
        {
            CX=CX0;
            FDofs=FDofs0;
            nFD=nFD0;
            R=&W[lW]; lW += ndof;
            AD=&W[lW]; lW += ndof;
            prepAD0c<<<grids1,threads1>>>(AD, CX, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>5)
               fprintI("MGCG initVcycle prepAD0c %d\n",lev);
        }
        else
        {
            FDofsP=FDofs;
            U=&W[lW];  lW += ndof;
            F=&W[lW];  lW += ndof;
            R=&W[lW];  lW += ndof;
            AP=A;
            A=&W[lW];  lW += 300*nnel;
            AD=&W[lW]; lW += ndof;
            FDofs=&W[lW];
            int nFDP=lnFD[lev-1];
            rstrFDcenv(FDofsP,FDofs, nelx, nely, nelz, nFDP, &nFD);
            lW += nFD;
            lnFD[lev]=nFD;
            nBlocksY = nelx;
            nBlocksX = nelz;
            nBlocksZ = 1;
            nThreadsX=nely;
            dim3 grids2(nBlocksX,nBlocksY,nBlocksZ);
            
            if (lev==1) {
                prepAelm(nu, hx, hy, hz);
                HANDLE_ERROR( cudaMemcpyToSymbol( Aelc, Aelm, sizeof(double) * 300*8) );
                coarseAe2c<<<grids2,nThreadsX>>>(A, CX, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>5)
                   fprintI("MGCG initVcycle coarseAe2c %d\n",lev);
            }
            else
            {
                coarseAe3c<<<grids2,nThreadsX>>>(AP, A, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>5)
                   fprintI("MGCG initVcycle coarseAe3c %d\n",lev);
            }
                
            if (lev==nl-1)
            {
                P=&W[lW]; lW += ndof;
                Q=&W[lW]; lW += ndof;
                Z=&W[lW]; lW += ndof;
                AQ=&W[lW]; lW += ndof*ndof;
                CorrCsAc<<<nFD/maxThreads+1,maxThreads>>>(A, FDofs, nFD, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>5)
                   fprintI("MGCG initVcycle CorrCsAc %d\n",lev);

                set0c<<<1,maxThreads>>>(AQ,ndof*ndof); if(printLev>3)check();
                nBlocksY = nelx+1;
                nBlocksX = nelz+1;
                nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
                nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
                dim3 grids3(nBlocksX,nBlocksY,nBlocksZ);
                //printf("nBlocksX %d, nBlocksY %d, nBlocksZ %d, nThreadsX %d\n\n", nBlocksX, nBlocksY, nBlocksZ, nThreadsX);
                assembAc<<<grids3,nThreadsX>>>(ndof, A, AQ, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>5)
                   fprintI("MGCG initVcycle assembAc %d\n",lev);
                //choldcc<<<1,1>>>(ndof, AQ,P); if(printLev>3)check();
                choldc1(ndof, AQ,P);
                if (printLev>5)
                   fprintI("MGCG initVcycle choldc1 %d\n",lev);
            }
            if (printLev>5)
               fprintIII("MGCG initVcycle before prepADc %d %d %d\n",nelx,nely,nelz);
            int nBlY = nelx+1;
            int nBlX = nelz+1;
            int nBlZ = (3*(nely+1)+maxThreads-1)/maxThreads;
            int nThrX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
            int nThrY=1;
            dim3 gridsAD(nBlX,nBlY,nBlZ);
            dim3 threadsAD(nThrX,nThrY);
            if (printLev>5)
               fprintIII("MGCG initVcycle before prepADc %d %d %d\n",nBlX,nBlY,nBlZ);
            if (printLev>5)
               fprintIII("MGCG initVcycle before prepADc %d %d %d\n",maxThreads,nThrX,nThrY);
            if (printLev>5 && lev==nl-1)
               fprintVc("A",A,300*nnel,10);
            if (printLev>5 && lev==nl-1)
               fprintVc("AD",AD,ndof,3);
               
            prepADc<<<gridsAD,nThrX>>>(A, AD, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>5)
               fprintI("MGCG initVcycle prepADc %d\n",lev);
        }
    }
    
    if (printLev>1) fprintII("Number of dof %d  Working space %d\n",ndof0,lW+6*ndof0+nnel0+nFD0);
}

void vcycle(double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0)
{
    int n, nnod, ndof, nnel, nelx, nely, nelz;
    double *U, *F, *R, *CX, *A, *AD, *AQ, *FDofs, *UP, *RP, *CXP, *FDofsP, *BC;
    double *P, *Q, *Z, *WRK;
    int nFDP, nFD, lW;
    int ndof0=3*(nelx0+1)*(nely0+1)*(nelz0+1);
    ndof=ndof0;
    U=U0;
    F=F0;
    CX=CX0;
    FDofs=FDofs0;
    nFD=nFD0;
    lW=levW[0];
    R=&W[lW];  lW += ndof0;
    AD=&W[lW]; lW += ndof0;
    int nBlocksY = nelx0+1;
    int nBlocksX = nelz0+1;
    int nBlocksZ = (3*(nely0+1)+maxThreads-1)/maxThreads;
    int nThreadsX=(3*(nely0+1)<=maxThreads) ? 3*(nely0+1) : maxThreads;
    int nThreadsY=1;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    dim3 threads(nThreadsX,nThreadsY);
    
    if (printLev>2)
    {
        defect0c<<<grids,threads>>>(F, U, R, CX, nelx0, nely0, nelz0); if(printLev>3)check();
        fprintID("lev %d Initial %12.6g\n",0,sqrt(scalVV(R,R,3*(nelx0+1)*(nely0+1)*(nelz0+1))));
    }
    
    for (int lev = 0; lev <nl; ++lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnel=nelx*nely*nelz;
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        nBlocksY = nelx+1;
        nBlocksX = nelz+1;
        nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
        nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
        nThreadsY=1;
        dim3 grids1(nBlocksX,nBlocksY,nBlocksZ);
        dim3 threads1(nThreadsX,nThreadsY);
        lW=levW[lev];
        if (lev==0)
        {
            U=U0;
            F=F0;
            CX=CX0;
            FDofs=FDofs0;
            nFD=nFD0;
            R=&W[lW]; lW += ndof;
            AD=&W[lW]; lW += ndof;
        }
        else
        {
            RP=R;
            CXP=CX;
            FDofsP=FDofs;
            U=&W[lW];  lW += ndof;
            F=&W[lW];  lW += ndof;
            R=&W[lW];  lW += ndof;
            A=&W[lW];  lW += 300*nnel;
            AD=&W[lW]; lW += ndof;
            nFD=lnFD[lev];
            FDofs=&W[lW]; lW += nFD;
            rstrFc<<<grids1,threads1>>>(RP,F,FDofs, nelx, nely, nelz, nFD); if(printLev>3)check();
            ZeroBCc<<<grids1,threads1>>>(F,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
        }

        if (lev<nl-1)
        {
            if (printLev>3) {
                if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
                else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
                fprintID("lev %d Bef preRelc %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
            preRelc<<<grids1,threads1>>>(AD, F, U, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>3) {
                if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
                else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
                fprintID("lev %d Aft preRelc %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
            for (int it = 0; it < nswp; ++it)
            {
                if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
                else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
                ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>3)
                    fprintID("lev %d Bef rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
                Jacobc<<<grids1,threads1>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
            }
            if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
            else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
            ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>3)
               fprintID("lev %d Aft Rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        }
        else
        {
            P=&W[lW]; lW += ndof;
            Q=&W[lW]; lW += ndof;
            Z=&W[lW]; lW += ndof;
            AQ=&W[lW]; lW += ndof*ndof;
            if (printLev>3)
            {
                if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
                else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
                fprintID("lev %d Bef DirSol %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
            if (lev==0) cg0(Ae, F, U, R, P, Q, Z, CX, FDofs, nFD, nelx, nely, nelz);
            else {cholsl1(ndof,AQ,P,F,U); if(printLev>3)check()};
            if (printLev>3)
            {
                if (lev==0) {defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
                else {defectc<<<grids1,threads1>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
                ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
                fprintID("lev %d Aft DirSol %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
        }
    }
    
    for (int lev = nl-2; lev >=0; --lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnel=nelx*nely*nelz;
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        nBlocksY = nelx+1;
        nBlocksX = nelz+1;
        nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
        nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
        nThreadsY=1;
        dim3 grids2(nBlocksX,nBlocksY,nBlocksZ);
        dim3 threads2(nThreadsX,nThreadsY);
        UP=U;
        nFD=lnFD[lev];
        lW=levW[lev];
        if (lev>0)
        {
            U=&W[lW]; lW += ndof;
            F=&W[lW]; lW += ndof;
            R=&W[lW]; lW += ndof;
            A=&W[lW];  lW += 300*nnel;
            AD=&W[lW]; lW += ndof;
            FDofs=&W[lW];
        }
        else
        {
            U=U0;
            F=F0;
            CX=CX0;
            FDofs=FDofs0;
            nFD=nFD0;
            R=&W[lW]; lW += ndof;
            AD=&W[lW]; lW += ndof;
        }
        
        interpc<<<grids2,threads2>>>(UP,U, nelx, nely, nelz); if(printLev>3)check();
        ZeroBCc<<<grids2,threads2>>>(U,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
        for (int it = 0; it < nswp; ++it)
        {
            if (lev==0) {defect0c<<<grids2,threads2>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
            else {defectc<<<grids2,threads2>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
            ZeroBCc<<<grids2,threads2>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            Jacobc<<<grids2,threads2>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>3)
                fprintID("lev %d Bef rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        }
        if (printLev>3)
        {
            if (lev==0) {defect0c<<<grids2,threads2>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();}
            else {defectc<<<grids2,threads2>>>(A, F, U, R, nelx, nely, nelz); if(printLev>3)check();}
            ZeroBCc<<<grids2,threads2>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            fprintID("lev %d Aft rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        }
        
    }
    if (printLev>2) fprintV("\n");
}

void cg0(double Ae[][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int maxit=ndof;
    int itcg;
    double rnorm, res,res0, rho, rho_p, beta, dpr, alpha, relres;
    double tol=1e-10;
    
    clock_t t0=clock();
    set0(U, ndof);
    defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
    res0=sqrt(scalVV(F,F,ndof));
    if (printLev>3) fprintD("CG Coarse level  Res0 %12.6g\n",res0);
    copyVV(F,R,ndof);
    copyVV(R,Z,ndof);
    rho=scalVV(R,R,ndof);
    rho_p=rho;
    for (itcg = 0; itcg < maxit; ++itcg)
    {
        multAV0(Ae, Z, P, CX, FDofs, nFD, nelx, nely, nelz);
        alpha=rho_p/scalVV(P,Z,ndof);
        addVsV(U,Z,U,alpha,ndof);
        addVsV(R,P,R,-alpha,ndof);
        
        rho=scalVV(R,R,ndof);
        beta = rho/rho_p;
        addVsV(R,Z,Z,beta,ndof);
        rho_p = rho;
        res = sqrt(scalVV(R,R,ndof));
        relres = res/res0;
        if (relres < tol || itcg>=maxit)
            break;
    }
    clock_t t1=clock();
    if (printLev>3) fprintIDD("CG It %d Time %12.6g   Res %12.6g\n",itcg,(double)(t1-t0)/CLOCKS_PER_SEC,res);
}

void cg(double A[][24][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int maxit=5*ndof;
    int itcg;
    double rnorm, res,res0, rho, rho_p, beta, dpr, alpha, relres;
    double tol=1e-10;
    
    clock_t t0=clock();
    set0(U, ndof);
    //printCsAe(A, nelx, nely, nelz);
    defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
    res0=sqrt(scalVV(F,F,ndof));
    if (printLev>3) fprintD("CG Coarse level  Res0 %12.6g\n",res0);
    copyVV(F,R,ndof);
    copyVV(R,Z,ndof);
    rho=scalVV(R,R,ndof);
    rho_p=rho;
    for (itcg = 0; itcg < maxit; ++itcg)
    {
        multAV(A, Z, P, FDofs, nFD, nelx, nely, nelz);
        alpha=rho_p/scalVV(P,Z,ndof);
        addVsV(U,Z,U,alpha,ndof);
        addVsV(R,P,R,-alpha,ndof);
        
        rho=scalVV(R,R,ndof);
        beta = rho/rho_p;
        addVsV(R,Z,Z,beta,ndof);
        rho_p = rho;
        res = sqrt(scalVV(R,R,ndof));
        relres = res/res0;
        if (relres < tol || itcg>=maxit)
            break;
    }
    clock_t t1=clock();
    if (printLev>3) fprintIDD("CG It %d Time %12.6g   Res %12.6g\n",itcg,(double)(t1-t0)/CLOCKS_PER_SEC,res);
}

void rstrFDcenv(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD)
{
    int nBlocksY = nelx+1;
    int nBlocksX = nelz+1;
    int nBlocksZ = (3*(nely+1)+maxThreads-1)/maxThreads;
    int nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
    int nThreadsY=1;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    dim3 threads(nThreadsX,nThreadsY);
    int *devnFD;
    HANDLE_ERROR( cudaMalloc( (void**)&devnFD, sizeof(int) ) );
    int nThreads=min(nFDP,maxThreads);
    rstrFDc<<<1,nThreads>>>(FDofsP,FDofs, nelx, nely, nelz, nFDP, devnFD); if(printLev>3)check();
    
    HANDLE_ERROR( cudaMemcpy( nFD, devnFD, sizeof(int), cudaMemcpyDeviceToHost ) );
}

__global__ void rstrFDc(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD)
{
    __shared__ int count;
    int i, j, k, di, dj, dk, n, d, nod, nodP, nel, nelP;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;

    int iFD=0;
    int idof;
    
    if (threadIdx.x == 0)
       count=0;
    __syncthreads();
    
    //n=blockIdx.y*3*(nely+1)*(nelz+1)+blockIdx.x*3*(nely+1)+ blockIdx.z*blockDim.x+threadIdx.x;
    //if (n<nFDP) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < nFDP; n += stride) {
        idof=((int)FDofsP[n]-1)%3;
        nodP=((int)FDofsP[n]-1)/3;
        iP=nodP/((nelzP+1)*(nelyP+1));
        if (iP%2==0) {
            kP=(nodP-(nelzP+1)*(nelyP+1)*iP)/(nelyP+1);
            if (kP%2==0) {
                jP=nodP-(nelzP+1)*(nelyP+1)*iP-(nelyP+1)*kP;
                if (jP%2==0) {
                    nod=(nelz+1)*(nely+1)*iP/2+(nely+1)*kP/2+jP/2; 
                    iFD=atomicAdd(&count,1);
                    FDofs[iFD]=3*nod+idof+1;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        *nFD=count;
    //__syncthreads();
}

__global__ void rstrFc(double *RP, double *F, double *FDofs, int nelx, int nely, int nelz, int nFD)
{
    int i, j, k, di, dj, dk, n, d0, nod, nodP;
    int iP, jP, kP;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double weight;
    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
        d0=threadIdx.x%3;
        
        k=blockIdx.x;
        i=blockIdx.y;
            
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        F[3*nod+d0]=0.0;
       
        for (di = -1; di <= 1; di++)
        {
            iP=2*i+di;
            if (iP<0 || iP>nelxP) continue;
            for (dk = -1; dk <= 1; dk++)
            {
                kP=2*k+dk;
                if (kP<0 || kP>nelzP) continue;
                for (dj = -1; dj <= 1; dj++)
                {
                    jP=2*j+dj;
                    if (jP<0 || jP>nelyP) continue;
                    weight=1.0/pow(2.0,(double)(abs(di)+abs(dj)+abs(dk)));
                    nodP=(nelzP+1)*(nelyP+1)*iP+(nelyP+1)*kP+jP;
                    F[3*nod+d0]=F[3*nod+d0]+weight*RP[3*nodP+d0];
                }
            }
        }

    }
}

void mulVsV(double *V1,double *V2, double s, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, s, ndof) private(n)
    for (n = 0; n < ndof; ++n)
    V2[n]=V1[n]*s;
}

void multAV0(double Ae[][24],double *U,double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d;
    int i0, j0, k0;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
        
    set0(R, ndof);
        
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(R, U, Ae, CX, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, nod2, i, j, k, n1, n2, d)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne[n2];
                for (d=0; d<3; ++d)
                {
                    R[3*nod1]  =R[3*nod1]  +CX[nel]*Ae[3*n1]  [3*n2+d]*U[3*nod2+d];
                    R[3*nod1+1]=R[3*nod1+1]+CX[nel]*Ae[3*n1+1][3*n2+d]*U[3*nod2+d];
                    R[3*nod1+2]=R[3*nod1+2]+CX[nel]*Ae[3*n1+2][3*n2+d]*U[3*nod2+d];
                }
            }
        }
    }
    }
    
    for (n = 0; n < nFD; ++n)
        R[(int)FDofs[n]-1]=0.0;
}

void multAV(double A[][24][24],double *U,double *R, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d;
    int i0, j0, k0;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
        
    set0(R, ndof);
        
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(R, U, A, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, nod2, i, j, k, n1, n2, d)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne[n2];
                for (d=0; d<3; ++d)
                {
                    R[3*nod1]   += A[nel][3*n1]  [3*n2+d]*U[3*nod2+d];
                    R[3*nod1+1] += A[nel][3*n1+1][3*n2+d]*U[3*nod2+d];
                    R[3*nod1+2] += A[nel][3*n1+2][3*n2+d]*U[3*nod2+d];
                }
            }
        }
    }
    }
    
    for (n = 0; n < nFD; ++n)
        R[(int)FDofs[n]-1]=0.0;
}

__global__ void multAV0c(double *U,double *R, double *CX, int nelx, int nely, int nelz)
{
    int nod, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d, d0;
    int i0, j0, k0, m, ia;
    int di, dj, dk;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nn[8]={6,5,2,1,7,4,3,0};

    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];


    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
        d0=threadIdx.x%3;
        
        k=blockIdx.x;
        i=blockIdx.y;
            
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        R[3*nod+d0]=0.0;
        for (di = -1; di <= 0; di++)
        {
            iP=i+di;
            if (iP<0 || iP>nelx-1) continue;
            for (dk = -1; dk <= 0; dk++)
            {
                kP=k+dk;
                if (kP<0 || kP>nelz-1) continue;
                for (dj = -1; dj <= 0; dj++)
                {
                    jP=j+dj;
                    if (jP<0 || jP>nely-1) continue;
                    nel=nelz*nely*iP+nely*kP+jP;
                    nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                    n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                    m=3*n1+d0;
                    for (n2 = 0; n2 < 8; ++n2)
                    {
                        nod2=nod1+ne[n2];
                        for (d=0; d<3; ++d)
                        {
                            n=3*n2+d;
                        R[3*nod+d0]   += CX[nel]*Ae[m][n]*U[3*nod2+d];
                        }
                    }
                }
            }
        }
    }
}

void set0(double *V, int n)
{
    int i;
    #pragma omp parallel for shared(V, n) private(i)
    for(i=0; i<n; i++)
        V[i]=0.0;
}

__global__ void set0c(double *V, int n)
{
    int i=threadIdx.x;
    while(i<n) {
        V[i]=0.0;
        i +=blockDim.x;
    }
}

__global__ void addVsVc(double *V1,double *V2,double *V3, double s, int n)
{
    int i=threadIdx.x;
    while(i<n) {
        V3[i]=V1[i]+s*V2[i];
        i +=blockDim.x;
    }
}

void addVsV(double *V1,double *V2,double *V3, double s, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, V3, s, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V3[n]=V1[n]+s*V2[n];
}

double scalVV(double *V1, double *V2, int n)
{
    double S=0.0;
    double *devS;
    HANDLE_ERROR( cudaMalloc( (void**)&devS, sizeof(double) ) );
    scalVVc<<<1,maxThreads>>>(V1,V2, n, devS); if(printLev>3)check();
    HANDLE_ERROR( cudaMemcpy( &S, devS, sizeof(double), cudaMemcpyDeviceToHost ) );
    return S;    
}

__global__ void scalVVc(double *V1, double *V2, int n, double *rnorm)
{
    __shared__ double cache[maxThreads];
    int cacheIndex = threadIdx.x;
    int i=threadIdx.x;
    double temp=0.0;
    while(i<n) {
        temp +=V1[i]*V2[i];
        i +=blockDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    
    i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0)
        *rnorm=cache[0];    
    __syncthreads();
}

void copyVV(double *V1, double *V2, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V2[n]=V1[n];
}

__global__ void copyVVc(double *V1, double *V2, int n)
{
    int i=threadIdx.x;
    while(i<n) {
        V2[i]=V1[i];
        i +=blockDim.x;
    }
}

void defect0(double Ae[][24],double *F,double *U, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d;
    int i0, j0, k0;
    double rnorm;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
    
    for (n = 0; n < ndof; ++n)
        R[n]=F[n];
        
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(R, U, Ae, CX, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, nod2, i, j, k, n1, n2, d)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne[n2];
                for (d=0; d<3; ++d)
                {
                    R[3*nod1]  =R[3*nod1]  -CX[nel]*Ae[3*n1]  [3*n2+d]*U[3*nod2+d];
                    R[3*nod1+1]=R[3*nod1+1]-CX[nel]*Ae[3*n1+1][3*n2+d]*U[3*nod2+d];
                    R[3*nod1+2]=R[3*nod1+2]-CX[nel]*Ae[3*n1+2][3*n2+d]*U[3*nod2+d];
                }
            }
        }
    }
    }
    
    for (n = 0; n < nFD; ++n)
        R[(int)FDofs[n]-1]=0.0;
}

void defect(double A[][24][24],double *F,double *U, double *R, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d;
    int i0, j0, k0;
    double rnorm;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
    
    for (n = 0; n < ndof; ++n)
        R[n]=F[n];
        
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(R, U, A, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, nod2, i, j, k, n1, n2, d)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne[n2];
                for (d=0; d<3; ++d)
                {
                    R[3*nod1]   -= A[nel][3*n1]  [3*n2+d]*U[3*nod2+d];
                    R[3*nod1+1] -= A[nel][3*n1+1][3*n2+d]*U[3*nod2+d];
                    R[3*nod1+2] -= A[nel][3*n1+2][3*n2+d]*U[3*nod2+d];
                }
            }
        }
    }
    }
    
    for (n = 0; n < nFD; ++n)
        R[(int)FDofs[n]-1]=0.0;
}

__global__ void preRelc(double *AD, double *F,double *U, int nelx, int nely, int nelz)
{
    int ndof=3*(nelx+1)*(nely+1)*(nelz+1);
    int n;
    n=blockIdx.y*3*(nely+1)*(nelz+1)+blockIdx.x*3*(nely+1)+ blockIdx.z*blockDim.x+threadIdx.x;
    if (n<ndof)
        U[n]=0.7*F[n]/AD[n];
}

__global__ void Jacobc(double *AD, double *U, double *R, int nelx, int nely, int nelz)
{
    int ndof=3*(nelx+1)*(nely+1)*(nelz+1);
    int n;
    n=blockIdx.y*3*(nely+1)*(nelz+1)+blockIdx.x*3*(nely+1)+ blockIdx.z*blockDim.x+threadIdx.x;
    if (n<ndof)
       U[n]=0.52*U[n]+0.48*(R[n]+AD[n]*U[n])/AD[n];
}

__global__ void defect0c(double *F,double *U,double *R, double *CX, int nelx, int nely, int nelz)
{
    int nod, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d, d0;
    int i0, j0, k0, m, ia;
    int di, dj, dk;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nn[8]={6,5,2,1,7,4,3,0};

    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];


    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
        d0=threadIdx.x%3;
        
        k=blockIdx.x;
        i=blockIdx.y;
            
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        R[3*nod+d0]=F[3*nod+d0];
        for (di = -1; di <= 0; di++)
        {
            iP=i+di;
            if (iP<0 || iP>nelx-1) continue;
            for (dk = -1; dk <= 0; dk++)
            {
                kP=k+dk;
                if (kP<0 || kP>nelz-1) continue;
                for (dj = -1; dj <= 0; dj++)
                {
                    jP=j+dj;
                    if (jP<0 || jP>nely-1) continue;
                    nel=nelz*nely*iP+nely*kP+jP;
                    nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                    n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                    m=3*n1+d0;
                    for (n2 = 0; n2 < 8; ++n2)
                    {
                        nod2=nod1+ne[n2];
                        for (d=0; d<3; ++d)
                        {
                            n=3*n2+d;
                        R[3*nod+d0]   -= CX[nel]*Ae[m][n]*U[3*nod2+d];
                        }
                    }
                }
            }
        }
    }
}

__global__ void defectc(double A[],double *F,double *U, double *R, int nelx, int nely, int nelz)
{
    int nod, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d, d0;
    int i0, j0, k0, m, ia;
    int di, dj, dk;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nn[8]={6,5,2,1,7,4,3,0};
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
        d0=threadIdx.x%3;
        
        k=blockIdx.x;
        i=blockIdx.y;
            
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        R[3*nod+d0]=F[3*nod+d0];
        for (di = -1; di <= 0; di++)
        {
            iP=i+di;
            if (iP<0 || iP>nelx-1) continue;
            for (dk = -1; dk <= 0; dk++)
            {
                kP=k+dk;
                if (kP<0 || kP>nelz-1) continue;
                for (dj = -1; dj <= 0; dj++)
                {
                    jP=j+dj;
                    if (jP<0 || jP>nely-1) continue;
                    nel=nelz*nely*iP+nely*kP+jP;
                    nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                    n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                    m=3*n1+d0;
                    for (n2 = 0; n2 < 8; ++n2)
                    {
                        nod2=nod1+ne[n2];
                        for (d=0; d<3; ++d)
                        {
                            n=3*n2+d;
                            if (n<=m)
                                ia=nel*300+((m+1)*m)/2+n;
                            else
                                ia=nel*300+((n+1)*n)/2+m;
                            R[3*nod+d0]   -= A[ia]*U[3*nod2+d];
                        }
                    }
                }
            }
        }
    }
}

__global__ void ZeroBCc(double *U, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int n=blockIdx.y*3*(nely+1)*(nelz+1)+blockIdx.x*3*(nely+1)+ blockIdx.z*blockDim.x+threadIdx.x;
    if (n<nFD)
        U[(int)FDofs[n]-1]=0.0;
}

void CorrCsA(double A[][24][24], double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ne[8]={5,6,1,2,4,7,0,3};
    
    int nod, nel, n1;
    int di, dj, dk;
    int i, j, k, n;
    int iP, jP, kP, idof;
    
    for (n = 0; n < nFD; n++)
    {
        idof=((int)FDofs[n]-1)%3;
        nod=((int)FDofs[n]-1)/3;
        i=nod/((nelz+1)*(nely+1));
        k=(nod-(nelz+1)*(nely+1)*i)/(nely+1);
        j=nod-(nelz+1)*(nely+1)*i-(nely+1)*k;
        for (di = -1; di <= 0; di++)
        for (dk = -1; dk <= 0; dk++)
        for (dj = -1; dj <= 0; dj++)
        {
            iP=i+di;
            if (iP<0 || iP>nelx-1) continue;
            jP=j+dj;
            if (jP<0 || jP>nely-1) continue;
            kP=k+dk;
            if (kP<0 || kP>nelz-1) continue;
            nel=nelz*nely*iP+nely*kP+jP; 
            n1=ne[4*(di+1)+2*(dk+1)+dj+1];
            A[nel][3*n1+idof][3*n1+idof]=1e10;
        }
    }
}

__global__ void CorrCsAc(double A[], double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int nn[8]={5,6,1,2,4,7,0,3};
    
    int nod, nel, idof;
    int di, dj, dk;
    int i, j, k, n, n1;
    int iP, jP, kP, m, ia;
    
    n=blockIdx.x*blockDim.x +threadIdx.x;
    if (n<nFD) {
    idof=((int)FDofs[n]-1)%3;
    nod=((int)FDofs[n]-1)/3;
    i=nod/((nelz+1)*(nely+1));
    k=(nod-(nelz+1)*(nely+1)*i)/(nely+1);
    j=nod-(nelz+1)*(nely+1)*i-(nely+1)*k;
    for (di = -1; di <= 0; di++)
    for (dk = -1; dk <= 0; dk++)
    for (dj = -1; dj <= 0; dj++)
    {
        iP=i+di;
        if (iP<0 || iP>nelx-1) continue;
        jP=j+dj;
        if (jP<0 || jP>nely-1) continue;
        kP=k+dk;
        if (kP<0 || kP>nelz-1) continue;
        nel=nelz*nely*iP+nely*kP+jP; 
        n1=nn[4*(di+1)+2*(dk+1)+dj+1];
        m=3*n1+idof;
        A[300*nel+(m*(m+1))/2+m]=1e10;
    }
    }
}

__global__ void prepADc(double A[], double *AD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int i, j, k, d0;
    int nod, nel;
    int n1, m, ia;
    int di, dj, dk;
    int iP, jP, kP;
    int nn[8]={6,5,2,1,7,4,3,0};
    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
    d0=threadIdx.x%3;
    
    k=blockIdx.x;
    i=blockIdx.y;
        
    nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
    AD[3*nod+d0]=0.0;
    for (di = -1; di <= 0; di++)
    {
        iP=i+di;
        if (iP<0 || iP>nelx-1) continue;
        for (dk = -1; dk <= 0; dk++)
        {
            kP=k+dk;
            if (kP<0 || kP>nelz-1) continue;
            for (dj = -1; dj <= 0; dj++)
            {
                jP=j+dj;
                if (jP<0 || jP>nely-1) continue;
                nel=nelz*nely*iP+nely*kP+jP;
                n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                m=3*n1+d0;
                ia=nel*300+((m+1)*m)/2+m;
                AD[3*nod+d0] += A[ia];
            }
        }
    }
    }
}

__global__ void prepAD0c(double *AD, double *CX, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int i, j, k, d0;
    int nod, nel;
    int n1, m, ia;
    int di, dj, dk;
    int iP, jP, kP;
    int nn[8]={6,5,2,1,7,4,3,0};
    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
    d0=threadIdx.x%3;
    
    k=blockIdx.x;
    i=blockIdx.y;
        
    nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
    AD[3*nod+d0]=0.0;
    for (di = -1; di <= 0; di++)
    {
        iP=i+di;
        if (iP<0 || iP>nelx-1) continue;
        for (dk = -1; dk <= 0; dk++)
        {
            kP=k+dk;
            if (kP<0 || kP>nelz-1) continue;
            for (dj = -1; dj <= 0; dj++)
            {
                jP=j+dj;
                if (jP<0 || jP>nely-1) continue;
                nel=nelz*nely*iP+nely*kP+jP;
                n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                m=3*n1+d0;
                AD[3*nod+d0] += CX[nel]*Ae[m][m];
            }
        }
    }
    }
}

void fprintV(char *format)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format);
    fclose(flog);
}

void fprintI(char *format, int v)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v);
    fclose(flog);
}

void fprintII(char *format, int v1, int v2)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}


void fprintIII(char *format, int v1, int v2, int v3)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v1, v2, v3);
    fclose(flog);
}

void fprintD(char *format, double v)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v);
    fclose(flog);
}

void fprintDD(char *format, double v1, double v2)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}

void fprintID(char *format, int v1, double v2)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}

void fprintIDD(char *format, int v1, double v2, double v3)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,format,v1, v2, v3);
    fclose(flog);
}

void fprintVc(char *fname, double *V0, int n, int ncol)
{
    flog=fopen("mgcg1.log", "a");
    fprintf(flog,fname);
    fprintf(flog," length=%d lines=%d\n",n,(n+2)/ncol);
    double *V = (double *)malloc(n*sizeof(double));
    HANDLE_ERROR( cudaMemcpy( V, V0, n * sizeof(double), cudaMemcpyDeviceToHost ) );
    
    for (int i = 0; i < n; ++i)
    {
        fprintf(flog,"%11.4g",V[i]);
        if ((i+1)%ncol==0)
            fprintf(flog,"\n");
    }
    //fprintf(fp,"\n");
    fprintf(flog,"\n");
    fclose(flog);
    free( V );
}

void printVc(char *fname, double *V0, int n, int ncol)
{
    FILE *fp;
    fp=fopen(fname, "w");
    fprintf(fp,fname);
    fprintf(fp," length=%d lines=%d\n",n,(n+2)/ncol);
    double *V = (double *)malloc(n*sizeof(double));
    HANDLE_ERROR( cudaMemcpy( V, V0, n * sizeof(double), cudaMemcpyDeviceToHost ) );
    
    for (int i = 0; i < n; ++i)
    {
        fprintf(fp,"%11.4g",V[i]);
        if ((i+1)%ncol==0)
            fprintf(fp,"\n");
    }
    //fprintf(fp,"\n");
    fprintf(fp,"\n");
    fclose(fp);
    free( V );
}

__global__ void interpc(double *UP, double *U, int nelx, int nely, int nelz)
{
    int i, j, k, d0, nod, idof, n;
    int nelxP=nelx/2;
    int nelyP=nely/2;
    int nelzP=nelz/2;
    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
        d0=threadIdx.x%3;
        
        k=blockIdx.x;
        i=blockIdx.y;
            
        idof=3*((nelz+1)*(nely+1)*i+(nely+1)*k+j)+d0;
        if (i%2==0 && j%2==0 && k%2==0)
        {
            U[idof] += UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*k/2+j/2)+d0];
        }
        if (i%2!=0 && j%2==0 && k%2==0)
        {
            U[idof] += 0.5*(UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*k/2+j/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*k/2+j/2)+d0]);
        }
        if (i%2==0 && j%2!=0 && k%2==0)
        {
            U[idof] += 0.5*(UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*k/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*k/2+(j+1)/2)+d0]);
        }
        if (i%2==0 && j%2==0 && k%2!=0)
        {
            U[idof] += 0.5*(UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k-1)/2+j/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k+1)/2+j/2)+d0]);
        }
        if (i%2==0 && j%2!=0 && k%2!=0)
        {
            U[idof] += 0.25*(UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k-1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k-1)/2+(j+1)/2)+d0]+
                             UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k+1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*i/2+(nelyP+1)*(k+1)/2+(j+1)/2)+d0]);
        }
        if (i%2!=0 && j%2==0 && k%2!=0)
        {
            U[idof] += 0.25*(UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k-1)/2+j/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k+1)/2+j/2)+d0]+
                             UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k-1)/2+j/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k+1)/2+j/2)+d0]);
        }
        if (i%2!=0 && j%2!=0 && k%2==0)
        {
            U[idof] += 0.25*(UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*k/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*k/2+(j+1)/2)+d0]+
                             UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*k/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*k/2+(j+1)/2)+d0]);
        }
        if (i%2!=0 && j%2!=0 && k%2!=0)
        {
            U[idof] += 0.125*(UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k-1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k-1)/2+(j+1)/2)+d0]+
                              UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k+1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i-1)/2+(nelyP+1)*(k+1)/2+(j+1)/2)+d0]+
                              UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k-1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k-1)/2+(j+1)/2)+d0]+
                              UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k+1)/2+(j-1)/2)+d0]+UP[3*((nelzP+1)*(nelyP+1)*(i+1)/2+(nelyP+1)*(k+1)/2+(j+1)/2)+d0]);
        }
    }
}

int readCX(double *CX)
{
    FILE *fp;
    fp=fopen("CX.txt", "r");
    int i=0;
    while (fscanf(fp,"%lf",&CX[i++])>0) ;
    fclose(fp);
    return i-1;
}

__constant__ double P[24][81]={{1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,},
{0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.125,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.25,0,0,0,0,0,1,0,0,0.5,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,},
{0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0,0,0.25,0,0,0,0,0,0.25,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};

__global__ void coarseAe3c(double AP[],double A[], int nelx, int nely, int nelz)
{
    double AL[81][81];
    double ALP[81][24];
    
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int nod0, nod1, nod2;
    int i, j, k, m, n, n1, n2, d1, d2;
    int iP, jP, kP, nel, nelP, iA, jA, l, iap;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    int ne22[8]={0,9,10,1,3,12,13,4};

    j=threadIdx.x;
    k=blockIdx.x;
    i=blockIdx.y;
    
    nel=nelz*nely*i+nely*k+j;
    for (jA = 0; jA < 81; jA++)
    for (iA = 0; iA < 81; iA++)
         AL[iA][jA]=0.0;
         
    for (iP = 0; iP < 2; ++iP)
    for (kP = 0; kP < 2; ++kP)
    for (jP = 0; jP < 2; ++jP)
    {
        nelP=nelzP*nelyP*(2*i+iP)+nelyP*(2*k+kP)+2*j+jP;
        nod0=9*iP+3*kP+jP;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne22[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne22[n2];
                for (d1=0; d1<3; ++d1)
                for (d2=0; d2<3; ++d2)
                {
                    m=3*n1+d1;
                    n=3*n2+d2;
                    if (n<=m)
                        iap=nelP*300+((m+1)*m)/2+n;
                    else
                        iap=nelP*300+((n+1)*n)/2+m;
                    AL[3*nod1+d1][3*nod2+d2]  += AP[iap];
                }
            }
        }
    }
    
    for (jA = 0; jA < 24; jA++)
    for (iA = 0; iA < 81; iA++)
    {
        ALP[iA][jA]=0.0;
        for (l = 0; l < 81; l++)
            ALP[iA][jA] +=AL[iA][l]*P[jA][l];
    }
    for (iA = 0; iA < 24; iA++)
    for (jA = 0; jA <= iA; jA++)
    {
        A[nel*300+((iA+1)*iA)/2+jA]=0.0;
        for (l = 0; l < 81; l++)
            A[nel*300+((iA+1)*iA)/2+jA] +=ALP[l][jA]*P[iA][l];
    }
}

//#define ASQ(i,j) AQ[i*ndof+j]

__global__ void assembAc(int nA, double *A, double *AQ, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d0, d1, ia, m;
    int di, dj, dk;
    int iP, jP, kP;
    int nn[8]={6,5,2,1,7,4,3,0};
    int ne[8]={0,0,0,0,0,0,0,0};
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];

    
    j=(blockIdx.z*blockDim.x)/3+threadIdx.x/3;
    if (j<=nely) {
    d0=threadIdx.x%3;
    k=blockIdx.x;
    i=blockIdx.y;
    
    nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
    
    for (di = -1; di <= 0; di++)
    {
        iP=i+di;
        if (iP<0 || iP>nelx-1) continue;
        for (dk = -1; dk <= 0; dk++)
        {
            kP=k+dk;
            if (kP<0 || kP>nelz-1) continue;
            for (dj = -1; dj <= 0; dj++)
            {
                jP=j+dj;
                if (jP<0 || jP>nely-1) continue;
                nel=nelz*nely*iP+nely*kP+jP;
                nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                m=3*n1+d0;
                for (n2 = 0; n2 < 8; ++n2)
                {
                    nod2=nod1+ne[n2];
                    for (d1=0; d1<3; ++d1)
                    {
                        n=3*n2+d1;
                        if (n<=m)
                            ia=nel*300+((m+1)*m)/2+n;
                        else
                            ia=nel*300+((n+1)*n)/2+m;
                        AQ[(3*nod0+d0)*ndof+3*nod2+d1] += A[ia];
                    }
                }
            }
        }
    }
    }
}

void printCsAe(double A[][24][24], int nelx, int nely, int nelz)
{
    int i, j, k, iA, jA, nel;
    
    FILE *fp;
    fp=fopen("logAF", "w");
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
        nel=nelz*nely*i+nely*k+j;
        fprintf(fp,"\n nel=%d\n",nel);
        for (jA = 0; jA < 24; jA++)
        {
            for (iA = 0; iA < 24; iA++)
                    fprintf(fp,"%9.5g,",A[nel][iA][jA]);
            fprintf(fp,"\n");
        }   
    }
    
    fclose(fp);
}

void CmpCsAe(double A[][24][24], double A1[][24][24], int nelx, int nely, int nelz)
{
    int i, j, k, iA, jA, nel;
    double dif, dc;
    FILE *fp;
    fp=fopen("logAF", "w");
    dif=0.0;
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
        //dif=0.0;
        nel=nelz*nely*i+nely*k+j;
        //fprintf(fp,"\n nel=%d\n",nel);
        for (jA = 0; jA < 24; jA++)
            for (iA = 0; iA < 24; iA++)
                dc=fabs(A1[nel][iA][jA]-A[nel][iA][jA]);
                if (dif<dc) dif=dc;
        //fprintf(fp,"%20.10g\n",dif);
    }
    fprintf(fp,"%20.10g\n",dif);
    fclose(fp);
}

void defD(double D[6][6],double v)
{
    double v1=1.0-v;
    double v2=(1.0-2.0*v)/2.0;
    double C1=0.5/(1+v)/v2;
    //double D[6][6]=0.5/(1+v)/v2*{ {v1, v,  v,  0,  0,  0},
    //                              {v,  v1, v,  0,  0,  0},
    //                              {v,  v,  v1, 0,  0,  0},
    //                              {0,  0,  0,  v2, 0,  0},
    //                              {0,  0,  0,  0,  v2, 0},
    //                              {0,  0,  0,  0,  0,  v2} };
    //set0(D,36);
    D[0][0]=C1*v1;
    D[1][1]=C1*v1;
    D[2][2]=C1*v1;
    D[3][3]=C1*v2;
    D[4][4]=C1*v2;
    D[5][5]=C1*v2;
    
    D[0][1]=C1*v;
    D[0][2]=C1*v;
    D[1][0]=C1*v;
    D[1][2]=C1*v;
    D[2][0]=C1*v;
    D[2][1]=C1*v;
}

void Bem3(double B[6][24],double s,double e,double t,double hx,double hy,double hz)
{
    double N1x=-0.25*(1-e)*(1-t)/hx;
    double N2x= 0.25*(1-e)*(1-t)/hx;
    double N3x= 0.25*(1+e)*(1-t)/hx;
    double N4x=-0.25*(1+e)*(1-t)/hx;
    double N5x=-0.25*(1-e)*(1+t)/hx;
    double N6x= 0.25*(1-e)*(1+t)/hx;
    double N7x= 0.25*(1+e)*(1+t)/hx;
    double N8x=-0.25*(1+e)*(1+t)/hx;
    double N1y=-0.25*(1-s)*(1-t)/hy;
    double N2y=-0.25*(1+s)*(1-t)/hy;
    double N3y= 0.25*(1+s)*(1-t)/hy;
    double N4y= 0.25*(1-s)*(1-t)/hy;
    double N5y=-0.25*(1-s)*(1+t)/hy;
    double N6y=-0.25*(1+s)*(1+t)/hy;
    double N7y= 0.25*(1+s)*(1+t)/hy;
    double N8y= 0.25*(1-s)*(1+t)/hy;
    double N1z=-0.25*(1-s)*(1-e)/hz;
    double N2z=-0.25*(1+s)*(1-e)/hz;
    double N3z=-0.25*(1+s)*(1+e)/hz;
    double N4z=-0.25*(1-s)*(1+e)/hz;
    double N5z= 0.25*(1-s)*(1-e)/hz;
    double N6z= 0.25*(1+s)*(1-e)/hz;
    double N7z= 0.25*(1+s)*(1+e)/hz;
    double N8z= 0.25*(1-s)*(1+e)/hz;

    //set0(B,24*6);
    B[0][ 0]=N1x;
    B[0][ 3]=N2x;
    B[0][ 6]=N3x;
    B[0][ 9]=N4x;
    B[0][12]=N5x;
    B[0][15]=N6x;
    B[0][18]=N7x;
    B[0][21]=N8x;

    B[1][ 1]=N1y;
    B[1][ 4]=N2y;
    B[1][ 7]=N3y;
    B[1][10]=N4y;
    B[1][13]=N5y;
    B[1][16]=N6y;
    B[1][19]=N7y;
    B[1][22]=N8y;

    B[2][ 2]=N1z;
    B[2][ 5]=N2z;
    B[2][ 8]=N3z;
    B[2][11]=N4z;
    B[2][14]=N5z;
    B[2][17]=N6z;
    B[2][20]=N7z;
    B[2][23]=N8z;

    B[3][ 0]=N1y;
    B[3][ 3]=N2y;
    B[3][ 6]=N3y;
    B[3][ 9]=N4y;
    B[3][12]=N5y;
    B[3][15]=N6y;
    B[3][18]=N7y;
    B[3][21]=N8y;

    B[3][ 1]=N1x;
    B[3][ 4]=N2x;
    B[3][ 7]=N3x;
    B[3][10]=N4x;
    B[3][13]=N5x;
    B[3][16]=N6x;
    B[3][19]=N7x;
    B[3][22]=N8x;

    B[4][ 1]=N1z;
    B[4][ 4]=N2z;
    B[4][ 7]=N3z;
    B[4][10]=N4z;
    B[4][13]=N5z;
    B[4][16]=N6z;
    B[4][19]=N7z;
    B[4][22]=N8z;

    B[4][ 2]=N1y;
    B[4][ 5]=N2y;
    B[4][ 8]=N3y;
    B[4][11]=N4y;
    B[4][14]=N5y;
    B[4][17]=N6y;
    B[4][20]=N7y;
    B[4][23]=N8y;

    B[5][ 0]=N1z;
    B[5][ 3]=N2z;
    B[5][ 6]=N3z;
    B[5][ 9]=N4z;
    B[5][12]=N5z;
    B[5][15]=N6z;
    B[5][18]=N7z;
    B[5][21]=N8z;

    B[5][ 2]=N1x;
    B[5][ 5]=N2x;
    B[5][ 8]=N3x;
    B[5][11]=N4x;
    B[5][14]=N5x;
    B[5][17]=N6x;
    B[5][20]=N7x;
    B[5][23]=N8x;

    //B=[N1x   0   0 N2x   0   0 N3x   0   0 N4x   0   0 N5x   0   0 N6x   0   0 N7x   0   0 N8x   0   0; ...
    //     0 N1y   0   0 N2y   0   0 N3y   0   0 N4y   0   0 N5y   0   0 N6y   0   0 N7y   0   0 N8y   0; ...
    //     0   0 N1z   0   0 N2z   0   0 N3z   0   0 N4z   0   0 N5z   0   0 N6z   0   0 N7z   0   0 N8z; ...
    //   N1y N1x   0 N2y N2x   0 N3y N3x   0 N4y N4x   0 N5y N5x   0 N6y N6x   0 N7y N7x   0 N8y N8x   0; ...
    //     0 N1z N1y   0 N2z N2y   0 N3z N3y   0 N4z N4y   0 N5z N5y   0 N6z N6y   0 N7z N7y   0 N8z N8y; ...
    //   N1z   0 N1x N2z   0 N2x N3z   0 N3x N4z   0 N4x N5z   0 N5x N6z   0 N6x N7z   0 N7x N8z   0 N8x];
}


void prepAelm(double v, double hx,double hy,double hz)
{
    int ia;
    int iP, jP, kP;
    double D[6][6]={0};
    double B[6][24]={0};
    double DB[6][24]={0};
    double integP[2]={-1/sqrt(3), 1/sqrt(3)};
    double s, e, t;
    int m, n, l, na;
    
    defD(D,v);
    for (iP = 0; iP < 2; ++iP)
    for (kP = 0; kP < 2; ++kP)
    for (jP = 0; jP < 2; ++jP)
    {
        na=4*iP+2*kP+jP;
        s=integP[iP];
        e=integP[jP];
        t=integP[kP];
        Bem3(B,s,e,t,hx,hy,hz);
        for (int m=0; m<6; m++)
        for (int n=0; n<24; n++)
        {
            DB[m][n]=0.0;
            for (int l=0; l<6; l++)
                DB[m][n] +=D[m][l]*B[l][n];
        }
                
        for (int m=0; m<24; m++)
        {
            for (int n=0; n<=m; n++)
            {
                ia=((m+1)*m)/2+n;
                Aelm[ia][na] = 0.0;
                for (int l=0; l<6; l++)
                    Aelm[ia][na] +=0.125*hx*hy*hz*B[l][m]*DB[l][n];
            }
        }
    }
}

__global__ void coarseAe2c(double *A, double *CXP, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int i, j, k, ia, ial;
    int iP, jP, kP, nel, nelP;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    int m, n, l, na;
    
    j=threadIdx.x;
    k=blockIdx.x;
    i=blockIdx.y;
    
    nel=nelz*nely*i+nely*k+j;
    for (m=0; m<300; m++)
        A[nel*300+m] = 0.0;
         
    for (iP = 0; iP < 2; ++iP)
    for (kP = 0; kP < 2; ++kP)
    for (jP = 0; jP < 2; ++jP)
    {
        na=4*iP+2*kP+jP;
        nelP=nelzP*nelyP*(2*i+iP)+nelyP*(2*k+kP)+2*j+jP;
        for (m=0; m<300; m++)
            A[nel*300+m] +=CXP[nelP]*Aelc[m][na];
    }
}

#define aq(i,j) aq[i*n+j]

void choldc(int n, double aq[], double p[])
//Given a positive-definite symmetric matrix a[1..n][1..n], this routine constructs its Cholesky
//decomposition, A = L · LT . On input, only the upper triangle of a need be given; it is not
//modified. The Cholesky factor L is returned in the lower triangle of a, except for its diagonal
//elements which are returned in p[1..n].
{
    int i,j,k;
    double sum;
    for (i=0;i<n;i++) {
    for (j=i;j<n;j++) {
        sum=aq(i,j);
        for (k=i-1;k>=0;k--)
        {
            sum -= aq(i,k)*aq(j,k);
        }
        if (i == j) {
            if (sum <= 0.0)                //a, with rounding errors, is not positive definite.
                printf("choldc failed %d %g\n",i,sum);
            p[i]=sqrt(sum);
        }
        else aq(j,i)=sum/p[i];
    }
    }
}

void choldc1(int n, double aq[], double p[])
//Given a positive-definite symmetric matrix a[1..n][1..n], this routine constructs its Cholesky
//decomposition, A = L · LT . On input, only the upper triangle of a need be given; it is not
//modified. The Cholesky factor L is returned in the lower triangle of a, except for its diagonal
//elements which are returned in p[1..n].
{
    int i;
    int nThr;
    for (i=0;i<n;i++)
    {
        nThr=(n-i<=maxThreads)? n-i : maxThreads;
        cholRow<<<1,nThr>>>(aq,p,i,n);
        if(printLev>5) check();
    }
}

__global__ void cholRow(double aq[], double p[], int i, int n)
{
    int j,k;
    double sum;
    j=i+threadIdx.x;
    if (j==i) {
        sum=aq(i,j);
        for (k=i-1;k>=0;k--)
            sum -= aq(i,k)*aq(j,k);
        if (sum <= 0.0)                //a, with rounding errors, is not positive definite.
           fprintIDc("choldc failed %d %g\n",i,sum);
        p[i]=sqrt(sum);
    }
    __syncthreads();
    
    if (j>i)
    {
        while (j < n)
        {
            sum=aq(i,j);
            for (k=i-1;k>=0;k--)
                sum -= aq(i,k)*aq(j,k);
            aq(j,i)=sum/p[i];
            j +=blockDim.x;
        }
    }
    //__syncthreads();
}

__global__ void choldcc(int n, double aq[], double p[])
//Given a positive-definite symmetric matrix a[1..n][1..n], this routine constructs its Cholesky
//decomposition, A = L · LT . On input, only the upper triangle of a need be given; it is not
//modified. The Cholesky factor L is returned in the lower triangle of a, except for its diagonal
//elements which are returned in p[1..n].
{
    int i,j,k;
    double sum;
    
    for (i=0;i<n;i++) {
    for (j=i;j<n;j++) {
        sum=aq(i,j);
        for (k=i-1;k>=0;k--)
        {
            sum -= aq(i,k)*aq(j,k);
        }
        if (i == j) {
            if (sum <= 0.0)                //a, with rounding errors, is not positive definite.
                fprintIDc("choldc failed %d %g\n",i,sum);
            p[i]=sqrt(sum);
        }
        else aq(j,i)=sum/p[i];
    }
    }
    //__syncthreads();
}

__device__ void fprintIDc(char *format, int v1, double v2)
{
}

void cholsl1(int n, double aq[], double p[], double b[], double x[])
// Solves the set of n linear equations A · x = b, where a is a positive-definite symmetric matrix.
// a[1..n][1..n] and p[1..n] are input as the output of the routine choldc. Only the lower
// subdiagonal portion of a is accessed. b[1..n] is input as the right-hand side vector. The
// solution vector is returned in x[1..n]. a, n, and p are not modified and can be left in place
// for successive calls with different right-hand sides b. b is not modified unless you identify b and
//  x in the calling sequence, which is allowed.
{
    int i;
    copyVVc<<<1,maxThreads>>>(b,x,n); if(printLev>3)check();
    int nThr=(n<=maxThreads)? n : maxThreads;

    //for (i=0;i<n;i++)
    //    x[i]=b[i];
    for (i=0;i<n;i++) {                                    //Solve L · y = b, storing y in x.
        //x[i]=x[i]/p[i];
        cholslCol1<<<1,nThr>>>(n, i, aq, p, x);
        //for (k=i+1; k<n; k++)
        //    x[k] -= aq(k,i)*x[i];
    }
    
    for (i=n-1;i>=0;i--) {                                    //Solve LT · x = y.
        //x[i]=x[i]/p[i];
        cholslCol2<<<1,nThr>>>(n, i, aq, p, x);
        //for (k=i-1; k>=0; k--)
        //    x[k] -= aq(i,k)*x[i];
    }
}

__global__ void cholslCol1(int n, int i, double aq[], double p[], double x[])
{
    if (threadIdx.x==0)
    x[i]=x[i]/p[i];
    __syncthreads();
    
    int k=i+1+threadIdx.x;
    while (k<n)
    {
        x[k] -= aq(k,i)*x[i];
        k +=blockDim.x;
    }
    //__syncthreads();
}

__global__ void cholslCol2(int n, int i, double aq[], double p[], double x[])
{
    if (threadIdx.x==0)
    x[i]=x[i]/p[i];
    __syncthreads();
    
    int k=i-1-threadIdx.x;
    while (k>=0)
    {
        x[k] -= aq(i,k)*x[i];
        k -=blockDim.x;
    }
    //__syncthreads();
}

void cholsl(int n, double aq[], double p[], double b[], double x[])
// Solves the set of n linear equations A · x = b, where a is a positive-definite symmetric matrix.
// a[1..n][1..n] and p[1..n] are input as the output of the routine choldc. Only the lower
// subdiagonal portion of a is accessed. b[1..n] is input as the right-hand side vector. The
// solution vector is returned in x[1..n]. a, n, and p are not modified and can be left in place
// for successive calls with different right-hand sides b. b is not modified unless you identify b and
//  x in the calling sequence, which is allowed.
{
    int i,k;
    double sum;
    for (i=0;i<n;i++) {                                    //Solve L · y = b, storing y in x.
        for (sum=b[i],k=i-1;k>=0;k--) sum -= aq(i,k)*x[k];
        x[i]=sum/p[i];
    }
    for (i=n-1;i>=0;i--) {                                    //Solve LT · x = y.
    for (sum=x[i],k=i+1;k<n;k++) sum -= aq(k,i)*x[k];
        x[i]=sum/p[i];
    }
}

__global__ void cholslc(int n, double aq[], double p[], double b[], double x[])
// Solves the set of n linear equations A · x = b, where a is a positive-definite symmetric matrix.
// a[1..n][1..n] and p[1..n] are input as the output of the routine choldc. Only the lower
// subdiagonal portion of a is accessed. b[1..n] is input as the right-hand side vector. The
// solution vector is returned in x[1..n]. a, n, and p are not modified and can be left in place
// for successive calls with different right-hand sides b. b is not modified unless you identify b and
//  x in the calling sequence, which is allowed.
{
    int i,k;
    double sum;
    for (i=0;i<n;i++) {                                    //Solve L · y = b, storing y in x.
        for (sum=b[i],k=i-1;k>=0;k--) sum -= aq(i,k)*x[k];
        x[i]=sum/p[i];
    }
    for (i=n-1;i>=0;i--) {                                    //Solve LT · x = y.
    for (sum=x[i],k=i+1;k<n;k++) sum -= aq(k,i)*x[k];
        x[i]=sum/p[i];
    }
}
