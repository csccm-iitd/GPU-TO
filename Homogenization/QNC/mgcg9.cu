// 
// revision 12:13 26.07.2021
// nswp parameter 
// 

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include "mex.h"
#include <assert.h>
#include "matrix.h"

extern "C" bool utIsInterruptPending();

void mgcg9(double Ae[][24],double *F,double *U, double *CX, double *FDofs, int nFD,  int nelx, int nely, int nelz, int nl, double tol, int maxit, int nswp, double *res, int *nit);
void initVcycle(double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0);
void vcycle(double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0);
void printV(const char *fname, double *V, int n, int ncol);
void set0(double *V, int n);
double scalVV(double *V1, double *V2, int ndof);
int readCX(double *CX);
void printCsAe(double A[][24][24], int nelx, int nely, int nelz);
void CmpCsAe(double A[][24][24],double A1[][24][24], int nelx, int nely, int nelz);
void defD(double D[6][6],double v);
void Bem3(double B[6][24],double s,double e,double t,double hx,double hy,double hz);
int estWorkSpace(int nFD0, int nl, int nelx0, int nely0, int nelz0);
void fprintV(const char *format);
void fprintI(const char *format, int v);
void fprintII(const char *format, int v1, int v2);
void fprintIII(const char *format, int v1, int v2, int v3);
void fprintD(const char *format, double v);
void fprintDD(const char *format, double v1, double v2);
void fprintID(const char *format, int v1, double v2);
void fprintIID(const char *format, int v1, int v2, double v3);
void fprintIDD(const char *format, int v1, double v2, double v3);
void fprintVc(const char *fname, int lev, double *V0, int n, int ncol);

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
      flog=fopen("mgcg9.log", "a");
      fprintf(flog,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      fclose(flog);
      if (abort) exit(code);
   }
}

#define check() gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize())

#define maxThreads 1024
#define maxDofs0 20000

__global__ void multAV0c(double *U,double *R,double *CX, int nelx, int nely, int nelz);
__global__ void defect0c(double *F,double *U,double *R, double *CX, int nelx, int nely, int nelz);
__global__ void copyVVc(double *V1, double *V2, int n);
__global__ void set0c(double *V, int n);
__global__ void scalVVc(double *V1, double *V2, int n, double *rnorm);

void choldc1(int n, double aq[], double p[]);
__global__ void cholRow1(double aq[], double p[], int i, int n);
__global__ void cholRow2(double aq[], double p[], int i, int n);
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

void printVc(const char *fname, double *V0, int n, int ncol);

void formCXcEnv(double *CXP, double *CX, int nelx, int nely, int nelz);
__global__ void formCXc(double *CXP, double *CX, int nelx, int nely, int nelz);
__global__ void formADc(double *AD, double *CX, int nelx, int nely, int nelz);
__global__ void assembAQc(double *CX, double *AQ, int nelx, int nely, int nelz);
__global__ void CorrCsAQc(double *AQ, double *FDofs, int nFD, int nelx, int nely, int nelz);

void interpEnv(double *UP, double *V, double *U, double *CX, int nelx, int nely, int nelz);
void restrEnv(double *FC, double *V, double *R, double *CX, double *AZ, int nelx, int nely, int nelz);
void relaxEnv(double *AD, double *U, double *F, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void relaxXYEnv(double *AD, double *U, double *F, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
__global__ void formAZc(double *AZ, double *CX, int nelx, int nely, int nelz);

static int printLev=1;

double Aelm[300][8];

__constant__ double Aelc[300][8];

__constant__ double Ae[24][24];

double nu=0.3, hx0=1, hy0=1, hz0=1;


/*-----------------------------------------------------------------------*/
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *Aep, *F, *U, *CX, *FDofs, *W;
    double Ae[24][24];
    double tol, res;
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
    int nFD = (int) mxGetScalar(prhs[5]); 
    //mexPrintf("\n%d nFD\n", nFD);
    nelx = (int) mxGetScalar(prhs[6]); 
    nely = (int) mxGetScalar(prhs[7]); 
    nelz = (int) mxGetScalar(prhs[8]); 
    nl   = (int) mxGetScalar(prhs[9]); 
    tol  =       mxGetScalar(prhs[10]); 
    maxit= (int) mxGetScalar(prhs[11]);
    nswp = (int) mxGetScalar(prhs[12]);
    
    printLev= (int) mxGetScalar(prhs[13]); 
    nu   =       mxGetScalar(prhs[14]); 
    hx0  =       mxGetScalar(prhs[15]); 
    hy0  =       mxGetScalar(prhs[16]); 
    hz0  =       mxGetScalar(prhs[17]); 
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
        //printV("mgcg9F",F, nelx, nely, nelz);
        mgcg9(Ae, F, U, CX, FDofs, nFD, nelx, nely, nelz, nl, tol, maxit, nswp, &res, &nit);
        plhs[0] = mxCreateDoubleScalar(res);
        plhs[1] = mxCreateDoubleScalar((double)nit);
    }
}

void mgcg9(double Ae0[][24],double *F0,double *U0, double *CX0, double *FDofs0, int nFD, int nelx, int nely, int nelz, int nl, double tol, int maxit, int nswp, double *res, int *nit)
{
    flog=fopen("mgcg9.log", "w");
    fclose(flog);
    //int nswp=10;
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
    //if (printLev>=3) {
    //      mexPrintf("\nReported GPU memory. Total: %d kb Free: %d kb\n",t/1024,f/1024);
    //      mexPrintf("\nNumber of dofs %d, Device memory required %d (Mb)\n", ndof, (lenW * sizeof(double))/1048576+1);
    //}
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
    
    set0c<<<1,maxThreads>>>(P, ndof); if(printLev>3)check();
    set0c<<<1,maxThreads>>>(Z, ndof); if(printLev>3)check();
    
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
        set0c<<<1,maxThreads>>>(Z, ndof); if(printLev>3)check();
        vcycle(R, Z, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
        rho=scalVV(R,Z,ndof);
        beta = rho/rho_p;
        addVsVc<<<1,maxThreads>>>(Z,P,P,beta,ndof); if(printLev>3)check();
        rho_p = rho;
        *res = sqrt(scalVV(R,R,ndof));
        relres = *res/res0;
        clock_t t1=clock();
        if (printLev>=3) {
            fprintIDD("CG It %d Time %8.3g  %12.6g\n",itcg, (double)(t1-t0)/CLOCKS_PER_SEC,relres);
            //mexPrintf("CG It %d Time %8.3g  %12.6g\n",itcg, (double)(t1-t0)/CLOCKS_PER_SEC,relres);
        }
        if (utIsInterruptPending()) {        /* check for a Ctrl-C event */
            mexPrintf("Ctrl-C Detected. END\n\n");
            break;
        }
        if (relres < tol || itcg>=maxit)
            break;
    }
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
            lW += 4*ndof + nnel + nFD;
            
        if (lev==nl-1 && lev!=0 && ndof<maxDofs0) {
            lW += ndof;
            lW += ndof*ndof;
        }
    }
    return lW;
}

static int levW[20], lnFD[20];

void initVcycle(double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0)
{
    int n, nnel, nnod, ndof, nelx, nely, nelz;
    double *U, *F, *R, *CX, *AD, *AQ, *FDofs, *UP, *RP, *CXP, *FDofsP, *P;
    double hx, hy, hz;
    int nFD, lW;
    int nnel0=nelx0*nely0*nelz0;
    int nnod0=(nelx0+1)*(nely0+1)*(nelz0+1);
    int ndof0=3*nnod0;
    ndof=ndof0;
    nFD=nFD0;
    lnFD[0]=nFD0;
    int nBlocksY, nBlocksX, nBlocksZ, nThreadsX, nThreadsY, nThreadsZ=1;

    lW=0;   
    for (int lev = 0; lev <nl; ++lev)
    {
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
        dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
        dim3 threads(nThreadsX,nThreadsY);
        if (lev==0)
        {
            CX=CX0;
            FDofs=FDofs0;
            nFD=nFD0;
            R=&W[lW]; lW += ndof;
            AD=&W[lW]; lW += ndof;
        }
        else
        {
            FDofsP=FDofs;
            U=&W[lW];  lW += ndof;
            F=&W[lW];  lW += ndof;
            R=&W[lW];  lW += ndof;
            CXP=CX;
            CX=&W[lW];  lW += nnel;
            AD=&W[lW]; lW += ndof;
            FDofs=&W[lW];
            int nFDP=lnFD[lev-1];
            rstrFDcenv(FDofsP,FDofs, nelx, nely, nelz, nFDP, &nFD);
            lW += nFD;
            lnFD[lev]=nFD;
            formCXcEnv(CXP, CX, nelx, nely, nelz);
        }
        formADc<<<grids,threads>>>(AD, CX, nelx, nely, nelz); if(printLev>3)check();

        if (lev==nl-1 && lev!=0 && ndof<maxDofs0)
        {
            P=&W[lW]; lW += ndof;
            AQ=&W[lW]; lW += ndof*ndof;
            set0c<<<1,maxThreads>>>(AQ,ndof*ndof); if(printLev>3)check();
            assembAQc<<<grids,threads>>>(CX, AQ, nelx, nely, nelz); if(printLev>3)check();
            CorrCsAQc<<<(nFD+maxThreads-1)/maxThreads,maxThreads>>>(AQ, FDofs, nFD, nelx, nely, nelz); if(printLev>3)check();
            choldc1(ndof, AQ,P);
        }
        //fprintI("MGCG initVcycle Lev=%d\n",lev);
    }
    if (printLev>1) fprintII("Number of dof %d  Working space %d\n",ndof0,lW+6*ndof0+nnel0+nFD0);
}

void vcycle(double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0)
{
    int gamma[8]={1,1,1,1,1,1,1,1};
    int numLevIt[8]={0,0,0,0,0,0,0,0};

    int n, nnod, ndof, nnel, nelx, nely, nelz;
    double *U, *F, *R, *CX, *A, *AD, *AQ, *FDofs, *UP, *RP, *CXP, *FDofsP, *P;
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
    
    int level=0;
    newcycle:
    for (int l=level+1; l<nl; ++l)
        numLevIt[l]=0;
    for (int lev = level; lev <nl; ++lev)
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
            CX=&W[lW];  lW += nnel;
            AD=&W[lW]; lW += ndof;
            nFD=lnFD[lev];
            FDofs=&W[lW]; lW += nFD;
            if (numLevIt[lev]==0) {
                //restrEnv(F, VP, RP, CXP, AZP, nelx*2, nely*2, nelz*2); if(printLev>3)check();
                rstrFc<<<grids1,threads1>>>(RP,F,FDofs, nelx, nely, nelz, nFD); if(printLev>3)check();
                if (printLev>3) fprintID("lev %d Aft Restr %12.6g\n",lev,sqrt(scalVV(F,F,ndof)));
                ZeroBCc<<<grids1,threads1>>>(F,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            }
        }
        if (printLev>9 && ndof<maxDofs0) fprintVc("F",lev,F,ndof,(nely+1)*3);

        if (lev<nl-1 || lev==0)
        {
            if (numLevIt[lev]==0) {
                if (printLev>5) {
                    defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
                    fprintID("lev %d Bef Rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
                }
                if (printLev>9 && ndof<maxDofs0) fprintVc("R",lev,R,ndof,(nely+1)*3);

                preRelc<<<grids1,threads1>>>(AD, F, U, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>4) {
                    defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
                    fprintID("lev %d Aft preRelc %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
                }
            }
            for (int it = 0; it < nswp; ++it)
            {
                //relaxEnv(AD, U, F, R, CX, FDofs, nFD, nelx, nely, nelz);
                //relaxXYEnv(AD, U, F, R, CX, FDofs, nFD, nelx, nely, nelz);
                defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
                ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
                if (printLev>4)
                    fprintIID("lev %d Rel1 %d %12.6g\n",lev, it+1, sqrt(scalVV(R,R,ndof)));
                Jacobc<<<grids1,threads1>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
            }
            defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
            ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>3)
               fprintID("lev %d Aft Rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        }
        else
        {
            if (ndof <maxDofs0) {
                P=&W[lW]; lW += ndof;
                AQ=&W[lW]; lW += ndof*ndof;
                if (printLev>5)
                {
                    defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
                    if (printLev>9 && ndof<maxDofs0) fprintVc("R",lev,R,ndof,(nely+1)*3);
                    fprintID("lev %d Bef DirSol %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
                }
                cholsl1(ndof,AQ,P,F,U);
                if (printLev>9 && ndof<maxDofs0) fprintVc("U",lev,U,ndof,(nely+1)*3);
                if (printLev>9 && nnel<maxDofs0) fprintVc("CX",lev,CX,nnel,nely);
            }
            else {
                for (int it = 0; it < 10*nswp; ++it)
                {
                    //relaxEnv(AD, U, F, R, CX, FDofs, nFD, nelx, nely, nelz);
                    defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
                    ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
                    if (printLev>4)
                        fprintIID("lev %d Rel1 %d %12.6g\n",lev, it+1, sqrt(scalVV(R,R,ndof)));
                    Jacobc<<<grids1,threads1>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
                }
            }
            if (printLev>3)
            {
                defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
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
            CX=&W[lW];  lW += nnel;
            AD=&W[lW]; lW += ndof;
            nFD=lnFD[lev];
            FDofs=&W[lW]; lW += nFD;
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
        
        //interpEnv(UP, V, U, CX, nelx, nely, nelz);
        interpc<<<grids2,threads2>>>(UP,U, nelx, nely, nelz); if(printLev>3)check();
        if (printLev>5 && lev==nl-2) fprintVc("U",lev,U,ndof,(nely+1)*3);
        ZeroBCc<<<grids2,threads2>>>(U,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
        if (printLev>9 && ndof<maxDofs0) fprintVc("U",lev,U,ndof,(nely+1)*3);
        defect0c<<<grids2,threads2>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
        ZeroBCc<<<grids2,threads2>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
        if (printLev>5 && lev==nl-2) fprintVc("R",lev,R,ndof,(nely+1)*3);
        if (printLev>3)
            fprintID("lev %d Bef rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        for (int it = 0; it < nswp; ++it)
        {
            //relaxEnv(AD, U, F, R, CX, FDofs, nFD, nelx, nely, nelz);
            //relaxXYEnv(AD, U, F, R, CX, FDofs, nFD, nelx, nely, nelz);
            Jacobc<<<grids2,threads2>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
            defect0c<<<grids2,threads2>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
            ZeroBCc<<<grids2,threads2>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
            if (printLev>4)
                fprintIID("lev %d Rel2 %d %12.6g\n",lev,it+1, sqrt(scalVV(R,R,ndof)));
        }
        if (printLev>3)
            fprintID("lev %d Aft rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        if (printLev>9 && ndof<maxDofs0) fprintVc("R",lev,R,ndof,(nely+1)*3);
        if (printLev>9 && ndof<maxDofs0) fprintVc("F",lev,F,ndof,(nely+1)*3);

        numLevIt[lev]++;
        if (numLevIt[lev]<gamma[lev]) {
            level=lev;
            goto newcycle;
        }
    }
    if (printLev>2) fprintV("\n");
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
    
    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
        d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
        
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

void formCXcEnv(double *CXP, double *CX, int nelx, int nely, int nelz)
// envelope for formCXc
{
    int nBlocksY = nelx;
    int nBlocksX = nelz;
    int nBlocksZ = (nely+maxThreads-1)/maxThreads;
    int nThreadsX=(nely<=maxThreads) ? nely : maxThreads;
    int nThreadsY=1;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    dim3 threads(nThreadsX,nThreadsY);
    formCXc<<<grids,threads>>>(CXP, CX, nelx, nely, nelz); if(printLev>3)check();
}

__global__ void formCXc(double *CXP, double *CX, int nelx, int nely, int nelz)
// define density array CX for lev>0 from densities CXP of lev-1
{
    int i, j, k, di, dj, dk, n, nod, nodP, nel, nelP;
    int iP, jP, kP;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double CHV,CHR, CA, CG;
    
    j=blockIdx.z*blockDim.x+threadIdx.x;
    if (j<nely) {
        k=blockIdx.x;
        i=blockIdx.y;
        
        nel=nelz*nely*i+nely*k+j;
        CHV = 0.0;
        CHR = 0.0;
        for (iP = 0; iP < 2; ++iP)
        for (kP = 0; kP < 2; ++kP)
        for (jP = 0; jP < 2; ++jP)
        {
            nelP=nelzP*nelyP*(2*i+iP)+nelyP*(2*k+kP)+2*j+jP;
            CHV += CXP[nelP]; 
            CHR += 1.0/CXP[nelP]; 
        }
        CHV = CHV/8;
        CHR = 8.0/CHR;
        CA=0.5*(CHV+CHR);
        CG=sqrt(CHV*CHR);
        //CX[nel]=CA+CG;  // Multipler 1/2 is ommited to take into account that Ke(2h)=2Ke(h)
                        // diverges on floor structure void 10^-6
        //CX[nel]=2*CHR;  // diverges on floor structure void 10^-6
        //CX[nel]=2*CA;  // ?? 
        CX[nel]=2*CHV;  // best stable convergence
    }
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


    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
        d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
        
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

__global__ void copyVVc(double *V1, double *V2, int n)
{
    int i=threadIdx.x;
    while(i<n) {
        V2[i]=V1[i];
        i +=blockDim.x;
    }
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
       U[n]=0.67*U[n]+0.33*(R[n]+AD[n]*U[n])/AD[n];
       //U[n]=0.52*U[n]+0.48*(R[n]+AD[n]*U[n])/AD[n];
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


    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
        d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
        
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
                    if (CX[nel]<1e-8) continue;
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

__global__ void ZeroBCc(double *U, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int n=blockIdx.y*3*(nely+1)*(nelz+1)+blockIdx.x*3*(nely+1)+ blockIdx.z*blockDim.x+threadIdx.x;
    if (n<nFD)
        U[(int)FDofs[n]-1]=0.0;
}

__global__ void CorrCsAQc(double *AQ, double *FDofs, int nFD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int idof, i, n;
    double AQii;

    n=blockIdx.x*blockDim.x +threadIdx.x;
    if (n<nFD) {
        idof=(int)FDofs[n]-1;
        //AQ[idof*ndof+idof] *= 1E10;
        //AQii=AQ[idof*ndof+idof]*1E10;
        for(i=0; i<ndof; i++)
            AQ[idof*ndof+i]=0.0;
    }
    __syncthreads();
    if (n<nFD) {
        idof=(int)FDofs[n]-1;
        for(i=0; i<ndof; i++)
            AQ[i*ndof+idof]=0.0;
        AQ[idof*ndof+idof]=1e10;
    }

}

__global__ void formADc(double *AD, double *CX, int nelx, int nely, int nelz)
// assembly diagonal of stiffness matrix
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int i, j, k, d0;
    int nod, nel;
    int n1, m;
    int di, dj, dk;
    int iP, jP, kP;
    int nn[8]={6,5,2,1,7,4,3,0};
    
    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
    d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
    
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

void fprintV(const char *format)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format);
    fclose(flog);
}

void fprintI(const char *format, int v)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v);
    fclose(flog);
}

void fprintII(const char *format, int v1, int v2)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}


void fprintIII(const char *format, int v1, int v2, int v3)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2, v3);
    fclose(flog);
}

void fprintD(const char *format, double v)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v);
    fclose(flog);
}

void fprintDD(const char *format, double v1, double v2)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}

void fprintID(const char *format, int v1, double v2)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2);
    fclose(flog);
}

void fprintIID(const char *format, int v1, int v2, double v3)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2, v3);
    fclose(flog);
}

void fprintIDD(const char *format, int v1, double v2, double v3)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,format,v1, v2, v3);
    fclose(flog);
}

void fprintVc(const char *fname, int lev, double *V0, int n, int ncol)
{
    flog=fopen("mgcg9.log", "a");
    fprintf(flog,fname);
    fprintf(flog," lev=%d length=%d lines=%d\n", lev, n, (n+2)/ncol);
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

void printVc(const char *fname, double *V0, int n, int ncol)
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
    
    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
        d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
        
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

__global__ void assembAQc(double *CX, double *AQ, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int nod0, nod1, nod2, nel;
    int i, j, k, n, n1, n2, d0, d1, m;
    int di, dj, dk;
    int iP, jP, kP;
    int nn[8]={6,5,2,1,7,4,3,0};
    int ne[8]={0,0,0,0,0,0,0,0};
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];

    
    j=(blockIdx.z*blockDim.x+threadIdx.x)/3;
    if (j<=nely) {
    d0=(blockIdx.z*blockDim.x+threadIdx.x)%3;
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
                        AQ[(3*nod0+d0)*ndof+3*nod2+d1] += CX[nel]*Ae[m][n];
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

#define aq(i,j) aq[i*n+j]

void choldc1(int n, double aq[], double p[])
//Given a positive-definite symmetric matrix a[1..n][1..n], this routine constructs its Cholesky
//decomposition, A = L · LT . On input, only the upper triangle of a need be given; it is not
//modified. The Cholesky factor L is returned in the lower triangle of a, except for its diagonal
//elements which are returned in p[1..n].
{
    int i;
    int nBl, nThr;
    nBl=(n+maxThreads-1)/maxThreads;
    nThr=maxThreads;
    dim3 grids1(nBl,1,1);
    for (i=0;i<n;i++)
    {
        cholRow1<<<1,1>>>(aq,p,i,n);check();
        //nThr=(n-i<=maxThreads)? n-i : maxThreads;
        cholRow2<<<grids1,maxThreads>>>(aq,p,i,n);
        if(printLev>5) check();
    }
}

__global__ void cholRow1(double aq[], double p[], int i, int n)
{
    int k;
    double sum;
    sum=aq(i,i);
    for (k=i-1;k>=0;k--)
        sum -= aq(i,k)*aq(i,k);
    if (sum <= 0.0)                //a, with rounding errors, is not positive definite.
        assert(0);
       //printf("choldc failed %d %g\n",i,sum);
    p[i]=sqrt(sum);
}


__global__ void cholRow2(double aq[], double p[], int i, int n)
{
    int j,k;
    double sum;
    j=i+blockIdx.x*blockDim.x+threadIdx.x;
    if (j>i && j<n)
    {
        sum=aq(i,j);
        for (k=i-1;k>=0;k--)
            sum -= aq(i,k)*aq(j,k);
        aq(j,i)=sum/p[i];
    }
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
        cholslCol1<<<1,maxThreads>>>(n, i, aq, p, x);
        //for (k=i+1; k<n; k++)
        //    x[k] -= aq(k,i)*x[i];
    }
    
    for (i=n-1;i>=0;i--) {                                    //Solve LT · x = y.
        //x[i]=x[i]/p[i];
        cholslCol2<<<1,maxThreads>>>(n, i, aq, p, x);
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


__global__ void relaxRGBc(double *AD, double *U, double *F, double *CX, int nelx, int nely, int nelz, int ipar);

void relaxEnv(double *AD, double *U, double *F, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz)
// envelope for relaxation
{
    int nBlocksY = nelx/2+1;
    int nBlocksX = nelz/2+1;
    int nBlocksZ = ((nely/2+1)+maxThreads-1)/maxThreads;
    int ipar;
    dim3 grids(nBlocksX,nBlocksY,nBlocksZ);
    int nBlocksY1 = nelx+1;
    int nBlocksX1 = nelz+1;
    int nBlocksZ1 = (3*(nely+1)+maxThreads-1)/maxThreads;
    int nThreadsX=(3*(nely+1)<=maxThreads) ? 3*(nely+1) : maxThreads;
    int nThreadsY=1;
    dim3 grids1(nBlocksX1,nBlocksY1,nBlocksZ1);
    dim3 threads1(nThreadsX,nThreadsY);
    for (ipar=7; ipar>=0; ipar-- ) {
        relaxRGBc<<<grids,maxThreads>>>(AD, U, F, CX, nelx, nely, nelz, ipar);
        ZeroBCc<<<grids1,threads1>>>(U,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
    }
    defect0c<<<grids1,threads1>>>(F, U, R, CX, nelx, nely, nelz); if(printLev>3)check();
    ZeroBCc<<<grids1,threads1>>>(R,FDofs,nFD, nelx, nely, nelz); if(printLev>3)check();
    Jacobc<<<grids1,threads1>>>(AD, U, R, nelx, nely, nelz); if(printLev>3)check();
}

__global__ void relaxRGBc(double *AD, double *U, double *F, double *CX, int nelx, int nely, int nelz, int ipar)
{
    int nod, nod1, nod2, nel;
    int i, j, k, n, n1, n2, m, d0, d;
    int di, dj, dk;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nn[8]={6,5,2,1,7,4,3,0};
    int npar[8][3]={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,1,1},{1,0,1},{1,1,0},{1,1,1}};
    double R;

    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];


    k=blockIdx.x*2 + npar[ipar][2];
    i=blockIdx.y*2 + npar[ipar][0];
    j=(blockIdx.z*blockDim.x+threadIdx.x)*2 + npar[ipar][1];
    if (j<=nely) {
    for (d0=2; d0>=0; d0--)
    {
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        R=F[3*nod+d0];
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
                    //if (CX[nel]<1e-8) continue;
                    nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                    n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                    m=3*n1+d0;
                    for (n2=0; n2<8; n2++)
                    {
                        nod2=nod1+ne[n2];
                        for (d=0; d<3; ++d)
                        {
                            n=3*n2+d;
                            if (n2==n1 && d==d0) continue;
                            R -= CX[nel]*Ae[m][n]*U[3*nod2+d];
                        }
                    }
                }
            }
        }
        //U[3*nod+d0] = R/AD[3*nod+d0];
        U[3*nod+d0] = 0.52*U[3*nod+d0]+0.48*R/AD[3*nod+d0];
    }
    }
}


__global__ void defectXYc(double *AD, double *U, double *F, double *R, double *CX, int nelx, int nely, int nelz, int k);
__global__ void JacobXYc(double *AD, double *U, double *R, int nelx, int nely, int nelz, int k);

void relaxXYEnv(double *AD, double *U, double *F, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz)
// envelope for XY relaxation, plane k=0 is fixed
{
    int nBlocksX = nelx+1;
    int nBlocksY = (3*(nely+1)+maxThreads-1)/maxThreads;
    dim3 grids(nBlocksX,nBlocksY,1);
    for (int k=1; k<nelz+1; k++) {
        defectXYc<<<grids,maxThreads>>>(AD, U, F, R, CX, nelx, nely, nelz, k);
        JacobXYc<<<grids,maxThreads>>>(AD, U, R, nelx, nely, nelz, k);
    }
}

__global__ void defectXYc(double *AD, double *U, double *F, double *R, double *CX, int nelx, int nely, int nelz, int k)
{
    int nod, nod1, nod2, nel;
    int i, j, n, n1, n2, m, d0, d;
    int di, dj, dk;
    int iP, jP, kP;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nn[8]={6,5,2,1,7,4,3,0};

    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];

    i=blockIdx.x;
    j=(blockIdx.y*blockDim.x+threadIdx.x)/3;
    d0=(blockIdx.y*blockDim.x+threadIdx.x)%3;
    
    if (j<=nely) {
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
                    //if (CX[nel]<1e-8) continue;
                    nod1=(nelz+1)*(nely+1)*iP+(nely+1)*kP+jP;
                    n1=nn[4*(di+1)+2*(dk+1)+dj+1];
                    m=3*n1+d0;
                    for (n2=0; n2<8; n2++)
                    {
                        nod2=nod1+ne[n2];
                        for (d=0; d<3; ++d)
                        {
                            n=3*n2+d;
                            //if (n2==n1 && d==d0) continue;
                            if (n==m) continue;
                            R[3*nod+d0] -= CX[nel]*Ae[m][n]*U[3*nod2+d];
                        }
                    }
                }
            }
        }
    }
}

__global__ void JacobXYc(double *AD, double *U, double *R, int nelx, int nely, int nelz, int k)
{
    int nod;
    int i, j, d0;

    i=blockIdx.x;
    j=(blockIdx.y*blockDim.x+threadIdx.x)/3;
    d0=(blockIdx.y*blockDim.x+threadIdx.x)%3;
    
    if (j<=nely) {
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        U[3*nod+d0] = 0.52*U[3*nod+d0]+0.48*R[3*nod+d0]/AD[3*nod+d0];
    }
}
