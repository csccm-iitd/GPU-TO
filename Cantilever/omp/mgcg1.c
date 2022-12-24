#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

void mgcg1(double Ae[][24],double *F,double *U, double *CX, double *FDofs, double *W, int nFD,  int nelx, int nely, int nelz, int nl, double tol, int maxit, double *res, int *nit);
void prepAD0(double Ae[][24], double *AD, double *CX, int nelx, int nely, int nelz);
void prepAD(double A[][24][24], double *AD, int nelx, int nely, int nelz);
void initVcycle(double Ae[][24], double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0);
void vcycle(double Ae[][24], double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0);
void cg0(double Ae[][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void cg(double Ae[][24][24], double *F, double *U, double *R, double *P, double *Q, double *Z, double *FDofs, int nFD, int nelx, int nely, int nelz);
void restr(double *RP, double *CXP, double *FDofsP, double *F, double *CX, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD);
void rstrFD(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD);
void rstrF(double *RP, double *F, double *FDofs, int nelx, int nely, int nelz, int nFD);
void interp(double *UP, double *U, int nelx, int nely, int nelz);
void printV(char *fname, double *V, int n, int ncol);
void set0(double *V, int n);
void defect0(double Ae[][24],double *F,double *U, double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void defect(double A[][24][24],double *F,double *U, double *R, double *FDofs, int nFD, int nelx, int nely, int nelz);
void multAV0(double Ae[][24],double *U,double *R, double *CX, double *FDofs, int nFD, int nelx, int nely, int nelz);
void multAV(double A[][24][24],double *U,double *R, double *FDofs, int nFD, int nelx, int nely, int nelz);
void subVV(double *V1,double *V2,double *V3, int ndof);
double scalVV(double *V1, double *V2, int ndof);
void copyVV(double *V1, double *V2, int ndof);
void mulsV(double *V,double s, int ndof);
void addVV(double *V1,double *V2,double *V3, int ndof);
void addVsV(double *V1,double *V2,double *V3, double s, int ndof);
int readCX(double *CX);
void coarseAe(double Ae[][24],double A[][24][24], double *CXP, int nelx, int nely, int nelz);
void coarseAe3(double AP[][24][24],double A[][24][24], int nelx, int nely, int nelz);
void coarseAe2(double A[][24][24], double *CXP, double v, double hx,double hy,double hz, int nelx, int nely, int nelz);
void coarseAe2L(double A[][24][24], double *CX0, int lev, double v, double hx,double hy,double hz, int nelx, int nely, int nelz);
void CorrCsA(double A[][24][24], double *FDofs, int nFD, int nelx, int nely, int nelz);
void assembA(int nA, double Ae[][24][24], double A[][nA], int nelx, int nely, int nelz);
void printCsAe(double A[][24][24], int nelx, int nely, int nelz);
void CmpCsAe(double A[][24][24],double A1[][24][24], int nelx, int nely, int nelz);
void Kem3(double Ae[][24],double v,double hx,double hy,double hz);
void defD(double D[6][6],double v);
void Bem3(double B[6][24],double s,double e,double t,double hx,double hy,double hz);
void choldc(int n, double a[][n], double p[]);
void cholsl(int n, double a[][n], double p[], double b[], double x[]);
int estWorkSpace(int nFD0, int nl, int nelx0, int nely0, int nelz0);

static int printLev=1;
FILE *flog;

/*-----------------------------------------------------------------------*/
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *Ae, *F, *U, *CX, *FDofs, *W;
    double dnelx, dnely, dnelz, dnl, dnswp, tol, dmaxit, res;
    int nelx, nely, nelz, nl, nswp=2, maxit;
    int nit;
    
    Ae = mxGetDoubles(prhs[0]);
    int mrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);
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
    W = mxGetDoubles(prhs[5]);
    int mrowsW = mxGetM(prhs[5]);
    int ncolsW = mxGetN(prhs[5]);
    //mexPrintf("\n%d %d WRK dimensions.\n", mrowsW, ncolsW);
    //int nFD = mxGetN(prhs[4]);
    int nFD = (int) mxGetScalar(prhs[6]); 
    //mexPrintf("\n%d nFD\n", nFD);
    nelx = (int) mxGetScalar(prhs[7]); 
    nely = (int) mxGetScalar(prhs[8]); 
    nelz = (int) mxGetScalar(prhs[9]); 
    nl   = (int) mxGetScalar(prhs[10]); 
    tol  =       mxGetScalar(prhs[11]); 
    maxit= (int) mxGetScalar(prhs[12]); 
    int lenW=estWorkSpace(nFD, nl, nelx, nely, nelz);
    if (lenW>mrowsW)
    {
        mexPrintf("\nMGCG - %d is not enough working space! %d is required.\n", mrowsW,lenW);
        plhs[0] = mxCreateDoubleScalar(0.0);
        plhs[1] = mxCreateDoubleScalar(-1.0);
    }
    else
    {
        //printV("mgcg1F",F, nelx, nely, nelz);
        mgcg1(Ae, F, U, CX, FDofs, W, nFD, nelx, nely, nelz, nl, tol, maxit, &res, &nit);
        plhs[0] = mxCreateDoubleScalar(res);
        plhs[1] = mxCreateDoubleScalar((double)nit);
    }
}

void mgcg1(double Ae[][24],double *F,double *U, double *CX, double *FDofs, double *W, int nFD, int nelx, int nely, int nelz, int nl, double tol, int maxit, double *res, int *nit)
{
    flog=fopen("mgcg1.log", "w");
    int nswp=3;
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    double *R=&W[0];
    double *P=&W[ndof];
    double *Q=&W[2*ndof];
    double *Z=&W[3*ndof];
    double *W1=&W[4*ndof];
    int n, itcg;
    double rnorm, res0, rho, rho_p, beta, dpr, alpha, relres;
    int threadsNum = 16;
    omp_set_num_threads(threadsNum);
    
    set0(U, ndof);
    copyVV(F,R,ndof);
    res0=sqrt(scalVV(F,F,ndof));
    initVcycle(Ae, CX, FDofs, W1, nFD, nl, nelx, nely, nelz);
    //printV("mgcgR",R, ndof,3);
    //copyVV(U,P,ndof);
    vcycle(Ae, R, P, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
    if (printLev>1)
    {
        defect0(Ae, R, P, Z, CX, FDofs, nFD, nelx, nely, nelz);
        fprintf(flog,"CG  res0=%12.6g\n",res0);
        fprintf(flog,"CG  res1=%12.6g\n",sqrt(scalVV(Z,Z,ndof)));
    }
    //printV("mgcgP",P, ndof,3);
    //printV("mgcFD",FDofs, nFD,3);
    //vcycle(Ae, F, U, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
    //fprintf(flog,"Time %8.3g  resf=%12.6g\n",0.0,res0);
    copyVV(P,Z,ndof);
    rho=scalVV(R,Z,ndof);
    rho_p=rho;
    for (itcg = 0; itcg < maxit; ++itcg)
    {
        clock_t t0=clock();
        multAV0(Ae, P, Q, CX, FDofs, nFD, nelx, nely, nelz);
        //printV("mgcgQ",Q, ndof,3);
        alpha=rho_p/scalVV(P,Q,ndof);
        addVsV(U,P,U,alpha,ndof);
        addVsV(R,Q,R,-alpha,ndof);
        //set0(Z, ndof);
        vcycle(Ae, R, Z, CX, FDofs, W1, nFD, nl, nswp, nelx, nely, nelz);
        
        rho=scalVV(R,Z,ndof);
        beta = rho/rho_p;
        addVsV(Z,P,P,beta,ndof);
        rho_p = rho;
        *res = sqrt(scalVV(R,R,ndof));
        relres = *res/res0;
        clock_t t1=clock();
        if (printLev>2)
            fprintf(flog,"CG It %d Time %8.3g  %12.6g\n",itcg, (double)(t1-t0)/CLOCKS_PER_SEC,relres);
        if (relres < tol || itcg>=maxit)
            break;
    }
    //multAV0(Ae, P, Q, CX, FDofs, nFD, nelx, nely, nelz);
    //alpha=rho_p/scalVV(P,Q,ndof);
    //addVsV(U,P,U,alpha,ndof);
    //defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
    //*res = sqrt(scalVV(R,R,ndof))/res0;
    for (n = 1; n < ndof; n +=3)
        U[n]=-U[n];

    *res=relres;
    *nit=itcg;
    if (printLev>1) 
          fprintf(flog,"CG Nit=%d res=%12.6g\n",itcg,relres);
    fclose(flog);
}

int estWorkSpace(int nFD0, int nl, int nelx0, int nely0, int nelz0)
{
    int n, nnod, ndof, nelx, nely, nelz;
    int nFD, lW;
    int ndof0=3*(nelx0+1)*(nely0+1)*(nelz0+1);
    ndof=ndof0;
    nFD=nFD0;
        
    lW=ndof0*4;   
    for (int lev = 0; lev <nl; ++lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        if (lev==0)
            lW += 2*ndof;
        else
            lW += (24*24+4)*ndof+nFD;
        if (lev==nl-1)
            lW += 3*ndof+ndof*ndof;
    }
    return lW;
}

static int levW[20], lnFD[20];

void initVcycle(double Ae[][24], double *CX0, double *FDofs0, double *W, int nFD0, int nl, int nelx0, int nely0, int nelz0)
{
    int n, nnod, ndof, nelx, nely, nelz;
    double *U, *F, *R, *CX, *A, *AP, *AD, *AQ, *FDofs, *UP, *RP, *CXP, *FDofsP, *BC;
    double *P, *Q, *Z, *WRK;
    double nu=0.3, hx0=1, hy0=1, hz0=1;
    double hx, hy, hz;
    int nFD, lW;
    int ndof0=3*(nelx0+1)*(nely0+1)*(nelz0+1);
    FDofs=FDofs0;
    nFD=nFD0;
    lnFD[0]=nFD0;
        
    lW=0;   
    for (int lev = 0; lev <nl; ++lev)
    {
        levW[lev]=lW;
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        hx=hx0*pow(2,lev);
        hy=hy0*pow(2,lev);
        hz=hz0*pow(2,lev);
        if (lev==0)
        {
            CX=CX0;
            FDofs=FDofs0;
            nFD=nFD0;
            R=&W[lW]; lW += ndof;
            AD=&W[lW]; lW += ndof;
            prepAD0(Ae, AD, CX, nelx, nely, nelz);
        }
        else
        {
            FDofsP=FDofs;
            U=&W[lW];  lW += ndof;
            F=&W[lW];  lW += ndof;
            R=&W[lW];  lW += ndof;
            AP=A;
            A=&W[lW];  lW += 24*24*ndof;
            AD=&W[lW]; lW += ndof;
            //fprintf(flog,"lev %d CX lW %d\n",lev,lW);
            FDofs=&W[lW];
            int nFDP=lnFD[lev-1];
            rstrFD(FDofsP,FDofs, nelx, nely, nelz, nFDP, &nFD);
            lW += nFD;
            lnFD[lev]=nFD;
            if (lev==1)
                coarseAe2(A, CX, nu, hx, hy, hz, nelx, nely, nelz);
            else
                //coarseAe2L(A, CX0, lev, nu, hx, hy, hz, nelx, nely, nelz);
                coarseAe3(AP, A, nelx, nely, nelz);
            if (lev==nl-1)
            {
                P=&W[lW]; lW += ndof;
                Q=&W[lW]; lW += ndof;
                Z=&W[lW]; lW += ndof;
                AQ=&W[lW]; lW += ndof*ndof;
                CorrCsA(A, FDofs, nFD, nelx, nely, nelz);
                //printCsAe(A, nelx, nely, nelz);
                assembA(ndof, A, AQ, nelx, nely, nelz);
                //printV("r6aAQ",AQ, ndof*ndof,ndof);
                
                choldc(ndof, AQ,P);
            }
            prepAD(A, AD, nelx, nely, nelz);
        }
    }
    
    if (printLev>0) fprintf(flog,"Number of dof %d  Working space %d\n",3*(nelx0+1)*(nely0+1)*(nelz0+1),lW+4*3*(nelx0+1)*(nely0+1)*(nelz0+1));
}

void vcycle(double Ae[][24], double *F0, double *U0, double *CX0, double *FDofs0, double *W, int nFD0, int nl,int nswp, int nelx0, int nely0, int nelz0)
{
    int n, nnod, ndof, nelx, nely, nelz;
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
    
    if (printLev>2)
    {
        defect0(Ae, F, U, R, CX, FDofs, nFD, nelx0, nely0, nelz0);
        fprintf(flog,"lev %d Initial %12.6g\n",0,sqrt(scalVV(R,R,3*(nelx0+1)*(nely0+1)*(nelz0+1))));
    }
    
    #pragma omp parallel for shared(U, F, AD, ndof) private(n)
    for (n = 0; n < ndof0; ++n)
        U[n]=0.7*F[n]/AD[n];
    //printV("mgcgU",U, ndof,3);
       
          
    for (int ivc = 0; ivc <1; ++ivc)
    {
    for (int lev = 0; lev <nl; ++lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
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
            A=&W[lW];  lW += 24*24*ndof;
            AD=&W[lW]; lW += ndof;
            FDofs=&W[lW];
            nFD=lnFD[lev];
            rstrF(RP,F,FDofs, nelx, nely, nelz, nFD);
            lW += nFD;
        }

        if (lev<nl-1)
        {
            if (lev>0)
                #pragma omp parallel for shared(U, F, AD, ndof) private(n)
                for (n = 0; n < ndof; ++n)
                    U[n]=0.7*F[n]/AD[n];
            for (int it = 0; it < nswp; ++it)
            {
                if (lev==0) defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
                else defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
                if (printLev>3)
                    fprintf(flog,"lev %d Bef rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));

                #pragma omp parallel for shared(U, R, AD, ndof) private(n)
                for (n = 0; n < ndof; ++n)
                    U[n]=0.52*U[n]+0.48*(R[n]+AD[n]*U[n])/AD[n];
                    //U[n]=U[n]+0.6*R[n]/AD[n];
            }
            if (lev==0) defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
            else defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
            if (printLev>3)
                fprintf(flog,"lev %d Aft rel1 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
        }
        else
        {
            P=&W[lW]; lW += ndof;
            Q=&W[lW]; lW += ndof;
            Z=&W[lW]; lW += ndof;
            AQ=&W[lW]; lW += ndof*ndof;
            if (lev==0) cg0(Ae, F, U, R, P, Q, Z, CX, FDofs, nFD, nelx, nely, nelz);
            else cholsl(ndof,AQ,P,F,U);
            for (n = 0; n < nFD; ++n)
                U[(int)FDofs[n]-1]=0.0;
            //else cg(A, F, U, R, P, Q, Z, FDofs, nFD, nelx, nely, nelz);
            if (printLev>3)
            {
                if (lev==0) defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
                else defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
                fprintf(flog,"lev %d Aft DirSol %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
        }
    }
    
    for (int lev = nl-2; lev >=0; --lev)
    {
        nelx=nelx0/pow(2,lev);
        nely=nely0/pow(2,lev);
        nelz=nelz0/pow(2,lev);
        nnod=(nelx+1)*(nely+1)*(nelz+1);
        ndof=3*nnod;
        UP=U;
        nFD=lnFD[lev];
        lW=levW[lev];
        if (lev>0)
        {
            U=&W[lW]; lW += ndof;
            F=&W[lW]; lW += ndof;
            R=&W[lW]; lW += ndof;
            A=&W[lW];  lW += 24*24*ndof;
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
        
        interp(UP,U, nelx, nely, nelz);
        for (int it = 0; it < nswp; ++it)
        {
            if (lev==0) defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
            else defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
            if (printLev>3)
            {
                //if (lev == nl-2) printV("r5bR",R, ndof,3);
                fprintf(flog,"lev %d Bef rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
            #pragma omp parallel for shared(U, R, AD, ndof) private(n)
            for (n = 0; n < ndof; ++n)
                U[n]=0.52*U[n]+0.48*(R[n]+AD[n]*U[n])/AD[n];
                //U[n]=U[n]+0.6*R[n]/AD[n];
            if (printLev>3)
            {
                if (lev==0) defect0(Ae, F, U, R, CX, FDofs, nFD, nelx, nely, nelz);
                else defect(A, F, U, R, FDofs, nFD, nelx, nely, nelz);
                fprintf(flog,"lev %d Aft rel2 %12.6g\n",lev,sqrt(scalVV(R,R,ndof)));
            }
        }
        
    }
    if (printLev>2) fprintf(flog,"\n");
    }

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
    //printV("3gF4",F,ndof,3);
    //printV("3gCX4",CX,nelx*nely*nelz,3);
    res0=sqrt(scalVV(F,F,ndof));
    if (printLev>3) fprintf(flog,"CG Coarse level  Res0 %12.6g\n",res0);
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
        //fprintf(flog,"%12.6g\n",res);
        relres = res/res0;
        if (relres < tol || itcg>=maxit)
            break;
    }
    clock_t t1=clock();
    if (printLev>3) fprintf(flog,"CG It %d Time %12.6g   Res %12.6g\n",itcg,(double)(t1-t0)/CLOCKS_PER_SEC,res);
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
    //printV("3gF4",F,ndof,3);
    //printV("3gCX4",CX,nelx*nely*nelz,3);
    res0=sqrt(scalVV(F,F,ndof));
    if (printLev>3) fprintf(flog,"CG Coarse level  Res0 %12.6g\n",res0);
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
        //fprintf(flog,"%12.6g\n",res);
        relres = res/res0;
        if (relres < tol || itcg>=maxit)
            break;
    }
    clock_t t1=clock();
    if (printLev>3) fprintf(flog,"CG It %d Time %12.6g   Res %12.6g\n",itcg,(double)(t1-t0)/CLOCKS_PER_SEC,res);
}

void rstrFD(double *FDofsP, double *FDofs, int nelx, int nely, int nelz, int nFDP, int *nFD)
{
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
    for (n = 0; n < nFDP; n++)
    {
        idof=((int)FDofsP[n]-1)%3;
        nodP=((int)FDofsP[n]-1)/3;
        iP=nodP/((nelzP+1)*(nelyP+1));
        if (iP%2>0) continue;
        kP=(nodP-(nelzP+1)*(nelyP+1)*iP)/(nelyP+1);
        if (kP%2>0) continue;
        jP=nodP-(nelzP+1)*(nelyP+1)*iP-(nelyP+1)*kP;
        if (jP%2>0) continue;
        nod=(nelz+1)*(nely+1)*iP/2+(nely+1)*kP/2+jP/2; 
        //fprintf(flog,"%d  %d  %d  %d\n",(int)FDofsP[n],iP,kP,jP);
        FDofs[iFD]=3*nod+idof+1;
        iFD++;
    }
    *nFD=iFD;
    //printV("mgcgFD0",FDofs,*nFD,3);
}

void rstrF(double *RP, double *F, double *FDofs, int nelx, int nely, int nelz, int nFD)
{
    int i, j, k, di, dj, dk, n, d, nod, nodP;
    int iP, jP, kP;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double weight;
    
    set0(F,3*(nelz+1)*(nely+1)*(nelx+1));
    #pragma omp parallel for shared(F, RP, nelx, nely, nelz, nelxP, nelyP, nelzP) private(nod, nodP, i, j, k, d, weight, di, dj, dk, iP, jP, kP)
    for (k = 0; k < nelz+1; k++)
    for (j = 0; j < nely+1; j++)
    for (i = 0; i < nelx+1; i++)
    {
        nod=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
       
        for (dk = -1; dk <= 1; dk++)
        {
            kP=2*k+dk;
            if (kP<0 || kP>nelzP) continue;
            for (dj = -1; dj <= 1; dj++)
            {
                jP=2*j+dj;
                if (jP<0 || jP>nelyP) continue;
                for (di = -1; di <= 1; di++)
                {
                    iP=2*i+di;
                    if (iP<0 || iP>nelxP) continue;
                    weight=1.0/pow(2,abs(di)+abs(dj)+abs(dk));
                    nodP=(nelzP+1)*(nelyP+1)*iP+(nelyP+1)*kP+jP;
                    for (d = 0; d < 3; d++)
                    F[3*nod+d]=F[3*nod+d]+weight*RP[3*nodP+d];
                }
            }
        }

    }
    
    for (n = 0; n < nFD; ++n)
        F[(int)FDofs[n]-1]=0.0;

    //printV("3cFD0",FDofsP,nFDP,3);
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

void set0(double *V, int n)
{
    int i;
    #pragma omp parallel for shared(V, n) private(i)
    for(i=0; i<n; i++)
        V[i]=0.0;
}

void mulsV(double *V,double s, int ndof)
{
    int n;
    #pragma omp parallel for shared(V, s, ndof) private(n)
    for (n = 0; n < ndof; ++n)
    V[n]=V[n]*s;
}

void addVV(double *V1,double *V2,double *V3, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, V3, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V3[n]=V1[n]+V2[n];
}

void addVsV(double *V1,double *V2,double *V3, double s, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, V3, s, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V3[n]=V1[n]+s*V2[n];
}

void subVV(double *V1,double *V2,double *V3, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, V3, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V3[n]=V1[n]-V2[n];
}

double scalVV(double *V1, double *V2, int ndof)
{
    int n;
    double rnorm=0.0;
    #pragma omp parallel for shared(V1, V2, ndof) private(n)  reduction (+: rnorm)
    for (n = 0; n < ndof; ++n)
        rnorm +=V1[n]*V2[n];
    return rnorm;    
}

void copyVV(double *V1, double *V2, int ndof)
{
    int n;
    #pragma omp parallel for shared(V1, V2, ndof) private(n)
    for (n = 0; n < ndof; ++n)
        V2[n]=V1[n];
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

void prepAD0(double Ae[][24], double *AD, double *CX, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i0, j0, k0;
    int i, j, k, n;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
        
    set0(AD, ndof);
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(AD, Ae, CX, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, i, j, k, n)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n = 0; n < 8; ++n)
        {
            nod1=nod0+ne[n];
            AD[3*nod1]  =AD[3*nod1]  +CX[nel]*Ae[3*n][3*n];
            AD[3*nod1+1]=AD[3*nod1+1]+CX[nel]*Ae[3*n+1][3*n+1];
            AD[3*nod1+2]=AD[3*nod1+2]+CX[nel]*Ae[3*n+2][3*n+2];
        }
    }
    }
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
        //fprintf(flog,"dof %d  idof=%d  nod=%d  i,j,k %d %d %d\n",(int)FDofs[n]-1,idof, nod, i,j,k);
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
            //fprintf(flog,"nel=%d  n1=%d  iP,jP,kP %d %d %d\n",nel, n1, iP,jP,kP);
        }
    }
}

void prepAD(double A[][24][24], double *AD, int nelx, int nely, int nelz)
{
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2, nel;
    int i0, j0, k0;
    int i, j, k, n;
    
    for (n = 0; n < 8; ++n)
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
        
    set0(AD, ndof);
    for (k0 = 0; k0 < 2; ++k0)
    for (j0 = 0; j0 < 2; ++j0)
    for (i0 = 0; i0 < 2; ++i0)
    {
    #pragma omp parallel for shared(AD, A, ne, nelx, nely, nelz, k0, j0, i0) private(nel, nod0, nod1, i, j, k, n)
    for (k = k0; k < nelz; k +=2)
    for (j = j0; j < nely; j +=2)
    for (i = i0; i < nelx; i +=2)
    {
        nel=nelz*nely*i+nely*k+j;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n = 0; n < 8; ++n)
        {
            nod1=nod0+ne[n];
            AD[3*nod1]   += A[nel][3*n][3*n];
            AD[3*nod1+1] += A[nel][3*n+1][3*n+1];
            AD[3*nod1+2] += A[nel][3*n+2][3*n+2];
        }
    }
    }
}

void printV(char *fname, double *V, int n, int ncol)
{
    FILE *fp;
    fp=fopen(fname, "w");
    fprintf(fp,fname);
    fprintf(fp," length=%d lines=%d\n",n,(n+2)/ncol);
    for (int i = 0; i < n; ++i)
    {
        fprintf(fp,"%11.4g",V[i]);
        if ((i+1)%ncol==0)
            fprintf(fp,"\n");
    }
    //fprintf(fp,"\n");
    fprintf(fp,"\n");
    fclose(fp);
}

void interp(double *UP, double *U, int nelx, int nely, int nelz)
{
    int i, j, k, d, n, nod, nodP;
    int di, dj, dk, i0, j0, k0, i1, j1, k1;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    int ke[8]={0,0,0,0,1,1,1,1};
    double weight;
    int nelxP=nelx/2;
    int nelyP=nely/2;
    int nelzP=nelz/2;
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;

    //for (k = 0; k < nelzP+1; k++)
    //for (j = 0; j < nelyP+1; j++)
    //for (i = 0; i < nelxP+1; i++)
    //{
    //    nodP=(nelzP+1)*(nelyP+1)*i+(nelyP+1)*k+j;
    //    UP[3*nodP]=i;
    //    UP[3*nodP+1]=i;
    //    UP[3*nodP+2]=i;
    //}
    //set0(U,ndof);
    
    for (k1 = 0; k1 < 2; ++k1)
    for (j1 = 0; j1 < 2; ++j1)
    for (i1 = 0; i1 < 2; ++i1)
    {
    #pragma omp parallel for shared(U, UP, nelx, nely, nelz, nelxP, nelyP, nelzP,k1,j1,i1) private(weight,nod,nodP,i,j,k,i0,j0,k0,di,dj,dk,d)
    for (k = k1; k < nelzP+1; k +=2)
    for (j = j1; j < nelyP+1; j +=2)
    for (i = i1; i < nelxP+1; i +=2)
    {
        nodP=(nelzP+1)*(nelyP+1)*i+(nelyP+1)*k+j;
        for (di = -1; di <= 1; di++)
        {
           i0=2*i+di;
           if (i0<0 || i0>nelx) continue;
            for (dk = -1; dk <= 1; dk++)
            {
               k0=2*k+dk;
               if (k0<0 || k0>nelz) continue;
                for (dj = -1; dj <= 1; dj++)
                {
                   j0=2*j+dj;
                   if (j0<0 || j0>nely) continue;
                   nod=(nelz+1)*(nely+1)*i0+(nely+1)*k0+j0;
                   weight=1.0/pow(2,abs(di)+abs(dj)+abs(dk));
                   for (d = 0; d < 3; d++)
                      U[3*nod+d]=U[3*nod+d]+weight*UP[3*nodP+d];
                }
            }
        }
    }
    }
    //printV("3fU",U,3*(nelz+1)*(nely+1)*(nelx+1),3);
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

void coarseAe(double Ae[][24],double A[][24][24], double *CXP, int nelx, int nely, int nelz)
{
    double AL[81][81];
    double ALP[81][24];
    
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2;
    int i, j, k, n, n1, n2, d1, d2;
    int iP, jP, kP, nel, nelP, iA, jA, l;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double P1[8][27]={{1,0.5,0, 0.5,0.25,0, 0,0,0,   0.5,0.25,0, 0.25,0.125,0, 0,0,0,      0,0,0,   0,0,0,      0,0,0},
                     {0,0,0,   0,0,0,      0,0,0,   0.5,0.25,0, 0.25,0.125,0, 0,0,0,      1,0.5,0, 0.5,0.25,0, 0,0,0},
                     {0,0,0,   0,0,0,      0,0,0,   0,0.25,0.5, 0,0.125,0.25, 0,0,0,      0,0.5,1, 0,0.25,0.5, 0,0,0},
                     {0,0.5,1, 0,0.25,0.5, 0,0,0,   0,0.25,0.5, 0,0.125,0.25, 0,0,0,      0,0,0,   0,0,0,      0,0,0},
                     {0,0,0,   0.5,0.25,0, 1,0.5,0, 0,0,0,      0.25,0.125,0, 0.5,0.25,0, 0,0,0,   0,0,0,      0,0,0},        
                     {0,0,0,   0,0,0,      0,0,0,   0,0,0,      0.25,0.125,0, 0.5,0.25,0, 0,0,0,   0.5,0.25,0, 1,0.5,0},
                     {0,0,0,   0,0,0,      0,0,0,   0,0,0,      0,0.125,0.25, 0,0.25,0.5, 0,0,0,   0,0.25,0.5, 0,0.5,1},
                     {0,0,0,   0,0.25,0.5, 0,0.5,1, 0,0,0,      0,0.125,0.25, 0,0.25,0.5, 0,0,0,   0,0,0,      0,0,0}  };
    double P[24][81];
    
    for (n = 0; n < 8; ++n)
        ne[n]=9*ie[n]+3*ke[n]+je[n];
        
    set0(P,24*81);
    for (jA = 0; jA < 27; jA++)
    for (iA = 0; iA < 8; iA++)
    {
        P[iA*3][jA*3]=P1[iA][jA];
        P[iA*3+1][jA*3+1]=P1[iA][jA];
        P[iA*3+2][jA*3+2]=P1[iA][jA];
    }
    
    #pragma omp parallel for shared(Ae,A,CXP, nelx,nely,nelz,nelxP,nelyP,nelzP) private(AL,ALP,nod0,nod1,nod2,nel,nelP,i,j,k,iP,jP,kP,iA,jA,l,n1,n2,d1,d2)
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
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
                nod1=nod0+ne[n1];
                for (n2 = 0; n2 < 8; ++n2)
                {
                    nod2=nod0+ne[n2];
                    for (d1=0; d1<3; ++d1)
                    for (d2=0; d2<3; ++d2)
                    {
                        AL[3*nod1+d1][3*nod2+d2]  += CXP[nelP]*Ae[3*n1+d1][3*n2+d2];
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
        for (jA = 0; jA < 24; jA++)
        for (iA = 0; iA < 24; iA++)
        {
            A[nel][iA][jA]=0.0;
            for (l = 0; l < 81; l++)
                A[nel][iA][jA] +=ALP[l][jA]*P[iA][l];
        }
    }
}

void coarseAe3(double AP[][24][24],double A[][24][24], int nelx, int nely, int nelz)
{
    double AL[81][81];
    double ALP[81][24];
    
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int nod0, nod1, nod2;
    int i, j, k, n, n1, n2, d1, d2;
    int iP, jP, kP, nel, nelP, iA, jA, l;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double P1[8][27]={{1,0.5,0, 0.5,0.25,0, 0,0,0,   0.5,0.25,0, 0.25,0.125,0, 0,0,0,      0,0,0,   0,0,0,      0,0,0},
                     {0,0,0,   0,0,0,      0,0,0,   0.5,0.25,0, 0.25,0.125,0, 0,0,0,      1,0.5,0, 0.5,0.25,0, 0,0,0},
                     {0,0,0,   0,0,0,      0,0,0,   0,0.25,0.5, 0,0.125,0.25, 0,0,0,      0,0.5,1, 0,0.25,0.5, 0,0,0},
                     {0,0.5,1, 0,0.25,0.5, 0,0,0,   0,0.25,0.5, 0,0.125,0.25, 0,0,0,      0,0,0,   0,0,0,      0,0,0},
                     {0,0,0,   0.5,0.25,0, 1,0.5,0, 0,0,0,      0.25,0.125,0, 0.5,0.25,0, 0,0,0,   0,0,0,      0,0,0},        
                     {0,0,0,   0,0,0,      0,0,0,   0,0,0,      0.25,0.125,0, 0.5,0.25,0, 0,0,0,   0.5,0.25,0, 1,0.5,0},
                     {0,0,0,   0,0,0,      0,0,0,   0,0,0,      0,0.125,0.25, 0,0.25,0.5, 0,0,0,   0,0.25,0.5, 0,0.5,1},
                     {0,0,0,   0,0.25,0.5, 0,0.5,1, 0,0,0,      0,0.125,0.25, 0,0.25,0.5, 0,0,0,   0,0,0,      0,0,0}  };
    double P[24][81];
    
    for (n = 0; n < 8; ++n)
        ne[n]=9*ie[n]+3*ke[n]+je[n];
        
    set0(P,24*81);
    for (jA = 0; jA < 27; jA++)
    for (iA = 0; iA < 8; iA++)
    {
        P[iA*3][jA*3]=P1[iA][jA];
        P[iA*3+1][jA*3+1]=P1[iA][jA];
        P[iA*3+2][jA*3+2]=P1[iA][jA];
    }
    
    #pragma omp parallel for shared(A,AP,nelx,nely,nelz,nelxP,nelyP,nelzP) private(AL,ALP,nod0,nod1,nod2,nel,nelP,i,j,k,iP,jP,kP,iA,jA,l,n1,n2,d1,d2)
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
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
                nod1=nod0+ne[n1];
                for (n2 = 0; n2 < 8; ++n2)
                {
                    nod2=nod0+ne[n2];
                    for (d1=0; d1<3; ++d1)
                    for (d2=0; d2<3; ++d2)
                    {
                        AL[3*nod1+d1][3*nod2+d2]  += AP[nelP][3*n1+d1][3*n2+d2];
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
        for (jA = 0; jA < 24; jA++)
        for (iA = 0; iA < 24; iA++)
        {
            A[nel][iA][jA]=0.0;
            for (l = 0; l < 81; l++)
                A[nel][iA][jA] +=ALP[l][jA]*P[iA][l];
        }
    }
}
void assembA(int nA, double Ae[][24][24], double A[][nA], int nelx, int nely, int nelz)
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
    double rnorm;
    
    for (n = 0; n < 8; ++n)
        //ne[n]=(nelx+1)*(nely+1)*ke[n]+(nely+1)*ie[n]+je[n];
        //ne[n]=(nelx+1)*(nely+1)*ke[n]+(nelx+1)*je[n]+ie[n];
        ne[n]=(nelz+1)*(nely+1)*ie[n]+(nely+1)*ke[n]+je[n];
    
    //#pragma omp parallel for shared(R, U, Ae) private(nod0, nod1, nod2, i, j, k, n1, n2, d)
    for (k = 0; k < nelz; ++k)
    for (j = 0; j < nely; ++j)
    for (i = 0; i < nelx; ++i)
    {
        nel=nelz*nely*i+nely*k+j;
        //nod0=(nelx+1)*(nely+1)*k+(nely+1)*i+j;
        //nod0=(nelx+1)*(nely+1)*k+(nelx+1)*j+i;
        nod0=(nelz+1)*(nely+1)*i+(nely+1)*k+j;
        for (n1 = 0; n1 < 8; ++n1)
        {
            nod1=nod0+ne[n1];
            for (n2 = 0; n2 < 8; ++n2)
            {
                nod2=nod0+ne[n2];
                for (int d1=0; d1<3; ++d1)
                for (int d2=0; d2<3; ++d2)
                {
                    A[3*nod1+d1][3*nod2+d2]  += Ae[nel][3*n1+d1][3*n2+d2];
                }
            }
        }
    }
    
    //FILE *fp;
    //fp=fopen("AQ", "w");
    //for (int i = 0; i < nA; ++i)
    //{
    //    for (int j = 0; j < nA; ++j)
    //        fprintf(fp,"%9.5g,",A[i][j]);
    //    fprintf(fp,"\n");
    //}
    //fclose(fp);
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

// 3d element stiffness matrix
void Kem3(double Ae[][24],double v,double hx,double hy,double hz)
{
    double D[6][6]={0};
    double B[6][24]={0};
    double DB[6][24]={0};

    //double integP[2]={-0.5, 0.5};
    double integP[2]={-1/sqrt(3), 1/sqrt(3)};
    double s, e, t;
    
    defD(D,v);
    set0(Ae,24*24);
    //set0(DB,24*6);
    for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
    for (int k=0; k<2; k++)
    {
        s=integP[i];
        e=integP[j];
        t=integP[k];
        Bem3(B,s,e,t,hx,hy,hz);
        for (int m=0; m<6; m++)
        for (int n=0; n<24; n++)
        {
            DB[m][n]=0.0;
            for (int l=0; l<6; l++)
                DB[m][n] +=D[m][l]*B[l][n];
        }
                
        for (int m=0; m<24; m++)
        for (int n=0; n<24; n++)
        {
            //Ae[m][n]=0.0;
            for (int l=0; l<6; l++)
                Ae[m][n] +=0.125*hx*hy*hz*B[l][m]*DB[l][n];
        }
    }
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

void coarseAe2(double A[][24][24], double *CXP, double v, double hx,double hy,double hz, int nelx, int nely, int nelz)
{
    double Ael[24][24][8];
    
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int i, j, k;
    int iP, jP, kP, nel, nelP;
    int nelxP=nelx*2;
    int nelyP=nely*2;
    int nelzP=nelz*2;
    double D[6][6]={0};
    double B[6][24]={0};
    double DB[6][24]={0};
    double integP[2]={-1/sqrt(3), 1/sqrt(3)};
    double s, e, t;
    int m, n, l, na;
    
    defD(D,v);
    set0(Ael,24*24*8);
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
        for (int n=0; n<24; n++)
        {
            for (int l=0; l<6; l++)
                Ael[m][n][na] +=0.125*hx*hy*hz*B[l][m]*DB[l][n];
        }
    }
    
    #pragma omp parallel for shared(Ael,A,CXP, nelx,nely,nelz,nelxP,nelyP,nelzP) private(nel,nelP,i,j,k,iP,jP,kP,m,n,na)
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
        nel=nelz*nely*i+nely*k+j;
        for (m=0; m<24; m++)
        for (n=0; n<24; n++)
            A[nel][m][n] = 0.0;
             
        for (iP = 0; iP < 2; ++iP)
        for (kP = 0; kP < 2; ++kP)
        for (jP = 0; jP < 2; ++jP)
        {
            na=4*iP+2*kP+jP;
            nelP=nelzP*nelyP*(2*i+iP)+nelyP*(2*k+kP)+2*j+jP;
            for (m=0; m<24; m++)
            for (n=0; n<24; n++)
                A[nel][m][n] +=CXP[nelP]*Ael[m][n][na];
        }
    }
}

void coarseAe2L(double A[][24][24], double *CX0, int lev, double v, double hx,double hy,double hz, int nelx, int nely, int nelz)
{
    double static Ael[24][24][4096];
    
    int nnod=(nelx+1)*(nely+1)*(nelz+1);
    int ndof=3*nnod;
    int ie[8]={0,1,1,0,0,1,1,0};
    int je[8]={0,0,1,1,0,0,1,1};
    //int je[8]={1,1,0,0,1,1,0,0};    // reverse order numeration over y-dir
    int ke[8]={0,0,0,0,1,1,1,1};
    int ne[8]={0,0,0,0,0,0,0,0};
    int i, j, k;
    int iP, jP, kP, nel, nelP;
    int mult;
    mult=pow(2,lev);
    printf("mult %d\n",mult);
    int nelxP=nelx*mult;
    int nelyP=nely*mult;
    int nelzP=nelz*mult;
    double D[6][6]={0};
    double B[6][24]={0};
    double DB[6][24]={0};
    //double integP[2]={-1/sqrt(3), 1/sqrt(3)};
    double integP[128];
    double s, e, t, weight;
    int m, n, l, na;
    int answ;
    
    weight=8.0/(mult*mult*mult);
    for (iP=0; iP<mult; iP++)
        integP[iP]=-1.0+1.0/mult+(2.0/mult)*iP;
    defD(D,v);
    set0(Ael,24*24*mult*mult*mult);
    for (iP = 0; iP < mult; ++iP)
    for (kP = 0; kP < mult; ++kP)
    for (jP = 0; jP < mult; ++jP)
    {
        na=mult*mult*iP+mult*kP+jP;
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
        for (int n=0; n<24; n++)
        {
            for (int l=0; l<6; l++)
                Ael[m][n][na] +=weight*0.125*hx*hy*hz*B[l][m]*DB[l][n];
        }
    }
    
    #pragma omp parallel for shared(Ael,A,CX0, nelx,nely,nelz,nelxP,nelyP,nelzP) private(nel,nelP,i,j,k,iP,jP,kP,m,n,na)
    for (k = 0; k < nelz; k++)
    for (j = 0; j < nely; j++)
    for (i = 0; i < nelx; i++)
    {
        nel=nelz*nely*i+nely*k+j;
        for (m=0; m<24; m++)
        for (n=0; n<24; n++)
            A[nel][m][n] = 0.0;
             
        for (iP = 0; iP < mult; ++iP)
        for (kP = 0; kP < mult; ++kP)
        for (jP = 0; jP < mult; ++jP)
        {
            na=mult*mult*iP+mult*kP+jP;
            nelP=nelzP*nelyP*(mult*i+iP)+nelyP*(mult*k+kP)+mult*j+jP;
            for (m=0; m<24; m++)
            for (n=0; n<24; n++)
                A[nel][m][n] +=CX0[nelP]*Ael[m][n][na];
        }
    }
}


void choldc(int n, double a[][n], double p[])
//Given a positive-definite symmetric matrix a[1..n][1..n], this routine constructs its Cholesky
//decomposition, A = L · LT . On input, only the upper triangle of a need be given; it is not
//modified. The Cholesky factor L is returned in the lower triangle of a, except for its diagonal
//elements which are returned in p[1..n].
{
    int i,j,k;
    double sum;
    for (i=0;i<n;i++) {
    for (j=i;j<n;j++) {
        sum=a[i][j];
        for (k=i-1;k>=0;k--)
        {
            sum -= a[i][k]*a[j][k];
        }
        if (i == j) {
            if (sum <= 0.0)                //a, with rounding errors, is not positive definite.
                printf("choldc failed %d %g\n",i,sum);
            p[i]=sqrt(sum);
        }
        else a[j][i]=sum/p[i];
    }
    }
}

void cholsl(int n, double a[][n], double p[], double b[], double x[])
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
        for (sum=b[i],k=i-1;k>=0;k--) sum -= a[i][k]*x[k];
        x[i]=sum/p[i];
    }
    for (i=n-1;i>=0;i--) {                                    //Solve LT · x = y.
        for (sum=x[i],k=i+1;k<n;k++) sum -= a[k][i]*x[k];
        x[i]=sum/p[i];
    }
}
