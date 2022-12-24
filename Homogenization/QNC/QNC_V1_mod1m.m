%function top3D125(nelx,nely,nelz,volfrac,penal,rmin,ft,ftBC,eta,beta,move,maxit)
clear;
nl=6;
nelx=48*2^(nl-1);
nely=2*2^(nl-1);
nelz=8*2^(nl-1);
volfrac=0.2;
volfrac0=0.1;
penal=3;
rmin=3.1*sqrt(3.0);
ft=1;
ftBC='N';
eta=0.4;
move=0.2;
maxit=200;
beta=2;

hx=1.0;
hy=1.0;
hz=1.0;

filename = strcat('QNC_V1',num2str(nl),'_', num2str(nelx),'_', num2str(nely),'_', num2str(nelz),'caml','.txt');    
fileID = fopen(filename,'w');
fprintf(fileID, ['Top3D125mod \n\nnl:%2d, nelx:%4d, nely:%4d, nelz:%4d, '...
    'volfrac:%.2f, penal:%2d, rmin:%4.4f, ft:%2d, eta:%.2f, move:%.2f, ',...
    'maxit:%3d, beta:%1.2f \n'],nl, nelx, nely, nelz, volfrac, penal, ...
    rmin, ft, eta, move, maxit, beta);
user = memory;
MemI = user.MemUsedMATLAB; 
fprintf(fileID,'\nInitial Memory USED: %6.4e Bytes = %6.4e GB\n\n',MemI,MemI/1e9);
fclose(fileID);

% ---------------------------- PRE. 1) MATERIAL AND CONTINUATION PARAMETERS
E0 = 1;                                                                    % Young modulus of solid
Emin = 1e-6;                                                               % Young modulus of "void"
nu = 0.3;                                                                  % Poisson ratio
penalCnt = { 1, 1, 25, 0.25 };                                             % continuation scheme on penal
betaCnt  = { 1, 1, 25,    2 };                                             % continuation scheme on beta
if ftBC == 'N', bcF = 'symmetric'; else, bcF = 0; end                      % filter BC selector
% ----------------------------------------- PRE. 2) DISCRETIZATION FEATURES
nEl = nelx * nely * nelz;                                                  % number of elements          #3D#
nodeNrs = int32( reshape( 1 : ( 1 + nelx ) * ( 1 + nely ) * ( 1 + nelz ), ...
    1 + nely, 1 + nelz, 1 + nelx ) );                                      % nodes numbering             #3D#
cVec = reshape( 3 * nodeNrs( 1 : nely, 1 : nelz, 1 : nelx ) + 1, nEl, 1 ); %                             #3D#
cMat = cVec+int32( [0,1,2,3*(nely+1)*(nelz+1)+[0,1,2,-3,-2,-1],-3,-2,-1,3*(nely+...
   1)+[0,1,2],3*(nely+1)*(nelz+2)+[0,1,2,-3,-2,-1],3*(nely+1)+[-3,-2,-1]]);% connectivity matrix         #3D#
nDof = ( 1 + nely ) * ( 1 + nelz ) * ( 1 + nelx ) * 3;                     % total number of DOFs        #3D#
%-[ sI, sII ] = deal( [ ] );
%-for j = 1 : 24
%-    sI = cat( 2, sI, j : 24 );
%-    sII = cat( 2, sII, repmat( j, 1, 24 - j + 1 ) );
%-end
%-[ iK , jK ] = deal( cMat( :,  sI )', cMat( :, sII )' );
%Iar = sort( [ iK( : ), jK( : ) ], 2, 'descend' ); clear iK jK              % reduced assembly indexing
Ke = 1/(1+nu)/(2*nu-1)/144 *( [ -32;-6;-6;8;6;6;10;6;3;-4;-6;-3;-4;-3;-6;10;...
    3;6;8;3;3;4;-3;-3; -32;-6;-6;-4;-3;6;10;3;6;8;6;-3;-4;-6;-3;4;-3;3;8;3;...
    3;10;6;-32;-6;-3;-4;-3;-3;4;-3;-6;-4;6;6;8;6;3;10;3;3;8;3;6;10;-32;6;6;...
    -4;6;3;10;-6;-3;10;-3;-6;-4;3;6;4;3;3;8;-3;-3;-32;-6;-6;8;6;-6;10;3;3;4;...
    -3;3;-4;-6;-3;10;6;-3;8;3;-32;3;-6;-4;3;-3;4;-6;3;10;-6;6;8;-3;6;10;-3;...
    3;8;-32;-6;6;8;6;-6;8;3;-3;4;-3;3;-4;-3;6;10;3;-6;-32;6;-6;-4;3;3;8;-3;...
    3;10;-6;-3;-4;6;-3;4;3;-32;6;3;-4;-3;-3;8;-3;-6;10;-6;-6;8;-6;-3;10;-32;...
    6;-6;4;3;-3;8;-3;3;10;-3;6;-4;3;-6;-32;6;-3;10;-6;-3;8;-3;3;4;3;3;-4;6;...
    -32;3;-6;10;3;-3;8;6;-3;10;6;-6;8;-32;-6;6;8;6;-6;10;6;-3;-4;-6;3;-32;6;...
    -6;-4;3;6;10;-3;6;8;-6;-32;6;3;-4;3;3;4;3;6;-4;-32;6;-6;-4;6;-3;10;-6;3;...
    -32;6;-6;8;-6;-6;10;-3;-32;-3;6;-4;-3;3;4;-32;-6;-6;8;6;6;-32;-6;-6;-4;...
    -3;-32;-6;-3;-4;-32;6;6;-32;-6;-32]+nu*[ 48;0;0;0;-24;-24;-12;0;-12;0;...
    24;0;0;0;24;-12;-12;0;-12;0;0;-12;12;12;48;0;24;0;0;0;-12;-12;-24;0;-24;...
    0;0;24;12;-12;12;0;-12;0;-12;-12;0;48;24;0;0;12;12;-12;0;24;0;-24;-24;0;...
    0;-12;-12;0;0;-12;-12;0;-12;48;0;0;0;-24;0;-12;0;12;-12;12;0;0;0;-24;...
    -12;-12;-12;-12;0;0;48;0;24;0;-24;0;-12;-12;-12;-12;12;0;0;24;12;-12;0;...
    0;-12;0;48;0;24;0;-12;12;-12;0;-12;-12;24;-24;0;12;0;-12;0;0;-12;48;0;0;...
    0;-24;24;-12;0;0;-12;12;-12;0;0;-24;-12;-12;0;48;0;24;0;0;0;-12;0;-12;...
    -12;0;0;0;-24;12;-12;-12;48;-24;0;0;0;0;-12;12;0;-12;24;24;0;0;12;-12;...
    48;0;0;-12;-12;12;-12;0;0;-12;12;0;0;0;24;48;0;12;-12;0;0;-12;0;-12;-12;...
    -12;0;0;-24;48;-12;0;-12;0;0;-12;0;12;-12;-24;24;0;48;0;0;0;-24;24;-12;...
    0;12;0;24;0;48;0;24;0;0;0;-12;12;-24;0;24;48;-24;0;0;-12;-12;-12;0;-24;...
    0;48;0;0;0;-24;0;-12;0;-12;48;0;24;0;24;0;-12;12;48;0;-24;0;12;-12;-12;...
    48;0;0;0;-24;-24;48;0;24;0;0;48;24;0;0;48;0;0;48;0;48 ] );             % elemental stiffness matrix  #3D#
Ke0( tril( ones( 24 ) ) == 1 ) = Ke';
Ke0 = reshape( Ke0, 24, 24 );
Ke0 = Ke0 + Ke0' - diag( diag( Ke0 ) );                                    % recover full matrix

% ----------------------------- PRE. 3) LOADS, SUPPORTS AND PASSIVE DOMAINS
fixed1=[3*nodeNrs(1:nely/2+1, 1, 5*nelx/16+1:6*nelx/16+1)];
fixed2=[3*nodeNrs(1:nely+1,1:nelz+1,1)-2];
fixed3=[3*nodeNrs(1,1:nelz+1,1:nelx+1)-1];
fixed = double(union(fixed3,union(fixed1,fixed2)));

[ pasS, pasV ] = deal( [], [] );                                            % passive solid and void elements
elmNrs = int32( reshape( 1 : nelx*nely*nelz ,nely, nelz, nelx));            % elements numbering             #3D#
pasS=elmNrs(1:nely,nelz*7/8+1:nelz,1:nelx);
pasV1=elmNrs(nely/2+1:nely,1:nelz*7/8,1:nelx);
pasV2=elmNrs(1:nely/2,1:nelz*7/8,1:1*nelx/16);
pasV=union(pasV1,pasV2);

F=zeros(nDof,1);
for i = 1:nelx
  for j = 1:nely
    F(3*nodeNrs(   j, nelz+1, i  )) = F(3*nodeNrs(   j, nelz+1, i  )) - 0.25*1.0;
    F(3*nodeNrs(   j, nelz+1, i+1)) = F(3*nodeNrs(   j, nelz+1, i+1)) - 0.25*1.0;
    F(3*nodeNrs( j+1, nelz+1, i+1)) = F(3*nodeNrs( j+1, nelz+1, i+1)) - 0.25*1.0;
    F(3*nodeNrs( j+1, nelz+1, i  )) = F(3*nodeNrs( j+1, nelz+1, i  )) - 0.25*1.0;
    % elpress=elmNrs(j,nelz,i);
    % for nn=5:8
    %   F(cMat(elpress,nn*3)) = F(cMat(elpress,nn*3)) - 0.25*1.0;
    % end
  end
end
F(fixed)=0.0;

free = setdiff( 1 : nDof, fixed );                                         % set of free DOFs
act = setdiff( ( 1 : nEl )', union( pasS, pasV ) );                        % set of active d.v.

% --------------------------------------- PRE. 4) DEFINE IMPLICIT FUNCTIONS
prj = @(v,eta,beta) (tanh(beta*eta)+tanh(beta*(v(:)-eta)))./...
    (tanh(beta*eta)+tanh(beta*(1-eta)));                                   % projection
deta = @(v,eta,beta) - beta * csch( beta ) .* sech( beta * ( v( : ) - eta ) ).^2 .* ...
    sinh( v( : ) * beta ) .* sinh( ( 1 - v( : ) ) * beta );                % projection eta-derivative 
dprj = @(v,eta,beta) beta*(1-tanh(beta*(v-eta)).^2)./(tanh(beta*eta)+tanh(beta*(1-eta)));% proj. x-derivative
cnt = @(v,vCnt,l) v+(l>=vCnt{1}).*(v<vCnt{2}).*(mod(l,vCnt{3})==0).*vCnt{4};
% -------------------------------------------------- PRE. 5) PREPARE FILTER
[dy,dz,dx]=meshgrid(-ceil(rmin)+1:ceil(rmin)-1,...
    -ceil(rmin)+1:ceil(rmin)-1,-ceil(rmin)+1:ceil(rmin)-1 );
h = max( 0, rmin - sqrt( dx.^2 + dy.^2 + dz.^2 ) );                        % conv. kernel                #3D#
Hs = imfilter( ones( nely, nelz, nelx ), h, bcF );                         % matrix of weights (filter)  #3D#
dHs = Hs;
% ------------------------ PRE. 6) ALLOCATE AND INITIALIZE OTHER PARAMETERS

[ x, dsK, dV ] = deal( zeros( nEl, 1 ) );                                  % initialize vectors

x( act ) = 0.2;
x( pasS ) = 1;                                                             % set x = 0.05 on pasS set
x( pasV ) = 1e-6;
for k = 1:nelz*7/8
  for i = 1*nelx/16:nelx
    for j = 1:nely/2
      if i<5/16*nelx - 3*k/2 - 1
         x(elmNrs(j,k,i))=1e-6;
      end
      if i>6/16*nelx + k*15/4 + 1
         x(elmNrs(j,k,i))=1e-6;
      end
      if i>11/32*nelx-k+32 & i<11/32*nelx + 3*k/2 - 32 & k>nelz/8+48
         x(elmNrs(j,k,i))=1e-6;
      end
    end
  end
end
% load x1ja;
volfrac=sum(x)/nEl;
%volfrac=(numel(act)*0.12+numel(pasS)*1+numel(pasV)*1e-6)/nEl;
dV( act, 1 ) = 1/nEl/volfrac;                                              % derivative of volume
[ xPhys, xOld, ch, loop, U ] = deal( x, 1, 1, 0, zeros( nDof, 1 ) );       % old x, x change, it. counter, U
chtol=5e-8;
% ================================================= START OPTIMIZATION LOOP
while ch > chtol && loop < maxit
  tic
  loop = loop + 1;                                                         % update iter. counter
  % ----------- RL. 1) COMPUTE PHYSICAL DENSITY FIELD (AND ETA IF PROJECT.)
  xTilde = imfilter( reshape( x, nely, nelz, nelx ), h, bcF ) ./ Hs;       % filtered field              #3D#
  xPhys( act ) = xTilde( act );                                            % reshape to column vector
  if ft > 1                                                                % compute optimal eta* with Newton
      f = ( mean( prj( xPhys, eta, beta ) ) - volfrac )  * (ft == 3);      % function (volume)
      while abs( f ) > 1e-6           % Newton process for finding opt. eta
          eta = eta - f / mean( deta( xPhys, eta, beta ) );
          f = mean( prj( xPhys, eta, beta ) ) - volfrac;
      end
      dHs = Hs ./ reshape( dprj( xPhys, eta, beta ), nely, nelz, nelx );   % sensitivity modification    #3D#
      xPhys = prj( xPhys, eta, beta );                                     % projected (physical) field
  end
  ch = norm( xPhys - xOld ) ./ nEl;
  xOld = xPhys;
  % -------------------------- RL. 2) SETUP AND SOLVE EQUILIBRIUM EQUATIONS
  %-sK = ( Emin + xPhys.^penal * ( E0 - Emin ) );
  dsK( act ) = -penal * ( E0 - Emin ) * xPhys( act ) .^ ( penal - 1 );
  %-sK = reshape( Ke( : ) * sK', length( Ke ) * nEl, 1 );
  %-K = fsparse( Iar( :, 1 ), Iar( :, 2 ), sK, [ nDof, nDof ] );
  %-L = chol( K( free, free ), 'lower' );
  %-U( free ) = L' \ ( L \ F( free ) );                                      % f/b substitution
  F=zeros(nDof,1);
  for iel=act
  for in=1:8
  F(cMat(iel,in*3)) = F(cMat(iel,in*3)) - 0.125*xPhys(iel);
  end
  end
  for iel=pasS
  for in=1:8
  F(cMat(iel,in*3)) = F(cMat(iel,in*3)) - 0.125*x(iel);
  end
  end
  F(fixed)=0.0;
  CX=Emin+xPhys.^penal*(E0-Emin);
  nfd=length(fixed);
  cgtol=1e-6;
  cgmax=500;
  
  nu=0.3; nswp=5; printLev=1;
%   [cgres,cgiters]=mgcg1(Ke0,F,U,CX,fixed',nfd,nelx,nely,nelz,nl,cgtol,cgmax,printLev);
  [cgres,cgiters]=mgcg9(Ke0,F,U,CX,fixed',nfd,nelx,nely,nelz,nl,cgtol,cgmax,nswp, printLev, nu, hx, hy, hz);
  if cgiters<0, break; end
  fprintf('MGCG relres: %4.2e iters: %4i \n',cgres,cgiters);
  
  % ------------------------------------------ RL. 3) COMPUTE SENSITIVITIES
  dc = dsK .* sum( ( U( cMat ) * Ke0 ) .* U( cMat ), 2 );                  % derivative of compliance
  dc(dc>0)=0.0;
  dc = imfilter( reshape( dc, nely, nelz, nelx ) ./ dHs, h, bcF );         % filter objective sens.      #3D#
  dV0 = imfilter( reshape( dV, nely, nelz, nelx ) ./ dHs, h, bcF );        % filter compliance sens.     #3D#
  % ----------------- RL. 4) UPDATE DESIGN VARIABLES AND APPLY CONTINUATION
  xT = x( act );
  [ xU, xL ] = deal( xT + move, xT - move );                               % current upper and lower bound
  ocP = xT .* sqrt( - dc( act ) ./ dV0( act ) );                           % constant part in resizing rule
  l = [ 0, mean( ocP ) / volfrac ];                                        % initial estimate for LM
  lmid=0.0;
  while ( l( 2 ) - l( 1 ) ) / ( l( 2 ) + l( 1 ) ) > 1e-4                   % OC resizing rule
      lmid = 0.5 * ( l( 1 ) + l( 2 ) );
      x( act ) = max( max( min( min( ocP / lmid, xU ), 1 ), xL ), 0 );
      if mean( x ) > volfrac, l( 1 ) = lmid; else, l( 2 ) = lmid; end
  end
  [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));   % apply conitnuation on parameters
  % -------------------------- RL. 5) PRINT CURRENT RESULTS AND PLOT DESIGN
  fprintf( 'It.:%5i C:%6.5e V:%7.3f ch.:%0.2e penal:%7.2f beta:%7.1f eta:%7.2f lm:%0.2e \n', ...
      loop, F'*U, mean(xPhys(:)), ch, penal, beta, eta, lmid );
%   isovals = shiftdim( reshape( xPhys, nely, nelz, nelx ), 2 );
%   isovals = smooth3( isovals, 'box', 1 );
%   patch(isosurface(isovals, .5),'FaceColor','b','EdgeColor','none');
%   patch(isocaps(isovals, .5),'FaceColor','r','EdgeColor','none');
%   drawnow; view( [ 145, 25 ] ); axis equal tight off; cla();
  t=toc;
  % fileID = fopen(filename,'a');
  % fprintf( fileID, 'It.:%5i C:%6.5e Time:%4.3f V:%7.3f ch.:%0.2e penal:%7.2f beta:%7.1f eta:%7.2f lm:%0.2e cgres: %3.4e,cgiters: %3d MemUsed:%6.4e \n', ...
  %           loop, F'*U, t, mean(xPhys(:)), ch, penal, beta, eta, lmid, cgres, cgiters, user.MemUsedMATLAB );
  % fclose(fileID);
  % isovals = shiftdim( reshape( xPhys, nely, nelz, nelx ), 2 );
  % isovals = smooth3( isovals, 'box', 1 );
  % patch(isosurface(isovals, .5),'FaceColor','b','EdgeColor','none');
  % patch(isocaps(isovals, .5),'FaceColor','r','EdgeColor','none');
  % drawnow; view( [ 145, 25 ] ); axis equal tight off; camlight;
  % 
  % savefig(strcat('QNC_V1',num2str(loop),'.fig'));
end

% isovals = shiftdim( reshape( xPhys, nely, nelz, nelx ), 2 );
% isovals = smooth3( isovals, 'box', 1 );
% patch(isosurface(isovals, .5),'FaceColor','b','EdgeColor','none');
% patch(isocaps(isovals, .5),'FaceColor','r','EdgeColor','none');
% drawnow; view( [ 145, 25 ] ); axis equal tight off; camlight;

%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab code was written by F. Ferrari, O. Sigmund                   %
% Dept. of Solid Mechanics-Technical University of Denmark,2800 Lyngby (DK)%
% Please send your comments to: feferr@mek.dtu.dk                          %
%                                                                          %
% The code is intended for educational purposes and theoretical details    %
% are discussed in the paper Ferrari, F. Sigmund, O. - A new generation 99 %
% line Matlab code for compliance Topology Optimization and its extension  %
% to 3D, SMO, 2020                                                         %
%                                                                          %
% The code as well as a postscript version of the paper can be             %
% downloaded from the web-site: http://www.topopt.dtu.dk                   %
%                                                                          %
% Disclaimer:                                                              %
% The authors reserves all rights but do not guaranty that the code is     %
% free from errors. Furthermore, we shall not be liable in any event       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
