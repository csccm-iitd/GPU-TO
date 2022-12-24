%function top3D125(nelx,nely,nelz,volfrac,penal,rmin,ft,ftBC,eta,beta,move,maxit)
nl=5;
nelx=16*2^(nl-1);
nely=4*2^(nl-1);
nelz=8*2^(nl-1);
volfrac=0.2;
volfrac0=0.1;
penal=3;
rmin=sqrt(3.0);
ft=3;
ftBC='N';
eta=0.4;
move=0.1;
maxit=100;
beta=2;

filename = strcat('ArchBridge2_',num2str(nl),'_', num2str(nelx),'_', num2str(nely),'_', num2str(nelz),'caml','.txt');    
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
Emin = 1e-9;                                                               % Young modulus of "void"
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

fixed1=[1:3];
fixed2=fixed1+3*nely;
fixed3=[3*nodeNrs(1, 1, nelx+1)-2 : 3*nodeNrs(1, 1, nelx+1)];
fixed4=[3*nodeNrs(nely+1, 1, nelx+1)-2 : 3*nodeNrs(nely+1, 1, nelx+1)];
fixed = double(union(fixed4,union(fixed3,union(fixed1,fixed2))));
[ pasS, pasV ] = deal( [], [] );                                            % passive solid and void elements
elmNrs = int32( reshape( 1 : nelx*nely*nelz ,nely, nelz, nelx));            % elements numbering             #3D#
pasV=elmNrs(5:nely-4,nelz/2+2:nelz,1:nelx);
pasS=elmNrs(1:nely,nelz/2:nelz/2+1,1:nelx);

F=zeros(nDof,1);

for i = 1:nelx
  for j = 5:nely-4
    elpress=elmNrs(j,nelz/2+1,i);
    for nn=5:8
      F(cMat(elpress,nn*3)) = F(cMat(elpress,nn*3)) - 0.25*1.0;
    end
  end
end

F(fixed)=0.0;

%lcDof = 3 * nodeNrs( 14 : 52, 97, 129: nelx );
%lcDof=pasS;
%F(lcDof)=-100;

free = setdiff( 1 : nDof, fixed );                                         % set of free DOFs
act = setdiff( ( 1 : nEl )', union( pasS, pasV ) );                        % set of active d.v.

actV=elmNrs(1:nely,1:nelz/2-32,33:nelx-32);
actS = setdiff( act, actV ); 

% --------------------------------------- PRE. 4) DEFINE IMPLICIT FUNCTIONS
prj = @(v,eta,beta) (tanh(beta*eta)+tanh(beta*(v(:)-eta)))./...
    (tanh(beta*eta)+tanh(beta*(1-eta)));                                   % projection
deta = @(v,eta,beta) - beta * csch( beta ) .* sech( beta * ( v( : ) - eta ) ).^2 .* ...
    sinh( v( : ) * beta ) .* sinh( ( 1 - v( : ) ) * beta );                % projection eta-derivative 
dprj = @(v,eta,beta) beta*(1-tanh(beta*(v-eta)).^2)./(tanh(beta*eta)+tanh(beta*(1-eta)));% proj. x-derivative
cnt = @(v,vCnt,l) v+(l>=vCnt{1}).*(v<vCnt{2}).*(mod(l,vCnt{3})==0).*vCnt{4};
% -------------------------------------------------- PRE. 5) PREPARE FILTER
[dy, dz, dx] = meshgrid(-ceil(rmin)+1:ceil(rmin)-1,...
    -ceil(rmin)+1:ceil(rmin)-1,-ceil(rmin)+1:ceil(rmin)-1 );
h = max( 0, rmin - sqrt( dx.^2 + dy.^2 + dz.^2 ) );                        % conv. kernel                #3D#
Hs = imfilter( ones( nely, nelz, nelx ), h, bcF );                         % matrix of weights (filter)  #3D#
dHs = Hs;
% ------------------------ PRE. 6) ALLOCATE AND INITIALIZE OTHER PARAMETERS

[ x, dsK, dV ] = deal( zeros( nEl, 1 ) );                                  % initialize vectors
%dV( act, 1 ) = 1/nEl/volfrac;                                              % derivative of volume

%x( act ) = (volfrac*( nEl - length(pasV) ) - length(pasS) )/length( act );% volume fraction on active set
%x( act ) = (volfrac*( nEl - numel(pasV) ) - numel(pasS) )/numel( act );    % volume fraction on active set

%x( act ) = volfrac0;

x( actS ) = 0.12;
x( actV ) = 5e-2;
x( pasS ) = 1;                                                             % set x = 0.05 on pasS set
x( pasV ) = 1e-6;

%volfrac=(numel(act)*volfrac0+numel(pasS)*1+numel(pasV)*1e-6)/nEl;
volfrac=(numel(actS)*0.1+numel(actV)*5e-2+numel(pasS)*1+numel(pasV)*1e-6)/nEl;
dV( act, 1 ) = 1/nEl/volfrac;                                              % derivative of volume

[ xPhys, xOld, ch, loop, U ] = deal( x, 1, 1, 0, zeros( nDof, 1 ) );       % old x, x change, it. counter, U

% ================================================= START OPTIMIZATION LOOP

while ch > 7e-7 && loop < maxit
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
  
  CX=Emin+xPhys.^penal*(E0-Emin);
  nfd=length(fixed);
  cgtol=1e-9;
  cgmax=700;
  printLev=1;
  [cgres,cgiters]=mgcg1(Ke0,F,U,CX,fixed',nfd,nelx,nely,nelz,nl,cgtol,cgmax,printLev);
  if cgiters<0, break; end
  fprintf('MGCG relres: %4.2e iters: %4i \n', cgres, cgiters);
  
  % ------------------------------------------ RL. 3) COMPUTE SENSITIVITIES
  dc = dsK .* sum( ( U( cMat ) * Ke0 ) .* U( cMat ), 2 );                  % derivative of compliance
  dc = imfilter( reshape( dc, nely, nelz, nelx ) ./ dHs, h, bcF );         % filter objective sens.      #3D#
  dV0 = imfilter( reshape( dV, nely, nelz, nelx ) ./ dHs, h, bcF );        % filter compliance sens.     #3D#
  % ----------------- RL. 4) UPDATE DESIGN VARIABLES AND APPLY CONTINUATION
  xT = x( act );
  [ xU, xL ] = deal( xT + move, xT - move );                               % current upper and lower bound
  ocP = xT .* sqrt( - dc( act ) ./ dV0( act ) );                           % constant part in resizing rule
  l = [ 0, mean( ocP ) / volfrac ];                                        % initial estimate for LM
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
  t = toc;
  fileID = fopen(filename,'a');
  fprintf( fileID, 'It.:%5i C:%6.5e Time:%4.3f V:%7.3f ch.:%0.2e penal:%7.2f beta:%7.1f eta:%7.2f lm:%0.2e cgres: %3.4e,cgiters: %3d MemUsed:%6.4e \n', ...
            loop, F'*U, t, mean(xPhys(:)), ch, penal, beta, eta, lmid, cgres, cgiters, user.MemUsedMATLAB );
  fclose(fileID);
end

isovals = shiftdim( reshape( xPhys, nely, nelz, nelx ), 2 );
isovals = smooth3( isovals, 'box', 1 );
patch(isosurface(isovals, .5),'FaceColor','b','EdgeColor','none');
patch(isocaps(isovals, .5),'FaceColor','r','EdgeColor','none');
drawnow; view( [ 145, 25 ] ); axis equal tight off; camlight;

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
