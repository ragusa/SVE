function u0 = init_sve

global npar dat
npar
% domain length, L
dat.L=1;
% gravity accel
dat.g=9.8;

% mesh
npar.nel=42; 
npar.x=linspace(0,dat.L,npar.nel+1); 
npar.dx=diff(npar.x);

% nbr of dofs per variable
npar.porder= 1;
npar.nphys = 2;
npar.ndofs(1) = npar.porder*npar.nel+1;
for i=2:npar.nphys
    npar.ndofs(i) = npar.ndofs(1);
end
npar.nnz_row=(2*npar.porder+1)*npar.nphys; % nnz per row of J; about 2 mass matrices for 2 physics

% connectivity
gn=zeros(npar.nel,npar.nphys*(npar.porder+1));
gn(1,1:npar.porder+1)=linspace(1,npar.porder+1,npar.porder+1);
for iel=2:npar.nel
    gn(iel,1:npar.porder+1)=[gn(iel-1,npar.porder+1) , gn(iel-1,2:npar.porder+1)+npar.porder ];
end
for i=2:npar.nphys
    i1=(npar.porder+1)*(i-1)+1;
    i2=(npar.porder+1)*(i  );
    gn(:,i1:i2) = gn(:,1:npar.porder+1) + npar.ndofs(i-1);
end
npar.gn=gn; clear gn;

% interpolation points used in basis function dfinition
npar.equidistant=true; 
npar.GLL=~npar.equidistant;

npar.M_lumped=true;

% % Robin with all values =0 is equivalent to Neumann BC
% var 1 = temperature, 2 = energy
bc.left.type(1)=0; %0=neumann, 1=robin, 2=dirichlet
bc.left.C(1)=0; % (that data is C in: -Ddu/dn=C // u/4+D/2du/dn=C // u=C)
bc.rite.type(1)=0;
bc.rite.C(1)=0;
%
bc.left.type(2)=1; %0=neumann, 1=robin, 2=dirichlet
bc.left.C(2)=1; % (that data is C in: -Ddu/dn=C // u/4+D/2du/dn=C // u=C)
bc.rite.type(2)=1;
bc.rite.C(2)=0;
dat.bc=bc; clear bc;


% compute the mass matrix
compute_mass_matrix()

% initial values
h=ones(npar.ndofs(1),1);
q=h*0;

u0=[h ; q];

return
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function compute_mass_matrix()

% make the problem-data a global variable
global dat npar

% shortcuts
porder= npar.porder;
gn    = npar.gn;
nel   = npar.nel;
% ideally, we would analyze the connectivity to determine nnz
nnz=npar.nphys*(2*porder+1)*nel; %this is an upperbound, not exact
% n: linear system size
n=sum(npar.ndofs);
% allocate memory
A=spalloc(n,n,nnz);
% initialize local matrices/vectors
m=zeros(porder+1,porder+1);

% compute local matrices
% load Gauss Legendre quadrature (GLQ is exact on polynomials of degree up to 2n-1,
% while using only integrand evaluations (n-point quadrature).
% estimate the max poly order (it comes from the mass matrix  when coef. are
% piecewise constant and the material mesh matches the computational mesh
poly_max=porder+1;
[xq,wq] = GLNodeWt(poly_max);
% store quadrature data
npar.xq=xq;
npar.wq=wq;
% store shapeset
[b,dbdx] =feshpln(xq,porder,npar.GLL);
% save shapeset in npar struct
npar.b=b;
npar.dbdx=dbdx;
% compute local matrices + load vector
for i=1:porder+1
    for j=1:porder+1
        m(i,j)= dot(wq.*b(:,i)    , b(:,j));
    end
end
% due to the special gn used here, whereby both variables are accessible in gn(iel,:)
m=kron(speye(npar.nphys),m);

% loop over elements
for iel=1:npar.nel
    % element extremities
    x0=npar.x(iel);
    x1=npar.x(iel+1);
    % jacobian of the transformation to the ref. element
    Jac=(x1-x0)/2;
    % assemble
    A(gn(iel,:),gn(iel,:)) = A(gn(iel,:),gn(iel,:)) + m*Jac;
end

% apply BC (Dirichlet 0 for the mass matrix where Dirichlet is required!!!)
Dirichlet_nodes=[];
Dirichlet_val=[];
for i=1:npar.nphys
    if(dat.bc.left.type(i)==2)
        i1=sum(npar.ndofs(1:i-1))+1;
        Dirichlet_nodes=[Dirichlet_nodes i1];
        Dirichlet_val=[Dirichlet_val dat.bc.left.C(i)];
    end
    if(dat.bc.rite.type(i)==2)
        i2=sum(npar.ndofs(1:i));
        Dirichlet_nodes=[Dirichlet_nodes i2];
        Dirichlet_val=[Dirichlet_val dat.bc.rite.C(i)];
    end
end
% apply Dirichlet BC
for i=1:length(Dirichlet_nodes);% loop on the number of constraints
    id=Dirichlet_nodes(i);      % extract the dof of a constraint
    bcval=Dirichlet_val(i);
    A(id,:)=0; % set all the id-th row to zero
    A(:,id)=0; % set all the id-th column to zero (symmetrize A)
    A(id,id)=1;            % set the id-th diagonal to unity
end

% store in structure
npar.Mass=A; clear A;

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
