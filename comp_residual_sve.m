function F=comp_residual_sve(u,time)

% global
global dat npar

% shortcuts
porder= npar.porder;
ndofs = npar.ndofs;
nel   = npar.nel;
gn    = npar.gn;
b     = npar.b;
dbdx  = npar.dbdx;
xq    = npar.xq;
wq    = npar.wq;

% set F to 0
F=0*u;
% initialize local residual vector
local_res=zeros(porder+1,2);

mu = compute_viscosity(u);

% tfn ( dq/dx + d/dx(-mu dh/dx) )
%   by parts:  tfn  dq/dx + mu dtfn/dx dh/dx) 

% eq.1: b dq/dx
% eq.2: b d(q^2/h+gh^2/2)/dx

for iel=1:nel
    % element extremities
    x0=npar.x(iel);
    x1=npar.x(iel+1);
    % jacobian of the transformation to the ref. element
    Jac=(x1-x0)/2;
    % get x values in the interval
    % x=(x1+x0)/2+xq*(x1-x0)/2;
    
    % b and dbdx(:,:) are of size (nbr of xq values) x (porder+1),
    % h/q are of length porder+1, 2/dx is the 1d jacobian
    gnh=gn(iel,1:porder+1);
    gnq=gn(iel,porder+2:end);
    h=u(gnh);
    q=u(gnq);
    
    % partial_x h
    local_dhdx = dbdx(:,:) * h;
    % partial_x q
    local_dqdx = dbdx(:,:) * q;
    % partial_x (q^2/h)
    local_dq2hdx = dbdx(:,:) * (q.^2./h);
    % partial_x (h^2)
    local_dh2dx = dbdx(:,:) * (h.^2);

    % compute local residual
    inviscid_flx_1=wq.*local_q;
    inviscid_flx_2=wq.*(local_dq2hdx+(dat.g/2)*local_dh2dx);
    viscous_flux_1=mu.*wq.*local_dhdx;
    viscous_flux_2=mu.*wq.*local_dqdx;
    for i=1:porder+1
        local_res(i,1) =  sum(inviscid_flx_1.*b(:,i))*Jac + sum(viscous_flux_1.*dbdx(:,i))/Jac;;
        local_res(i,2) =  sum(inviscid_flx_2.*b(:,i))*Jac + sum(viscous_flux_2.*dbdx(:,i))/Jac;;
    end
    F(gnT) = F(gnT) + local_res(:,1);
    F(gnE) = F(gnE) + local_res(:,2);

end

for iphys=1:npar.nphys
    % apply natural BC
    Dirichlet_nodes=[];
    Dirichlet_val=[];
    % LEFT
    ind = 1+sum(npar.ndofs(1:iphys-1));
    switch dat.bc.left.type(iphys)
        case 0 % Neumann, int_bd_domain (b D grad u n) is on the RHS
            F(ind)=F(ind)-dat.bc.left.C(iphys);
        case 1 % Robin
            F(ind)=F(ind)+1/2*u(ind);
            F(ind)=F(ind)-2*dat.bc.left.C(iphys);
        case 2 % Dirichlet
            Dirichlet_nodes=[Dirichlet_nodes ind];
            Dirichlet_val=[Dirichlet_val dat.bc.left.C(iphys)];
    end
    % RIGHT
    ind = sum(npar.ndofs(1:iphys));
    switch dat.bc.rite.type(iphys)
        case 0 % Neumann, int_bd_domain (b D grad u n) is on the RHS
            F(ind)=F(ind)-dat.bc.rite.C(iphys);
        case 1 % Robin
            F(ind)=F(ind)+1/2*u(ind);
            F(ind)=F(ind)-2*dat.bc.rite.C(iphys);
        case 2 % Dirichlet
            Dirichlet_nodes=[Dirichlet_nodes ind];
            Dirichlet_val=[Dirichlet_val dat.bc.rite.C(iphys)];
    end
    % apply Dirichlet BC
    for i=1:length(Dirichlet_nodes);% loop on the number of constraints
        id=Dirichlet_nodes(i);      % extract the dof of a constraint
        bcval=Dirichlet_val(i);
        F(id)=u(id)-bcval;         % put the constrained value in the rhs
    end
end

% move to rhs
F=-F;

return
end

