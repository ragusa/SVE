function visc=viscosity(pbID,u1,RKci_dt,logi)

global npar dat

% short-cuts
dx   =  npar.dx;
nel  =  npar.nel;
shape = npar.b;
dhdr  = npar.dbdx;
gn    = npar.gn;
wquad = npar.wq;
% recall that shape is a matrix of size (qorder,porder+1)
qorder=length(wquad);
porder=npar.porder;
% first-order visc
c1=npar.c1;
% entr visc
cE=npar.cE;

% upper bound to entropy viscosity for all quadrature points
% compute at new time because it will be used in the next RK stage
speed=zeros(qorder,nel);
% entropies: current and old ones
ent =zeros(qorder,nel);
%ento=zeros(qorder,nel);
ent0 = npar.entro.ent0;
eave=zeros(nel,1);

for iel=1:nel,
    g=gn(iel,:);
    % coef_u = uu(g(:)); % vector of length porder+1
    % recall that shape is a matrix of size (qorder,porder+1)
    %local_u0 = shape * u0(g(:)); % vector of length qorder
    local_u1 = shape * u1(g(:)); % vector of length qorder
    
    % speed(:,iel) = abs( dfdu(pbID,local_u1) );
    % speed is the maximum over the cell:
    speed(:,iel) = max(abs( dfdu(pbID,local_u1) ));
    ent1(:,iel) = entropy(pbID,local_u1);
    % ent0(:,iel) = entropy(pbID,local_u0);
    eave(iel) = dot( wquad, ent1(:,iel))/2; % sum quad=2
end

% domain averaged entropy
ent_ave = dot(eave,dx)/sum(dx);
% compute the normalization value
norm_ = max( max( abs( ent1(:,:)-ent_ave) ) );

% residual = dE/dt + dF/dx
% where F = entropy flux.
% however, dF/dx = F' * du/dx
% and F' = dF/du = E' * f'
% thus, residual = dE/dt + E'*f'*du/dx
%
% short demo: du/dt + df/dx = 0 <==> du/dt + f'(u) du/dx = 0
% for an entropy pair (E(u),F(u)) satisfying dE/dt + dF/dx = 0
% we have (smoothness condition) E'(u) du/dt + F'(u) du/dx = 0
% which is compared to: E'(u) [ du/dt + f'(u) du/dx ]
resi=zeros(qorder,nel);
% time derivative portion
resi = (ent1-ent0)/RKci_dt;
for iel=1:nel,
    g=gn(iel,:);
    coef_u = u1(g(:)); % vector of length porder+1
    % recall that shape is a matrix of size (qorder,porder+1)
    local_u  = shape * coef_u; % vector of length qorder
    %wrong   local_fp = flux(pbID,local_u);
    local_fp = dfdu(pbID,local_u);
    local_Ep = dEdu(pbID,local_u);
    % vector of length qorder, 2/dx is the 1d jacobian
    local_dudx = 2/dx(iel) * dhdr * coef_u;
    
    resi(:,iel) = resi(:,iel) + local_Ep.*local_fp.*local_dudx;
    % normalize by |E_q-Eave|
    %    resi(:,iel) = abs( resi(:,iel) ./ (ent1(:,iel)-eave(iel)) );
    %     resi(:,iel) = abs( resi(:,iel) ./ (ent1(:,iel)-ent_ave) );
    resi(:,iel) = abs( resi(:,iel)) / norm_ ;
    resi(:,iel) = max(resi(:,iel));
end

% just to get xq again
xq=npar.xq;
del=max(abs(diff(xq)));

visc = zeros(qorder,nel);
aux_dx = kron(ones(1,qorder),dx)';
aux_dx_p = aux_dx/porder;
%
aux_dx_quad = aux_dx*del/2;
aux_dx_quad = aux_dx_p;

if(cE>1.e5)
    %     for iel=1:nel,
    %         visc = dx(iel)*c1*speed(:,iel);
    %     end
    visc = c1 * aux_dx_p .* speed;
else
    % here, the size h for the entropy visc is the minimum
    % between 2 quadrature points = dx/porder for lagrange polynomials
    % where the size for the maximum entropy (speed) is dx BUT
    % cmax(=c1) is a constant/porder in JLG paper, so it is like using
    % dx/porder everywhere in the formula
    visc = min( c1*aux_dx_p.*speed , cE*(aux_dx_quad).^2.*resi );
end

% operator S, smoother
a=reshape(visc,qorder*nel,1);
% n=qorder*nel;
% T=spalloc(n,n,3*n-2);
% T=T+speye(n)/2;
% i=n;T(i,i-1)=0.25;
% i=1;T(i,i+1)=0.25;
% for i=2:n-1
%     T(i,i-1)=0.25;
%     T(i,i+1)=0.25;
% end
% a=T*a;

% a=smoothing(a,3);

% a=smoothing_2(a,1);

visc=reshape(a,qorder,nel);

return
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = smoothing(uu,howmany)

out = uu;
for i=1:howmany
    out(1) = ( uu(end) + 4*uu(1) + uu(2) ) /6;
    out(2:end-1) = (uu(1:end-2) + 4*uu(2:end-1) + uu(3:end) ) /6;
    out(end) = (uu(end-1) + 4*uu(end) + uu(1) ) /6;
    uu=out;
end

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = smoothing_2(uu,howmany)

out = uu;
for i=1:howmany
    out(1) = max( [uu(end);uu(1:2)] ) ;
    for i=2:length(uu)-1
        out(i) = max( uu(i-1:i+1) );
    end
    out(end) = max( [uu(end-1:end);uu(1)] );
    uu=out;
end

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

