function mu = compute_viscosity(u,u_old,u_older,dt)

global dat npar
% shortcuts
porder= npar.porder;
ndofs = npar.ndofs;
nel   = npar.nel;
gn    = npar.gn;
dx    = npar.dx;
b     = npar.b;
dbdx  = npar.dbdx;
xq    = npar.xq;
wq    = npar.wq;

qorder=length(wq);
ent =zeros(qorder,nel);
eave=zeros(nel,1);


mu_1 = zeros(nel,qorder);
mu_e = zeros(nel,qorder);

for iel=1:nel,
    gnh=gn(iel,1:porder+1);
    gnq=gn(iel,porder+2:end);
    h=u(gnh);
    q=u(gnq);

    [ent,dummy] = entropy(h,q);
    eave(iel) = dot(wq,ent)/2; % sum quad=2
end
% domain averaged entropy
ent_ave = dot(eave,dx)/sum(dx);
% compute the normalization value
norm_ = max( max( abs( ent(:,:)-ent_ave) ) );


for iel=1:nel
    % b and dbdx(:,:) are of size (nbr of xq values) x (porder+1),
    % h/q are of length porder+1, 2/dx is the 1d jacobian
    gnh=gn(iel,1:porder+1);
    gnq=gn(iel,porder+2:end);
    h=u(gnh);
    q=u(gnq);
    h_old=u_old(gnh);
    q_old=u_old(gnq);
    
    % h, q
    local_h = b(:,:) * h;
    local_q = b(:,:) * q;

    % first order viscosity
    eig = abs(q./h + sqrt(dat.g*h));
    mu_1(iel,:)=0.5*dx(iel)*max(eig);
    
    % entropy viscosity
    [eta    ,psi  ] = entropy(h    ,q    );
    [eta_old,dummy] = entropy(h_old,q_old);
    residual = b(:,:) * (eta - eta_old)/dt ...
             + (local_q./local_h).* dbdx(:,:)*psi;
    mu_e(iel,:) = (dx(iel))^2*residual/norm_;
    mu(iel,:) = min(mu_1(iel,:), mu_e(iel,:) );
    
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
    viscous_flux_1=mu(iel,:).*wq.*local_dhdx;
    viscous_flux_2=mu(iel,:).*wq.*local_dqdx;
    for i=1:porder+1
        local_res(i,1) =  sum(inviscid_flx_1.*b(:,i))*Jac + sum(viscous_flux_1.*dbdx(:,i))/Jac;;
        local_res(i,2) =  sum(inviscid_flx_2.*b(:,i))*Jac + sum(viscous_flux_2.*dbdx(:,i))/Jac;;
    end
    F(gnT) = F(gnT) + local_res(:,1);
    F(gnE) = F(gnE) + local_res(:,2);

end

