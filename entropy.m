function [eta,psi]=entropy(h,q)
% computes SVE entropy pair
global dat

eta=0.5*(q.^2./h + dat.g*h.^2);
psi=q./h.*eta + dat.g*h.*q;

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
