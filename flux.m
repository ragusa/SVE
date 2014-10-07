function out=flux(pbID,uu)
% computes f(u)
switch(pbID)
    case{1}
        out = 0.5 * uu.^2;
    case{11,12,13,14}
        out = uu;
end

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
