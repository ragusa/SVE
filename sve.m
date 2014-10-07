function sve
% cleanup
clear all; close all; clc;

% global variables
global dat npar

% initialize data
u0 = init_sve;

t_end=10;
n_steps=100;
dt=t_end/n_steps;
time_0=0;

figure; hold on;
subplot(2,1,1);plot(npar.x,u0(1:npar.ndofs(1)));
subplot(2,1,2);plot(npar.x,u0(npar.ndofs(1)+1:end));
for it=1:n_steps
    % compute time at the end of current time step
    time_1=(it  )*dt;
    % compute sve residual
    resi=comp_residual_sve(u0,time_0)
    % forward Euler
    u1=u0+dt*resi;
    % plots
    subplot(2,1,1);plot(npar.x,u1(1:npar.ndofs(1)));
    subplot(2,1,2);plot(npar.x,u1(npar.ndofs(1)+1:end));
    % next time step
    u0=u1;
    time_0=time_1;
end

return;
end
