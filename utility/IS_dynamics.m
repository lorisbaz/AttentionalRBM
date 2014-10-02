function [x_curr,vel_curr] = IS_dynamics(x_prec,vel_prec,pdf_prec,sigma,sigma_v,N)
% [x_curr,vel_curr] = IS_dynamics(x_prec,vel_prec,pdf_prec,sigma,sigma_v,N)
%
% This function perform Importance Sampling with a simple linear, Gaussina 
% Dynamics Model as proposal distribution. Note that using this simple
% proposal distribution make simpler the computation of the importance
% weights, but it is a very poor proposal and it could be improved.
%
% - x_prec, vel_prec: particles at previous time (position and velocity)
% - pdf_prec: importance weights
% - sigma,sigma_v: gaussian dynamics parameters
% - N: number of particles
%
% - x_curr,vel_curr: new proposed particles for the current time
%
% Loris Bazzani
% loris.bazzani@univr.it

cSum    =   cumsum(pdf_prec);
x_curr = x_prec;
for n = 1:N
    ind         = pick_base_sample(1,N,cSum);  % sampling
    
    x_curr(n,1) = x_prec(ind,1) + vel_prec(ind,1) + normrnd(0,sigma); % dinamics
    x_curr(n,2) = x_prec(ind,2) + vel_prec(ind,2) + normrnd(0,sigma);
    vel_curr(n,1) = vel_prec(ind,1) + normrnd(0,sigma_v); % dinamics
    vel_curr(n,2) = vel_prec(ind,2) + normrnd(0,sigma_v);
end