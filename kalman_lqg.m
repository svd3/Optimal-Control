%[K,L,Cost,Xa,XSim,CostSim] = 
%   kalman_lqg( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1  [NSim,Init,Niter] )
%
% Compute optimal controller and estimator for generalized LQG
%
% u(t)    = -L(t) x(t)
% x(t+1)  = A x(t) + B (I + Sum(C(i) rnd_1)) u(t) + C0 rnd_n
% y(t)    = H x(t) + Sum(D(i) rnd_1) x(t) + D0 rnd_n
% xhat(t+1) = A xhat(t) + B u(t) + K(t) (y(t) - H xhat(t)) + E0 rnd_n
% x(1)    ~ mean X1, covariance S1
%
% cost(t) = u(t)' R u(t) + x(t)' Q(t) x(t)
%
% NSim    number of simulated trajectories (default 0)  (optional)
% Init    0 - open loop; 1 (default) - LQG; 2 - random  (optional)
% Niter   iterations; 0 (default) - until convergence   (optional)
%
% K       Filter gains
% L       Control gains
% Cost    Expected cost (per iteration)
% Xa      Expected trajectory
% XSim    Simulated trajectories
% CostSim Empirical cost
%
% This is an implementation of the algorithm described in:
%  Todorov, E. (2005) Stochastic optimal control and estimation
%  methods adapted to the noise characteristics of the
%  sensorimotor system. Neural Computation 17(5): 1084-1108
% The paper is available online at www.cogsci.ucsd.edu/~todorov

% Copyright (C) Emanuel Todorov, 2004-2006


function [K,L,Cost,Xa,XSim,CostSim,iter] = ...
   kalman_lqg( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1, NSim,Init,Niter )


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
format long;

% numerical parameters
MaxIter = 500;
Eps = 10^-15;

% determine sizes
szX = size(A,1);
szU = size(B,2);
szY = size(H,1);
szC = size(C,3);
szC0 = size(C0,2);
szD = size(D,3);
szD0 = size(D0,2);
szE0 = size(E0,2);
N = size(Q,3);

% initialize missing optional parameters
if nargin<13,
   NSim = 0;
end;
if nargin<14,
   Init = 1;
end;
if nargin<15,
   Niter = 0;
end;

% if C or D are scalar, replicate them into vectors
if size(C,1)==1 & szU>1,
   C = C*ones(szU,1);
end;
if length(D(:))==1,
    if D(1)==0,
        D = zeros(szY,szX);
    else
        D = D*ones(szX,1);
        if szX ~= szY,
            error('D can only be a scalar when szX = szY');
        end
    end
end;

% if C0,D0,E0 are scalar, set them to 0 matrices and adjust size
if length(C0(:))==1 & C0(1)==0,
    C0 = zeros(szX,1);
end
if length(D0(:))==1 & D0(1)==0,
    D0 = zeros(szY,1);
end
if length(E0(:))==1 & E0(1)==0,
    E0 = zeros(szX,1);
end


% initialize policy and filter
K = zeros(szX,szY,N-1);
L = zeros(szU,szX,N-1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run iterative algorithm - until convergence or MaxIter

for iter = 1:MaxIter
   
   % initialize covariances
   SiE = S1;
   SiX = X1*X1';
   SiXE = zeros(szX,szX);
   
   % forward pass - recompute Kalman filter   
   for k = 1:N-1
      
      % compute Kalman gain
      temp = SiE + SiX + SiXE + SiXE';
      if size(D,2)==1,
          DSiD = diag(diag(temp).*D.^2);
      else
          DSiD = zeros(szY,szY);
          for i=1:szD
              DSiD = DSiD + D(:,:,i)*temp*D(:,:,i)';
          end;
      end;
      K(:,:,k) = A*SiE*H'*pinv(H*SiE*H'+D0*D0'+DSiD);
      
      % compute new SiE
      newE = E0*E0' + C0*C0' + (A-K(:,:,k)*H)*SiE*A';
      LSiL = L(:,:,k)*SiX*L(:,:,k)';
      if size(C,2)==1,
         newE = newE + B*diag(diag(LSiL).*C.^2)*B';
      else
         for i=1:szC
            newE = newE + B*C(:,:,i)*LSiL*C(:,:,i)'*B';
         end;
      end;
      
      % update SiX, SiE, SiXE
      SiX = E0*E0' + K(:,:,k)*H*SiE*A' + (A-B*L(:,:,k))*SiX*(A-B*L(:,:,k))' + ...
          (A-B*L(:,:,k))*SiXE*H'*K(:,:,k)' + K(:,:,k)*H*SiXE'*(A-B*L(:,:,k))';
      SiE = newE;
      SiXE = (A-B*L(:,:,k))*SiXE*(A-K(:,:,k)*H)' - E0*E0';
   end;
   
   
   % first pass initialization
   if iter==1,
      if Init==0,         % open loop
         K = zeros(szX,szY,N-1); 
      elseif Init==2,     % random
         K = randn(szX,szY,N-1);
      end;
   end;
   
   
   % initialize optimal cost-to-go function
   Sx = Q(:,:,N);
   Se = zeros(szX,szX);
   Cost(iter) = 0;
   
   % backward pass - recompute control policy
   for k=N-1:-1:1
      
      % update Cost
      Cost(iter) = Cost(iter) + trace(Sx*C0*C0') + ...
         trace(Se*(K(:,:,k)*D0*D0'*K(:,:,k)' + E0*E0' + C0*C0'));
      
      % Controller
      temp = R + B'*Sx*B;
      BSxeB = B'*(Sx+Se)*B;
      if size(C,2)==1,
         temp = temp + diag(diag(BSxeB).*C.^2);
      else
         for i=1:size(C,3)
            temp = temp + C(:,:,i)'*BSxeB*C(:,:,i);
         end;
      end;
      L(:,:,k) = pinv(temp)*B'*Sx*A;
      
      % compute new Se
      newE = A'*Sx*B*L(:,:,k) + (A-K(:,:,k)*H)'*Se*(A-K(:,:,k)*H);
      
      % update Sx and Se
      Sx = Q(:,:,k) + A'*Sx*(A-B*L(:,:,k));
      KSeK = K(:,:,k)'*Se*K(:,:,k);
      if size(D,2)==1,
          Sx = Sx + diag(diag(KSeK).*D.^2);
      else
          for i=1:szD
              Sx = Sx + D(:,:,i)'*KSeK*D(:,:,i);
          end;
      end;     
      Se = newE;
   end;
   
   
   % adjust cost
   Cost(iter) = Cost(iter) + X1'*Sx*X1 + trace((Se+Sx)*S1);
   
   % progress bar
   if ~rem(iter,10),
      fprintf('.');
   end;
   
   % check convergence of Cost
   if (Niter>0 & iter>=Niter) | ...
      (Niter==0 & iter>1 & abs(Cost(iter-1)-Cost(iter))<Eps) | ...
      (Niter==0 & iter>20 & sum(diff(dist(iter-10:iter))>0)>3),
      break;
   end;
end;

% print result
fprintf('\n');
fprintf('iterations %d\n', iter);
fprintf('\n%.15f\n', Cost(iter-1));
fprintf('%.15f\n', Cost(iter));
if Cost(iter-1)~=Cost(iter)
   fprintf(' Log10DeltaCost = %.2f\n',log10(abs(Cost(iter-1)-Cost(iter))));
else
   fprintf(' DeltaCost = 0\n' );
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute average trajectory

Xa = zeros(szX,N);
Xa(:,1) = X1;

for k=1:N-1
   u = -L(:,:,k)*Xa(:,k);
   Xa(:,k+1) = A*Xa(:,k) + B*u;
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulate noisy trajectories

if NSim > 0,
   
   % square root of S1
   [u,s,v] = svd(S1);
   sqrtS = u*diag(sqrt(diag(s)))*v';
   
   % initialize
   XSim = zeros(szX,NSim,N);
   Xhat = zeros(szX,NSim,N);
   Xhat(:,:,1) = repmat(X1, [1 NSim]);
   XSim(:,:,1) = repmat(X1, [1 NSim]) + sqrtS*randn(szX,NSim);
   
   CostSim = 0;
   
   % loop over N
   for k=1:N-1
      
      % update control and cost
      U = -L(:,:,k)*Xhat(:,:,k);
      CostSim = CostSim + sum(sum(U.*(R*U)));
      CostSim = CostSim + sum(sum(XSim(:,:,k).*(Q(:,:,k)*XSim(:,:,k))));
      
      % compute noisy control
      Un = U;
      if size(C,2)==1,
         Un = Un + U.*randn(szU,NSim).*repmat(C,[1,NSim]);
      else
         for i=1:szC
            Un = Un + (C(:,:,i)*U).*repmat(randn(1,NSim),[szU 1]);
         end;
      end;
      
      % compute noisy observation
      y = H*XSim(:,:,k) + D0*randn(szD0,NSim);
      if size(D,2)==1,
         y = y + XSim(:,:,k).*randn(szY,NSim).*repmat(D,[1,NSim]);
      else
         for i=1:szD
            y = y + (D(:,:,i)*XSim(:,:,k)).*repmat(randn(1,NSim),[szY 1]);
         end;
      end;
      
      XSim(:,:,k+1) = A*XSim(:,:,k) + B*Un + C0*randn(szC0,NSim);
      Xhat(:,:,k+1) = A*Xhat(:,:,k) + B*U + K(:,:,k)*(y-H*Xhat(:,:,k)) + ...
          E0*randn(szE0,NSim);
   end;
   
   % final cost update
   CostSim = CostSim + sum(sum(XSim(:,:,N).*(Q(:,:,N)*XSim(:,:,N))));
   CostSim = CostSim / NSim;
   
else
   XSim = [];
   CostSim = [];
end;
