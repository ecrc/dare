// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

clear all
close all

load ../Feb21_DARE_matrices.mat
gamma_hist=fitsread('../Feb21_gamma_hist.fits');

%n=10;
%N=5;
%mavis=25;
maxit=100;

tol=1e-15;

% Non-symmetric matrices
%A=rand(n*N);
%B=rand(n*N,mavis);

% Symmetric matrix
%Q=rand(n*N); Q=Q+Q';

% Diagonal matrix
%R=diag(rand(mavis,1));

% Non-symmetric matrices
alpha = A;

% Diagonal matrix
sr_invR=zeros(size(R,1),1);
newB=zeros(size(A,1), size(R,1));
for i=1:size(R,1)
    sr_invR(i)=sqrt(1/R(i,i));
    newB(:,i)=B(:,i)*sr_invR(i); % xscal
end

fitswrite(newB','BinvRt.fits')


% Symmetric matrices
%beta  = B*invR*B';
beta  = newB*newB'; % xsyrk
gamma = Q; % xlacpy

%Id    = eye(size(A,1));
%old   = gamma;

for it=1:maxit
    % Non-symmetric matrices
    %common = Id + beta*gamma;
    common = beta*gamma; % xgemm
    for i=1:size(common,1)
        common(i,i)=common(i,i)+1; % use routine for making matrix diag dominant
    end

    %inv_common = inv(common);
    %inv_commonxalpha = inv_common*alpha;
    [L U P] = lu(common); % xgetrf
    y = L\(P*alpha); % xgetrs: fwd and bwd sub
    inv_commonxalpha = U\y;

    %tmp = alpha'*gamma*inv_commonxalpha;
    tmp   = alpha'*gamma; % xgemm
    gamma = gamma + tmp*inv_commonxalpha; % xgemm

    if (it <= 15)
       diff_bis = norm(gamma-gamma_hist(:,:,it)');
    else
       diff_bis = -1;
    end

    diff = norm(tmp, 'fro');
    it
    diff
    tol
    % diff_bis
    if (diff < tol)
       break
    end

    % beta  = beta + alpha*inv_common*beta*alpha';
    y = L\(P*beta); % xgetrs: fwd and bwd sub
    inv_commonxbeta = U\y;
    tmp = inv_commonxbeta*alpha'; % xgemm
    beta  = beta + alpha*tmp; % xgemm

    alpha = alpha*inv_commonxalpha; % xgemm
    alpha(1:5,1:5)
    save('alpha.mat','alpha')
    %return

    %norm(gamma-A'*gamma*A+(A'*gamma*B)*inv(R+B'*gamma*B)*(B'*gamma*A)-Q)
end

%newK = inv(R+B'*gamma*B)*B'*gamma*A;
tmp = B'*gamma; % xgemm
newK = inv(R+tmp*B)*tmp*A;

norm(newK-K,'fro')
