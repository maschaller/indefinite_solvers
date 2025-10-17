function [x,j]=widlund(A,H,S,b,n,tol)

nb = sqrt(b'*H*b);
u = b/nb;
U(:,1) = u;

maxit = min(n);

alpha = [];
beta = [];
T = [];
TT = [];
x= 0*b;
for j = 1:maxit
    if mod(j,10)==0
     %   disp(j)
    end
    if isa (A,'function_handle')
        U(:,j+1) = A(U(:,j));
    else
        U(:,j+1) = A*U(:,j);
    end
    alpha(j) = U(:,j+1)'*(H*U(:,j));
    if j == 1
        U(:,j+1) = U(:,j+1)-alpha(j)*U(:,j);
    else
        U(:,j+1) = U(:,j+1)-alpha(j)*U(:,j)+beta(j-1)*U(:,j-1);
    end
    beta(j) = sqrt((U(:,j+1))'*(H*U(:,j+1)));
    U(:,j+1) = U(:,j+1)/beta(j);    
    T(j,j+1) = -beta(j);
    T(j+1,j) = beta(j);
    % T(1:j,1:j) = U(:,1:j)'*S*U(:,1:j);

    % [EV,EW] = eig(T(1:j,1:j));
    e = zeros(j,1); e(1)=1;
    y = (speye(j)+T(1:j,1:j))\(nb*e);

    if isa (A,'function_handle')
        aux = U(:,1:j)*y+A(U(:,1:j)*y);
    else
        aux =U(:,1:j)*y+A*(U(:,1:j)*y);
    end
    if norm(aux-b)/norm(b)<tol
        x=U(:,1:j)*y;
        break;
    end
    % A*U(:,1:j)-U(:,1:j+1)*T(1:j+1,1:j)
end
% U(:,1:n)'*H*U(:,1:n)