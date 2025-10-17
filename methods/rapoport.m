function [x,j]=rapoport(A,H,S,b,n,tol)

nb = sqrt(b'*H*b);
u = b/nb;
U(:,1) = u;

maxit = 1000; min(n);

alpha = [];
beta = [];
T = [];
x=b;
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
    Ttilde = [speye(j)+T(1:j,1:j);[zeros(1,j-1) T(j+1,j)]];
    e = zeros(j+1,1);
    e(1)=1;
    y = (Ttilde'*Ttilde)\(nb*Ttilde'*e);
    ynew = Ttilde\(nb*e);

    if isa (A,'function_handle')
        aux = A(U(:,1:j)*y);
    else
        aux = A*(U(:,1:j)*y);
    end
    if norm(aux-b)/norm(b)<tol
        x=U(:,1:j)*y;
        break;
    end
end
% U(:,1:n)'*H*U(:,1:n)