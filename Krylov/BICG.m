function xold = BICG(A, xold, b, tol)
% Bi-conjugate gradient
r = b - A*xold;
rtilde = r;

% fprintf('   RHS (b) and x0: \n');
% for i=1:N
%     fprintf('   N: %d, b: %+6.10e, x0: %+6.10e, CPT_E: %+6.10e, ECV: %+6.10e \n',i, RightHandSide(i), xold(i), ExpliciteCouplingTerm(i), ExplicitCouplingValues(i));
% end
% 
% fprintf('   Initial residuals: \n');
% for i=1:N
%     fprintf('   N: %d, r: %+6.10e, rtilde: %+6.10e \n',i, r(i), rtilde(i));
% end

for i=1:20
%     z = r;
%     ztilde = rtilde;
    
%     rho1 = dot(z, rtilde);
    rho1 = dot(r, rtilde);

%     if i==1
%         r
%         rtilde
%         rho1
%     end
    
    if i==1
%         p = z;
%         ptilde = ztilde;
        p = r;
        ptilde = rtilde;
    else
        beta = rho1/rho2;
%         p = z +beta*p;
%         ptilde = ztilde + beta*ptilde;
        p = r +beta*p;
        ptilde = rtilde + beta*ptilde;
    end
    q = A*p;
    qtilde = transpose(A)*ptilde;
    alpha = rho1 / dot(ptilde,q);
    xold = xold + alpha*p;
    r = r - alpha*q;
    rtilde = rtilde - alpha*qtilde;
    rho2=rho1;
    
    if tol > norm(r)
        break;
    end
end

end

