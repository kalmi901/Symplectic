function LUP

A = [8, 1, 6; 
	 4, 9, 2;
	 3, 5, 7];

A = magic(6)	 
b = [1; 23; 3];


x = SolveLinearSystem(A, b)

end



function x = SolveLinearSystem(A, b)
    n = length(A);
    x = zeros(n, 1);
    y = zeros(n, 1);
    % decomposition of matrix, Doolittle’s Method
    for i = 1:1:n
        for j = 1:1:(i - 1)
            alpha = A(i,j);
            for k = 1:1:(j - 1)
                alpha = alpha - A(i,k)*A(k,j);
            end
            A(i,j) = alpha/A(j,j);
        end
        for j = i:1:n
            alpha = A(i,j);
            for k = 1:1:(i - 1)
                alpha = alpha - A(i,k)*A(k,j);
            end
            A(i,j) = alpha;
        end
    end
    %A = L+U-I
	A,
	kbhit()
    % find solution of Ly = b
    for i = 1:1:n
        alpha = 0;
        for k = 1:1:i
            alpha = alpha + A(i,k)*y(k);
        end
        y(i) = b(i) - alpha;
    end
    % find solution of Ux = y
    for i = n:(-1):1
        alpha = 0;
        for k = (i + 1):1:n
            alpha = alpha + A(i,k)*x(k);
        end
        x(i) = (y(i) - alpha)/A(i, i);
    end    
end