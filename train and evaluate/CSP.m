function [W_csp] = CSP(X, Y, num_filters)    
    cov1 = cov(X');
    cov2 = cov(Y');
    
    % whitening
    [V, D] = eig(cov1 + cov2);
    P = sqrt(inv(D)) * V';

    S1 = P * cov1 * P';
    S2 = P * cov2 * P';

    % GEVD
    [U, D] = eig(S1, S1 + S2);
    [~, idx] = sort(diag(D), 'descend');  
    W_all = U(:, idx);  
    
    W = [W_all(:, 1:num_filters/2), W_all(:, end - num_filters/2 + 1:end)];
    
    W_csp = W' * P; % filters
end
