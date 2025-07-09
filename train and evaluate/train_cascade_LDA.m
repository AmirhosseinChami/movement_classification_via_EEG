function cascade_model = train_cascade_LDA(X, Y)
    cascade_model = {};

    % class 4 vs rest
    Y1 = (Y == 4); 
    model4 = fitcdiscr(X, Y1, 'DiscrimType', 'diagLinear');
    cascade_model{end+1} = struct('model', model4, 'class', 4);

    % class 3 vs 2 and 1 
    idx = Y ~= 4;           
    X2 = X(idx, :);
    Y2 = Y(idx) == 3;      
    model3 = fitcdiscr(X2, Y2, 'DiscrimType', 'diagLinear');
    cascade_model{end+1} = struct('model', model3, 'class', 3);

    % class 2 vs class 1
    idx = Y == 1 | Y == 2;
    X3 = X(idx, :);
    Y3 = Y(idx) == 2;      
    model2 = fitcdiscr(X3, Y3, 'DiscrimType', 'diagLinear');
    cascade_model{end+1} = struct('model', model2, 'class', 2);
end
