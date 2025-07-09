function preds = predict_cascade_LDA(cascade_model, X)
    preds = zeros(size(X,1),1);
    for i = 1:size(X,1)
        for j = 1:length(cascade_model)
            model = cascade_model{j}.model;
            cls   = cascade_model{j}.class;
            label = predict(model, X(i,:));
            if label == 1
                preds(i) = cls;
                break;
            end
        end
        if preds(i) == 0
            preds(i) = 1;  % default to class 1
        end
    end
end
