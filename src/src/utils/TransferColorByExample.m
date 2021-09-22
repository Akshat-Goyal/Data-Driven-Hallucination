function result=TransferColorByExample(target, example, space)
if max(max(target)) > 2.0
    flag='255MAX';
else
    flag='1.0MAX';
end
target = RGB2Lab(target);
example = RGB2Lab(example);
target_1 = reshape( target, size(target,1)*size(target,2),3);
example = reshape( example, size(example,1)*size(example,2),3);
MEAN_U = mean(target_1);
MEAN_V = mean(example);
T = diag( (diag(cov(example)).^0.5)./ (diag(cov(target_1)).^0.5));
result = (target_1 - repmat( MEAN_U , length(target_1),1) )*T' + repmat( MEAN_V , length(target_1),1)  ;
result = reshape( result , size(target,1) , size(target,2), 3 ) ;
result = Lab2RGB(result);
if strcmp(flag,'255MAX');
    result=result*255;
end
end
