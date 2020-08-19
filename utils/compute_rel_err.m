function [rel] = compute_rel_err(x,y)
rel=norm(x-y,'fro')/norm(x,'fro');
end

