%%Computes the Rbf value Input:(DataPoint,µ,∑) Output:Scalar Value
function value = rbf(Input,Mu,Var)
value = exp(-(0.5)*(Input-Mu)*((Var)\(Input-Mu)'));
end