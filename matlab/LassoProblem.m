classdef LassoProblem
    properties
        x % data
        eta % noise standard deviation
        A % measurement matrix
        y % measured vector
        method % convex recovery algorithm
        parms % parameters for method
    end
    methods
        function self = LassoProblem(x, A, eta, y, method, parms)
            self.x = x;
            self.A = A;
            self.eta = eta;
            self.y = y;
            self.method = method;
            self.parms = parms;
        end
    end
end