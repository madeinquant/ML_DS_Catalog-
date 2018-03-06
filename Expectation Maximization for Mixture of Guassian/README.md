
**Description**
The probability a point belonging to a Mixture of Guassians(each Gussian represented by a class Cj) is given by

P(x) = sum P(x,Cj)  = sum P(x/Cj)P(Cj)  


Expectation Maximization for a Mixture of Guassian attempts to find the parameters of each of Guassian Class i.e the mean vector and the Covariance matrix for each Guassian class and also the class probabilities i.e the P(Cj) s 

The parameters are determined by maximizing the likelihood or minimizing the negative loglikelihood. The negative log likelihood is not minimized by using gradient based methods but by using an Algorithm 
called expectation maximization. 

**Expectation Step**



**Maximization Step**






 



 
    


