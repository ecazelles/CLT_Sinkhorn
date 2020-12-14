function [OT,lower,l,m,alpha,beta]=choose_sinkhorn(sinkhorn_algo,a,b,C,U,lambda,crit,norme,tol,iter,VERBOSE);

% Choose the version of the algorithm to launch

switch(sinkhorn_algo)
    case 'log'
        [OT,lower,l,m,alpha,beta]=sinkhornTransport_log_stabilization(a,b,C,U,lambda,crit,norme,tol,iter,VERBOSE);
    case 'acc'
        [OT,lower,l,m,alpha,beta]=sinkhornTransport_acc(a,b,C,U,lambda,crit,norme,tol,iter,VERBOSE);
    otherwise
        [OT,lower,l,m,alpha,beta]=sinkhornTransport(a,b,C,U,lambda,crit,norme,tol,iter,VERBOSE);
end


        