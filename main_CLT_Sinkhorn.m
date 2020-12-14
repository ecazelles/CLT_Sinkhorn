%% Set global parameters
clear all 
close all
addpath('tools')

%% Parameters

% Size of the image
n_row = 20; % Row
n_col = 20; % Column

grid_lambda = 10; %[1 10 100];

% Choose between Sinkhorn divergence or Sinkhorn loss
%choice_Sink = 1; % for Sinkhorn divergence
choice_Sink = 2; % for Sinkhorn loss

% Number of samples
choice_sample{1} = 'one_sample';
choice_sample{2} = 'two_samples';
which_sample = 1; %which_sample = [1 2];

% Choose the hypothesis
% Choose the null cass ie a = b (hypothesis=0) or the alternative one is a!=b (hypothesis=1)
choice_hyp = 1; %choice_hyp = [0 1];

% Choice of the p-Wasserstein metric
p = 2;

% Number of points in the support
d = n_row*n_col;

% Cost matrix
[Y,Z] = meshgrid(1:n_col,1:n_row);
Vect_Y=reshape(Y, [d 1]);
Vect_Z=reshape(Z, [d 1]);

Y=repmat(Vect_Y, [1 d]);
Z=repmat(Vect_Z, [1 d]);    
C=(abs((Y-Y')).^p + abs((Z-Z')).^p);

% Choice of the distribution a and b
a=ones(d,1); a=a/sum(a);
slope = 0.5;
b=1+slope*(1:d)'; b=b/sum(b);

% Multinomial covariance matrix
covM=@(a)-a*a'+diag(a);



for  N= [1000]; % Number of observations of a for each empirical measure
    fprintf('N = %d\n',N)
    
    M = N; % Number of observations of a for each empirical measure

    L = 1000; %  Number of empirical measure in order to visualize the convergence in law
    L_Boot = 1000; % Number of Bootstrap repetitions


    for (nb_sample = which_sample)
    
        sample = choice_sample{nb_sample};
        fprintf('sample = %s\n',sample)

    % Case considered
    % Choose the null cass ie a = b (hypothesis=0) or the alternative one is a!=b (hypothesis=1)
        for (hypothesis = choice_hyp)
            fprintf('hypothesis = %d\n',hypothesis)
            
            % Set lambda
            for (lambda= grid_lambda)
                fprintf('lambda = %d\n',lambda)

                sinkhorn_algo='normal'; %choose between 'log, 'acc', 'normal'
                % set 'log' for small values of lambda and/or hypothesis 0 otherwise 'acc' is faster


                %% Initialization of variables


                % the matrix to be scaled.
                K = exp(-C/lambda); % in practical situations it might be a good idea to do the following: K(K<1e-100)=1e-100;
              
                % pre-compute matrix U, the Schur product of K and M.
                U=K.*C;
            
                % Matrices of the divergences
                DN=zeros(1,L);


%% GO!

switch sample
    case 'one_sample'
        if hypothesis==0
            b=a;
        end
 
        AN=zeros(size(a,1),L);
        for h=1:L
          AN(:,h)=mnrnd(N,a)'/N;
        end
        B=repmat(b, [1 L]);
        
        if choice_Sink==1
        
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[a AN],[b B],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);
            
            T = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));T(T<=0)=1;
            D=OTN(1)+sum(T(:).*log(T(:)))*lambda;
            
            for h=1:L
                TN = bsxfun(@times,m(:,h+1)',(bsxfun(@times,l(:,h+1),K)));TN(TN<=0)=1;
                DN(h)=OTN(h+1)+sum(TN(:).*log(TN(:)))*lambda;
            end
            
            % Empirical distribution of W(aN,b)
            L_law=sqrt(N)*(DN-D);
            
            % Limit distribution
            R_var = theta(:,1)'*covM(a)*theta(:,1);

            
        else
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[a a b AN AN],[b a b B AN],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);
            
            Tab = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));Tab(Tab<=0)=1;
            D_ab=OTN(1)+sum(Tab(:).*log(Tab(:)))*lambda;
            Taa = bsxfun(@times,m(:,2)',(bsxfun(@times,l(:,2),K)));Taa(Taa<=0)=1;
            D_aa=OTN(2)+sum(Taa(:).*log(Taa(:)))*lambda;
            Tbb = bsxfun(@times,m(:,3)',(bsxfun(@times,l(:,3),K)));Tbb(Tbb<=0)=1;
            D_bb=OTN(3)+sum(Tbb(:).*log(Tbb(:)))*lambda;
             
            for h=1:L
                T_AN_B = bsxfun(@times,m(:,3+h)',(bsxfun(@times,l(:,3+h),K)));T_AN_B(T_AN_B<=0)=1;
                T_AN_AN=bsxfun(@times,m(:,3+h+L)',(bsxfun(@times,l(:,3+h+L),K)));T_AN_AN(T_AN_AN<=0)=1;
                DN(h)=OTN(3+h)+sum(T_AN_B(:).*log(T_AN_B(:)))*lambda-0.5*(OTN(3+h+L)+sum(T_AN_AN(:).*log(T_AN_AN(:)))*lambda+D_bb);
            end
   
            
            if hypothesis==0
                D=0;
                theta_loss=0.5*(theta(:,1)-beta(:,1));
                else
                D=D_ab-0.5*(D_aa+D_bb);
                theta_loss=theta(:,1)-0.5*(theta(:,2)+beta(:,2));
            end

            % Empirical distribution of W(aN,b)
            L_law=sqrt(N)*(DN-D);

            % Limite distribution
            R_var=0.5*theta(:,1)'*covM(a)*theta(:,1)-0.5*(beta(:,1)'*covM(a)*beta(:,2));
                
        end
        
            
     
            
    case 'two_samples'
        rho=sqrt((N*M)/(N+M));
        gamma=M/(M+N);
        
        if hypothesis==0
            b=a;
        end

        % We concatenate a and AN (and b and BM) for the sake of order of the stooping criterion
        AN=zeros(size(a,1),L);
        BM=zeros(size(b,1),L);
        for h=1:L
          AN(:,h)=mnrnd(N,a)'/N;
          BM(:,h)=mnrnd(M,b)'/M;
        end
        
        if choice_Sink==1

            B=repmat(b, [1 L]);
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[a AN],[b BM],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            T =bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));T(T<=0)=1;
            D=OTN(1)+sum(T(:).*log(T(:)))*lambda;

            for h=1:L
                T = bsxfun(@times,m(:,h+1)',(bsxfun(@times,l(:,h+1),K)));
                T(T<=0)=1;
                DN(h)=OTN(h+1)+sum(T(:).*log(T(:)))*lambda;
            end

              L_law=rho*(DN-D);

            if hypothesis==0
                R_var = theta(:,1)'*covM(a)*theta(:,1);
            else
                R_var = gamma*theta(:,1)'*covM(a)*theta(:,1) + (1-gamma)*beta(:,1)'*covM(b)*beta(:,1);
            end

        
            
        else
            
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[a a b AN AN BM],[b a b BM AN BM],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            Tab = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));Tab(Tab<=0)=1;
            D_ab=OTN(1)+sum(Tab(:).*log(Tab(:)))*lambda;
            Taa = bsxfun(@times,m(:,2)',(bsxfun(@times,l(:,2),K)));Taa(Taa<=0)=1;
            D_aa=OTN(2)+sum(Taa(:).*log(Taa(:)))*lambda;
            Tbb = bsxfun(@times,m(:,3)',(bsxfun(@times,l(:,3),K)));Tbb(Tbb<=0)=1;
            D_bb=OTN(3)+sum(Tbb(:).*log(Tbb(:)))*lambda;
             
            for h=1:L
                T_AN_B = bsxfun(@times,m(:,3+h)',(bsxfun(@times,l(:,3+h),K)));T_AN_B(T_AN_B<=0)=1;
                T_AN_AN=bsxfun(@times,m(:,3+h+L)',(bsxfun(@times,l(:,3+h+L),K)));T_AN_AN(T_AN_AN<=0)=1;
                T_BM_BM=bsxfun(@times,m(:,3+h+2*L)',(bsxfun(@times,l(:,3+h+2*L),K)));T_BM_BM(T_BM_BM<=0)=1;
                DN(h)=OTN(3+h)+sum(T_AN_B(:).*log(T_AN_B(:)))*lambda-0.5*(OTN(3+h+L)+sum(T_AN_AN(:).*log(T_AN_AN(:)))*lambda+OTN(3+h+2*L)+sum(T_BM_BM(:).*log(T_BM_BM(:)))*lambda);
            end

            if hypothesis==0
                D=0;
                else
                D=D_ab-0.5*(D_aa+D_bb);
            end

            % Distribution empirique de W(aN,b)
            L_law=rho*(DN-D);

            
            if hypothesis==0
                theta_loss=0.5.*(theta(:,1)-beta(:,1));
                R_var = theta_loss'*covM(a)*theta_loss;
            else
                theta_loss_a=theta(:,1)-0.5.*(theta(:,2)+beta(:,2));
                theta_loss_b=beta(:,1)-0.5.*(theta(:,3)+beta(:,3));
                R_var = gamma*theta_loss_a'*covM(a)*theta_loss_a + (1-gamma)*theta_loss_b'*covM(b)*theta_loss_b;
            end
        
            
        end
            
           

end
 

%%%%%%%%%%%%%%%%%%%%%%%%
%% BOOTSTRAP EXPERIMENTS

switch sample
    case 'one_sample'

        % Observed sample
        aN_basic = mnrnd(N,a)'/N;

        % Computation of W(aN_basic,b)
        if choice_Sink==1
                    
            [OTN,lowerEMB,l,m,theta,beta] = choose_sinkhorn(sinkhorn_algo,aN_basic,b,C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);
            T = bsxfun(@times,m',(bsxfun(@times,l,K)));T(T<=0)=1;
            DN_basic =OTN+sum(T(:).*log(T(:)))*lambda;

        else
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[aN_basic aN_basic b],[b aN_basic,b],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);
            TaNb = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));TaNb(TaNb<=0)=1;
            TaNaN = bsxfun(@times,m(:,2)',(bsxfun(@times,l(:,2),K)));TaNaN(TaNaN<=0)=1;
            Tbb = bsxfun(@times,m(:,3)',(bsxfun(@times,l(:,3),K)));Tbb(Tbb<=0)=1;
            Dbb =OTN(3)+sum(Tbb(:).*log(Tbb(:)))*lambda;
            DN_basic =OTN(1)+sum(TaNb(:).*log(TaNb(:)))*lambda-0.5*(OTN(2)+sum(TaNaN(:).*log(TaNaN(:)))*lambda+Dbb);
        end
        
        % Bootstrap
        DN_Boot=zeros(1,L_Boot);
        
        AN_Boot=zeros(size(a,1),L);
        for h=1:L_Boot
          AN_Boot(:,h)=mnrnd(N,aN_basic)'/N;
        end
        B=repmat(b, [1 L_Boot]);
        
        if choice_Sink==1
            
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,AN_Boot,B,C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            for h=1:L
                T = bsxfun(@times,m(:,h)',(bsxfun(@times,l(:,h),K)));T(T<=0)=1;
                DN_Boot(h)=OTN(h)+sum(T(:).*log(T(:)))*lambda;
            end
            
        else
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[AN_Boot AN_Boot],[B AN_Boot],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            for h=1:L
                TaNB = bsxfun(@times,m(:,h)',(bsxfun(@times,l(:,h),K)));TaNB(TaNB<=0)=1;
                TaNaN = bsxfun(@times,m(:,h+L)',(bsxfun(@times,l(:,h+L),K)));TaNaN(TaNaN<=0)=1;
                DN_Boot(h)=OTN(h)+sum(TaNB(:).*log(TaNB(:)))*lambda-0.5*(OTN(h+L)+sum(TaNaN(:).*log(TaNaN(:)))*lambda+Dbb);
            end
            
        end
            
       
        % Empirical measure by Bootstrap
        L_law_Boot=sqrt(N)*(DN_Boot-DN_basic);

    case 'two_samples'

        % Observed samples
        aN_basic = mnrnd(N,a)'/N;
        bM_basic = mnrnd(M,b)'/M;

        % Bootstrap
        DN_Boot=zeros(1,L_Boot);


        % We concatenate aN_basic and AN_Boot for the sake of order of the stooping criterion
        AN_Boot=zeros(size(a,1),L);
        BM_Boot=zeros(size(b,1),L);
        for h=1:L_Boot
          AN_Boot(:,h)=mnrnd(N,aN_basic)'/N;
          BM_Boot(:,h)=mnrnd(M,bM_basic)'/M;
        end
        
        if choice_Sink==1
        
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[aN_basic AN_Boot],[bM_basic BM_Boot],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            T = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));T(T<=0)=1;
            DN_basic =OTN(1)+sum(T(:).*log(T(:)))*lambda;

            for h=1:L_Boot
                T = bsxfun(@times,m(:,h+1)',(bsxfun(@times,l(:,h+1),K)));T(T<=0)=1;
                DN_Boot(h)=OTN(h+1)+sum(T(:).*log(T(:)))*lambda;
            end
            
        else
            
            [OTN,lowerEMB,l,m,theta,beta]=choose_sinkhorn(sinkhorn_algo,[aN_basic aN_basic bM_basic AN_Boot AN_Boot BM_Boot],[bM_basic aN_basic bM_basic BM_Boot AN_Boot BM_Boot],C,U,lambda,'distanceRelativeDecrease',inf,1e-7,1e7,0);

            T1 = bsxfun(@times,m(:,1)',(bsxfun(@times,l(:,1),K)));T1(T1<=0)=1;
            T2 = bsxfun(@times,m(:,2)',(bsxfun(@times,l(:,2),K)));T2(T2<=0)=1;
            T3 = bsxfun(@times,m(:,3)',(bsxfun(@times,l(:,3),K)));T3(T3<=0)=1;
            DN_basic =OTN(1)+sum(T1(:).*log(T1(:)))*lambda-0.5*(OTN(2)+sum(T2(:).*log(T2(:)))*lambda+OTN(3)+sum(T3(:).*log(T3(:)))*lambda);

            for h=1:L_Boot
                T1 = bsxfun(@times,m(:,h+3)',(bsxfun(@times,l(:,h+3),K)));T1(T1<=0)=1;
                T2 = bsxfun(@times,m(:,h+3+L)',(bsxfun(@times,l(:,h+3+L),K)));T2(T2<=0)=1;
                T3 = bsxfun(@times,m(:,h+3+2*L)',(bsxfun(@times,l(:,h+3+2*L),K)));T3(T3<=0)=1;
                DN_Boot(h)=OTN(h+3)+sum(T1(:).*log(T1(:)))*lambda-0.5*(OTN(h+3+L)+sum(T2(:).*log(T2(:)))*lambda+OTN(h+3+2*L)+sum(T3(:).*log(T3(:)))*lambda);
                        

            
            end
        end

        % Empirical measure by Bootstrap
        L_law_Boot=rho*(DN_Boot-DN_basic);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display of the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mini=min([min(L_law),-5*sqrt(R_var),min(L_law_Boot)]);
maxi=max([max(L_law),5*sqrt(R_var),max(L_law_Boot)]);

nb_bin=1000;
dt=(maxi-mini)/nb_bin;
nn=mini:dt:maxi;

aa=ksdensity(L_law,nn);
%% bb=ksdensity(R_law,nn);
bb = normpdf(nn,0,sqrt(R_var));

clear theta
figure;hold on;
fill(nn,bb,'b');plot(nn,bb,'b','linewidth',2);
fill(nn,aa,'r');plot(nn,aa,'r','linewidth',2);
alpha 0.2
xlim([nn(1) nn(end)])

aa_Boot=ksdensity(L_law_Boot,nn);

figure;hold on;
fill(nn,bb,'b');plot(nn,bb,'b','linewidth',2)
fill(nn,aa,'r');plot(nn,aa,'r','linewidth',2);
fill(nn,aa_Boot,'g');plot(nn,aa_Boot,'g','linewidth',2);
alpha 0.2
xlim([nn(1) nn(end)])

ylim = get(gca,'YLim');
switch sample
    case 'one_sample'
    plot(sqrt(N)*(DN_basic-D)*[1,1],ylim*1.05,'r-','LineWidth',2);
    case 'two_samples'
    plot(rho*(DN_basic-D)*[1,1],ylim*1.05,'r-','LineWidth',2);
end

% Confidence intervals
bootstrapCI = prctile(L_law_Boot,[2.5 97.5]);
plot(bootstrapCI(1)*[1,1],ylim*1.05,'g-','LineWidth',2);
plot(bootstrapCI(2)*[1,1],ylim*1.05,'g-','LineWidth',2);

%close all

end

end

end

end
