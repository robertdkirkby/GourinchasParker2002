% Gourinchas & Parker (2002) - Consumption over the Life-Cycle

% First, does 'robust' estimation with the identity matrix as the weighting matrix
% Second, then does 'efficient' estimation, with the inverse of the
% variance-covariance matrix of the moment conditions, as the weighting matrix

% Currently, I just take the data moments and the weighting matrix from some online materials.

% Alternatively, GP2002_FirstStage.m reproduces the data work from scratch
% (NOT YET COMPLETED)

% Important difference: GP2002 solve based on the FOCs (the consumption
% Euler Eqn). As a result, they do NOT need V_{J+1}, they instead just need
% C_{J+1} [think about writing out a sequence problem, and then think about 
% the FOC between periods J and J+1, all you need is C_{J+1} to iterate from.
% GP2002 write this FOC with C_{J+1} as eqn below eqn (5) on pg 55]. 
% They write out the original model with a kappa and h that determine
% V_{J+1}. But they then can essentially drop kappa, replace h with gamma0
% and gamma1, and because gamma0 and gamma1 determine C_{J+1}, they can
% solve the model without ever finding V_{J+1}. VFI Toolkit works based on
% value function iteration, and so C_{J+1} is not enough for us, we need
% V_{J+1}. We therefore need to work with kappa and h, instead of gamma0
% and gamma1. As a result, we are going to be estimating different
% parameters to what GP2002 do, but we are just estimating a different
% parametrization of the model (with kappa and h to pin down V_{J+1}) than
% what GP2002 used (with gamma0 and gamma1 to pin down C_{J+1}).
% [Both this code and GP2002 also estimate R and rho]

% Another important difference: GP2002 do 'two-iteration efficient SMM'
% (they call it two-step). Here because of the nature of the moments we can
% just directly do efficient GMM (with no need for two-iterations).
 

%% Age and model periods

N_j=40;
Params.agejshifter=25; % first period is age 26
% Age
Params.J=N_j;
Params.agej=1:1:N_j;
Params.age=Params.agejshifter+Params.agej;

% Unit-root
kirkbyoptions.nSigmas=1; % originally I used 2 % GP2002 do not discuss what this should be. If I let it be substantial, then the minimum grid point gets silly-low at higher ages 
% This is the number of standard deviations used in discertization/quadrature
% Turns out to essentially completely determine the value of rho (the risk aversion)


%% Empirical data
% From: https://github.com/ThomasHJorgensen/Sensitivity/tree/master/GP2002
% Who says: Empirical files from the orignal paper is based on the online code for the paper by Andrews et al. (2017, QJE) and can be found here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LLARSN
% The data files are:
% income.txt:          age profile of average income (in 1000s of 1987 dollars)
% sample_moments.txt:  age profile of average consumption (in 1000s of 1987 dollars)
% weight.txt:          weight matrix used to estimate the structural parameters.
% I copy-pasted the contents of these here
avgincome=[18690.960, 19163.940, 19621.110, 20064.870, 20497.210, 20919.710, 21333.540, 21739.350, 22137.360, 22527.250, 22908.310, 23279.320, 23638.630, 23984.270, 24313.830, 24624.660, 24913.860, 25178.370, 25415.050, 25620.790, 25792.540, 25927.470, 26023.090, 26077.300, 26088.440, 26055.600, 25978.380, 25857.240, 25693.430, 25488.990, 25246.820, 24970.680, 24665.150, 24335.610, 23988.180, 23629.620, 23267.470, 22909.670, 22564.940, 22242.540];
avgconsumption=[20346.439999999999, 20608.549999999999, 20553.389999999999, 20568.669999999998, 20832.849999999999, 20960.669999999998, 21566.720000000001, 21459.770000000000, 21282.630000000001, 22127.910000000000, 21929.880000000001, 21460.419999999998, 21619.970000000001, 22359.830000000002, 22107.360000000001, 23016.139999999999, 22424.619999999999, 22871.310000000001, 23250.590000000000, 23839.259999999998, 22803.639999999999, 22548.490000000002, 23354.259999999998, 22359.360000000001, 21651.770000000000, 21383.049999999999, 21787.459999999999, 21454.160000000000, 20358.779999999999, 19842.860000000001, 20311.130000000001, 20353.930000000000, 19331.619999999999, 19082.180000000000, 17613.980000000000, 19077.070000000000, 18321.340000000000, 18501.980000000000, 17788.040000000001, 18201.759999999998];
WeightingMatrix_AGS2017=diag([3.6872008,3.8049959,3.5500436,3.7745263,4.2091920,4.1346511,3.8382528,3.7693568,4.0865395,3.8544915,3.9664480,3.8415740,4.0701883,3.7211355,3.7487563,3.7335098,3.7867473,3.7138973,3.9722601,3.2203741,3.6383268,3.3823610, 3.7156815,3.3014884,3.1814975,3.7363199,3.7048994, 3.7809048, 3.6250560, 3.2918133, 3.5717417,3.9574252,3.4803905,3.1553934,3.9773247,3.2377166,3.4513555,3.5322114,3.5292536, 3.4207858]);
% Note: I have double-checked these against contents of ./derived/Transparent Identification (Gourinchas and Parker Replication)/output/make.log
%       from the materials of https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LLARSN
% Note: This later source got the following as the fitted (estimated) moments for consumption:
fittedavgconsumption_AGS2017=[18675.330192338559, 19388.212600096318, 19931.409678033669, 20416.983059913120, 20846.511902463310, 21249.797274646731, 21568.638059828245, 21960.326011461413, 22250.073312477685, 22572.663315759091, 22762.477900147911, 22965.855194085474, 23083.242442513980, 23172.036295533137, 23106.696851993544, 23010.418890175795, 22932.704122827741, 22774.784936048960, 22610.201008821856, 22425.937563838477, 22235.859576262708, 22038.096024257269, 21875.939447598190, 21683.238568007921, 21469.783149753606, 21258.301954897222, 21061.824954368018, 20832.183568772063, 20577.220515166311, 20348.595988976027, 20110.078731585425, 19875.656107118601, 19617.792926036545, 19368.268110153349, 19120.868715666104, 18842.115913731086, 18605.273565004958, 18327.670847333808, 18072.160850591466, 17806.156351514812];
% Note: these are the fitted moments for AGS2017 estimates:
% beta	0.9574
% rho	0.6526024
% x0	0.061146601185
% gamma0	0.0015
% gamma1	0.071
% From AGS2017: /derived/Transparent Identification (Gourinchas and Parker Replication)/code/gp_procedures.gau
% Is the original code of GP2002 for solving the model.

% Note: GP2002 say WeightingMatrix is diagonal matrix with elements being the inverse of the variance of the moments,
% this would give variance as around 0.3, so presumably is already for log(C)

% Note: average consumption is the raw data (can tell from Fig 2, shown just below)
% GP2002 estimate based on a smooth profile
% GP2002, pg 86: "Smooth profiles are constructed by estimating an equation
% similar to (14) that fixed pi_2 at the value estimated on the unsmoothed
% data, replaces the age and cohort dummies by fifth order polynomials, and
% extends the highest age to 70 to avoid some of the endpoint problems
% commonly encountered with polynomial smoothers"
% Eqn 14: lnCtilde_i=f_i pi_1 + a_i pi_2 + b_i pi_3 + U_i pi_4 + Ret_i pi_5 +epsilon_i
% Where: Ctilde=observed household consumption in the CEX
%        f_i=set of family dummies, a_i=complete set of age dummies
%        b_i=complete set of cohort dummies (less the middle one)
%        U_i=census region unemployment rate in year tau
%        Ret_i=dummy for each group that is equal to 1 when the respondent is retired
smoothavgconsumption=avgconsumption; % PLACEHOLDER


figure(2)
plot(26:1:65,smoothavgconsumption,'-r')
hold on
plot(26:1:65,avgconsumption,'or')
plot(26:1:65,avgincome,'-|b')
hold off
title('Household consumption and income over the life cycle')
legend('Consumption (smoothed)','Consumption (raw)','Income')
ylabel('1987 dollars')
xlabel('Age')

% From the average income profile we derive the expected income growth profile
% Params.g=[log(avgincome(2:end))-log(avgincome(1:end-1)),0]; % GP2002 pg 65 gives this formula
% But actually this is done off of log avg income fitted to a fifth order polynomial in age
% Got polynomial coeffs for upsilon from (Ctrl+F for "permanent income growth" https://github.com/ThomasHJorgensen/Sensitivity/blob/master/GP2002/solve_GP2002.py
Ybar=exp(6.8013936+0.3264338*Params.age-0.0148947*Params.age.^2+0.000363424*Params.age.^3-4.411685*10^(-6)*Params.age.^4+2.056916*10^(-8)*Params.age.^5);
Params.g=[log(Ybar(2:end))-log(Ybar(1:end-1)),0]; % GP2002 pg 65 gives this formula
% Note: Ybar ends up almost identical to avgincome in any case

%% Now, set up the model
n_d=0; % no decision variable
n_a=1001; % 501 % 1000 % assets is the only endogenous state
n_z=75; %51 % 75 Permanent shock (GP2002 renormalize to eliminate this, I just keep it as I'm lazy)
n_e=12; % zero-income shock (since the calibration follow Caroll (1997) I use 12 points as that is what Caroll (1997) used. I am just guessing GP2002 do the same)


%% Parameters
% Note: rho, beta, gamma0, gamma1 are going to be estimated by SMM, so these are anyway just initial guesses
% I set them to the values that GP2002 use for Figure 1 (so that I can reproduce it)

% Preferences
Params.rho=0.514; % CRRA utility param
% Params.upsilon= % Consumption equivalence scale
% GP2002 describe upsilon as a preference shifter that depends on household
% characteristics. But then just simplify it to an consumption equivalence scale anyway.
% Note: GP2002 is weird, as put consumption equivalence scale outside the utility fn, rather than directly on consumption in the utility fn.
% GP2002 say it is a fifth-order polynomial in age
% Got polynomial coeffs for upsilon from (Ctrl+F for "family shifter" https://github.com/ThomasHJorgensen/Sensitivity/blob/master/GP2002/solve_GP2002.py
Params.upsilonraw=0+0.13964975*Params.age-0.0047742190*Params.age.^2+8.5155210*10^(-5)*Params.age.^3-7.9110880*10^(-7)*Params.age.^4+2.9789550*10^(-9)*Params.age.^5;
% ThomasHJorgensen then does exp(upsilonraw-upsilonraw[lag])
Params.upsilonintermediate=exp(Params.upsilonraw(2:end)-Params.upsilonraw(1:end-1));
% ThomasHJorgensen says this is (upsilon_{t+1}/upsilon_t)^(1/rho) [a few lines below "family shifter"].
% [He, like GP2002, works directly, so all they need is (upsilon_{t+1}/upsilon_t), as that is what appears in the consumption Euler eqn, see eqn 5 of GP2002]
% But I think this is a typo, and that it should say
% (upsilon_{t+1}/upsilon_t)^(1-rho) as it is used to 'normalize the FOCs to for growth)
% Because I work with the value function, I need the actual upsilon_t. I
% will normalize first period value to 1, and then use upsilonintermediate to iterate them
Params.upsilon=ones(N_j,1);
for jj=2:N_j
    Params.upsilon(jj)=Params.upsilon(jj-1)*Params.upsilonintermediate(jj-1)^(1/(1-Params.rho));
end
% Note: Seems weird that GP2002 can have upsilon defined in a way that
% depends on rho, when upsilon is not part of the estimation step but rho
% is part of the estimation step?


% Discount factor
Params.beta=0.9598;

% Asset returns
Params.R=1.0344;

% Retirement
Params.gamma0=0.0015;
Params.gamma1=0.0710;
% We do not use gamma0 and gamma1 for anything (as we solve by VFI, while GP2002 solved the FOCs)
% Instead we need
Params.h=Params.gamma0/Params.gamma1;
% Note: GP2002 define gamma0=gamma1*h on pg 55 on line below eqn (6)


% Exogenous shocks
% Permanent shocks
% Params.g was already set above
Params.sigma_z_n=sqrt(0.0212); % standard dev of innovations to z
% Transitory shocks
Params.p=0.00302; % probability of zero earnings
Params.sigma_u=sqrt(0.0440); % std dev of innovations to u (exp(u) occurs with prob 1-p)
% GP2002 report 'variance of permanent shock' and 'variance of transitory shock' in Table 1 on pg 61

% Bequests
Params.kappa=500; 
% GP2002 don't seem to say what value kappa takes
% Codes at
% https://github.com/ThomasHJorgensen/Sensitivity/tree/master/GP2002 don't
% even seem to include a kappa parameter anywhere (probably not needed as
% only iterate on policy, not value fn, so just need c_{J+1})



% Initial assets at age 26
% Done below



%% GP2002 do a first stage estimation of R, p , sigma_z_n and sigma_u
% They get the point estimates as above
% R is interest rate of 3.440, p=0.00302, sigma_z_n=sqrt(0.0212), sigma_u=sqrt(0.0440)
% They also report that the standard error of R is 0.281 and the standard error of p is 0.000764.
% They do not report standard errors of the estimates of sigma_z_n^2 and sigma_u^2


%% Grids
% Because there is a positive probability in every period of zero earnings and agents must die without debt, there is implicitly a no-borrowing constaint
a_grid=(5*10^5)*linspace(0,1,n_a)'.^3; % annual income is about 20000, so use max assets of half million

d_grid=[];

% # Set permanent income to the first predicted value:
%         self.par.init_P = Ybar[0]/1000.0 # line 1651

% Permanent shocks
% GP2002 has that z_t=G_t z_{t-1} N_t
% G=(1+g) is the growth factor for permanent labor income
% ln(N) ~ N(0, sigma_n^2)
% If we log this process we get: lnZ_t= g + lnZ_t-1 + lnN_t
% Since N_t is log-normal, we just get n_t=log(N_t) ~ N(0, sigma_z_n^2), which is nice and easy
% We need to use the extension of Farmer-Toda to age-dependent parameters to handle the income growth g and that there are permanent shocks
kirkbyoptions.nSigmas=1; % originally I used 2 % GP2002 do not discuss what this should be. If I let it be substantial, then the minimum grid point gets silly-low at higher ages 
kirkbyoptions.initialj0mewz=log(Ybar(1)); % Based on codes of Jorgensen (2023), he seems to set period 1 to be a permanent shock on this Ybar(1). This is just like making period 0 the Ybar(1).
[lnz_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_Kirkby(Params.g,ones(1,N_j),Params.sigma_z_n*ones(1,N_j),n_z,N_j,kirkbyoptions);
% Note: GP2002 does what you should do in a model with exogenous
% labor and  permament shocks, namely renormalize and then solve. We just
% take the lazy (but computationally costly) option here, on the plus side
% it makes it easy to modify this shock process.
z_grid_J=exp(lnz_grid_J);
% jequaloneDistz is the initial distribution

% Turns out that while the original discretization is good, taking the exponential of the grid messes up the drift.
% (This is fairly standard, discretizing and then taking exponential often messes things up)
% Following few lines are showing what goes wrong
[lnzmean,lnzvariance,lnzautocorrelation,~]=MarkovChainMoments_FHorz(lnz_grid_J,pi_z_J,jequaloneDistz);
[zmean,zvariance,zautocorrelation,~]=MarkovChainMoments_FHorz(z_grid_J,pi_z_J,jequaloneDistz);
% lnzmean./log(Ybar) % Shows that mean of discretized income is always within 1% of data
% zmean./Ybar % shows that after exponentials are taken it yoes wrong
% exp(lnzmean)./Ybar % Shows just a 2-3% difference, so mostly has to do with how exponential interacts with grid and probabilities
% I just do something fairly brute-force to correct this, namely I renormalize each age grid to get the correct growth in mean income
for jj=1:N_j
    z_grid_J(:,jj)=z_grid_J(:,jj)*(Ybar(jj)/zmean(jj));
end
% Check this has done a decent job
[zmean,zvariance,zautocorrelation,~]=MarkovChainMoments_FHorz(z_grid_J,pi_z_J,jequaloneDistz);
% zmean./Ybar % Now these are all equal to 1
% Note: You could definitely do something nicer here, but this will do (as paper is anyway just about age-conditional means)


% Transitory shocks
% First, the e shock which is i.i.d.
% Consists of a first stage, which GP2002 call u
farmertodaoptions.nSigmas=2; % GP2002 don't appear to say what this should be
[u_grid,pi_u]=discretizeAR1_FarmerToda(0,0,Params.sigma_u,n_e-1,farmertodaoptions);
pi_u=pi_u(1,:)';
% [sum(lnZ_grid.*pi_Z),Params.mew_lnZ] % should be equal, they are
% Now add in the zero-income event, and use Z instead of lnZ
e_grid=[0;exp(u_grid)];
pi_e=[Params.p; (1-Params.p)*pi_u]; % There is no difference between pi_lnZ and pi_Z
  
% sum(e_grid.*pi_e) 
% Not exactly equal to one because of the discretization, so I renormalize the grid so that we get exactly 1
e_grid=e_grid./sum(e_grid.*pi_e);
sum(e_grid.*pi_e) % This should be 1 (it is :)

% Note: GP2002 use zero mean for innovations n and u, which is kind of an odd departure from Carroll (1997) given how closely the income process otherwise follows it
% (Carroll (1997) uses means for u and n that mean that exp(u) and exp(n) are a specific non-zero mean and mean one repectively)

vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;


%% Return fn
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime,a,z,e,R,rho,upsilon,agej,J,beta,kappa,h) GourinchasParker2002_ReturnFn(aprime,a,z,e,R,rho,upsilon,agej,J,beta,kappa,h);



%% Initial values for parameters to be estimated
% Use the AGS estimates
Params.beta=0.9574;
Params.rho=0.6526;
Params.gamma0=0.0015;
Params.gamma1=0.071;
Params.h=Params.gamma0/Params.gamma1;
% WHAT IS x0: log(0.0611)=-2.79, so this is omega26_mean, which is anyway almost exactly what I set below: exp(-2.794)=0.0612


% Overwrite with some better initial guesses based on the solution (calculated on a previous run)
Params.rho=1.5;
Params.beta=0.93;
Params.kappa=15;

%% Try solving value function
tic;
vfoptions.divideandconquer=1; % take advantage of monotonicity for a faster solution algorithm
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
vftime=toc

%% Initial distribution

% GP2002 do not talk about initial distribution of permanent shocks.
% GP2002 say they set initial assets to be log-normal distribution with
Params.omega26_mean=-2.794; % parameter for the mean of log assets
Params.omega26_stddev=1.784; % parameter for the std dev of log assets
% Note that from these two, we get that the mean of assets is
% exp(Params.omega26_mean+0.5*Params.omega26_stddev^2)

% So we need to put this log-normal distribution onto our asset grid
tic;
jequaloneDistassets=MVNormal_ProbabilitiesOnGrid(log(a_grid+[10^(-9);zeros(n_a-1,1)]),Params.omega26_mean,Params.omega26_stddev,n_a); % note: first point in a_grid is zero, so have to add something tiny before taking log
initdisttime=toc

% So initial dist is given by
jequaloneDist=zeros(n_a,n_z,n_e);
% I decided to add some variance to the income shocks, otherwise the first moment has almost zero variance at low ages which seems silly (clearly conflicts with data)
jequaloneDist=jequaloneDistassets.*jequaloneDistz'.*shiftdim(pi_e,-2);
z_grid_J(:,1)=linspace(0.95,1.05,n_z)'.*z_grid_J(:,1); % add 

% GP2002 don't specify age weights, but as their only outputs are
% life-cycle profiles they don't have too. So we will just put equal weights.
AgeWeightParamNames={'mewj'};
Params.mewj=ones(1,N_j)/N_j;

%% Solve agent dist
tic;
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);
disttime=toc

%% Plot some outputs
FnsToEvaluate2.Income=@(aprime,a,z,e) z*e;
FnsToEvaluate2.Consumption=@(aprime,a,z,e,R) R*a+z*e-aprime;
FnsToEvaluate2.z=@(aprime,a,z,e) z;
FnsToEvaluate2.e=@(aprime,a,z,e) e;
FnsToEvaluate2.a=@(aprime,a,z,e) a;

simoptions.whichstats=[1,0,0,0,0,0,0];
tic;
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);
acstime=toc

figure(3);
plot(Params.age,AgeConditionalStats.Income.Mean)
hold on
plot(Params.age,AgeConditionalStats.Consumption.Mean)
hold off
title('Model life-cycle profiles')
legend('Income','Consumption')


figure(4)
subplot(3,1,1); plot(Params.age,AgeConditionalStats.a.Mean)
title('Assets (a)')
subplot(3,1,2); plot(Params.age,AgeConditionalStats.z.Mean)
title('Permanent shock (z)')
subplot(3,1,3); plot(Params.age,AgeConditionalStats.e.Mean)
title('Transitory shock (e)')


figure(5)
subplot(2,1,1); plot(26:1:65,avgincome,Params.age,AgeConditionalStats.Income.Mean)
title('Income')
legend('Data','Model')
subplot(2,1,2); plot(26:1:65,smoothavgconsumption,Params.age,AgeConditionalStats.Consumption.Mean,26:1:65,fittedavgconsumption_AGS2017)
title('Consumption')
legend('Data','Model','AGS2017')


%% Okay, let's estimate

%% Now, we estimate the model parameters to target the mean life cycle profile of consumption
% To do this using the toolkit there are two things we need to setup
% First, just say all the model parameters we want to calibrate
% For exercise of GP2002 there are four parameters
EstimParamNames={'beta','rho','h','kappa'};
% EstimParamNames gives the names of all the model parameters that will be
% estimated, these parameters must all appear in Params, and the values in
% Params will be used as the initial values.
% All other parameters in Params will remain fixed.

% Second, we need to say which model statistics we want to target
% We can target any model stat generated by the AllStats, and LifeCycleProfiles commands
% We set up a structure containing all of our targets
TargetMoments.AgeConditionalStats.Consumption.Mean=smoothavgconsumption';
estimoptions.logmoments=1; % We target log(C), not C. [Note: we put in the target C, and use estimoptions.logmoments to say we want to target log(C). Internally the log() will be taken, do not take the logs when setting up target here]
% Note: When setting up TargetMoments there are some rules you must follow
% There are two options TargetMomements.AgeConditionalStats and
% TargetMoments.AllStats (you can use both). Within these you must 
% follow the structure that you get when you run the commands
% AgeConditionalStats=LifeCycleProfiles_FHorz_Case1()
% and
% AllStats=EvalFnOnAgentDist_AggVars_FHorz_Case1()
% [If you are using PType, then these will be the PType equivalents]

% So we want a FnsToEvaluate which is just consumption (this will be faster as it only includes what we actually need)
FnsToEvaluate.Consumption=@(aprime,a,z,e,R) R*a+z*e-aprime;

save ./SavedOutput/GP2002setup.mat

% load ./SavedOutput/GP2002setup.mat

estimoptions.CalibParamsNames={'R'};

%% Need the Variance-Covariance matrix of the moment conditions
% GP2002 say they set weights as diagonal matrix, with elements being inverse of the variance of the data moments (given the actual numbers, these are presumably for log(C))
% AGS2017 give the weights, but not the original covarinance matrix, so just invert it
CovarMatrixDataMoments=zeros(size(WeightingMatrix_AGS2017));
CovarMatrixDataMoments(logical(eye(size(CovarMatrixDataMoments))))=diag(WeightingMatrix_AGS2017).^(-1); 
% Note: Because the data source used by GP2002 is cross-sectional, the
% covariances between the different age-conditional-means is zero by definition.

% Note: The Variance-Covariance matrix of the moment conditions is not
% typically equal to the Variance-Covariance matrix if the data moments.
% But we are using 'seperable moments' and GMM estimation, so the two
% coincide for us here. [Note, GP2002 and AGS2017 use SMM, so it would not
% be true for them.]

%% Done, setting up. 

%% First, the robust estimation
% Robust: use the identity matrix as weighting matrix

RobustWeightingMatrix=eye(size(CovarMatrixDataMoments));

estimoptions.verbose=1; % give feedback
[EstimParams_robust, EstimParamsConfInts_robust, estsummary_robust]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, RobustWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, estimoptions, vfoptions,simoptions);
% EstimParams is the estimated parameter values
% estsummary is a structure containing various info on how the estimation
% went, plus some output useful for analysis

save ./SavedOutput/GP2002estimation_robust.mat EstimParams_robust EstimParamsConfInts_robust Params estsummary_robust estimoptions

save ./SavedOutput/GP2002_1.mat


%% Second, the efficient estimation
% Efficient: use the inverse of the variance-covariance matrix of the moment conditions as the weighting matrix
% Typically, efficient estimation requires a two-iteration estimator.
% But...
% Because we use GMM and separable moments we can do GMM in a single
% iteration. GMM with separable moments means that the variance-covariance 
% matrix of the moment conditions is just equal to the variance-covariance
% matrix of the data moments.

EfficientWeightingMatrix=CovarMatrixDataMoments^(-1);

estimoptions.verbose=1; % give feedback
[EstimParams_eff, EstimParamsConfInts_eff, estsummary_eff]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, EfficientWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, estimoptions, vfoptions,simoptions);
% EstimParams is the estimated parameter values
% estsummary is a structure containing various info on how the estimation
% went, plus some output useful for analysis

save ./SavedOutput/GP2002estimation_eff.mat EstimParams_eff EstimParamsConfInts_eff Params estsummary_eff estimoptions

save ./SavedOutput/GP2002_2.mat



%% Do I get the same estimates:
% GP2002 report that: 
% beta=0.960
% rho=0.514;
% h=gamma0/gamma1=0.001/0.071=0.0141
% AGS2017 report that:
% beta=0.9574
% rho=0.6526

% My estimates differ, but because I have a different parametrization of
% period J+1 to GP2002 (I parametrize V_{J+1} by h and kappa, they
% parametrize C_{J+1} by gamma0 and gamma1) it is not possible to just
% compare the value of the objective function under my parameter estimates
% with the value of the objective function under their parameter estimates.




%% Plot some outputs
for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_eff.(EstimParamNames{cc});
end

[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);

% FnsToEvaluate2.Income=@(aprime,a,z,e) z*e;
% FnsToEvaluate2.Consumption=@(aprime,a,z,e,R) R*a+z*e-aprime;
% FnsToEvaluate2.z=@(aprime,a,z,e) z;
% FnsToEvaluate2.e=@(aprime,a,z,e) e;
% FnsToEvaluate2.a=@(aprime,a,z,e) a;

AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

figure(11);
plot(Params.age,AgeConditionalStats.Income.Mean)
hold on
plot(Params.age,AgeConditionalStats.Consumption.Mean)
hold off
title('Model life-cycle profiles')
legend('Income','Consumption')


figure(12)
subplot(3,1,1); plot(Params.age,AgeConditionalStats.a.Mean)
title('Assets (a)')
subplot(3,1,2); plot(Params.age,AgeConditionalStats.z.Mean)
title('Permanent shock (z)')
subplot(3,1,3); plot(Params.age,AgeConditionalStats.e.Mean)
title('Transitory shock (e)')


figure(13)
subplot(2,1,1); plot(26:1:65,avgincome,Params.age,AgeConditionalStats.Income.Mean)
title('Income')
legend('Data','Model')
subplot(2,1,2); plot(26:1:65,smoothavgconsumption,Params.age,AgeConditionalStats.Consumption.Mean,26:1:65,fittedavgconsumption_AGS2017)
title('Consumption')
legend('Data','Model','AGS2017')


% As a check on the graph, calculate the GMM objective function both for our estimate and AGS2017 estimate
Mdiff_eff=(smoothavgconsumption-AgeConditionalStats.Consumption.Mean)';
Mdiff_AGS=(smoothavgconsumption-fittedavgconsumption_AGS2017)';
GMMobj_eff=Mdiff_eff'*EfficientWeightingMatrix*Mdiff_eff;
GMMobj_AGS=Mdiff_AGS'*EfficientWeightingMatrix*Mdiff_AGS;
% GMMobj_AGS is lower, but because it is based on EE I cannot check if
% their parameters actually give the consumption means that they report (or
% if their mean consumption profile is actually impossible to acheive due
% to V_{J+1} having different parametrization to C_{J+1}.)
% Given their parameter estimates are essentailly my initial guess, I expect the
% alternative parametrization makes it impossible to achieve an objective
% function value as low as theirs.

%% Check asset grid (make sure people are not trying to leave the top end)
Sdist=squeeze(sum(sum(StationaryDist,3),2));
Sdist=cumsum(Sdist,1);
Sdist=Sdist./Sdist(end,:); % normalize to 1 conditional on age
figure(14)
plot(Sdist(:,1))
hold on
plot(Sdist(:,10))
plot(Sdist(:,20))
plot(Sdist(:,30))
plot(Sdist(:,40))
hold off
legend('period 1','10','20','30','40')
% Seems fine, no-one runs into the top of the grid

%% Runtime Comparisons
% needed for paper
simoptions.whichstats=[1,0,0,0,0,0,0];
simoptions.numbersims=1000; % for panel data

FnsToEvaluate_temp.Consumption=FnsToEvaluate.Consumption; % Just consumption

R=100; % Across 100 runs
vftime=zeros(1,R);
disttime=zeros(1,R);
acstime=zeros(1,R);
paneltime=zeros(1,R);
acspaneltime=zeros(1,R);
for rr=1:R
    fprintf('Runtime comparison: rr=%i \n',rr)

    tic;
    [V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
    vftime(rr)=toc;

    tic;
    StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);
    disttime(rr)=toc;

    tic;
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate_temp,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);
    acstime(rr)=toc;

    tic;
    simPanelValues=SimPanelValues_FHorz_Case1(jequaloneDist,Policy,FnsToEvaluate_temp,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,simoptions);
    paneltime(rr)=toc;

    tic;
    AgeConditionalStats_PanelValues=PanelValues_LifeCycleProfiles_FHorz(simPanelValues,N_j,simoptions);
    acspaneltime(rr)=toc;

end

%% Runtime comparisons
fprintf('When doing iteration runtimes are \n')
[mean(disttime(2:end)),mean(acstime(2:end)),mean(disttime(2:end)+acstime(2:end))]
fprintf('When doing simulation runtimes are \n')
[mean(paneltime(2:end)),mean(acspaneltime(2:end)),mean(paneltime(2:end)+acspaneltime(2:end))] % first panel simulation contains time to open cpu pool, so drop it
fprintf('For comparison, the value fn runtime is \n')
mean(vftime)

fprintf('Ratio of iteration-to-simulation runtimes is \n')
mean(disttime(2:end)+acstime(2:end))/mean(paneltime(2:end)+acspaneltime(2:end))



%% Can we tell in advance which are likely to be best moments to target?
% Calculate all the derivatives of moments with respect to parameters
% Want to think about Gourinchas & Parker (2002) vs Cagetti (2003)
% Both essentially estimate model of Carroll (1997)
% GP2002 targets age-conditional mean of consumption
% C2003 targets age-conditional median of wealth
% Use calibrated parameters of Carroll (1997) 

% load ./SavedOutput/GP2002_1.mat

FnsToEvaluate3.Consumption=@(aprime,a,z,e,R) R*a+z*e-aprime;
FnsToEvaluate3.Wealth=@(aprime,a,z,e) a;


Params.beta=1/(1+0.04);
Params.rho=2;
% Carroll (1997) doesn't need/use a parametrization of V_{T+1}, so I will just use our estimated one.
Params.h=EstimParams_eff.h;
Params.kappa=EstimParams_eff.kappa;

% Compute the derivatives of model moments with respect to parameters to be estimated.
[MomentDerivatives, SortedMomentDerivatives, momentderivsummary]=EstimateLifeCycleModel_MomentDerivatives(EstimParamNames, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate3, estimoptions, vfoptions,simoptions);


% How does beta matter?
% Look at the derivatives w.r.t. beta for age-conditional mean of consumption
MomentDerivatives.wrt_beta.AgeConditionalStats.Consumption.Mean
% Look at the derivatives w.r.t. beta for age-conditional median of wealth
MomentDerivatives.wrt_beta.AgeConditionalStats.Wealth.Median
MomentDerivatives.wrt_beta.AgeConditionalStats.Wealth.Mean

% How does rho matter?
% Look at the derivatives w.r.t. beta for age-conditional mean of consumption
MomentDerivatives.wrt_rho.AgeConditionalStats.Consumption.Mean
% Look at the derivatives w.r.t. beta for age-conditional median of wealth
MomentDerivatives.wrt_rho.AgeConditionalStats.Wealth.Median
MomentDerivatives.wrt_rho.AgeConditionalStats.Wealth.Mean

save ./SavedOutput/MomentDerivatives.mat MomentDerivatives Params momentderivsummary

% What about std deviations of the empirical moments?
% GP2002 get consumption data using 40,000 households, but across all the age it is nearer 1000 per age.
% GP2002, pg 48: "from a sample of roughly 40,000 households from the Consumer Expenditure Survey (CEX) from 1980 to 1993"
% C2003, apparently just 20 to 100 observations per age 
% C2003 gets wealth data from Survey of Consumer Finances (SCF), has
% roughly 3275 individual households worth of data (top pg 343)
% I have the variance of the empirical moments for GP2002 (see near top of code, they were used for the weighting matrix)
% C2003 does not appear to report hem (and don't think codes are available, I have not tried emailing Marco Cagetti to ask if he still has them)


% Note: C2003 does simulations with 10,000 individuals (C2003, pg 352: "the
% life cycle profiles are simulated for 10,000 households while each age
% group contains from 20 to 100 observations, so the ratio of observations
% to simulated points is extremely low")
% Note: C2003, pg 352 "The variance of the estimator due to the simulation
% is not considered", not sure what he means by this? Guessing he just
% omits the (1+1/s) term from the covar matrix of the parameter estimates?

% GP2002 parameter estimates:
% beta    0.9598 (0.0101)
% rho     0.5140 (0.1690)
% gamma0  0.0015 (5.68x10^(-6))
% gamma1  0.0710 (0.0613)
%
% C2003 parameter estimates:
% beta    









