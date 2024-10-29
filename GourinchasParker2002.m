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

%%
% A line I needed for running on the Server
addpath(genpath('./MatlabToolkits/'))
% gpuDevice(1) % reset gpu to clear out memory


%% Age and model periods

N_j=40;
Params.agejshifter=25; % first period is age 26
% Age
Params.J=N_j;
Params.agej=1:1:N_j;
Params.age=Params.agejshifter+Params.agej;

showFigures=0; % set to zero to run on server

% some stuff I want to skip sometimes (1=do them, 0=skip)
preCalibFigures=0;
doRuntimes=0;
doMomentDerivs=0;
doCheckVariousParams=0;

% alternative calibration for final period
altPeriodJ=1;

Params.nSigmas=2; % for both permanent and transitory shocks
% GP2002 did Gauss-Hermite quadrature with 5 points for permanent shocks (and same for for transitory shocks)
% I use KFTT for permanent and Farmer-Toda for transitory with way more points (actually, setting nSigmas=1 this was no longer possible, so switched to Tauchen)
% Set small nSigmas, used for both, to be more in line with what a 5 point Gauss-Hermite is likely to produce in terms of amount of income risk

estimoptions.fminalgo=8; % lsqnonlin()

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

if showFigures==1
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
end

% From the average income profile we derive the expected income growth profile
% Params.g=[log(avgincome(2:end))-log(avgincome(1:end-1)),0]; % GP2002 pg 65 gives this formula
% But actually this is done off of log avg income fitted to a fifth order polynomial in age
% Got polynomial coeffs for upsilon from (Ctrl+F for "permanent income growth" https://github.com/ThomasHJorgensen/Sensitivity/blob/master/GP2002/solve_GP2002.py
Ybar=exp(6.8013936+0.3264338*Params.age-0.0148947*Params.age.^2+0.000363424*Params.age.^3-4.411685*10^(-6)*Params.age.^4+2.056916*10^(-8)*Params.age.^5);
Params.g=[log(Ybar(2:end))-log(Ybar(1:end-1)),0]; % GP2002 pg 65 gives this formula
% Note: Ybar ends up almost identical to avgincome in any case

%% Now, set up the model
n_d=0; % no decision variable
n_a=1001; % 501 % 1001 % assets is the only endogenous state
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
% Which makes sense looking at eqns (12) and (13) of GP2002.
% [He, like GP2002, works directly, so all they need is (upsilon_{t+1}/upsilon_t)^(1/rho), as that is what appears in the consumption Euler eqn, see eqn 5 of GP2002]
% Because I work with the value function, I need the actual upsilon_t. I will normalize first period value to 1, 
% and then use upsilonintermediate to iterate them
%   Params.upsilon=ones(N_j,1);
%   for jj=2:N_j
%       Params.upsilon(jj)=Params.upsilon(jj-1)*Params.upsilonintermediate(jj-1)^Params.rho;
%   end
% Note: To solve VFI, we need upsilon, but to solve Consumption Euler Eqn
% you only need upsilonintermediate. Hence the dependence on rho is
% implicit, rather than explicit, in the Consumption Euler Eqn estimation.
% But here we do VFI, so we cannot use upsilon in the model (as it depends
% on rho which needs to be estimated) and hence we create the following
% 'upsilon pre rho' that will be modified by rho in the return function.
Params.upsilon_prerho=[1,cumprod(Params.upsilonintermediate)];
% We can get upsilon from this as: upsilon=upsilonprerho^rho [which is done inside the return function+


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
% GP2002 don't say what value kappa takes as they use euler eqn so just
% need gamma0 and gamma1 (instead of h and kappa)


% Initial assets at age 26
% Done below



%% GP2002 do a first stage estimation of R, p , sigma_z_n and sigma_u
% They get the point estimates as above
% R is interest rate of 3.440, p=0.00302, sigma_z_n=sqrt(0.0212), sigma_u=sqrt(0.0440)
% They also report that the standard error of R is 0.281 and the standard error of p is 0.000764.
% They do not report standard errors of the estimates of sigma_z_n^2 and sigma_u^2


%% Grids
% Because there is a positive probability in every period of zero earnings and agents must die without debt, there is implicitly a no-borrowing constaint
a_grid=(10^6)*linspace(0,1,n_a)'.^3; % average annual income is about 20,000, so use max assets of 600,000

d_grid=[];

% # Set permanent income to the first predicted value:
%         self.par.init_P = Ybar[0]/1000.0 # line 1651

% Shocks: Jorgensen codes does Gauss-Hermite for both the permanent and transitory shocks using 5 points for each.
%         My impression from his codes is that this is also what GP2002 did.

% Permanent shocks
% GP2002 has that z_t=G_t z_{t-1} N_t
% G=(1+g) is the growth factor for permanent labor income
% ln(N) ~ N(0, sigma_n^2)
% If we log this process we get: lnZ_t= g + lnZ_t-1 + lnN_t
% Since N_t is log-normal, we just get n_t=log(N_t) ~ N(0, sigma_z_n^2), which is nice and easy
% We need to use the extension of Farmer-Toda to age-dependent parameters to handle the income growth g and that there are permanent shocks
if Params.nSigmas>=1.2
    kfttoptions.nSigmas=Params.nSigmas;
    kfttoptions.nMoments=2; % discretization targets first two conditional moments
    kfttoptions.initialj0mewz=log(Ybar(1)); % Based on codes of Jorgensen (2023), he seems to set period 1 to be a permanent shock on this Ybar(1). This is just like making period 0 the Ybar(1).
    [lnz_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_KFTT(Params.g,ones(1,N_j),Params.sigma_z_n*ones(1,N_j),n_z,N_j,kfttoptions);
else % KFTT is better than Tauchen, but doesn't really work if you have nSigmas<1.2 (is hard to hit the variance if your max/min points are only one std dev!!)
    fellagallipolipanoptions.nSigmas=Params.nSigmas;
    fellagallipolipanoptions.initialj0mewz=log(Ybar(1)); % Based on codes of Jorgensen (2023), he seems to set period 1 to be a permanent shock on this Ybar(1). This is just like making period 0 the Ybar(1).
    [lnz_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_FellaGallipoliPanTauchen(Params.g,ones(1,N_j),Params.sigma_z_n*ones(1,N_j),n_z,N_j,fellagallipolipanoptions);
end
z_grid_J=exp(lnz_grid_J);
% Note: GP2002 does what you should do in a model with exogenous labor and  permament shocks, namely renormalize and then solve. We just
% take the lazy (but computationally costly) option here, on the plus side it makes it easy to modify this shock process.

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
if Params.nSigmas>=1.2
    farmertodaoptions.nSigmas=Params.nSigmas;
    [u_grid,pi_u]=discretizeAR1_FarmerToda(0,0,Params.sigma_u,n_e-1,farmertodaoptions);
else
    tauchenoptions=struct();
    Tauchen_q=Params.nSigmas;
    [u_grid,pi_u]=discretizeAR1_Tauchen(0,0,Params.sigma_u,n_e-1,Tauchen_q,tauchenoptions);
end
pi_u=pi_u(1,:)';
% [sum(lnZ_grid.*pi_Z),Params.mew_lnZ] % should be equal, they are
% Now add in the zero-income event, and use Z instead of lnZ
e_grid=[0; exp(u_grid)];
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

ReturnFn=@(aprime,a,z,e,R,rho,upsilon_prerho,agej,J,beta,kappa,h,altPeriodJ,rhoJ) GourinchasParker2002_ReturnFn(aprime,a,z,e,R,rho,upsilon_prerho,agej,J,beta,kappa,h,altPeriodJ,rhoJ);



%% Initial values for parameters to be estimated
% Use the AGS estimates
Params.beta=0.9574;
Params.rho=0.6526;
Params.gamma0=0.0015;
Params.gamma1=0.071;
Params.h=Params.gamma0/Params.gamma1;
% WHAT IS x0: log(0.0611)=-2.79, so this is omega26_mean, which is anyway almost exactly what I set below: exp(-2.794)=0.0612


Params.kappa=3;
% Overwrite with some better initial guesses based on the solution (calculated on a previous run)
Params.rho=0.3;
% Params.beta=0.94;

if altPeriodJ==1
    % rho_J is the rho for the terminal period (used in the 'warm-glow of bequests), altPeriodJ allows it to differ from rho.
    Params.rhoJ=Params.rho;
    Params.altPeriodJ=1;
elseif altPeriodJ==0
    Params.rhoJ=0; % Not used for anything
    Params.altPeriodJ=0;
end


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
% exp(Params.omega26_mean+0.5*Params.omega26_stddev^2)=0.300;
% These numbers are from Table II, pg 62 of GP2002.


% So we need to put this log-normal distribution onto our asset grid
tic;
jequaloneDistassets=MVNormal_ProbabilitiesOnGrid(log((a_grid+[10^(-9);zeros(n_a-1,1)])/10000),Params.omega26_mean,Params.omega26_stddev,n_a);
initdisttime=toc
% Note that the normal distribution is put onto log(a_grid), and hence this is like putting a log-normal distribution onto a_grid.
%     First point in a_grid is zero, so have to add something tiny before taking log; this is the +[10^(-9);zeros(n_a-1,1)]
%     This initial distribution is actually on 10,000s of dollars, not on dollars. So I put it onto a_grid/10000, rather than onto a_grid.


% So initial dist is given by
jequaloneDist=zeros(n_a,n_z,n_e);
% I decided to add some variance to the income shocks, otherwise the first moment has almost zero variance at low ages which seems silly (clearly conflicts with data)
jequaloneDist=jequaloneDistassets.*jequaloneDistz'.*shiftdim(pi_e,-2);
% z_grid_J(:,1)=linspace(0.95,1.05,n_z)'.*z_grid_J(:,1); % add 

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

simoptions.whichstats=[1,0,0,0,0,0,0]; % just the mean
tic;
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);
acstime=toc


if preCalibFigures==1 && showFigures==1
    figure(3);
    plot(Params.age,AgeConditionalStats.Income.Mean)
    hold on
    plot(Params.age,AgeConditionalStats.Consumption.Mean)
    hold off
    title('Model life-cycle profiles (pre-calibration)')
    legend('Income','Consumption')


    figure(4)
    subplot(3,1,1); plot(Params.age,AgeConditionalStats.a.Mean)
    title('Assets (a) (pre-calibration)')
    subplot(3,1,2); plot(Params.age,AgeConditionalStats.z.Mean)
    title('Permanent shock (z) (pre-calibration)')
    subplot(3,1,3); plot(Params.age,AgeConditionalStats.e.Mean)
    title('Transitory shock (e) (pre-calibration)')


    figure(5)
    subplot(2,1,1); plot(26:1:65,avgincome,Params.age,AgeConditionalStats.Income.Mean)
    title('Income (pre-calibration)')
    legend('Data','Model')
    subplot(2,1,2); plot(26:1:65,smoothavgconsumption,Params.age,AgeConditionalStats.Consumption.Mean,26:1:65,fittedavgconsumption_AGS2017)
    title('Consumption (pre-calibration)')
    legend('Data','Model','AGS2017')
end

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
TargetMoments.AgeConditionalStats.Consumption.Mean=log(smoothavgconsumption');
estimoptions.logmoments=1; % We target log(C), not C. [Note: we put in the target log(C), and use estimoptions.logmoments to say we want to target log(C) so that the model also uses log()]
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

estimoptions.verbose=1; % give feedback
estimoptions.CalibParamsNames={'R'};

if altPeriodJ==1
    EstimParamNames={'beta','rho','h','kappa','rhoJ'}; % include rhoJ in the parameters to estimate
end

%% When using small nSigmas get negative h, so I rule this out
% Constrain h>=0 
estimoptions.constrainpositive={'h'}; % 

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
ParametrizeParamsFn=[]; % not something we use

save ./SavedOutput/GP2002setup.mat

%% First, the robust estimation
% Robust: use the identity matrix as weighting matrix

RobustWeightingMatrix=eye(size(CovarMatrixDataMoments));

[EstimParams_robust, EstimParamsConfInts_robust, estsummary_robust]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, RobustWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions);
% EstimParams is the estimated parameter values
% estsummary is a structure containing various info on how the estimation
% went, plus some output useful for analysis

save ./SavedOutput/GP2002estimation_robust.mat EstimParams_robust EstimParamsConfInts_robust Params estsummary_robust estimoptions

% % Put the robust estimates into Params, so I can use them as an initial guess for the efficient estimates (should substantially reduce the runtime)
% for cc=1:length(EstimParamNames)
%     Params.(EstimParamNames{cc})=EstimParams_robust.(EstimParamNames{cc});
% end

%% Second, the efficient estimation
% Efficient: use the inverse of the variance-covariance matrix of the moment conditions as the weighting matrix
% Typically, efficient estimation requires a two-iteration estimator.
% But...
% Because we use GMM and separable moments we can do GMM in a single
% iteration. GMM with separable moments means that the variance-covariance 
% matrix of the moment conditions is just equal to the variance-covariance
% matrix of the data moments.

EfficientWeightingMatrix=CovarMatrixDataMoments^(-1);

[EstimParams_eff, EstimParamsConfInts_eff, estsummary_eff]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, EfficientWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions);
% EstimParams is the estimated parameter values
% estsummary is a structure containing various info on how the estimation
% went, plus some output useful for analysis

save ./SavedOutput/GP2002estimation_eff.mat EstimParams_eff EstimParamsConfInts_eff Params estsummary_eff estimoptions


%% Save progress
save ./SavedOutput/GP2002_1.mat


%% Do I get the same estimates:
% GP2002 report that: 
% beta=0.960 (std dev. 0.01)
% rho=0.514 (std dev. 0.169)
% h=gamma0/gamma1=0.001/0.071=0.0141
% AGS2017 report that:
% beta=0.9574
% rho=0.6526

% GP2002 had:
% gamma0=0.0015 (std dev. 3.84)
% gamma1=0.0710 (std dev. 0.1215)

% My estimates differ, but because I have a different parametrization of
% period J+1 to GP2002 (I parametrize V_{J+1} by h and kappa, they
% parametrize C_{J+1} by gamma0 and gamma1) it is not possible to just
% compare the value of the objective function under my parameter estimates
% with the value of the objective function under their parameter estimates.


%% Plot some outputs
for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_robust.(EstimParamNames{cc});
end
simoptions.whichstats=ones(1,7);
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);

AgeConditionalStats_robust=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

if showFigures==1
    figure(11);
    plot(Params.age,AgeConditionalStats.Income.Mean)
    hold on
    plot(Params.age,AgeConditionalStats_robust.Consumption.Mean)
    hold off
    title('Model life-cycle profiles')
    legend('Income','Consumption')


    figure(12)
    subplot(3,1,1); plot(Params.age,AgeConditionalStats_robust.a.Mean)
    title('Assets (a)')
    subplot(3,1,2); plot(Params.age,AgeConditionalStats_robust.z.Mean)
    title('Permanent shock (z)')
    subplot(3,1,3); plot(Params.age,AgeConditionalStats_robust.e.Mean)
    title('Transitory shock (e)')


    figure(13)
    subplot(2,1,1); plot(26:1:65,avgincome,Params.age,AgeConditionalStats_robust.Income.Mean)
    title('Income')
    legend('Data','Model')
    subplot(2,1,2); plot(26:1:65,smoothavgconsumption,Params.age,AgeConditionalStats_robust.Consumption.Mean,26:1:65,fittedavgconsumption_AGS2017)
    title('Consumption')
    legend('Data','Model','AGS2017')
end

% As a check on the graph, calculate the GMM objective function both for our estimate and AGS2017 estimate
Mdiff_mine=(log(smoothavgconsumption)-log(AgeConditionalStats.Consumption.Mean))';
Mdiff_AGS=(log(smoothavgconsumption)-log(fittedavgconsumption_AGS2017))';
GMMobj_mine=Mdiff_mine'*EfficientWeightingMatrix*Mdiff_mine;
GMMobj_AGS=Mdiff_AGS'*EfficientWeightingMatrix*Mdiff_AGS;
% GMMobj_AGS is lower, but because it is based on EE I cannot check if
% their parameters actually give the consumption means that they report (or
% if their mean consumption profile is actually impossible to acheive due
% to V_{J+1} having different parametrization to C_{J+1}.)
% Given their parameter estimates are essentailly my initial guess, I expect the
% alternative parametrization makes it impossible to achieve an objective
% function value as low as theirs.


%% Check asset grid (make sure people are not trying to leave the top end)
if showFigures==1
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
    title('CDF of households over assets')
    % Seems fine, no-one runs into the top of the grid

    % Take a look at dist over z shocks too (just to see)
    Sdistz=squeeze(sum(sum(StationaryDist,3),1));
    Sdistz=cumsum(Sdistz,1);
    Sdistz=Sdistz./Sdistz(end,:); % normalize to 1 conditional on age
    figure(15)
    plot(Sdistz(:,1))
    hold on
    plot(Sdistz(:,10))
    plot(Sdistz(:,20))
    plot(Sdistz(:,30))
    plot(Sdistz(:,40))
    hold off
    legend('period 1','10','20','30','40')
    title('CDF of households over z (unit-root shocks)')
    % Looks fine
end

%% Runtime Comparisons
% needed for paper
if doRuntimes==1
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

end % doRuntimes


%% Check the standard errors, using much larger grids
n_a=2501;
a_grid=(7*10^5)*linspace(0,1,n_a)'.^3; % average annual income is about 20,000, so use max assets of 600,000
% Changing a_grid, we also need to change
jequaloneDistassets=MVNormal_ProbabilitiesOnGrid(log(a_grid/1000+[10^(-9);zeros(n_a-1,1)]),Params.omega26_mean,Params.omega26_stddev,n_a); % note: first point in a_grid is zero, so have to add something tiny before taking log
% I decided to add some variance to the income shocks, otherwise the first moment has almost zero variance at low ages which seems silly (clearly conflicts with data)
jequaloneDist=jequaloneDistassets.*jequaloneDistz'.*shiftdim(pi_e,-2);

estimoptions.skipestimation=1
vfoptions.level1n=floor(n_a/100);

for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_robust.(EstimParamNames{cc});
end
[EstimParams_robust2, EstimParamsConfInts_robust2, estsummary_robust2]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, RobustWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions);

for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_eff.(EstimParamNames{cc});
end
% estimoptions.efficientW=1; % Not really needed as in principle the formulas evaulate to the the same anyway
[EstimParams_eff2, EstimParamsConfInts_eff2, estsummary_eff2]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, EfficientWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions);


% Use the robust point estimates, and then compute the efficient standard
% deviations (to back up my claim that the larger std dev of efficient
% estimate is because of the different point estimate)
for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_robust.(EstimParamNames{cc});
end
[EstimParams_effcheck, EstimParamsConfInts_effcheck, estsummary_effcheck]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, EfficientWeightingMatrix,CovarMatrixDataMoments, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions);


save ./SavedOutput/GP2002_2.mat


%% Table for my paper
FID = fopen('./SavedOutput/GP2002estimates.tex', 'w');
fprintf(FID, '\\begin{tabular}{|l|c|c|l|} \\hline \\hline \n');
fprintf(FID, '\\multicolumn{4}{|c|}{\\textbf{Estimated Parameters}} \\\\ \\hline \n');
fprintf(FID, '\\multicolumn{4}{|c|}{\\textbf{Robust estimates ($W=\\mathbb{I}$)}} \\\\ \n');
fprintf(FID, '\\multicolumn{1}{|c}{} & \\multicolumn{1}{c}{\\textit{Point Estimate}} & \\multicolumn{1}{c}{\\textit{90\\%% Conf. Int}} & \\multicolumn{1}{c|}{\\textit{Description}}  \\\\ \n');
fprintf(FID, '\\hline \n');
fprintf(FID, '$\\beta$ & %1.3f & [%1.3f, %1.3f] & discount factor \\\\ \n', EstimParams_robust.beta, EstimParamsConfInts_robust2.beta(1), EstimParamsConfInts_robust2.beta(2));
fprintf(FID, '$\\rho$ & %1.3f & [%1.3f, %1.3f] & curvature of utility fn \\\\ \n', EstimParams_robust.rho, EstimParamsConfInts_robust2.rho(1), EstimParamsConfInts_robust2.rho(2));
fprintf(FID, '$h$ & %1.3f & [%1.3f, %1.3f] & role of permanent income in warm-glow \\\\ \n', EstimParams_robust.h, EstimParamsConfInts_robust2.h(1), EstimParamsConfInts_robust2.h(2));
fprintf(FID, '$\\kappa$ & %1.3f & [%1.3f, %1.3f] & relative importance of warm-glow \\\\ \n', EstimParams_robust.kappa, EstimParamsConfInts_robust2.kappa(1), EstimParamsConfInts_robust2.kappa(2));
if altPeriodJ==1
    fprintf(FID, '$\\rho_J$ & %1.3f & [%1.3f, %1.3f] & curvature of warm-glow utility fn \\\\ \n', EstimParams_robust.rhoJ, EstimParamsConfInts_robust2.rhoJ(1), EstimParamsConfInts_robust2.rhoJ(2));
end
fprintf(FID, '\\hline \n');
fprintf(FID, '\\multicolumn{4}{|c|}{\\textbf{Efficient estimates ($W=\\Omega^{-1}$)}} \\\\ \n');
fprintf(FID, '\\multicolumn{1}{|c}{} & \\multicolumn{1}{c}{\\textit{Point Estimate}} & \\multicolumn{1}{c}{\\textit{90\\%% Conf. Int}} & \\multicolumn{1}{c|}{\\textit{Description}}  \\\\ \n');
fprintf(FID, '\\hline \n');
fprintf(FID, '$\\beta$ & %1.3f & [%1.3f, %1.3f] & discount factor \\\\ \n', EstimParams_eff.beta, EstimParamsConfInts_eff2.beta(1), EstimParamsConfInts_eff2.beta(2));
fprintf(FID, '$\\rho$ & %1.3f & [%1.3f, %1.3f] & curvature of utility fn \\\\ \n', EstimParams_eff.rho, EstimParamsConfInts_eff2.rho(1), EstimParamsConfInts_eff2.rho(2));
fprintf(FID, '$h$ & %1.3f & [%1.3f, %1.3f] & role of permanent income in warm-glow \\\\ \n', EstimParams_eff.h, EstimParamsConfInts_eff2.h(1), EstimParamsConfInts_eff2.h(2));
fprintf(FID, '$\\kappa$ & %1.3f & [%1.3f, %1.3f] & relative importance of warm-glow \\\\ \n', EstimParams_eff.kappa, EstimParamsConfInts_eff2.kappa(1), EstimParamsConfInts_eff2.kappa(2));
if altPeriodJ==1
    fprintf(FID, '$\\rho_J$ & %1.3f & [%1.3f, %1.3f] & curvature of warm-glow utility fn \\\\ \n', EstimParams_eff.rhoJ, EstimParamsConfInts_eff2.rhoJ(1), EstimParamsConfInts_eff2.rhoJ(2));
end
fprintf(FID, '\\hline \\hline \n \\end{tabular} \n');
fprintf(FID, '\\begin{minipage}[t]{0.90\\textwidth}{\\baselineskip=.5\\baselineskip \\vspace{.3cm} \\textit{\\footnotesize{Notes: (i) The parameter $h$ was constrained to be positive.}}} \\end{minipage}');
fclose(FID);


%% Graph for appendix in my paper
simoptions.whichstats=ones(1,7);
% Consumption moments for robust estimates
for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_robust.(EstimParamNames{cc});
end
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);
AgeConditionalStats_robust=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);
% Consumption moments for efficient estimates
for cc=1:length(EstimParamNames)
    Params.(EstimParamNames{cc})=EstimParams_eff.(EstimParamNames{cc});
end
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);
AgeConditionalStats_eff=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

if showFigures==1
    fig16=figure(16);
    subplot(2,1,1); plot(26:1:65,avgincome,Params.age,AgeConditionalStats_robust.Income.Mean)
    title('Income')
    legend('Data','Model')
    subplot(2,1,2); plot(26:1:65,smoothavgconsumption,Params.age,AgeConditionalStats_robust.Consumption.Mean,Params.age,AgeConditionalStats_eff.Consumption.Mean,26:1:65,fittedavgconsumption_AGS2017)
    title('Consumption')
    legend('Data','Robust','Efficient','AGS2017')
    saveas(fig16,'./SavedOutput/FigureConsumptionIncomeFit.pdf','pdf')
end
% And double-check (as not in figure)
max(abs(AgeConditionalStats_robust.Income.Mean-AgeConditionalStats_eff.Income.Mean)) % should be zero

