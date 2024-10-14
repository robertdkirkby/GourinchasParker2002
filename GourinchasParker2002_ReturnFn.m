function F=GourinchasParker2002_ReturnFn(aprime,a,z,e,R,rho,upsilonprerho,agej,J,beta,kappa,h,altPeriodJ,rhoJ)

F=-Inf;

upsilon=upsilonprerho^rho;

c=R*a+z*e-aprime;

if c>0
    F=upsilon*(c^(1-rho))/(1-rho);
end    

if agej==J
    % Final period, so add warm glow from retirement
    % Value function for retirement is on pg 54 of GP2002
    % Fret=beta*kappa*upsilon*(X_{J+1}+H_{J+1})^(1-rho)
    % Because GP2002 work with FOCs, rather than value fn, they never
    % actually define kappa, and instead can simply get an expression for C_J+1
    % based on how consumption evolves under some extreme simplifying assumptions
    % Specifically, they use parameters gamma0 and gamma1
    % But here we are doing value function iteration, so that formulation
    % is not enough for us to solve the problem.
    % [To solve the problem with VFI we need V_{J+1}, while they just need
    % C_{J+1} which you can see if you write it as a sequence problem and
    % then look at the consumption Euler Eqn connecting periods J and J+1 (pretend 
    % the agent survives a few period after retirement). Note that C_{J+1} is not 
    % enough to tell us the value of V_{J+1}.]
    % So we will drop gamma0 and gamma1, and instead use h and kappa.
    % We define h=gamma0/gamma1, and H=hZ (both of these two eqns are actually used by GP2002, so nothing novel here).
    % We have to find the value of kappa, whereas GP2002 did not (they
    % assume an implicit value for it in using the FOC with the values of
    % gamma0 and gamma1, but it is too awkward to try figure it out).

    % Note that we can modify the value fn on retirement to
    % Fret=beta*kappa*upsilon*((X_{J+1}+H_{J+1})^(1-rho))/(1-rho)
    % Just renormalizes kappa, which eases the interpretation of kappa as makes it look more like the utility fn

    % Now evaluate X_{J+1}+H_{J+1}
    XplusH=(R*aprime+z)+h*z;
    % H_{J+1}=h z_{J+1}=h z_J (near top of pg 54 of GP2002)
    % X_{J+1}=R*aprime+z
    % Note that X_{J+1} is the cash-on-hand, which equals R*aprime+zprime*eprime
    % I simplify this to R*aprime+z 
    % Note: this is giving you an income of z in first period of retirement, but this is implicit in the eqn halfway down pg 55 of GP2002 so it is not my fault :P
    % (GP2002 don't need this because they use FOCs so ignore the bequests, but they make same zprime=z assumption for H_{J+1} near top of pg 54)
    % Note that E[eprime]=1, so I just put in this certainty equivalent.

    if altPeriodJ==0 % Do final period following GP2002 paper
        Fret=beta*kappa*upsilon*(XplusH^(1-rho))/(1-rho); % Add retirement utility to this period utility
        % Note: implicitly assume upsilon is constant from period J to J+1 (is anyway hardly changing in last few periods)
    else % altPeriodJ==1
        % Use different rho for the final period
        Fret=beta*kappa*upsilon*(XplusH^(1-rhoJ))/(1-rhoJ); % Add retirement utility to this period utility
        % Note: still uses rho in upsilon.
    end

    F=F+Fret;

    % What GP2002 did:
    % Eqn between eqns (5) and (6) on pg 55 of GP2002 tells us that
    % (immediately below it is that gamma0=gamma1*h; top of pg 54 says H=h*z)
    % C_{j+1}=gamma1*((R*aprime+z*1)+h*z); % In principle this is zprime, but permanent shock is constant in retirement, so zprime is just equal to z for this last period
    % And then because they solve based on FOCs, C_{j+1} is all they need (no need for V_{J+1})

end

