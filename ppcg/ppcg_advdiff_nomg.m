clear all
close all

%% convection diffusion
m = load('../data/adv_diff/3d_21.mat');
smallscale = false;

m.b = m.b';
n = size(m.A,1);
disp(['Dimension is ',num2str(n)])

%% some preparations
m.H=(m.A+m.A')/2; % H part
m.S=(m.A-m.A')/2; % S part
m.lam = 1e-4;

globtol = 1e-4;
global innerlgm;
innerlgm = 0;
global innergm;
innergm = 0;
global innerrapo;
innerrapo = 0;
global innerwid;
innerwid = 0;
global innerlgm_mg;
innerlgm_mg = 0;
global innerrapo_mg;
innerrapo_mg = 0;
global innerwid_mg;
innerwid_mg = 0;

tols = 1e-6*[1,1,1];


%% functions
tic;
m.L = ichol(m.H,struct('type','ict','droptol',1e-1));
time_ass_ichol = toc;
disp(['Time for setting up incomp. chol ', num2str(time_ass_ichol)])


tic;
m.L_genau = chol(m.H,'lower'); %ichol(m.H,struct('type','ict','droptol',1e-5));
time_ass_chol = toc;
disp(['Time for setting up chol ', num2str(time_ass_chol)])

 m.A_fun=@(x)apply_matvec(x,m.L_genau,m.S);
 m.IA_fun=@(x)apply_matvec_with_I(x,m.L_genau,m.S);
 m.At_fun=@(x)apply_matvec(x,m.L_genau,-m.S);
 m.IAt_fun=@(x)apply_matvec_with_I(x,m.L_genau,-m.S);

tic;
if smallscale
    [m.LA,m.RA] = ilu(m.A,struct('type','ilutp','droptol',globtol*1e-1));
end
time_ass_ilu = toc;
disp(['Time for setting up incomp. lu ', num2str(time_ass_ilu)])

tic
if smallscale
    [m.LA_true,m.RA_true] = lu(m.A);
end
time_ass_lu = toc;
disp(['Time for setting up lu ', num2str(time_ass_lu)])


function y=apply_matvec_mg(x,S)
y = hsl_mi20_precondition((S*x));
end

function y=apply_matvec_with_I_mg(x,S)
y = x+ hsl_mi20_precondition((S*x));
end

m.A_fun_mg=@(x)apply_matvec_mg(x,m.S);
m.IA_fun_mg=@(x)apply_matvec_with_I_mg(x,m.S);
m.At_fun_mg=@(x)apply_matvec_mg(x,-m.S);
m.IAt_fun_mg=@(x)apply_matvec_with_I_mg(x,-m.S);


s = whos;
disp(['Memory used ', num2str(1e-9*sum([s.bytes])),'GB'])

function x = A_inv(rhs,m,mode)
global innerlgm;
global innergm;
global innerrapo;
global innerwid;
global innerrapo_mg;
global innerwid_mg;
global innerlgm_mg;
iter = 0;
    switch mode
        case "lgmres"
            [x,~,~,iter] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,m.L,m.L');
            %[x,~,~,iter] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,m.LA,m.RA);
            innerlgm = innerlgm + iter(2);
        case "lgmres_mg"
            [x,~,~,iter] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,@(x) hsl_mi20_precondition(x));
            %[x,~,~,iter] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,m.LA,m.RA);
            innerlgm_mg = innerlgm_mg + iter(2);
        case "gmres"
            [x,~,~,iter] = gmres(m.A,rhs,[],m.innertol,m.innermaxit); 
            innergm = innergm + iter(2);
        case "rapoport"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = rapoport(m.IA_fun,m.H,m.S,m.L_genau'\(m.L_genau\rhs),m.innermaxit,m.innertol);
                innerrapo = innerrapo + iter;
            end
        case "rapoport_mg"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = rapoport(m.IA_fun_mg,m.H,m.S,hsl_mi20_precondition(rhs),m.innermaxit,m.innertol);
                innerrapo_mg = innerrapo_mg + iter;
            end
        case "widlund"
             x = zeros(size(m.A,1),1);
             if norm(rhs) > 1e-12
                [x,iter] = widlund(m.A_fun,m.H,m.S,m.L_genau'\(m.L_genau\rhs),m.innermaxit,m.innertol);
                innerwid = innerwid + iter;
             end
        case "widlund_mg"
             x = zeros(size(m.A,1),1);
             if norm(rhs) > 1e-12
                [x,iter] = widlund(m.A_fun_mg,m.H,m.S,hsl_mi20_precondition(rhs),m.innermaxit,m.innertol);
                innerwid_mg = innerwid_mg + iter;
             end
        case "exact"
             x = m.RA_true\(m.LA_true\ rhs);
        case "ilu"
             x = m.RA\(m.LA\ rhs);
        case "biCG"
             [x,~]= bicgstab(m.A,rhs,m.innertol,m.innermaxit); 
        case "lbiCG"
             [x,~]= bicgstab(m.A,rhs,m.innertol,m.innermaxit,m.L,m.L'); 
        case "lgmres_genau"
             [x,~] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,m.L_genau,m.L_genau');
    end
end

function x = Astar_inv(rhs,m,mode)
global innerlgm;
global innergm;
global innerrapo;
global innerwid;
global innerrapo_mg;
global innerwid_mg;
global innerlgm_mg;
iter = 0;
    switch mode
        case "lgmres"
            [x,~,~,iter] = gmres(m.A',rhs,[],m.innertol,m.innermaxit,m.L,m.L');
            innerlgm = innerlgm + iter(2);
        case "lgmres_mg"
            [x,~,~,iter] = gmres(m.A',rhs,[],m.innertol,m.innermaxit,@(x) hsl_mi20_precondition(x));
            innerlgm_mg = innerlgm_mg + iter(2);
        case "gmres"
            [x,~,~,iter]= gmres(m.A',rhs,[],m.innertol,m.innermaxit); 
            innergm = innergm + iter(2);
        case "rapoport"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = rapoport(m.IAt_fun,m.H,m.S,m.L_genau'\(m.L_genau\rhs),m.innermaxit,m.innertol);
            end
            innerrapo = innerrapo + iter;
        case "rapoport_mg"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = rapoport(m.IAt_fun_mg,m.H,m.S,hsl_mi20_precondition(rhs),m.innermaxit,m.innertol);
            end
            innerrapo_mg = innerrapo_mg + iter;
        case "widlund"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = widlund(m.At_fun,m.H,m.S,m.L_genau'\(m.L_genau\rhs),m.innermaxit,m.innertol);
            end
            innerwid = innerwid + iter;
        case "widlund_mg"
            x = zeros(size(m.A,1),1);
            if norm(rhs) > 1e-12
                [x,iter] = widlund(m.At_fun_mg,m.H,m.S,hsl_mi20_precondition(rhs),m.innermaxit,m.innertol);
            end
            innerwid_mg = innerwid_mg + iter;
        case "exact"
             x  =  m.LA_true'\(m.RA_true'\rhs);
        case "ilu"
             x  =  m.LA'\(m.RA'\rhs);
        case "biCG"
             [x,~]= bicgstab(m.A',rhs,m.innertol,m.innermaxit); 
        case "lbiCG"
             [x,~]= bicgstab(m.A',rhs,m.innertol,m.innermaxit,m.L,m.L'); 
        case "lgmres_genau"
              disp(num2str(m.innertol))
             [x,~] = gmres(m.A,rhs,[],m.innertol,m.innermaxit,m.L_genau,m.L_genau');
    end
   %disp(['adj ',num2str(norm(rhs)),"   ", num2str(norm(x)),"   ", num2str(norm(m.A'*x-rhs))])
end

function x = P(rhs,m,mode)
    n = size(m.A,1);
    rhs_p = rhs(1:n);
    rhs_u = rhs(n+1:2*n);
    rhs_y = rhs(2*n+1:end);
    p = Astar_inv(rhs_p,m,mode);
    u = (m.lam*m.Mass) \ (m.B'*p + rhs_u);
    y = A_inv(m.B*u+rhs_y ,m,mode);
    x = [y;u;p];
end



%%

M = [m.C 0*speye(n) m.A'; 0*speye(n) m.lam*m.Mass -m.B'; m.A -m.B 0*speye(n)];

m.innertol = 1e-6;
m.innermaxit = 1000;
res_changed = [-m.C* (A_inv(m.b,m,"lgmres")); zeros(n,1); zeros(n,1)];
z0 = [A_inv(m.b,m,"lgmres"); zeros(n,1); zeros(n,1)];

outermaxit = 1000;
outertol = globtol;
failtolerance = outertol*1e2;

% direct as inner
if smallscale
    mode = "ilu";
    tic
    [z_ilu,flag,~,iter_ilu] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
    time_ilu = toc;
    res_ilu = norm(M*z_ilu - res_changed);
    if(res_ilu > failtolerance || flag ~=0)
        disp(['ilu failed with flag ', num2str(flag),' and residual ', res_ilu/norm(res_changed)])
    else
        disp(['ILU  succeeded in ', num2str(time_ilu+time_ass_ilu), 's with ', num2str(iter_ilu), ...
            ' iterations at rel residual of ', num2str(res_ilu/norm(res_changed))])
    end
end


% gmres as inner
mode = "gmres";
m.innertol = tols(1);
tic
[z_gmres,flag,relres_gm,iter_gm] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
time_gm = toc;
res_gm = norm(M*z_gmres - res_changed);
if(res_gm > outertol || flag ~=0)
    disp(['GMRES failed with flag ', num2str(flag),' and residual ', res_gm/norm(res_changed)])
else
    disp(['GMRES  succeeded in ', num2str(time_gm), 's with ', num2str(iter_gm),'\',num2str(innergm), ...
        ' iterations at rel residual of ', num2str(res_gm/norm(res_changed))])
end

% lgmres as inner
mode = "lgmres";
m.innertol = tols(2);
tic
[z_lgmres,flag,~,iter_lgm] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
time_lgm = toc;
res_lgm = norm(M*z_lgmres - res_changed);
if(res_lgm > failtolerance || flag ~=0)
    disp(['LGMRES(ICHOL) failed with flag ', num2str(flag),' and residual ', res_lgm/norm(res_changed)])
else 
    disp(['LGMRES(ICHOL) succeeded in ', num2str(time_lgm+time_ass_ichol), 's with ', num2str(iter_lgm), '\',num2str(innerlgm), ...
        ' iterations at rel residual of ', num2str(res_lgm/norm(res_changed))])
end


% rapoport as inner
mode = "rapoport";
tic
[z_rap,flag,~,iter_rap] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
time_rap = toc;
res_rap = norm(M*z_rap - res_changed);
if(res_rap > failtolerance|| flag ~=0)
    disp(['Rapoport failed with flag ', num2str(flag),' and residual ', res_rap/norm(res_changed)])
else 
    disp(['Rapoport succeeded in ', num2str(time_rap+time_ass_chol), 's with ', num2str(iter_rap), '\',num2str(innerrapo), ...
        ' iterations at rel residual of ', num2str(res_rap/norm(res_changed))])
end
% 
% %widlund as inner
mode = "widlund";
tic
[z_wid,flag,~,iter_wid] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
time_wid = toc;
res_wid = norm(M*z_wid - res_changed);
if(res_wid > failtolerance)
    disp(['Widlund failed with flag ', num2str(flag),' and residual ', res_wid/norm(res_changed)])
else 
    disp(['Widlund succeeded in ', num2str(time_wid+time_ass_chol), 's with ', num2str(iter_wid), '\',num2str(innerwid), ...
        ' iterations at rel residual of ', num2str(res_wid/norm(res_changed))])
end


% direct as inner
if smallscale
    mode = "exact";
    tic
    [z_ex,flag,~,iter_ex] = pcg(M,res_changed,outertol,outermaxit,@(b) P(b,m,mode) );
    time_ex = toc;
    res_ex = norm(M*z_ex - res_changed);
    if(res_ex > failtolerance || flag ~=0)
        disp('exact failed')
    else
        disp(['Exact  succeeded in ', num2str(time_ex+time_ass_lu), 's with ', num2str(iter_ex), ...
            ' iterations at rel residual of ', num2str(res_ex/norm(res_changed))])
    end
end


function y=apply_matvec(x,L,S)
y = L'\(L\(S*x));
end

function y=apply_matvec_with_I(x,L,S)
y = x+(L'\(L\(S*x)));
end

