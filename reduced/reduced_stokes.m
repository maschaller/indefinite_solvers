clear all
%close all
%clc

%% load matrices
m = load('../data/Stokes/stokes_blockwise_11.mat');
%cd 'hsl_mi20-1.5.1/matlab/'
%hsl_mi20_install();

n = size(m.A,1);
m.b = m.b';

disp(['Dimension is ',num2str(n)])
%% some preparations
m.H=(m.A+m.A')/2; % H part
m.S=(m.A-m.A')/2; % S part
m.lam = 1e-1; %1e-5

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

tols = [globtol*1e-1,globtol*1e-1,globtol*1e-1];


%% functions
tic;
m.L = ichol(m.H,struct('type','ict','droptol',1e-2));
time_ass_ichol = toc;
disp(['Time for setting up incomp. chol ', num2str(time_ass_ichol)])


tic;
%m.L_genau = chol(m.H,'lower'); %ichol(m.H,struct('type','ict','droptol',1e-5));
time_ass_chol = toc;
disp(['Time for setting up chol ', num2str(time_ass_chol)])


m.A_fun=@(x)apply_matvec(x,m.L_genau,m.S);
m.IA_fun=@(x)apply_matvec_with_I(x,m.L_genau,m.S);
m.At_fun=@(x)apply_matvec(x,m.L_genau,-m.S);
m.IAt_fun=@(x)apply_matvec_with_I(x,m.L_genau,-m.S);

tic;
[m.LA,m.RA] = ilu(m.A,struct('type','ilutp','droptol',globtol*1e-2));
time_ass_ilu = toc;
disp(['Time for setting up incomp. lu ', num2str(time_ass_ilu)])

tic
[m.LA_true,m.RA_true] = lu(m.A);
time_ass_lu = toc;
disp(['Time for setting up lu ', num2str(time_ass_lu)])

disp(['Smallest eig of A ', num2str(eigs(m.A, 1, 'smallestreal'))])
disp(['Smallest eig of H ', num2str(eigs(m.H, 1, 'smallestreal'))])

tic
hsl_mi20_startup
control = hsl_mi20_control;
control.v_iterations = 4;
%control.coarse_solver= 4;
% control.one_pass_coarsen = 1;
% control.damping = .1;
% control.pre_smoothing = 5;
% control.post_smoothing = 5;
% control.err_tol = .9;
% control.smoother = 1;
% control.aggressive = 1.2;
inform = hsl_mi20_setup(m.H,control);
time_ass_mg = toc;
disp(['Time for setting up mg ', num2str(time_ass_mg)])

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

function Mx = OptSys_inex(u,m,mode)
    Mx = m.B'*Astar_inv(m.C*A_inv(m.B*u,m,mode),m,mode) + m.lam*m.Mass*u;
end

%%
m.innertol = 1e-8;
m.innermaxit = min(n,1000);

disp('Computing RHS...')
rhs_opt = -m.B'*Astar_inv(m.C*A_inv(m.b,m,"lgmres_mg"),m,"lgmres_mg");
disp('...done computing RHS')

fig = figure;
plot(rhs_opt);
print(fig, 'figs/testing_r', '-dpng', '-r300');  

outertol = 1e-4;
outermaxit = min(n,1000);

failtolerance = outertol*1e2;

%% Compute exact reference

% disp('solving exact...')
% tic
% u_ex = (m.B'*(m.A'\m.C'*m.C*(m.A\m.B)) + m.lam*m.Mass) \ rhs_opt;
% toc

% disp('applying backslash...')
% tic
% res1 = (m.B'*(m.A'\m.C'*m.C*(m.A\m.B)) + m.lam*m.Mass)*u_ex;
% toc

function normres = exact_residual(u,m,rhs_opt)
    %disp("computing residual...")

    %normres = norm((m.B'*(m.LA_true'\(m.RA_true'\m.C'*m.C*(m.RA_true\(m.LA_true\m.B*u)))) ...
    %+ m.lam*m.Mass*u - rhs_opt));
    
    %disp("preconditioned residual ")
    %norm(m.L\m.L'\(m.B'*(m.LA_true'\(m.RA_true'\m.C'*m.C*(m.RA_true\(m.LA_true\m.B*u)))) ...
    %+ m.lam*m.Mass*u - rhs_opt))

    %normres1 = norm((m.B'*(m.LA'\(m.RA'\m.C'*m.C*(m.RA\(m.LA\m.B*u)))) ...
    %   + m.lam*m.Mass*u - rhs_opt));
    
    mode = "lgmres_mg";
    OptSyscallable = @(u) OptSys_inex(u,m,mode);
    normres = norm(OptSyscallable(u) - rhs_opt);    
end

m.innertol = 1e-6; %10
ex_res = @(u) exact_residual(u,m,rhs_opt);

m.innertol = 1e-4; %-5
outertol = globtol;


%% Compare solvers 
disp('Comparing solvers...')

% direct as inner
mode = "ilu";
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_ilu,~,~,iter_ilu] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_ilu = toc;
res_ilu = ex_res(opt_u_ilu);
if(res_ilu > failtolerance)
    disp('ilu failed')
    disp(num2str(res_ilu));
else
    disp(['ILU  succeeded in ', num2str(time_ilu+time_ass_ilu), 's with ', num2str(iter_ilu), ...
        ' iterations at rel residual of ', num2str(res_ilu/norm(rhs_opt))])
end


%gmres as inner
m.innertol = tols(1);
mode = "gmres";
OptSyscallable = @(u) OptSys_inex(u,m,mode);
tic
[opt_u_gm,~,relres_1,iter_gm] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_gm = toc;
res_gm = ex_res(opt_u_gm);
if(res_gm > failtolerance)
    disp('GMRES failed')
    disp(num2str(res_gm))
else
    disp(['GMRES  succeeded in ', num2str(time_gm), 's with ', num2str(iter_gm),'\',num2str(innergm), ...
        ' iterations at rel residual of ', num2str(res_gm/norm(rhs_opt))])
    %disp(relres_1)
end


% lgmres as inner
mode = "lgmres";
m.innertol = tols(2);
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_lgm,~,~,iter_lgm] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_lgm = toc;
res_lgm = ex_res(opt_u_lgm);
if(res_lgm > failtolerance)
    disp('LGMRES(ICHOL) failed')
    disp(num2str(res_lgm))
else 
    disp(['LGMRES(ICHOL) succeeded in ', num2str(time_lgm+time_ass_ichol), 's with ', num2str(iter_lgm), '\',num2str(innerlgm), ...
        ' iterations at rel residual of ', num2str(res_lgm/norm(rhs_opt))])
end

mode = "lgmres_mg";
m.innertol = tols(3);
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_lgm_mg,~,~,iter_lgm_mg] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_lgm_mg = toc;
res_lgm_mg = ex_res(opt_u_lgm_mg);
if(res_lgm_mg > failtolerance)
    disp('LGMRES(MG) failed')
    disp(num2str(res_lgm_mg))
else 
    disp(['LGMRES(MG) succeeded in ', num2str(time_lgm_mg+time_ass_mg), 's with ', num2str(iter_lgm_mg), '\',num2str(innerlgm_mg), ...
        ' iterations at rel residual of ', num2str(res_lgm_mg/norm(rhs_opt))])
end

% rapoport as inner
% mode = "rapoport";
% tic
% OptSyscallable = @(u) OptSys_inex(u,m,mode);
% [opt_u_rap,~,~,iter_rap] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
% time_rap = toc;
% res_rap = ex_res(opt_u_rap);
% if(res_rap > failtolerance)
%     disp('Rapoport failed')
%     disp(res_rap)
% else 
%     disp(['Rapoport succeeded in ', num2str(time_rap+time_ass_chol), 's with ', num2str(iter_rap), '\',num2str(innerrapo), ...
%         ' iterations at rel residual of ', num2str(res_rap/norm(rhs_opt))])
% end
% 
% %widlund as inner
% mode = "widlund";
% tic
% OptSyscallable = @(u) OptSys_inex(u,m,mode);
% [opt_u_wid,~,~,iter_wid] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
% time_wid = toc;
% res_wid = ex_res(opt_u_wid);
% if(res_wid > failtolerance)
%     disp('Widlund failed')
%     disp(res_wid)
% else 
%     disp(['Widlund succeeded in ', num2str(time_wid+time_ass_chol), 's with ', num2str(iter_wid), '\',num2str(innerwid), ...
%         ' iterations at rel residual of ', num2str(res_wid/norm(rhs_opt))])
% end

% rapoport multigrid as inner
mode = "rapoport_mg";
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_rap_mg,~,~,iter_rap_mg] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_rap_mg = toc;
res_rap_mg = ex_res(opt_u_rap_mg);
if(res_rap_mg > failtolerance)
    disp('Rapoport failed')
    disp(num2str(res_rap_mg))
else 
    disp(['Rapoport(MG) succeeded in ', num2str(time_rap_mg+time_ass_mg), 's with ', num2str(iter_rap_mg), '\',num2str(innerrapo_mg), ...
        ' iterations at rel residual of ', num2str(res_rap_mg/norm(rhs_opt))])
end

%widlund multigrid as inner
mode = "widlund_mg";
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_wid_mg,~,~,iter_wid_mg] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_wid_mg = toc;
res_wid_mg = ex_res(opt_u_wid_mg);
if(res_wid_mg > failtolerance)
    disp('Widlund failed')
    disp(num2str(res_wid_mg))
else 
    disp(['Widlund(MG) succeeded in ', num2str(time_wid_mg+time_ass_mg), 's with ', num2str(iter_wid_mg), '\',num2str(innerwid_mg), ...
        ' iterations at rel residual of ', num2str(res_wid_mg/norm(rhs_opt))])
end


mode = "exact";
tic
OptSyscallable = @(u) OptSys_inex(u,m,mode);
[opt_u_ex,~,~,iter_ex] = pcg(OptSyscallable,rhs_opt,outertol,outermaxit);
time_ex = toc;
res_ex = ex_res(opt_u_ex);
if(res_ex > failtolerance)
    disp('exact failed')
    disp(num2str(res_ex))
    disp(num2str(iter_ex))
else
    disp(['Exact  succeeded in ', num2str(time_ex+time_ass_lu), 's with ', num2str(iter_ex), ...
        ' iterations at rel residual of ', num2str(res_ex/norm(rhs_opt))])
end




%disp([num2str(norm(opt_u_lgm)),' ',num2str(norm(opt_u_ilu)),' ',num2str(norm(opt_u_lgm_mg))])

%disp([num2str(norm(opt_u_lgm-opt_u_ilu)),' ',num2str(norm(opt_u_lgm-opt_u_lgm_mg))])



% disp(['& cg it.\ & ','--','&','--','&',num2str(iter_gm), ...
%     '&',num2str(iter_lgm),'&',num2str(iter_lgm_mg),'&',num2str(iter_rap_mg),'&',num2str(iter_wid_mg),'\\'])
% disp(['& inner it.\ & ','--','&','--','&',num2str(innergm), ...
%     '&',num2str(innerlgm),'&',num2str(innerlgm_mg_s),'&',num2str(innerrapo_mg),'&',num2str(innerwid_mg),'\\'])
% disp(['\multirow{-3}{*}{$n=', num2str(n),'$} '])
% disp(['& time & ','--','&','--', ...
%     '&',num2str(round(time_gm,2,'significant')), '&',num2str(round(time_lgm + time_ass_ichol,2,'significant')),'&', ...
%     num2str(round(time_lgm_mg + time_ass_mg,2,'significant')),'&', num2str(round(time_rap_mg+time_ass_mg,2,'significant')),'&',...
%     num2str(round(time_wid_mg+time_ass_mg,2,'significant')),'\\\hline'])

% disp(['& res & ','--','&','--','&', ...
%     num2str(round(res_gm/norm(rhs_opt),1,'significant')), '&',num2str(round(res_lgm/norm(rhs_opt),1,'significant')),'&', num2str(round(res_lgm_mg/norm(rhs_opt),1,'significant')),'&', ...
%     num2str(round(res_rap_mg/norm(rhs_opt),1,'significant')),'&',num2str(round(res_wid_mg/norm(rhs_opt),1,'significant')),'\\\hline'])


disp(['& cg it.\ & ',num2str(iter_ex),'&',num2str(iter_ilu),'&',num2str(iter_gm), ...
    '&',num2str(iter_lgm),'&',num2str(iter_lgm_mg),'&',num2str(iter_rap_mg),'&',num2str(iter_wid_mg),'\\'])
disp(['& inner it.\ & ','--','&','--','&',num2str(innergm), ...
    '&',num2str(innerlgm),'&',num2str(innerlgm_mg_s),'&',num2str(innerrapo_mg),'&',num2str(innerwid_mg),'\\'])
disp(['\multirow{-3}{*}{$n=', num2str(n),'$} '])
disp(['& time & ',num2str(round(time_ex+time_ass_lu,2,'significant')),'&',num2str(round(time_ilu+time_ass_ilu,2,'significant')), ...
    '&',num2str(round(time_gm,2,'significant')), '&',num2str(round(time_lgm + time_ass_ichol,2,'significant')),'&', ...
    num2str(round(time_lgm_mg + time_ass_mg,2,'significant')),'&', num2str(round(time_rap_mg+time_ass_mg,2,'significant')),'&',...
    num2str(round(time_wid_mg+time_ass_mg,2,'significant')),'\\\hline'])

%disp(['& res & ',num2str(round(res_ex/norm(rhs_opt),1,'significant')),'&',num2str(round(res_ilu/norm(rhs_opt),1,'significant')),'&', ...
  %  num2str(round(res_gm/norm(rhs_opt),1,'significant')), '&',num2str(round(res_lgm/norm(rhs_opt),1,'significant')),'&', num2str(round(res_lgm_mg/norm(rhs_opt),1,'significant')),'&', ...
 %   num2str(round(res_rap_mg/norm(rhs_opt),1,'significant')),'&',num2str(round(res_wid_mg/norm(rhs_opt),1,'significant')),'\\\hline'])



function y=apply_matvec(x,L,S)
y = L'\(L\(S*x));
end

function y=apply_matvec_with_I(x,L,S)
y = x+(L'\(L\(S*x)));
end





