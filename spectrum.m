clc
close all
clear all

% Set default font size globally
set(groot, 'DefaultAxesFontSize', 14);           % Axes tick labels
set(groot, 'DefaultTextFontSize', 14);           % Text objects like titles, labels
set(groot, 'DefaultUicontrolFontSize', 14);      % UI controls, if applicable

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

mode = "stokes";

switch mode
    case "stokes"
        list = [6,11,21,31];
    case "adv_diff"
        list = [6,11,21,31,41,51];
    case "wave"
        list = [21,31,41,51]; %,21,51,101];
    case "oseen"
        list = [6,11,15,21,26];
    case "kelvin-voigt"
        list = [11,21,31,41,51];
end

cond_H = [];
cond_A = [];
cond_precond = [];
cond_i_precond = [];
cond_i_precond2 = [];
cond_eyeS = [];
cond_ilu = [];
cond_B = [];


cond_A11 = [];
cond_H11 = [];
cond_A11_precond = [];

cond_schur = [];
cond_schur_precond = [];
cond_schur_massprecond = [];

for grids = list
    switch mode
        case "stokes"
             m = load(['data/Stokes/stokes_H1_',num2str(grids),'.mat']);
        case "adv_diff"
            m = load(['data/adv_diff/1d_',num2str(grids),'.mat']);
        case "wave"
            m = load(['data/wave/1d_',num2str(grids),'.mat']);
        case "oseen"
            m = load(['data/Oseen/Oseen_',num2str(grids),'.mat']);
        case "kelvin-voigt"
            m = load(['data/KV/KV_',num2str(grids),'.mat']);
    end   
    disp(num2str(grids))

    if mode ~= "kelvin-voigt"
        n = size(m.A,1);
        
        m.H=(m.A+m.A')/2; % H part
        m.S=(m.A-m.A')/2; % S part
    end
   

    if mode == "oseen"
        condmethod = @(mat) condest(mat);
        H11 = (m.A11 + m.A11')/2;
        S11 = (m.A11 - m.A11')/2;
        cond_H11 = [cond_H11, condest(H11)];
        cond_A11 = [cond_A11 condest(m.A11)];
        cond_A11_precond = [cond_A11_precond condest(H11\m.A11)];

        m.Ared = m.B*(m.A11\m.B'); %+ 1e-8*eye(size(m.B,2));
        Hred = (m.Ared + m.Ared')/2;
        Sred = (m.Ared - m.Ared')/2;
        Hpre = Hred\m.Ared;
        Mpre = m.Mp\m.Ared;
        eigsAred = sort(abs(eig(full(m.Ared))));
        eigsHpre = sort(abs(eig(full(Hpre))));
        eigsMpre = sort(abs(eig(full(Mpre))));

        
            % eigS = eig(full(S11));
            % mini = 10.;
            % for i = 1:size(eigS)
            %     if abs(eigS(i)) < mini
            %         if abs(eigS(i)) > 1e-14
            %             mini = abs(eigS(i));
            %         end
            %     end
            % end
            % 
            % 
            % cond_eyeS = [cond_eyeS max(abs(eigS))/ mini];

        cond_schur = [cond_schur eigsAred(end)/eigsAred(2)];
        cond_schur_precond = [cond_schur_precond eigsHpre(end)/eigsHpre(2)];
        cond_schur_massprecond = [cond_schur_massprecond eigsMpre(end)/eigsMpre(2)];
        %cond_schur = [cond_schur condmethod(m.Ared)];
        %cond_schur_precond = [cond_schur_precond condmethod(Hred\m.Ared)];
        %cond_schur_massprecond = [cond_schur_massprecond condmethod(m.Mp\m.Ared)];
    elseif mode == "kelvin-voigt"
        condmethod = @(mat) condest(mat);
        dum = m.A11;
        m.A11 = m.A11*m.A11;
        H11 = (m.A11 + m.A11')/2;
        S11 = (m.A11 - m.A11')/2;
        cond_H11 = [cond_H11, condest(H11)];
        cond_A11 = [cond_A11 condest(m.A11)];
        cond_A11_precond = [cond_A11_precond condest(H11\m.A11)];
        %disp([ " smallest and largest A11", min(eig(full(m.A11))), " ", max(eig(full(m.A11)))])
        %max(eig(full(m.A11)))

        m.Ared = m.B*inv(m.A11)*m.B; %m.B*inv(m.A11)*m.B; %1*eye(size(m.B,2));
        Hred = (m.Ared + m.Ared')/2;
        Sred = (m.Ared - m.Ared')/2;
        Hpre = Hred\m.Ared;
        Mpre = m.Mp\m.Ared;
        eigsAred = sort(abs(eig(full(m.Ared))));
        eigsHpre = sort(abs(eig(full(Hpre))));
        eigsMpre = sort(abs(eig(full(Mpre))));
        eigsB = sort(abs(eig(full(m.B))));
        %disp([ " smallest and largest Ared", min(eig(full(m.Ared))), " ", max(eig(full(m.Ared)))])

        cond_B = [cond_B eigsB(end)/eigsB(3)];
        cond_schur = [cond_schur eigsAred(end)/eigsAred(3)];
        cond_schur_precond = [cond_schur_precond eigsHpre(end)/eigsHpre(3)];
        cond_schur_massprecond = [cond_schur_massprecond eigsMpre(end)/eigsMpre(3)];

    else
        condmethod = @(mat) condest(mat);
        m.L = ichol(m.H,struct('type','ict','droptol',1e-2));
        m.L2 = ichol(m.H,struct('type','ict','droptol',1e-4));


        cond_H = [cond_H condmethod(m.H)];
        cond_A = [cond_A condmethod(m.A)];
        cond_precond = [cond_precond condmethod(m.H\m.A)];
        cond_i_precond = [cond_i_precond condmethod(m.L'\(m.L\m.A))];
        cond_i_precond2 = [cond_i_precond2 condmethod(m.L2'\(m.L2\m.A))];


        if mode ~= "stokes"
            eigS = eig(full(m.S));
            mini = 10.;
            for i = 1:size(eigS)
                if abs(eigS(i)) < mini
                    if abs(eigS(i)) > 0
                        mini = abs(eigS(i));
                    end
                end
            end
    
            cond_eyeS = [cond_eyeS max(abs(eigS))/ mini]; %max(eig(full(m.S)))];
        end
    end
    %m.L = chol(m.H,'lower');
    %disp(["min ", min(eig(full(m.A)))])
    %disp(["max ", max(eig(full(m.A)))])

    
    % if grids == 11
    %     h_eigs =  eigs(m.H(2:end-1,2:end-1));
    %     lambda_max = max(h_eigs);
    %     lambda_min = eigs(m.H(2:end-1,2:end-1), 1, 'smallestreal');
    % 
    % 
    % else
    %     h_eigs_temp = eigs(m.H(2:end-1,2:end-1));
    %     lambda_max = [lambda_max;max(h_eigs_temp)];
    %     lambda_min = [lambda_min;eigs(m.H(2:end-1,2:end-1), 1, 'smallestreal')];
    %     h_eigs = [h_eigs_temp h_eigs];
    % end
    %eigs(m.S)


end

switch mode
    case "oseen"
        fig = figure('Units', 'inches', 'Position', [1 1 8.27 3]);
        subplot(1,2,1);
        loglog(1./list, cond_A11, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_H11, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list, cond_A11_precond, '--x', 'LineWidth', 3, 'MarkerSize', 8);
       % loglog(1./list, cond_eyeS, '--x', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list,list,'k-', 'LineWidth', 3);
        loglog(1./list,list.^2,'k--', 'LineWidth', 3);
        hold off;
        set ( gca, 'xdir', 'reverse' )
        legend('$A_{11}$', '$H_{11}$','$H_{11}^{-1} A_{11}$','$h^{-1}$','$h^{-2}$','FontSize', 12, 'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'northwest'); %'ichol$(H,10^{-4})^{-1}A$'
        xlabel('$h$','FontSize',14);
        ylabel('condition number','FontSize',14);
        grid on;

        subplot(1,2,2);
        loglog(1./list, cond_schur, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_schur_precond, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list, cond_schur_massprecond, '--x', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list,list,'k-', 'LineWidth', 3);
        loglog(1./list,list.^2,'k--', 'LineWidth', 3);
        hold off;
        set ( gca, 'xdir', 'reverse' )
        legend('$W$', '$H_S^{-1} W$', '$M_p^{-1}W$','$h^{-1}$','$h^{-2}$','FontSize', 12, 'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'southeast'); %'ichol$(H,10^{-4})^{-1}A$'
        xlabel('$h$','FontSize',14)
        ylabel('condition number','FontSize',14)
        grid on;
        set(fig, 'Renderer', 'painters');
        print(fig, 'figs/cond_oseen', '-dpng', '-r300');  

    case "kelvin-voigt"
        fig = figure('Units', 'inches', 'Position', [1 1 8.27 3]);
        subplot(1,2,1);
        loglog(1./list, cond_A11, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_H11, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list, cond_A11_precond, '--x', 'LineWidth', 3, 'MarkerSize', 8);
       % loglog(1./list, cond_eyeS, '--x', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list,list,'k-', 'LineWidth', 3);
        loglog(1./list,list.^4,'k--', 'LineWidth', 3);
        loglog(1./list,list.^2,'k--', 'LineWidth', 3);

        loglog(1./list, cond_B, '--x', 'LineWidth', 3, 'MarkerSize', 8);
        hold off;
        set ( gca, 'xdir', 'reverse' )
        legend('$A_{11}$', '$H_{11}$','$H_{11}^{-1} A_{11}$','$h^{-1}$','$h^{-4}$','$h^{-2}$','CondB','FontSize', 12, 'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'east'); %'ichol$(H,10^{-4})^{-1}A$'
        xlabel('$h$','FontSize',14);
        ylabel('condition number','FontSize',14);
        grid on;

        subplot(1,2,2);
        loglog(1./list, cond_schur, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_schur_precond, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list, cond_schur_massprecond, '--x', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list,list,'k-', 'LineWidth', 3);
        loglog(1./list,list.^2,'k--', 'LineWidth', 3);
        loglog(1./list,list.^4,'k--', 'LineWidth', 3);

        hold off;
        set ( gca, 'xdir', 'reverse' )
        legend('$W$', '$H_S^{-1} W$','$M_p^{-1}W$', '$h^{-1}$','$h^{-2}$','$h^{-4}$','FontSize', 12, 'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'east'); %'ichol$(H,10^{-4})^{-1}A$'
        xlabel('$h$','FontSize',14)
        ylabel('condition number','FontSize',14)
        grid on;
        set(fig, 'Renderer', 'painters');
        print(fig, 'figs/cond_kv', '-dpng', '-r300');  

    case "stokes"
        fig = figure('Units', 'inches', 'Position', [1 1 5 3]);
        loglog(1./list, cond_A, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_H, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        %semilogy(list, cond_eyeS, '--x', 'LineWidth', 2, 'MarkerSize', 6);
        loglog(1./list, cond_precond, '-^', 'LineWidth', 3, 'MarkerSize', 8);hold on
        loglog(1./list, cond_i_precond, '--d', 'LineWidth', 3, 'MarkerSize', 8);
        %loglog(1./list, cond_i_precond2, '--d', 'LineWidth', 3, 'MarkerSize', 8);
        %loglog(1./list,list,'k-', 'LineWidth', 3);
        %loglog(1./list,list.^2,'k--', 'LineWidth', 3);
        grid on;
        yticks([1e3,1e4,1e5,1e6])

        hold off;
        legend('$A$', '$H$','$H^{-1}A$', 'ichol$(H,10^{-2})^{-1}A$','FontSize', 12, 'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'east'); %'ichol$(H,10^{-4})^{-1}A$'
        xlabel('$h$','FontSize',14)
        set ( gca, 'xdir', 'reverse' )
        ylabel('condition number','FontSize',14)
        set(fig, 'Renderer', 'painters');
        print(fig, 'figs/cond_stokes', '-dpng', '-r300');  
        
    otherwise
        fig = figure('Units', 'inches', 'Position', [1 1 8.27 3]);
        subplot(1,2,1)
        loglog(1./list, cond_A, '-o', 'LineWidth', 3, 'MarkerSize', 8); hold on;
        loglog(1./list, cond_H, '--s', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list, cond_eyeS, '--x', 'LineWidth', 3, 'MarkerSize', 8); 
        loglog(1./list,list,'k-', 'LineWidth', 3);
        loglog(1./list,list.^2,'k--', 'LineWidth', 3);
        set ( gca, 'xdir', 'reverse' )

        hold off;
        legend('$A=H+S$', '$H$','$S$','$h^{-1}$','$h^{-2}$','FontSize',12,'Orientation','horizontal', ...
            'NumColumns', 2, 'Location', 'northwest');
        xlabel('$h$','FontSize',14)
        ylabel('condition number','FontSize',14)
        grid on;
        
        subplot(1,2,2)
        loglog(1./list, cond_precond, '-^', 'LineWidth', 3, 'MarkerSize', 8);hold on
        loglog(1./list, cond_i_precond, '--d', 'LineWidth', 3, 'MarkerSize', 8);
        loglog(1./list,list,'k-', 'LineWidth', 3);
        set ( gca, 'xdir', 'reverse' )
        hold off;
        legend('$H^{-1}A$', 'ichol$(H)^{-1}A$','$h^{-1}$','FontSize',14,'Orientation','horizontal', 'NumColumns', 2);
        xlabel('$h$','FontSize',14)
        ylabel('condition number','FontSize',14)
        grid on;
        %tightfig;
        set(fig, 'Renderer', 'painters');
        print(fig, 'figs/cond_heat_ilu', '-dpng', '-r300');  % Outputs plot_a4.png
end   

