clc; clear all;
% Set N =1 for obtaining the results of run 0
% N = 1;
% Set N > 1 for obtaining the average results of run 0 to run N-1
N = 5;
set(0, 'DefaultAxesFontName', 'SimSun');
set(0, 'DefaultTextFontName', 'SimSun');
%% plot reward
figure;
for i = 1 : N
    r(i, :) = csvread(['results\data\average_reward', num2str(i), '.csv']);
end
[~, x] = size(r);
set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
plot(500*[1:x], mean(r, 1),'LineWidth',1.5,'Color','#5861AC');
grid on;
xlabel('Iteration');
ylabel('Average effective SE (bits/s/Hz)');
set(gca,'FontName','Times New Roman','fontsize',12);
exportgraphics(gcf,'reward.pdf','ContentType','vector');

%% plot g
figure;
for i = 1 : N
    g(i, :) = csvread(['results\data\average_loss', num2str(i), '.csv']);
end
set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
plot(500*[1:x], mean(g, 1),'LineWidth',1.5,'Color','#5861AC');
exportgraphics(gcf,'Qnet.pdf','ContentType','vector');

% plot c
figure;
for i = 1 : N
    cp(i, :) = csvread(['results\data\average_constraint', num2str(i), '.csv']);
end
plot(500*[1:x], mean(cp, 1)/5, 'LineWidth',1.5,'Color','#5861AC');
% plot(500*[1:x], mean(cp, 1)/5, 'LineWidth',1.5,'Color','#000000');
% ylim([-0.1, 0.35]);
ylim([-10, 35]);
xl=get(gca,'Xlim');
set(gcf,'unit','normalized','position',[0.2,0.2,0.24,0.28]);
hold on
plot(xl,[0,0],'--','LineWidth',1.5,'Color','#000000');
grid on;

xlabel('Iteration');
ylabel('Constraint violation probability (%)');
set(gca,'FontName','Times New Roman','fontsize',12);
exportgraphics(gcf,'constraint.pdf','ContentType','vector');



