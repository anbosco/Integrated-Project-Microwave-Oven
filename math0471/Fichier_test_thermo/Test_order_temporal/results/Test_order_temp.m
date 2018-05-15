%% This function plot the results for the temporal order of convergence of 
% the thermal solver.
function Test_order_temp
close all
clear all
%% Importation

% Theta = 1
cd theta_1
cd dt_0_1
Temperature2_theta1 = load('Probe1_step0_to_step1000.txt');
cd ..

cd dt0_05
Temperature3_theta1 = load('Probe1_step0_to_step2000.txt');
cd ..

cd dt_0_025
Temperature4_theta1 = load('Probe1_step0_to_step4000.txt');
cd ../..

% Theta = 0.5
cd theta_0_5
cd dt_0_1
Temperature2_theta0_5 = load('Probe1_step0_to_step1000.txt');
cd ..

cd dt0_05
Temperature3_theta0_5 = load('Probe1_step0_to_step2000.txt');
cd ..

cd dt_0_025
Temperature4_theta0_5 = load('Probe1_step0_to_step4000.txt');
cd ../..

%% Analytic solution
dt2 = 0.1;
dt3 = 0.05;
dt4 = 0.025;

t2 = 0:dt2:100-dt2;
t3 = 0:dt3:100-dt3;
t4 = 0:dt4:100-dt4;

Q = 50000;
rho = 1;
cp = 1;
omega = (2*3.141592)/20;

T_anal2 = (Q/(rho*cp*omega))*(1-cos(omega*t2));
T_anal3 = (Q/(rho*cp*omega))*(1-cos(omega*t3));
T_anal4 = (Q/(rho*cp*omega))*(1-cos(omega*t4));

%% Calcul de l'erreur
err2_theta1 = max(abs(transpose(T_anal2)-Temperature2_theta1));
err3_theta1 = max(abs(transpose(T_anal3)-Temperature3_theta1));
err4_theta1 = max(abs(transpose(T_anal4)-Temperature4_theta1));

err2_theta0_5 = max(abs(transpose(T_anal2)-Temperature2_theta0_5));
err3_theta0_5 = max(abs(transpose(T_anal3)-Temperature3_theta0_5));
err4_theta0_5 = max(abs(transpose(T_anal4)-Temperature4_theta0_5));

T = [log10(dt2/100) log10(dt3/100) log10(dt4/100)];
err_log_theta1 = [log10(err2_theta1/(Q*100)) log10(err3_theta1/(Q*100)) log10(err4_theta1/(Q*100))];
err_log_theta0_5 = [log10(err2_theta0_5/(Q*100)) log10(err3_theta0_5/(Q*100)) log10(err4_theta0_5/(Q*100))];


%% Calcul de l'ordre
temp1 = polyfit(T,err_log_theta1 ,1);
Order_theta1= temp1(1)
temp2 = polyfit(T,err_log_theta0_5 ,1);
Order_theta_0_5 = temp2(1)
%% Plotting of the results
Figure1=figure(1);clf;set(Figure1,'defaulttextinterpreter','latex');
hold on;
set(gca,'fontsize',40,'fontname','Times','LineWidth',0.5); 

plot(T,polyval(temp1,T),'b','linewidth', 4);
plot(T,polyval(temp2,T),'k','linewidth', 4);
plot(T,err_log_theta1,'r.','Markersize',30);
plot(T,err_log_theta0_5,'r.','Markersize',30);
  axis([-3.7 -2.9 -6.5 -3]);
legend('\theta = 1','\theta = 0.5');
xlabel('$\log(\frac{dt}{T_f})$ [-]');
ylabel('$\log(\frac{\varepsilon\,Q\,T_f}{\rho c_p})$ [-]');
grid on;
box on;
end