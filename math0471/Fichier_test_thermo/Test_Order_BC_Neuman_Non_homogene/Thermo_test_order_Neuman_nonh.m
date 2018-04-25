function Thermo_test_order
close all 
clear ll
Q = 50000;
L = 0.16;	
pos = 0.16;
h = 1;
k = 1;
Tinf = 20;
C1 = Q*(L/pi);
C1=C1+h*Tinf;
% C1=C1/(0.85*k +h*L)
C1=C1/(k+h*L)

%% dx == O.02
dx = 0.02;
x = 0:dx:0.3;
eta = 0 : dx: L;
cd 'dx_0_02'
y_num1 = load('Cut_alongY_step2.txt');

y_numX1 = load('Cut_alongX_step2.txt');
y_numZ1 = load('Cut_alongZ_step2.txt');
cd ..

temp = find(y_num1==0)
i_min = 5;
i_max = 13;
y_num1 = y_num1(i_min:i_max);
y_numX1 = y_numX1(i_min:i_max);
y_numZ1 = y_numZ1(i_min:i_max);
y_anal1 = (sin(eta*(pi/L))*(L/pi)^2*Q)/k + C1*eta;
% hold on 
% plot(eta,y_num1,'b.','markersize',20);

err_1 = max(abs(y_num1-transpose(y_anal1)));

%% dx == O.01
% figure
dx = 0.01;
x = 0:dx:0.3;
eta = 0 : dx: L;
cd 'dx_0_01'
y_num2 = load('Cut_alongY_step2.txt');
y_numX2 = load('Cut_alongX_step2.txt');
y_numZ2 = load('Cut_alongZ_step2.txt');
cd ..

temp = find(y_num2==0)
i_min = 9;
i_max = 25;
y_num2 = y_num2(i_min:i_max);
y_numX2 = y_numX2(i_min:i_max);
y_numZ2 = y_numZ2(i_min:i_max);
y_anal2 = (sin(eta*(pi/L))*(L/pi)^2*Q)/k + C1*eta;


Figure1=figure(1);clf;set(Figure1,'defaulttextinterpreter','latex');
hold on;
set(gca,'fontsize',40,'fontname','Times','LineWidth',0.5); 
plot(eta, y_anal2,'b','linewidth',2);
plot(eta,y_num2,'r.','markersize',20);

legend('Analytic solution','Numerical results');
xlabel('$x [m]$');
ylabel('Temperature [°C]');
grid on;
box on;
err_2 = max(abs(y_num2-transpose(y_anal2)));

% %% dx == O.005
% dx = 0.005;
% x = 0:dx:0.3;
% eta = 0 : dx: L;
% cd 'dx_0_005'
% y_num3 = load('Cut_alongY_step2.txt');
% y_numX3 = load('Cut_alongX_step2.txt');
% y_numZ3 = load('Cut_alongZ_step2.txt');
% cd ..
% 
% temp = find(y_num3==0)
% i_min = 17;
% i_max = 49;
% y_num3 = y_num3(i_min:i_max);
% y_numX3 = y_numX3(i_min:i_max);
% y_numZ3 = y_numZ3(i_min:i_max);
% y_anal3 = (sin(eta*(pi/L))*(L/pi)^2*Q)/k + C1*eta;
% % hold on 
% % plot(eta, y_anal3,'linewidth',2);
% % plot(eta,y_num3,'b.','markersize',20);
% 
% err_3 = max(abs(y_num3-transpose(y_anal3)));
% 
% 
% 
% %% Compute error 
% figure;
% h =[0.02 0.01 0.005];
% err = [err_1 err_2 err_3];
% 
% Figure2=figure(2);clf;set(Figure2,'defaulttextinterpreter','latex');
% hold on;
% set(gca,'fontsize',40,'fontname','Times','LineWidth',0.5); 
% plot(log(h/L),log(err/50),'b','linewidth',2);
% plot(log(h/L),log(err/50),'r.','markersize',20);
% xlabel('$\log(\frac{dx}{L})\,[-]$');
% ylabel('$\log(\varepsilon/T_1)\,[-]$');
% temp = polyfit(log(h),log(err),1);
% box on;
% grid on;
% 
% Order = temp(1)
% 
% %% Profile of the temperature in the perpendicular direction
% eta1 = 0:0.02:L;
% eta2 = 0:0.01:L;
% eta3 = 0:0.005:L;
% 
% Figure3=figure(3);clf;set(Figure3,'defaulttextinterpreter','latex');
% hold on;
% set(gca,'fontsize',40,'fontname','Times','LineWidth',0.5); 
% plot(eta1,y_numX1,'b','linewidth',2);
% plot(eta2,y_numX2,'r','linewidth',2);
% plot(eta3,y_numX3,'g','linewidth',2);
% plot([0 L],[sin((L/2)*(pi/L))*(L/pi)^2*Q + (50/L)*(L/2) sin((L/2)*(pi/L))*(L/pi)^2*Q + (50/L)*(L/2)],'k','linewidth',2);
% xlabel('$y [m]$');
% ylabel('Temperature [°C]');
% legend('dx = 0.02','dx = 0.01','dx = 0.005', 'Exact solution','Location','eastoutside');
% axis([0 0.16 154.5 156.5]);
% grid on;
% box on;

end