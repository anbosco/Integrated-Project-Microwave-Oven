%% This function plot the results for the homogeneous wave guide
function plot_impulse_evo
dx = 0.005
x=0:dx:0.6;
x = transpose(x);       % To plot Ey
x2 = 0:dx:0.6+dx;       % To plot Hz or Hx
x2 = transpose(x2);
cd Cut
%% Ey
load('Cut_Ey_alongX_step50_rank0.txt'); 
load('Cut_Ey_alongX_step50_rank4.txt');

y1_Ey = [Cut_Ey_alongX_step50_rank0; Cut_Ey_alongX_step50_rank4];
 
load('Cut_Ey_alongX_step100_rank0.txt');
load('Cut_Ey_alongX_step100_rank4.txt');
y2_Ey = [Cut_Ey_alongX_step100_rank0; Cut_Ey_alongX_step100_rank4];

load('Cut_Ey_alongX_step150_rank0.txt');
load('Cut_Ey_alongX_step150_rank4.txt');
y3_Ey = [Cut_Ey_alongX_step150_rank0; Cut_Ey_alongX_step150_rank4];
 
load('Cut_Ey_alongX_step200_rank0.txt'); 
load('Cut_Ey_alongX_step200_rank4.txt');
y4_Ey = [Cut_Ey_alongX_step200_rank0; Cut_Ey_alongX_step200_rank4];

load('Cut_Ey_alongX_step300_rank0.txt');
load('Cut_Ey_alongX_step300_rank4.txt');
y5_Ey = [Cut_Ey_alongX_step300_rank0; Cut_Ey_alongX_step300_rank4];

% %%Hx
% load('Cut_Hx_alongX_step25_rank0.txt');
% load('Cut_Hx_alongX_step25_rank4.txt');
% y1_Hx = [Cut_Hx_alongX_step25_rank0; Cut_Hx_alongX_step25_rank4];
% 
% load('Cut_Hx_alongX_step50_rank0.txt');
% load('Cut_Hx_alongX_step50_rank4.txt');
% y2_Hx = [Cut_Hx_alongX_step50_rank0; Cut_Hx_alongX_step50_rank4];
% 
% load('Cut_Hx_alongX_step75_rank0.txt');
% load('Cut_Hx_alongX_step75_rank4.txt');
% y3_Hx = [Cut_Hx_alongX_step75_rank0; Cut_Hx_alongX_step75_rank4];
% 
% load('Cut_Hx_alongX_step150_rank0.txt');
% load('Cut_Hx_alongX_step150_rank4.txt');
% y5_Hx = [Cut_Hx_alongX_step150_rank0; Cut_Hx_alongX_step150_rank4];
% 
% %% Hz
% load('Cut_Hz_alongX_step25_rank0.txt');
% load('Cut_Hz_alongX_step25_rank4.txt');
% y1_Hz = [Cut_Hz_alongX_step25_rank0; Cut_Hz_alongX_step25_rank4];
% 
% load('Cut_Hz_alongX_step50_rank0.txt');
% load('Cut_Hz_alongX_step50_rank4.txt');
% y2_Hz = [Cut_Hz_alongX_step50_rank0; Cut_Hz_alongX_step50_rank4];
% 
% load('Cut_Hz_alongX_step75_rank0.txt');
% load('Cut_Hz_alongX_step75_rank4.txt');
% y3_Hz = [Cut_Hz_alongX_step75_rank0; Cut_Hz_alongX_step75_rank4];
%  
% load('Cut_Hz_alongX_step150_rank0.txt');
% load('Cut_Hz_alongX_step150_rank4.txt');
% y5_Hz = [Cut_Hz_alongX_step150_rank0; Cut_Hz_alongX_step150_rank4];
cd ..

Figure1=figure(1);clf;set(Figure1,'defaulttextinterpreter','latex');
hold on;
subplot(2,2,1)
set(gca,'fontsize',150,'fontname','Times','LineWidth',0.5);
plot(x,y1_Ey,'r','linewidth',2);
xlabel('$x [m]$');
ylabel('$E_y [V\cdot m^{-1}]$');
ylim([-150 150]);
box on;
grid on;

subplot(2,2,2)
set(gca,'fontsize',80,'fontname','Times','LineWidth',0.5);
plot(x,y2_Ey,'r','linewidth',2);
xlabel('$x [m]$');
ylabel('$E_y [V\cdot m^{-1}]$');
ylim([-150 150]);
box on;
grid on;

subplot(2,2,3)
set(gca,'fontsize',80,'fontname','Times','LineWidth',0.5);
plot(x,y4_Ey,'r','linewidth',2);
xlabel('$x [m]$');
ylabel('$E_y [V\cdot m^{-1}]$');
ylim([-150 150]);
box on;
grid on;

subplot(2,2,4)
hold on
set(gca,'fontsize',80,'fontname','Times','LineWidth',0.5);
plot(x,y5_Ey,'r','linewidth',2);
plot([0.450 0.450], [-150 150],'k-.','linewidth',2)
xlabel('$x [m]$');
ylabel('$E_y [V\cdot m^{-1}]$');
ylim([-150 150]);
box on;
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',30)
end