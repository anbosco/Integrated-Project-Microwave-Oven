function Test_order_temp
close all
clear all
cd dt_1
Temperature1 = load('Probe1_step0_to_step100.txt');
cd ..

cd dt_0_1
Temperature2 = load('Probe1_step0_to_step1000.txt');
cd ..

cd dt0_05
Temperature3 = load('Probe1_step0_to_step2000.txt');
cd ..

cd dt_0_025
Temperature4 = load('Probe1_step0_to_step4000.txt');
cd ..

dt1 = 1;
dt2 = 0.1;
dt3 = 0.05;
dt4 = 0.025;

t1 = 0:dt1:100-dt1;
t2 = 0:dt2:100-dt2;
t3 = 0:dt3:100-dt3;
t4 = 0:dt4:100-dt4;
hold on;
plot(t1,Temperature1,'r.','Markersize',15);
plot(t2,Temperature2,'r.','Markersize',15);
plot(t3,Temperature3,'k.','Markersize',15);

Q = 50000;
rho = 1;
cp = 1;
omega = (2*3.141592)/20;
T_anal1 = (Q/(rho*cp*omega))*sin(omega*t1);
T_anal2 = (Q/(rho*cp*omega))*sin(omega*t2);
T_anal3 = (Q/(rho*cp*omega))*sin(omega*t3);
T_anal4 = (Q/(rho*cp*omega))*sin(omega*t4);

err1 = max(abs(transpose(T_anal1)-Temperature1))
err2 = max(abs(transpose(T_anal2)-Temperature2))
err3 = max(abs(transpose(T_anal3)-Temperature3))
err4 = max(abs(transpose(T_anal4)-Temperature4))


plot(t3,T_anal3,'b');
end