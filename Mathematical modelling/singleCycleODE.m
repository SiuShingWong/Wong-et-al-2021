function [] = singleCycleODE()
konR       = 10;
koffR      = 20;
konP       = 100;
koffP      = 50;
kcatP      = 20;
kdisP      = 10;
konS       = 20;
koffS      = 200;
kcatS      = 100;
kdisSbar   = 1;
konC       = 100;
koffC      = 100;
kcatC      = 50;
kdisS      = 20;
kdisC      = 0.05;
deltaP     = 10;
deltaS     = 0.001;
deltaC     = 0.001;
deltaR     = 5;
Cinit      = 1;

tspan = 0:0.01:1;

poloFun = @(t) 0.5*(1 - cos(2*pi*t))';

V0Model1 = [0 0 0 0];

[~,Vout] = ode23(@(t,V) model1(t,V,1,1,konS,deltaS,konC,deltaC),tspan,V0Model1);

V0Model1 = [Vout(end,1), Vout(end,2), Vout(end,3), Cinit];

[~,Vout] = ode23(@(t,V) model1(t,V,1,1,konS,deltaS,konC,deltaC),tspan,V0Model1);

spdScaffold  = Vout(:,1) + Vout(:,2) + Vout(:,3);
cnnScaffold  = Vout(:,3) + Vout(:,4);
poloScaffold = poloFun(tspan) + deltaP*Vout(:,3);
spdScaffold  = spdScaffold/max(spdScaffold);
cnnScaffold  = cnnScaffold/max(cnnScaffold);
poloScaffold = poloScaffold/max(poloScaffold);

V0Model2 = [0, 0, 0, 0, 1, 0, 0];

[~,VoutModel2] = ode23(@(t,V) model2(t,V,1,1,konS,deltaS,konC,deltaC,koffR),tspan,V0Model2);

V0Model2 = [VoutModel2(end,1), VoutModel2(end,2), VoutModel2(end,3), Cinit, 1 - VoutModel2(end,6) - VoutModel2(end,7), VoutModel2(end,6), VoutModel2(end,7)];

[~,VoutModel2] = ode23(@(t,V) model2(t,V,1,1,konS,deltaS,konC,deltaC,koffR),tspan,V0Model2);

spdScaffoldModel2  = VoutModel2(:,1) + VoutModel2(:,2) + VoutModel2(:,3);
cnnScaffoldModel2  = VoutModel2(:,3) + VoutModel2(:,4);
poloScaffoldModel2 = VoutModel2(:,7) + deltaR*VoutModel2(:,3);
poloActivityModel2 = VoutModel2(:,7);
spdScaffoldModel2  = spdScaffoldModel2/max(spdScaffoldModel2);
cnnScaffoldModel2  = cnnScaffoldModel2/max(cnnScaffoldModel2);
poloMax            = max(poloScaffoldModel2);
poloScaffoldModel2 = poloScaffoldModel2/max(poloScaffoldModel2);
poloActivityModel2 = poloActivityModel2/max(poloActivityModel2);

[~,VoutModel2HalfAna1] = ode23(@(t,V) model2(t,V,1,1,konS,deltaS,konC,deltaC,0.5*koffR),tspan,V0Model2);

poloScaffoldModel2HalfAna1 = VoutModel2HalfAna1(:,7) + 2*deltaR*VoutModel2HalfAna1(:,3);
poloScaffoldModel2HalfAna1 = 0.5*poloScaffoldModel2HalfAna1/poloMax;

[~,VoutModel2HalfSpd2] = ode23(@(t,V) model2(t,V,1,1,0.5*konS,2*deltaS,konC,deltaC,koffR),tspan,V0Model2);

poloScaffoldModel2HalfSpd2 = VoutModel2HalfSpd2(:,7) + deltaR*VoutModel2HalfSpd2(:,3);
poloScaffoldModel2HalfSpd2 = poloScaffoldModel2HalfSpd2/poloMax;

V0Model3 = [0, 0, 0, 0, 1, 0, 0, 0];

[~,VoutModel3] = ode23(@(t,V) model3(t,V,1,1,konS,deltaS,konC,deltaC,koffR),tspan,V0Model3);

V0Model3 = [VoutModel3(end,1), VoutModel3(end,2), VoutModel3(end,3), Cinit, 1 - VoutModel3(end,6) - VoutModel3(end,7), VoutModel3(end,6), VoutModel3(end,7), VoutModel3(end,8)];

[~,VoutModel3] = ode23(@(t,V) model3(t,V,1,1,konS,deltaS,konC,deltaC,koffR),tspan,V0Model3);

spdScaffoldModel3  = VoutModel3(:,1) + VoutModel3(:,2) + VoutModel3(:,3);
cnnScaffoldModel3  = VoutModel3(:,3) + VoutModel3(:,4);
poloScaffoldModel3 = VoutModel3(:,8) + VoutModel3(:,7) + deltaR*VoutModel3(:,3);
poloActivityModel3 = VoutModel3(:,8) + VoutModel3(:,7);
spdScaffoldModel3  = spdScaffoldModel3/max(spdScaffoldModel3);
cnnScaffoldModel3  = cnnScaffoldModel3/max(cnnScaffoldModel3);
poloScaffoldModel3 = poloScaffoldModel3/max(poloScaffoldModel3);
poloActivityModel3 = poloActivityModel3/max(poloActivityModel3);
      
% Model 1
figure(1)
clf
hold on
plot(tspan,spdScaffold,'color',[231/255 147/255 61/255],'linewidth',2)
plot(tspan,cnnScaffold/max(cnnScaffold),'color',[32/255 122/255 184/255],'linewidth',2)
plot(tspan,poloScaffold,'color',[0 164/255 82/255],'linewidth',2)
plot(tspan,poloFun(tspan)/max(poloFun(tspan)),'linestyle','--','color',[0 164/255 82/255],'linewidth',2)
legend('Spd2','Cnn','Total Polo','Centriolar Polo','location','southeast')

ylim([0 1])
xlim([0 1])
xlabel('Time')
ylabel('Fluorescence (A.U.)')
set(gca,'linewidth',2,'Fontsize',20)

% Model 2
figure(2)
clf
hold on
plot(tspan,spdScaffoldModel2,'color',[231/255 147/255 61/255],'linewidth',2)
plot(tspan,cnnScaffoldModel2/max(cnnScaffoldModel2),'color',[32/255 122/255 184/255],'linewidth',2)
plot(tspan,poloScaffoldModel2,'color',[0 164/255 82/255],'linewidth',2)
plot(tspan,poloActivityModel2,'color',[0 164/255 82/255],'linestyle','--','linewidth',2)
legend('Spd2','Cnn','Total Polo','Centriolar Polo','location','southeast')

ylim([0 1])
xlim([0 1])
xlabel('Time')
ylabel('Fluorescence (A.U.)')
set(gca,'linewidth',2,'Fontsize',20)

% Model 3
figure(3)
clf
hold on
plot(tspan,spdScaffoldModel3,'color',[231/255 147/255 61/255],'linewidth',2)
plot(tspan,cnnScaffoldModel3/max(cnnScaffoldModel3),'color',[32/255 122/255 184/255],'linewidth',2)
plot(tspan,poloScaffoldModel3,'color',[0 164/255 82/255],'linewidth',2)
plot(tspan,poloActivityModel3,'color',[0 164/255 82/255],'linestyle','--','linewidth',2)
legend('Spd2','Cnn','Total Polo','Centriolar Polo','location','southeast')

ylim([0 1])
xlim([0 1])
xlabel('Time')
ylabel('Fluorescence (A.U.)')
set(gca,'linewidth',2,'Fontsize',20)

% Half dose Ana1
figure(4)
clf
hold on
plot(tspan,poloScaffoldModel2,'color',[0 164/255 82/255],'linestyle','-','linewidth',2)
plot(tspan,poloScaffoldModel2HalfAna1,'color',[0 164/255 82/255],'linestyle','--','linewidth',2)
legend('Total Polo 1x Ana1','Total Polo 0.5x Ana1','location','southeast')

ylim([0 1])
xlim([0 1])
xlabel('Time')
ylabel('Fluorescence (A.U.)')
set(gca,'linewidth',2,'Fontsize',20)

% Half dose Spd2
figure(5)
clf
hold on
plot(tspan,poloScaffoldModel2,'color',[0 164/255 82/255],'linestyle','-','linewidth',2)
plot(tspan,poloScaffoldModel2HalfSpd2,'color',[0 164/255 82/255],'linestyle','--','linewidth',2)
legend('Total Polo 1x Spd2','Total Polo 0.5x Spd2','location','southeast')

ylim([0 1])
xlim([0 1])
xlabel('Time')
ylabel('Fluorescence (A.U.)')
set(gca,'linewidth',2,'Fontsize',20)

    
    function dV = model1(t,Vin,Scyt_fun,Ccyt_fun,konS,deltaS,konC,deltaC)
        rS       = Vin(1);
        sstar    = Vin(2);
        sbar     = Vin(3);
        cstar    = Vin(4);
        
        r = 1 - rS;
        S = Scyt_fun - deltaS*(sstar + sbar + rS);
        C = Ccyt_fun - deltaC*(cstar + sbar);
        dV = [konS*r*S - (koffS + kcatS*poloFun(t))*rS;
              kcatS*poloFun(t)*rS - (konC*C +kdisS)*sstar + (koffC + kcatC)*sbar;
              konC*C*sstar - (koffC + kcatC)*sbar - kdisSbar*sbar;
              kcatC*sbar - kdisC*cstar^2];
    end

    function dV = model2(t,Vin,Scyt_fun,Ccyt_fun,konS,deltaS,konC,deltaC,koffR)
        rS       = Vin(1);
        sstar    = Vin(2);
        sbar     = Vin(3);
        cstar    = Vin(4);
        rPoff    = Vin(5);
        rP       = Vin(6);
        rPbar    = Vin(7);

        r = 1 - rS;
        S = Scyt_fun - deltaS*(sstar + sbar + rS);
        C = Ccyt_fun - deltaC*(cstar + sbar);
        dV = [konS*r*S - (koffS + kcatS*rPbar)*rS;
              kcatS*rPbar*rS - (konC*C +kdisS)*sstar + (koffC + kcatC)*sbar;
              konC*C*sstar - (koffC + kcatC)*sbar - kdisSbar*sbar;
              kcatC*sbar - kdisC*cstar^2;
              -konR*rPoff;
              konR*rPoff - konP*rP + koffP*rPbar - koffR*rPbar*rP;
              konP*rP - koffP*rPbar];
    end

    function dV = model3(t,Vin,Scyt_fun,Ccyt_fun,konS,deltaS,konC,deltaC,koffR)
        rS       = Vin(1);
        sstar    = Vin(2);
        sbar     = Vin(3);
        cstar    = Vin(4);
        rPoff    = Vin(5);
        rP       = Vin(6);
        rPbar    = Vin(7);
        Pstar    = Vin(8);
        Pact     = Pstar + rPbar;

        r = 1 - rS;
        S = Scyt_fun - deltaS*(sstar + sbar + rS);
        C = Ccyt_fun - deltaC*(cstar + sbar);
        dV = [konS*r*S - (koffS + kcatS*Pact)*rS;
              kcatS*Pact*rS - (konC*C +kdisS)*sstar + (koffC + kcatC)*sbar;
              konC*C*sstar - (koffC + kcatC)*sbar - kdisSbar*sbar;
              kcatC*sbar - kdisC*cstar^2;
              -konR*rPoff;
              konR*rPoff - konP*rP + (koffP + kcatP)*rPbar - koffR*Pact*rP;
              konP*rP - (koffP + kcatP)*rPbar;
              kcatP*rPbar - kdisP*Pstar];
    end

end