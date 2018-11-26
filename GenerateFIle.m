NP=10;
NC = 8;
T=1;
xP=rand(1,NP)*20;
yP = rand(1,NP)*20;
v= rand(1,NP)*2;
xC=rand(1,NC)*20;
yC = rand(1,NC)*20;
figure()
scatter(xP,yP,'b');
h = labelpoints (xP, yP,v, 'FontSize', 11.5, 'Color', 'k');
hold on;
scatter(xC,yC);
grid on;
csvwrite('InM1.csv',[NP NC T]);
csvwrite('InM1.csv',[xP' yP' v']);
csvwrite('InM1.csv',[xC' yC']);