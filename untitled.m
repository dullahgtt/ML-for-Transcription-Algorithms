clear all
close all

x0 = 1
x1 = 2

for i =1:10
    fx0=x0^2-2;
    fx1=x1^2-2;
    x2 = x1-fx1*(x1-x0)/(fx1-fx0)
    x0=x1;
    x1=x2;
end

x0 = 1
x1 = 1
for i=1:10
    x2 = .5*(x1+2/x1)
    x1=x2;
end