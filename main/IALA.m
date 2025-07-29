%______________________________________________________________________________________________
function [Score,Position,Convergence_curve]=IALA(X,N,Max_iter,lb,ub,dim,fobj)

Position=zeros(1,dim); % Best position
Score=inf; %Best score (initially infinite)
fitness=zeros(1,size(X,1));% Fitness of each individual
Convergence_curve=[];% Store convergence information
vec_flag=[1,-1]; % Directional flag
%% Record the initial optimal solution and fitness
for i=1:size(X,1)
    fitness(1,i)=fobj(X(i,:));
    if fitness(1,i)<Score
        Position=X(i,:); Score=fitness(1,i);
    end
end

[~,idx1]=sort(fitness);
second_best=X(idx1(2),:);
third_best=X(idx1(3),:);
sum1=0;
N1=N*0.5;
for i=1:N1
    sum1=sum1+X(idx1(i),:);
end

half_best_mean=sum1/N1;
Elite_pool(1,:)=Position;
Elite_pool(2,:)=second_best;
Elite_pool(3,:)=third_best;
Elite_pool(4,:)=half_best_mean;



for i=1:N
    index(i)=i;
end
Na=N*0.2;
Nb=N*0.8;
Xnew=X;
Iter=1; %Iteration number

%% Main optimization loop
while Iter<=Max_iter
    RB=randn(N,dim);  % Brownian motion
    F=vec_flag(floor(2*rand()+1)); % Random directional flag
    index1=randperm(N,Na);
    index2=setdiff(index,index1);
    if rand>0.5
        w = 1/(pi*exp(1))*cos(rand);
    else
        w = 1/(pi*exp(1))*sin(rand);
    end
    for i=1:Na

        for j=1:size(X,2) 
            if rand>0.5
                if rand<0.3
                    r1 = 2 * rand(1,dim) - 1;
                    Xnew(index1(i),:)= w*Elite_pool(1,:)+F.*RB(index1(i),:).*(r1.*(Position-X(index1(i),:))+(1-r1).*(X(index1(i),:)-X(randi(N),:)));
                else
                    if rand>rand
                        r2 = rand ()* (1 + sin(0.5 * Iter));
                        Xnew(index1(i),:)= w*Elite_pool(1,:)+ F.* r2*(Position-X(randi(N),:));
                    else
                        learning_rate = 0.01; 
                        current_position = X(index1(i), :);
                        gradient = calculate_gradient(current_position,fobj); 
                        Xnew(index1(i), :) = current_position - learning_rate * gradient; 
                    end
                end

            else
                k=1;
                L = 2*rand-1;
                if rand>0.5
                    a=cos(pi*(1-(Iter/Max_iter)));
                else
                    a=sin(pi*(1-(Iter/Max_iter)));
                end
                z = exp(k*a);
                Xnew(index1(i),:)= w*Elite_pool(1,:)+exp(z*L)*cos(2*pi*L)*abs(Position-Xnew(index1(i),:));
            end

        end
    end
    if Na<N
        Na=Na+1;
        Nb=Nb-1;
    end

    if Nb>=1
        for i=1:Nb
            k1=randperm(4,1);
            for j=1:size(X,2) 
                if rand>0.5
                    if rand<0.5
                        radius = sqrt(sum((Position-X(index2(i), :)).^2));
                        r3=rand();
                        spiral=radius*(sin(2*pi*r3)+cos(2*pi*r3));
                        Xnew(index2(i),:) =w*Elite_pool(k1,:) + F.* X(index2(i),:).*spiral*rand;
                    else
                        G=2*(sign(rand-0.5))*(1-Iter/Max_iter);
                        Xnew(index2(i),:) = w*Elite_pool(k1,:) + F.* G*Levy(dim).* (Position - Xnew(index2(i),:)) ;

                    end
                else
                    rg=0.1-((0.1)*Iter/Max_iter);
                    r=rand*rg;
                    L = Position-Xnew(index2(i),:);
                    LP = L.*rand(1,dim);
                    alph = L.*L+LP.*LP-2*LP.*L.*cos(2*pi*rand(1,dim));
                    Xnew(index2(i),:) = w*Elite_pool(k1,:)+r*alph.*randn(size(Xnew(index2(i),:)));

                end

            end
        end
    end




    %%  Boundary check and evaluation
    for i=1:size(X,1)
        Flag4ub=Xnew(i,:)>ub;
        Flag4lb=Xnew(i,:)<lb;
        Xnew(i,:)=(Xnew(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % Boundary correction
        newPopfit=fobj(Xnew(i,:)); % Evaluate new solution
        if newPopfit<fitness(i)
            X(i,:)=Xnew(i,:);
            fitness(1,i)=newPopfit;
        end
        if fitness(1,i)<Score
            Position=X(i,:);
            Score=fitness(1,i);
        end
    end
    %% Record convergence curve
    Convergence_curve(Iter)=Score;


    [~,idx1]=sort(fitness);
    second_best=X(idx1(2),:);
    third_best=X(idx1(3),:);
    sum1=0;
    for i=1:N1
        sum1=sum1+X(idx1(i),:);
    end

    half_best_mean=sum1/N1;
    Elite_pool(1,:)=Position;
    Elite_pool(2,:)=second_best;
    Elite_pool(3,:)=third_best;
    Elite_pool(4,:)=half_best_mean;


    Iter=Iter+1;
end

end
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end

function grad = calculate_gradient(x, fobj)
    epsilon = 1e-6; %微小偏移量
    grad = zeros(size(x));
    for i = 1:length(x)
        %计算偏导数的分母
        x_plus = x;
        x_plus(i) = x_plus(i) + epsilon;
        
        x_minus = x;
        x_minus(i) = x_minus(i) - epsilon;
        
        %计算中心差分的偏导数
        grad(i) = (fobj(x_plus) - fobj(x_minus)) / (2 * epsilon);
    end
end

