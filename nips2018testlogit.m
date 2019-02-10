%% LAG - logistic regression real data
clear all
close all

%% data allocation for linear regression
[Xdata_28] = load('data2/data.txt'); % UCLA Housing dataset
[ydata_28] = load('data2/y.txt'); 
[Xdata_29] = load('data9/data.txt'); % Body Fat dataset
[ydata_29] = load('data9/y.txt');   
[Xdata_30] = load('data11/data.txt'); % Age of abalone dataset
[ydata_30] = load('data11/y.txt');   

accuracy=1e-8;
num_iter=4000;
num_split=6;
num_workers=num_split*3;
X=cell(num_workers);
y=cell(num_workers);

num_feature=min(min(size(Xdata_28,2),size(Xdata_29,2)),size(Xdata_30,2));
num_sample=size(Xdata_28,1);
per_split=floor(num_sample/num_split);

for n=1:num_split
X{n}=Xdata_28(per_split*(n-1)+1:per_split*n,1:num_feature);
y{n}=ydata_28(per_split*(n-1)+1:per_split*n);
end

num_sample=size(Xdata_29,1);
per_split=floor(num_sample/num_split);
for n=1:num_split
X{n+num_split}=Xdata_29(per_split*(n-1)+1:per_split*n,1:num_feature);
y{n+num_split}=ydata_29(per_split*(n-1)+1:per_split*n);
end

num_sample=size(Xdata_30,1);
per_split=floor(num_sample/num_split);
for n=1:num_split
X{n+2*num_split}=Xdata_30(per_split*(n-1)+1:per_split*n,1:num_feature);
y{n+2*num_split}=ydata_30(per_split*(n-1)+1:per_split*n);
end


X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

% [lambda]=eig(Xdata'*Xdata);
Hmax=zeros(num_workers,1);
% for i=1:num_workers
%    X{i}=1^(i-1)*Q(:,i)*Q(:,i)'+diag(ones(total_sample,1));
%    Hmax(i)=max(abs(eig(X{i}'*X{i}))); 
% end

for i=1:num_workers
   Hmax(i)=max(eig(X{i}'*X{i})); 
end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
nonprob=Hmax/Hmax_sum;

%% data pre-analysis
lambda=0.0001;
Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=0.25*max(abs(eig(X{i}'*X{i})))+lambda; 
end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
nonprob=Hmax/Hmax_sum;

Hmin=zeros(num_workers,1);
Hcond=zeros(num_workers,1);
for i=1:num_workers
   Hmin(i)=lambda; 
   Hcond(i)=Hmax(i)/Hmin(i);
end

X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

triggerslot=10;
Hmaxall=0.25*max(eig(X_fede'*X_fede))+lambda;
[cdff,cdfx] = ecdf(Hmax*num_workers/Hmaxall);
comm_save=0;
for i=1:triggerslot
    comm_save=comm_save+(1/i-1/(i+1))*cdff(find(cdfx>=min(max(cdfx),sqrt(1/(triggerslot*i))),1));
end

heterconst=mean(exp(Hmax/Hmaxall));
heterconst2=mean(Hmax/Hmaxall);
rate=1/(1+sum(Hmin)/(4*sum(Hmax)));
%% parameter initialization
%triggerslot=100;
theta=zeros(num_feature,num_iter);
grads=ones(num_feature,num_workers);
%stepsize=1/(num_workers*max(Hmax));
stepsize=1/Hmaxall;
thrd=10/(stepsize^2*num_workers^2)/triggerslot;
comm_count=ones(num_workers,1);

theta2=zeros(num_feature,num_iter);
grads2=ones(num_feature,1);
stepsize2=stepsize;

theta3=zeros(num_feature,num_iter);
grads3=ones(num_feature,num_workers);
stepsize3=stepsize2/num_workers; % cyclic access learning

theta4=zeros(num_feature,num_iter);
grads4=ones(num_feature,num_workers);
stepsize4=0.5*stepsize/sqrt(num_workers); % nonuniform-random access learning


thrd5=1/(stepsize^2*num_workers^2)/triggerslot;
theta5=zeros(num_feature,1);
grads5=ones(num_feature,num_workers);
stepsize5=stepsize;
comm_count5=ones(num_workers,1);

%thrd6=2/(stepsize*num_workers);
theta6=zeros(num_feature,1);
grads6=ones(num_feature,1);
stepsize6=0.5*stepsize;
comm_count6=ones(num_workers,1);


theta7=zeros(num_feature,1);
grads7=ones(num_feature,num_workers);
stepsize7=stepsize;
comm_count7=ones(num_workers,1);

% lambda=0.000;
%%  GD
comm_error2=[];
comm_grad2=[];
for iter=1:num_iter*2
    if mod(iter,100)==0
        iter
    end
    % central server computation
    if iter>1
    grads2=-(X_fede'*(y_fede./(1+exp(y_fede.*(X_fede*theta2(:,iter))))))+num_workers*lambda*theta2(:,iter);
        end
    grad_error2(iter)=norm(sum(grads2,2),2);

    loss2(iter)=num_workers*lambda*0.5*norm(theta2(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta2(:,iter)))));
    theta2(:,iter+1)=theta2(:,iter)-stepsize2*grads2;
    comm_error2=[comm_error2;iter*num_workers,loss2(iter)]; 
    comm_grad2=[comm_grad2;iter*num_workers,grad_error2(iter)]; 
end

for iter=i:num_iter
   if abs(loss2(iter)-loss2(end))<accuracy
    fprintf('Communication rounds of GD\n');
       iter*num_workers  
       break
   end
end

%% LAG-PS
comm_iter=1;
comm_index=zeros(num_workers,num_iter);
comm_error=[];
comm_grad=[];
theta_temp=zeros(num_feature,num_workers);
for iter=1:num_iter
    
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta(:,iter-(n-1))-theta(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if Hmax(i)^2*norm(theta_temp(:,i)-theta(:,iter),2)^2>thrd*trigger
                grads(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta(:,iter))))))+lambda*theta(:,iter);
                theta_temp(:,i)=theta(:,iter);
                comm_index(i,iter)=1;
                comm_count(i)=comm_count(i)+1;
                comm_iter=comm_iter+1;
                comm_flag=1;
            end
        end
    end
    
    % central server computation
    grad_error(iter)=norm(sum(grads,2),2);
    loss(iter)=num_workers*lambda*0.5*norm(theta(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta(:,iter)))));
    theta(:,iter+1)=theta(:,iter)-stepsize*sum(grads,2);
    

    if comm_flag==1
        comm_error=[comm_error;comm_iter,loss(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    elseif  mod(iter,1000)==0
        iter
        comm_iter=comm_iter+1;
        comm_error=[comm_error;comm_iter,loss(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    end
    if abs(loss(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of LAG-PS\n');
        comm_iter
        break
    end
end

%% LAG-WK
comm_iter5=1;
comm_index5=zeros(num_workers,num_iter);
comm_error5=[];
comm_grad5=[];
for iter=1:num_iter

    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        grad_temp=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta5(:,iter))))))+lambda*theta5(:,iter);
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta5(:,iter-(n-1))-theta5(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if norm(grad_temp-grads5(:,i),2)^2>thrd5*trigger
                grads5(:,i)=grad_temp;
                comm_count5(i)=comm_count5(i)+1;
                comm_index5(i,iter)=1;
                comm_iter5=comm_iter5+1;
                comm_flag=1;
            end
        end       
    end
    grad_error5(iter)=norm(sum(grads5,2),2);
    loss5(iter)=num_workers*lambda*0.5*norm(theta5(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta5(:,iter)))));
    if comm_flag==1
       comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    elseif  mod(iter,1000)==0
        iter
        comm_iter5=comm_iter5+1; 
        comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    end
    theta5(:,iter+1)=theta5(:,iter)-stepsize5*sum(grads5,2);
    if abs(loss5(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of LAG-WK\n');
        comm_iter5
        break
    end
end

%% cyclic IAG
for iter=1:num_iter*floor(num_workers)
    if mod(iter,100)==0
        iter
    end
    if iter>1
    % local worker computation
    i=mod(iter,num_workers)+1;
    grads3(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta3(:,iter))))))+lambda*theta3(:,iter);
    end
    % central server computation
    grad_error3(iter)=norm(sum(grads3,2),2);
    loss3(iter)=num_workers*lambda*0.5*norm(theta3(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta3(:,iter)))));
    theta3(:,iter+1)=theta3(:,iter)-stepsize3*sum(grads3,2);
    
    if abs(loss3(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of Cyc-IAG\n');
        iter
        break
    end
end

%% non-uniform RANDOMIZED IAG
for iter=1:num_iter*floor(num_workers)
    if mod(iter,100)==0
        iter
    end
    % local worker computation
    workprob=rand;
    for i=1:num_workers
        if workprob<=sum(nonprob(1:i));
           break;
        end
    end
    %i=randi(num_workers);   
    if iter>1
    grads4(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta4(:,iter))))))+lambda*theta4(:,iter);
    end
    % central server computation
    grad_error4(iter)=norm(sum(grads4,2),2);
    loss4(iter)=num_workers*lambda*0.5*norm(theta4(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta4(:,iter)))));
    theta4(:,iter+1)=theta4(:,iter)-stepsize4*sum(grads4,2);
    
    if abs(loss4(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of Num-IAG\n');
        iter
        break
    end
end

%% CoCoA
% lambda=lambda*num_workers;
comm_iter6=1;
comm_index6=zeros(num_workers,num_iter);
comm_error6=[];
comm_grad6=[];
dual_wk=cell(num_workers);
dualgrads=cell(num_workers);
% iner_num=5000;
% for i=1:num_workers
%     dual_wk{i}=zeros(size(X{i},1),1);
%     dualgrads{i}=zeros(size(X{i},1),1);
% end

% for iter=1:num_iter
%     if mod(iter,100)==0
%         iter
%     end
%     dualgrads=cell(num_workers);
%     iner_num=10;
%     if iter>1
%         for i=1:num_workers
%             dualgrads{i}=zeros(size(X{i},1),1);
%             for n=1:iner_num
%                 intergrads6=-(1/(lambda*num_workers)*(X{i}*X{i}')+1)*dualgrads{i}+dual_wk{i}-1/(lambda*num_workers)*X{i}*theta6(:,iter)+y{i};
%                 dualgrads{i}=dualgrads{i}+0.0000001*intergrads6;
%             end
% %             grads6(:,i)=X{i}'*dualgrads{i}/lambda;
%             grads6(:,i)=X{i}'*dualgrads{i};
%             dual_wk{i}=dual_wk{i}+dualgrads{i};
%         end
%     end
%     grad_error6(iter)=norm(sum(grads6,2),2);
%     loss6(iter)=0.5*norm(X_fede*theta6(:,iter)-y_fede,2)^2+0.5*lambda*num_workers*norm(theta6(:,iter),2)^2;
%     theta6(:,iter+1)=theta6(:,iter)+sum(grads6,2);
%     comm_error6=[comm_error6;iter*num_workers,loss6(iter)]; 
%     comm_grad6=[comm_grad6;iter*num_workers,grad_error6(iter)]; 
% end


%% COCOA-low comp
comm_iter7=1;
comm_index7=zeros(num_workers,num_iter);
comm_error7=[];
comm_grad7=[];
dual_wk=cell(num_workers);
dualgrads=cell(num_workers);
% iner_num=5000;
% for i=1:num_workers
%     dual_wk{i}=zeros(size(X{i},1),1);
%     dualgrads{i}=zeros(size(X{i},1),1);
% end
% 
% for iter=1:num_iter
%     if mod(iter,100)==0
%         iter
%     end
%     dualgrads=cell(num_workers);
%     iner_num_low=1;
%     if iter>1
%         for i=1:num_workers
%             dualgrads{i}=zeros(size(X{i},1),1);
%             for n=1:iner_num_low
%                 intergrads7=-(1/(lambda)*(X{i}*X{i}')+1)*dualgrads{i}+dual_wk{i}-X{i}*theta7(:,iter)+y{i};
%                 dualgrads{i}=dualgrads{i}+0.00000001*intergrads7;
%             end
% %             grads6(:,i)=X{i}'*dualgrads{i}/lambda;
%             grads7(:,i)=X{i}'*dualgrads{i}/lambda/num_workers;
%             dual_wk{i}=dual_wk{i}+dualgrads{i}/num_workers;
%         end
%     end
%     grad_error7(iter)=norm(sum(grads7,2),2);
%     loss7(iter)=0.5*norm(X_fede*theta7(:,iter)-y_fede,2)^2+0.5*lambda*num_workers*norm(theta7(:,iter),2)^2;
%     theta7(:,iter+1)=theta7(:,iter)+sum(grads7,2);
%     comm_error7=[comm_error7;iter*num_workers,loss7(iter)]; 
%     comm_grad7=[comm_grad7;iter*num_workers,grad_error7(iter)]; 
% end

%% figure
% figure
% semilogy(grad_error,'r-','LineWidth',2);
% hold on
% semilogy(grad_error3,'k-','LineWidth',2);
% hold on
% semilogy(grad_error4,'g--','LineWidth',4);
% hold on
% semilogy(grad_error5,'r--','LineWidth',2);
% hold on
% % semilogy(grad_error6,'c-','LineWidth',2);
% % hold on
% % semilogy(grad_error7,'c--','LineWidth',2);
% % hold on
% semilogy(grad_error2,'b-','LineWidth',2);
% xlabel('Number of iteration','fontsize',16,'fontname','Times New Roman')
% ylabel('Gradient residual','fontsize',16,'fontname','Times New Roman')
% % legend('Cyclic-IAG','Non-IAG','LAG-PS','LAG-WK','CoCoA','CoCoA-low','Batch-GD')
% legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')

figure
semilogy(abs(loss3-loss2(end)),'k-','LineWidth',2);
hold on
semilogy(abs(loss4-loss2(end)),'g--','LineWidth',4);
hold on
semilogy(abs(loss-loss2(end)),'r-','LineWidth',2);
hold on
semilogy(abs(loss5-loss2(end)),'r--','LineWidth',2);
hold on
% semilogy(loss6-loss2(end),'c-','LineWidth',2);
% hold on
% semilogy(loss7-loss2(end),'c--','LineWidth',2);
% hold on
semilogy(abs(loss2-loss2(end)),'b-','LineWidth',2);
xlabel('Number of iteration','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')


% figure
% semilogy(grad_error3,'k-','LineWidth',2);
% hold on
% semilogy(grad_error4,'g--','LineWidth',4);
% hold on
% semilogy(comm_grad(:,1),comm_grad(:,2),'r-','LineWidth',2);
% hold on
% semilogy(comm_grad5(:,1),comm_grad5(:,2),'r--','LineWidth',2);
% hold on
% % semilogy(comm_grad6(:,1),comm_grad6(:,2),'c-','LineWidth',2);
% % hold on
% % semilogy(comm_grad7(:,1),comm_grad7(:,2),'c--','LineWidth',2);
% % hold on
% semilogy(comm_grad2(:,1),comm_grad2(:,2),'b-','LineWidth',2);
% xlabel('Number of communications (uploads)','fontsize',16,'fontname','Times New Roman')
% ylabel('Gradient residual','fontsize',16,'fontname','Times New Roman')
% legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')


figure
semilogy(abs(loss3-loss2(end)),'k-','LineWidth',2);
hold on
semilogy(abs(loss4-loss2(end)),'g--','LineWidth',4);
hold on
semilogy(comm_grad(:,1),abs(comm_error(:,2)-loss2(end)),'r-','LineWidth',2);
hold on
semilogy(comm_grad5(:,1),abs(comm_error5(:,2)-loss2(end)),'r--','LineWidth',2);
hold on
% semilogy(comm_grad6(:,1),comm_error6(:,2)-loss2(end),'c-','LineWidth',2);
% hold on
% semilogy(comm_grad7(:,1),comm_error7(:,2)-loss2(end),'c--','LineWidth',2);
% hold on
semilogy(comm_grad2(:,1),abs(comm_error2(:,2)-loss2(end)),'b-','LineWidth',2);
xlabel('Number of communications (uploads)','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')