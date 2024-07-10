%频域变换和特征提取
clear;
clc;
fs=51200;
filename1=[400,800,1200,1600,2000,2400,2800,3200,3600,4000];
filename2='pro.mat';
for idx=10%1:length(filename1)
    filename=[num2str(filename1(idx)),filename2];
    load(filename);
    for ind=10%1:size(prodata,2)
        cutdata=prodata(:,ind);
        %对选定信号进行频域变换
        fs=51200;
        x=(0:length(cutdata)-1)/fs; 
        y=abs(fft(cutdata));
        xf=(0:length(cutdata)-1)'*fs/length(y);
        subplot(2,1,1)
        plot(x,cutdata);
        xlabel('时间/s');
        ylabel('幅度');
        title('时域图');
        subplot(2,1,2);
        plot(xf,y);
        xlabel('频率/Hz');
        ylabel('幅度');
        title('频域图');
        %计算MFCC
        cutdata=filter(0.0625,1,cutdata);%预加重
        [coeffs,delta,deltaDelta,loc]= mfcc(cutdata,fs,'WindowLength',128,...
            'OverlapLength',32,"LogEnergy","Ignore",'DeltaWindowLength',9);
        %处理矩阵使其维度相同
        if idx==5    %(35*13)->（37*13）
            a=coeffs(end-1:end,:)+randn(2,13);
            coeffs=[coeffs;a];
            a=delta(end-1:end,:)+randn(2,13);
            delta=[delta;a];
            a=deltaDelta(end-1:end,:)+randn(2,13);
            deltaDelta=[deltaDelta;a];
        end
        if idx==10%(39*13)->（37*13）
            coeffs(38:39,:)=[];
            delta(38:39,:)=[];
            deltaDelta(38:39,:)=[];
        end
        % 保存输出矩阵
        eval(['coeffs',num2str(ind),'=','coeffs',';']);
        eval(['delta',num2str(ind),'=','delta',';']);
        eval(['deltaDelta',num2str(ind),'=','deltaDelta',';']);
        matname=['coeffs',num2str(filename1(idx)),'.mat'];
        if ind==1
            save(matname,['coeffs',num2str(ind)]);
        else
            save(matname,['coeffs',num2str(ind)],'-append');
        end
        matname=['delta',num2str(filename1(idx)),'.mat'];
        if ind==1
            save(matname,['delta',num2str(ind)]);
        else
            save(matname,['delta',num2str(ind)],'-append');
        end
        matname=['deltaDelta',num2str(filename1(idx)),'.mat'];
        if ind==1
            save(matname,['deltaDelta',num2str(ind)]);
        else
            save(matname,['deltaDelta',num2str(ind)],'-append');
        end
    end      
end
