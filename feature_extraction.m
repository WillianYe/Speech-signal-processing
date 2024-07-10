%Ƶ��任��������ȡ
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
        %��ѡ���źŽ���Ƶ��任
        fs=51200;
        x=(0:length(cutdata)-1)/fs; 
        y=abs(fft(cutdata));
        xf=(0:length(cutdata)-1)'*fs/length(y);
        subplot(2,1,1)
        plot(x,cutdata);
        xlabel('ʱ��/s');
        ylabel('����');
        title('ʱ��ͼ');
        subplot(2,1,2);
        plot(xf,y);
        xlabel('Ƶ��/Hz');
        ylabel('����');
        title('Ƶ��ͼ');
        %����MFCC
        cutdata=filter(0.0625,1,cutdata);%Ԥ����
        [coeffs,delta,deltaDelta,loc]= mfcc(cutdata,fs,'WindowLength',128,...
            'OverlapLength',32,"LogEnergy","Ignore",'DeltaWindowLength',9);
        %�������ʹ��ά����ͬ
        if idx==5    %(35*13)->��37*13��
            a=coeffs(end-1:end,:)+randn(2,13);
            coeffs=[coeffs;a];
            a=delta(end-1:end,:)+randn(2,13);
            delta=[delta;a];
            a=deltaDelta(end-1:end,:)+randn(2,13);
            deltaDelta=[deltaDelta;a];
        end
        if idx==10%(39*13)->��37*13��
            coeffs(38:39,:)=[];
            delta(38:39,:)=[];
            deltaDelta(38:39,:)=[];
        end
        % �����������
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
