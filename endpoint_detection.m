% 双门限法语音端点检测
clear;
clc;
startpoint=[54000,49500,26500,20000,500,9000,30000,95000,16000,70000];
filename1=[400,800,1200,1600,2000,2400,2800,3200,3600,4000];
filename2='raw.mat';
for ind=1:length(filename1)
    filename=[num2str(filename1(ind)),filename2];
    load(filename);
    fs=51200;
    start=startpoint(ind);                  % 起始点
    x=data/max(abs(data));                  % 幅度归一化
    N=length(x);
    time=(0:N-1)/fs; 
    % 数据分帧
    wlen=500;                               % 帧长
    inc=200;                                % 帧移位(相邻两帧非重叠部分）
    fn=fix((N-wlen)/inc)+1;                 % 帧数
    NIS=fix((start-wlen)/inc)+1;            % 前导噪声帧数
    frameTime=(((1:fn)-1)*inc+wlen/2)/fs;   % 计算每帧对应的时间
    [voiceseg,vsl]=vad_ezm1(x,wlen,inc,NIS);% 端点检测
    % 输出矩阵初始化
    dur=0;
    for j=1 : vsl       % 计算最长信号的长度
        i=fix(fs*frameTime(voiceseg(j).end))-fix(fs*frameTime(voiceseg(j).begin));
        if dur <= i
            dur=i;
        end
    end
    prodata=zeros(dur,vsl);
     % 画出起止点位置
    plot(time,x);    
    title('端点检测');
    ylabel('幅值');axis([0 max(time) -1 1]);   grid;
    xlabel('时间/s');   
    for k=1 : vsl      
        nx1=voiceseg(k).begin; 
        nx2=voiceseg(k).end;
        a=fix((fs*frameTime(nx1)));
        b=fix((fs*frameTime(nx2)));
        nxl=b-a+1;
        prodata(1:nxl,k)=x(fix(fs*frameTime(nx1)):fix(fs*frameTime(nx2)));      
         line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','r','LineStyle','-');
         line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','r','LineStyle','--');
    end
    % 保存输出矩阵
    matname=[num2str(filename1(ind)),'pro.mat'];
    save(matname,'prodata');
end