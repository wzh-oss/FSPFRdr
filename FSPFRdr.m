%%% Feature Set Partition-based Approach to Fuzzy Rough dimensionality reduction (FSPFRdr) algorithm
%%% Please refer to the following papers:
%%% Zhihong Wang, Hongmei Chen, Xiaoling Yang, Jihong Wan, Tianrui Li, Chuan Luo. Fuzzy rough dimensionality reduction:
%%% A feature set partition-based approach[J].
%%%  Information Science.
function compound_feature_data=FSPFRdr(data,K,T1)
%%%input:
% data is data matrix, where rows for samples and columns for attributes.
% Numerical attributes should be normalized into [0,1].
% K is a given parameter for the number of the nearest neighbors adjustment.
%T1 is used to adjust the fuzzy radius.
%%%output
% Reduction data
[row,column]=size(data);
label=data(:,end);
class=unique(label);
r_D=rD(label);
fea_num=column-1;
C_num=1:fea_num;
C=data(:,C_num);
S_num=[];
gy=zeros(1,fea_num);
for i=1:length(C_num)
    gy1=dep([C(:,i) label],T1);
    gy(:,i)=gy(:,i)+ gy1;
end
sig=gy;
[~,k_s]=find(sig==max(sig));
k_k=min(k_s);
S_num=[S_num C_num(k_k)];
C_num=setdiff(C_num,C_num(k_k));
rs=compute_r([data(:,k_k) label],T1);
while ~isempty(C_num)
    c=zeros(1,length(C_num));
    for i=1:length(C_num)
        ri=compute_r([C(:,C_num(i)) label],T1);
        C_I=CI(ri,rs,r_D);
        c(1,i)=c(1,i)+ C_I;
    end
    c_c=(sum(c.^(1))/length(C_num))^(1/1);
    [~,c_l]=find(c>c_c);
    SF_num=C_num(c_l);
    [~,c_2]=find(c<c_c&c>-c_c);
    WS_num=C_num(c_2);
    C_num=[];
    while ~isempty(WS_num)
        if length(WS_num)>length(class)-1
            d_p=length(class)-1;
        else
            d_p=1;
        end
        e=FRslle([C(:,WS_num) label],K,d_p,T1,1)';
        if ~isreal(e)
            E=abs(e);
            E=normalize(E,'range');
        else
            E=normalize(e,'range');
        end
        [~,col1]=size(C);
        e_num=col1+1:+col1+d_p;
        colf=length(SF_num);
        sf_num=zeros(1,colf+d_p);
        s_c=zeros(row,col1+d_p);
        sf_num(1:colf)=sf_num(1:colf)+SF_num;
        sf_num(colf+1:end)=sf_num(colf+1:end)+e_num;
        SF_num=sf_num;
        s_c(:,1:col1)=s_c(:,1:col1)+C;
        s_c(:,col1+1:end)=s_c(:,col1+1:end)+E;
        C=s_c;
        WS_num=[];
    end
    while  ~isempty(SF_num)
        LS=length(S_num);
        sig=Sig(C(:,S_num),C(:,SF_num),label,T1);        
        k_K=find(sig==max(sig));
        s_num=zeros(1,LS+length(k_K));
        if max(sig)>0
            s_num(1:LS)=s_num(1:LS)+S_num;
            s_num(LS+1:end)=s_num(LS+1:end)+SF_num(k_K);
            S_num=s_num;
        end
        SF_num=setdiff(SF_num,SF_num(k_K));
        C_num=SF_num;
        SF_num=[];
    end
end
compound_feature_data=[C(:,S_num) label];
end