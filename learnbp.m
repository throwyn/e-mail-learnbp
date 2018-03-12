clear all %usuwa wszystkie obiekty
nntwarn off %wy³¹cza ostrze¿enia 
format long %ustawia format wypisywania wyników

file = load('spambase_new.txt') ;

%%%%%%%%-------NORMALIZACJA-------%%%%%%%%
P=file(:,1:57);
T=file(:,58)';
maxP = max(P');
minP = min(P');

n=[min(P); max(P)]';

Pn=zeros(size(P));

for i=1:length(maxP),
        Pn(i,:) = (1-(-1))*(P(i,:)-minP(i))/(maxP(i)-minP(i))+(-1);
end

[min(Pn'); max(Pn')]';
P=Pn';

%%%%%%%%-------WARTOŒCI DO EDYCJI-------%%%%%%%%
[R,Q] = size(P);            %liczba neuronów w warstwie wejœciowej
[S3,Q] = size(T);           %liczba neuronów w warstwie wyjœciowej
S1 = 10:2:12;               %liczba neuronów w warstwie pierwszej ustawiana rêcznie
S2 = S1;                    %liczba neuronów w warstwie drugiej ustawiana rêcznie
lr = [0.0001 0.00001];      %poziomy stopnia nauki 
disp_freq= 1000;            %czêstotliwoœæ wyœwietlania wyniku
max_epoch=20000;            %maksymalna liczba epok ustawiana rêcznie
err_goal=.25;               %b³¹d docelowy
error = [];                 %tablica przechowuj¹ca b³êdy

%%%%%%%%-------TWORZENIE MACIERZY DO ZAPISYWANIA WYNIKÓW-------%%%%%%%%
wyniki_pp=zeros(length(S1), length(S2),length(lr));     %procentu poprawnoœci
wyniki_epoch=zeros(length(S1), length(S2),length(lr));  %liczby epok
wyniki_SSE=zeros(length(S1), length(S2),length(lr));    %koñcowego b³êdu uczenia
hist_pp = zeros(1,(max_epoch/disp_freq));   
hist_sse = zeros(1,(max_epoch/disp_freq));

%%%%%%%%-------PÊTLE PROGRAMU UCZ¥CEGO DLA WSZYSTKICH PRZYPADKÓW-------%%%%%%%%
for ind_S1=1:length(S1),
    for ind_S2=1:ind_S1,
        for ind_lr=1:length(lr)
            

%generowanie macierzy wag i wektora biasu dla poszczególnych warstw
[W1,B1] = nwtan(S1(ind_S1),R);
[W2,B2] = nwtan(S2(ind_S2),S1(ind_S1));
[W3,B3] = rands(S3,S2(ind_S2));

%%%%%%%%-------G£ÓWNA PÊTLA UCZ¥CA-------%%%%%%%%
for epoch=1:max_epoch,      
   
    A1 = tansig(W1*P,B1);   %wyjœcie 1 warstwy
    A2 = tansig(W2*A1,B2);  %wyjœcie 2 warstwy
    A3 = purelin(W3*A2,B3); %wyjœcie 3 warstwy
    E = T -A3;              %b³¹d miêdzy aktualnym stanem sieci a wyjœciem 3 warstwy

    D3 = deltalin(A3,E);        %delta dla warstwy 3
    D2 = deltatan(A2,D3,W3);    %delta dla warstwy 2
    D1 = deltatan(A1,D2,W2);    %delta dla warstwy 1
    
    

    [dW1,dB1] = learnbp(P,D1,lr(ind_lr));   %wyliczanie nowych ró¿nic wag i bias dla neuronów warstwy pierwszej
    W1 = W1 + dW1;
    B1 = B1 + dB1;
    [dW2,dB2] = learnbp(A1,D2,lr(ind_lr));  %wyliczanie nowych ró¿nic wag i bias dla neuronów warstwy drugiej
    W2 = W2 + dW2;
    B2 = B2 + dB2;
    [dW3,dB3] = learnbp(A2,D3,lr(ind_lr));  %wyliczanie nowych ró¿nic wag i bias dla neuronów warstwy trzeciej
    W3 = W3 + dW3;
    B3 = B3 + dB3;
 
     SSE = sumsqr(E);               %okreœlenie sumy kwadratów b³edów "E"
    error = [error SSE];            %do³¹czneie SSE do tablicy "error"
   
   
    if SSE < err_goal,              %je¿eli SSE jest mniejsze od b³êdu docelowego
        epoch = epoch - 1;          %dekrementacja epoki
        break,                      %przerwanie pêtli
    end,
    
   
%%%%%%%%-------WYPISANIE/ZAPISYWANIE WYNIKÓW OKRESOWYCH-------%%%%%%%%
    if(rem(epoch,disp_freq)==0)     %warunek wypisania aktualnego wyniku
        hist_pp(epoch/1000) = 100*(1-sum((abs(T-A3)>=.5)')/length(T));  
        hist_sse(epoch/1000) = SSE;
        
        epoch
        SSE
        100*(1-sum((abs(T-A3)>=.5)')/length(T))
          plot([1:length(T)],sort(T),'r',[1:length(T)],sort(A3),'g')
         %rysowanie wykresu 
        %plot(error)
      
        pause(1e-100)
    end 
end
%%%%%%%%---////G£ÓWNA PÊTLA UCZ¥CA-------%%%%%%%%
%%%%%%%%-------ZAPIS WYNIKU-------%%%%%%%%
fileID = fopen('wyniki_14_maj.txt','a');
fprintf(fileID,'%2.0f %2.0f %1.5f %5.0f %3.3f %5.2f\r\n',S1(ind_S1),S2(ind_S2),lr(ind_lr),epoch,(100*(1-sum((abs(T-A3)>=.5)')/length(T))),SSE);
fclose(fileID);

wyniki_pp(ind_S1, ind_S2,ind_lr)=100*(1-sum((abs(T-A3)>=.5)')/length(T));
wyniki_epoch(ind_S1, ind_S2,ind_lr)=epoch;  
wyniki_SSE(ind_S1, ind_S2,ind_lr)=SSE;    


%%%%%%%%-------WYPISANIE WYNIKÓW-------%%%%%%%%
[T' A3' (T-A3)' (abs(T-A3)>.5)']
[S1(ind_S1) S2(ind_S2)]     %liczba neuronów w warstwarch
[lr(ind_lr) ]               %learning rate
100*(1-sum((abs(T-A3)>=.5)')/length(T)) %wynik uczenia w procentach
SSE     %b³¹d koñcowy SSE

%%%%%%%%-------ZAKOÑCZENIE PÊTLI G£ÓWNYCH I ZAPIS STANU MACIERZY-------%%%%%%%%
end
save 'projekt_14_maj.mat'
end
save 'projekt_14_maj.mat'
end
%%%%%%%%---////PÊTLE PROGRAMU UCZ¥CEGO DLA WSZYSTKICH PRZYPADKÓW-------%%%%%%%%