clear all %usuwa wszystkie obiekty
nntwarn off %wy��cza ostrze�enia 
format long %ustawia format wypisywania wynik�w

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

%%%%%%%%-------WARTO�CI DO EDYCJI-------%%%%%%%%
[R,Q] = size(P);            %liczba neuron�w w warstwie wej�ciowej
[S3,Q] = size(T);           %liczba neuron�w w warstwie wyj�ciowej
S1 = 10:2:12;               %liczba neuron�w w warstwie pierwszej ustawiana r�cznie
S2 = S1;                    %liczba neuron�w w warstwie drugiej ustawiana r�cznie
lr = [0.0001 0.00001];      %poziomy stopnia nauki 
disp_freq= 1000;            %cz�stotliwo�� wy�wietlania wyniku
max_epoch=20000;            %maksymalna liczba epok ustawiana r�cznie
err_goal=.25;               %b��d docelowy
error = [];                 %tablica przechowuj�ca b��dy

%%%%%%%%-------TWORZENIE MACIERZY DO ZAPISYWANIA WYNIK�W-------%%%%%%%%
wyniki_pp=zeros(length(S1), length(S2),length(lr));     %procentu poprawno�ci
wyniki_epoch=zeros(length(S1), length(S2),length(lr));  %liczby epok
wyniki_SSE=zeros(length(S1), length(S2),length(lr));    %ko�cowego b��du uczenia
hist_pp = zeros(1,(max_epoch/disp_freq));   
hist_sse = zeros(1,(max_epoch/disp_freq));

%%%%%%%%-------P�TLE PROGRAMU UCZ�CEGO DLA WSZYSTKICH PRZYPADK�W-------%%%%%%%%
for ind_S1=1:length(S1),
    for ind_S2=1:ind_S1,
        for ind_lr=1:length(lr)
            

%generowanie macierzy wag i wektora biasu dla poszczeg�lnych warstw
[W1,B1] = nwtan(S1(ind_S1),R);
[W2,B2] = nwtan(S2(ind_S2),S1(ind_S1));
[W3,B3] = rands(S3,S2(ind_S2));

%%%%%%%%-------G��WNA P�TLA UCZ�CA-------%%%%%%%%
for epoch=1:max_epoch,      
   
    A1 = tansig(W1*P,B1);   %wyj�cie 1 warstwy
    A2 = tansig(W2*A1,B2);  %wyj�cie 2 warstwy
    A3 = purelin(W3*A2,B3); %wyj�cie 3 warstwy
    E = T -A3;              %b��d mi�dzy aktualnym stanem sieci a wyj�ciem 3 warstwy

    D3 = deltalin(A3,E);        %delta dla warstwy 3
    D2 = deltatan(A2,D3,W3);    %delta dla warstwy 2
    D1 = deltatan(A1,D2,W2);    %delta dla warstwy 1
    
    

    [dW1,dB1] = learnbp(P,D1,lr(ind_lr));   %wyliczanie nowych r�nic wag i bias dla neuron�w warstwy pierwszej
    W1 = W1 + dW1;
    B1 = B1 + dB1;
    [dW2,dB2] = learnbp(A1,D2,lr(ind_lr));  %wyliczanie nowych r�nic wag i bias dla neuron�w warstwy drugiej
    W2 = W2 + dW2;
    B2 = B2 + dB2;
    [dW3,dB3] = learnbp(A2,D3,lr(ind_lr));  %wyliczanie nowych r�nic wag i bias dla neuron�w warstwy trzeciej
    W3 = W3 + dW3;
    B3 = B3 + dB3;
 
     SSE = sumsqr(E);               %okre�lenie sumy kwadrat�w b�ed�w "E"
    error = [error SSE];            %do��czneie SSE do tablicy "error"
   
   
    if SSE < err_goal,              %je�eli SSE jest mniejsze od b��du docelowego
        epoch = epoch - 1;          %dekrementacja epoki
        break,                      %przerwanie p�tli
    end,
    
   
%%%%%%%%-------WYPISANIE/ZAPISYWANIE WYNIK�W OKRESOWYCH-------%%%%%%%%
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
%%%%%%%%---////G��WNA P�TLA UCZ�CA-------%%%%%%%%
%%%%%%%%-------ZAPIS WYNIKU-------%%%%%%%%
fileID = fopen('wyniki_14_maj.txt','a');
fprintf(fileID,'%2.0f %2.0f %1.5f %5.0f %3.3f %5.2f\r\n',S1(ind_S1),S2(ind_S2),lr(ind_lr),epoch,(100*(1-sum((abs(T-A3)>=.5)')/length(T))),SSE);
fclose(fileID);

wyniki_pp(ind_S1, ind_S2,ind_lr)=100*(1-sum((abs(T-A3)>=.5)')/length(T));
wyniki_epoch(ind_S1, ind_S2,ind_lr)=epoch;  
wyniki_SSE(ind_S1, ind_S2,ind_lr)=SSE;    


%%%%%%%%-------WYPISANIE WYNIK�W-------%%%%%%%%
[T' A3' (T-A3)' (abs(T-A3)>.5)']
[S1(ind_S1) S2(ind_S2)]     %liczba neuron�w w warstwarch
[lr(ind_lr) ]               %learning rate
100*(1-sum((abs(T-A3)>=.5)')/length(T)) %wynik uczenia w procentach
SSE     %b��d ko�cowy SSE

%%%%%%%%-------ZAKO�CZENIE P�TLI G��WNYCH I ZAPIS STANU MACIERZY-------%%%%%%%%
end
save 'projekt_14_maj.mat'
end
save 'projekt_14_maj.mat'
end
%%%%%%%%---////P�TLE PROGRAMU UCZ�CEGO DLA WSZYSTKICH PRZYPADK�W-------%%%%%%%%