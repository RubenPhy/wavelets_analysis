function [variabsig] = detcrash(signal,level,window)

le=length(signal);

variabsig=zeros(le,1);

%% Primero simetrizamos la se�al alrededor de cada instante, con longitud 
%% total 2*window


for i=window:le
    s1=signal(i-window+1:i);
    s2=flipud(s1);
    s=cat(1,s1,s2);

%% Hacemos la descomposici�n wavelet haar hasta nivel=level

        [C,L] = wavedec(s,level,'haar');

%% C�lculo detalles a todos los niveles

    for j=1:level 
        d(:,j)=wrcoef('d',C,L,'haar',j);
    end

%% C�lculo de aproximaci�n a �ltimo nivel 

    ap=wrcoef('a',C,L,'haar',level);

%% Calculamos la variabilidad de los detalles a �ltimo 
%% nivel correspondientes a la ventana 'window'

   
    difdet=abs(d(1:end-1,level)-d(2:end,level));
  
    varydet=sum(difdet);
    

%% Ahora normalizamos la variabilidad de los detalles

    varydetnorm=varydet/sum(ap); 

%% Finalmente posicionamos esa variabilidad normalizada en el instante i

    variabsig(i)=varydetnorm; 

end
