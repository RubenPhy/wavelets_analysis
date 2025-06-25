function [variabsig,outs] = detcrash(signal,level,window)

le=length(signal);

variabsig=zeros(le,1);

%% Primero simetrizamos la señal alrededor de cada instante, con longitud 
%% total 2*window


for i=window:le
    s1=signal(i-window+1:i);
    s2=flipud(s1);
    s=cat(1,s1,s2);

%% Hacemos la descomposición wavelet haar hasta nivel=level

        [C,L] = wavedec(s,level,'haar');

%% Cálculo detalles a todos los niveles

    for j=1:level 
        d(:,j)=wrcoef('d',C,L,'haar',j);
    end

%% Cálculo de aproximación a último nivel 

    ap=wrcoef('a',C,L,'haar',level);

%% Calculamos la variabilidad de los detalles a último 
%% nivel correspondientes a la ventana 'window'

   
    difdet=abs(d(1:end-1,level)-d(2:end,level));
  
    varydet=sum(difdet);
    

%% Ahora normalizamos la variabilidad de los detalles

    varydetnorm=varydet/sum(ap); 

%% Finalmente posicionamos esa variabilidad normalizada en el instante i

    variabsig(i)=varydetnorm; 

end

%% Ahora generamos el vector de outliers, pero necesitamos tener primero
%% un 10% de la señal para poder computarlos

outs=zeros(le,1);

for i=300:le

    ou=isoutlier(variabsig(i-299:i));

    outs(i)=ou(end);

end
