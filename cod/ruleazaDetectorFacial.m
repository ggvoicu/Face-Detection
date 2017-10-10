function [detectii, scoruriDetectii, imageIdx] = ruleazaDetectorFacial(parametri)
% 'detectii' = matrice Nx4, unde 
%           N este numarul de detectii  
%           detectii(i,:) = [x_min, y_min, x_max, y_max]
% 'scoruriDetectii' = matrice Nx1. scoruriDetectii(i) este scorul detectiei i
% 'imageIdx' = tablou de celule Nx1. imageIdx{i} este imaginea in care apare detectia i
%               (nu punem intregul path, ci doar numele imaginii: 'albert.jpg')

% Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din parametri.numeDirectorExempleTest
% Directorul cu numele parametri.numeDirectorExempleTest contine imagini ce
% pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
% date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
% Functia 'suprimeazaNonMaximele' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
% Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.

% Functia voastra ar trebui sa calculeze pentru fiecare imagine
% descriptorul HOG asociat. Apoi glisati o fereastra de dimeniune paremtri.dimensiuneFereastra x  paremtri.dimensiuneFereastra (implicit 36x36)
% si folositi clasificatorul liniar (w,b) invatat poentru a obtine un scor. Daca acest scor este deasupra unui prag (threshold) pastrati detectia
% iar apoi mporcesati toate detectiile prin suprimarea non maximelor.
% pentru detectarea fetelor de diverse marimi folosit un detector multiscale

imgFiles = dir( fullfile( parametri.numeDirectorExempleTest, '*.jpg' ));
%initializare variabile de returnat
detectii = zeros(0,4);
scoruriDetectii = zeros(0,1);
imageIdx = cell(0,1);



%for i = 1:length(imgFiles)   
for i = 1:length(imgFiles)
    fprintf('Rulam detectorul facial pe imaginea %s\n', imgFiles(i).name)
    img = imread(fullfile( parametri.numeDirectorExempleTest, imgFiles(i).name ));    
    
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end    
    %completati codul functiei in continuare
    detectiiImg = zeros(1000000, 4);
    scoruriImg = zeros(1, 1000000);
    imageIdxImg = cell(1000000, 1);
    
    res = 0.1;
    contor = 0;
    
    while res <= 1.1

        imgModif = imresize(img,res);
        descriptorHOGImg = vl_hog(single(imgModif), parametri.dimensiuneCelulaHOG);

        nrCeluleVerticaleFereastra = parametri.dimensiuneFereastra / parametri.dimensiuneCelulaHOG;
        nrCeluleOrizontaleFereastra = parametri.dimensiuneFereastra / parametri.dimensiuneCelulaHOG;

        nrCeluleVerticaleImagine = size(descriptorHOGImg, 1); %6
        nrCeluleOrizontaleImagine = size(descriptorHOGImg, 2); %6
        %dimDescriptorFereastra = size(descriptorHOGImg, 3); %31
        %size(img)
        %disp(nrCeluleOrizontaleImagine);

            for celulaX = 1 : nrCeluleOrizontaleImagine - nrCeluleOrizontaleFereastra + 1
            %disp(celulaX);
            %disp(contor);
                for celulaY = 1 : nrCeluleVerticaleImagine - nrCeluleVerticaleFereastra + 1
                %disp(celulaY);
                   contor = contor + 1;

                   xmin = (celulaX - 1) * parametri.dimensiuneCelulaHOG + 1;
                   xmax = xmin + parametri.dimensiuneFereastra - 1;

                   ymin = (celulaY - 1) * parametri.dimensiuneCelulaHOG + 1;
                   ymax = ymin + parametri.dimensiuneFereastra - 1;

                   detectiiImg(contor, :) = [int64(xmin/res), int64(ymin/res), int64(xmax/res), int64(ymax/res)];
                   descriptorFereastra = descriptorHOGImg(celulaY : celulaY + nrCeluleVerticaleFereastra - 1, ...
                                                          celulaX : celulaX + nrCeluleOrizontaleFereastra - 1, : );
                   [a, b, c] = size(descriptorFereastra);

                   descriptorFereastraRedimensionat = reshape(descriptorFereastra, 1, a * b * c);

                   scoruriImg(contor) = (parametri.w)' * descriptorFereastraRedimensionat' + parametri.b;
                   imageIdxImg{contor} = imgFiles(i).name;
                   %size(detectiiImg)
                end
            end
         res=res+0.1;
    end
    %keyboard;
    %size(img)
    %disp(nrCeluleOrizontaleImagine);
    %size(img)
    %disp(nrCeluleOrizontaleImagine);
    
    detectiiImg(contor + 1 : end, :) = [];
    scoruriImg(contor + 1 : end) = [];
    imageIdxImg(contor + 1 : end) = [];
    
    index = find(scoruriImg < parametri.threshold);
    detectiiImg(index, :) = [];
    scoruriImg(index) = [];
    imageIdxImg(index) = [];
    
    esteMaximLocal = eliminaNonMaximele(detectiiImg, scoruriImg, size(img));
    
    ndx = find(esteMaximLocal == 1);
    
    detectii(end + 1 : end + length(ndx), :) = detectiiImg(ndx, :);
    scoruriDetectii(end + 1 : end + length(ndx)) = scoruriImg(ndx);
    imageIdx(end + 1 : end + length(ndx)) = imageIdxImg(ndx);
    
end
