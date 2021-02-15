A = imread('test.png');

%sigma = 5
%B = fspecial('gaussian',5*sigma,sigma);


% PSF width factor
PSF_width = 3;
% size of convolving kernel
size = 60;
Q = zeros(size/2);
%Generate bessel matrix
for x = 1:round(size/2)
    for y = 1:round(size/2)
        r = (1/PSF_width)*(x^2+y^2)^0.5;
        Q(x,y) = besselj(1,r)/r;

    end
end

B = [flip(flip(Q,2)),flip(Q);flip(Q,2),Q];
PSF = B.^2;
C = convn(A,PSF);

lucy = deconvlucy(C,PSF,8);

%match brightness: note data is now float(not uint8) so convert 0-1.

m_C = max(C, [], 'all');
m_lucy = max(lucy, [], 'all');

%show samples

subplot(2,2,1), imshow(C/m_C) 
title('PSF applied')
subplot(2,2,2), imshow(lucy/m_lucy)
title('Deconvolved')
subplot(2,2,3), imshow(A)
title('Ground truth')
subplot(2,2,4), surf(B.^2)
title('PSF')


%careful about current working directory, so path is valid
imwrite(C,'testoutput.png')