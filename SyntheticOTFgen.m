A = imread('test.png');

%sigma = 5
%B = fspecial('gaussian',5*sigma,sigma);


% PSF width factor
PSF_width = 1.5;
% size of convolving kernel
size = 128;
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
surf(PSF)
OTF = abs(fftshift(fft2(PSF)));
scaled_OTF = OTF/max(OTF, [], 'all');
imwrite(scaled_OTF,'OTF.png')