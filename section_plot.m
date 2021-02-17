blurred = imread('testdata/input/noisy_512/17-1_testimg.png');
gt = imread('testdata/ground_truth/noisy_512/17-1.png');
deblurred = imread('testdata/output/17-1_testimg_out.png');

%specify segment start and end-point
x = [80 300];
y = [432 200];

a = improfile(blurred, x, y);
b = improfile(gt, x, y);
c = improfile(deblurred, x, y);
seg = linspace(1,length(a),length(a));

hold off

hold on

%plot intensity segment graph
subplot(2,3,[1,2,3]), plot(seg, a/255,'b', seg, c/255,'g', seg, b/255,'-')
xlabel('Segment pixel')
ylabel('Intensity')
legend('Blurred', 'Deblurred', 'Ground truth')
axis([0,length(a),0,0.6])

%plot images with segment shown
subplot(2,3,4), imshow(gt)
line(x,y)
title('Ground truth')
subplot(2,3,5), imshow(blurred)
line(x,y)
title('Blurred')
subplot(2,3,6), imshow(deblurred)
line(x,y)
title('Deblurred')
sgtitle('Intensity plot(top) along cross-sections(pictured, bottom)','fontweight','bold')