%This program is for runing FER algorithm on camera
%Ridvan Ozdemir

%Start webcam
camera=webcam;

%Call face detection algorithm
faceDetector = vision.CascadeObjectDetector;
pause(1);

%Real-time test for 10 seconds 
tic;
while toc < 10

%while true
    
   picture = camera.snapshot;
   bboxes = step(faceDetector, picture);
   [m,n] = size(bboxes);
   
   for i=1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[227,227]);
        [label,scr] = classify( myNet, I2);
        %emotions(i)=cellstr(char(label));
        per=sprintf('%0.2f',max(scr));
        emotions(i)=cellstr([ num2str(per), '  ',char(label)]);
   end
   
   if m ~= 0
       IFaces = insertObjectAnnotation(picture, 'rectangle', bboxes(1:m,:), emotions(1:m));  
   end
   
       image(picture);
       imshow(IFaces);
       %title(char(label));
       drawnow;  
end

close all;
clear camera;