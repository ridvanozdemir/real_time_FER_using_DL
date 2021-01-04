%This program is for runing FER algorithm on single image
%Ridvan Ozdemir

%Call face detection algorithm
faceDetector = vision.CascadeObjectDetector;
pause(1);

%choice your test image
im = imread('Friends2.jpg');
picture = im;  

bboxes = step(faceDetector, picture);
[m,n] = size(bboxes);

%FER for every face on image
   for i=1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[227,227]);
        [label,scr] = classify( myNet, I2);
        per=sprintf('%0.2f',max(scr));
        emotions(i)=cellstr([ num2str(per), '  ',char(label)]);
   end
   
   if m ~= 0
       IFaces = insertObjectAnnotation(picture, 'rectangle', bboxes(1:m,:), emotions(1:m));  
   end
   
   image(picture);
   imshow(IFaces);
   drawnow;