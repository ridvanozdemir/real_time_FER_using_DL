%This program detect the faces on the image from crop_input folder 
%using Viola-Jones face detection algorithm then crop them and
% write with new size and name to crop_out folder.

%path of your crop_input folder
imagefiles = dir('D:\ridvan_16\01_lisansustu\doktora\2018_BSEU\emotion_recognition\f_e_r_adobe_and_home_made\crop_input\*.jpg');     
nfiles = length(imagefiles);    

%calling Viola-Jones face detection algorithm
faceDetector = vision.CascadeObjectDetector;

%creating loop for detecting all faces in the crop_input folder
for ii=1:nfiles
   currentfilename =fullfile('D:\ridvan_16\01_lisansustu\doktora\2018_BSEU\emotion_recognition\f_e_r_adobe_and_home_made\crop_input\', imagefiles(ii).name);
   currentimage = imread(currentfilename);
   images{ii} = currentimage;
   
   picture = images{ii};
   bboxes = step(faceDetector, picture);
   [m,n] = size(bboxes);
   %writing resized and cropped face images to crop_out folder
   for i=1:1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[227,227]);
        folder = 'D:\ridvan_16\01_lisansustu\doktora\2018_BSEU\emotion_recognition\f_e_r_adobe_and_home_made\crop_out\';
        newimagename = [folder imagefiles(ii).name '_v4v4v' '.jpg'];
        imwrite(I2,newimagename);
   end 
   
end
