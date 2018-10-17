base_path = 'E:/Project/OTB100/sequences/';% 'sequences/';%'C:/Users/iva/Desktop/其他/Temple-color-128/Temple-color-128/'%C:\Users\iva\Desktop\其他\vot2016

OTB2013Set = {'Basketball', 'Doll', 'Bolt', 'Soccer', 'Deer', 'Boy', 'CarDark', 'CarScale',...
      'David', 'David3', 'Football1', 'Girl', 'Crossing',...
      'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1',...
      'Trellis', 'Walking', 'Walking2', 'Woman', 'Tiger1', 'MotorRolling',...
      'Lemming', 'Matrix', 'Coke', 'FaceOcc1', 'Liquor', 'Skiing', 'Tiger2', 'Ironman', 'Couple',...
      'Car4', 'Subway', 'David2', 'Dog1', 'Dudek', 'FaceOcc2', 'Fish', 'FleetFace',...
      'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Jogging-1', 'Jogging-2',...
      'Jumping', 'Mhyang', 'Suv', 'Sylvester'};
OTB100Set = {'Basketball', 'Doll', 'Bolt', 'Soccer', 'Deer', 'Boy', 'CarDark', 'CarScale',...
      'David', 'David3', 'Football1', 'Girl', 'Crossing',...
      'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1',...
      'Trellis', 'Walking', 'Walking2', 'Woman', 'Tiger1', 'MotorRolling',...
      'Lemming', 'Matrix', 'Coke', 'FaceOcc1', 'Liquor', 'Skiing', 'Tiger2', 'Ironman', 'Couple',...
      'Car4', 'Subway', 'David2', 'Dog1', 'Dudek', 'FaceOcc2', 'Fish', 'FleetFace',...
      'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Jogging-1', 'Jogging-2',...
      'Jumping', 'Mhyang', 'Suv', 'Sylvester','Biker','Bird1','Bird2','BlurBody','BlurCar1','BlurCar2','BlurCar3','BlurCar4',...
      'BlurFace','BlurOwl','Board','Bolt2','Box','Car1','Car2','Car24','ClifBar','Coupon','Crowds','Dancer','Dancer2',...
      'Diving','Dog','DragonBaby','Girl2','Gym','Human2','Human3','Human4-2','Human5','Human6','Human7','Human8','Human9',...
      'Jump','KiteSurf', 'Man','Panda','RedTeam','Rubik','Skater','Skater2','Skating2-1','Skating2-2','Surfer','Toy','Trans','Twinnings','Vase'};
testSet = OTB100Set; 
scale = 0.6;
num_pp = zeros(10,1); s = 1;
pp = zeros(10,1);
% for scale = 0.1:0.1:1;
   
    p = 0;
    num_p = 0;
    for i = 1:length(testSet)
        video_path = [base_path testSet{i} '/'];
        text_files = dir([video_path '*_gt.txt']);
        f = [video_path text_files(1).name];
        ground_truth = load(f);
        len = size(ground_truth,1);
        count = 0;
        for j = 1:len-1
            target_sz = [ground_truth(j,4), ground_truth(j,3)];
            norm_sz = sqrt(ground_truth(j,4)*ground_truth(j,3));
            pos1 = [ground_truth(j,2), ground_truth(j,1)];
            pos2 = [ground_truth(j+1,2), ground_truth(j+1,1)];
            juli = norm(pos1-pos2);
            if juli > scale*norm_sz
                count = count + 1;
            end
            
        end
        if count ~= 0
            num_p = num_p + 1;
%             count
            disp([testSet{i}]);
        end
        d = count/len;
        p = p + d;
    end
    num_p
% pp(s) = p;
%     num_pp(s) = num_p;
%     s = s + 1;
% end
% x = 0.1:0.1:1;
% num_pp;
%  bar(x,pp);%axis([0 2 0 100]);
% title('Motion Statistics');
% xlabel('ratio');
% ylabel('Frames \%');
% saveas(gcf,['motion2'],'png')