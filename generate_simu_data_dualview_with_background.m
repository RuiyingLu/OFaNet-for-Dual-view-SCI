

clc;
clear;
close all;

video_path='F:\sci-dual-view\Data\DAVIS-2017\DAVIS';


meas_save_path='./train_data/meas/';
gt_save_path='./train_data/gt/';

resolution='Full-Resolution/';
block_size=256;
block_size_square=block_size*block_size;
compress_frame=10;


if exist(gt_save_path,'dir')==0
   mkdir(gt_save_path);
end
if exist(meas_save_path,'dir')==0
   mkdir(meas_save_path);
end
load(['256_simu_mask',num2str(compress_frame),'.mat'])
mask = double(simu_mask);

num_yb=0;

name_obj=dir([video_path,'/JPEGImages/',resolution]);

for image1=3:length(name_obj)
   for image2=image1+1:length(name_obj)
       
       fprintf('%d - %d\n',image1-2,image2-2)
       path1=[video_path,'/JPEGImages/',resolution,name_obj(image1).name];
       name_frame_1=dir(path1);
       pic1=imread([path1,'/',name_frame_1(3).name]);
       w_pic1=size(pic1);
       x_1=[1:block_size/2:w_pic1(1)-block_size w_pic1(1)-block_size];
       y_1=[1:block_size/2:w_pic1(2)-block_size w_pic1(2)-block_size];
       
       path2=[video_path,'/JPEGImages/',resolution,name_obj(image2).name];
       name_frame_2=dir(path2);
       pic2=imread([path2,'/',name_frame_2(3).name]); 
       w_pic2=size(pic2);
       x_2=[1:block_size/2:w_pic2(1)-block_size w_pic2(1)-block_size];
       y_2=[1:block_size/2:w_pic2(2)-block_size w_pic2(2)-block_size];
      
       num=0;
       sample_num=0;
       while num<8
               num=num+1;
               im_1 = unidrnd(length(name_frame_1)-compress_frame-2)+2;
               im_2 = unidrnd(length(name_frame_2)-compress_frame-2)+2;
               % compute the big motion in image1
               pic_block_1=zeros(size(pic1));
             for mm=1:compress_frame
                 pic=imread([path1,'/',name_frame_1(im_1+mm-1).name]);
                 pic=rgb2ycbcr(pic);
                 pic_block_1(:,:,mm)=pic(:,:,1);
             end
             pic_block_mean_1=mean(pic_block_1,3);
             pic_block_sigma_1=var(pic_block_1,0,3);
             m_1=zeros(length(x_1),length(y_1));
             for i=1:length(x_1)
                for j=1:length(y_1)
                    x1=pic_block_sigma_1(x_1(i):x_1(i)+block_size-1,y_1(j):y_1(j)+block_size-1,:);
                    a1=max(x1(:))-min(x1(:));
                    m_1(i,j)=a1;
                end
             end
             [a_1,index_1]=sort(m_1(:),'descend');
             
             % compute the big motion in image2
             pic_block_2=zeros(size(pic2));
             for mm=1:compress_frame
                 pic=imread([path2,'/',name_frame_2(im_2+mm-1).name]);
                 pic=rgb2ycbcr(pic);
                 pic_block_2(:,:,mm)=pic(:,:,1);
             end
             pic_block_mean_2=mean(pic_block_2,3);
             pic_block_sigma_2=var(pic_block_2,0,3);
             m_2=zeros(length(x_2),length(y_2));
             for i=1:length(x_2)
                for j=1:length(y_2)
                    x2=pic_block_sigma_2(x_2(i):x_2(i)+block_size-1,y_2(j):y_2(j)+block_size-1,:);
                    a2=max(x2(:))-min(x2(:));
                    m_2(i,j)=a2;
                end
             end
             [a_2,index_2]=sort(m_2(:),'descend');
             
            Num=1;
            for n=1:Num
              x1=index_1(1);
              i1=mod((x1-1),length(x_1))+1;
              j1=floor((x1-1)/length(x_1))+1;
              x2=index_2(1);
              i2=mod((x2-1),length(x_2))+1;
              j2=floor((x2-1)/length(x_2))+1;
              meas=zeros(block_size,block_size,'uint16');
              sum1=meas;
              sum2=meas;
              patch_save=zeros(block_size,block_size,compress_frame*2);
              for mm=1:compress_frame
                  p=pic_block_1(x_1(i1):x_1(i1)+block_size-1,y_1(j1):y_1(j1)+block_size-1,mm);
                  patch_save(:,:,mm)=p;
                  sum1=sum1+uint16(p.*mask(:,:,mm));
                  a=sum1;
              end
              if length(find(a>0))/block_size_square<0.09
                  continue
              end    
              for mm=1:compress_frame
                  p=pic_block_2(x_2(i2):x_2(i2)+block_size-1,y_2(j2):y_2(j2)+block_size-1,mm);
                  patch_save(:,:,mm+compress_frame)=p;
                  sum2=sum2+uint16(p.*mask(:,:,mm+compress_frame));
                  b=sum2;
              end
              if length(find(b>0))/block_size_square<0.09
                  continue
              end     
             sample_num=sample_num+1;
             num_yb=num_yb+1;
             meas=uint16(a+b);
             patch_save = uint8(patch_save);
             save([gt_save_path,num2str(num_yb),'.mat'],'patch_save');      
             save([meas_save_path,num2str(num_yb),'.mat'],'meas');
             if sample_num==2
                 break
             end    
             
            end
       end
       
   end
  
end

