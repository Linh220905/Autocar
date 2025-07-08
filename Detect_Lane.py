import cv2
import numpy as np
import os
total = 0
count =0
image_folder = "E:/AutoCar/NewMap"
gray_red = "E:/AutoCar/Gray_red"
gray_black =  "E:/AutoCar/Gray_black"
result = "E:/AutoCar/Result"




def detect_lane(image_path, result_path,black_gray,red_gray):
    global total,count
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    width_left = width * 2//5
    width_right = width * 3//4
    top = height//2
    cropped_img = img[top:, :]
    red_img = img[top:, :]

    hsv_red = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_black = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    center = width // 2

    x_black = None
    x_red = None
    lower_black1 = np.array([0, 0, 0])
    upper_black1 = np.array([180, 255, 50])  

    lower_black2 = np.array([10//2,15*2.55,2*2.55])
    upper_black2 = np.array([50//2,45*2.55,25*2.55])

    lower_black3 = np.array([20//2,10*2.55,20*2.55])
    upper_black3 = np.array([40//2,30*2.55,30*2.55])

    lower_black4 = np.array([20//2,45*2.55,10*2.55])
    upper_black4 = np.array([40//2,60*2.55,30*2.55])

    mask_black1 = cv2.inRange(hsv_black, lower_black1,upper_black1)
    mask_black2 = cv2.inRange(hsv_black, lower_black2,upper_black2)
    mask_black3 = cv2.inRange(hsv_black, lower_black3,upper_black3)
    mask_black4 = cv2.inRange(hsv_black, lower_black4,upper_black4)
    mask_black = mask_black1



    lower_red1 = np.array([0, 45 * 2.55, 18*2.55])
    upper_red1 = np.array([10//2, 105*2.55,50 *2.55])
    lower_red2 = np.array([340//2, 45*2.55, 18*2.55])
    upper_red2 = np.array([360//2, 100*2.55, 55*2.55])
    lower_red3 = np.array([347//2, 68*2.55 , 50*2.55])
    upper_red3 = np.array([360*2.55,100*2.55, 75*2.55])
    lower_red4 = np.array([0//2, 68*2.55 , 10*2.55])
    upper_red4 = np.array([10*2.55,100*2.55, 30*2.55])

    mask_red1 = cv2.inRange(hsv_red, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_red, lower_red2, upper_red2)
    mask_red3 = cv2.inRange(hsv_red, lower_red3, upper_red3)
    mask_red4 = cv2.inRange(hsv_red, lower_red4, upper_red4)
    mask_red = mask_red1  | mask_red2 
   
    mask_red = cv2.erode(mask_red, None, iterations=1)
    mask_red = cv2.dilate(mask_red, None, iterations=1)



    mask_black = cv2.erode(mask_black, None, iterations=1)
    mask_black = cv2.dilate(mask_black, None, iterations=1)
    #mask_red = extract_color_mask_kmeans(hsv_red, target_hue_range=(170, 180))  # tương ứng 340° – 360° trong HSV


    
    red_lines = cv2.HoughLinesP(mask_red, 1, np.pi/180, threshold=60, minLineLength=100, maxLineGap=100)
    if red_lines is not None:
        top_red = sorted(
            [(x1, y1, x2, y2, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
             for [[x1, y1, x2, y2]] in red_lines],
            key=lambda x: x[4],
            reverse=True
        )
        x1, y1, x2, y2, _ = top_red[0]
        
       
        x_red =  x2
        
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  
        

   
    black_lines = cv2.HoughLinesP(mask_black, 1, np.pi/180, threshold=60, minLineLength=100, maxLineGap=100)
    if black_lines is not None:
        top_black = sorted(
            [(x1, y1, x2, y2, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
             for [[x1, y1, x2, y2]] in black_lines],
            key=lambda x: x[4],
            reverse=True
        )
        x1, y1, x2, y2, _ = top_black[0]
        y1 += top
        y2+= top
        
     
        x_black = (x1 + x2) / 2
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    error = None
    x_center = -1
    width_lane = 430
    if x_red is not None and x_black is not None:
        if x_black < center:
           x_center = (x_red - x_black) // 2
        else:
            x_center = (x_black - x_red ) // 2


    if x_red is not None:
        if x_red < center:
          
            x_center = x_red + width_lane // 2
        else: 
            x_center = x_red - width_lane //  2
    cv2.line(img, (x_center,0), (x_center,height), (0,255,0),2)
    cv2.line(img, (center,0), (center,height), (255,0,0),2)

    

    cv2.imwrite(result_path,img)
    cv2.imwrite(red_gray,mask_red)
    cv2.imwrite( black_gray,mask_black)
    return error

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    gray_red_path = os.path.join(gray_red,filename.replace("image","gray")).replace("\\","/")
    result_path = os.path.join(result,filename.replace("image", "result")).replace("\\","/")
    gray_black_path = os.path.join(gray_black,filename.replace("image","gray")).replace("\\","/")
    error =detect_lane(image_path,result_path,gray_black_path,gray_red_path)


