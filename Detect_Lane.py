import cv2
import numpy as np
import os
total = 0
count =0
image_folder = "E:/AutoCar/Data"
gray_red = "E:/AutoCar/Gray_red"
gray_black =  "E:/AutoCar/Gray_black"
result = "E:/AutoCar/Result"


def detect_lane(image_path, result_path,black_gray,red_gray):
    global total,count
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    width_left = width // 4
    width_right = width * 3//4
    top = height* 2//3
    cropped_img = img[top:, width_left:width_right]
    red_img = img[top:, :width_right]

    hsv_red = cv2.cvtColor(red_img, cv2.COLOR_BGR2HSV)
    hsv_black = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    center = width // 2

    x_black = None
    x_red = None
    lower_black1 = np.array([1.5, 5*2.55, 10*2.55])
    upper_black1 = np.array([42//2, 17*2.55, 37*2.55])  

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
    mask_black = mask_black1 | mask_black2 | mask_black3 | mask_black4



    lower_red1 = np.array([340//2, 25*2.55, 30*2.55])
    upper_red1 = np.array([360//2, 50*2.55, 55*2.55])
    lower_red2 = np.array([0, 35*2.55, 28*2.55])
    upper_red2 = np.array([10//2, 80*2.55, 55*2.55])
    lower_red3 = np.array([347//2, 68*2.55 , 25*2.55])
    upper_red3 = np.array([360*2.55,100*2.55, 50*2.55])
    lower_red4 = np.array([0//2, 68*2.55 , 10*2.55])
    upper_red4 = np.array([10*2.55,100*2.55, 30*2.55])

    mask_red1 = cv2.inRange(hsv_red, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_red, lower_red2, upper_red2)
    mask_red3 = cv2.inRange(hsv_red, lower_red3, upper_red3)
    mask_red4 = cv2.inRange(hsv_red, lower_red4, upper_red4)
    mask_red = mask_red1  | mask_red2 | mask_red4 | mask_red3
   
    mask_red = cv2.erode(mask_red, None, iterations=1)
    mask_red = cv2.dilate(mask_red, None, iterations=1)

    mask_black = cv2.erode(mask_black, None, iterations=1)
    mask_black = cv2.dilate(mask_black, None, iterations=1)

  
    red_lines = cv2.HoughLinesP(mask_red, 1, np.pi/180, threshold=60, minLineLength=100, maxLineGap=100)
    if red_lines is not None:
        top_red = sorted(
            [(x1, y1, x2, y2, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
             for [[x1, y1, x2, y2]] in red_lines],
            key=lambda x: x[4],
            reverse=True
        )
        x1, y1, x2, y2, _ = top_red[0]
        
       
        x_red = (x1 + x2) / 2
        
        y1 += top
        y2 += top
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
        x1+= width_left
        x2+=width_left
        y1+= top
        y2+= top
     
        x_black = (x1 + x2) / 2
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    error = None
    if x_red is not None and x_black is not None:
        x_center = (x_black+x_red)//2
        error = x_center - center
       

    elif x_red is not None:
        x_center = int(x_red - 190)
        error = x_center - center
        

    else:
        print(f"{image_path}Mất vạch")
    if x_black is not None and x_red is not None:
        with_lane = int(x_black - x_red)
    else:
        
        with_lane = None  

    if with_lane:
        total += with_lane
        count +=1

    

    cv2.imwrite(result_path,img)
    cv2.imwrite(red_gray,mask_red)
    cv2.imwrite( black_gray,mask_black)
    return error
        
class PIDController:
    def __init__(self, Kp=0.4, Ki=0.01, Kd=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.loi_truoc = 0
        self.tong_loi = 0

    def cap_nhat(self, loi, dt=1):
        self.tong_loi += loi * dt
        dao_ham_loi = (loi - self.loi_truoc) / dt
        output = (self.Kp * loi +
                  self.Ki * self.tong_loi +
                  self.Kd * dao_ham_loi)
        self.loi_truoc = loi
        return output

pid = PIDController(Kp=0.4, Ki=0.01, Kd=0.1)

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    gray_red_path = os.path.join(gray_red,filename.replace("image","gray")).replace("\\","/")
    result_path = os.path.join(result,filename.replace("image", "result")).replace("\\","/")
    gray_black_path = os.path.join(gray_black,filename.replace("image","gray")).replace("\\","/")
    error =detect_lane(image_path,result_path,gray_black_path,gray_red_path)

print(total/count)


