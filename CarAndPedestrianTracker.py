import cv2 as cv

# Reading the image
#img = img = cv.imread(r'C:\Users\Lenovo\Pictures\Camera Roll\traffic.jpg')
#video = cv.VideoCapture(r'C:\Users\Lenovo\Pictures\Camera Roll\video2.mp4')
video = cv.VideoCapture(r'C:\Users\Lenovo\Pictures\Camera Roll\video4.mp4')

#Our pre trained classifiers 
classifier_path1 = (r'C:\Users\Lenovo\Desktop\Haarcascades\cars.xml')
classifier_path2 = (r'C:\Users\Lenovo\Desktop\Haarcascades\fullbody.xml') 

# Create the car classifier
car_tracker = cv.CascadeClassifier(classifier_path1)
pedestrian_tracker = cv.CascadeClassifier(classifier_path2)

#Read until the car stops
while True:
     
    #Read the current frame 
    (read_successful,frame) = video.read()
    
    #Safe Code
    if read_successful:
        #Converting into grayscale
        grayscaled_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    else:
        break
    
    #Detect cars and pedestrians and print 
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
        
    #Draw rectangle around cars    
    for(x,y,w,h) in cars:
        cv.rectangle(frame,(x+1,y+2),(x+w,y+h),(255,0,0),2)
        cv.rectangle(frame,(x,y), (x+w, y+h), (0,0,255),2)
        
    #Draw rectangle around the pedestrians
    for(x,y,w,h) in pedestrians:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
   
   
   #Display the image with faces spotted
    cv.imshow('Current Frame',frame)
    
   #Dont autoclose (waiting n code for a key press)
    key = cv.waitKey(1)
    
    #Stop if Q key is pressed
    if key==81 or key==113:
        break
    
#Release the VideoCapture object
video.release()
    


# Convert the image to grayscale
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect cars
# cars = car_tracker.detectMultiScale(gray)
#print(cars) 

#Draw rectangle around the car
#car1 = cars[0]
#(x,y,w,h)= car1
#cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    


#cv.imshow('Gray', gray)
#cv.waitKey()
