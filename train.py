import os
import datetime
import time
import numpy
import cv2
import csv
import pandas
import tkinter
from tkinter import Message, Text
import tkinter.ttk as ttk
import tkinter.font as font
import shutil
from PIL import Image, ImageTk
 
def register():        
    studentID=(studentID_entry.get())
    studentName=(studentName_entry.get())
    if(studentID.isnumeric() and studentName.isalpha()):
        camera = cv2.VideoCapture(0)
        pathHarcascade = r"C:\Users\Parikshit\Documents\ai-final-project\haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(pathHarcascade)
        picNo=0
        while(True):
            ret, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            person = detector.detectMultiScale(gray, 1.1, 5)
            for (x,y,w,h) in person:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        
                cv2.imwrite("Images\ "+studentName +"."+studentID +'.'+ str(picNo) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',frame)
                picNo=picNo+1
            if cv2.waitKey(100) & 0xFF == ord('x'):
                break
            elif picNo>60:
                break
        camera.release()
        cv2.destroyAllWindows() 
        status_text = "Student registered with ID : " + studentID +" and Name : "+ studentName
        r = [studentID,  studentName]
        with open('Data\Data.csv','a+') as csv:
            w = csv.writer(csv)
            w.writerow(r)
        csv.close()
        status_text.configure(text=status_text)

    
def train():
    trainer = cv2.face.LBPHFaceRecognizer_create()
    pathHarcascade = r"C:\Users\Parikshit\Documents\ai-final-project\haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(pathHarcascade)
    persons,IDs = fetchImgData(r"C:\Users\Parikshit\Documents\ai-final-project\Images")
    trainer.train(persons, numpy.array(IDs))
    trainer.save(r"C:\Users\Parikshit\Documents\ai-final-project\Trainner.yml")
    train_text = "Training Complete!"
    status_text.configure(text= train_text)

def fetchImgData(path):
    imageFiles=[os.path.join(path,f) for f in os.listdir(path)] 
    persons=[]
    IDs=[]

    for imageFile in imageFiles:
        img=Image.open(imageFile).convert('L')
        grey=numpy.array(img,'uint8')
        ID=int(os.path.split(imageFile)[-1].split(".")[1])
        persons.append(grey)
        IDs.append(ID)        
    return persons,IDs

def record():
    recorder = cv2.face.LBPHFaceRecognizer_create()
    pathHarcascade = r"C:\Users\Parikshit\Documents\ai-final-project\haarcascade_frontalface_default.xml"
    recorder.read(r"C:\Users\Parikshit\Documents\ai-final-project\Trainner.yml")
    cascadeFace = cv2.CascadeClassifier(pathHarcascade);    
    data=pandas.read_csv(r"C:\Users\Parikshit\Documents\ai-final-project\StudentDetails\StudentDetails.csv")
    camera = cv2.VideoCapture(0)
    columns =  ['StudentID','StudentName','Date','Time']
    f = cv2.FONT_HERSHEY_COMPLEX        
    logs = pandas.DataFrame(columns = columns)    
    while True:
        ret, im =camera.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=cascadeFace.detectMultiScale(gray, 1.1,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            ID, confidence = recorder.predict(gray[y:y+h,x:x+w])                                   
            if(confidence < 50):
                timestamp = time.time()      
                date_object = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                time_object = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                xx=data.loc[data['StudentID'] == ID]['StudentName'].values
                img_text=str(ID)+"-"+xx
                logs.loc[len(logs)] = [ID,xx,date_object,time_object]
                
            else:
                ID='Unknown'                
                img_text=str(ID)  
            if(confidence > 75):
                noOfFile=len(os.listdir("Unknown"))+1
                cv2.imwrite("Unknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(img_text),(x,y+h), f, 1,(255,255,255),2)        
        logs=logs.drop_duplicates(subset=['StudentID'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('x')):
            break
    timestamp = time.time()      
    date_object = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    time_object = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    Hour,Minute,Second=time_object.split(":")
    filePath="Logs\Attendance_"+date_object+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    logs.to_csv(filePath,index=False)
    camera.release()
    cv2.destroyAllWindows()
    logs_text=logs
    attendance_text.configure(text=logs_text)


application = tkinter.Tk()
application.title("Automated Attendance System")
application.grid_rowconfigure(0, weight=1)
application.configure(background='white')
application.grid_columnconfigure(0, weight=1)

title = tkinter.Label(application, text="Automated Attendance System", width=60, height=3, bg="blue", fg="black", font=('arial', 30, 'bold')) 
title.place(x=0, y=0)

studentID_label = tkinter.Label(application, text="Enter Student ID", width=20, height=2, fg="black", bg="blue", font=('arial', 14)) 
studentID_label.place(x=300, y=180)

studentID_entry = tkinter.Entry(application, width=20, bg="blue", fg="black",font=('arial', 14))
studentID_entry.place(x=600, y=195)

studentName_label = tkinter.Label(application, text="Enter Student Name", width=20, fg="black", bg="blue", height=2, font=('arial', 14)) 
studentName_label.place(x=300, y=280)

studentName_entry = tkinter.Entry(application, width=20, bg="blue", fg="black",font=('arial', 14))
studentName_entry.place(x=600, y=295)

status_label = tkinter.Label(application, text="Status", width=20, fg="black", bg="blue", height=2, font=('arial', 15)) 
status_label.place(x=300, y=380)

status_text = tkinter.Label(application, text="", bg="blue", fg="black", width=30, height=2, font=('arial', 15)) 
status_text.place(x=600, y=380)

registerStudent = tkinter.Button(application, text="Register", command=register, bg="blue", fg="black", width=20, height=3, font=('arial', 15))
registerStudent.place(x=100, y=460)

trainButton = tkinter.Button(application, text="Train", command=train, bg="blue", fg="black", width=20, height=3, font=('arial', 15))
trainButton.place(x=400, y=460)

recordButton = tkinter.Button(application, text="Record", command=record, bg="blue", fg="black", width=20, height=3, font=('arial', 15))
recordButton.place(x=700, y=460)

quitButton = tkinter.Button(application, text="Exit", command=application.destroy, bg="blue", fg="black", width=20, height=3, font=('arial', 15))
quitButton.place(x=1000, y=460)

attendance_label = tkinter.Label(application, text="Attendance Logs", width=20, fg="black", bg="blue", height=2, font=('arial', 15)) 
attendance_label.place(x=200, y=580)

attendance_text = tkinter.Label(application, text="", bg="blue", fg="black", width=60, height=3, font=('arial', 15)) 
attendance_text.place(x=500, y=570)

 
application.mainloop()