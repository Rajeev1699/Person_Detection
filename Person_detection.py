import cv2

cap = cv2.VideoCapture('video.mp4')

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)

    # Display the resulting frame
    c=1

    for (x,y,w,h) in humans:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame,
                'PERSON DETECTION',
                (200, 30),
                font, 1,        # Font Size
                (0, 0, 255),    # Color Code
                3,              # Thikness
                cv2.LINE_4)


        cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        c += 1

        cv2.putText(frame, "PERSONS: "+str(c), (10, 90), font, 0.8, (0, 0, 255), 3, cv2.LINE_4)



    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
