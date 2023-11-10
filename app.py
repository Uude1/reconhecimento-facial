import cv2
import dlib
import os
import numpy as np

cap = cv2.VideoCapture(2) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

webcam_face_landmarks = None

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('Webcam Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Face Detection', original_width, original_height)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        webcam_face_landmarks = landmarks.parts()

        for point in webcam_face_landmarks:
            x, y = point.x * original_width / frame.shape[1], point.y * original_height / frame.shape[0]
            point.x, point.y = int(x), int(y)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        img_folder_path = "img/"
        found_similar_face = False

        if webcam_face_landmarks is not None:
            for img_filename in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, img_filename)

                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                frame_resized = cv2.resize(frame, (int(frame.shape[1] * img.shape[0] / frame.shape[0]), img.shape[0]))

                img_faces = detector(gray_img, 0)

                if len(img_faces) > 0:
                    face_img = img_faces[0]
                    landmarks_img = predictor(gray_img, face_img)
                    img_face_landmarks = landmarks_img.parts()

                    for point in img_face_landmarks:
                        x, y = point.x * original_width / img.shape[1], point.y * original_height / img.shape[0]
                        point.x, point.y = int(x), int(y)

                    distance = np.mean([np.linalg.norm(np.array((x1.x - x2.x, x1.y - x2.y))) for x1, x2 in zip(webcam_face_landmarks, img_face_landmarks)])

                    threshold = 20  
                    if distance < threshold:
                        found_similar_face = True
                        print(f"Rosto na webcam é parecido com {img_filename} (Distância: {distance})")

                        combined_image = np.concatenate((frame_resized, img), axis=1)
                        cv2.imshow('Webcam vs. Image', combined_image)
                        cv2.waitKey(0) 

        if not found_similar_face:
            cv2.imshow('Webcam Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
