import base64
import cv2
import dlib
import numpy as np
from confluent_kafka import Producer
import requests

conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'webcam_face_detection_producer'
}

producer = Producer(conf)

user_data = []

def produce_kafka_message(message):
    try:
        topic = 'FACE_MATCHED'

        if not isinstance(message, str):
            message = str(message)

        message_bytes = message.encode('utf-8')
        producer.produce(topic, key=None, value=message_bytes)
        producer.flush()

    except Exception as e:
        print(f"Erro ao produzir mensagem no Kafka: {e}")

def get_user_data():
    try:
        response = requests.get('http://localhost:8080/user')
        if response.status_code == 200:
            user_data = response.json()
            return user_data
        else:
            print(f"Erro na solicitação HTTP GET: Status Code {response.status_code}")
            return []
    except Exception as e:
        print(f"Erro na solicitação HTTP GET: {e}")
        return []

def load_image_from_user_data(user_data):
    image_data = base64.b64decode(user_data['profileImg'])
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def main():
    user_data = get_user_data()

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

            found_similar_face = False

            if webcam_face_landmarks is not None:
                for user in user_data:
                    image = load_image_from_user_data(user)
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    frame_resized = cv2.resize(frame, (int(frame.shape[1] * image.shape[0] / frame.shape[0]), image.shape[0]))

                    img_faces = detector(gray_img, 0)

                    if len(img_faces) > 0:
                        face_img = img_faces[0]
                        landmarks_img = predictor(gray_img, face_img)
                        img_face_landmarks = landmarks_img.parts()

                        for point in img_face_landmarks:
                            x, y = point.x * original_width / image.shape[1], point.y * original_height / image.shape[0]
                            point.x, point.y = int(x), int(y)

                        distance = np.mean([np.linalg.norm(np.array((x1.x - x2.x, x1.y - x2.y))) for x1, x2 in zip(webcam_face_landmarks, img_face_landmarks)])

                        threshold = 20
                        if distance < threshold:
                            message = user['idUser']
                            produce_kafka_message(message)

            if not found_similar_face:
                cv2.imshow('Webcam Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
