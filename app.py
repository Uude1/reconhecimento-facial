import base64
import cv2
import dlib
import numpy as np
import requests

def verify_need_update_user_list():
    try:
        response = requests.get('https://ifmatch-api.onrender.com/conf/need-update-users')
        if response.status_code == 200:
            if response.json():
                disable_flag_conf_update_users()
                return True
        else:
            print(f"Erro na chamada verify_need_update_user_list")
            return False
    except Exception as e:
        print(f"Erro ao chamar verify_need_update_user_list")
        return False

    return False

def disable_flag_conf_update_users():
    try:
        response = requests.put('https://ifmatch-api.onrender.com/conf/false')
        if response.status_code == 200:
            print(f"disable_flag_conf_update_users OK")
        else:
            print(f"Erro na chamada disable_flag_conf_update_users")
            return []
    except Exception as e:
        print(f"Erro ao chamar disable_flag_conf_update_users")
        return []

def get_user_data():
    try:
        response = requests.get('https://ifmatch-api.onrender.com/user')
        if response.status_code == 200:
            user_data = response.json()
            return user_data
        else:
            print(f"Erro na chamada get_user_data")
            return []
    except Exception as e:
        print(f"Erro ao chamar get_user_data")
        return []


def load_image_from_user_data(user_data):
    image_data = base64.b64decode(user_data['profileImg'])
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def change_user_status(frame, user_id):
    try:
        mensagem = "Rosto reconhecido!"
        cv2.putText(frame, mensagem, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)
        response = requests.get(
            'https://ifmatch-api.onrender.com/user/change-status/' + user_id + '/AGUARDANDO_ATENDIMENTO')
        if response.status_code == 200:
            get_user_data()
        else:
            print(f"Erro na chamada change_user_status")
            return []
    except Exception as e:
        print(f"Erro ao chamar change_user_status")
        return []

def main():
    user_data = get_user_data()

    if not user_data:
        print("Não foi possível obter os dados do usuário.")
        return

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('Webcam Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Face Detection', original_width, original_height)

    while True:
        if verify_need_update_user_list():
            user_data = get_user_data()

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if webcam_face_landmarks is not None:
                for user in user_data:
                    image = load_image_from_user_data(user)
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    img_faces = detector(gray_img, 0)

                    if len(img_faces) > 0:
                        face_img = img_faces[0]
                        landmarks_img = predictor(gray_img, face_img)
                        img_face_landmarks = landmarks_img.parts()

                        for point in img_face_landmarks:
                            x, y = point.x * original_width / image.shape[1], point.y * original_height / image.shape[0]
                            point.x, point.y = int(x), int(y)

                        distance = np.mean([np.linalg.norm(np.array((x1.x - x2.x, x1.y - x2.y))) for x1, x2 in
                                            zip(webcam_face_landmarks, img_face_landmarks)])

                        threshold = 20
                        if distance < threshold:
                            change_user_status(frame, str(user['idUser']))

                        cv2.imshow('Webcam Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()