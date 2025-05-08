import cv2
import face_recognition
import mediapipe as mp
from deepface import DeepFace

# Загрузка изображения с известным лицом
known_image = face_recognition.load_image_file("me.jpg")
# Определение местоположения лица на изображении
known_face_locations = face_recognition.face_locations(known_image)
if known_face_locations:
    # Получение кодировки лица
    known_face_encoding = face_recognition.face_encodings(known_image, known_face_locations)[0]
else:
    raise ValueError("Не удалось обнаружить лицо на изображении 'me.jpg'.")

# Инициализация MediaPipe Hands для распознавания рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Режим для видео
    max_num_hands=1,               # Максимум 1 рука
    min_detection_confidence=0.5   # Минимальная уверенность для обнаружения
)
mp_draw = mp.solutions.drawing_utils  # Утилиты для рисования аннотаций

# Инициализация захвата видео с веб-камеры
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Установка ширины кадра
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Установка высоты кадра

def detect_emotion(face_roi):
    """
    Определение эмоции на основе области лица с помощью DeepFace
    """
    try:
        # Преобразование ROI в RGB и изменение размера до 224x224
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        # Анализ эмоций с использованием DeepFace
        result = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)
        # DeepFace возвращает список; извлекаем первую запись
        return result[0]['dominant_emotion']
    except Exception as e:
        print("Ошибка определения эмоции:", e)
        return "unknown"

while True:
    ret, frame = cap.read()  # Чтение кадра с камеры
    if not ret:
        break

    # Уменьшение размера кадра для ускорения обработки
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Преобразование BGR в RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    # Обнаружение лиц на уменьшенном кадре
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # Получение кодировок лиц
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    is_known = False       # Флаг для известного лица
    face_detected = False  # Флаг для обнаружения лица
    top, right, bottom, left = 0, 0, 0, 0  # Координаты рамки лица

    # Обработка каждого обнаруженного лица
    for (top_sm, right_sm, bottom_sm, left_sm), face_encoding in zip(face_locations, face_encodings):
        # Сравнение с известной кодировкой лица
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if matches[0]:
            is_known = True
            face_detected = True
            # Масштабирование координат обратно к оригинальному размеру кадра
            top = top_sm * 4
            right = right_sm * 4
            bottom = bottom_sm * 4
            left = left_sm * 4
            # Рисование зелёной рамки вокруг известного лица
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            face_detected = True
            top = top_sm * 4
            right = right_sm * 4
            bottom = bottom_sm * 4
            left = left_sm * 4
            # Рисование красной рамки вокруг неизвестного лица
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Преобразование кадра в RGB для обработки MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Обнаружение рук
    results = hands.process(rgb_frame)

    count = 0  # Счётчик поднятых пальцев
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Рисование аннотаций руки
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            fingers = []
            # Определение состояния каждого пальца (поднят или нет)
            # Большой палец
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
            # Указательный палец
            if landmarks[8].y < landmarks[6].y:
                fingers.append(1)
            else:
                fingers.append(0)
            # Средний палец
            if landmarks[12].y < landmarks[10].y:
                fingers.append(1)
            else:
                fingers.append(0)
            # Безымянный палец
            if landmarks[16].y < landmarks[14].y:
                fingers.append(1)
            else:
                fingers.append(0)
            # Мизинец
            if landmarks[20].y < landmarks[18].y:
                fingers.append(1)
            else:
                fingers.append(0)
            # Подсчёт поднятых пальцев
            count = sum(fingers)

    # Обработка действий на основе количества поднятых пальцев
    if face_detected:
        if is_known:
            if count == 1:
                # Отображение имени при 1 поднятом пальце
                cv2.putText(frame, "Ayrat", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif count == 2:
                # Отображение фамилии при 2 поднятых пальцах
                cv2.putText(frame, "Fakhrutdinov", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif count == 3:
                # Определение и отображение эмоции при 3 поднятых пальцах
                face_roi = frame[top:bottom, left:right]
                if face_roi.size != 0:
                    emotion = detect_emotion(face_roi)
                    if emotion:
                        cv2.putText(frame, f"Emotion: {emotion}", (left, bottom + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Отображение "Unknown" для неизвестного лица
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Отображение кадра с аннотациями
    cv2.imshow('Face & Hand Recognition', frame)

    # Выход из цикла при нажатии esc
    if cv2.waitKey(1) == 27:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
