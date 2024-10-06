import mediapipe as mp
import os
import numpy as np
from tensorflow import lite as lt
import cv2
import string
import keyboard
import pyttsx3
import threading
import time

def image_process(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    kp = np.concatenate([lh, rh])
    return kp

engine = pyttsx3.init()
def speak(speech):
    threading.Thread(target=lambda: (engine.say(speech), engine.runAndWait())).start()

def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)


PATH = os.path.join('sign_data')

actions = ['hello','how','i','love','thanks','you']

interpreter = lt.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sentence, keypoints, last_prediction, last_printed_sentence = [], [], [], []
last_prediction_time = time.time()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()


with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        _, image = cap.read()
        results = image_process(image, holistic)
        draw_landmarks(image, results)
        image = cv2.flip(image, 1)
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 10:
            keypoints = np.array(keypoints, dtype=np.float32)
            keypoints = keypoints[np.newaxis, :, :]

            interpreter.set_tensor(input_details[0]['index'], keypoints)

            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])

            keypoints = []

            if np.amax(prediction) > 0.9:
                if last_prediction != actions[np.argmax(prediction)]:
                    sentence.append(actions[np.argmax(prediction)])
                    last_prediction = actions[np.argmax(prediction)]
                    last_prediction_time = time.time()

        if time.time() - last_prediction_time > 5:
            if sentence:
                speak(' '.join(sentence))
                sentence, keypoints, last_prediction = [], [], []

        if len(sentence) > 5:
            sentence = sentence[-5:]

        if keyboard.is_pressed('q'):
            sentence = sentence[:-1]

        if sentence:
            sentence[0] = sentence[0].capitalize()

        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (
                        sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        if sentence:
            # Join the sentence list into a string and display it
            text = ' '.join(sentence)
            cv2.putText(image,text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Camera', image)

        cv2.waitKey(1)

        if keyboard.is_pressed('esc'):
            break

    cap.release()
    cv2.destroyAllWindows()

