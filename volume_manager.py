import cv2
import time
import shutil
import mediapipe as mp
import numpy as np
from pathlib import Path

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def delete_cache(input_path):
    for path in input_path.iterdir():
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
        elif path.name == 'tempCodeRunnerFile.py':
            path.unlink()
        elif path.is_dir():
            delete_cache(path)


def create_fps(img, prev_time):
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    return cur_time


def draw_rectangle(img, fact_length, min_length, max_length, bar_min, bar_max):
    interp_y = np.interp(fact_length, (min_length, max_length), (bar_min, bar_max))
    percentage = np.interp(fact_length, (min_length, max_length), (0, 100))
    cv2.rectangle(img, (50, bar_min), (85, bar_max), (255, 0, 0), 3)
    cv2.rectangle(img, (50, bar_min), (85, int(interp_y)), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{str(int(percentage))}%', (55, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    return img


class VolumeManager:
    def __init__(self) -> None:
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))

    def get_volume_range(self):
        return self.volume.GetVolumeRange()
    
    def get_current_volume(self):
        return self.volume.GetMasterVolumeLevel()

    def set_volume(self, landmarks, fact_length, min_length, max_length):
        if landmarks:
            min_vol, max_vol = self.get_volume_range()[0], self.get_volume_range()[1]
            interp_x = np.interp(fact_length, (min_length, max_length), (min_vol, max_vol))
            self.volume.SetMasterVolumeLevel(interp_x, None)
        else:
            self.volume.SetMasterVolumeLevel(self.get_current_volume(), None)


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_confid=0.5, track_confid=0.5) -> None:
        self.hands = mp.solutions.hands.Hands(static_image_mode=mode, max_num_hands=max_hands,
                                              min_detection_confidence=detection_confid,
                                              min_tracking_confidence=track_confid)
        self.draw_utils = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self.results = None
        self.thumb_no = 4
        self.index_finger_no = 8
        self.mid_finger_no = 12
        self.mid_phal_no = 10

    def draw_hand(self, img):
        self.results = self.hands.process(img)
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            self.draw_utils.draw_landmarks(img, hand, self.hand_connections)

        return img
    
    def find_positions(self, img):
        landmarks = {}
        x_positions = []
        y_positions = []

        if self.results.multi_hand_landmarks:
            height, width, _ = img.shape
            hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmarks[id] = [cx, cy]
                x_positions.append(cx)
                y_positions.append(cy)

                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            max_x = max(x_positions)
            max_y = max(y_positions)
            min_x = min(x_positions)
            min_y = min(y_positions)

            cv2.rectangle(img, (min_x - 30, min_y - 30), (max_x + 30, max_y + 30), (255, 0, 255), 2)

        return img, landmarks
    
    def check_mid_finger_up(self, landmarks):
        return landmarks[self.mid_phal_no][1] > landmarks[self.mid_finger_no][1] if landmarks else False

    def draw_mid_finger_up(self, img, landmarks):
        if landmarks:
            thumb_x, thumb_y = landmarks[self.thumb_no]
            index_finder_x, index_finger_y = landmarks[self.index_finger_no]
            mid_finder_x, mid_finger_y = landmarks[self.mid_finger_no]

            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_finder_x, index_finger_y), 15, (255, 0, 0), cv2.FILLED) 
            cv2.circle(img, (mid_finder_x, mid_finger_y), 15, (255, 0, 0), cv2.FILLED)

        return img 

    def draw_line(self, img, landmarks, min_length, max_length):
        length = 0

        if landmarks:
            thumb_x, thumb_y = landmarks[self.thumb_no]
            index_finder_x, index_finger_y = landmarks[self.index_finger_no]
            length = np.hypot((index_finder_x - thumb_x), (index_finger_y - thumb_y))

            tip_circle_color = (255, 0, 0) if length < max_length else (0, 255, 0)
 
            cv2.circle(img, (thumb_x, thumb_y), 15, tip_circle_color, cv2.FILLED)
            cv2.circle(img, (index_finder_x, index_finger_y), 15, tip_circle_color, cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_finder_x, index_finger_y), (255, 0, 0), 2)

            mid_circle_color = (255, 0, 0) if length >= min_length else (0, 255, 0)
            cv2.circle(img, ((thumb_x + index_finder_x) // 2, (thumb_y + index_finger_y) // 2), 15, mid_circle_color, cv2.FILLED)

        return img, length


def main():
    prev_time = 0

    length = 0
    min_length = 40
    max_length = 300

    bar_min = 400
    bar_max = 150

    window_name = 'Image'
    path = Path(__file__).parent

    hand_detector = HandDetector()
    volume_manager = VolumeManager()
    cap = cv2.VideoCapture(0)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 768))

        img = hand_detector.draw_hand(img=img)
        img, landmarks = hand_detector.find_positions(img=img)
        
        if landmarks and hand_detector.check_mid_finger_up(landmarks):
            img = hand_detector.draw_mid_finger_up(img, landmarks)
            cur_volume = volume_manager.get_current_volume()
            min_vol, max_vol = volume_manager.get_volume_range()[0], volume_manager.get_volume_range()[1]
            img = draw_rectangle(img=img, fact_length=cur_volume, min_length=min_vol, max_length=max_vol, bar_min=bar_min, bar_max=bar_max)
        elif not landmarks:
            cur_volume = volume_manager.get_current_volume()
            min_vol, max_vol = volume_manager.get_volume_range()[0], volume_manager.get_volume_range()[1]
            img = draw_rectangle(img=img, fact_length=cur_volume, min_length=min_vol, max_length=max_vol, bar_min=bar_min, bar_max=bar_max)
        else:
            img, length = hand_detector.draw_line(img=img, landmarks=landmarks, min_length=min_length, max_length=max_length)
            volume_manager.set_volume(landmarks=landmarks, fact_length=length, min_length=min_length, max_length=max_length)
            img = draw_rectangle(img=img, fact_length=length, min_length=min_length, max_length=max_length, bar_min=bar_min, bar_max=bar_max)

        prev_time = create_fps(img, prev_time)
        cv2.imshow(window_name, img)
        
        if cv2.waitKey(1) == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
    delete_cache(path)


if __name__ == '__main__':
    main()
