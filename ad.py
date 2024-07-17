import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import time
import scipy.spatial
from multiprocessing import Process, Queue
import copy

kernel = np.ones((7, 7))

def create_mask(img):
    img1_filter = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img1_filter, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype="uint8")
    canny = cv2.Canny(gray, 30, 30)
    dil = cv2.dilate(canny, kernel, iterations=3)
    contours, h = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = tuple(contour for contour in contours if cv2.contourArea(contour) > 5000)
    for contour in big_contours:
        p = cv2.arcLength(contour, True)
        cv2.approxPolyDP(contour, 0.001 * p, True)
    cv2.fillPoly(mask, big_contours, 255)
    return mask

def diff_to_image2(img1, img2, queue):
    frame_diff1 = cv2.absdiff(img1, img2)
    hsv = cv2.cvtColor(frame_diff1, cv2.COLOR_BGR2HSV)
    brightness = 12
    saturation = 0
    saturation_u = 20
    lower = (0, saturation, brightness)
    upper = (255, saturation_u, 255)
    gray_mask = cv2.inRange(hsv, lower, upper)
    hsv[gray_mask != 0] = [0, 0, 255]
    image_bold = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    diffs_set1 = np.where(image_bold.sum(axis=2) > 700)
    np_out = np.column_stack([diffs_set1[0], diffs_set1[1]])
    queue.put(np_out)

class MyApp(App):
    def build(self):
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.mask1 = False
        self.mask2 = False

        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)

        btn = Button(text='Start Camera')
        btn.bind(on_press=self.start_camera)
        layout.add_widget(btn)

        return layout

    def start_camera(self, instance):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            if self.img1 is None:
                self.img1 = frame
            elif self.img2 is None:
                self.img2 = frame
            else:
                self.img3 = frame
                c3, self.mask1, self.mask2 = self.get_123_sets_from_videostream_np_array(self.img1, self.img2, self.img3, self.mask1, self.mask2)
                if c3 is not None:
                    for j in range(len(c3)):
                        pt1 = (c3[j, 1] - 40, c3[j, 0] - 40)
                        pt2 = (c3[j, 1] + 40, c3[j, 0] + 40)
                        cv2.rectangle(frame, pt1, pt2, (0, 69, 255), 3)
                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image.texture = texture
                self.img1, self.img2 = self.img2, self.img3

    def get_123_sets_from_videostream_np_array(self, img1, img2, img3, mask1=False, mask2=False):
        np_out1, np_out2, np_out3, mask1, mask2 = self.make_a_list_of_diffs_3_on_numpy2(img1, img2, img3, mask1, mask2)
        c3 = self.find_near_two_points_in_numpy(np_out1, np_out2, np_out3)
        return c3, mask1, mask2

    def make_a_list_of_diffs_3_on_numpy2(self, img1, img2, img3, mask1=False, mask2=False):
        if type(mask1) == bool:
            mask1 = create_mask(img1)
        if type(mask2) == bool:
            mask2 = create_mask(img2)
        mask3 = create_mask(img3)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        img1[mask == 255] = (0, 0, 0)
        img2[mask == 255] = (0, 0, 0)
        img3[mask == 255] = (0, 0, 0)
        queue = Queue()
        queue2 = Queue()
        queue3 = Queue()
        p = Process(target=diff_to_image2, args=(img1, img2, queue))
        p2 = Process(target=diff_to_image2, args=(img2, img3, queue2))
        p3 = Process(target=diff_to_image2, args=(img1, img3, queue3))
        p.start()
        p2.start()
        p3.start()
        np_out1 = queue.get()
        np_out2 = queue2.get()
        np_out3 = queue3.get()
        return np_out1, np_out2, np_out3, mask1, mask2

    def find_near_two_points_in_numpy(self, s1, s2, s3):
        try:
            p1 = scipy.spatial.distance.cdist(s1, s2)
            dif_p1 = np.where((p1 > 5))
            a1 = s1[dif_p1[0]]
            b1 = s2[dif_p1[1]]
            point3_expect = ((2 * b1) - a1)
            p2 = scipy.spatial.distance.cdist(point3_expect, s3)
            d2 = p2.min(axis=0)
            dif_p2 = np.where((p2 < 20) & (p2 == d2))
            c3 = s3[dif_p2[1]]
            return c3
        except:
            pass

if __name__ == '__main__':
    MyApp().run()
