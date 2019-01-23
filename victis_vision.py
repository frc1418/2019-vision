#!/usr/bin/env python3

try:
    import cscore as cs
    CSCORE = True
except ImportError:
    CSCORE = False

import cv2
import numpy as np

from enum import Enum
from image_processor import ImageProcessor
import argparse

from networktables import NetworkTable
from networktables.util import ntproperty

import math

class ImageProcessor:
    # Values for the lifecam-3000
    VFOV = 45.6 # Camera's vertical field of view
    HFOV = 61 # Camera's horizontal field of view

    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)

    min_width = ntproperty('/camera/min_width', 2)
    min_height = ntproperty('/camera/min_height', 2)

    thresh_hue_lower = ntproperty('/camera/thresholds/hue_low', 60)
    thresh_hue_high = ntproperty('/camera/thresholds/hue_high', 100)
    thresh_sat_lower = ntproperty('/camera/thresholds/sat_low', 150)
    thresh_sat_high = ntproperty('/camera/thresholds/sat_high', 255)
    thresh_val_lower = ntproperty('/camera/thresholds/val_low', 140)
    thresh_val_high = ntproperty('/camera/thresholds/val_high', 255)

    square_tolerance = ntproperty('/camera/square_tolerance', 10)
    broken_tolerance_x = ntproperty('/camera/broken_tolerance_x', 2)
    broken_tolerance_y = ntproperty('/camera/broken_tolerance_y', 20)

    target_spacing = ntproperty('/camera/target_spacing', 2)

    draw_thresh = ntproperty('/camera/draw_thresh', True)
    draw_approx = ntproperty('/camera/draw_approx', False)
    draw_approx2 = ntproperty('/camera/draw_approx2', False)
    draw_contours = ntproperty('/camera/draw_contours', False)
    draw_target_patch = ntproperty('/camera/draw_target_patch', False)
    draw_target = ntproperty('/camera/draw_target', True)

    def __init__(self):
        self.size = None
        self.thresh_low = np.array([self.thresh_hue_lower, self.thresh_sat_lower, self.thresh_val_lower], dtype=np.uint8)
        self.thresh_high = np.array([self.thresh_hue_high, self.thresh_sat_high, self.thresh_val_high], dtype=np.uint8)

        self.nt = NetworkTable.getTable('/camera')

    def preallocate(self, img):
        if self.size is None or self.size[0] != img.shape[0] or self.size[1] != img.shape[1]:
            h, w = img.shape[:2]
            self.size = (h, w)

            self.img = np.empty((h, w, 3), dtype=np.uint8)

            self.hsv = np.empty((h, w, 3), dtype=np.uint8)
            self.bin = np.empty((h, w, 1), dtype=np.uint8)
            self.bin2 = np.empty((h, w, 1), dtype=np.uint8)

            self.out = np.empty((h, w, 3), dtype=np.uint8)

            # for overlays
            self.zeros = np.zeros((h, w, 1), dtype=np.bool)
            self.black = np.zeros((h, w, 3), dtype=np.uint8)

            self.morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))

        cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)

    def threshhold(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=self.hsv)
        cv2.inRange(self.hsv, self.thresh_low, self.thresh_high, dst=self.bin)

        cv2.morphologyEx(self.bin, cv2.MORPH_CLOSE, self.morphKernel, dst=self.bin2, iterations=1)

        if self.draw_thresh:
            b = (self.bin2 != 0)
            cv2.copyMakeBorder(self.black, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
            self.out[np.dstack((b, b, b))] = 255

        return self.bin2

    def find_contours(self, img):
        thresh_img = self.threshhold(img)

        _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

            if self.draw_approx:
                cv2.drawContours(self.out, [approx], -1, self.BLUE, 2, lineType=8)

            if len(approx) > 3 and len(approx) < 15:
                _,_,w,h = cv2.boundingRect(approx)
                if h > self.min_height and w > self.min_width:
                    hull = cv2.convexHull(cnt)
                    approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)

                    if self.draw_approx2:
                        cv2.drawContours(self.out, [approx2], -1, self.GREEN, 2, lineType=8)

                    result.append(approx2)
        return result

    def get_contour_info(self, contour):
        contour_info = {}

        contour_info['x'], contour_info['y'], contour_info['w'], contour_info['h'] = cv2.boundingRect(contour)

        contour_info['cx'] = contour_info['x'] + contour_info['w'] / 2
        contour_info['cy'] = contour_info['y'] + contour_info['h'] / 2

        return contour_info

    def process_for_target(self, contours, time):
        # Filter contours for complete targets and possible 'broken targets'
        self.targets = []

        for c in contours:
            target_info = self.get_contour_info(c)
            target_info['cnt'] = c

            self.targets.append(target_info)

        self.full_targets = []
        # Group contours together if within a certain tolerance
        for i, b in enumerate(self.targets[:]):
            matched = False
            for i2, b2 in enumerate(self.targets[i+1:]):
                if b['cx'] >= b2['cx'] - self.broken_tolerance_x and b['cx'] <= b2['cx'] + self.broken_tolerance_x:
                    matched = True
                    new_blob = np.concatenate([b['cnt'], b2['cnt']])

                    hull = cv2.convexHull(new_blob)
                    new_blob = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)

                    target_info = self.get_contour_info(new_blob)
                    target_info['cnt'] = new_blob

                    self.full_targets.append(target_info)
                    self.targets.remove(b)

                    break
            if not matched:
                self.full_targets.append(b)

        # Draw targets after `patching` them together
        if self.draw_target_patch:
            contours = []
            for g in self.full_targets:
                cv2.drawContours(self.out, [g['cnt']], -1, self.YELLOW, 2, lineType=8)
                contours.append(g['cnt'])

        # Break out of loop if no complete targets
        if len(self.full_targets) == 0:
            self.nt.putBoolean('target_present', False)
            return self.out

        # Find the target that is closest to the center
        h = float(self.size[0])
        w = float(self.size[1])

        primary_target = None
        for i, g in enumerate(self.full_targets[:]):
            greater_than = True

            for g2 in self.full_targets[i+1:]:
                if g['cx'] - (h / 2) < g2['cx'] - (h / 2):
                    greater_than = False

            if greater_than:
                primary_target = g
                self.full_targets.remove(g)
                break


        # Find the another close target if present
        main_target_contour = primary_target['cnt']
        secondary_target = None
        partial = True
        if len(self.full_targets) > 0:
            for i, g in enumerate(self.full_targets):
                greater_than = True

                if abs(g['cx'] - primary_target['cx']) < self.target_spacing * primary_target['h']:

                    for g2 in self.full_targets[i:]:
                        if g['cx'] - (h / 2) < g2['cx'] - (h / 2):
                            greater_than = False
                else:
                    greater_than = False

                if greater_than:
                    secondary_target = self.get_contour_info(g['cnt'])
                    main_target_contour = np.concatenate([g['cnt'],main_target_contour])
                    partial = False
                    break

        # Preform math on contours to make them useful
        hull = cv2.convexHull(main_target_contour)
        main_target_contour = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)

        cnt_info = self.get_contour_info(main_target_contour)
        height = self.VFOV * target_info['cy'] / h - self.VFOV/2.0
        angle = self.HFOV * target_info['cx'] / w - self.HFOV/2.0
        print('Height %s' % height)
        print('Angle %s' % angle)

        self.nt.putBoolean('target_present', True)
        self.nt.putBoolean('target_partial', partial)

        if not partial:
            skew = 0
            if primary_target['h'] < secondary_target['h']:
                skew = secondary_target['h']/primary_target['h']
                skew -= 1
                if primary_target['cx'] < secondary_target['cx']:
                    skew *= -1
            else:
                skew = primary_target['h']/secondary_target['h']
                skew -= 1
                if secondary_target['cx'] < primary_target['cx']:
                    skew *= -1
            print("Skew %s" % skew)
            self.nt.putNumber('target_skew', skew)

        self.nt.putNumber('target_angle', angle)
        self.nt.putNumber('target_height', height)

        if self.draw_target:
            cv2.drawContours(self.out, [main_target_contour], -1, self.RED, 2, lineType=8)


    def process_frame(self, frame, time):
        self.preallocate(frame)

        cnt = self.find_contours(frame)

        self.process_for_target(cnt, time)

        return self.out

class VisionMode:
    CSCORE_STREAM_ONLY = 1
    CSCORE_WITH_STREAM = 2
    PHOTO_WITH_IMSHOW = 3

class VictisVision:
    enabled = ntproperty('/camera/enabled', False)

    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop('mode', VisionMode.CSCORE_WITH_STREAM)

        # Don't mess with these values for now.
        self.width = 320 #kwargs.pop('width', 320)
        self.height = 240 #kwargs.pop('height', 240)

        if self.mode == VisionMode.CSCORE_WITH_STREAM or self.mode == VisionMode.CSCORE_STREAM_ONLY:
            if not CSCORE:
                raise 'Error: cscore option selected but cscore failed to import'

        NetworkTable.setIPAddress(kwargs.pop('nt_address', 'localhost'))
        NetworkTable.setClientMode()
        NetworkTable.initialize()

        self.nt = NetworkTable.getTable('/camera')

        self.processor = ImageProcessor()

        if self.mode in (1,2):
            self.setup_cscore_stream(kwargs.pop('camera_port', 0), kwargs.pop('stream_port', 8081))
        if self.mode == VisionMode.CSCORE_WITH_STREAM:
            self.setup_cscore_cv(kwargs.pop('stream_cv', True), kwargs.pop('cv_stream_port', 8082))
        if self.mode == VisionMode.PHOTO_WITH_IMSHOW:
            self.process_photo(kwargs.pop('photo_path', None))

        self.process()

    def setup_cscore_stream(self, camera_port, stream_port):
        self.camera = cs.UsbCamera('usbcam', camera_port)
        self.camera.setVideoMode(cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)

        self.mjpegServer = cs.MjpegServer('httpserver', stream_port)
        self.mjpegServer.setSource(self.camera)

    def setup_cscore_cv(self, stream, port):
        if self.camera is None:
            raise 'Camera not intialized'

        self.cvsink = cs.CvSink('cvsink')
        self.cvsink.setSource(self.camera)

        if stream:
            self.cvSource = cs.CvSource('cvsource', cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)
            self.cvMjpegServer = cs.MjpegServer('cvhttpserver', port)
            self.cvMjpegServer.setSource(self.cvSource)

    def process(self):
        img = np.zeros(shape=(self.height, self.width, 3), dtype=np.uint8)

        while True:
            if self.mode == VisionMode.CSCORE_WITH_STREAM:
                time, img = self.cvsink.grabFrame(img)

                if time == 0:
                    print('error:', self.cvsink.getError())
                    continue
                if not self.enabled:
                    self.nt.putBoolean('target_present', False)
                    self.cvSource.putFrame(img)
                    continue

                out = self.processor.process_frame(img, time)

                self.cvSource.putFrame(out)

    def process_photo(self, path):
        if path is None:
            raise 'photo path must be provided'

        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))
        out = self.processor.process_frame(img)

        cv2.imshow('Frame', out)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nt-address', default='localhost', help='Adress of NetworkTables server')

    parser.add_argument('-s','--stream-only', action='store_true', default=False, help='Streams only camera output')
    parser.add_argument('-cv','--cv-stream', action='store_true', default=False, help='Streams camera output and runs it through OpenCv processing')
    parser.add_argument('-i','--image', action='store_true', default=False, help='Processes single photo image')

    parser.add_argument('--camera-port', default=0, help='Port of camera if using \'-s\' or \'-cv\'')
    parser.add_argument('--stream-port', default=8081, help='Port of camera stream if using \'-s\' or \'-cv\'')

    parser.add_argument('--stream-cv', action='store_true', default=False, help='Stream out after OpenCV processing if using \'-cv\'')
    parser.add_argument('--cvstream-port', default=8082, help='Port of OpenCV stream if using \'-cv\'')

    parser.add_argument('--photo-path', default=None, help='Path of photo if using \'-i\'')

    args = parser.parse_args()

    mode = None
    if args.stream_only:
        mode = VisionMode.CSCORE_STREAM_ONLY
    if args.cv_stream:
        if mode is not None:
            raise 'Multiple modes set please use only \'-i\', \'-s\', or \'-cv\''
        mode = VisionMode.CSCORE_WITH_STREAM
    if args.image:
        if mode is not None:
            raise 'Multiple modes set please use only \'-i\', \'-s\', or \'-cv\''

        if args.photo_path is None:
            raise 'Photo path must be passed in image mode'

        mode = VisionMode.PHOTO_WITH_IMSHOW

    if mode is None:
        raise 'No vision mode set!'

    vision = VictisVision(mode=mode,
            nt_address=args.nt_address,
            camera_port=int(args.camera_port),
            stream_port=int(args.stream_port),
            stream_cv=args.stream_cv,
            cv_stream_port=int(args.cvstream_port),
            photo_path=args.photo_path)
