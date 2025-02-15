#!/usr/bin/env python3
import json
import time
import sys

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
import math


# Lifecam 3000
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
DIAGONAL_FOV = math.radians(68.5)

# 16:9 aspect ratio
HORIZONTAL_ASPECT = 16
VERTICAL_ASPECT = 9

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270

# Used to calculate horizontal FOV
DIAGONAL_ASPECT = math.hypot(HORIZONTAL_ASPECT, VERTICAL_ASPECT)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
HORIZONTAL_FOV = math.atan(math.tan(DIAGONAL_FOV/2) * (HORIZONTAL_ASPECT / DIAGONAL_ASPECT)) * 2
VERTICAL_FOV = math.atan(math.tan(DIAGONAL_FOV/2) * (VERTICAL_ASPECT / DIAGONAL_ASPECT)) * 2

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = IMAGE_WIDTH / (2 * math.tan(HORIZONTAL_FOV / 2))
V_FOCAL_LENGTH = IMAGE_HEIGHT / (2 * math.tan(VERTICAL_FOV / 2))

MIN_CONTOUR_SIZE = 1

def threshold_frame(frame):
    """
    Calculate mask by thresholding input frame.
    """
    img = frame.copy()
    blur = cv2.medianBlur(img, 3)

    # Get image in HSV colorspace
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Define HSV range of bright green features
    lower_threshold = np.array([50, 150, 50])
    upper_threshold = np.array([100, 255, 255])
    # Extract qualifying pixels from image
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    return mask


def find_contours(mask):
    """
    Find contours of image mask, displaying them over original stream.
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    return contours


def find_targets(contours, frame):
    """
    Find targets and draw on frame.

    :param contours: list of contours among which to find targets.
    :param frame: image upon which to draw contours.
    """
    # If there aren't any contours present, return frame without drawing
    if len(contours) == 0:
        return frame
    # Copy frame, TODO why do we need to do this?
    image = frame.copy()
    screen_height, screen_width, _ = image.shape;
    # TODO: Why subtract?
    center_x = screen_width / 2 - .5
    center_y = screen_height / 2 - .5
    # List for storing found targets
    targets = []

    if len(contours) >= 2:
        # Sort contours in descending order by size
        contours.sort(key=lambda contour: cv2.contourArea(contour), reverse=True)

        valid_contours = []
        for contour in contours:
            # Calculate areas of contour
            contour_area = cv2.contourArea(contour)
            if contour_area >= MIN_CONTOUR_SIZE:
                # Get moments of contour for centroid calculations
                moments = cv2.moments(contour)
                # Find centroid of contour
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = 0, 0

                ### CALCULATE CONTOUR ROTATION BY FITTING ELLIPSE ###
                rotation = get_ellipse_rotation(image, contour)

                ### DRAW CONTOUR ###
                # Draw white circle at center of contour
                cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                # Draw contour in green
                cv2.drawContours(image, [contour], 0, (0, 200, 0), 1)

                # Append important info to array
                valid_contours.append({"cx": cx, "cy": cy, "rotation": rotation})

        # Sort array based on coordinates (left to right) to make sure contours are adjacent
        valid_contours.sort(key=lambda contour: contour["cx"])

        # Find targets from contours
        for i in range(len(valid_contours) - 1):
            # Check rotation of adjacent contours
            tilt_left = valid_contours[i]["rotation"]
            tilt_right = valid_contours[i + 1]["rotation"]

            # Contour coordinates
            cx_left = valid_contours[i]["cx"]
            cx_right = valid_contours[i + 1]["cx"]
            cy_left = valid_contours[i]["cy"]
            cy_right = valid_contours[i + 1]["cy"]

            # If contour angles are opposite
            # Negative tilt -> Rotated to the right
            # NOTE: if using rotated rect (min area rectangle), negative tilt means rotated to left
            # If left contour rotation is tilted to the left then skip iteration
            # If right contour rotation is tilted to the right then skip iteration
            if (len(valid_contours) == 2) or (np.sign(tilt_left) != np.sign(tilt_right) and
                    not (tilt_left > 0 and cx_left < cx_right or tilt_right > 0 and cx_right < cx_left)):

                target_cx = (cx_left + cx_right) / 2
                target_cy = (cy_left + cy_right) / 2

                target_yaw = calculate_yaw(target_cx, center_x)
                target_pitch = calculate_pitch(target_cy, center_y)

                targets.append({"cx": target_cx,
                                "cy": target_cy,
                                "yaw": target_yaw,
                                "pitch": target_pitch})

    # Check if there are targets seen
    if len(targets) > 0:
        # Get target with smallest yaw
        nearest_target = min(targets, key=lambda target: math.fabs(target["yaw"]))
        # Write yaw of target in corner of image
        cv2.putText(image, "Yaw: %.3f" % nearest_target["yaw"], (1, 12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # Draw line at center of target
        cv2.line(image, (int(nearest_target["cx"]), screen_height), (int(nearest_target["cx"]), 0), (255, 0, 0), 1)
        # Draw line at center of screen
        cv2.line(image, (round(center_x), screen_height), (round(center_x), 0), (255, 255, 255), 1)

        # Send our final data to NetworkTables
        table.putBoolean("target_present", True)
        table.putNumber("targets_seen", len(targets))
        table.putNumber("target_yaw", nearest_target["yaw"])
        table.putNumber("target_pitch", nearest_target["pitch"])
    else:
        table.putBoolean("target_present", False)
        table.putNumber("targets_seen", 0)
        table.putNumber("target_yaw", 0)
        table.putNumber("target_pitch", 0)
        table.putNumber("target_distance", 0)

    return image


# Forgot how exactly it works, but it works!
def translate_rotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculate_distance(pitch):
    """
    Use trig and pitch to find distance to target.

    d = distance
    h = height between camera and target
    a = angle = pitch

    tan a = h/d (opposite over adjacent)

    d = h / tan a

                 .
                /|
               / |
              /  |h
             /a  |
      camera -----
               d

    :param camera_height: height of camera from ground.
    :param target_height: height of target from ground.
    :param pitch: angle of camera.
    """
    # TODO: Use actual values calculated
    target_height = 60
    camera_height = 40
    height_difference = target_height - camera_height
    distance = math.fabs(height_difference / math.tan(math.radians(pitch)))

    return distance


def calculate_yaw(pixel_x, center_x) -> float:
    """
    Use trig and focal length of camera to find yaw.

    Explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    """
    yaw = math.degrees(math.atan((pixel_x - center_x) / H_FOCAL_LENGTH))
    return yaw


def calculate_pitch(pixel_y, center_y) -> float:
    """
    Explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    """
    pitch = -math.degrees(math.atan((pixel_y - center_y) / V_FOCAL_LENGTH))
    return pitch

def get_ellipse_rotation(image, contour):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(contour)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translate_rotation(rotation, widthE, heightE)

        # Gets smaller side
        if widthE > heightE:
            smaller_side = heightE
        else:
            smaller_side = widthE

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(contour)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translate_rotation(rotation, width, height)
        return rotation

#################### FRC VISION PI Image Specific #############
config_file = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

def parseError(message):
    """
    Cleanly report config parsing error.
    """
    print("config error in " + config_file + ": " + message, file=sys.stderr)

def read_camera_config(config):
    """
    Read single camera configuration.
    """
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("{}: could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

def read_config():
    """
    Read configuration file.
    """
    global team
    global server

    # parse file
    try:
        with open(config_file, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open {}: {}".format(config_file, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    team = 1418

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not read_camera_config(camera):
            return False

    return True


def start_camera(config):
    """
    Begin running the camera.
    """
    print("Starting {} on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
    # read configuration
    if not read_config():
        sys.exit(1)

    # start NetworkTables and create table instance
    ntinst = NetworkTablesInstance.getDefault()
    table = NetworkTables.getTable("vision")
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client")
        ntinst.startClientTeam(team)

    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = start_camera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    # Get the first camera
    camera_server = streams[0]
    # Get a CvSink. This will capture images from the camera
    cv_sink = camera_server.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    output_stream = camera_server.putVideo("stream", IMAGE_WIDTH, IMAGE_HEIGHT)
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # loop forever
    while True:

        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        # TODO: Why can't we just use frame for everything?
        timestamp, img = cv_sink.grabFrame(img)
        frame = img
        if timestamp == 0:
            # Send the output the error.
            output_stream.notifyError(cv_sink.getError());
            # Skip the rest of the current iteration
            continue

        mask = threshold_frame(frame)
        contours = find_contours(mask)
        print("Found %d contours initially." % len(contours))
        processed_frame = find_targets(contours, frame)
        # (optional) send image back to the dashboard
        output_stream.putFrame(processed_frame)
