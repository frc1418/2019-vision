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
diagonal_fov = math.radians(68.5)

# 16:9 aspect ratio
horizontal_aspect = 16
vertical_aspect = 9

image_width = 480
image_height = 270

# Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonal_aspect = math.hypot(horizontal_aspect, vertical_aspect)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontal_view = math.atan(math.tan(diagonal_fov/2) * (horizontal_aspect / diagonal_aspect)) * 2
vertical_view = math.atan(math.tan(diagonal_fov/2) * (vertical_aspect / diagonal_aspect)) * 2

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = image_width / (2*math.tan((horizontal_view/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((vertical_view/2)))

def threshold_video(frame):
    """
    Calculate masked frame based on thresholding input video.
    """
    img = frame.copy()
    blur = cv2.medianBlur(img, 5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # define HSV range to extract bright green features
    lower_color = np.array([50, 150,160])
    upper_color = np.array([100, 255, 255])
    # extract qualifying pixels from image
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask


def find_contours(frame, mask):
    """
    Find contours of image mask, displaying them over original stream.
    """
    # Calculate contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    print("Found %d contours initially." % len(contours))
    # Get frame resolution
    screenHeight, screenWidth, _ = frame.shape
    # Calculate center of screen
    center_x = (screenWidth / 2) - .5
    center_y = (screenHeight / 2) - .5
    # Copy frame to image
    image = frame.copy()
    # Processes contours
    if len(contours) != 0:
        image = find_targets(contours, image, center_x, center_y)
    # Return image of contours overlayed on original video
    return image


def find_targets(contours, image, center_x, center_y):
    """
    Draw contours, calculating target angle.

    :param contours: list of contours among which to find targets.
    :param image: image upon which to draw contours.
    :param center_x: x coordinate of image center.
    :param center_y: y coordinate of image center.
    """
    screenHeight, screenWidth, _ = image.shape;
    # List for storing found targets
    targets = []

    if len(contours) >= 2:
        # Sort contours in descending order by size
        contours.sort(key=lambda contour: cv2.contourArea(contour), reverse=True)

        largest_contours = []
        for contour in contours:
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(contour)
            # Calculate areas of contour and hull
            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull)
            # Get moments of contour for centroid calculations
            moments = cv2.moments(contour)
            # Find centeroids of contour
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = 0, 0

            ### CALCULATE CONTOUR ROTATION BY FITTING ELLIPSE ###
            rotation = getEllipseRotation(image, contour)

            ### DRAW CONTOUR ###
            # Get rotated bounding rectangle of contour
            rect = cv2.minAreaRect(contour)
            # Create box around that rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw white circle at center of contour
            cv2.circle(image, (cx, cy), 6, (255, 255, 255))

            # Draw contours
            cv2.drawContours(image, [contour], 0, (23, 184, 80), 1)

            # Get coordinates and radius of contour's enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Append important info to array
            largest_contours.append([cx, cy, rotation, contour])

        # Sort array based on coordinates (left to right) to make sure contours are adjacent
        largest_contours.sort(key=lambda contour: contour[0])

        # Find targets from contours
        for i in range(len(largest_contours) - 1):
            # Check rotation of adjacent contours
            tilt_left = largest_contours[i][2]
            tilt_right = largest_contours[i + 1][2]

            # Contour coordinates
            cx_left = largest_contours[i][0]
            cx_right = largest_contours[i + 1][0]
            cy_left = largest_contours[i][1]
            cy_right = largest_contours[i + 1][1]

            # If contour angles are opposite
            # Negative tilt -> Rotated to the right
            # NOTE: if using rotated rect (min area rectangle), negative tilt means rotated to left
            # If left contour rotation is tilted to the left then skip iteration
            # If right contour rotation is tilted to the right then skip iteration
            if (np.sign(tilt_left) != np.sign(tilt_right) and
                    not (tilt_left > 0 and cx_left < cx_right or tilt_right > 0 and cx_right < cx_left)):

                target_center = (cx_left + cx_right) / 2
                # Angle from center of camera to target (what you should pass into gyro)
                target_yaw = calculate_yaw(target_center, center_x, H_FOCAL_LENGTH)

                # Push to NetworkTable
                table.putNumber("target_yaw", target_yaw)

                # Make sure no duplicates, then append
                targets.append([target_center, target_yaw])
    # Check if there are targets seen
    if len(targets) > 0:
        table.putBoolean("target_seen", True)
        # Sort targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        final_target = min(targets, key=lambda x: math.fabs(x[1]))
        # Draw yaw of target + line where center of target is
        cv2.putText(image, "Yaw: " + str(final_target[1]), (1, 8), cv2.FONT_HERSHEY_PLAIN, .6, (255, 255, 255))
        cv2.line(image, (final_target[0], screenHeight), (final_target[0], 0), (255, 0, 0), 2)

        current_angle_error = final_target[1]

        table.putNumber("current_angle_error", current_angle_error)

    cv2.line(image, (round(center_x), screenHeight), (round(center_x), 0), (255, 255, 255), 2)

    return image


# Forgot how exactly it works, but it works!
def translate_rotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculate_distance(camera_height, target_height, pitch):
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
    pitch = math.degrees(math.atan((pixel_y - center_y) / V_FOCAL_LENGTH))
    # Just stopped working have to do this:
    pitch *= -1
    return pitch

def getEllipseRotation(image, contour):
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
    print("Starting camera '{}' on {}".format(config.name, config.path))
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
    #Get the first camera
    cameraServer = streams[0]
    # Get a CvSink. This will capture images from the camera
    cvSink = cameraServer.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cameraServer.putVideo("stream", image_width, image_height)
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

    # loop forever
    while True:
        table.putBoolean("target_seen", False)
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        timestamp, img = cvSink.grabFrame(img)
        frame = img
        if timestamp == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError());
            # skip the rest of the current iteration
            continue

        threshold = threshold_video(frame)
        processed = find_contours(frame, threshold)
        # (optional) send some image back to the dashboard
        outputStream.putFrame(processed)
