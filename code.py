import rospy
import math
from clover import srv
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock

bridge = CvBridge()
telem_lock = Lock()

rospy.init_node('flight')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

recorded_objects = {}
position = 1
f = open('nea.txt', 'w')

# Предположим, что у нас есть информация о реальном размере объекта
real_object_width = 0.2  # в метрах

def get_telemetry_locked(frame_id='aruco_map'):
    with telem_lock:
        return get_telemetry(frame_id=frame_id)

def goto(x=0, y=0, z=2.5, yaw=float('nan'), speed=0.5, frame_id='aruco_map', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    while not rospy.is_shutdown():
        telem = get_telemetry_locked(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

def detect_color(hsv_image):
    color_ranges = {
        'Yellow': [(25, 70, 120), (35, 255, 255)],
        "Red": [(0, 70, 50), (10, 255, 255), (172, 70, 50), (180, 255, 255)],
        "Blue": [(100, 150, 0), (140, 255, 255)],
        "Green": [(40, 40, 40), (70, 255, 255)],
        'Pink': [(140, 50, 50), (170, 255, 255)],
        'Purple': [(130, 50, 50), (145, 255, 255)]
    }

    detected_colors = []
    for color, ranges in color_ranges.items():
        mask = cv.inRange(hsv_image, *ranges[:2])
        if len(ranges) > 2:
            mask = cv.bitwise_or(mask, cv.inRange(hsv_image, *ranges[2:]))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            detected_colors.append((color, contours))
    return detected_colors

def image_callback(data):
    global position
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)

    detected_colors = detect_color(hsv_image)

    for detected_color, contours in detected_colors:
        for contour in contours:
            if cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                
                # Рисуем контур объекта белым цветом
                cv.drawContours(cv_image, [contour], -1, (255, 255, 255), 2)
                
                # Рисуем прямоугольник вокруг объекта
                cv.rectangle(cv_image, (x, y), (x + w, y + h), color_to_bgr(detected_color), 2)
                
                # Вычисляем и рисуем геометрический центр
                M = cv.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv.circle(cv_image, (cx, cy), 5, (0, 0, 0), -1)  # Черная точка

                    # Используем известный размер объекта для расчета координат
                    focal_length = 500  # Примерное значение, нужно уточнить для вашей камеры
                    distance = (real_object_width * focal_length) / w
                    pos = get_telemetry_locked(frame_id='aruco_map')

                    # Вычисляем координаты объекта
                    object_x = pos.x + (cx - cv_image.shape[1] // 2) * distance / focal_length
                    object_y = pos.y + (cy - cv_image.shape[0] // 2) * distance / focal_length

                    # Выводим координаты объекта на изображение
                    coords_text = f'({object_x:.2f}, {object_y:.2f})'
                    cv.putText(cv_image, coords_text, (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_to_bgr(detected_color), 2)

                    if detected_color not in recorded_objects:
                        log_position(detected_color, object_x, object_y)

    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

def set_effect_based_on_color(color):
    color_effects = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Pink': (255, 192, 203),
        'Yellow': (255, 255, 0),
        'Purple': (128, 0, 128)
    }
    if color in color_effects:
        set_effect(r=color_effects[color][0], g=color_effects[color][1], b=color_effects[color][2])

def color_to_bgr(color):
    color_bgr = {
        'Red': (0, 0, 255),
        'Green': (0, 255, 0),
        'Blue': (255, 0, 0),
        'Pink': (203, 192, 255),
        'Yellow': (0, 255, 255),
        'Purple': (128, 0, 128)
    }
    return color_bgr.get(color, (255, 255, 255))

def is_recorded(x, y, threshold=0.5):
    for obj in recorded_objects.values():
        if math.sqrt((obj[0] - x) ** 2 + (obj[1] - y) ** 2) < threshold:
            return True
    return False

def log_position(color, x, y):
    global position
    avg_x = round(x, 1)
    avg_y = round(y, 1)
    print(f'object {position}: {color} {avg_x} {avg_y} (aruco_map)')
    f.write(f'object {position}: {color} {avg_x} {avg_y} (aruco_map)\n')
    recorded_objects[color] = (avg_x, avg_y)
    position += 1

image_pub = rospy.Publisher('Rusin_Egor_debug', Image, queue_size=1)
image_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback)

goto(z=1, frame_id='body', auto_arm=True)
goto()

for marker in ['aruco_0', 'aruco_1', 'aruco_21', 'aruco_22', 'aruco_2', 'aruco_3', 'aruco_23']:
    goto(frame_id=marker)

f.close()
goto()
land()
