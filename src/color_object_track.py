import cv2
import sys
import json
import numpy as np
from functools import partial
from datetime import datetime


WINDOW_NAME = "LAB Filter"
l_min = 0
a_min = 0
b_min = 0
l_max = 255
a_max = 255
b_max = 255


def update_color_value(x, color, is_min):
    global l_min, a_min, b_min, l_max, a_max, b_max
    match color:
        case "L":
            if is_min:
                l_min = x
            else:
                l_max = x
        case "A":
            if is_min:
                a_min = x
            else:
                a_max = x
        case "B":
            if is_min:
                b_min = x
            else:
                b_max = x
        case _:
            pass


# Trackbar UI
cv2.namedWindow(WINDOW_NAME)
cv2.resizeWindow(WINDOW_NAME, 800, 200)
cv2.createTrackbar("L Min", WINDOW_NAME, l_min, 255,
                   partial(update_color_value, color="H", is_min=True))
cv2.createTrackbar("L Max", WINDOW_NAME, l_max, 255,
                   partial(update_color_value, color="L", is_min=False))
cv2.createTrackbar("A Min", WINDOW_NAME, a_min, 255,
                   partial(update_color_value, color="A", is_min=True))
cv2.createTrackbar("A Max", WINDOW_NAME, a_max, 255,
                   partial(update_color_value, color="A", is_min=False))
cv2.createTrackbar("B Min", WINDOW_NAME, b_min, 255,
                   partial(update_color_value, color="B", is_min=True))
cv2.createTrackbar("B Max", WINDOW_NAME, b_max, 255,
                   partial(update_color_value, color="B", is_min=False))


# Load if config file is given
if len(sys.argv) > 1:

    with open(sys.argv[1], "r") as f1:
        config = json.load(f1)
        l_min = config.get("l_min", l_min)
        l_max = config.get("l_max", l_max)
        a_min = config.get("a_min", a_min)
        a_max = config.get("a_max", a_max)
        b_min = config.get("b_min", b_min)
        b_max = config.get("b_max", b_max)

    cv2.setTrackbarPos("L Min", WINDOW_NAME, l_min)
    cv2.setTrackbarPos("L Max", WINDOW_NAME, l_max)
    cv2.setTrackbarPos("A Min", WINDOW_NAME, a_min)
    cv2.setTrackbarPos("A Max", WINDOW_NAME, a_max)
    cv2.setTrackbarPos("B Min", WINDOW_NAME, b_min)
    cv2.setTrackbarPos("B Max", WINDOW_NAME, b_max)

    print(f"Loaded config:")
    print(f"L : {l_min} - {l_max}")
    print(f"A : {a_min} - {a_max}")
    print(f"B : {b_min} - {b_max}")

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if ret is False:
        break

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Filter in BGR space (OpenCV uses BGR)
    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])
    mask = cv2.inRange(lab, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"Rect: ({x} {y} {w} {h})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show filtered result
    combined = np.hstack((img, result))
    cv2.imshow("LAB Filter Result", cv2.resize(combined, (1280, 600)))

    # Exit on ESC key press
    key = cv2.waitKey(1) & 0xff
    match key:
        case 27:
            break
        case 115: # 's' key
            # Save current filter settings
            save_data = {
                "l_min": l_min, "l_max": l_max,
                "a_min": a_min, "a_max": a_max,
                "b_min": b_min, "b_max": b_max
            }
            timestampestr = f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
            with open("LAB-cal.json", "w") as f1, open(f"LAB-cal-{timestampestr}.json", "w") as f2:
                json.dump(save_data, f1, indent=4)
                json.dump(save_data, f2, indent=4)
                print(f"File saved to LAB-cal.json and LAB-cal-{timestampestr}.json")
        case _:
            pass
cap.release()
cv2.destroyAllWindows()
