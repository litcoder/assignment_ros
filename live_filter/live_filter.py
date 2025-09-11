import cv2
import numpy as np

# TODO: Image kernels
kernels = {
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TODO: Get kernel(filter)'s name
    filter_name = "TODO: filter name"
    # TODO: Get kernel data and apply it to the frame using filter2D()
    filtered = frame

    # Overlay text
    cv2.putText(filtered, f"Filter: {filter_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera Filter", filtered)

    # Key control
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 82:  # UP arrow
        # TODO: apply the previous filter
        pass
    elif key == 84:  # DOWN arrow
        # TODO: apply the next filter
        pass


cap.release()
cv2.destroyAllWindows()
