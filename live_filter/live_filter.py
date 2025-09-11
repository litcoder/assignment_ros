import cv2
import numpy as np


class Filters:
    # TODO: Image kernels
    kernels = {
    }

    def __init__(self):
        # TODO: Implement internal variables
        pass

    def apply_filter(self, frame, filter_name) -> np.array:
        # TODO: Apply the filter to the frame and return
        return frame

    def get_current_filter_name(self) -> str:
        # TODO: Return currently set kernels's name
        return

    def switch_next_filter(self):
        # TODO: Update currently selected kernel to the next
        pass

    def switch_previous_filter(self):
        # TODO: Update currently selected kernel to the previous
        pass


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Filters
    filters = Filters()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current filter's name
        filter_name = filters.get_current_filter_name()

        # Apply the filter.
        filtered_frame = filters.apply_filter(frame, filter_name)

        # Overlay text
        cv2.putText(filtered_frame, f"Filter: {filter_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Camera Filter", filtered_frame)

        # Key control
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 82:  # UP arrow
            filters.switch_previous_filter()
        elif key == 84:  # DOWN arrow
            filters.switch_next_filter()

    cap.release()
    cv2.destroyAllWindows()
