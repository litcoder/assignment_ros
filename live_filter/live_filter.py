import cv2
import numpy as np


class Filters:
    # TODO: Image kernels
    Kernels = {
        "Original": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        "Gausian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Sobel (X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Sobel (Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "Edge detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    }

    def __init__(self, kernels=Kernels):
        self.kernels = kernels

        # TODO: Implement internal variables
        self.filter_idx = 0
        self.filter_names = list(self.kernels.items())

    def apply_filter(self, frame, filter_name) -> np.array:
        # TODO: Apply the filter to the frame and return
        filter = self.kernels[filter_name][1:]
        return cv2.filter2D(frame, -1, filter)

    def get_current_filter_name(self) -> str:
        # TODO: Return currently set kernels's name
        return self.filter_names[self.filter_idx][0]

    def switch_next_filter(self):
        # TODO: Update currently selected kernel to the next
        self.filter_idx = abs(self.filter_idx + 1 % (len(self.kernels))) - 1

    def switch_previous_filter(self):
        # TODO: Update currently selected kernel to the previous
        self.filter_idx = self.filter_idx - 1 % (len(self.kernels))


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
