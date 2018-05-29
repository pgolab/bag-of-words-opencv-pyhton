import cv2

from config import DRAW_DETECTED_KEYPOINTS


def get_image_descriptors(image_path, sift):
    image = cv2.imread(image_path.as_posix())
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray_image, None)

    if DRAW_DETECTED_KEYPOINTS:
        cv2.drawKeypoints(image, key_points, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('processing image', image)
        cv2.waitKey(0)

    return descriptors
