import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import time
import warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
from pathlib import Path


def dir_img_files(path_dir, img_extensions=["png", "jpg", "jpeg"]):
    dir_files = os.listdir(path_dir)

    for file_path in dir_files[:]:  # dir_files[:] makes a copy of dir_files.
        if file_path.split(".")[-1] not in img_extensions:
            dir_files.remove(file_path)
    # print(dir_files)
    return dir_files


def display_image(image, name):

    cv2.imshow("Image", image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()


def save_image(image, dir_out, image_path):

    path_out = str(dir_out / image_path.stem) + ".png"
    cv2.imwrite(path_out, image)
    print("\nSaved image to ", path_out)


def read_image(image_path, metadata=False):

    image, width, height = None, -1, -1
    image = cv2.imread(image_path)

    if metadata:
        height, width = image.shape[:2]

    return image, width, height


def load_image_into_numpy_array(path):

    return np.array(Image.open(path))


def load_detection_model(path_to_model_dir, path_to_labels):

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
    tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # LOAD THE MODEL

    PATH_TO_SAVED_MODEL = path_to_model_dir / "saved_model"

    print("Loading model...", end="")
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    tf_model = tf.saved_model.load(str(PATH_TO_SAVED_MODEL))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(round(elapsed_time, 3)))

    # LOAD LABEL MAP DATA FOR PLOTTING

    category_index = label_map_util.create_category_index_from_labelmap(
        str(path_to_labels), use_display_name=True
    )

    warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings

    return tf_model, category_index


def read_img_to_tensor(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Image read error. Check image path!")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    image_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    image_tensor = image_tensor[tf.newaxis, ...]

    return image_tensor, image


def visualize_detections(image, detections, min_conf, category_index, dir_out, img_path, save=True):

    img_h, img_w, _ = image.shape

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    # print(detections)

    scores = detections["detection_scores"]
    boxes = detections["detection_boxes"]
    classes = detections["detection_classes"]
    count = 0
    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            # increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * img_h)))
            xmin = int(max(1, (boxes[i][1] * img_w)))
            ymax = int(min(img_h, (boxes[i][2] * img_h)))
            xmax = int(min(img_w, (boxes[i][3] * img_w)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            # Draw label
            object_name = category_index[int(classes[i])][
                "name"
            ]  # Look up object name from "labels" array using class index

            label = "%d%%:%s" % (int(scores[i] * 100), object_name)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                image, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )  # Draw label text

    cv2.putText(
        image,
        "Number of kept detections : " + str(count),
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (10, 255, 0),
        1,
        cv2.LINE_AA,
    )

    if save:
        save_image(image, dir_out, img_path)


def main():

    PATH_TO_MODEL_DIR = Path(
        "training/exported_models/ear_detection_ssd_mobilenet_v2_fpnlite_model"
    )
    PATH_TO_LABELS = Path(
        "training/exported_models/ear_detection_ssd_mobilenet_v2_fpnlite_model/saved_model/label_map.pbtxt"
    )
    MIN_CONF_THRESH = float(0.25)
    DIR_IMAGES_TEST = Path("test/images")
    DIR_IMAGES_OUT = Path("test/output")

    list_dir_imgs = dir_img_files(DIR_IMAGES_TEST)

    detection_model, category_index = load_detection_model(PATH_TO_MODEL_DIR, PATH_TO_LABELS)

    for img_name in list_dir_imgs:

        print("Running detections on {}... ".format(img_name), end="")
        img_path = DIR_IMAGES_TEST / img_name

        image_tensor, image = read_img_to_tensor(str(img_path))

        # image_tensor = np.expand_dims(image_np, 0)
        detections = detection_model(image_tensor)

        visualize_detections(
            image, detections, MIN_CONF_THRESH, category_index, DIR_IMAGES_OUT, img_path,
        )


if __name__ == "__main__":
    main()
