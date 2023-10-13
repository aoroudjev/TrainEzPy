import os
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box, visualization

# Constants
SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
IMAGE_RESIZE = (640, 640)


def load_data_from_dir(path_images: str, path_annot: str, class_mapping: dict):
    """
    Load data in pascal voc format from designated directories.
    Args:
        path_images (str): The path to the directory containing the images.
        path_annot (str): The path to the directory containing the xml annotations.
        class_mapping (dict): The mapping of class names to class IDs.
    Returns:
        tuple: A tuple containing the image paths, bounding boxes, and class IDs.
    """
    # Get a list of all xml files in the annotation directory
    xml_files = sorted([
        os.path.join(path_annot, file_name) for file_name in os.listdir(path_annot)
        if file_name.endswith('.xml')
    ])
    image_paths, bbox, classes = [], [], []
    for xml_file in tqdm(xml_files):
        # Parse the annotations for each xml file and get the image path, bounding boxes, and class IDs
        image_path, boxes, class_ids = parse_annotation(xml_file, path_images, class_mapping)
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)
    return image_paths, bbox, classes


def parse_annotation(xml_file: str, path_images: str, class_mapping: dict):
    """
    Parse the annotations for a given xml file.
    :param xml_file: Path to the xml file
    :param path_images: Path to the directory containing the images
    :param class_mapping: Mapping of class names to class IDs
    :return: Lists of image paths, bounding boxes, and class IDs
    """

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the image name and path
    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    # Collect the bounding boxes and class names
    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    # Map the class names to class IDs
    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]

    return image_path, boxes, class_ids


def load_image(image_path: str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def main():
    class_ids = ['palm']

    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    path_images = "./data/palm/"
    path_annot = "./data/bboxes/"

    images, bboxes, classes = load_data_from_dir(path_images, path_annot, class_mapping)
    images = tf.ragged.constant(images)
    bboxes = tf.ragged.constant(bboxes)
    classes = tf.ragged.constant(classes)

    data = tf.data.Dataset.from_tensor_slices((images, classes, bboxes))
    num_val = int(len(data) * SPLIT_RATIO)
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    bboxes = keras_cv.bounding_box.convert_format(
        bboxes, images=images, target="xywh", source="xyxy"
    )




    # train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == "__main__":
    main()
