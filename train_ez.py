try:
    import os
    import xml.etree.ElementTree as ET
    import tensorflow as tf
    from keras_cv import visualization, bounding_box
    from tqdm.auto import tqdm
    import cv2
    import numpy as np
    import random
    import os
    from pathlib import Path
    from PIL import Image, ImageDraw
except ImportError as e:
    raise ImportError(f"Required library not found: {e.name}. Please install the dependencies.")


def load_data_from_dir(path_images: str, path_annot: str, class_mapping: dict):
    """
    Load data from a directory containing image files and corresponding XML annotations.
    :param path_images: Path to the directory containing images.
    :param path_annot: Path to the directory containing XML annotations.
    :param class_mapping: A mapping from class names to class IDs.
    :return: A tuple containing lists of image paths, bounding boxes, and class IDs.
    """
    # Validate input paths
    if not os.path.exists(path_images):
        raise ValueError(f"Image path does not exist: {path_images}")
    if not os.path.exists(path_annot):
        raise ValueError(f"Annotation path does not exist: {path_annot}")

    xml_files = sorted([
        os.path.join(path_annot, file_name) for file_name in os.listdir(path_annot)
        if file_name.endswith('.xml')
    ])

    image_paths, bbox, classes = [], [], []
    for xml_file in xml_files:
        # Parse the annotations for each xml file and get the image path, bounding boxes, and class IDs
        image_path, boxes, class_ids = _parse_annotation(xml_file, path_images, class_mapping)
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)
    return image_paths, bbox, classes


def _parse_annotation(xml_file: Path | str, path_images: Path | str, class_mapping: dict):
    """
    (Private Function) Parse an XML file containing annotations for an image.
    :param xml_file: Path to the XML file.
    :param path_images: Path to the directory containing images.
    :param class_mapping: Dict mapping of class names to class IDs.
    :return: A tuple containing the image path, list of bounding boxes, and list of class IDs.
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


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    """
    Load a dataset by combining images and bounding boxes in the proper format.
    :param image_path: Path to the image file.
    :param classes: List of class IDs for the bounding boxes.
    :param bbox: List of bounding boxes.
    :return: A dictionary with keys 'images' and 'bounding_boxes' containing the respective data.
    """
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    """
    Visualize a dataset containing images and their associated bounding boxes.
    :param inputs: An iterable containing the dataset elements.
    :param value_range: The range of values for the images (e.g., (0, 255)).
    :param rows: Number of rows in the visualization grid.
    :param cols: Number of columns in the visualization grid.
    :param bounding_box_format: The format of bounding boxes.
    """
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]

    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        # TODO: Remove hard code
        class_mapping={0: 'palm'},
    )


def replace_white_background(image, background_images):
    """
    Remove and replace white elements of image.
    :param image: Path to image to be transformed.
    :param background_images: Path to random backgrounds.
    :return: Image with changed background (previously white).
    """
    # Define the white threshold
    white_threshold = [210, 210, 210]
    lower_white = np.array(white_threshold, dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower_white, upper_white)
    mask_inv = cv2.bitwise_not(mask)

    for _ in range(len(background_images)):
        bg_image_path = random.choice(background_images)
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            print(f"Failed to load background image: {bg_image_path}")
            continue

        # Resize the background image to match the input image size
        bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))

        # Extract the region of interest from the background based on the mask
        bg = cv2.bitwise_and(bg_image, bg_image, mask=mask)

        # Extract the region of interest from the original image
        foreground = cv2.bitwise_and(image, image, mask=mask_inv)
        combined_image = cv2.add(foreground, bg)
        return combined_image

    return None


def visualize_detections(model, dataset, bounding_box_format):
    """
    Visualize a dataset using a model as a detector and using the saved dataset annotations.
    :param model: Model to detect objects in image.
    :param dataset: Tensor dataset to run model on.
    :param bounding_box_format: The format of bounding boxes.
    """
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        # TODO: Remove hard code
        class_mapping={0: 'palm'},
    )
