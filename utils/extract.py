from pathlib import Path
import numpy as np
from enum import Enum
import docx2txt
from utils.labels import doc_name_to_moid
from utils.parse import parse_annotation_pdf
import cv2
from typing	import *
import matplotlib.pyplot as plt


def transform_coordinates_from_document_to_image(
        coords: List[List[float]], 
        true_height: int, 
        true_width: int, 
        coord_height: int, 
        coord_width: int) -> List[List[float]]:
    """
    Transform coordinates from PDF to image coordinates.
    """
    # Scale the coordinates
    coords = [[x * true_width / coord_width, y * true_height / coord_height] for x, y in coords]

    # Flip the coordinates
    # coords = [[x, true_height - y] for x, y in coords]

    return coords


class Mode(Enum):
    PDF = 1
    WORD = 2


class Extractor:

    def __init__(self, document: Path,  template: np.ndarray, file_directory: Path) -> None:
        
        suffix              = document.suffix
        self.moid           = doc_name_to_moid(document, suffix)
        self.file_directory = file_directory
        self.document       = document
        self.template       = template

        if suffix == ".pdf":
            self.mode = Mode.PDF
        elif suffix == ".docx":
            self.mode = Mode.WORD
        else:
            raise ValueError(f"Unknown document suffix: {suffix}")
    
    def image_from_word(self, path: Path) -> np.ndarray:
        """
        Extract image from a Word document.

        Args:
            path (Path): Path to the Word document.

        Returns:
            np.ndarray: Image of the VFA analysis with a DXA scan.
        """
        
        if self.file_directory.exists():
            for f in self.file_directory.glob("*"):
                f.unlink()
        else:
            self.file_directory.mkdir()

        docx2txt.process(path, self.file_directory)

        n_files = len(list(self.file_directory.glob("*")))

        if n_files == 1:
            image_path = list(self.file_directory.glob("*"))[0]
        elif n_files > 1:
            image_path = self.file_directory / "image1.png"
        else:
            raise ValueError(f"No images found in {self.file_directory}")
            
        image =  cv2.imread(str(image_path), cv2.IMREAD_ANYCOLOR)

        return image

    def vertebrae(self, size_factor: float = 0.9, threshold: float = 0.8, degree=3, residual_threshold=11, method = cv2.TM_CCOEFF_NORMED, plot = False) -> np.ndarray:
        """
        Extract vertebrae from a Word document or a PDF.
        
        
        Args:
            size_factor (float, optional): Size factor for the template. Defaults to 0.9.
            threshold (float, optional): Threshold for the matching. Defaults to 0.8.
            method ([type], optional): Matching method. Defaults to cv2.TM_CCOEFF_NORMED.
        
        Returns:
            np.ndarray: Vertebrae coordinates of the shape (n, 6, 2)
        """

        if self.mode == Mode.WORD:

            self.image = self.image_from_word(self.document)
            roi, template = self.match_image_with_dxa(size_factor=size_factor, method=method)
            points, mask = self.points_from_image(roi, threshold=threshold, method=method)
            vertebrae = self.vertebrae_from_points(points, degree=degree, residual_threshold=residual_threshold, plot=plot)
            height, width = roi.shape[:2]
            names = None
            
        elif self.mode == Mode.PDF:

            points, width, height, names = parse_annotation_pdf(self.document)

            # Switch x and y coordinates
            points = points[:, ::-1]

            vertebrae = self.vertebrae_from_points(points, degree=degree, residual_threshold=residual_threshold)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Get DXA image size
        true_width, true_height = self.template.shape[::-1]
        vs = []
        for v in vertebrae:
            coordinates = transform_coordinates_from_document_to_image(v, true_height, true_width, height, width)
            vs.append(coordinates)

        vertebrae = np.array(vs)
        return vertebrae, names



    def match_image_with_dxa(self, size_factor=0.9, method=cv2.TM_CCOEFF_NORMED) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match an image from a Word document with a DXA scan.

        Args:
            size_factor (float, optional): Size factor for the template. Defaults to 0.9.
            method ([type], optional): Matching method. Defaults to cv2.TM_CCOEFF_NORMED.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of the extracted region of interest and the template
        """

        # Convert to grayscale
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        image_height, image_width = image.shape
        template_height, template_width = self.template.shape

        scale = (image_height / template_height)*size_factor
        new_width = int(template_width * scale)
        new_height = int(template_height * scale)
        template = cv2.resize(self.template, (new_width, new_height))

        # Match template on image
        result = cv2.matchTemplate(image, template, method)

        # Get the location of the best match
        min_val, max_val, min_loc, top_left = cv2.minMaxLoc(result)
        bottom_right = (top_left[0] + new_width, top_left[1] + new_height)

        # Extract the region of interest
        roi = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        return roi, template
    
    def points_from_image(self, image: np.ndarray, threshold: float = 0.8, method=cv2.TM_CCOEFF_NORMED) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts colored points with the shape of a cross from an image.
        
        Args:
            image (np.ndarray): Image to extract points from
            threshold (float, optional): Threshold for the match. Defaults to 0.8.
            method ([type], optional): Matching method. Defaults to cv2.TM_CCOEFF_NORMED.
        
        Raises:
            ValueError: If the number of points is not a multiple of 6

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of points and mask
        """

        # Convert to HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((image[:,:,0] > 0) | (image[:,:,1] > 0)).astype(np.uint8)


        # Match with a cross template
        cross = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=np.uint8)

        result = cv2.matchTemplate(mask,cross, method)

        locations = np.where( result >= threshold )
        points = np.array(list(zip(*locations[::-1])))

        if len(points) % 6 != 0:
            raise ValueError("Number of points is not a multiple of 6")

        return points, mask
    
    def vertebrae_from_points(self, points: np.ndarray, degree=3, residual_threshold=15, plot=False) -> np.ndarray:
        """
        Creates a grouped list of points for each vertebrae, given an array of points.

        Args:
            points (np.ndarray): Array of points of shape (n, 2) where n is a multiple of 6 and each row is a point (x, y)
            degree (int, optional): Degree of the polynomial to fit. Defaults to 3.
            residual_threshold (int, optional): Threshold for the residuals. Defaults to 15.
        """
        if len(points) == 6:
            return points.reshape(1, 6, 2)
        
        if len(points) <= 18:
            degree = 3

        # Fit a third degree polynomial to the points
        z = np.polyfit(points[:,1], points[:,0], degree)
        f = np.poly1d(z)

        # Calculate the residuals for all points to the curve
        residuals = points[:,0] - f(points[:,1])

        # Get point indices with a residual larger than 0.1
        r = residual_threshold if residual_threshold else np.floor(points[:,0].std() / 2)
        right   = residuals > r
        left    = residuals < -r
        middle  = np.logical_and(residuals < r, residuals > -r)


        right_points    = points[right]
        left_points     = points[left]
        middle_points   = points[middle]

        # Sort right_points by x
        right_points = right_points[np.argsort(right_points[:,1])]
        left_points = left_points[np.argsort(left_points[:,1])]
        middle_points = middle_points[np.argsort(middle_points[:,1])]

        # Chunk the points into groups of 2
        right_points = np.array_split(right_points, len(right_points) / 2)
        left_points = np.array_split(left_points, len(left_points) / 2)
        middle_points = np.array_split(middle_points, len(middle_points) / 2)

        if plot:
            x_new = np.linspace(points[:,1].min(), points[:,1].max(), 50)
            y_new = f(x_new)
            f, ax = plt.subplots(2,1, figsize=(10,10))
            ax[0].plot(points[left,1], residuals[left], 'o')
            ax[0].plot(points[right,1], residuals[right], 'o')
            ax[0].plot(points[middle,1], residuals[middle], 'o')
            ax[1].plot(points[:,1], points[:,0], '.', x_new, y_new, '-', c='k')
            plt.show()

        if not (right.sum() == left.sum() == middle.sum()):

            raise ValueError(f"Residuals not grouped correctly: {right.sum()}, {left.sum()}, {middle.sum()}")


        vertebrae = np.array([
            np.concatenate((l, m, r)).tolist() for r, l, m in zip(right_points, left_points, middle_points)
            ])

        return vertebrae

