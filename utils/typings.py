from dataclasses import dataclass
from typing import *
import numpy as np
from pathlib import Path
from PIL import Image
import json
# from utils.structure import transform_coordinates_from_pdf_to_image
from utils.extract import resize_coordinates
from models.augmentations import Preprocess
from torchvision.transforms.functional import crop
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision.ops import box_convert, clip_boxes_to_image
import matplotlib.pyplot as plt
import torch

THORACIC = [
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4"
]

LUMBAR = [
    "L4", "L3", "L2", "L1"
]

VERTEBRA_NAMES = [
    *LUMBAR,
    *THORACIC
]

@dataclass
class DXA:
    """
    A class representing a lateral/sagittal spine image.
    """

    path: Path

    @property
    def image(self) -> Image:
        return Image.open(self.path)
    
    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.image)
    
    def crop(self, bbox: "Bbox") -> np.ndarray:
        return crop(self.image, bbox.y, bbox.x, bbox.height, bbox.width)

    
@dataclass
class CT:
    """
    A class representing a CT image.
    """

    path: Path

    @property
    def image(self) -> Image:
        return Image.open(self.path)


@dataclass
class Point:
    """
    A class representing a point.
    """

    x: float
    y: float
    visible: int = 1

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.visible])

@dataclass
class Annotation:
    points: List[Point]
    image_height: Optional[float] = None
    image_width: Optional[float] = None

    def to_numpy(self) -> np.ndarray:
        return np.array([p.to_numpy() for p in self.points])
    
    def to_bbox(self, image_height: float, image_width: float) -> "Bbox":
        return Bbox.from_annotation(self, image_height, image_width)
    
    @property
    def posterior(self) -> List[Point]:
        return self.points[0:2]
    
    @property
    def middle(self) -> List[Point]:
        return self.points[2:4]
    
    @property
    def anterior(self) -> List[Point]:
        return self.points[4:6]
    
    @property
    def centroid(self) -> Point:
        return Point(np.mean([p.x for p in self.points]), np.mean([p.y for p in self.points]))
    
    def set_size(self, image_height: float, image_width: float) -> "Annotation":
        self.image_height = image_height
        self.image_width = image_width

        return self
    
    def resize(self, new_img_height: float, new_img_width: float):
        return Annotation([Point(p.x * new_img_width / self.image_width, p.y * new_img_height / self.image_height, p.visible) for p in self.points], new_img_height, new_img_width)

@dataclass
class Bbox:
    """
    A class representing a bounding box, with x, y, width and height. 
    Coordinates are in pixels, relative to the original image size.
    """

    x: float
    y: float
    width: float
    height: float
    image_height: Optional[float] = None
    image_width: Optional[float] = None

    @staticmethod
    def from_annotation(annotation: Annotation, image_height: float, image_width: float) -> "Bbox":
        points  = annotation.points

        x       = min([p.x for p in points])
        y       = min([p.y for p in points])
        width   = max([p.x for p in points]) - x
        height  = max([p.y for p in points]) - y

        return Bbox(x, y, width, height, image_height, image_width)
    
    def to_normalized(self, height: int, width: int) -> "NormalizedBbox":
        x = self.x / width
        y = self.y / height
        width = self.width / width
        height = self.height / height

        return NormalizedBbox(x, y, width, height, self.image_height, self.image_width)
    
    def to_expanded(self, expand: float = 0.2) -> "ExpandedBbox":
        x = self.x - expand * self.width
        y = self.y - expand * self.height
        width = self.width + 2 * expand * self.width
        height = self.height + 2 * expand * self.height

        return ExpandedBbox(x, y, width, height, self.image_height, self.image_width)
    
    def resize(self, new_img_height: float, new_img_width: float):
        
        x = self.x * new_img_width / self.image_width
        y = self.y * new_img_height / self.image_height
        width = self.width * new_img_width / self.image_width
        height = self.height * new_img_height / self.image_height

        return Bbox(x, y, width, height, new_img_height, new_img_width)
    
    def to_x1y1x2y2(self) -> np.ndarray:
        assert self.x > 0 and self.y > 0 and self.width + self.x < self.image_width and self.height + self.y < self.image_height, "Bbox is outside image"
        return np.array([self.x, self.y, self.x + self.width, self.y + self.height])
    
    def to_xcycwh(self) -> np.ndarray:
        return np.array([self.x + self.width / 2, self.y + self.height / 2, self.width, self.height])
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.width, self.height])
    
@dataclass
class NormalizedBbox(Bbox):
    """
    A class representing a bounding box, with x, y, width and height.
    Coordinates are normalized, relative to the original image size.
    """

    @staticmethod
    def from_annotation(annotation: Annotation, height: int, width: int) -> "NormalizedBbox":
        bbox = Bbox.from_annotation(annotation)

        x = bbox.x / width
        y = bbox.y / height
        width = bbox.width / width
        height = bbox.height / height

        return NormalizedBbox(x, y, width, height)
    
@dataclass
class ExpandedBbox(Bbox):
    """
    A class representing a bounding box, with x, y, width and height.
    Coordinates are in pixels, relative to the original image size.
    """
    @staticmethod
    def from_bbox(bbox: Bbox, expand: float = 0.1) -> "ExpandedBbox":
        x = bbox.x - expand * bbox.width
        y = bbox.y - expand * bbox.height
        width = bbox.width + 2 * expand * bbox.width
        height = bbox.height + 2 * expand * bbox.height

        return ExpandedBbox(x, y, width, height)
    
@dataclass
class Vertebra:
    name: str
    grad_morf: str
    grad_visuell: str
    typ: str
    ej_bedombbar: str
    kommentar: str
    coordinates: Optional[Annotation] = None

    
@dataclass
class Vertebrae:
    T4: Vertebra
    T5: Vertebra
    T6: Vertebra
    T7: Vertebra
    T8: Vertebra
    T9: Vertebra
    T10: Vertebra
    T11: Vertebra
    T12: Vertebra
    L1: Vertebra
    L2: Vertebra
    L3: Vertebra
    L4: Vertebra
    ULL: float
    K: float
    R: float
    A: float
    DF: float
    S: float
    BAC: float
    FR: float
    SK: float
    VLS: float
    HLS: float
    VBS: float
    HBS: float
    GRANSKAD: str
    Reader: str
    EJ_BEDÖMBAR: str
    pixel_spacing: Tuple[float, float]
    height: int
    width: int
    annotations: List[Annotation]

    def __iter__(self) -> Iterator[Vertebra]:
        for vertebra in VERTEBRA_NAMES:
            yield getattr(self, vertebra)

    @staticmethod
    def from_json(json_path: Path) -> "Vertebrae":
        with open(json_path) as f:
            data = json.load(f)

        vertebrae = {}
        for vertebra in VERTEBRA_NAMES:

            coords = data[vertebra].get("coordinates", None)
            if coords is not None:
                transformed = coords
                try:
                    annotation = Annotation([Point(*p) for p in transformed])
                except Exception as e:
                    annotation = None
            else:
                annotation = None

            vertebrae[vertebra] = Vertebra(
                vertebra,
                data[vertebra]["GRAD_MORF"],
                data[vertebra]["GRAD_VISUELL"],
                data[vertebra]["TYP"],
                data[vertebra]["_EJ_BEDÖMBAR"],
                data[vertebra]["KOMMENTAR"],
                annotation
            )

        annotations = []
        if data.get("keypoints") is not None:
            for v in data["keypoints"]:
                points = [Point(*p) for p in v]
                annotations.append(Annotation(points))

        other = {
            "ULL": data["ULL"],
            "K": data["K"],
            "R": data["R"],
            "A": data["A"],
            "DF": data["DF"],
            "S": data["S"],
            "BAC": data["BAC"],
            "FR": data["FR"],
            "SK": data["SK"],
            "VLS": data["VLS"],
            "HLS": data["HLS"],
            "VBS": data["VBS"],
            "HBS": data["HBS"],
            "GRANSKAD": data["GRANSKAD"],
            "Reader": data["Reader"],
            "EJ_BED\u00d6MBAR": data["EJ_BED\u00d6MBAR"],
            "pixel_spacing": tuple(data["pixel_spacing"]),
            "height": data.get("height", None),
            "width": data.get("width", None),
            "annotations": annotations
        }
        
        # print(other)

        
        return Vertebrae(**vertebrae, **other)
    
    def labels_to_numpy(self) -> np.ndarray:
        return np.array([v.grad_visuell for v in self.__dict__.values() if isinstance(v, Vertebra)])
    
    def coordinates_to_numpy(self) -> np.ndarray:
        return np.array([v.coordinates.to_numpy() for v in self.__dict__.values() if isinstance(v, Vertebra)])


@dataclass
class Patient:
    moid: str
    root: Path
    spine: DXA
    leg: CT
    arm: CT
    vertebrae: Vertebrae

    @staticmethod
    def from_moid(moid: str, root: Path) -> "Patient":
        patient_dir = root / moid
        spine_path  = patient_dir / "lateral" / "patient.tiff"
        label_path  = patient_dir / "lateral" / "patient.json"
        dxa = DXA(spine_path)
        leg = None
        arm = None

        vertebrae= Vertebrae.from_json(label_path) if label_path.exists() else None
        
        return Patient(
            moid, 
            root, 
            dxa, 
            leg, 
            arm, 
            vertebrae
            )
    
class Compression:
    """
    Convenience class for checking if a vertebra has compression. Has two modes:
    - exists: returns True if the vertebra has compression
    - severity: returns the severity of the compression (0, 1, 2, 3)
    """

    def __init__(self, severity: int, mode: str = "exists") -> None:
        
        self.severity = severity
        self.mode = mode

    def __call__(self, v: Vertebra) -> bool:

        if v.grad_morf is None:
            grad_morf = 0
        else:
            grad_morf = float(v.grad_morf)

        if v.grad_visuell is None:
            grad_visuell = 0
        else:
            grad_visuell = int(v.grad_visuell)

        if v.typ is None:
            typ = 0
        else:
            typ = int(v.typ)

        has_compression = all([grad_morf > 0, grad_visuell > self.severity, typ > 0])

        if self.mode == "exists":
            return has_compression
        elif self.mode == "severity":
            return grad_visuell
        else:
            raise ValueError(f"Unknown compression mode: {self.mode}")
        
def patient_to_image_bbox(patient: Patient, height: int, width: int, bbox_expansion: float = 0.1) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    bboxes = []
    for v in patient.vertebrae:
        if v.coordinates is not None:

            # Get label
            label = v.name
            compression = v.grad_visuell

            bbox = v.coordinates.to_bbox(patient.spine.image.height, patient.spine.image.width).resize(height, width)
            bbox = clip_boxes_to_image(box_convert(torch.Tensor(bbox.to_expanded(bbox_expansion).to_numpy()), "xywh", "xyxy"), (height, width))
            bboxes.append(
                {
                    "label": label,
                    "compression": compression,
                    "bbox": np.array(bbox)
                }
            )

    image = patient.spine.to_numpy(height, width, dtype=np.uint8)

    return image, bboxes


def plot_image_with_points(x: np.ndarray, targets: List[Dict[str, Any]]):
    pass
    
def plot_image_with_bbox(x: np.ndarray, targets: List[Dict[str, Any]]):


    boxes = [BoundingBox(*t["bbox"], label=f"{t['label']}: {int(t['compression'])}") for t in targets]
    bbs = BoundingBoxesOnImage(boxes, shape=x.shape)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(bbs.draw_on_image(np.transpose(x, (1, 2, 0))), )

    return fig, ax