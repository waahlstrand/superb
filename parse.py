#%%
from lxml import etree
from typing import *
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import os
import itertools
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment
import numpy as np

import warnings
warnings.filterwarnings("ignore")
#%%
def parse_html(html):
    parser = etree.HTMLParser()
    tree = etree.parse(html, parser)
    return tree

def extract_short_spans(tree, limit=3):
    spans = tree.xpath(f"//span[number(substring-before(substring-after(@style, 'width:'),'px;')) < {limit}]")
    
    return spans

def get_div_by_height(tree, height):
    spans = tree.xpath(f"//div[number(substring-before(substring-after(@style, 'height:'),'px;')) > {height}]")
    
    return spans

def extract_props_from_element(element):

    style = element.get("style")
    width = int(style.split("width:")[1].split("px")[0])
    height = int(style.split("height:")[1].split("px")[0])
    top = int(style.split("top:")[1].split("px")[0])
    left = int(style.split("left:")[1].split("px")[0])

    return width, height, top, left

def extract_vertebrae_names(tree, pixel_size=16):
    elements = tree.xpath(f"//div/span[number(substring-before(substring-after(@style, 'font-size:'),'px')) = {pixel_size}]")
    names = [element.text.strip() for element in elements if len(element.text.strip()) <= 4]

    return names

def offset_by_figure(element: Tuple[int, int, int, int], f: Tuple[int, int, int, int]):
    return element[0], element[1], element[2] - f[2], element[3] - f[3]

def parse_document(path: str):
    tree = parse_html(path)
    short_spans = extract_short_spans(tree)
    figure = get_div_by_height(tree, 600)[0]
    f = extract_props_from_element(figure)

    offset_spans = []

    for span in short_spans:
        props = extract_props_from_element(span)
        offset_props = offset_by_figure(props, f)

        offset_spans.append(offset_props)

    return offset_spans, tree

def bounding_box(spans: Tuple[int, int, int, int]):
    top = min([span[2] for span in spans])
    left = min([span[3] for span in spans])
    bottom = max([span[2] + span[0] for span in spans])
    right = max([span[3] + span[1] for span in spans])

    return top, left, bottom, right

def middle_point_of_bounding_box(spans: List[Tuple[int, int, int, int]]):
    top, left, bottom, right = bounding_box(spans)
    return (top + bottom) / 2, (left + right) / 2

def middle_point_of_list_of_points(points: List[Tuple[float, float]]):
    top = min([point[0] for point in points])
    left = min([point[1] for point in points])
    bottom = max([point[0] for point in points])
    right = max([point[1] for point in points])

    return (top + bottom) / 2, (left + right) / 2

def calculate_approximate_vertebrae_coordinates(spans: List[Tuple[int, int, int, int]]):

    n_spans = len(spans)
    n_spans_per_cross = 4
    n_crosses = n_spans // n_spans_per_cross

    midpoints = [[span[2] + span[0] / 2, span[3] + span[1] / 2] for span in spans]

    X = np.array(midpoints)

    nbrs = NearestNeighbors(n_neighbors=n_spans_per_cross, algorithm='ball_tree', metric="minkowski").fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Set of indices
    set_of_indices = set(frozenset(i) for i in indices) 
    centers = np.array([np.mean(X[np.array(list(i))], axis=0) for i in set_of_indices])

    kmeans = KMeans(n_clusters=n_crosses, random_state=0, algorithm="full", max_iter=1000000, tol=1e-10).fit(centers)
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers



def create_grouped_spans(groups, offset_spans):
    grouped_spans = []
    for group in groups:
        group_spans = [offset_spans[i] for i in group]
        grouped_spans.append(group_spans)
    return grouped_spans

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_annotation_pdf(path: Path) -> np.ndarray:

    filename = path.name
    output_path = f"output.html"
    os.system(f"pdf2txt.py -o '{output_path}' '{path}'")

    # Parse the html to extract lines relative to the figure
    offset_spans, tree = parse_document(output_path)

    coordinates = calculate_approximate_vertebrae_coordinates(offset_spans)

    return coordinates

# %%

path = Path("/home/victor/data/balder/datasets/superb/analyzes/VFA-analyser MO0544-/")
pdfs = list(path.rglob("*.pdf"))

problem_pdfs = []
vertebrae_list = []
for pdf in tqdm(pdfs):
    try:
        vertebrae = parse_annotation_pdf(pdf)
        vertebrae_list.append({"pdf": pdf, "vertebrae": vertebrae})
    except Exception as e:
        problem_pdfs.append({"pdf": pdf, "error": e})
        # print(e)

#%%
idx = 22
moid = "MO"+vertebrae_list[idx]["pdf"].name.removesuffix(".pdf").split("MO")[-1].zfill(4)
vertebrae = vertebrae_list[idx]["vertebrae"]
img = cv2.imread(f"/home/victor/data/balder/datasets/superb/patients/{moid}/lateral/{moid}.tiff", cv2.IMREAD_GRAYSCALE)

# Resize
tree = parse_html("output.html")
f = get_div_by_height(tree, 600)[0]
props = extract_props_from_element(f)
img = cv2.resize(img, (props[0], props[1]))

# Flip y axis
# img = cv2.flip(img, 0)

fig, ax = plt.subplots(1, 1, figsize=(5, 10))

ax.imshow(img, cmap="gray", origin="lower")
ax.invert_yaxis()

ax.scatter(vertebrae[:, 1], vertebrae[:, 0], c='r', marker="x")
