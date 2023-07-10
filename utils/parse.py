#%%
from lxml import etree
from typing import *
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import itertools
from sklearn.cluster import KMeans
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
import shapely
import matplotlib

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

def vertebrae_coordinates(spans: List[Tuple[int, int, int, int]]):

    valid_span = lambda span: span[0] >= 0 and span[1] >= 0 and span[2] >= 0 and span[3] >= 0

    spans = np.array([(span[2], span[3]) for span in spans if valid_span(span)])

    if len(spans) % (4*6) != 0:
        raise ValueError("Number of spans is not a multiple of 24")

    idxs = np.argsort(np.arange(len(spans)))
    splits = np.split(idxs, len(idxs) / 4)
    coords = np.array([np.mean(spans[split], axis=0) for split in splits])

    return coords

def calculate_approximate_vertebrae_coordinates(spans: List[Tuple[int, int, int, int]], n_vertebrae: int):

    n_spans = len(spans)
    n_spans_per_cross = 3
    # n_crosses = n_spans // n_spans_per_cross
    n_crosses = n_vertebrae * 6

    midpoints = [[span[2] + span[0] / 2, span[3] + span[1] / 2] for span in spans if span[0] > 0 or span[1] > 0]

    X = np.array(midpoints)

    nbrs = NearestNeighbors(n_neighbors=n_spans_per_cross, algorithm='ball_tree', metric="minkowski").fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Set of indices
    set_of_indices = set(frozenset(i) for i in indices) 
    centers = np.array([np.mean(X[np.array(list(i))], axis=0) for i in set_of_indices])

    kmeans = KMeans(n_clusters=n_crosses, random_state=0, n_init=10, algorithm="lloyd", max_iter=1000000, tol=1e-10).fit(centers)
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

def parse_annotation_pdf(path: Path) -> Tuple[np.ndarray, int, int, List[str]]:

    filename = path.name
    output_path = f"output.html"
    os.system(f"pdf2txt.py -o '{output_path}' '{path}'")

    # Parse the html to extract lines relative to the figure
    offset_spans, tree = parse_document(output_path)
    f = get_div_by_height(tree, 600)[0]
    names = extract_vertebrae_names(tree)

    n_names = len(names)

    # coordinates = calculate_approximate_vertebrae_coordinates(offset_spans, n_names)
    coordinates = vertebrae_coordinates(offset_spans)
    props = extract_props_from_element(f)

    return coordinates, props[0], props[1], names

def height_width_ratio(hull):

    box = hull.minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x, y = box.exterior.coords.xy

    # get length of bounding box edges
    width  = max(x)-min(x)
    height =  max(y)-min(y)

    return height/width

def rectangularity(hull: shapely.geometry.Polygon):
    
    return hull.area / hull.minimum_rotated_rectangle.area

def any_point_in_hull(hull: shapely.geometry.Polygon, points: np.ndarray):
    return any(filter(hull.contains, points.geoms))

def plot_hull(hull: shapely.geometry.Polygon, ax: plt.Axes, **kwargs):
    x, y = hull.exterior.xy
    ax.plot(x,y, **kwargs)

def rotational_sort(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]

def point_candidates(remaining_points: np.ndarray, 
                     n_points_in_vertebra: int, 
                     n_neighbours: int, 
                     ax: Union[plt.Axes,None], 
                     area_threshold: float, 
                     rectangularity_threshold: float, 
                     height_width_ratio_threshold: float):

    distances = squareform(pdist(remaining_points))

    # For each point, select the top 10 closest points
    closest = np.argsort(distances, axis=1)[:, 0:n_neighbours]
    distance = np.sort(distances, axis=1)[:, 0:n_neighbours]
        
    # Points including index
    combinations = np.array(list(itertools.combinations(closest[0][1:], n_points_in_vertebra-1)))
    idx = closest[0][0]
    
    if ax is not None:
        ax.plot(remaining_points[idx, 0], remaining_points[idx, 1], "rx")

    candidates = []
    all_hulls = []
    conditions = []
    for c in combinations:
            
        # All combinations must include the first point
        idxs = np.append(c, idx)

        rotated = rotational_sort(remaining_points[idxs])

        # Create a concave hull from the points
        points  = shapely.MultiPoint(remaining_points[idxs])
        
        hull    = shapely.Polygon(rotated)
        # points  = shapely.MultiPoint(rotated)
        # hull    = shapely.convex_hull(points)
        # hull    = shapely.concave_hull(points, ratio=0.99)
        # try:
        #     hull = alphashape.alphashape([((geom.xy[0][0], geom.xy[1][0])) for geom in points.geoms ],0.0)
        # except TypeError as e:
        #     continue
        #     print(e, [((geom.xy[0][0], geom.xy[1][0])) for geom in points.geoms ])



        all_hulls.append({
            "hull": hull,
            "idxs": idxs,
            "points": points
        })

        # plot_hull(hull, ax)

        # Check if the hull contains any other points
        other_idxs      = np.setdiff1d(closest[0], idxs)
        other_points    = shapely.MultiPoint(remaining_points[other_idxs])
            
            
        if hull.area < area_threshold \
            and hull.is_valid \
            and not any_point_in_hull(hull, other_points) \
            and rectangularity(hull) > rectangularity_threshold \
            and height_width_ratio(hull) > height_width_ratio_threshold:

            score = rectangularity_threshold-rectangularity(hull) 
            # score = rectangularity_threshold-rectangularity(hull) + height_width_ratio(hull) - height_width_ratio_threshold

            print("Area", hull.area)
            print("Rectangularity", rectangularity(hull))
            print("Height width ratio", height_width_ratio(hull))
            print("Any point in hull", any_point_in_hull(hull, other_points))

            candidates.append({
                    "hull": hull,
                    "idxs": idxs,
                    "score": score
            })


    return candidates, all_hulls

def vertebrae_from_points(
        points_list: np.ndarray, 
        names: List[str], 
        n_points_in_vertebra: int = 6, 
        n_neighbours: int = 10, 
        area_threshold: float = 2700, 
        rectangularity_threshold: float = .74, 
        height_width_ratio_threshold: float = 1.0,
        plot: bool = False) -> Dict[str, np.ndarray]:

    polygon_to_points = lambda p: list(zip(*p.exterior.coords.xy))

    f, ax = plt.subplots(figsize=(10,5))

    hulls = []
    remaining_points = points_list.copy()

    if plot:
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1], c='k', marker="x")

    i = 0


    while len(remaining_points) > 0:

        # Sort the points by x coordinate
        remaining_points = remaining_points[np.argsort(remaining_points[:, 0])]

        candidates, all_hulls = point_candidates(
            remaining_points, 
            n_points_in_vertebra, 
            n_neighbours, 
            ax, 
            area_threshold, 
            rectangularity_threshold, 
            height_width_ratio_threshold
            )

        # If there are multiple candidates, select the one with the smallest area
        if len(candidates) > 0:

            candidate = min(candidates, key=lambda x: x['score']) if len(candidates) > 0 else candidates[0]
            # candidate = min(candidates, key=lambda x: x['hull'].area) if len(candidates) > 0 else candidates[0]
            
            hulls.append({
                "points": polygon_to_points(candidate['hull']),
                "name": names[i]
                })

            # Remove the finished points from the list
            remaining_points = np.delete(remaining_points, candidate['idxs'], axis=0)

            # Plot the hull
            if plot:
                plot_hull(candidate['hull'], ax, alpha=0.8, color='k')
                plt.text(candidate['hull'].centroid.x, candidate['hull'].centroid.y, names[i], fontsize=12)

            i += 1


        else:

            break

    hulls_dict = {hull['name']: hull['points'] for hull in hulls}    
    
    return hulls_dict