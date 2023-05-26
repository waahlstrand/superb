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
matplotlib.use('Agg')
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

def parse_annotation_pdf(path: Path) -> Tuple[np.ndarray, int, int, List[str]]:

    filename = path.name
    output_path = f"output.html"
    os.system(f"pdf2txt.py -o '{output_path}' '{path}'")

    # Parse the html to extract lines relative to the figure
    offset_spans, tree = parse_document(output_path)
    f = get_div_by_height(tree, 600)[0]
    names = extract_vertebrae_names(tree)

    coordinates = calculate_approximate_vertebrae_coordinates(offset_spans)
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

    f, ax = plt.subplots()

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

#%%
# path = Path("/data/balder/datasets/superb/analyzes/VFA-analyser MO0544-/")
# pdfs = list(path.rglob("*.pdf"))

# problem_pdfs = []
# vertebrae_list = []

# for pdf in tqdm(pdfs):
#     try:
#         vertebrae, height, width, names = parse_annotation_pdf(pdf)
#         vertebrae_list.append({"pdf": pdf, "vertebrae": vertebrae, "height": height, "width": width, "names": names})
#     except Exception as e:
#         problem_pdfs.append({"pdf": pdf, "error": e})

# #%%
# import json
# from preprocessing import labels as lbl

# def pdf_name_to_moid(pdf_name: Path):
#     file_moid = "MO"+pdf_name.name.removesuffix(".pdf").split("MO")[-1].zfill(4)
    
#     # Split and remove e.g. "_LJ, _Revert, _1"
#     file_moid = file_moid.split("_")[0]

#     # Remove LJ and KR from moid
#     file_moid = file_moid.removesuffix("LJ").removesuffix("KR")

    
#     if len(file_moid) == 5:
#         moid = lbl.image_id_to_excel_id(file_moid)
#     else:
#         moid = file_moid

#     return moid

# for vpdf in vertebrae_list:
#     file_moid = "MO"+vpdf["pdf"].name.removesuffix(".pdf").split("MO")[-1].zfill(4)
    
#     # Split and remove e.g. "_LJ, _Revert, _1"
#     file_moid = file_moid.split("_")[0]

#     # Remove LJ and KR from moid
#     file_moid = file_moid.removesuffix("LJ").removesuffix("KR")

    
#     if len(file_moid) == 5:
#         moid = lbl.image_id_to_excel_id(file_moid)
#     else:
#         moid = file_moid

#     with open(f"data/vertebrae/{moid}.json", "w") as f:
#         d = {
#             "pdf": vpdf["pdf"].name,
#             "moid": file_moid,
#             "height": vpdf["width"],
#             "width": vpdf["height"],
#             "vertebrae": vpdf["vertebrae"].tolist(),
#             "names": vpdf["names"]
#         }
#         json.dump(d, f, indent=4)

#%%
# Bad: 4, 5
# removed = [
#         "1977-05-15",
#         "360410-4848",
#         "MO1704",
#          "MO1863",
#          "MO2004",
#          "MO2005",
#          "MO2129",
#          "MO2335",
#          "MO2585",
#          "MO2799",
#          "MO2806",
#          "MO3018",
#          "MO2154"]

# ds = BinaryDataset(
#     Path("/data/balder/datasets/superb/patients"), 
#     removed=removed, 
#     target_size=(1800, 600))
        

# pos_idxs = ds.where_label(lambda x: x > 0)
# ids = [ds.get_idx(idx, label_override=True)[1]["id"] for idx in pos_idxs]

# files = Path("data/vertebrae").glob("*.json")
# vertebrae_list = []
# in_dataset = set()
# for file in files:
#     with open(file, "r") as f:
#         v = json.load(f)
#         vertebrae_list.append(v)

#     if v["moid"] in ids:
#         in_dataset.add(v["moid"])
 

# # Filter if they are in dataset
# filtered_vertebrae_list = [v for v in vertebrae_list if v["moid"] in in_dataset]

# #%%
# from tqdm import tqdm
# bad = []
# for vertebrae in tqdm(filtered_vertebrae_list):

#     points = np.array(vertebrae["vertebrae"])
#     hulls = vertebrae_from_points(points, 
#                                   names=vertebrae["names"],
#                                   n_points_in_vertebra=6, 
#                                   n_neighbours=8, 
#                                   area_threshold=3000, 
#                                   rectangularity_threshold=.5, 
#                                   height_width_ratio_threshold=0.85)
    

    
#     if len(hulls) != len(points) // 6:
#         bad.append(vertebrae)
#     else:
#         with open(f"/data/balder/datasets/superb/patients/{vertebrae['moid']}/lateral/annotation.json", "w") as f:
#             json.dump(hulls, f, indent=4)

# print(len(bad))

# # %%
# # Test a single vertebrae
# # Test: 8
# vertebrae = np.array(filtered_vertebrae_list[8]["vertebrae"])

# hulls = vertebrae_from_points(vertebrae, 
#                                     names=filtered_vertebrae_list[8]["names"],
#                                   n_points_in_vertebra=6, 
#                                   n_neighbours=8, 
#                                   area_threshold=5000, 
#                                   rectangularity_threshold=.4, 
#                                   height_width_ratio_threshold=0.9)

# # %%
# def create_labelstudio_annotation(vertebra: np.ndarray, names: List[str], moid: str, height: int, width: int, local_root: Path = Path("/data/balder/datasets/superb/patients") ,  image_root: Path = Path("/data/balder/datasets/superb/patients")):

#     valid = True
#     hulls = vertebrae_from_points(vertebra, 
#                                   n_points_in_vertebra=6, 
#                                   names=names,
#                                   n_neighbours=8, 
#                                   area_threshold=5000, 
#                                   rectangularity_threshold=.4, 
#                                   height_width_ratio_threshold=0.9)
    
#     if len(hulls) != len(vertebra) // 6:
#         valid = False

#     path = local_root / moid / "lateral" / (moid + ".tiff")
#     remote_path = image_root / moid / "lateral" / (moid + ".tiff")

#     image = Image.open(path)
#     true_width, true_height = image.size
    
#     annotation = {
#         "data": {
#             "img": remote_path.as_posix(),
#         },
#         "predictions": [
#             {
#                 "result": [
#                     {
#                         "original_width": true_width,
#                         "original_height": true_height,
#                         "value": {
#                             "x": 100 * point[1] / width,
#                             "y": 100 * point[0] / height,
#                         },
#                         "id": f"{moid}_{i}_{j}",
#                         "type": "keypointlabels",
#                         "from_name": "keypoint",
#                         "to_name": "image",
#                         "keypointlabels": [
#                             hull["name"] if valid else "Unknown"
#                         ]
#                     }
#                     for j, point in enumerate(hull["points"])
#                 ],
#             }
#             for i, hull in enumerate(hulls)
#         ]
#     }



#     return annotation
# # %%
# vertebrae =filtered_vertebrae_list[8]
# ann = create_labelstudio_annotation(
#     np.array(vertebrae["vertebrae"]), 
#     vertebrae["names"], 
#     vertebrae["moid"], 
#     vertebrae["height"], 
#     vertebrae["width"],
#     local_root=Path("/data/balder/datasets/superb/patients"),
#     image_root=Path("http://snotra.e2.chalmers.se:8081/")
#     )

# with open("test.json", "w") as f:
#     json.dump(ann, f, indent=4)
# # %%
