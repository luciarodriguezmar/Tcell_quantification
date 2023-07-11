import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from skimage.measure import regionprops
from scipy import sparse
import matplotlib.patches as mpatches
import numpy.ma as ma
import napari
import numba
from numba import njit
from skimage.measure import regionprops
from napari_segment_blobs_and_things_with_membranes import connected_component_labeling
import glob
from patchify import patchify, unpatchify
import cv2


# for SINGLE SLICE.

properties = {}
cell_folder = ''
sl_num = 1
#cell_labels = None

def draw_bbox(bbox):
    '''
    Takes a tuple of x and y values for bounding box. Returns an array of coordinates to be passed as data points in napari shape layer
    :param bbox: Tuple of (minr, minc, maxr, maxc). returned from skimage's region.bbox.
    :return: 4 x 2 array of path length of a rectangle
    '''
    rect = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    return rect

def remake_bbox(bbox_asstring, stack=True):
    numbers = ''.join([i if i!='[' and i!=']' and i!='\n ' else '' for i in bbox_asstring])
    arr_flat = np.array([int(i) for i in numbers.split(' ') if i!=''])
    if stack:
        bbox = arr_flat.reshape((4,3))
    else:
        bbox = arr_flat.reshape((4,2))
    return bbox

def make_bbox_stack(slices, bbox_slice):
    sls = [np.array([[sl_num]*4]).T for sl_num in slices]
    bbox_stack = [np.hstack((sl, bbox)) for sl,bbox in zip(sls, bbox_slice)]
    return bbox_stack

# for plotting in io - ignore this
# def get_bbox(labels:'uint8 ndarray'):
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # assert type(labels) == np.ndarray
    # ax.imshow(labels)
    #
    # for region in regionprops(labels):
    #     minr, minc, maxr, maxc = region.bbox
    #
    #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                               fill=False, edgecolor='red', linewidth=2)
    #
    #     ax.add_patch(rect)
    #     ax.plot(minc, minr, 'or')
    #     ax.plot(minc, maxr, 'or')
    #     ax.plot(maxc, maxr, 'or')
    #     ax.plot(maxc, minr, 'or')
    #
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()


@njit
def mask_it(labels:'uint8 ndarray', organelle_mask:'ndarray'):
    '''
    Takes image label (ndarray of 2 dimensions) and binary mask of the same dimension. Returns labelled organelles corresponding to the cell.
    :param labels: ndarray
    :param organelle_mask: ndarray. non organelle should be 0 or false
    :return: organelles labelled with their corresponding cells
    '''
    # assert type(labels) == np.ndarray # why assert dosen't work with numba?
    # assert type(organelle_mask) == np.ndarray and (organelle_mask.dtype == bool or np.bool)

    #trying masked array - does not work
    # nroi = labels.max()
    # for i in range(nroi):

    # organelle_mask = ~organelle_mask
    # organelle_mask = organelle_mask * 1
    #
    # organelle_labels = ma.array(labels, mask = organelle_mask)
    # io.imshow(organelle_labels)

    organelle_labels = np.zeros((labels.shape), dtype=np.uint8)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if organelle_mask[i][j] != 0:
                organelle_labels[i][j] = labels[i][j]

    # list comprehension does not work
    # organelle_labels = np.array([[labels[i][j] for i in range(labels.shape[0]) for j in range(labels.shape[1]) if organelle_mask[i][j] == True]])
    # organelle_labels = np.array([[labels[i][j] for (i, j) in zip(range(labels.shape[0]), range(labels.shape[1])) if organelle_mask[i][j] == True]])

    return organelle_labels




def img_from_tiles(folder, sl):
    '''
    A folder containing image tiles FROM 1 SEGMENTATION OBJECT exported from VAST. Has the format <filename>.vsseg_export_s%z%_Y%y%_X%x%.tif
    run this function for every segmentation object
    :param folder: folder containing 1 segmentation object exported from VAST
    :param slice: list of slice number to export. use slice='all' for exporting all slices
    :return: patched image from the tiles.
    '''

    file = r'\*_s{}_*'.format(str(sl).zfill(2))
    imgfile_list = []

    R = 0
    C = 0

    patchsize = (8192, 8192)

    for imgfile in glob.glob(folder + file):
        imgfile_list.append(imgfile)

        r = int(imgfile[-8])
        c = int(imgfile[-5])

        if r > R:
            R = r
        if c > C:
            C = c

    imageshape = (R, C, *patchsize)
    imagepatches = np.zeros(shape=imageshape, dtype=np.uint8)

    for imgfile in imgfile_list:
        r = int(imgfile[-8])
        c = int(imgfile[-5])

        imagepatches[r - 1, c - 1, :, :] = io.imread(imgfile).astype(np.uint8)

    image = unpatchify(imagepatches, (patchsize[0] * R, patchsize[1] * C))
    image = image.astype(np.uint8)

    return image

def label_cells(image):
    '''
    converts an ndarray of image to labels. Uses connected compoennt labelling. (needs to proof read with image)
    :param image: 2D image of ndarray
    :return: 2D labelled image
    '''
    binary_image = image.astype(bool)
    labels = connected_component_labeling(binary_image)
    labels = labels.astype(np.uint8)
    return labels

# def get_ratio(labels1, labels2):
#     '''
#     calculate ratio of 2 areas
#     :param labels1: labelled ndarray of item of choice e.g. organelle, nucleus
#     :param labels2: labelled ndarray of item of choice e.g. organelle, nucleus
#     :return: updates properties['ratio xx']
#     '''
#     ratios = []
#     for i in np.unique(labels2)[1:]:
#         organelle_area = len(np.argwhere(labels1 == i))
#         labels_area = len(np.argwhere(labels2 == i))
#         ratio = organelle_area/labels_area
#         ratios.append(ratio)
#
#     ratios = np.array(ratios)
#     return ratios

def ER_length(ER, labels): # put labels as global variable
    '''
    Calculates total length of ER in a cell.
    :param ER: patched ER from 1 slice. dtype bool or int
    :param labels: cell labels
    :return: properties['ER_length']
    '''
    if ER.dtype != np.uint8:
        if ER.dtype == bool:
            ER = ER * 255
        ER = ER.astype(np.uint8)
    
    # properties['ER_length'] = []
    ER_lengths = [] #remove after class is created

    # labels_draw = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    for region in regionprops(labels):
        label_num = region.label
        # threshold = label_num - 1

        bbox = region.bbox  # returns minr, minc, maxr, maxc
        ER_crop = ER[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        # ER_crop = ER_crop > threshold #thresholding to remove overlapping bboxes with other labels
        # ER_crop = ER_crop.astype(np.uint8)

        contours, hierachy = cv2.findContours(image=ER_crop,
                                              mode=cv2.RETR_TREE,
                                              method=cv2.CHAIN_APPROX_SIMPLE
                                              )
        
        # cv2.drawContours(image=labels_draw[bbox[0]:bbox[2], bbox[1]:bbox[3]],
        #                  contours=contours,
        #                  contourIdx=-1,
        #                  color=(255, 255, 255),
        #                  thickness=1
        #                  )

        ER_len = 0
        for cnt in contours:
            ER_len_single = cv2.arcLength(cnt, True)
            # add include only if cnt[0] location is in label number,
            ER_len += ER_len_single
            
        ER_lengths.append(ER_len)

        #properties['ER_length'].append(ER_len)
    #properties['ER_length'] = np.array(properties['ER_length'])
    ER_lengths = np.array(ER_lengths)
    return ER_lengths #remove after class is created

"""
def count_organelles(organelles_labels, labels):
    '''
    Calculates number of organelles per cell
    :param organelles_labels: ndarray of organelles lavels
    :param labels: cell labels
    :return: properties['organelle_count']
    '''
    #properties['organelle_count'] = []
    organelles_count = []

    for region in regionprops(labels): #optimise this part - also used in ER functions
        bbox = region.bbox
        organelles_label = organelles_labels[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        # bbox_rect = draw_bbox(bbox)
        # bbox_rects.append(bbox_rect)

        organelles_label_num = region.label
        # print(lys_label_num)
        threshold = organelles_label_num - 1
        organelles_bi = organelles_label > threshold  # anyway to optimise this?
        organelles_bi = organelles_bi * 1
        organelles_bi = organelles_bi.astype(np.uint8)


        num_labels, l = cv2.connectedComponents(organelles_bi, connectivity=4)  #why num_labels is 1 extra?

        organelles_count.append(num_labels - 1)

    organelles_count = np.array(organelles_count)
    return organelles_count
"""



@njit #(nopython=True)
def pad(img, new_arr, n, m):
    for i in range(n):
        for j in range(m):
            new_arr[i + 1][j + 1] = img[i][j]


@njit #(nopython=True)
def find_first_2d(arr, start):
    m = len(arr[0])
    for i in range(start, len(arr)):
        for j in range(m):
            if arr[i][j] > 0:
                return (i, j)
    return (-1, -1)


@njit #(nopython=True)
def get_neighbours(px, n, m):
    nbs = [(px[0], px[1] + 1), (px[0], px[1] - 1), (px[0] + 1, px[1]), (px[0] - 1, px[1])]
    
    if px[0] == 0:
        nbs.remove((px[0], px[1] - 1))
    elif px[0] == n:
        nbs.remove((px[0], px[1] + 1))
        
    if px[1] == 0:
        nbs.remove((px[0], px[1] - 1))
    elif px[1] == m:
        nbs.remove((px[0], px[1] + 1))
    
    return nbs


@njit #(nopython=True)
def expand(origin, arr, n, m):
    cell = arr[origin[0]][origin[1]]
    new_gen = [origin]
    while new_gen:
        cur_gen = [*set(new_gen)]
        new_gen = []
        for px in cur_gen:
            if arr[px[0]][px[1]] == cell:
                arr[px[0]][px[1]] = 0
                new_gen.extend(get_neighbours(px, n, m))
    return cell


@njit #(nopython=True)
def contains(l, val):
    for el in l:
        if el == val:
            return True
    return False


@njit(nopython=True)
def count_organelles(arr, n, m, cells):
    '''
    Calculates number of organelles.
    :param arr: numpy array with labeled organelles
    :param n: number of rows 
    :param m: number of columns
    :param cells: list of cell values (sorted)
    :return: list of organelle count, sorted by cell value
    '''
    org_count = {}

    origin = find_first_2d(arr, 0)
    while origin != (-1, -1):
        val = int(expand(origin, arr, n, m))
        if not contains(org_count.keys(), val):
            org_count[val] = 0
        org_count[val] += 1
        origin = find_first_2d(arr, origin[0])

    count_by_cell = []
    keys = org_count.keys()
    for c in cells:
        if c in keys:
            count_by_cell.append(org_count[c])
        else:
            count_by_cell.append(0)

    return count_by_cell


def individual_cells(cell_labels):
    return np.unique(cell_labels)[1:]


if __name__ == "__main__":
    pass
