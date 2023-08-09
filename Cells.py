import numpy as np
import os
from skimage.measure import regionprops
import pandas
import cv2
import glob
import re
from patchify import patchify, unpatchify
from napari_segment_blobs_and_things_with_membranes import connected_component_labeling
from numba import njit
from skimage import io

class Cells:
    def __init__(self, sl_num):
        self.sl_num = sl_num
        self.bboxes = None
        self.labels = None
        self.imageshape = None
        # self.image = None
        self.file = r'\*_export_s{}_*'.format(str(sl_num).zfill(3))
    def get_imageshape(self, folder):
        '''
        params
        '''
        imgfile_list = []
        patchsize = (9360,8792)
        
        R = 0
        C = 0
        
        for imgfile in glob.glob(folder + self.file):
            imgfile_list.append(imgfile)
        
            ind = re.findall('\d+', imgfile[-15:])
            r, c = int(ind[1]), int(ind[2])
        
            if r > R:
                R = r
            if c > C:
                C = c
        
        self.imageshape = (4, 4, *patchsize)
        return self.imageshape

    def img_from_tiles(self, folder):
        '''
        A folder containing image tiles FROM 1 SEGMENTATION OBJECT exported from VAST. Has the format <filename>.vsseg_export_s%z%_Y%y%_X%x%.tif
        run this function for every segmentation object. If cell==False, imageshape should not be None
        :param folder: folder containing 1 segmentation object exported from VAST
        :param slice: list of slice number to export. use slice='all' for exporting all slices
        :return: patched image from the tiles.
        '''
        imagepatches = np.zeros(shape=self.imageshape, dtype=np.uint8)

        for imgfile in glob.glob(folder + self.file):
            ind = re.findall('\d+', imgfile[-15:])
            r, c = int(ind[1]), int(ind[2])

            imagepatches[r-1, c-1, :, :] = io.imread(imgfile).astype(np.uint8)

        image = unpatchify(imagepatches, (self.imageshape[0] * self.imageshape[2], self.imageshape[1] * self.imageshape[3]))
        image = image.astype(np.uint8)
        
        return image


    def label_cells(self, image):
        '''
        converts an ndarray of image to labels. Uses connected compoennt labelling. (needs to proof read with image)
        :param image: 2D image of ndarray
        :return: 2D labelled image
        '''
        binary_image = image.astype(bool)
        labels = connected_component_labeling(binary_image)
        self.labels = labels.astype(np.uint8)
        return self.labels

    @staticmethod
    def draw_bbox(bbox):
        '''
        Takes a tuple of x and y values for bounding box. Returns an array of coordinates to be passed as data points in napari shape layer
        :param bbox: Tuple of (minr, minc, maxr, maxc). returned from skimage's region.bbox.
        :return: 4 x 2 array of path length of a rectangle
        '''
        rect = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        return rect

    #def get_bbox(self, labels):
        #self.bboxes = [[],[]]
        #for region in regionprops(labels):
           # bbox=region.bbox
            #self.bboxes[0].append(bbox)

            #bbox_rect = Cells.draw_bbox(bbox)
            #self.bboxes[1].append(bbox_rect)

        #return self.bboxes



if __name__ == "__main__":
    pass
    # folder = r'D:\Hanyi temp\MG\cell-png'
    # cell5 = Cells(5)
    # imageshape = cell5.get_imageshape(folder)
    # cell5_image = cell5.img_from_tiles(folder)
    # cell5_labels = cell5.label_cells(cell5_image)
    # cell5_bboxes = cell5.get_bbox(cell5_labels)

    # print(cell5_bboxes[0][0])
    # print(cell5_bboxes[1][0])

    # print(cell5_bboxes[0][25])
    # print(cell5_bboxes[1][25])


