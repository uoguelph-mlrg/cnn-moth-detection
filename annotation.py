import json
import logging

logger = logging.getLogger(__name__)


def get_annotation(ann_path):
    '''
    For reading the txt annotation files
    '''
    with open(ann_path, 'r') as file:
        return json.load(file)
# end def get_annotation


def get_bbs(annotation):
    """
    Return data structure of bounding boxes ready for plot.
    It reads the output of get_annotation.
    """

    bbs = []

    # need to deal with no boundingboxes case, if happens assign bb_structure
    # as empty list
    try:
        bb_structure = annotation['Image_data']['boundingboxes']
    except KeyError:
        bb_structure = []

    # there's a bug in the annotation structure, where if there is only
    # a single bb it's not written out as a list
    if not isinstance(bb_structure, list):
        logger.warn('bad bb detected, %s' % annotation['Image_data']['Filename'])
        bb_structure = [bb_structure]

    for bb in bb_structure:

        width = bb['corner_bottom_right_x'] - bb['corner_top_left_x']
        height = bb['corner_bottom_right_y'] - bb['corner_top_left_y']

        # get bottom left co-ordinate
        xy = (bb['corner_top_left_x'], bb['corner_top_left_y'])
        bbs.append((xy, width, height))

    return bbs
# end def get_bbs
