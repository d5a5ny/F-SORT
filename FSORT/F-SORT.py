"""
    F-SORT (featured SORT tracker)
    paper : Aircraft Tracking in Aerial Videos Based on Fused RetinaNet and Low-Score Detection Classification

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

"""
from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier

import glob
import time
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# detections=bb_test, trackers=bb_gt
def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_test = bb_test[..., 0:7]
    bb_gt = bb_gt[..., 0:5]

    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def similarity(dets_fvector, trks_vector):
    sim_matrix = np.zeros((len(dets_fvector), len(trks_vector)), dtype=np.float32)
    for d, d_fvector in enumerate(dets_fvector):
        for t, t_fvector in enumerate(trks_vector):
            sim_matrix[d, t] = distance.correlation(d_fvector, t_fvector)
    return 1 - sim_matrix


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """

    if x[2] <= 0:
        x[2] = 0.0000001
    if x[3] <= 0:
        x[3] = 0.0000001

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # print('==== init =====')
        # print(bbox[0:4])
        # define constant velocity model

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # to force returning the detection bbox and not the predicted bbox
        self.bbox = bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.previous_position = []  # store the previous x,y points of the trk id
        self.current_position = []  # store the current x,y points of the trk id
        self.update_position(bbox)  # store the new box position
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update_position(self, bbox):
        if len(bbox) > 0:
            # the if condition is to overcome the problem that bbox comes initially on 1D array while it comes in 2D
            # array for other frames
            x1 = bbox[0] if len(bbox) != 1 else bbox[0][0]
            y1 = bbox[1] if len(bbox) != 1 else bbox[0][1]
            width = (bbox[2] if len(bbox) != 1 else bbox[0][2]) - x1
            height = (bbox[3] if len(bbox) != 1 else bbox[0][3]) - y1
            cx = round(x1 + round(float(width / 2), 2), 2)
            cy = round(y1 + round(float(height / 2), 2), 2)

            if len(self.previous_position) == 0:
                self.previous_position = [cx, cy]
            elif len(self.current_position) == 0:
                self.current_position = [cx, cy]
            else:
                self.previous_position = self.current_position
                self.current_position = [cx, cy]

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        # print('=== update ======')
        # print(bbox[0:4])
        # to force returning the detection bbox and not the predicted bbox
        self.bbox = bbox

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        # print(self.kf)
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        # print('============ self.kf.x ====================')
        # print(self.kf)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        # print(self.bbox)
        bbox = convert_x_to_bbox(self.kf.x)
        self.update_position(bbox)
        return bbox


def associate_detections_to_trackers(detections, trackers, dets_fvector, trks_vector, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    similarity_matrix = similarity(dets_fvector, trks_vector)

    # the similarity function
    combined_matrix = mode_iou_weight * iou_matrix + mode_sim_weight * similarity_matrix

    # check the length before assignment and then do the assignment using hungarian algorithm
    matched_indices = linear_assignment(-combined_matrix) if combined_matrix.size > 0 else np.empty(shape=(0, 2))

    unmatched_detections = []
    unmatched_dets_fvector = []

    # for new targets initialization
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            unmatched_dets_fvector.append(d)

    unmatched_trackers = []
    unmatched_trks_fvector = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            unmatched_trks_fvector.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if combined_matrix[m[0], m[1]] < iou_threshold:  # adjust combined_matrix
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            unmatched_dets_fvector.append(m[0])  # for the feature vector
            unmatched_trks_fvector.append(m[1])  # for the feature vector
        else:
            matches.append(m.reshape(1, 2))

    matches = np.empty((0, 2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    pool_x = []
    pool_y = []
    buffer_high = 0
    buffer_low = 0
    buffer_size = 40
    lock_lazy_classification_model = 0  # lock the model when the buffer fills
    lazy_classification_model = []

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.fvector_trks = []  # fvector of trackers
        # self.fvector_trks = np.array(self.fvector_trks)  # converted to array by awd
        self.frame_count = 0

        self.clusters = 5  # KNN
        self.low_dets_limit = 5  # KNN the limit of the dets score to get

    def fill_pool(self, dets, fvector_dets):
        if len(dets) > 0:
            for d, df in zip(dets, fvector_dets):
                if (d[4] > dets_score) and self.buffer_high <= self.buffer_size:
                    self.buffer_high += 1
                    self.pool_x.append(df.tolist())
                    self.pool_y.append(1)
                if (d[4] < dets_score_low + self.low_dets_limit) and self.buffer_low <= self.buffer_size:
                    self.buffer_low += 1
                    self.pool_x.append(df.tolist())
                    self.pool_y.append(0)

    def get_classification(self, det, fvector):
        fvector = fvector.tolist()
        if len(self.pool_y) >= self.clusters and len(self.pool_x) >= self.clusters and (
                det[4] >= dets_score_low + self.low_dets_limit):
            if not self.lock_lazy_classification_model:
                neigh = KNeighborsClassifier(n_neighbors=self.clusters)  # default distance metric = minkowski
                neigh.fit(self.pool_x, self.pool_y)
                classified = neigh.predict([fvector])

                if self.buffer_high >= self.buffer_size and self.buffer_low >= self.buffer_size:
                    self.lock_lazy_classification_model = 1
                    self.lazy_classification_model = neigh
                    print('========== K-nearest algorithm model locked ========')
            else:
                classified = self.lazy_classification_model.predict([fvector])
        else:
            classified = 0  # not ready
        return classified

    # get the ID
    def update(self, dets, fvector_dets):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
      fvector - a numpy array of features for each detected object.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))

        fvector_trks = self.fvector_trks  ####### get the previous feature vector

        to_del = []
        ret = []
        ret_fvector = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)  # to remove bad trackers ??

        # append the feature vector to the dets and trks except for the first frame
        # dets, trks = get_frame_with_fvector([dets, trks])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks,
                                                                                   fvector_dets,
                                                                                   fvector_trks,
                                                                                   self.iou_threshold)

        self.fvector_trks = []  # reset to append the assigned new LBPs trackers

        # update matched trackers with assigned detections and LBPs
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:  # matched
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                if len(d) > 0:
                    trk.update(dets[d][0])
                    self.fvector_trks.append(fvector_dets[d][0])
                else:
                    self.fvector_trks.append(np.zeros(fvector_lenght))
            else:
                self.fvector_trks.append(fvector_trks[t])

        # create and initialise new trackers for unmatched detections
        low_dets = []
        sim_scores = []
        if len(dets) > 0:
            for i, det in enumerate(dets):
                if dets_score > det[4]:
                    low_dets.append(i)
            self.fill_pool(dets, fvector_dets)
            if len(low_dets) > 0:
                for i, det in enumerate(dets):
                    if i in low_dets:
                        if self.get_classification(dets[i], fvector_dets[i]):
                            sim_scores.append(i)

        for i in unmatched_dets:
            if dets[i][4] >= dets_score or (
                    i in sim_scores if len(sim_scores) > 0 else False):  # enable byteTrack paper
                # if dets[i][4] < dets_score:
                # print(dets[i][4])
                det_kalmn = KalmanBoxTracker(dets[i])
                self.trackers.append(det_kalmn)
                self.fvector_trks.append(fvector_dets[i])

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret_fvector.append(np.array(self.fvector_trks[i - 1]).reshape(1, -1))

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                self.fvector_trks.pop(i)
        if len(ret) > 0:
            return [np.concatenate(ret)]
        return np.empty((0, 5))


##################################### End of program #############################################


##################################### Start of Configuration #####################################

##################################### MOT15 #####################################
MOT15 = ['ADL-Rundle-6_frames', 'ADL-Rundle-8_frames', 'ETH-Bahnhof_frames', 'ETH-Sunnyday_frames', 'KITTI-13_frames',
         'PETS09-S2L1_frames', 'TUD-Stadtmitte_frames', 'Venice-2_frames', 'KITTI-17_frames', 'TUD-Campus_frames',
         'ETH-Pedcross2_frames']

MOT17 = ['MOT17-02_frames', 'MOT17-04_frames', 'MOT17-05_frames', 'MOT17-09_frames',
         'MOT17-10_frames', 'MOT17-11_frames', 'MOT17-13_frames']

string_fused = '_fusedRetina'
MOT15_FusedRetinaNet = [x + string_fused for x in MOT15]
string_retina = '_retina'
MOT15_retina = [x + string_retina for x in MOT15]

MOT17_FusedRetinaNet = [x + string_fused for x in MOT17]
MOT17_retina = [x + string_retina for x in MOT17]

resolution = '_800'
MOT17_FusedRetinaNet = [x + resolution for x in MOT17_FusedRetinaNet]

##################################### Airport #####################################
resolution = '_400'

airport_fused = ['airport-1_frames_fusedRetina', 'airport-2_frames_fusedRetina',
                 'airport-3_frames_fusedRetina', 'airport-4_frames_fusedRetina']

airport_fused = [x + resolution for x in airport_fused]
####################################################################################
# tracking_dataset = MOT15_FusedRetinaNet
# tracking_dataset = MOT17_FusedRetinaNet
tracking_dataset = airport_fused

##################################### SORT parameters #####################################

print('---------- the default values -----------------')

# add the dataset name to the track evaluation file
track_eval_path = 'data/gt/mot_challenge/seqmaps/MOT15-train.txt'
track_eval_arr = np.append(['name'], tracking_dataset)
np.savetxt(track_eval_path, track_eval_arr, fmt='%s')
print('evaluation dataset name is save in MOT15-train.txt')

display = 0  # for display the video track
draw_det_box = 0  # display the det bbox
save_video = 0  # if save video frames on HDD

seq_path = 'data'
phase = 'track'

mode_iou_weight = 0.5
mode_sim_weight = 0.5

# filter the dets file based on score (50%) default = 50
dets_score_low = 15  # to get individually the low score dets (from 20% to 50%)  default = 20
dets_score = 50  # 0 to 25  #60

# the effective part in the 256 feature vector
fvector_lenght = 256  # 28 # 256  # default = 256  #28

# iou_threshold
iou_threshold = 0.5

max_age = 1  # default: 1   # 20  #1   # best mot test is #10
min_hits = 5  # default: 3  # 4   #1    # best mot test is #5

file_name = 'FSORT'

save_path = '/home/mostafa1/PycharmProjects/keras-retinanet-inference/examples/results_jpg/'

print('seq_path = ' + seq_path,
      ', phase = ' + phase + ', max_age = ' + str(max_age) + ', min_hits = ' + str(
          min_hits) + ', iou_threshold = ' + str(iou_threshold))
print('dets_threshold = ' + str((dets_score)) + ', dets_threshold_low = ' + str((dets_score_low)))

print('------------------------------------------------------------')
##################################### start of program #####################################
cnt = 0  # pressing input counter

high_dets_indices = []  # array to store detections with scores more than 0.5 (dets_threshold)
low_dets_indices = []  # array to store detections with scores between 0.5 (dets_threshold) and 0.2 (dets_threshold_low)
high_low_dets_indices = []  # array to store detections with scores more than 0.2 (dets_threshold_low)


for sub_folder in tracking_dataset:

    total_time = 0.0
    total_frames = 0
    total_dets = 0

    colours = np.random.rand(32, 3)  # used only for display

    # set display
    images_folder = os.path.join(seq_path, phase)
    if display:
        if not os.path.exists(images_folder):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    ('
                'https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s '
                '/path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

        if not os.path.exists(os.path.join(save_path, sub_folder)):
            os.makedirs(os.path.join(save_path, sub_folder))

    # out_folder = 'sort_output'
    out_folder = 'data/trackers/mot_challenge/MOT15-train/MPNTrack/data'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # global variable for image name
    image_name = ''

    pattern = os.path.join(seq_path, phase, sub_folder, 'det', 'det.txt')

    # cnt trackers number
    cnt_trackers = 0

    for seq_dets_fn in glob.glob(pattern):
        print(seq_dets_fn)
        mot_tracker = Sort(max_age=max_age,
                           min_hits=min_hits,
                           iou_threshold=iou_threshold)  # create instance of the SORT tracker

        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        # print(len(seq_dets))

        # seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        seq = sub_folder

        # cnt for detections ina file
        # print('Number of detections are ' + str(filtered_dets) + ' of '+str(total_dets))
        print('Number of detections are {}, and Number of frames are {}'.format(str(len(seq_dets)),
                                                                                int(seq_dets[:, 0].max())))

        # open each file in the folder
        with open(os.path.join(out_folder, '%s.txt' % seq), 'w') as out_file:
            print("Processing %s." % seq)

            # prev_dets, prev_fvector = [], []
            for frame in range(int(seq_dets[:, 0].max())):

                frame += 1  # detection and frame numbers begin at 1

                cnt = cnt + 1  # increment for the press key display

                image_name = '{0:03}'.format(frame) if "3" not in seq else '{}'.format(frame)

                # if image_name =='186':
                #    break

                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]

                # open me and don't forget
                dets[:, 2:4] += dets[:, 0:2]  # convert from [x1,y1,w,h] to [x1,y1,x2,y2]

                dets_indices_h, dets_indices_l, all_dets = [], [], []

                if len(dets) != 0:
                    for t, det in enumerate(dets):
                        if det[4] >= dets_score:
                            dets_indices_h.append(t)
                            all_dets.append(t)
                        elif det[4] >= dets_score_low:
                            dets_indices_l.append(t)
                            all_dets.append(t)

                total_frames += 1

                # set display
                if display:
                    # fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % frame)
                    # print("{}----{}".format(frame,seq))
                    if "airport-3" in seq:
                        fn = os.path.join(images_folder, sub_folder, 'img1',
                                          '{}.jpg'.format(frame))  # for seq 3 when using --display
                    elif "airport" in seq:
                        fn = os.path.join(images_folder, sub_folder, 'img1', '{0:03}.jpg'.format(frame))
                    else:
                        fn = os.path.join(images_folder, sub_folder, 'img1', '{0:06}.jpg'.format(frame))

                    im = io.imread(fn)
                    ax1.imshow(im, aspect='auto')
                    plt.title(seq + ' Tracked Targets (' + str(frame) + ')')

                fvector_dets = seq_dets[seq_dets[:, 0] == frame, 0: fvector_lenght]

                dets = dets[all_dets]
                fvector_dets = fvector_dets[all_dets]

                start_time = time.time()
                total_dets += len(dets) if len(dets) > 0 else 0
                trackers = mot_tracker.update(dets, fvector_dets)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                if len(trackers) > 0:
                    for t in trackers[0]:
                        cnt_trackers += 1
                        # print(d)
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                            frame, t[4], t[0], t[1], t[2] - t[0], t[3] - t[1]),
                              file=out_file)
                        if display:
                            t = t.astype(np.int32)
                            rectangle = patches.Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1], fill=False, lw=2,
                                                          ec=colours[t[4] % 32, :])
                            ax1.add_patch(rectangle)

                            rx, ry = rectangle.get_xy()
                            cx = rx + rectangle.get_width() / 2
                            cy = ry - 10
                            cx = cx + 10
                            ax1.annotate(t[4], (cx, cy), color='w', weight='bold',
                                         fontsize=6, ha='center', va='center')
                # draw the dets bbox
                if len(dets) > 0 and draw_det_box:
                    for det in dets:
                        # if det[4] < dets_score:
                        rectangle = patches.Rectangle((det[0], det[1]), det[2] - det[0], det[3] - det[1],
                                                      fill=False,
                                                      lw=1, ec=[0.7, 0.2, 0.7], ls='dashed')
                        ax1.add_patch(rectangle)

                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width() / 2
                        cy = ry - 10
                        cx = cx - 10
                        ax1.annotate('[{}]'.format(int(det[4])), (cx, cy), color='w', weight='bold',
                                     fontsize=6, ha='center', va='center')


                if display:
                    fig.canvas.flush_events()
                    plt.draw()

                    # save frames
                    if save_video:
                        fig1 = plt.gcf()
                        # remove title and axis labels
                        plt.gca().set_title('')
                        plt.gca().axes.get_xaxis().set_visible(False)
                        plt.gca().axes.get_yaxis().set_visible(False)

                        fig1.savefig(os.path.join(save_path, sub_folder,
                                                  '{}_{}_{}'.format(seq.split('.')[0], frame, file_name) + ".png"),
                                     bbox_inches='tight', pad_inches=0, dpi=150)
                    # clear axis
                    ax1.cla()

            if save_video:
                plt.close(fig1)

    print('total dets = {}'.format(total_dets))

    # cnt for detections ina file
    print('Number of trackers are ' + str(cnt_trackers))
    # initialize the counter
    cnt_trackers = 0

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")

print('\n\n=========== Run the evaluation ================')
exec(open(os.path.join(os.path.abspath("."), 'track_eval.py')).read())
