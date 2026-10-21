import os
import h5py
import numpy as np
import cv2

from .calibration import Calibration


class MVSEC():
    def __init__(self, path, name, location='left'):


        self.name = name
        self.location = location
        data_path = os.path.join(path, name + '_data.hdf5')
        gt_path = os.path.join(path, name + '_gt.hdf5')


        data_raw = h5py.File(data_path, 'r')
        if not (name == 'outdoor_day2' and location =='right'):
            self.image_data = data_raw['davis'][location]['image_raw']
            self.image_ts = data_raw['davis'][location]['image_raw_ts']
            self.image_event_inds = data_raw['davis'][location]['image_raw_event_inds']
        self.event_data = data_raw['davis'][location]['events']
        self.event_data_ts = self.event_data[:,2]


        if not (name == 'outdoor_day2' and location =='right'):
            self.gt_raw = h5py.File(gt_path, 'r')
            self.gt_flow_data = self.gt_raw['davis'][location]['flow_dist']  if location == 'left' else None
            self.gt_timestamps = self.gt_raw['davis'][location]['flow_dist_ts'] if location == 'left' else None
            self.gt_depth = self.gt_raw['davis'][location]['depth_image_rect']
            self.gt_depth_ts = self.gt_raw['davis'][location]['depth_image_rect_ts']


        if not (name == 'outdoor_day2' and location =='right'):
            self.__len_i = len(self.image_data)
            self.__len_gt = len(self.gt_depth)
        self.__len_e = len(self.event_data)


        self.DISPARITY_MULTIPLIER = 7.0
        self.INVALID_DISPARITY = 255
        self.FOCAL_LENGTH_X_BASELINE = {
            'indoor_flying': 19.941772,
            'outdoor_night': 19.651191,
            'outdoor_day': 19.635287
        }


        calibration_path = os.path.join(path, name[:-1]+'_calib.zip')
        self.calibration = Calibration(calibration_path, name[:-1])
        self.rectify_map = self.calibration.left_map if location == 'left' else self.calibration.right_map


    def binary_search_h5_dset(self, dset, x, l=None, r=None, side='left'):


        l = 0 if l is None else l
        r = len(dset)-1 if r is None else r
        while l <= r:
            mid = l + (r - l)//2
            midval = dset[mid]
            if midval == x:
                return mid
            elif midval < x:
                l = mid + 1
            else:
                r = mid - 1
        if side == 'left':
            return l
        return r


    def find_ts_index(self, events_t):
        idx = self.binary_search_h5_dset(self.event_data_ts, events_t)
        return idx

    def disparity_to_depth(self, disparity_image, INVALID_DISPARITY=float('inf')):
        unknown_disparity = disparity_image == INVALID_DISPARITY
        depth_image = \
            self.FOCAL_LENGTH_X_BASELINE[self.name[:-1]] / (
            disparity_image + 1e-7)
        depth_image[unknown_disparity] = INVALID_DISPARITY
        return depth_image

    def _depth2disparity(self, depth_image, focal_length_x_baseline):
        disparity_image = np.round(self.DISPARITY_MULTIPLIER *
                                np.abs(focal_length_x_baseline) /
                                (depth_image + 1e-15))
        invalid = np.isnan(disparity_image) | (disparity_image == float('inf')) | (disparity_image >= 255.0)
        disparity_image[invalid] = self.INVALID_DISPARITY

        return disparity_image.astype(np.uint8)

    def get_disparityAndValid(self, index:int):
        depth = self.gt_depth[index]
        disparity = self._depth2disparity(depth, self.FOCAL_LENGTH_X_BASELINE[self.name[:-1]])

        invalid_disparity = (disparity == self.INVALID_DISPARITY)
        valid = (disparity != self.INVALID_DISPARITY)

        disparity = (disparity / self.DISPARITY_MULTIPLIER)
        disparity[invalid_disparity] = float('inf')

        return disparity, valid

    def get_time_ofDisparity(self, index:int) -> np.float64:
        return self.gt_depth_ts[index]

    def get_image(self, index:int) -> np.ndarray:


        return self.image_data[index]

    def _rectify_events(self, events, distorted_to_rectified, image_size):
        rectified_events = []
        width, height = image_size
        for event in events:
            x, y, timestamp, polarity = event
            x_rectified = round(distorted_to_rectified[int(y), int(x)][0])
            y_rectified = round(distorted_to_rectified[int(y), int(x)][1])
            if (0 <= x_rectified < width) and (0 <= y_rectified < height):
                rectified_events.append(
                    [x_rectified, y_rectified, timestamp, polarity])

        return np.array(rectified_events)

    def get_events(self, start:int, end:int, rectify=False) -> np.ndarray:


        events = self.event_data[start:end]
        if rectify:
            events = self._rectify_events(events, self.rectify_map, (346,260))

        return events

    def get_flow(self, index:int) -> np.ndarray:


        return self.gt_flow_data[index]

    def get_time_ofimage(self, index:int) -> np.float64:
        return self.image_ts[index]

    def get_idx_imageToevent(self, index:int) -> np.int64:
        return self.image_event_inds[index]

    def estimate_flow(self, T_start:int, T_end:int) -> np.ndarray:


        U_gt, V_gt = self.__estimate_corresponding_gt_flow(self.gt_flow_data, self.gt_timestamps, T_start, T_end)
        gt_flow = np.stack((U_gt, V_gt), axis=2)
        return gt_flow

    def len_image(self) -> int:
        return self.__len_i

    def len_event(self) -> int:
        return self.__len_e

    def len_flow(self) -> int:
        return self.__len_gt

    def __estimate_corresponding_gt_flow(self, flows, gt_timestamps, start_time, end_time):


        gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
        gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
        x_flow = np.squeeze(flows[gt_iter, 0, ...])
        y_flow = np.squeeze(flows[gt_iter, 1, ...])

        dt = end_time - start_time


        if gt_dt > dt:
            return x_flow * dt / gt_dt, y_flow * dt / gt_dt

        x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                        np.arange(x_flow.shape[0]))
        x_indices = x_indices.astype(np.float32)
        y_indices = y_indices.astype(np.float32)

        orig_x_indices = np.copy(x_indices)
        orig_y_indices = np.copy(y_indices)


        x_mask = np.ones(x_indices.shape, dtype=bool)
        y_mask = np.ones(y_indices.shape, dtype=bool)

        scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
        total_dt = gt_timestamps[gt_iter + 1] - start_time

        self.__prop_flow(x_flow, y_flow,
                x_indices, y_indices,
                x_mask, y_mask,
                scale_factor=scale_factor)

        gt_iter += 1

        while gt_timestamps[gt_iter + 1] < end_time:
            x_flow = np.squeeze(flows[gt_iter, 0, ...])
            y_flow = np.squeeze(flows[gt_iter, 1, ...])

            self.__prop_flow(x_flow, y_flow,
                             x_indices, y_indices,
                             x_mask, y_mask)
            total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

            gt_iter += 1

        final_dt = end_time - gt_timestamps[gt_iter]
        total_dt += final_dt

        final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        x_flow = np.squeeze(flows[gt_iter, 0, ...])
        y_flow = np.squeeze(flows[gt_iter, 1, ...])

        scale_factor = final_dt / final_gt_dt

        self.__prop_flow(x_flow, y_flow,
                x_indices, y_indices,
                x_mask, y_mask,
                scale_factor)

        x_shift = x_indices - orig_x_indices
        y_shift = y_indices - orig_y_indices
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0

        return x_shift, y_shift


    def __prop_flow(self, x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):


        flow_x_interp = cv2.remap(x_flow,
                                x_indices,
                                y_indices,
                                cv2.INTER_NEAREST)

        flow_y_interp = cv2.remap(y_flow,
                                x_indices,
                                y_indices,
                                cv2.INTER_NEAREST)

        x_mask[flow_x_interp == 0] = False
        y_mask[flow_y_interp == 0] = False

        x_indices += flow_x_interp * scale_factor
        y_indices += flow_y_interp * scale_factor

        return
