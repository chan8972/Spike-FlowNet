import numpy as np
import os
import h5py
import argparse

parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS', help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='../datasets/indoor_flying1/indoor_flying1_data.hdf5', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()


save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
  os.makedirs(save_path)

count_dir = os.path.join(save_path, 'count_data')
if not os.path.exists(count_dir):
  os.makedirs(count_dir)
  
gray_dir = os.path.join(save_path, 'gray_data')
if not os.path.exists(gray_dir):
  os.makedirs(gray_dir)
  

class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)], shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0, gray=0, image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0):
        print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)

        split_interval = image_raw_ts_temp.shape[0]
        data_split = 10 # N * (number of event frames from each groups)

        td_img_c = np.zeros((2, self.height, self.width, data_split), dtype=np.uint8)

        t_index = 0
        for i in range(split_interval-(dt_time_temp-1)):
            if image_raw_event_inds_temp[i-1] < 0:
                frame_data = input_event[0:image_raw_event_inds_temp[i+(dt_time_temp-1)], :]
            else:
                frame_data = input_event[image_raw_event_inds_temp[i-1]:image_raw_event_inds_temp[i+(dt_time_temp-1)], :]

            if frame_data.size > 0:
                td_img_c.fill(0)

                for m in range(data_split):
                    for vv in range(int(frame_data.shape[0]/data_split)):
                        v = int(frame_data.shape[0] / data_split)*m + vv
                        if frame_data[v, 3].item() == -1:
                            td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                        elif frame_data[v, 3].item() == 1:
                            td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1

            t_index = t_index + 1

            np.save(os.path.join(count_dir, str(i)), td_img_c)
            np.save(os.path.join(gray_dir, str(i)), gray[i,:,:])



d_set = h5py.File(args.data_path, 'r')

raw_data = d_set['davis']['left']['events']
image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
gray_image = d_set['davis']['left']['image_raw']
d_set = None

dt_time = 1

td = Events(raw_data.shape[0])
# Events
td.generate_fimage(input_event=raw_data, gray=gray_image, image_raw_event_inds_temp=image_raw_event_inds, image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
raw_data = None


print('Encoding complete!')
