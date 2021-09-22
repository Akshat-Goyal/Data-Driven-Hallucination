import cv2
import numpy as np
from os.path import join, exists
import pickle
from utils import videoReader


class Match_Frame:

    def ImageHistogram(self, image, nbins=256):
        hist_b = cv2.calcHist([image], [0], None, [nbins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [nbins], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [nbins], [0, 256])

        hist_des = np.vstack((hist_r, hist_g, hist_b)).ravel()
        hist_des = hist_des/image.size
        return hist_des

    def save_ImageHistogram(self, video_path, sample_frames=100, dir="../data"):
        cap = videoReader(video_path)
        numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_rate = max(1, numberOfFrames//sample_frames)

        idx = []
        frame_descriptors = []
        for i in range(0, numberOfFrames, sampling_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, frame = cap.read()
            idx.append(i)
            frame_descriptors.append(self.ImageHistogram(frame))

        idx = np.array(idx)
        frame_descriptors = np.array(frame_descriptors)
        with open(join(dir, video_path.split("/")[-1].split(".")[0] + ".pickle"), 'wb') as f:
            pickle.dump({'idx': idx, 'descriptors': frame_descriptors},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_ImageHistogram(self, video_path, sample_frames=100, dir="../data"):
        filename = join(dir, video_path.split(
            "/")[-1].split(".")[0] + ".pickle")
        if not exists(filename):
            self.save_ImageHistogram(video_path, sample_frames, dir)

        with open(filename, 'rb') as f:
            return pickle.load(f)

    def best_distribution_frame(self, image, video_path, sample_frames=100):
        '''Search for best frame given a particular video'''
        image_descriptor = self.ImageHistogram(image).reshape((1, -1))

        cap = videoReader(video_path)
        frame_descriptors = self.load_ImageHistogram(video_path, sample_frames)
        distance = np.linalg.norm(
            frame_descriptors['descriptors'] - image_descriptor, axis=1)
        min_index = np.argmin(distance)
        min_distance = distance[min_index]
        min_index = frame_descriptors['idx'][min_index]

        # Resizing the matched frame to image size, for future operations
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_index)
        res, matched_frame = cap.read()
        matched_frame = cv2.resize(
            matched_frame, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
        return matched_frame, min_index, min_distance


if __name__ == '__main__':
    image = cv2.imread('../data/img/singapore.jpg')
    obj = Match_Frame()
    # obj.save_ImageHistogram('../data/videos/window_02.mp4')
    # print(obj.load_ImageHistogram("../data/videos/window_02.mp4")["descriptors"].shape)
    img, min_idx, min_dist = obj.best_distribution_frame(
        image, '../data/videos/window_02.mp4')
    print(min_idx, min_dist)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
