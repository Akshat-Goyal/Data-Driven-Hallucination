import cv2
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join, exists
from hog import HOG
from utils import videoReader


class Match_Video:

    def __init__(self):
        self.hog = HOG()

    def processVideo(self, path_video, width=256, height=256, samples_per_video=5):
        video = videoReader(path_video)

        numberOfFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("Video info", path_video, "Frames", numberOfFrames,
              "Dimensions", video_width, video_height)

        video_descriptors = []
        idx = np.round(np.linspace(0, numberOfFrames-1,
                                   samples_per_video)).astype(int)
        for i in range(samples_per_video):
            video.set(cv2.CAP_PROP_POS_FRAMES, idx[i])
            _, frame = video.read()
            video_descriptors.append(
                self.hog.pyramid_hog(frame, (width, height)))

        return np.array(video_descriptors)

    def save_HOGFeatures(self, filename='../data/timelapseHOG.pickle', videos_dir='../data/videos'):

        videos_paths = [join(videos_dir, f) for f in listdir(
            videos_dir) if isfile(join(videos_dir, f))]
        count = 0
        total = len(videos_paths)
        hogs = []
        paths = []
        for video_path in videos_paths:
            hogs.append(self.processVideo(video_path))
            paths.append(video_path)
            count += 1
            print('Video processed', count, '/', total)
        hogs = np.array(hogs)
        paths = np.array(paths)
        with open(filename, 'wb') as handle:
            pickle.dump({'hogs': hogs, 'paths': paths}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def checkVideo(self, video_descriptor, image_descriptor):
        return np.max(np.apply_along_axis(self.hog.pyramid_intersection, -1, video_descriptor, image_descriptor))

    def load_HOGFeatures(self, filename='../data/timelapseHOG.pickle'):
        if not exists(filename):
            self.save_HOGFeatures()
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def findBestVideo(self, image, videos_dir='../data/videos'):

        timeLapseVideoHOGs = self.load_HOGFeatures()

        image_descriptor = self.hog.pyramid_hog(image)

        similarity = np.apply_along_axis(
            self.hog.pyramid_intersection, -1, timeLapseVideoHOGs['hogs'], image_descriptor)
        similarity = np.max(similarity, axis=1)
        idx = np.argsort(similarity)
        similarities_array = np.vstack(
            [similarity[idx], timeLapseVideoHOGs['paths'][idx]])
        max_similarity, max_index = similarities_array[:, -1]

        # max_index = None
        # max_similarity = 0
        # similarities_array = []
        # count = 0
        # for video_path,video_descriptor in timeLapseVideoHOGs.items():
        #     similarity = self.checkVideo(video_descriptor,image_descriptor)
        #     similarities_array.append((similarity,video_path))
        #     if similarity > max_similarity:
        #         max_similarity = similarity
        #         max_index = video_path
        #     count += 1
        #     print(count,'\t',similarity,'\t',video_path,)
        # similarities_array.sort()

        print(max_similarity)
        print(max_index)
        return similarities_array[:, :min(5, similarities_array.shape[1])]


if __name__ == '__main__':
    match_video = Match_Video()
    # match_video.save_HOGFeatures()
    # img = np.random.randn(2, 2, 3)
    img = cv2.imread('../data/img/singapore.jpg')
    arr = match_video.findBestVideo(img)
    # print("arr", arr)
