import cv2
import numpy as np
from math import floor


class HOG:
    def __init__(self):
        self.loaded = False
        self.words = None

    def load_words(self, filename='../data/words.npy'):
        if not self.loaded:
            self.loaded = True
            self.words = np.load(filename)

    def reduced_HoG_Features(self, image, stride=8):
        """
        Reduced Dimensional HOG Features
        Takes colored image as input
        """
        image = np.array(image).astype(np.double)
        w, h, _ = image.shape

        epsilon = 0.00001
        # unit vectors used to compute gradient orientation
        x_component = np.array([1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736,
                                -0.5000, -0.7660, -0.9397]).astype(np.double)
        y_component = np.array([0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848,
                                0.8660, 0.6428, 0.3420]).astype(np.double)

        blocks = (w//stride, h//stride)
        histograms = np.zeros((*blocks, 18))

        output_features = np.zeros((w//stride - 2, h//stride - 2, 31))

        visible_x = w//stride * stride
        visible_y = h//stride * stride

        for x in range(1, visible_x-1):
            for y in range(1, visible_y-1):
                grad_x = image[x+1, y, :] - image[x-1, y, :]
                grad_y = image[x, y+1, :] - image[x, y-1, :]
                magnitude = grad_x**2 + grad_y**2
                # Select the strongest gradient
                max_idx = np.argmax(magnitude)
                grad_x = grad_x[max_idx]
                grad_y = grad_y[max_idx]
                magnitude = np.sqrt(magnitude[max_idx])

                # Place the gradient in a bin of 20 degrees each
                best_orientation = 0
                dot = x_component * grad_x + y_component * grad_y
                max_idx = np.argmax(np.abs(dot))
                if dot[max_idx] >= 0:
                    best_orientation = max_idx
                else:
                    # This handles directed edges
                    best_orientation = max_idx + 9

                # add to 4 histograms around pixel using linear interpolation
                xp = (x + 0.5) / stride - 0.5
                yp = (y + 0.5) / stride - 0.5
                ixp = floor(xp)
                iyp = floor(yp)
                vx = xp - ixp
                vy = yp - iyp
                if ixp >= 0 and iyp >= 0:
                    histograms[ixp][iyp][best_orientation] += (
                        1-vx) * (1-vy) * magnitude
                if ixp+1 < blocks[1] and iyp >= 0:
                    histograms[ixp+1][iyp][best_orientation] += vx * \
                        (1-vy) * magnitude
                if ixp >= 0 and iyp+1 < blocks[0]:
                    histograms[ixp][iyp +
                                    1][best_orientation] += (1-vx) * vy * magnitude
                if ixp+1 < blocks[1] and iyp+1 < blocks[0]:
                    histograms[ixp+1][iyp +
                                      1][best_orientation] += vx * vy * magnitude
        # normalizing factors
        norm_fac = np.sum(histograms, axis=2)

        # write reduced features
        out_x = w // stride - 2
        out_y = h // stride - 2
        for x in range(out_x):
            for y in range(out_y):
                n1 = 1.0 / np.sqrt(norm_fac[x+1][y+1] + norm_fac[x+2]
                                   [y+1] + norm_fac[x+1][y+2] + norm_fac[x+2][y+2] + epsilon)
                n2 = 1.0 / np.sqrt(norm_fac[x+1][y] + norm_fac[x+2][y] +
                                   norm_fac[x+1][y+1] + norm_fac[x+2][y+1] + epsilon)
                n3 = 1.0 / np.sqrt(norm_fac[x][y+1] + norm_fac[x+1]
                                   [y+1] + norm_fac[x][y+2] + norm_fac[x+1][y+2] + epsilon)
                n4 = 1.0 / np.sqrt(norm_fac[x][y] + norm_fac[x+1][y] +
                                   norm_fac[x][y+1] + norm_fac[x+1][y+1] + epsilon)

                h1 = np.minimum(histograms[x+1, y+1, :] * n1, 0.2)
                h2 = np.minimum(histograms[x+1, y, :] * n2, 0.2)
                h3 = np.minimum(histograms[x, y+1, :] * n3, 0.2)
                h4 = np.minimum(histograms[x, y, :] * n4, 0.2)
                output_features[x, y, :18] = 0.5 * (h1 + h2 + h3 + h4)
                t1 = np.sum(h1)
                t2 = np.sum(h2)
                t3 = np.sum(h3)
                t4 = np.sum(h4)

                h1 = np.minimum(
                    (histograms[x+1, y+1, :9] + histograms[x+1, y+1, 9:]) * n1, 0.2)
                h2 = np.minimum(
                    (histograms[x+1, y, :9] + histograms[x+1, y, 9:]) * n2, 0.2)
                h3 = np.minimum(
                    (histograms[x, y+1, :9] + histograms[x, y+1, 9:]) * n3, 0.2)
                h4 = np.minimum(
                    (histograms[x, y, :9] + histograms[x, y, 9:]) * n4, 0.2)

                output_features[x, y, 18:27] = 0.5 * (h1 + h2 + h3 + h4)
                output_features[x][y][27] = 0.2357 * t1
                output_features[x][y][28] = 0.2357 * t2
                output_features[x][y][29] = 0.2357 * t3
                output_features[x][y][30] = 0.2357 * t4
        return output_features

    def stack_neigbours(self, features):
        """
        Stacking of neighbouring pixels (2x2)
        """
        W, H, _ = features.shape
        W = W - 1
        H = H - 1
        descriptors = np.zeros((124, W * H))
        descriptors[:31, :] = np.uint8(
            features[:W, :H, :].reshape((-1, 31)).T * 255)
        descriptors[31:62, :] = np.uint8(
            features[1:W+1, :H, :].reshape((-1, 31)).T * 255)
        descriptors[62:93, :] = np.uint8(
            features[:W, 1:H+1, :].reshape((-1, 31)).T * 255)
        descriptors[93:, :] = np.uint8(
            features[1:W+1, 1:H+1, :].reshape((-1, 31)).T * 255)
        return descriptors

    def word2vec(self, descriptors):
        """
        Visual Words Dictionary
        """
        self.load_words()
        word_histogram = np.zeros(300).astype(np.int16)
        n = descriptors.shape[1]
        for i in range(n):
            distances = np.sum((self.words.T - descriptors[:, i])**2, axis=1)
            word_histogram[np.argmin(distances)] += 1
        return word_histogram

    def hog_pipeline(self, image):
        features = self.reduced_HoG_Features(image)
        descriptors = self.stack_neigbours(features)
        word_histogram = self.word2vec(descriptors)
        return word_histogram

    def pyramid_hog(self, image, resize=(256, 256), levels=2):
        image = cv2.resize(image, resize, cv2.INTER_CUBIC)
        w, h, _ = image.shape
        descriptors = []
        for level in range(levels+1):
            level_descriptors = []

            for i in range(2**level):
                for j in range(2**level):
                    image_part = image[i*(w//(2**level)):(i+1)*(w//(2**level)),
                                       j*(h//(2**level)):(j+1)*(h//(2**level))]
                    level_descriptors.append(self.hog_pipeline(image_part))

            descriptors.append(np.array(level_descriptors))

        return np.array(descriptors)

    def histogram_intersection(self, hist_A, hist_B):
        return np.sum(np.minimum(hist_A, hist_B))

    def pyramid_intersection(self, pyramid_hist_A, pyramid_hist_B):
        levels = len(pyramid_hist_A)
        score = 0
        for level in range(levels):
            weight = 1.0/2**(levels-level)

            similarity = self.histogram_intersection(
                pyramid_hist_A[level][:2**level], pyramid_hist_B[level][:2**level])
            score += weight * similarity
        return score


if __name__ == '__main__':
    img = np.random.randn(32, 32, 3)
    hog = HOG()
    print(hog.pyramid_hog(img).shape)
