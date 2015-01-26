# coding: utf-8
import cv2, numpy as np
import math
import argparse as ap

DEBUG = False

## 1. Extract SURF keypoints and descriptors from an image. [4] ----------
def extract_features(image, algorithm, surfThreshold=1000):

  # Convert image to grayscale (for SIFT detector).
  image_gs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  
  if DEBUG:
      cv2.imwrite("out/gray.jpg", image_gs)
  
  # Detect SIFT features and compute descriptors.
  detector = cv2.FeatureDetector_create(algorithm) # what happens with SURF?
  descriptor = cv2.DescriptorExtractor_create(algorithm) # what happens with SURF?
  
  kp = detector.detect(image_gs)
  (keypoints,descriptors) = descriptor.compute(image_gs,kp)
  #print descriptors.shape
  
  return (keypoints, descriptors)


## 2. Find corresponding features between the images. [2] ----------------
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

  ## Find corresponding features.
  match = match_flann(descriptors1, descriptors2)
  
  points1 = np.array([keypoints1[i].pt for (i, j) in match], np.float32)
  points2 = np.array([keypoints2[j].pt for (i, j) in match], np.float32)
  

  return (points1, points2)


## 3. Calculate the size and offset of the stitched panorama. [5] --------



def calculate_size(size_image1, size_image2, homography):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  if DEBUG:
    print top_left
    print top_right
    print bottom_left
    print bottom_right
  
  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  if DEBUG:
    print np.int32(top_left)
    print np.int32(top_right)
    print np.int32(bottom_left)
    print np.int32(bottom_right)
  
  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  if DEBUG:
    print 'Panodimensions'
    print pano_top
    print pano_bottom
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)
  
  if DEBUG:
    print 'Calculated size:'
    print size
    print 'Calculated offset:'
    print offset
      

  return (size, offset)


## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  ## Combine the two images into one.
  
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (ox, oy) = offset
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  if DEBUG:
    print homography
  homography = translation * homography
  # print homography
  
  # draw the transformed image2
  cv2.warpPerspective(image2, homography, size, panorama)
  
  panorama[oy:h1+oy, ox:ox+w1] = image1  
  # panorama[:h1, :w1] = image1  

  

  return panorama

# def merge_images_translation(image1, image2, offset):

  # ## Put images side-by-side into 'image'.
  # (h1, w1) = image1.shape[:2]
  # (h2, w2) = image2.shape[:2]
  # (ox, oy) = offset
  # ox = int(ox)
  # oy = int(oy)
  # oy = 0
  
  # image = np.zeros((h1+oy, w1+ox, 3), np.uint8)
  
  # image[:h1, :w1] = image1
  # image[:h2, ox:ox+w2] = image2
  
  # return image


##---- No need to change anything below this point. ----------------------


def match_flann(desc1, desc2, r_threshold = 0.12):
  'Finds strong corresponding features in the two given vectors.'
  ## Adapted from <http://stackoverflow.com/a/8311498/72470>.

  ## Build a kd-tree from the second feature vector.
  FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

  ## For each feature in desc1, find the two closest ones in desc2.
  (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

  ## Create a mask that indicates if the first-found item is sufficiently
  ## closer than the second-found, to check if the match is robust.
  mask = dist[:,0] / dist[:,1] < r_threshold
  
  ## Only return robust feature pairs.
  idx1  = np.arange(len(desc1))
  pairs = np.int32(zip(idx1, idx2[:,0]))
  return pairs[mask]
  
def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  ## Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

  return image


if __name__ == "__main__":
      
##  parser = ap.ArgumentParser()
##  parser.add_argument('im1')
##  parser.add_argument('im2')
##  parser.add_argument('-a', '--algorithm', 
##                      help='feature detection algorithm',
##                      choices=['SURF', 'SIFT'],
##                      default='SURF')
##  
##  args = parser.parse_args()
  
  ## Load images.
  image1 = cv2.imread('Image1.jpg')
  image2 = cv2.imread('Image2.jpg')

  ## Detect features and compute descriptors.
  (keypoints1, descriptors1) = extract_features(image1, algorithm='SIFT')
  (keypoints2, descriptors2) = extract_features(image2, algorithm='SIFT')
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  ## Find corresponding features.
  (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  print len(points1), "features matched"
  
  ## Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("correspondences.jpg", correspondences)
  print 'Wrote correspondences.jpg'
  
  ## Find homography between the views.
  (homography, _) = cv2.findHomography(points2, points1)
  
  ## Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  print "output size: %ix%i" % size
  
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  cv2.imwrite("panorama.jpg", panorama)
  print 'Wrote panorama.jpg'
