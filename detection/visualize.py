import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
from functools import partial
assert torch.__version__.split('.')[1] == '4'
import model
"""
Usage:
srun --mem 8G --gres=gpu:1 python visualize.py --dataset csv --csv_classes /imatge/ppalau/work/Fishes/data/fishes_classes.csv --csv_val /imatge/ppalau/work/Fishes/test_image.csv --model /imatge/ppalau/work/Fishes/model_weights/0/fishes_retinanet_4.pt
"""

print('CUDA available: {}'.format(torch.cuda.is_available()))

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	"""
	boxes is N x 4
	"""
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	load_your_model = True

	retinanet = None
	if(load_your_model):
		# retinanet = model.resnet50(num_classes=2,)
		retinanet = torch.load(parser.model)
	else:
		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		retinanet = torch.load(parser.model, pickle_module=pickle)
	# Print model to see what resnet backbone uses
	print(retinanet)
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()
	
	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):
		
		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			#scores, classification, transformed_anchors = retinanet(cropped_img.cuda().float())
			print("Scores:")
			print(scores)
			print("Classification")
			print(classification)
			print("Transformed anchors")
			print(transformed_anchors)
			print('Elapsed time: {}'.format(time.time()-st))
			# This low threshold is to check the feature proposal network (resnet)
			print("Scores")
			print(scores)
			idxs = np.where(scores>0.1)
			print("idxs")
			print(idxs)
			# TODO: figure out why plotting the raw bboxes that it gives as output does not work. If we save the image as a .npy and then visualize it is correct, but don't know why
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
			#print("data")
			#print(data)
			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
			#non_maxima_supressed = transformed_anchors[idxs[0,:],:]
			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)
			np.save('/imatge/ppalau/work/Fishes/output_image', img)
			# x11 is not supported so we comment the imshow and save the image then plot it in local machine (cutre)
			#cv2.imshow('img', img)
			#cv2.waitKey(0)


if __name__ == '__main__':
 main()
 print("Finished deteciton")
