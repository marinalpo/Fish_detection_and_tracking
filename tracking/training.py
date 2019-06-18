import copy
import cv2
import glob
import random
import numpy as np
import time
import torch
from torchvision import transforms

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from tracker import network_resnet
from re3_utils.simulator import simulator
from re3_utils.util import im_util, bb_util

from constants import CROP_SIZE
from constants import CROP_PAD
from constants import CROP_SIZE
from constants import LSTM_SIZE

basedir = os.path.dirname(__file__)

USE_SIMULATOR = 0.5
USE_NETWORK_PROB = 0.8
REAL_MOTION_PROB = 1.0 / 8
AREA_CUTOFF = 0.25

BATCH_SIZE = 1
UNROLL_SIZE = 2
NUM_EPOCHS = 1

##class alovDataset(torch.utils.data.Dataset):
##    def __init__(self):
##        labels = sorted(glob.glob(os.path.join(basedir, 'alov300', 'alov300++_rectangleAnnotation_full', '*', '*')))
##        self.len = len(labels)
##        self.image_paths = []
##        self.labels = []
##        for label_sequence in labels:
##            annot = np.loadtxt(label_sequence, delimiter = ' ', dtype = np.float32)
##            annot = torch.from_numpy(annot)
##            self.labels.append(annot)
##            
##            split = os.path.split(label_sequence)
##            resplit = os.path.split(split[0])
##            image_paths = sorted(glob.glob(os.path.join(basedir, 'alov300', 'imagedata++', resplit[1], os.path.splitext(split[1])[0], '*')))
##            annotated = []
##            current_annotated = 0
##            for i, image_path in enumerate(image_paths):
##                if current_annotated<annot.shape[0] and i == int(annot[current_annotated,0]):
##                    annotated.append(image_path)
##                    current_annotated+=1
##            self.image_paths.append(annotated)
##        
##
##    def __getitem__(self, index):
##        images = []
##        print("test")
##        print(self.image_paths[index][0])
##        for image_path in self.image_paths[index]:
##            image = cv2.imread(image_path)
##            # Tracker expects RGB, but opencv loads BGR.
##            imageRGB = image[:,:,::-1]
##            #images.append(imageRGB)
##            #zero padding to have constant vectors
##            pad_image = np.pad(np.array(imageRGB), ((0,1080-imageRGB.shape[0]),(0,1920-imageRGB.shape[1]), (0,0)), 'constant')
##            images.append(pad_image)
##            
##        return images, self.labels[index]
##
##    def __len__(self):
##        return self.

class alovDataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.seed = random.seed()
        labels = sorted(glob.glob(os.path.join(basedir, 'alov300', 'alov300++_rectangleAnnotation_full', '*', '*')))
        self.len = len(labels)
        self.image_paths = []
        self.labels = []
        for label_sequence in labels:
            annot = np.loadtxt(label_sequence, delimiter = ' ', dtype = np.float32)
            #annot = torch.from_numpy(annot)
            
            split = os.path.split(label_sequence)
            resplit = os.path.split(split[0])
            image_paths = sorted(glob.glob(os.path.join(basedir, 'alov300', 'imagedata++', resplit[1], os.path.splitext(split[1])[0], '*')))
            annotated = []
            current_annotated = 0
            #
            for i, image_path in enumerate(image_paths):
                if current_annotated<annot.shape[0] and i+1 == int(annot[current_annotated,0]):
                    annotated.append(image_path)
                    current_annotated+=1
            self.labels.append(annot[:,[3,4,7,8]])
            self.image_paths.append(annotated)
        
    def __getitem(self, index):
        images = []
        print("test")
        print(self.image_paths[index][0])
        print(len(self.image_paths[index]))
        for image_path in self.image_paths[index]:
            image = cv2.imread(image_path)
            # Tracker expects RGB, but opencv loads BGR.
            imageRGB = image[:,:,::-1]
            #images.append(imageRGB)
            #zero padding to have constant vectors
            pad_image = np.pad(np.array(imageRGB), ((0,1080-imageRGB.shape[0]),(0,1920-imageRGB.shape[1]), (0,0)), 'constant')
            images.append(pad_image)
            
        return images, self.labels[index]

    def getbatch(self):
        image_sequences = []
        label_sequences = []
        for _ in range(self.batch_size):
            index = random.randint(0, self.len-1)
            images, labels = self.__getitem(index)
            image_sequences.append(images)
            label_sequences.append(labels)
        return image_sequences, label_sequences
            
        
    def __len__(self):
        return self.len
    

# Randomly jitter the box for a bit of noise.
def add_noise(bbox, prevBBox, imageWidth, imageHeight):
    numTries = 0
    bboxXYWHInit = bb_util.xyxy_to_xywh(bbox)
    while numTries < 10:
        bboxXYWH = bboxXYWHInit.copy()
        centerNoise = np.random.laplace(0,1.0/5,2) * bboxXYWH[[2,3]]
        sizeNoise = np.clip(np.random.laplace(1,1.0/15,2), .6, 1.4)
        bboxXYWH[[2,3]] *= sizeNoise
        bboxXYWH[[0,1]] = bboxXYWH[[0,1]] + centerNoise
        if not (bboxXYWH[0] < prevBBox[0] or bboxXYWH[1] < prevBBox[1] or
            bboxXYWH[0] > prevBBox[2] or bboxXYWH[1] > prevBBox[3] or
            bboxXYWH[0] < 0 or bboxXYWH[1] < 0 or
            bboxXYWH[0] > imageWidth or bboxXYWH[1] > imageHeight):
            numTries = 10
        else:
            numTries += 1

    return fix_bbox_intersection(bb_util.xywh_to_xyxy(bboxXYWH), prevBBox, imageWidth, imageHeight)

# Make sure there is a minimum intersection with the ground truth box and the visible crop.
def fix_bbox_intersection(bbox, gtBox, imageWidth, imageHeight):
    if type(bbox) == list:
        bbox = np.array(bbox)
    if type(gtBox) == list:
        gtBox = np.array(gtBox)

    gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
    bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    while IOU.intersection(bboxLarge, gtBox) / gtBoxArea < AREA_CUTOFF:
        bbox = bbox * .9 + gtBox * .1
        bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    return bbox


# Read a sequence from the batch cache or simulate one and get the ground truth and crops for it.
def data_augmentation(debug=False):
    # Preallocate the space for the images and labels.
    tImage = np.zeros((delta, 2, CROP_SIZE, CROP_SIZE, 3),
            dtype=np.uint8)
    xywhLabels = np.zeros((delta, 4))

    mirrored = random.random() < 0.5
    useSimulator = random.random() < USE_SIMULATOR
    gtType = random.random()
    realMotion = random.random() < REAL_MOTION_PROB

    # Initialize first frame (give the network context).

    if useSimulator:
        # Initialize the simulation and run through a few frames.
        trackingObj, trackedObjects, background = simulator.create_new_track()
        for _ in range(random.randint(0,200)):
            simulator.step(trackedObjects)
            bbox = trackingObj.get_object_box()
            occlusion = simulator.measure_occlusion(bbox, trackingObj.occluder_boxes, cropPad=1)
            if occlusion > .2:
                break
        for _ in range(1000):
            bbox = trackingObj.get_object_box()
            occlusion = simulator.measure_occlusion(bbox, trackingObj.occluder_boxes, cropPad=1)
            if occlusion < 0.01:
                break
            simulator.step(trackedObjects)
        initBox = trackingObj.get_object_box()
        if debug:
            images = [simulator.get_image_for_frame(trackedObjects, background)]
        else:
            images = [np.zeros((SIMULATION_HEIGHT, SIMULATION_WIDTH))]

    else:
        # Read a new data sequence from batch cache and get the ground truth.
        (batchKey, images) = getData()
        gtKey = batchKey
        imageIndex = key_lookup[gtKey]
        initBox = datasets[gtKey[0]][imageIndex, :4].copy()
    if debug:
        bboxes = []
        cropBBoxes = []

    # bboxPrev starts at the initial box and is the best guess (or gt) for the image0 location.
    # noisyBox holds the bboxPrev estimate plus some noise.
    bboxPrev = initBox
    lstmState = None

    for dd in range(delta):
        # bboxOn is the gt location in image1
        if useSimulator:
            bboxOn = trackingObj.get_object_box()
        else:
            newKey = list(gtKey)
            newKey[3] += dd
            newKey = tuple(newKey)
            imageIndex = key_lookup[newKey]
            bboxOn = datasets[newKey[0]][imageIndex, :4].copy()
        if dd == 0:
            noisyBox = bboxOn.copy()
        elif not realMotion and not useSimulator and gtType >= USE_NETWORK_PROB:
            noisyBox = add_noise(bboxOn, bboxOn, images[0].shape[1], images[0].shape[0])
        else:
            noisyBox = fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])

        if useSimulator:
            patch = simulator.render_patch(bboxPrev, background, trackedObjects)
            tImage[dd,0,...] = patch
            if dd > 0:
                simulator.step(trackedObjects)
                bboxOn = trackingObj.get_object_box()
                noisyBox = fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])
        else:
            tImage[dd,0,...] = im_util.get_cropped_input(
                    images[max(dd-1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)[0]

        if useSimulator:
            patch = simulator.render_patch(noisyBox, background, trackedObjects)
            tImage[dd,1,...] = patch
            if debug:
                images.append(simulator.get_image_for_frame(trackedObjects, background))
        else:
            tImage[dd,1,...] = im_util.get_cropped_input(
                    images[dd], noisyBox, CROP_PAD, CROP_SIZE)[0]

        shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, noisyBox, CROP_PAD, 1)
        shiftedBBoxXYWH = bb_util.xyxy_to_xywh(shiftedBBox)
        xywhLabels[dd,:] = shiftedBBoxXYWH


        if gtType < USE_NETWORK_PROB:
            # Run through a single forward pass to get the next box estimate.
            if dd < delta - 1:
                if dd == 0:
                    lstmState = initialLstmState

                feed_dict = {
                        forwardNetworkImagePlaceholder : tImage[dd,...],
                        prevLstmState : lstmState
                        }
                networkOuts, s1, s2 = sess.run([networkOutputs, state1, state2], feed_dict=feed_dict)
                lstmState = (s1[0], s1[1], s2[0], s2[1])

                xyxyPred = networkOuts.squeeze() / 10
                outputBox = bb_util.from_crop_coordinate_system(xyxyPred, noisyBox, CROP_PAD, 1)

                bboxPrev = outputBox
                if debug:
                    bboxes.append(outputBox)
                    cropBBoxes.append(xyxyPred)
        else:
            bboxPrev = bboxOn

        if debug:
            # Look at the inputs to make sure they are correct.
            image0 = tImage[dd,0,...].copy()
            image1 = tImage[dd,1,...].copy()

            xyxyLabel = bb_util.xywh_to_xyxy(xywhLabels[dd,:].squeeze())
            print('xyxy raw', xyxyLabel, 'actual', xyxyLabel * CROP_PAD)
            label = np.zeros((CROP_PAD, CROP_PAD))
            drawing.drawRect(label,  xyxyLabel * CROP_PAD, 0, 1)
            drawing.drawRect(image0, bb_util.xywh_to_xyxy(np.full((4,1), .5) * CROP_SIZE), 2, [255,0,0])
            bigImage0 = images[max(dd-1,0)].copy()
            bigImage1 = images[dd].copy()
            if dd < len(cropBBoxes):
                drawing.drawRect(bigImage1, bboxes[dd], 5, [255,0,0])
                drawing.drawRect(image1, cropBBoxes[dd] * CROP_SIZE, 1, [0,255,0])
                print('pred raw', cropBBoxes[dd], 'actual', cropBBoxes[dd] * CROP_PAD)
            print('\n')

            label[0,0] = 1
            label[0,1] = 0
            plots = [bigImage0, bigImage1, image0, image1, label]
            subplot = drawing.subplot(plots, 3, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5)
            cv2.imshow('debug', subplot[:,:,::-1])
            cv2.waitKey(0)

    if mirrored:
        tImage = np.fliplr(
                tImage.transpose(2,3,4,0,1)).transpose(3,4,0,1,2)
        xywhLabels[...,0] = 1 - xywhLabels[...,0]

    tImage = tImage.reshape([delta * 2] + list(tImage.shape[2:]))
    xyxyLabels = bb_util.xywh_to_xyxy(xywhLabels.T).T * 10
    return {imagePlaceholder: tImage, labelPlaceholder: xyxyLabels}


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

##    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print("TOTAAAAAAAAALLLLLLLLLLLLL")
            # Iterate over data.
            inputs_sequences, labels_sequences = dataloaders.getbatch()
            for inputs_sequence, labels_sequence in zip(inputs_sequences, labels_sequences):
                print("AAAAAAAAAAAAAAAAAAAH")
                print(phase)
##                print(len(inputs_sequence))
##                for image, label in zip(inputs_sequence, labels_sequence):
##                    print(label)
##                    cv2.rectangle(image,
##                        (int(label[0]), int(label[1])),
##                        (int(label[2]), int(label[3])),
##                        [0,0,255], 2)
##                    cv2.imshow('Image', image)
##                    cv2.waitKey(0)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                        
                # initialize the lstm states
                lstmState = [torch.autograd.Variable(torch.rand(1, 1, LSTM_SIZE)).to(device) for _ in range(4)]
                             
                # Load a data sequence in memory
                # preallocate the spaces for the inputs and labels (useful??)
                len_sequence = labels_sequence.shape[0]
                inputs_temp = np.zeros([len_sequence-1, 2, 3, CROP_SIZE, CROP_SIZE])
                inputs_vec = torch.zeros(len_sequence-1, 2, 3, CROP_SIZE, CROP_SIZE, dtype=torch.float, device=device)
                labels_output_temp = np.zeros([len_sequence-1, 4])
                labels_output_vec = torch.zeros(len_sequence-1, 4, dtype=torch.float, device=device)
                labels_input_vec = labels_sequence[:-1]
                for i, labels_input in enumerate(labels_input_vec):
                    # get the input values
                    cropped_input0, _ = im_util.get_cropped_input(inputs_sequence[i], labels_input, CROP_PAD, CROP_SIZE)
                    cropped_input1, _ = im_util.get_cropped_input(inputs_sequence[i+1], labels_input, CROP_PAD, CROP_SIZE)
                    inputs_temp[i] = np.concatenate((cropped_input0, cropped_input1)).reshape(2, 3, CROP_SIZE, CROP_SIZE)
                    # fit the output labels to the network output
                    labels_output_temp[i] = bb_util.to_crop_coordinate_system(labels_sequence[i+1], labels_input, CROP_PAD, CROP_SIZE)
                # Transfer to the gpu memory only once for each sequence (very slow operation)
                inputs_vec = torch.Tensor(inputs_temp).to(device)
                labels_output_vec = torch.Tensor(labels_output_temp).to(device)
                
                # Learn/Test this data sequence
                for i, _ in enumerate(labels_input_vec):
                        # set variables
                        inputs = inputs_vec[i]
                        labels_output = labels_output_vec[i]
                             
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs, s1, s2 = model(inputs=inputs,batch_size=1, prevLstmState=lstmState)
                            lstmState = [s1[0], s1[1], s2[0], s2[1]]
                            loss = criterion(outputs, labels_output[i])

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

            epoch_loss = 100#running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100#running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Data augmentation and normalization for training
# Just normalization for validation
##input_size = CROP_SIZE
##data_transforms = {
##    'train': transforms.Compose([
##        transforms.RandomResizedCrop(input_size),
##        transforms.RandomHorizontalFlip(),
##        transforms.ToTensor(),
##        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
##    ]),
##    'val': transforms.Compose([
##        transforms.Resize(input_size),
##        transforms.CenterCrop(input_size),
##        transforms.ToTensor(),
##        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
##    ]),
##}

# Import the model 
model_ft = network_resnet.torch_net()

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
###dataset = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", 'imagenet_video')
##dataset_name = 'imagenet_video'
##dataset_gt = get_datasets.get_data_for_dataset(dataset_name, 'train')['gt']
##image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
### Create training and validation dataloaders
##dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataloaders_dict = alovDataset(batch_size=BATCH_SIZE)
##dataloaders_dict = {x: torch.utils.data.DataLoader(dataset=dataset,
##                                                  batch_size=BATCH_SIZE,
##                                                  shuffle=True,
##                                                  num_workers=0) for x in ['train', 'val']}
##dataloaders_dict = torch.utils.data.DataLoader(dataset=dataset,
##                                                batch_size=BATCH_SIZE,
##                                                shuffle=True,
##                                                num_workers=4)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
##print("Params to learn:")
##if feature_extract:
##    params_to_update = []
##    for name,param in model_ft.named_parameters():
##        if param.requires_grad == True:
##            params_to_update.append(param)
##            print("\t",name)
##else:
##    for name,param in model_ft.named_parameters():
##        if param.requires_grad == True:
##            print("\t",name)

##tests1, tests2 = dataloaders_dict.getbatch()
##for test1, test2 in zip(tests1, tests2):
##    print("test")
##print("ok")
##print(len(test2))

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(params_to_update, lr=0.01)

# Setup the loss fxn
#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

# Save the trained model
torch.save(model_ft.state_dict(), "checkpoint.pth")
