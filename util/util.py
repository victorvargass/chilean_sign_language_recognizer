from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
import os
import random
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

#Convert a img source to tensor
def image_to_tensor(img_path, use_gpu):
    image = cv2.imread(img_path)
    image = torch.tensor((np.asarray(image).T/255.).astype('float32'))
    image = image.unsqueeze(0)
    if use_gpu:
        image = image.cuda()
    return image

#Plot an image from path
def plot_image(img_path):
    image = plt.imread(img_path)
    plt.imshow(image)
    plt.title("Test original image")

#Plot image and show the predicted class
def plot_image_and_class(model, img_path):
    image = plt.imread(img_path)
    plt.imshow(image)
    tensor_image = image_to_tensor(img_path, True)
    tensor_model_image = model(tensor_image)
    sm, detected_label, accuracy = detect_class(tensor_model_image)
    plt.title('Accuracy: '+ str(accuracy*100)[0:5] + '%')
    plt.suptitle(detected_label)

#Return the class of a Tensor image
def detect_class(tensor_image):
    classes = list('ABCDEFHIKLMNOPQRTUVWY')
    sm = torch.softmax(tensor_image, dim=1).detach().cpu().numpy()
    max_value = sm.max()
    arg_max = sm.argmax()
    return sm[0], classes[arg_max], max_value

#Print results of an image path
def print_results(model, tensor_image):
    tensor_model_image = model(tensor_image)
    sm, detected_label, accuracy = detect_class(tensor_model_image)
    print("Softmax")
    classes = list('ABCDEFHIKLMNOPQRTUVWY')
    for i, acc in enumerate(sm):
        print(classes[i], '\t', str(acc*100)[0:5] + '%')
    print("Detected label: "+detected_label)
    print('Accuracy: '+ str(accuracy*100)[0:5] + '%')

#Get feature maps from a lenet model
def get_lenet_feat_maps(lenet_model, tensor_image):
    f_map_1 = lenet_model.activation(lenet_model.conv1(tensor_image))
    f_map_2 = lenet_model.activation(lenet_model.conv2(f_map_1))
    adapool = lenet_model.adapool(f_map_2)
    view = adapool.view(adapool.size(0), -1)
    f_conn1 = lenet_model.activation(lenet_model.fc1(view))
    f_conn2 = lenet_model.activation(lenet_model.fc2(f_conn1))
    output = lenet_model.fc3(f_conn2)
    return f_map_1, f_map_2#, adapool, f_conn1

#Get feature maps from a alexnet model
def get_alexnet_feat_maps(alexnet_model, tensor_image):
    #f_map = (model.features[1](tensor_image)).detach().cpu().numpy()
    cv1 = (alexnet_model.features[0](tensor_image))
    ac1 = (alexnet_model.features[1](cv1))
    pool1 = (alexnet_model.features[2](ac1))
    cv2 = (alexnet_model.features[3](pool1))
    ac2 = (alexnet_model.features[4](cv2))
    pool2 = (alexnet_model.features[5](ac2))
    cv3 = (alexnet_model.features[6](pool2))
    ac3 = (alexnet_model.features[7](cv3))
    cv4 = (alexnet_model.features[8](ac3))
    ac4 = (alexnet_model.features[9](cv4))
    cv5 = (alexnet_model.features[10](ac4))
    ac5 = (alexnet_model.features[11](cv5))
    pool3 = (alexnet_model.features[12](ac5))
    #print(ac5.shape)
    #print(pool3.shape)
    avgpool = (alexnet_model.avgpool(pool3))
    #print(avgpool.shape)
    #print(pool3)
    view = avgpool.view(avgpool.size(0), 256 * 6 * 6)
    drop1 = alexnet_model.classifier[0](view)
    linear1 = alexnet_model.classifier[1](drop1)
    ac6 = alexnet_model.classifier[2](linear1)
    drop2 = alexnet_model.classifier[3](ac6)
    linear2 = alexnet_model.classifier[4](drop2)
    ac7 = alexnet_model.classifier[5](linear2)
    output = alexnet_model.classifier[6](ac7)

    return pool3

#Plot feature map from a model
def plot_feat_map(f_map):
    feat_map_number = f_map.shape[1]
    print("Feature output maps number: ", feat_map_number)
    divisions = 16
    fig, axes = plt.subplots(nrows=divisions, ncols=divisions, figsize=(20, 20))
    for i in range(0, divisions):
        for j in range(0, divisions):
            index = j + (divisions * i)
            axes[i,j].imshow(f_map[0, index].detach().cpu().numpy())
            axes[i,j].axis('off')
    plt.show()

def show_layer_output(layer):
    #print(layer)
    print("Output layer shape: ", layer.shape)
    #print(layer.argmax(dim = 1))

def test_image_from_path(model, test_img_path, img_name, use_gpu, transform):
    if(img_name == ''):
        r = random.choice(os.listdir(test_img_path)) #change dir name to whatever
        img_path = test_img_path + r

    pil_image = Image.open(img_path)
    transformed_image = transform_test_image(pil_image, transform)
    print("Transform image shape: ", transformed_image.shape)
    if use_gpu:
        transformed_image = transformed_image.cuda()

    print_results(model, transformed_image)
    plot_image(img_path)
    if(model.name != 'VGG16'):
        if model.name == 'AlexNet': feat_maps = get_alexnet_feat_maps(model, transformed_image)
        if model.name == 'LeNet5': feat_maps = get_lenet_feat_maps(model, transformed_image)
        print("Feature maps shape: ", feat_maps.shape)
        plot_feat_map(feat_maps)

def test_image_from_dataset(model, use_gpu, transform, dataset):
    r = random.randint(0, len(dataset))
    img_path = 'balanced/' + dataset[r]

    pil_image = Image.open(img_path)
    transformed_image = transform_test_image(pil_image, transform)
    #print("Transform shape ", transformed_image.shape)
    if use_gpu:
        transformed_image = transformed_image.cuda()

    print_results(model, transformed_image)
    plot_image(img_path)
    if(model.name != 'VGG16'):
        if model.name == 'AlexNet': feat_maps = get_alexnet_feat_maps(model, transformed_image)
        if model.name == 'LeNet5': feat_maps = get_lenet_feat_maps(model, transformed_image)
        print("Feature maps shape: ", feat_maps.shape)
        plot_feat_map(feat_maps)

def transform_test_image(pil_image, transform):
    transformed_image = transform(pil_image)
    numpy_image = transformed_image.numpy()[0]
    plt.imshow(numpy_image)
    plt.title("Resized image transform")
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image