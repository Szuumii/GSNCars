from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model.models import DVMModel
import torchvision.transforms as transforms
from dataset.CarFronts import FrontDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def tensor_to_image(tensor_image):
    img_size = (300, 300)
    to_pil_image = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image

if __name__ == "__main__":
    dataset_root = "/home/shades/Datasets/resized_DVM/"
    ckp_path = "/home/shades/GitRepos/GSNCars/lightning_logs/logs/version_34/checkpoints/epoch=3-step=47979.ckpt"
    csv_path = '/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_test_0.csv'
    batch_size = 1

    model = DVMModel.load_from_checkpoint(ckp_path)

    transform = transforms.ToTensor()
    dataset = FrontDataset(csv_path, dataset_root, transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    input_tensor, prod_year = next(iter(data_loader))

    target_layers = [model.net.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=True)

    # plt.imshow(tensor_to_image(torch.tensor(grayscale_cam)))
    # plt.show()

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    viz_tensor = input_tensor[0].numpy().reshape(300, 300, 3)
    # viz = np.array([viz_tensor[2], viz_tensor[1], viz_tensor[0]])
    # visualization = show_cam_on_image(viz_tensor, grayscale_cam, use_rgb=True)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    img = tensor_to_image(input_tensor[0])

    visualization = heatmap + img
    visualization = visualization / np.max(visualization)
    vis = np.uint8(255 * visualization)

    superimposed_img = heatmap * 0.4 + img

    # plt.imshow(heatmap)
    # plt.imshow(img)
    # plt.show()

    # Adds a subplot at the 1st position
    fig = plt.figure()
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Car")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(heatmap)
    plt.axis('off')
    plt.title("GradCam")
    plt.show()

    # plt.imshow(tensor_to_image(input_tensor[0]))
    # plt.imshow(grayscale_cam)