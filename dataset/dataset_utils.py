import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(tensor_image):
    img_size = (400, 400)
    to_pil_image = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image

def display_batch(batch):
    imgs, prod_year = batch
    n_images_per_row = 2
    n_rows = int(np.ceil(len(imgs) / n_images_per_row))
    fig, axis = plt.subplots(n_rows, n_images_per_row)
    for ndx in range(len(imgs)):
        row = ndx // n_images_per_row
        col = ndx % n_images_per_row
        tensor = tensor_to_image(imgs[ndx])
        axis[row, col].imshow(tensor)
        axis[row, col].set_xlabel(prod_year[ndx].item())

def display_batch_with_predictions(model, batch):
    imgs, prod_year = batch
    predictions = model.forward(imgs)
    predicted_class = predictions.argmax(dim=1)

    n_images_per_row = 2
    n_rows = int(np.ceil(len(imgs) / n_images_per_row))
    fig, axis = plt.subplots(n_rows, n_images_per_row)
    for ndx in range(len(imgs)):
        row = ndx // n_images_per_row
        col = ndx % n_images_per_row
        tensor = tensor_to_image(imgs[ndx])
        axis[row, col].imshow(tensor)
        axis[row, col].set_xlabel(f"Prod_year: {prod_year[ndx].item()} vs predicted: {2018 - predicted_class[ndx].item()}")

    plt.show()
