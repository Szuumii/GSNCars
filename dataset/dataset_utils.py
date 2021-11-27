def tensor_to_image(tensor_image):
    img_size = (400, 400)
    to_pil_image = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image
