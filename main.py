import sys
import cv2 as cv
import numpy as np
import skimage as ski


def image_display(image, title: str):
    if image is None:
        sys.exit("Could not read the image.")

    cv.imshow(title, image)
    cv.waitKey(0)

    if image.dtype != np.uint8:
        image = (255*image).astype(np.uint8)

    cv.imwrite(f"{title.replace(' ', '_')}.jpg", image)


def additive_noise(image, mean=4, sigma=0.1):
    rng = np.random.default_rng()
    lognormal = rng.lognormal(mean, sigma**0.5, image.shape)
    image_f = image.astype(np.float32)
    image_out = (image_f + lognormal).clip(0, 255).astype(np.uint8)
    return image_out


def noise_creation(option: int, image, mode: str = 's&p'):
    match option:
        case 1:
            im = ski.util.random_noise(image, mode, amount=0.15)
            image_display(im, 'Impulse noise')
        case 2:
            ad = additive_noise(image)
            image_display(ad, 'Additive noise')
        case 3:
            gs = ski.util.random_noise(image, mode='gaussian', mean=0.01, var=0.1)
            image_display(gs, 'Gaussian noise')
        case 4:
            sp = ski.util.random_noise(image, mode='speckle', mean=0.01, var=0.1)
            image_display(sp, 'Speckle noise')
        case 5:
            po = ski.util.random_noise(image, mode='poisson')
            image_display(po, 'Poisson Noise')
        case _:
            image_display(image, 'Source image')

    return 0


src = cv.imread("lewis-hine-taschen-main-3.jpg", 0)
noise_creation(2, src)
