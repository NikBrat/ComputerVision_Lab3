import sys

import cv2
import cv2 as cv
import numpy as np
import skimage as ski


def image_display(image, title: str, folder: str):
    """Displays image"""
    if image is None:
        sys.exit("Could not read the image.")

    cv.imshow(title, image/255)
    cv.waitKey(0)

    # cv.imwrite(f"{folder}/{title.replace(' ', '_')}.jpg", image)


def additive_noise(image, mean=4, sigma=0.1):
    """
    Additive noise function.

    Adds log-normal distribution noise to image

    Defined with the following expression:

    NOISY_IMAGE(x,y) = SOURCE_IMAGE(x,y) + NOISE(x,y)
    """
    rng = np.random.default_rng()
    lognormal = rng.lognormal(mean, sigma**0.5, image.shape)
    image_f = image.astype(np.float32)
    image_out = (image_f + lognormal).clip(0, 255).astype(np.uint8)
    return image_out


def noise_creation(option: int, image, mode: str = 's&p'):
    """Adds noise to image"""
    match option:
        case 1:
            # Impulse noise
            im = ski.util.random_noise(image, mode, amount=0.15)
            image_display(im, 'Impulse noise', 'Noisy_images')
        case 2:
            # Additive noise
            ad = additive_noise(image)
            image_display(ad, 'Additive noise', 'Noisy_images')
        case 3:
            # Gaussian noise
            gs = ski.util.random_noise(image, mode='gaussian', mean=0.01, var=0.1)
            image_display(gs, 'Gaussian noise', 'Noisy_images')
        case 4:
            # Speckle noise
            sp = ski.util.random_noise(image, mode='speckle', mean=0.01, var=0.1)
            image_display(sp, 'Speckle noise', 'Noisy_images')
        case 5:
            # Poisson noise
            po = ski.util.random_noise(image, mode='poisson')
            image_display(po, 'Poisson noise', 'Noisy_images')
        case _:
            # displaying source image
            image_display(image, 'Source image', 'Noisy_images')

    return 0


def contraharmonic(image, m: int, n: int, q):
    """
    Contraharmonic mean filter.

    Filter based on contraharmonic mean:

    FILTERED_IMAGE(x,y)=(SRC_IMAGE(0,0))^(q+1)+...+(SRC_IMAGE(m,n))^(q+1) / (SRC_IMAGE(0,0))^q+...+(SRC_IMAGE(m,n))^q
    """
    kernel = np.ones((m, n), dtype=np.float32)
    num = np.power(image, q+1)  # numerator
    den = np.power(image, q)    # denominator
    filtered_image = cv2.filter2D(src=num, ddepth=-1, kernel=kernel) / cv2.filter2D(src=den, ddepth=-1, kernel=kernel)
    return filtered_image


def image_filtering(option: int, noisy_image, name: str, kx: int = 3, ky: int = 3, m=3, n=3, q=0.0):
    """Image Filtering"""
    match option:
        case 1:
            # Gaussian Blur
            gb = cv.GaussianBlur(noisy_image, (kx, ky), 0)
            image_display(gb, f'Gaussian Blur_{name}_({kx},{ky})', 'Gaussian_Blur')
        case 2:
            # Contraharmonic mean filter
            ch = contraharmonic(noisy_image, m, n, q)
            image_display(ch, f'Contraharmonic_{name}_(m,n={(m,n)},q={q})', 'Contraharmonic_Filter')
            pass
        case _:
            # displaying noisy image
            image_display(noisy_image, 'Noisy_image', 'Noisy_images')
    return 0


# src = cv.imread("lewis-hine-taschen-main-3.jpg", 0)
# noise_creation(5, src)

# noisy images titles
titles = ["Additive_noise", "Gaussian_noise", "Impulse_noise", "Speckle_noise", "Poisson_noise"]

nim = cv.imread(f"Noisy_images/Impulse_noise.jpg", 0)
image_filtering(2, nim, "Impulse_noise", m=3, n=3, q=-1.0)
