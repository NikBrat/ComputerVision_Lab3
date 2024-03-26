import sys
import cv2 as cv
import numpy as np
import skimage as ski
import scipy


def image_display(image, title: str, folder: str):
    """Displays image"""
    if image is None:
        sys.exit("Could not read the image.")

    # cv.imshow(title, image)
    # cv.waitKey(0)
    cv.imwrite(f"{folder}/{title.replace(' ', '_')}.jpg", image)


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
    filtered_image = cv.filter2D(src=num, ddepth=-1, kernel=kernel) / cv.filter2D(src=den, ddepth=-1, kernel=kernel)
    return filtered_image.clip(0, 255).astype(np.uint8)


def rang_filter(src_image, k, rank):
    """Rang filtering"""
    # Filter parameters
    k_size = (k, k)
    kernel = np.ones(k_size, dtype=np.float32)
    rows, cols = src_image.shape[0:2]
    # Convert to float
    # and make image with border
    if src_image.dtype == np.uint8:
        copied_image = src_image.astype(np.float32) / 255
    else:
        copied_image = src_image
    copied_image = cv.copyMakeBorder(copied_image, int((k_size[0] - 1) / 2), int(k_size[0] / 2),
                                     int((k_size[1] - 1) / 2), int(k_size[1] / 2), cv.BORDER_REPLICATE)
    # Fill arrays for each kernel item
    I_layers = np.zeros(src_image.shape + (k_size[0] * k_size[1],), dtype=np.float32)
    if src_image.ndim == 2:
        for i in range(k_size[0]):
            for j in range(k_size[1]):
                I_layers[:, :, i * k_size[1] + j] = kernel[i, j] * copied_image[i:i + rows, j:j + cols]
    else:
        for i in range(k_size[0]):
            for j in range(k_size[1]):
                I_layers[:, :, :, i * k_size[1] + j] = kernel[i, j] * copied_image[i:i + rows, j:j + cols, :]
    # Sort arrays
    I_layers.sort()
    # Choose layer with rank
    if src_image.ndim == 2:
        filtered_image = I_layers[:, :, rank]
    else:
        filtered_image = I_layers[:, :, :, rank]

    return filtered_image


def wiener(src, k):
    """Wiener filter"""
    rows, cols = src.shape[0:2]
    # Define parameters
    k_size = (k, k)
    kernel = np.ones((k_size[0], k_size[1]))
    # Convert to float
    # and make image with border
    if src.dtype == np.uint8:
        img_copy = src.astype(np.float32) / 255
    else:
        img_copy_nb = src
    img_copy = cv.copyMakeBorder(img_copy, int((k_size[0] - 1) / 2), int(k_size[0] / 2), int((k_size[1] - 1) / 2),
                                 int(k_size[1] / 2), cv.BORDER_REPLICATE)
    # Split into layers
    bgr_planes = cv.split(img_copy)
    bgr_planes_2 = []
    k_power = np.power(kernel, 2)
    for plane in bgr_planes:
        plane_power = np.power(plane, 2)
        m = np.zeros(src.shape[0:2], np.float32)
        q = np.zeros(src.shape[0:2], np.float32)
        for i in range(k_size[0]):
            for j in range(k_size[1]):
                m = m + kernel[i, j] * plane[i:i + rows, j:j + cols]
                q = q + k_power[i, j] * plane_power[i:i + rows, j:j + cols]
        m = m / np.sum(kernel)
        q = q - m * m
        v = np.sum(q) / src.size
        # Do filter
        plane_2 = plane[(k_size[0] - 1) // 2:
                        (k_size[0] - 1) // 2 + rows, (k_size[1] - 1) // 2: (k_size[1] - 1) // 2 + cols]
        plane_2 = np.where(q < v, m, (plane_2 - m) * (1 - v / q) + m)
        bgr_planes_2.append(plane_2)
    # Merge image back
    filtered_image = cv.merge(bgr_planes_2)
    if src.dtype != np.uint8:
        filtered_image = (255 * filtered_image).clip(0, 255).astype(np.uint8)

    return filtered_image


def image_filtering(option: int, noisy_image, name: str, kx: int = 3, ky: int = 3, m=3, n=3, q=0.0, k=0, r=0):
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
        case 3:
            # Median Filter
            md = cv.medianBlur(noisy_image, k)
            image_display(md, f'Median_{name}_(k={k})', 'Median_Filter')
        case 4:
            # 2D-Median Filter
            mdf = scipy.signal.medfilt2d(noisy_image, k)
            image_display(mdf, f'Median_2D{name}_(k={k})', 'Median_2D_Filter')
        case 5:
            # Rang Filter
            rn = rang_filter(noisy_image, k, r)
            print(rn)
            image_display(rn, f'Rang_{name}_(k={k},r={r+1})', 'Rang_Filter')
        case 6:
            # Wiener Filter
            wn = wiener(noisy_image, k)
            image_display(wn, f'Wiener_{name}_(k={k})', 'Wiener_Filter')
        case _:
            # displaying noisy image
            image_display(noisy_image, 'Noisy_image', 'Noisy_images')
    return 0


# src = cv.imread("lewis-hine-taschen-main-3.jpg", 0)
# noise_creation(5, src)

# noisy images titles
titles = ["Additive_noise", "Gaussian_noise", "Impulse_noise", "Speckle_noise", "Poisson_noise"]
for title in titles:
    nim = cv.imread(f"Noisy_images/{title}.jpg", 0)
    image_filtering(5, nim, title, k=3, r=0)
