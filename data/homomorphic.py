import cv2
import numpy as np

def homomorphic(img, gamma_l=0.8, gamma_h=1.5, c=1.0):
    # Load the input image and convert to float32

    # Compute the logarithm of the image
    img_log = np.log(img + 1)

    # Compute the Fourier transform of the log image
    f = np.fft.fft2(img_log)

    # Shift the zero-frequency component to the center of the spectrum
    f_shift = np.fft.fftshift(f)

    # Set the filter parameters

    # Compute the filter
    H = np.zeros_like(img)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - rows / 2) ** 2 + (j - cols / 2) ** 2)
            H[i, j] = (gamma_h - gamma_l) * (1 - np.exp(-c * (d ** 2))) + gamma_l

    # Apply the filter to the spectrum
    g_shift = H * f_shift

    # Shift the zero-frequency component back to the corner of the spectrum
    g = np.fft.ifftshift(g_shift)

    # Compute the inverse Fourier transform to obtain the filtered image
    g_log = np.fft.ifft2(g)
    g = np.exp(g_log.real) - 1

    # Scale the output image to the range [0, 255] and convert to uint8
    out = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out