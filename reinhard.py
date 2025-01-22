import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Define conversion matrices
## RGB to LAB
_rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])

_lms2lab = np.dot(
    np.array([[1 / (3**0.5), 0, 0],
              [0, 1 / (6**0.5), 0],
              [0, 0, 1 / (2**0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)
# ----------------------------------------------------------------------------------------------------------------------

# Define conversion matrices
## LAB to RGB
_lms2rgb = np.linalg.inv(_rgb2lms)
_lab2lms = np.linalg.inv(_lms2lab)
# ----------------------------------------------------------------------------------------------------------------------

# Function that converts an RGB image to LAB
def rgb_to_lab(im_rgb):

    # Get input image dimensions
    m = im_rgb.shape[0]
    n = im_rgb.shape[1]

    # Calculate im_lms values from RGB
    im_rgb = np.reshape(im_rgb, (m * n, 3))
    im_lms = np.dot(_rgb2lms, np.transpose(im_rgb))
    im_lms[im_lms == 0] = np.spacing(1)

    # Calculate LAB values from im_lms
    im_lab = np.dot(_lms2lab, np.log(im_lms))

    # Reshape to 3-channel image
    im_lab = np.reshape(im_lab.transpose(), (m, n, 3))

    return(im_lab)
# ----------------------------------------------------------------------------------------------------------------------

# Function that converts an LAB image to RGB
def lab_to_rgb(im_lab):

    # Get input image dimensions
    m = im_lab.shape[0]
    n = im_lab.shape[1]

    # Calculate im_lms values from LAB
    im_lab = np.reshape(im_lab, (m * n, 3))
    im_lms = np.dot(_lab2lms, np.transpose(im_lab))

    # Calculate RGB values from im_lms
    im_lms = np.exp(im_lms)
    im_lms[im_lms == np.spacing(1)] = 0

    im_rgb = np.dot(_lms2rgb, im_lms)

    # Reshape to 3-channel image
    im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))

    return(im_rgb)
# ----------------------------------------------------------------------------------------------------------------------

# Function that applies color normalization (Reinhard Normalization)
def reinhard(im_src, target_mu, target_sigma, src_mu=None, src_sigma=None, mask_out=None):

    # convert input image to LAB color space
    im_lab = rgb_to_lab(im_src)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    # calculate src_mu and src_sigma if either is not provided
    if (src_mu is None) or (src_sigma is None):
        src_mu = [im_lab[..., i].mean() for i in range(3)]
        src_sigma = [im_lab[..., i].std() for i in range(3)]

    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    # convert back to RGB colorspace
    im_normalized = lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0

    # return masked values and reconstruct unmasked LAB image
    if mask_out is not None:
        im_normalized = im_normalized.data
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_normalized[:, :, i].copy()
            original[np.not_equal(mask_out[:, :, 0], True)] = 0
            new[mask_out[:, :, 0]] = 0
            im_normalized[:, :, i] = new + original
    im_normalized = im_normalized.astype(np.uint8)

    return(im_normalized)

# ----------------------------------------------------------------------------------------------------------------------
