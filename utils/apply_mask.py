import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random

def generate_colormap(num_class, seed=20230609):
    # Generate random RGB colors for each class.
    random.seed(seed)
    rgb_colors = [(random.random(), random.random(), random.random()) for _ in range(num_class)]
    return mcolors.ListedColormap(rgb_colors)

def apply_mask(image, mask, alpha=0.8, file_path='./gen/test.png'):
    '''
    @image: tensor.Size(H, W, 3)
    @mask: tensor.Size(H, W)
    @alpha: float, form [0, 1]
        It decide the weight between image and mask.
    '''
    # Create a colormap
    cmap = generate_colormap(151)
    # Normalize between 0 and 1
    norm = mcolors.Normalize(vmin=mask.min(), vmax=mask.max())
    # Create a color mask
    colored_mask = cmap(norm(mask))
    # Apply mask onto image
    overlay = (1-alpha)*image + alpha*colored_mask[:,:,:3]
    
    # Store the original image and the overlayed image
    fig, ax = plt.subplots(1, 2, figsize=(8, 10))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(overlay)
    ax[1].set_title('Mask')
    fig.savefig(file_path, bbox_inches='tight')

    return overlay