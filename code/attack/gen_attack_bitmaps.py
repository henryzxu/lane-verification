import numpy as np
import os


import matplotlib.pyplot as plt

from verifier.defense import LaneDefense

os.chdir("../../")

os.makedirs("defense/attack_images_v2/", exist_ok=True)
i = 0
while i < 500:
    lane_pts = []

    fit_params = np.polyfit(np.random.randint(50,720, 3), np.random.randint(0, 1280, 3), 3)
    ys_expand = np.arange(250, 720, 0.1)
    xs_expand = np.polyval(fit_params, ys_expand)
    xs = xs_expand
    ys = ys_expand

    l_coords = list(zip(xs, ys))

    l_coords = np.array(l_coords)

    if np.count_nonzero(l_coords[:, 0] > 1280) > 200 or np.count_nonzero(l_coords[:, 0] < 0) > 200:
        continue
    l_coords = np.clip(l_coords, 0, 1279)
    tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
    tmp_mask[np.int_(l_coords[:, 1]), np.int_(l_coords[:, 0])] = 255

    tmp_mask_original = tmp_mask.copy()

    tmp_mask_expanded = LaneDefense.get_stabilized_lane(xs,ys, tmp_mask, set_pixels=True)
    if tmp_mask_expanded is False:
        continue


    plt.imshow(tmp_mask_original, cmap="gray")
    plt.axis('off')
    plt.savefig("defense/attack_images_v2/{}.png".format(i), bbox_inches='tight', pad_inches=0)

    plt.imshow(tmp_mask_expanded, cmap="gray")
    plt.axis('off')
    plt.savefig("defense/attack_images_v2/{}_expanded.png".format(i), bbox_inches='tight', pad_inches=0)

    thinning_mask = np.random.randn(tmp_mask_expanded.shape[0], tmp_mask_expanded.shape[1]) > 0.5
    thinned = thinning_mask * tmp_mask_expanded

    plt.imshow(thinned, cmap="gray")
    plt.axis('off')
    plt.savefig("defense/attack_images_v2/{}_thinned.png".format(i), bbox_inches='tight', pad_inches=0)

    i += 1

