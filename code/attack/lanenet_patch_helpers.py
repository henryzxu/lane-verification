import torch
import numpy as np
from torch.autograd import Variable

def patch_attack(image, applied_patch, mask, guide, lanenet_loss, model, device, lr=1e2, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch).type(torch.FloatTensor).cuda(device=device)
    mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda(device=device)
    count = 0

    perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)

    while count < max_iteration:
        count += 1
        # Optimize the patch
        perturbed_image = Variable(perturbed_image.data, requires_grad=True)
        output = model(perturbed_image)
        cost = -lanenet_loss(output, guide)
        model.zero_grad()
        cost.backward()


        applied_patch = lr * perturbed_image.grad + applied_patch



        perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)





    return perturbed_image, applied_patch


def patch_attack_orig_classifier(image, applied_patch, mask_numpy, patch_shape, guide, classifier_loss, model, device, lr=1e-2, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch).type(torch.FloatTensor).cuda(device=device)
    mask = torch.from_numpy(mask_numpy).type(torch.FloatTensor).cuda(device=device)
    count = 0

    perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)


    while count < max_iteration:

        # Optimize the patch
        perturbed_image = Variable(perturbed_image.data, requires_grad=True)



        output = model(perturbed_image)
        cost = -classifier_loss(output, guide)
        model.zero_grad()
        cost.backward()
        applied_patch = lr * perturbed_image.grad + applied_patch
        perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
        count += 1




    return perturbed_image, applied_patch


def patch_attack_lane(image, applied_patch, mask_numpy, patch_shape, guide, classifier_loss, model, device, lr=1e2, max_iteration=500):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch).type(torch.FloatTensor).cuda(device=device)
    mask = torch.from_numpy(mask_numpy).type(torch.FloatTensor).cuda(device=device)
    count = 0


    perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)



    while count < max_iteration:

        # Optimize the patch
        perturbed_image = Variable(perturbed_image.data, requires_grad=True)

        if count % 50 == 0:
            z, x, y = np.nonzero(mask_numpy)
            rand_idx = max(np.random.choice(x), patch_shape[1])


        perturbed_patch = perturbed_image[:, :, rand_idx - patch_shape[1]: rand_idx, :]


        output = model(perturbed_patch)
        cost = -classifier_loss(output, guide)
        model.zero_grad()
        cost.backward()
        applied_patch = lr * perturbed_image.grad + applied_patch
        perturbed_image = torch.mul(mask, applied_patch) + torch.mul((1 - mask), image)
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
        count += 1




    return perturbed_image, applied_patch



def mask_generation(guides, mask_length, image_size=(3, 288, 512), min_length=5, fixed=False):
    applied_patch = np.zeros(image_size)

    target = guides[0].detach().cpu().numpy()

    target_nonzero_x, target_nonzero_y = np.nonzero(target)

    b = -210.50523386652551
    a = 2.216172914515555

    max_patch_size_y = b + a * image_size[1]
    min_patch_size_y = max_patch_size_y // 10
    patch_ratio_y = min_patch_size_y / max_patch_size_y

    scaled_min_length_y = patch_ratio_y * mask_length



    b_x = -47.89771525764763
    a_x = 0.4234650503137426

    max_patch_size_x = b_x + a_x * image_size[1]
    min_patch_size_x = max_patch_size_x // 10
    patch_ratio_x = min_patch_size_x / max_patch_size_x

    scaled_min_length_x = patch_ratio_x * mask_length


    x_mask = (mask_length // 2 < target_nonzero_x) & (target_nonzero_x < image_size[1] - mask_length//2)
    y_mask = (mask_length // 2 < target_nonzero_y) & (target_nonzero_y < image_size[2] - mask_length//2)
    full_mask = x_mask & y_mask

    target_nonzero_x = target_nonzero_x[full_mask]
    target_nonzero_y = target_nonzero_y[full_mask]

    found_idx = False
    counter = 0

    while not found_idx:
        assert counter <= 10
        try:
            location_idx = np.random.randint(0, len(target_nonzero_x))
        except:
            counter += 1
        else:
            found_idx = True

    x_location, y_location = target_nonzero_x[location_idx], target_nonzero_y[location_idx]

    scaled_patch_size_x = min(max(int((x_location * a_x + b_x) * mask_length / max_patch_size_x), 10), (image_size[1] - x_location)*2)
    if scaled_patch_size_x % 2 == 1:
        scaled_patch_size_x += 1

    scaled_patch_size_y = min(max(int((x_location * a + b) * mask_length / max_patch_size_y), 10), (image_size[2] - y_location)*2)
    if scaled_patch_size_y % 2 == 1:
        scaled_patch_size_y += 1

    if fixed:
        patch = patch_initialization(mask_length_x=mask_length, mask_length_y=mask_length)
    else:
        patch = patch_initialization(mask_length_x=scaled_patch_size_x, mask_length_y=scaled_patch_size_y)
    applied_patch[:, x_location - patch.shape[1]//2:x_location + patch.shape[1]//2, y_location-patch.shape[2]//2:y_location + patch.shape[2]//2] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location



def patch_initialization(image_size=(3, 288, 512), mask_length_x=20, mask_length_y=20):
    patch = np.random.rand(image_size[0], mask_length_x, mask_length_y)
    return patch