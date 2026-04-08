import numpy as np


class ImageDataAugmentor:
    def __init__(self, augmentations=None, random_seed=None):
        self.augmentations = augmentations or []
        self.random_state = np.random.RandomState(random_seed)

    def antialiasing(self, src_x, src_y, channel):
        h, w = channel.shape[:2]
        is_gray = len(channel.shape) == 2 or channel.shape[-1] == 1
        if is_gray:
            channel = np.squeeze(channel)
        x_floor, y_floor = np.floor(src_x).astype(int), np.floor(src_y).astype(int)
        x_ceil, y_ceil = np.ceil(src_x).astype(int), np.ceil(src_y).astype(int)
        x_floor = np.clip(x_floor, 0, w-1)
        x_ceil = np.clip(x_ceil, 0, w-1)
        y_floor = np.clip(y_floor, 0, h-1)
        y_ceil = np.clip(y_ceil, 0, h-1)
        lt = channel[y_floor, x_floor]
        rt = channel[y_floor, x_ceil]
        lb = channel[y_ceil, x_floor]
        rb = channel[y_ceil, x_ceil]
        dx, dy = src_x - x_floor, src_y - y_floor
        result = lt * (1-dx) * (1-dy) + rt * dx * (1-dy) + lb * (1-dx) * dy + rb * dx * dy
        if is_gray:
            result = np.expand_dims(result, axis=-1)
        return np.squeeze(result)

    def resize_image(self, image, target_shape):
        h, w = image.shape[:2]
        target_h, target_w = target_shape[:2]
        scale_x, scale_y = w / target_w, h / target_h
        is_gray = len(image.shape) == 2
        num_channels = 1 if is_gray else image.shape[2]
        new_image_shape = (target_h, target_w, num_channels) if not is_gray else (target_h, target_w)
        new_image = np.zeros(new_image_shape, dtype=np.float32)
        y_indices, x_indices = np.indices((target_h, target_w))
        src_x, src_y = x_indices * scale_x, y_indices * scale_y
        if is_gray:
            new_image = self.antialiasing(src_x, src_y, image)
            new_image = new_image[:, :, np.newaxis]
        else:
            for c in range(num_channels):
                new_image[..., c] = self.antialiasing(src_x, src_y, image[..., c])
        return new_image

    def rotate(self, image, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos, sin = np.cos(angle_rad), np.sin(angle_rad)
        h, w = image.shape[:2]
        is_gray = len(image.shape) == 2 or image.shape[-1] == 1
        num_channels = 1 if is_gray else image.shape[2]
        center_y, center_x = np.array([h, w]) // 2
        y_indices, x_indices = np.indices((h, w))
        coords = np.stack([x_indices - center_x, y_indices - center_y], axis=-1)
        new_coords = coords @ np.array([[cos, -sin], [sin, cos]])
        src_x = new_coords[..., 0] + center_x
        src_y = new_coords[..., 1] + center_y
        rotated_image = np.zeros((h, w, num_channels), dtype=np.float32)
        if is_gray:
            rotated_image[..., 0] = self.antialiasing(src_x, src_y, np.squeeze(image))
            rotated_image = np.expand_dims(rotated_image[..., 0], axis=-1)
        else:
            for c in range(num_channels):
                rotated_image[..., c] = self.antialiasing(src_x, src_y, image[..., c])
        return rotated_image

    def translate(self, image, dx, dy):
        h, w = image.shape[:2]
        num_channels = image.shape[2] if len(image.shape) > 2 else 1
        new_image_shape = (h, w, num_channels) if num_channels > 1 else (h, w)
        translated_image = np.zeros(new_image_shape, dtype=np.float32)
        y_indices, x_indices = np.indices((h, w))
        src_x, src_y = x_indices - dx, y_indices - dy
        if num_channels == 1:
            translated_image = self.antialiasing(src_x, src_y, image)
        else:
            for c in range(num_channels):
                translated_image[..., c] = self.antialiasing(src_x, src_y, image[..., c])
        return translated_image

    def random_crop(self, image, crop_size):
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        return image[top:top + crop_h, left:left + crop_w]

    def flip(self, image, horizontal=False, vertical=False):
        if horizontal:
            image = np.fliplr(image)
        if vertical:
            image = np.flipud(image)
        return image

    def adjust_brightness(self, image, factor):
        return np.clip(image * factor, 0, 1)

    def add_gaussian_noise(self, image, mean=0, std=0.1):
        return np.clip(image + np.random.normal(mean, std, image.shape), 0, 1)

    def add_salt_pepper_noise(self, image, salt_prob=0.05, pepper_prob=0.05):
        noise = np.random.choice([0, 1, 2], size=image.shape[:2], p=[salt_prob, pepper_prob, 1-salt_prob-pepper_prob])
        noisy_image = np.copy(image)
        noisy_image[noise == 0] = 0
        noisy_image[noise == 1] = 1
        return noisy_image

    def add_poisson_noise(self, image, lam=0.1):
        return np.clip(image + np.random.poisson(lam, image.shape) * 0.01, 0, 1)

    def jitter(self, image, sigma=1.0):
        h, w = image.shape[:2]
        num_channels = image.shape[2] if len(image.shape) > 2 else 1
        jittered_image = np.zeros_like(image, dtype=np.float32)
        x_offsets = np.random.normal(0, sigma, (h, w))
        y_offsets = np.random.normal(0, sigma, (h, w))
        y_indices, x_indices = np.indices((h, w))
        src_x, src_y = x_indices + x_offsets, y_indices + y_offsets
        if num_channels == 1:
            jittered_image = self.antialiasing(src_x, src_y, image)
        else:
            for c in range(num_channels):
                jittered_image[..., c] = self.antialiasing(src_x, src_y, image[..., c])
        return jittered_image

    def channel_shuffle(self, image):
        if len(image.shape) < 3 or image.shape[2] < 2:
            return image
        channel_indices = np.arange(image.shape[2])
        self.random_state.shuffle(channel_indices)
        return image[..., channel_indices]

    def augment(self, images, target_shape=(28, 28)):
        augmented_images = []
        for image in images:
            for aug_fn, prob in self.augmentations:
                if self.random_state.rand() < prob:
                    image = aug_fn(self, image)
            image = self.resize_image(image, target_shape)
            augmented_images.append(image)
        return np.stack(augmented_images, axis=0)
