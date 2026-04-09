import numpy as np
import pytest
from numpyml import ImageDataAugmentor


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


class TestAugmentor:
    def test_augment_preserves_shape(self):
        aug = ImageDataAugmentor()
        imgs = np.random.rand(5, 28, 28, 1).astype(np.float32)
        out = aug.augment(imgs)
        assert out.shape == imgs.shape

    def test_augment_infers_target_shape(self):
        aug = ImageDataAugmentor()
        imgs = np.random.rand(3, 32, 32, 3).astype(np.float32)
        out = aug.augment(imgs)
        assert out.shape == imgs.shape

    def test_gaussian_noise(self):
        aug = ImageDataAugmentor(augmentations=[
            (lambda self, img: self.add_gaussian_noise(img, std=0.1), 1.0),
        ])
        imgs = np.zeros((2, 8, 8, 1), dtype=np.float32)
        out = aug.augment(imgs)
        assert not np.allclose(out, 0)

    def test_rotate(self):
        aug = ImageDataAugmentor()
        img = np.random.rand(8, 8, 1).astype(np.float32)
        rotated = aug.rotate(img, 45)
        assert rotated.shape == img.shape

    def test_flip(self):
        aug = ImageDataAugmentor()
        img = np.arange(12).reshape(3, 4, 1).astype(np.float32)
        flipped = aug.flip(img, horizontal=True)
        np.testing.assert_array_equal(flipped[:, 0, 0], img[:, -1, 0])

    def test_random_crop(self):
        aug = ImageDataAugmentor()
        img = np.random.rand(28, 28, 1).astype(np.float32)
        cropped = aug.random_crop(img, (20, 20))
        assert cropped.shape == (20, 20, 1)

    def test_brightness(self):
        aug = ImageDataAugmentor()
        img = np.full((4, 4, 1), 0.5, dtype=np.float32)
        bright = aug.adjust_brightness(img, 2.0)
        assert bright.max() <= 1.0
        assert bright.min() >= 0.0

    def test_values_clipped(self):
        aug = ImageDataAugmentor(augmentations=[
            (lambda self, img: self.add_gaussian_noise(img, std=10.0), 1.0),
        ])
        imgs = np.random.rand(2, 8, 8, 1).astype(np.float32)
        out = aug.augment(imgs)
        assert out.min() >= 0
        assert out.max() <= 1
