import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=(1,))
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        padding = ((self.padding, self.padding), (self.padding, self.padding), (0,0))
        img_padded = np.pad(img, padding, 'constant', constant_values=0)
        x_from = self.padding + shift_x
        x_to = x_from + img.shape[0]
        y_from = self.padding + shift_y
        y_to = y_from + img.shape[1]
        return img_padded[x_from:x_to, y_from:y_to, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_idx = 0
        if self.shuffle:
            ordering = np.arange(len(self.dataset))
            np.random.shuffle(ordering)
            batch_ranges = range(self.batch_size, len(self.dataset), self.batch_size)
            self.ordering = np.array_split(ordering, batch_ranges)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx < len(self.ordering):
            idx = self.ordering[self.batch_idx]
            self.batch_idx += 1
            result = self.dataset[idx]
            result = tuple([Tensor(x) for x in result])
            return result
        else:
            raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        import gzip, struct
        with gzip.open(image_filename, 'rb') as f:
            magic, size, nrows, ncols = struct.unpack(">IIII", f.read(16))
            image_data = np.frombuffer(f.read(),
                                    dtype=np.dtype(np.uint8).newbyteorder('>'))
            image_data = image_data.reshape((size, nrows, ncols))

        image_data = image_data.reshape(size, -1).astype(np.float32)
        image_data = (image_data - image_data.min()) / (image_data.max() -
                                                        image_data.min())

        with gzip.open(label_filename, 'rb') as f:
            magic, nlabels = struct.unpack(">II", f.read(8))
            label_data = np.frombuffer(f.read(),
                                    dtype=np.dtype(np.uint8).newbyteorder('>'))
        self.X = image_data
        self.y = label_data
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.X[index]
        y = self.y[index]
        if isinstance(index, (slice, np.ndarray)):
            y = np.reshape(y, y.shape[0])
            X = np.reshape(X, (X.shape[0], 28, 28, 1))
            for idx, x in enumerate(X):
                X[idx] = self.apply_transforms(x)
        else:
            X = np.reshape(X, (28, 28, 1))
            X = self.apply_transforms(X)
        return (X, y)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
