from os import listdir
from os.path import join

import torch.utils.data as data

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.a2_path = join(image_dir, "a2")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.image_filenames2 = [x for x in listdir(self.a2_path) if is_image_file(x)]

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.a_path, self.image_filenames[index]))
        input2 = load_img(join(self.a2_path, self.image_filenames2[index]))
        target = load_img(join(self.b_path, self.image_filenames[index]))

        filename=self.image_filenames[index]

        return input, input2, target, filename

    def __len__(self):
        return len(self.image_filenames)
