import os, zlib, numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import sys

class CVNDataset(Dataset):
    """
    ViT transformer expect 3 channels RGB-like, the input should fit this scheme
    DUNE CVN pixel-maps as 3-channel images (U,V,Z) from .gz files.
    Also we can do a single plane in 3 channels so it would be training on 1 plane instead of 3
    Returns (image, dummy_target). DINOv2 will apply its own multi-crop transform.
    """
    def __init__(self, root,
                 split=None,
                 swap_axes=False,
                 transform=None,
                 target_transform=None,
                 **kwargs):
        self.root = root
        self.split = split
        self.swap_axes = bool(swap_axes)
        self.transform = transform
        self.target_transform = target_transform

        # -------------------------------
        # Parse optional flags from :extra=
        # -------------------------------
        # Accept comma-separated K=V pairs, e.g. extra="plane=Z,mono=mono3,swap_axes=1"
        self.plane = None       # 'U','V','Z','0','1','2' -> choose single plane; None -> use all 3
        self.mono = "mono1"     # 'mono3' (replicate single plane to 3ch) | 'mono1' (true 1ch 'L' image), mono1 is actually not implemented, it requires changing the model which I think will be done anyway, so just keep config for now
        extra = kwargs.get("extra", None)
        if isinstance(extra, str):
            for token in extra.split(","):
                if not token.strip():
                    continue
                k, _, v = token.partition("=")
                k = k.strip().lower()
                v = v.strip()
                if k == "plane":
                    self.plane = v.upper()  # allow 'U','V','Z' or '0','1','2'
                elif k == "mono":
                    self.mono = v.lower()   # 'mono3' or 'mono1'
                elif k == "swap_axes":
                    self.swap_axes = v.lower() in ("1", "true", "t", "yes", "y")

        # -------------------------------
        # Index all available events
        # -------------------------------
        # cand_flavs = ["nue", "numu", "NC", "nutau", "nuecc", "numucc", "nutaucc"]
        # self.entries = []
        # for flav in cand_flavs:
        #     d = os.path.join(root, flav)
        #     if not os.path.isdir(d):
        #         continue
        #     for gz in sorted(glob(os.path.join(d, "event*.gz"))):
        #         key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
        #         self.entries.append((flav, key, gz))
        cand_flavs = ["nu", "nue", "nc"]
        self.entries = []
        for flav in cand_flavs:
            folder_name = f'prodgenie_dunevd_1x8x6_{flav}/cvn_gaushit'
            d = os.path.join(root, folder_name)
            if not os.path.isdir(d):
                continue
            for dd in os.listdir(d):
                subdir = os.path.join(d, dd)
                if not os.path.isdir(subdir):
                    continue
                # print(subdir)
                for gz in sorted(glob(os.path.join(subdir, "event*.gz"))):
                    key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
                    self.entries.append((flav, key, gz))
        # print("=============================== LENGTH OF DATASET : -------========" )
        # print(len(self.entries))
        # print(self.entries[0])
        # print('================================')
        # print('================================')
        # sys.exit()

        if not self.entries:
            raise RuntimeError(f"No .gz files found under {root}/<flavor>/event*.gz")

    def __len__(self):
        #  if (len(self.entries) % 64 != 0):
        #      return len(self.entries) - (len(self.entries) % 64)
        return len(self.entries)

    def _read_array(self, gz_path):
        with open(gz_path, "rb") as f:
            arr = np.frombuffer(bytearray(zlib.decompress(f.read())), dtype=np.uint8).reshape(3, 500, 500)
        if self.swap_axes:
            # swap wire/time -> (3, W, T) -> (3, T, W)
            arr = arr.transpose(0, 2, 1)
        return arr  # (3, H, W)

    def _to_pil(self, arr3):
        """
        Convert a (3,H,W) numpy uint8 to PIL according to options:
          - plane=None: return RGB-like from [U,V,Z] (3ch)
          - plane=..., mono=mono3: replicate selected plane -> 3ch RGB-like
          - plane=..., mono=mono1: return single-channel 'L' image (requires in_chans=1 model)
        """
        if self.plane is None:
            img = np.moveaxis(arr3, 0, -1)                     # (H,W,3)
            return Image.fromarray(img, mode="RGB")
        # choose one plane
        idx_map = {"U": 0, "V": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        idx = idx_map[self.plane]
        plane = arr3[idx]                                      # (H,W)
        if self.mono == "mono1":
            return Image.fromarray(plane, mode="L")            # true 1-channel
        # default: replicate into 3 channels (Option A)
        img = np.repeat(plane[..., None], 3, axis=2)           # (H,W,3)
        return Image.fromarray(img, mode="RGB")

    def __getitem__(self, idx):
        _, _, gz = self.entries[idx]
        arr = self._read_array(gz)
        img = self._to_pil(arr)
        target = 0  # dummy label for SSL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


