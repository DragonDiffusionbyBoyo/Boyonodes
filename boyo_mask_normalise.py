"""
BoyoMaskToImage - Converts a mask tensor [b,h,w] to an image tensor [b,h,w,3]
for feeding into DiffuEraser's video_mask_image input, bypassing its broken
len(video_mask)>3 check which fires on any video longer than 3 frames.
"""


class BoyoMaskToImage:

    CATEGORY = "BoyoNodes/Masks"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    def convert(self, mask):
        original_shape = tuple(mask.shape)

        # Normalise to [b,h,w] first
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)                        # [h,w] -> [1,h,w]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)                         # [b,h,w,1] -> [b,h,w]
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)                          # [b,1,h,w] -> [b,h,w]

        # [b,h,w] -> [b,h,w,1] -> [b,h,w,3]
        image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)

        # Ensure contiguous memory layout after expand
        image = image.contiguous()

        print(f"[BoyoMaskToImage] mask {original_shape} -> image {tuple(image.shape)}")
        return (image,)


NODE_CLASS_MAPPINGS = {
    "BoyoMaskToImage": BoyoMaskToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoMaskToImage": "Boyo Mask To Image (DiffuEraser Fix)",
}
