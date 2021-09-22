import albumentations as A
    
# Augmentation list. Here, we use the Albumentations library
# IMPORTANT: You MUST add A.Normalize(...) in the list.
# Also, you don't need to add Resizer in the end
# because it has been implemented inside
# CocoAlbumentationsDataset class.
# Efficient-Det input image sizes : [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
def Albumentations(params):
    augmentation = A.Compose([
        A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
        A.OneOf([A.CLAHE(p=1.), A.RandomBrightnessContrast(0.15, 0.15, p=1.),A.RandomGamma(p=1.), A.HueSaturationValue(p=1.)], p=0.8),
        A.OneOf([A.Blur(4, p=1.), A.MotionBlur(5, p=1.), A.Sharpen(p=1.)], p=0.75),
        A.Normalize(mean=params.mean, std=params.std, max_pixel_value=255., always_apply=True)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.2))
    
    return augmentation