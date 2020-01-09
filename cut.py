from PIL import Image

def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image

image_list = [
    "train2014/COCO_train2014_000000004571.jpg",
    "train2014/COCO_train2014_000000005169.jpg",
    "train2014/COCO_train2014_000000006631.jpg",
    "train2014/COCO_train2014_000000364913.jpg",
    "train2014/COCO_train2014_000000367375.jpg",
    "train2014/COCO_train2014_000000370760.jpg"
]

for image_name in image_list:
    image = Image.open(image_name)
    image = center_crop(image, 512, 512)
    image.save(image_name.split("/")[-1])