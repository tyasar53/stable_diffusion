from PIL import ImageOps, Image, ImageChops


image, mask =   Image.open('dress-masked.jpg'), Image.open('dress.jpg')
alpha_mask  = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
image = image.convert("RGB")

image.show()
alpha_mask.show()