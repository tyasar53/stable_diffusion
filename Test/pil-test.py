from PIL import ImageOps, Image
import gr as gr



image = Image.open('dress.jpg')



#image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
image = image.split()[-1]
image.show()

image = ImageOps.invert(image)
image.show()

image = image.convert('L')
image.show()

image = image.point(lambda x: 255 if x > 0 else 0, mode='1')


image_m = ImageChops.lighter(image, image_m.convert('L')).convert('L')
image_m = image_m.convert("RGB")
image.show()



 elif mode == 2:  # inpaint
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        image = image.convert("RGB")

#alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
#mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
#image = image.convert("RGB")

#x_invert = ImageOps.invert(x)


#x_invert.show()

