from PIL import Image

image = Image.open(r'C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\static\assets\images\default.png')
new_image = image.resize((299, 299))
new_image = new_image.convert('RGB')
new_image.save('default.png')