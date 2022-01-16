from PIL import Image
def convertImg(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")

    datas = img.getdata()

    newData = []

    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255,255, 255, 0))
        else:
            newData.append(item)


    img.putdata(newData)
    img.save("./logo.png", "PNG")


convertImg(r"C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\static\assets\images\logo.png")