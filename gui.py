from tkinter import *
from PIL import ImageTk,Image,ImageFont,ImageDraw
from tkinter import filedialog
from prediction import imgPrediction

img = Image.open("example.jpg")

print(img.size)

def image_text(img,text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial", 24)
    draw.text((img.size[0]/2 - 50, img .size[1]-50),text,(255,255,255),font=font)
    img.save('sample-out.jpg')

root = Tk()
root.title('Images')

def open():
    global my_img
    root.filename = filedialog.askopenfilename(initialdir='/fatiheminoge/Desktop',title='Select A File',filetypes=(('all files','*.*'),
    ('png files','*.png'),('jpg files','*.jpg'),('jpeg files','*.jpeg')))
    img = Image.open(root.filename)
    description = imgPrediction(root.filename)
    image_text(img,description)
    my_img = ImageTk.PhotoImage(Image.open('sample-out.jpg'))
    my_img_label = Label(image=my_img).grid(row=1,column=0)


my_btn = Button(root,text='Open File',command=open).grid(row=0,column=0)

button_quit = Button(root,text='Exit Program',command=root.quit).grid(row=3,column=0)

root.mainloop()
