import tkinter
from tkinter import *
from tkinter import filedialog
from display import *

import cv2

gui = Tk()
gui.title('Insights-Mask-Off GUI')
gui.resizable(False, False)


def introduction():
    intro_text = "Motivated by the inconvenience of not being able to use facial recognition software to unlock our phones while wearing masks, we set out to build a model that could accurately recognize one's face while partially obstructed by a mask. Namely, this required recognizing the upper portion of the subject's face while everything below the nose and along the chin was covered. To do so, we preprocessed a dataset of 13,000 people and correctly drew masks on all of them (coloring over their nose, mouth, and along their jawline with a solid blue). Then, we built and trained a Siamese Network using these matched pictures, ultimately leading to our final product, which will allow the user to select two images of the dataset and determine whether or not the two images portray the same person."
    introduction = Toplevel(gui)
    introduction.title("Introduction")
    introduction.geometry("450x500")
    label = Label(introduction, text=intro_text,
                  wraplength=400, justify="left")
    label.pack()


def capture():
  gui.filename1 = filedialog.askopenfilename(initialdir = "/insights-mask_off",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
  gui.filename2 = filedialog.askopenfilename(initialdir = "/insights-mask_off",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
  script = open("display.py")
  a_script = script.read()
  sys.argv = ["display.py", gui.filename1, gui.filename2]
  exec(a_script)
  script.close()

canvas = Canvas(gui, width=600, height=600)
img = PhotoImage(file="/home/datavis_1/insights-mask_off/data/Aaron_Eckhart_0001.png")
img2 = PhotoImage(file="/home/datavis_1/insights-mask_off/data/Aaron_Eckhart_0001_m.png")
# img3 = PhotoImage(file="croppedlips.png")

canvas.create_image(150, 300, image=img, anchor="center")

canvas.create_image(450, 300, image=img2, anchor="center")
# # canvas.create_image(190, 280, image=img3, anchor="se")
canvas.pack()
pixelVirtual = PhotoImage(width=1, height=1)

introduction_button = Button(gui, anchor=tkinter.CENTER, text="Intro", foreground='black', image=pixelVirtual,
                             width=75, height=30, compound="c", command=introduction)
introduction_button.place(x=50, y=50)

# crop_lips_button = Button(gui, anchor=tkinter.CENTER, text="Crop Lips", foreground='blue', image=pixelVirtual,
#                           width=75, height=30, compound="c", command=crop_lips)
# crop_lips_button.place(x=133, y=300)

capture_button = Button(gui, anchor=tkinter.CENTER, text="Capture", foreground='purple', image=pixelVirtual,
                        width=75, height=30, compound="c", command=capture)
capture_button.place(x=450, y=50)

# future_button = Button(gui, anchor=tkinter.CENTER, text="Next Steps", foreground='red', image=pixelVirtual,
#                        width=75, height=30, compound="c", command=future)
# future_button.place(x=333, y=300)


gui.geometry("600x600")
gui.mainloop()