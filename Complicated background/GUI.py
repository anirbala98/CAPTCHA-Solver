from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
from solve_captcha import solve
root = Tk()
root.title("CAPTCHA_SOLVER")
root.geometry("500x500")
frame = Frame(root)
frame.pack()
path = ""
def details():

    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\Balachandar\\Pictures", title="Select a file")
    #my_label = Label(root, text=root.filename).pack()
    path = root.filename
    
    img = ImageTk.PhotoImage(Image.open(root.filename))
    panel = Label(root, image = img)
    panel.image = img
    panel.configure(image=img)
    panel.pack(side = "top", fill = "both", expand = "no")
    L1 = Label(frame, text = "Input image", font = 14)
    L1.pack( side = BOTTOM)
    
    solvec(path)

def solvec(path):
    text1 = solve(path)
    L4 = Label(frame, text = "Output: {}".format(text1), font = 14)
    L4.pack( side = BOTTOM)

bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM)

L2 = Label(frame, text="CAPTCHA SOLVER", font = 18)
L2.pack(side = TOP)
L5 = Label(frame, text=" ", font = 18)
L5.pack(side = TOP)

L3 = Label(frame, text="Select a CAPTCHA image", font = 14)
L3.pack (side = TOP)
L6 = Label(frame, text=" ", font = 18)
L6.pack(side = TOP)
L8 = Label(frame, text=" ", font = 18)
L8.pack(side = TOP)
B1 = Button(bottomframe, text ="Select", command = details)

B1.pack()
