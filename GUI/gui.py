from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

if __name__ == "__main__":
    win = Tk()
    win.title('image processing')
    # win.resizable(0, 0)
    win.geometry('600x400+100+100')



    # setting up a tkinter canvas with scrollbars
    frame = Frame(win, bd=2, relief=FLAT)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    # xscroll = Scrollbar(frame, orient=HORIZONTAL)
    # xscroll.grid(row=1, column=0, sticky=E + W)
    # yscroll = Scrollbar(frame)
    # yscroll.grid(row=0, column=1, sticky=N + S)
    # canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    tx_note = Label(frame, text="please choose the image that need to be processed:")
    tx_note.grid(row=0, column=0)
    canvas = Canvas(frame, bd=0)
    canvas.grid(row=1, column=0, sticky=N + S + E + W)
    # xscroll.config(command=canvas.xview)
    # yscroll.config(command=canvas.yview)



    # function to be called when mouse is clicked
    def printcoords():
        File = filedialog.askopenfilename(parent=win, initialdir="D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images", title='Choose an image.')
        filename = ImageTk.PhotoImage(Image.open(File))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0, 0, anchor='nw', image=filename)

    Button(frame, text='choose', command=printcoords).grid(row=0, column=1)
    frame.pack(fill=BOTH, expand=1)

    Button(win, text='sharpening').pack()

    win.mainloop()
