import tkinter as tk
from enum import Flag, auto
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt

ASSETS_DIR = Path(__file__).parent / 'assets'  # NOTE: Switch to importlib.resources later


class BloodComponent(Flag):
    ANTI_A = auto()
    ANTI_B = auto()
    PLUS = auto()

q = 1
# f = 0
# v = 0
p1 = ''
p2 = ''
p3 = ''
p4 = ''


class Login(tk.Frame):
    def __init__(self, master: tk.Tk | None = None) -> None:
        super().__init__(master)
        self.blood: BloodComponent | None = BloodComponent(0)
        self.anti_a_img: Path | None = None
        self.anti_b_img: Path | None = None
        self.anti_d_img: Path | None = None
        self.reagent_img: Path | None = None
        self.init_window()

    def init_window(self):
        self.configure(background='powder blue')
        self.pack(fill=tk.BOTH, expand=1)
        self.master.title("Blood Group Detection System")
        self.master.iconbitmap(ASSETS_DIR / 'Blood.ico')

        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        file = tk.Menu(menu)
        stage = tk.Menu(menu)
        file.add_cascade(label="Process", menu=stage)
        file.add_command(label="Restart",command=self.restart)
        file.add_command(label="Exit",command=self.quit)
        menu.add_cascade(label="File", menu=file)

        stage.add_command(label="Process 1: Green Plane Extraction", command=self.gp)
        stage.add_command(label="Process 2: Auto Threshold", command=self.autothresh)
        stage.add_command(label="Process 3: Adaptive Threshold:Ni Black", command=self.Adapthresh)
        stage.add_command(label="Process 4: Morphology: Fill Holes", command=self.Fill_holes)
        stage.add_command(label="Process 5: Advanced Morphology: Remove small objects", command=self.Remove_small_objects)
        stage.add_command(label="Process 6: Histogram", command=self.Histogram)
        stage.add_command(label="Process 7: Quantification", command=self.HSV_Luminance)

        l1 = tk.Label(self, text="Reagent Anti-A", font=("Helvetica", 12))
        l2 = tk.Label(self, text="Reagent Anti-B", font=("Helvetica", 12))
        l3 = tk.Label(self, text="Reagent Anti-D", font=("Helvetica", 12))
        l4 = tk.Label(self, text="Control Reagent", font=("Helvetica", 12))
        l1.place(x=160, y=475)
        l2.place(x=480, y=475)
        l3.place(x=780, y=475)
        l4.place(x=1070, y=475)

        e1 = tk.Button(self, text="Choose Image", command=self.select_anti_a_image)
        e2 = tk.Button(self, text="Choose Image", command=self.select_anti_b_image)
        e3 = tk.Button(self, text="Choose Image", command=self.select_anti_d_image)
        e4 = tk.Button(self, text="Choose Image", command=self.select_control_reagent_image)
        self.ep = tk.Button(self, text="Process", font=("Helvetica", 12), fg='red', relief=tk.SUNKEN)
        self.ep.place(x=650, y=575)
        e1.place(x=170, y=500)
        e2.place(x=490, y=500)
        e3.place(x=790, y=500)
        e4.place(x=1080, y=500)

    def quit(self):
        global q
        q = 0
        root.destroy()

    def restart(self):
        global q
        q = 1
        root.destroy()

    def message(self,q):
        messagebox.showinfo("Result",q+"Confirmed")

    def select_image(self, label_coordinates: tuple[int, int]) -> Path:
        path = Path(filedialog.askopenfilename())
        with Image.open(path) as picture:
            resized = picture.resize((300, 425), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(resized)
            label = tk.Label(self, image=image)  # type: ignore[arg-type]
            x, y = label_coordinates
            label.place(x=x, y=y)

        return path
    
    def select_anti_a_image(self) -> None:
        self.anti_a_img = self.select_image(label_coordinates=(75, 50))
        if not any(path is None for path in (self.anti_a_img, self.anti_b_img, self.anti_d_img, self.reagent_img)):
            # NOTE: Work on this
            self.ep.configure(relief=tk.RAISED, fg='green', command=self.start1)
    
    def select_anti_b_image(self) -> None:
        self.anti_b_img = self.select_image(label_coordinates=(375, 50))
        if not any(path is None for path in (self.anti_a_img, self.anti_b_img, self.anti_d_img, self.reagent_img)):
            # NOTE: Work on this
            self.ep.configure(relief=tk.RAISED, fg='green', command=self.start1)
    
    def select_anti_d_image(self) -> None:
        self.anti_d_img = self.select_image(label_coordinates=(675, 50))
        if not any(path is None for path in (self.anti_a_img, self.anti_b_img, self.anti_d_img, self.reagent_img)):
            # NOTE: Work on this
            self.ep.configure(relief=tk.RAISED, fg='green', command=self.start1)
    
    def select_control_reagent_image(self) -> None:
        self.reagent_img = self.select_image(label_coordinates=(975, 50))
        if not any(path is None for path in (self.anti_a_img, self.anti_b_img, self.anti_d_img, self.reagent_img)):
            # NOTE: Work on this
            self.ep.configure(relief=tk.RAISED, fg='green', command=self.start1)

    def start1(self):
        self.start(p1,"Anti A")
        self.start2()

    def start2(self):
        self.start(p2, "Anti B")
        self.start3()

    def start3(self):
        self.start(p3, "Anti D")
        self.start4()

    def start4(self):
        self.start(p4, "Control")
        self.check()

    def extract_green_plane(self, p,r):  # Extracting the Green plane
        img = cv2.imread(p)
        gi = img[:, :, 1]
        cv2.imwrite("p1"+r+".png", gi)
        return gi

    def obtain_threshold(self, p,r):  # Obtaining the threshold
        gi = self.extract_green_plane(p,r)
        _, th = cv2.threshold(gi, 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite("p2"+r+".png", th)

    def obtain_black_image(self, p,r):  # Obtaining Ni black image
        img = cv2.imread('p2'+r+'.png', 0)
        th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 14)
        cv2.imwrite("p3"+r+".png", th4)

    def fill_holes(self,r):  # Morphology: fill holes
        gi = cv2.imread('p3'+r+'.png', cv2.IMREAD_GRAYSCALE)
        _, gi_th = cv2.threshold(gi, 220, 255, cv2.THRESH_BINARY_INV)
        gi_floodFill=gi_th.copy()
        h, w = gi_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(gi_floodFill, mask, (0, 0), 255)
        gi_floodFill_inv = cv2.bitwise_not(gi_floodFill)
        gi_out = gi_th | gi_floodFill_inv
        cv2.imwrite('p4'+r+'.png', gi_out)

    def eliminate_small_objects(self,r):  # Morphing To eliminate small objects
        img = cv2.imread('p4'+r+'.png')
        kernel = np.ones((5, 5), np.uint8)
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('p5'+r+'.png', close)

    def histogram(self,r):  #Histogram
        img = cv2.imread('p5'+r+'.png', 0)
        img2 = cv2.imread('p1'+r+'.png', 0)
        mask = np.ones(img.shape[:2], np.uint8)
        hist = cv2.calcHist([img2], [0], mask, [256], [0, 256])
        min = 1000
        max = 0
        n = 0
        s = 0
        ss = 0
        for y in hist:
            if y > max:
                max = y
            if y < min:
                min = y
            s += y
            n += 1

        mean = s/n
        for y in hist:
            ss += (y-mean)**2
        ss /= n
        sd = abs(ss)**0.5
        print(r,"-",sd,"\n")
        return sd < 580


    def start(self, p,r):
        self.extract_green_plane(p,r)
        self.obtain_threshold(p,r)
        self.obtain_black_image(p,r)
        self.fill_holes(r)
        self.eliminate_small_objects(r)
        a = self.histogram(r)
        print(a," - ",r)
        if a == 1:
            blood = BloodComponent(0)
            if r == "Anti A":
                blood |= BloodComponent.ANTI_A
            elif r == "Anti B":
                blood |=  BloodComponent.ANTI_B
            elif r == "Anti D":
                blood |= BloodComponent.PLUS
            elif r == "Control":
                blood = None
            self.blood = blood

    def check(self) -> None:
        if self.blood is None:
            self.message("Invalid")
            return
        
        message = ""

        if self.blood & BloodComponent.ANTI_A:
            message += 'A'
        if self.blood & BloodComponent.ANTI_B:
            message += 'B'
        if not self.blood & BloodComponent.ANTI_A and not self.blood & BloodComponent.ANTI_B:
            message += 'O'

        message += '+' if self.blood & BloodComponent.PLUS else '-'

        self.message(message)


    def gp(self) -> None:
        im1 = cv2.imread('p1Anti A.png')
        cv2.imshow('Anti-A',im1)
        im2 = cv2.imread('p1Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p1Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p1Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def autothresh(self) -> None:
        im1 = cv2.imread('p2Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p2Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p2Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p2Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Adapthresh(self):
        im1 = cv2.imread('p3Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p3Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p3Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p3Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Fill_holes(self):
        im1 = cv2.imread('p4Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p4Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p4Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p4Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Remove_small_objects(self):
        im1 = cv2.imread('p5Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p5Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p5Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p5Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Histogram(self):
        img1 = cv2.imread('p5Anti A.png', 0)
        img2 = cv2.imread('p5Anti B.png', 0)
        img3 = cv2.imread('p5Anti D.png', 0)
        img4 = cv2.imread('p5Control.png', 0)
        plt.hist(img1.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img2.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img3.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img4.ravel(), 256, [0, 256])
        plt.show()

    def HSV_Luminance(self):
        img1 = cv2.imread(p1)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv1, 0)

        img2 = cv2.imread(p2)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv2, 0)

        img3 = cv2.imread(p3)
        hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv3, 0)

        img4 = cv2.imread(p4)
        hsv4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv4, 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def stp_full(self,event=None):
        root.attributes("-fullscreen", False)
        root.geometry("1020x720")



if __name__ == '__main__':
    while q == 0:
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        app = Login(root)
        root.bind("<Escape>", app.stp_full)
        root.mainloop()
