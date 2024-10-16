"""Blood group detection system."""

from __future__ import annotations

import tkinter as tk
from enum import Flag, auto
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from typing import Any, Never

    from numpy.typing import ArrayLike

ASSETS_DIR = Path(__file__).parent / 'assets'  # NOTE: Switch to importlib.resources later
USE_FULLSCREEN = True
LABEL_FONT = ("Helvetica", 12)
LABELS = [
    ("Reagent Anti-A", 160, 475),
    ("Reagent Anti-B", 480, 475),
    ("Reagent Anti-D", 780, 475),
    ("Control Reagent", 1070, 475),
]


class BloodComponent(Flag):
    """Blood components."""

    ANTI_A = auto()
    ANTI_B = auto()
    PLUS = auto()

class RestartProgram(Exception):  # noqa: N818
    """Restart the program."""


class Main(tk.Tk):
    """Main application."""

    def __init__(self) -> None:
        """Init window."""
        super().__init__()
        self.title("Blood Group Detection System")
        self.iconbitmap(ASSETS_DIR / 'Blood.ico')
        self.attributes("-fullscreen", USE_FULLSCREEN)
        self.exit_application = False
        self.app = Login(self)
        self.bind("<Escape>", self.exit_fullscreen)
        self.create_menu()

    def create_menu(self) -> None:
        """Create menu."""
        menu = tk.Menu(self)
        self.config(menu=menu)

        file = tk.Menu(menu)
        stage = tk.Menu(menu)
        file.add_cascade(label="Process", menu=stage)
        file.add_command(label="Restart",command=self.restart_program)
        file.add_command(label="Exit",command=self.end_program)
        menu.add_cascade(label="File", menu=file)

        stage.add_command(label="Process 1: Green Plane Extraction", command=gp)
        stage.add_command(label="Process 2: Auto Threshold", command=autothresh)
        stage.add_command(label="Process 3: Adaptive Threshold:Ni Black", command=adapthresh)
        stage.add_command(label="Process 4: Morphology: Fill Holes", command=fill_holes)
        stage.add_command(label="Process 5: Advanced Morphology: Remove small objects", command=remove_small_objects)
        stage.add_command(label="Process 6: Histogram", command=histogram)
        stage.add_command(label="Process 7: Quantification", command=self.app.hsv_luminance)

    def exit_fullscreen(self, _event: Any = None) -> None:  # noqa: ANN401
        """Exit fullscreen mode."""
        self.attributes("-fullscreen", False)  # noqa: FBT003
        self.geometry("1020x720")

    def restart_program(self) -> Never:
        """Tell the main loop to restart the program."""
        raise RestartProgram

    def end_program(self) -> Never:
        """Tell the main loop to end the program."""
        raise SystemExit


class Login(tk.Frame):
    """Lorem Ipsum."""

    def __init__(self, master: tk.Tk | None = None) -> None:
        """Lorem Ipsum."""
        super().__init__(master)
        self.blood: BloodComponent | None = BloodComponent(0)
        self.anti_a_img: Path | None = None
        self.anti_b_img: Path | None = None
        self.anti_d_img: Path | None = None
        self.reagent_img: Path | None = None
        self.init_window()

    def init_window(self) -> None:
        """Add menu entries."""
        self.configure(background='powder blue')
        self.pack(fill=tk.BOTH, expand=1)

        for label_text, x, y in LABELS:
            label = tk.Label(self, text=label_text, font=LABEL_FONT)
            label.place(x=x, y=y)

        image_buttons = [
            (170, 500, self.select_anti_a_image),
            (490, 500, self.select_anti_b_image),
            (790, 500, self.select_anti_d_image),
            (1080, 500, self.select_control_reagent_image),
        ]

        for x, y, command in image_buttons:
            button = tk.Button(self, text="Choose Image", command=command)
            button.place(x=x, y=y)

        self.process_button = tk.Button(self, text="Process", font=LABEL_FONT, fg='red', relief=tk.SUNKEN)
        self.process_button.place(x=650, y=575)

    def message(self, text: str) -> None:
        """Create a message box with info."""
        messagebox.showinfo(f"Result {text} Confirmed")

    def select_image(self, label_coordinates: tuple[int, int]) -> Path:
        """Choose an image file from the filesystem."""
        path = Path(filedialog.askopenfilename())
        with Image.open(path) as picture:
            resized = picture.resize((300, 425), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(resized)
            label = tk.Label(self, image=image)  # type: ignore[arg-type]
            x, y = label_coordinates
            label.place(x=x, y=y)

        return path

    def verify_all_images_chosen(self) -> None:
        """Check all images have been chosen."""
        if all(path is not None
            for path in (self.anti_a_img, self.anti_b_img, self.anti_d_img, self.reagent_img)):
            self.process_button.configure(relief=tk.RAISED, fg='green', command=self.process_images)

    def select_anti_a_image(self) -> None:
        """Select image for A."""
        self.anti_a_img = self.select_image(label_coordinates=(75, 50))
        self.verify_all_images_chosen()

    def select_anti_b_image(self) -> None:
        """Select image for B."""
        self.anti_b_img = self.select_image(label_coordinates=(375, 50))
        self.verify_all_images_chosen()

    def select_anti_d_image(self) -> None:
        """Select image for D."""
        self.anti_d_img = self.select_image(label_coordinates=(675, 50))
        self.verify_all_images_chosen()

    def select_control_reagent_image(self) -> None:
        """Select image for reagent."""
        self.reagent_img = self.select_image(label_coordinates=(975, 50))
        self.verify_all_images_chosen()

    def process_images(self) -> None:
        """Lorem Ipsum."""
        self.start(self.anti_a_img, "Anti A")  # type: ignore[arg-type]
        self.start(self.anti_b_img, "Anti B")  # type: ignore[arg-type]
        self.start(self.anti_d_img, "Anti D")  # type: ignore[arg-type]
        self.start(self.reagent_img, "Control")  # type: ignore[arg-type]
        self.check()

    def extract_green_plane(self, image_path: Path, experiment_title: str) -> ArrayLike:  # Extracting the Green plane
        """Lorem Ipsum."""
        img = cv2.imread(image_path)  # type: ignore[call-overload]
        gi = img[:, :, 1]
        cv2.imwrite(f"p1{experiment_title}.png", gi)
        return gi

    def obtain_threshold(self, image_path: Path, experiment_title: str) -> None:  # Obtaining the threshold
        """Lorem Ipsum."""
        gi = self.extract_green_plane(image_path, experiment_title)
        _, th = cv2.threshold(gi, 0, 255, cv2.THRESH_OTSU)  # type: ignore[call-overload]
        cv2.imwrite(f"p2{experiment_title}.png", th)

    def obtain_black_image(self, experiment_title: str) -> None:  # Obtaining Ni black image
        """Lorem Ipsum."""
        img = cv2.imread(f'p2{experiment_title}.png', 0)
        th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 14)
        cv2.imwrite(f"p3{experiment_title}.png", th4)

    def fill_holes(self, experiment_title: str) -> None:  # Morphology: fill holes
        """Lorem Ipsum."""
        gi = cv2.imread(f'p3{experiment_title}.png', cv2.IMREAD_GRAYSCALE)
        _, gi_th = cv2.threshold(gi, 220, 255, cv2.THRESH_BINARY_INV)
        gi_flood_fill = gi_th.copy()
        h, w = gi_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(gi_flood_fill, mask, (0, 0), 255)  # type: ignore[call-overload]
        gi_flood_fill_inv = cv2.bitwise_not(gi_flood_fill)
        gi_out = gi_th | gi_flood_fill_inv  # type: ignore[operator]
        cv2.imwrite(f'p4{experiment_title}.png', gi_out)

    def eliminate_small_objects(self, experiment_title: str) -> None:  # Morphing To eliminate small objects
        """Lorem Ipsum."""
        img = cv2.imread(f'p4{experiment_title}.png')
        kernel = np.ones((5, 5), np.uint8)
        open_ = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        close = cv2.morphologyEx(open_, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'p5{experiment_title}.png', close)

    def histogram(self, experiment_title: str) -> bool:
        """Histogram."""
        img = cv2.imread(f'p5{experiment_title}.png', 0)
        img2 = cv2.imread(f'p1{experiment_title}.png', 0)
        mask = np.ones(img.shape[:2], np.uint8)
        hist = cv2.calcHist([img2], [0], mask, [256], [0, 256])
        n = len(hist)
        s = sum(hist)
        mean = s / n
        ss = sum((value - mean) ** 2 for value in hist) / n

        standard_deviation = abs(ss) ** 0.5

        print(f"{experiment_title}-{standard_deviation}\n")  # noqa: T201
        return standard_deviation < 580  # noqa: PLR2004


    def start(self, image_path: Path, experiment_title: str) -> None:
        """Lorem Ipsum."""
        self.extract_green_plane(image_path, experiment_title)
        self.obtain_threshold(image_path, experiment_title)
        self.obtain_black_image(experiment_title)
        self.fill_holes(experiment_title)
        self.eliminate_small_objects(experiment_title)
        lower_deviation = self.histogram(experiment_title)
        print(f"{lower_deviation} - {experiment_title}")  # noqa: T201
        if lower_deviation:
            blood: BloodComponent = BloodComponent(0)
            if experiment_title == "Anti A":
                blood |= BloodComponent.ANTI_A
            elif experiment_title == "Anti B":
                blood |=  BloodComponent.ANTI_B
            elif experiment_title == "Anti D":
                blood |= BloodComponent.PLUS
            elif experiment_title == "Control":
                self.blood = None
                return
            self.blood = blood

    def check(self) -> None:
        """Check blood type."""
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

    def hsv_luminance(self) -> None:
        """I don't know what this does."""
        img1 = cv2.imread(self.anti_a_img)  # type: ignore[call-overload]
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv1, 0)  # type: ignore[call-overload]

        img2 = cv2.imread(self.anti_b_img)  # type: ignore[call-overload]
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv2, 0)  # type: ignore[call-overload]

        img3 = cv2.imread(self.anti_d_img)  # type: ignore[call-overload]
        hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv3, 0)  # type: ignore[call-overload]

        img4 = cv2.imread(self.reagent_img)  # type: ignore[call-overload]
        hsv4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv4, 0)  # type: ignore[call-overload]

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def gp() -> None:
    """I don't know what this does."""
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

def autothresh() -> None:
    """I don't know what this does."""
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

def adapthresh() -> None:
    """I don't know what this does."""
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

def fill_holes() -> None:
    """I don't know what this does."""
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

def remove_small_objects() -> None:
    """I don't know what this does."""
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

def histogram() -> None:
    """I don't know what this does."""
    img1 = cv2.imread('p5Anti A.png', 0)
    img2 = cv2.imread('p5Anti B.png', 0)
    img3 = cv2.imread('p5Anti D.png', 0)
    img4 = cv2.imread('p5Control.png', 0)
    plt.hist(img1.ravel(), 256, [0, 256])  # type: ignore[arg-type]
    plt.show()
    plt.hist(img2.ravel(), 256, [0, 256])  # type: ignore[arg-type]
    plt.show()
    plt.hist(img3.ravel(), 256, [0, 256])  # type: ignore[arg-type]
    plt.show()
    plt.hist(img4.ravel(), 256, [0, 256])  # type: ignore[arg-type]
    plt.show()


if __name__ == '__main__':
    while True:
        app = Main()
        try:
            app.mainloop()
        except RestartProgram:
            pass
        except SystemExit as err:
            if err.code == 0:
                break
            raise
        finally:
            app.destroy()
