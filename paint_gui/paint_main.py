import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageGrab
import numpy as np

class PaintApp:
    def __init__(self, root, base_image_path: str):
        self.root = root
        self.root.title("Paint App")
        self.base_image_path = base_image_path

        # Current color
        self.current_color = "black"

        # Create the canvas for drawing
        self.canvas = Canvas(root, bg="white", width=IMAGE_WIDTH+2, height=IMAGE_HEIGHT+2)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Stores the coordinate at which the current click initially happened
        self.init_x, self.init_y = None, None

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_tracker)

        # Create color palette
        self.create_color_palette()

        # Loads the base image
        self.current_image = ImageTk.PhotoImage(Image.open(base_image_path))
        self.base_image = self.current_image
        self.canvas.create_image(2, 2, anchor=tk.NW, image=self.current_image)

    def create_color_palette(self):
        colors = [
            "black", "gray", "red", "orange", "yellow",
            "green", "blue", "purple", "brown", "pink"
        ]

        palette_frame = tk.Frame(self.root, padx=5, pady=5)
        palette_frame.pack(side=tk.LEFT, fill=tk.Y)
    

        # Creates the color buttons
        for i, color in enumerate(colors):
            color_button = tk.Button(
                palette_frame, bg=color, width=3, height=2,
                command=lambda col=color: self.change_color(col)
            )
            color_button.grid(row=i // 2, column=i % 2,pady=2)
    
        # Creates the shift buttons
        for i,direction in enumerate(["^", "v", "<", ">"]):
            button = tk.Button(
                palette_frame, text=direction, font=("Helvetica", 12),
                width=3, height=2, command=lambda d=direction: self.shift(d)
            )
            button.grid(row=5+i//2, column=i%2,pady=4)

        erase_button = tk.Button(
                palette_frame, text="reset", font=("Helvetica",12),width = 6, height = 2, command= self.erase
        )
        erase_button.grid(row=10, column=0, columnspan=2, pady=4)


    def erase(self):
        self.canvas.delete("all")
        self.current_image = ImageTk.PhotoImage(Image.open(self.base_image_path))
        self.base_image = self.current_image
        self.canvas.create_image(2, 2, anchor=tk.NW, image=self.current_image)



    def change_color(self, new_color):
        self.current_color = new_color

    def paint(self, event: tk.Event):
        x, y = event.x, event.y
        if self.init_x is not None:
            self.canvas.create_line(self.init_x, self.init_y, x, y, fill=self.current_color, width=5)
        self.init_x, self.init_y = x, y
        
    def reset_tracker(self, event):
        self.init_x, self.init_y = None, None

    def shift(self, direction: str):
        # Computes the direction to shift
        dx, dy = 0, 0
        if direction == "^":
            dy = -IMAGE_STEP
        elif direction == "v":
            dy = IMAGE_STEP
        elif direction == "<":
            dx = -IMAGE_STEP
        elif direction == ">":
            dx = IMAGE_STEP

        # Shifts the canvas
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 2: # Handles images
                x,y = coords
                x,y = x+dx,y+dy
                self.canvas.coords(item, x, y)
            elif len(coords) == 4: # Handles drawn lines
                x0,y0,x1,y1 = coords
                x0,y0,x1,y1 = x0+dx,y0+dy,x1+dx,y1+dy
                self.canvas.coords(item, (x0,y0,x1,y1))


    # def rotate(self,direction: str, degrees: int):



        
        
IMAGE_STEP = 10
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGE_PATH = "./pic.png"

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root, IMAGE_PATH)
    root.mainloop()
