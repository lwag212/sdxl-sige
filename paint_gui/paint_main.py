import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageGrab
import numpy as np

class PaintApp:
    def __init__(self, root, base_image_path: str):
        self.root = root
        self.root.title("Paint App")
        self.base_image_path = base_image_path

        image = Image.open(base_image_path)
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size
        image.close()

        # Current color
        self.current_color = "black"

        # Create the canvas for drawing

        self.canvas = Canvas(root, bg="white", width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Stores the coordinate at which the current click initially happened
        self.init_x, self.init_y = None, None

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_tracker)

        # Create color palette
        self.create_color_palette()

        #track if user has painted. For shifting 
        self.rot_value = 0
        self.has_paint = False
        self.x = 0
        self.y = 0 
        self.drawn_lines = [] 
        self.DONE_EDITING = False 

        #keep as pil so can manipulate properly 
        self.pil_base_image = Image.open(base_image_path)
        self.pil_width, self.pil_height = self.pil_base_image.size

        self.temp_base_image = self.pil_base_image
        self.current_image = ImageTk.PhotoImage(Image.open(base_image_path))
        self.base_image = self.current_image
        self.canvas.create_image(2, 2, anchor=tk.NW, image=self.current_image,tags="image_layer")


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

        #rotate directions 
        for i,dir in enumerate(["L","R"]):
            button = tk.Button(
                palette_frame, text=dir, font=("Helvetica", 12),
                width=3, height=2, command=lambda d=dir: self.rotate(d)
            )
            button.grid(row=10, column=i%2,pady=4,padx=2)

        # Createse reset button 
        reset_button = tk.Button(
                palette_frame, text="reset", font=("Helvetica",12),width = 6, height = 2, command= self.reset
        )
        reset_button.grid(row=12, column=0, columnspan=2, pady=4)

        generate_button = tk.Button(
                        palette_frame, text="generate", font=("Helvetica",12),width = 6, height = 2, command= self.generate
                )
        generate_button.grid(row=14, column=0, columnspan=2, pady=4)

        #reset fn, resets everything - paint and img loc/orientaiton 
    def reset(self):
        self.canvas.delete("all")
        self.current_image = ImageTk.PhotoImage(Image.open(self.base_image_path))
        self.base_image = self.current_image
        self.canvas.create_image(2, 2, anchor=tk.NW, image=self.current_image,tags="image_layer")
        self.drawn_lines = []
        self.has_paint = False
        self.x = 0 
        self.y = 0
        self.DONE_EDITING = False

    def change_color(self, new_color):
        self.current_color = new_color

    def paint(self, event: tk.Event):
        if not self.DONE_EDITING:
            x, y = event.x, event.y
            if self.init_x is not None:
                self.canvas.create_line(self.init_x, self.init_y, x, y, fill=self.current_color, width=5,tags="line")
                #toggle paint var 
                self.drawn_lines.append((self.init_x,x,self.init_y,y,self.current_color))
                self.has_paint = True 
            self.init_x, self.init_y = x, y

    #this is so broken, don't really need it though 
    # def redraw_lines(self):
    #     self.canvas.delete("line")
    #     for line in self.drawn_lines:
    #         self.canvas.create_line(line[0],line[1],line[2],line[3],fill=line[4],width=5,tags="line")

    def reset_tracker(self, event):
        self.init_x, self.init_y = None, None

    def shift(self, direction: str):
        if not self.DONE_EDITING:
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
            
                if len(coords) == 2:  # Handles images
                    x, y = coords
                    # Update the image's position directly
                    self.canvas.coords(item, x + dx, y + dy)
                    
                elif len(coords) == 4:  # Handles drawn lines
                    x0, y0, x1, y1 = coords
                    # Shift the drawn lines by updating their coordinates
                    self.canvas.coords(item, x0 + dx, y0 + dy, x1 + dx, y1 + dy)

        # Update the image's anchor position
            self.x += dx
            self.y += dy

    # func to rotate 
    def rotate(self, dir: str):
        if not self.DONE_EDITING:
        #temp copy of pil image 
            working_image = self.pil_base_image.copy()
            #calc center so rotating doesn't go crazy
            center =  (self.pil_width/2,self.pil_height/2)
            if dir == "L": 
                rotated_image = working_image.rotate(self.rot_value + 5, center = center,  resample=Image.BICUBIC, fillcolor = "white") #resample the image so it sucks less 
                self.rot_value+=5 
            elif dir == "R":
                rotated_image = working_image.rotate(self.rot_value -5, center = center,  resample=Image.BICUBIC, fillcolor = "white") 
                self.rot_value-=5
            rotated_image = self.crop_original(rotated_image)
            self.current_image = ImageTk.PhotoImage(rotated_image)
            self.canvas.delete("image_layer")  # Remove the old image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image, tags="image_layer")
            # Update the base image to reflect the rotation
            self.base_image = rotated_image
            self.temp_base_image = rotated_image
        
    def crop_original(self, rotated_image: Image):
        #crop the image to the canvas size, but maintain the original image (secretly). Crop final does actual cropping 
        rotated_width, rotated_height = rotated_image.size
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()

        # Center of the rotated image
        left = (rotated_width - canvas_width) / 2
        upper = (rotated_height - canvas_height) / 2
        right = left + canvas_width
        lower = upper + canvas_height
        # Crop the rotated image to fit within the bounds of the canvas
        rotated_image_cropped = rotated_image.crop((left, upper, right, lower))
        #convert image         
        self.current_image = ImageTk.PhotoImage(rotated_image_cropped)
        return rotated_image_cropped  


    def crop_to_canvas(self, image: Image) -> Image:
        # Get the width and height of the image
        canvas_width = self.canvas.winfo_width()   
        canvas_height = self.canvas.winfo_height()
       
        # Crop the image to the canvas
        cropped_image = image.crop((0, 0, canvas_width, canvas_height))
        self.current_image = ImageTk.PhotoImage(cropped_image)  
        self.canvas.delete("image_layer")  # Remove the old image
        self.canvas.create_image(self.x,self.y, anchor=tk.NW, image=self.current_image, tags="image_layer")
        return cropped_image

    def generate(self):
        self.rot_value = 0 
        self.crop_to_canvas(self.temp_base_image) 
        #lock editing 
        self.DONE_EDITING = True 

        
        
IMAGE_STEP = 10
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGE_PATH = "./pic2.jpg"

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root, IMAGE_PATH)
    root.mainloop()
