import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv

class AutoLabeler:
    def __init__(self):
        self.original_img = None
        self.display_img = None
        self.window_name = "Auto Labeler"
        self.drawing = False
        self.scale_factor = 1.0
        

    def load_img(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.original_img = cv.imread(file_path)
            if self.original_img is None:
                messagebox.showerror("Error", "Could not load the image. Please try another file.")
                return False

            self.prepare_display_img()
            return True
        return False


    def prepare_display_img(self):
        if self.original_img is None:
            return

        height, width = self.original_img.shape[:2]
        max_display_size = 800

        if height > max_display_size or width > max_display_size:
            if height > width:
                self.scale_factor = max_display_size / height
            else:
                self.scale_factor = max_display_size / width

            new_width = int(width*self.scale_factor)
            new_height = int(height*self.scale_factor)

            self.display_img = cv.resize(self.original_img, (new_width, new_height))
        else:
            self.display_img = self.original_img.copy()
            self.scale_factor = 1.0


    # Mouse callback functions
    def mouse_callback(self):
        pass

    # Update display following mouse events
    def update_display(self):
        if self.display_img is None:
            return

        img_copy = self.display_img.copy()

        cv.imshow(self.window_name, img_copy)


    def save_img(self):
        if True: # if self.annotated_img is None:
            messagebox.showwarning("Warning", "No image to save. Please label the image first.")
            return
        
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            # cv.imwrite(file_path, self.annotated_img)
            messagebox.showinfo("Success", f"Image saved to {file_path}")


    def run(self):
        print("Auto Labeler")

        if not self.load_img():
            print("No image selected. Exiting.")
            return

        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

        while True:
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_img()
            elif key == ord('r'):
                cv.destroyAllWindows()
                if self.load_img():
                    cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
                    cv.setMouseCallback(self.window_name, self.mouse_callback)
                    self.update_display()
                    # Reset anything else
                else:
                    break


    


if __name__ == "__main__":
    app = AutoLabeler()
    app.run()