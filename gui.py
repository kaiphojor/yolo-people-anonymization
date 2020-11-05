from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import imageio
import cv2
import pandas as pd
import detect2
import webbrowser
# FileNotFoundError: [Errno 2] No such file or directory: ''


class Window(Frame):
    global video_path
    global text_list


    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.rect = []
        self.master.title()
        self.pack(fill=BOTH, expand=1)

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        analyze = Menu(menu,tearoff=True)
        analyze.add_command(label="Convert", command=self.detecting, font=("Verdana", 10))
        analyze.add_command(label="Play", command=self.playing, font=("Verdana", 10))
        analyze.add_separator()
        analyze.add_command(label="Exit", command=self.client_exit, font=("Verdana", 10))
        menu.add_cascade(label="Run", menu=analyze)

        file = Menu(menu, tearoff=True)
        file.add_command(label="Video", command=self.open_file, font=("Verdana", 10))
        file.add_command(label="Text", command=self.open_text, font=("Verdana", 10))
        menu.add_cascade(label="Upload", menu=file)

        help_ = Menu(menu, tearoff=True)
        menu.add_cascade(label='Help', menu=help_)

        help_.add_command(label='Github', command=lambda: self.openweb(), font=("Verdana", 10))


        self.filename = "img/main_.png"
        self.imgSize = Image.open(self.filename)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas = Canvas(master=root, width=self.w, height=self.h,bg='white')
        self.canvas.create_image(95, 40, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def openweb(self):
        url = "https://github.com/Cha-Euy-Sung/"
        webbrowser.open(url, new=1)

    def open_file(self):
        global video_path
        self.filename = filedialog.askopenfilename()
        video_path = self.filename


    def open_text(self):
        global text_list
        self.filename = filedialog.askopenfilename()
        df = pd.read_excel(self.filename)
        text_list = df['name'][df['agreement'] == 'Y'].to_list()
        print("동의자 명단:",text_list)


    def playing(self):
        # NameError: name 'video_path' is not defined
        try:
            capture = cv2.VideoCapture(video_path)
            while True:
                if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                    capture.open(video_path)

                ret, frame = capture.read()
                cv2.imshow("VideoFrame", frame)

                if cv2.waitKey(33) > 0: break

            capture.release()
            cv2.destroyAllWindows()
        except(NameError):
            pass


    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy()

        self.canvas = Canvas(master=root, width=self.w, height=self.h,bg='white')
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def detecting(self):
        detect2.main(text_list, video_path)

    def client_exit(self):
        exit()


    def main_process(self):

        video_src = self.filename
        cap = cv2.VideoCapture(video_src)
        reader = imageio.get_reader(video_src)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer('./output.mp4', fps=fps)


root = Tk()
app = Window(root)
root.geometry("%dx%d" % (535, 380))
root.title(":: Identify People Only IN lisT::")
root.configure()
root.mainloop()