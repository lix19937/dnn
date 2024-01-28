try:
    from Tkinter import *
except:
    from tkinter import *
import numpy as np
import threading
import os.path as osp

class WindowConfig:
    def __init__(self, input, path, wrong, truth):
        self.input = input
        self.path = path
        self.wrong = wrong
        self.truth = truth

    def update(self, input, path, wrong, truth):
        self.input = input
        self.path = path
        self.wrong = wrong
        self.truth = truth

    def update_window(self, myWindow):
        myWindow.entry1.delete(0, END)
        myWindow.entry1.insert(END, self.path)
        myWindow.entry2.delete(0, END)
        myWindow.entry2.insert(END, str(self.input))
        myWindow.entry3.delete(0, END)
        myWindow.entry3.insert(END, str(self.wrong))
        myWindow.entry4.delete(0, END)
        myWindow.entry4.insert(END, str(self.truth))
    
    def load(self, path):
        txt = open(osp.join(path, 'doing.txt'), 'r').readlines()[0]
        params = txt.split(' ')
        self.path = params[0]
        self.input = int(params[1])
        self.wrong = int(params[2])
        self.truth = int(params[3])
    
    def save(self, path):
        # txt = open(osp.join(path, 'doing.txt'), 'r').readlines()[0]
        # params = txt.split(' ')
        params = []
        params.append(self.path)
        self.path = params[0]
        self.input = int(params[1])
        self.wrong = int(params[2])
        self.truth = int(params[3])
        with open(osp.join(path, 'doing.txt'), 'w') as f:
            f.write(str(params) + '\n')

class Window:
    def __init__(self, length, cb_bn1, cb_bn2, up, down, enter):
        self.window = Tk()
        self.window.title("label change")
        self.window.resizable(False, False)
        self.window.geometry("+50+40")
        self.window.attributes("-topmost",1)

        self.frame = LabelFrame(self.window, text="all data count:{}".format(length), font=('Arial 12 bold'))
        frame = self.frame
        frame.grid(row=0,sticky=W, padx=5, pady=5)
        x_label1 = Label(frame, text="file name:", font=('Arial 12 bold'), width=10, height=2)
        x_label1.grid(row=0)
        Label(frame, text="input:", font=('Arial 12 bold'), width=10, height=2).grid(row=1)
        #Entry
        self.entry1=Entry(frame, font=('Arial 12'), width=30)
        def validate(text):
            try:
                z=int(text)
                return isinstance(z, int)
            except:
                return False
        v_cmd = (frame.register(validate), '%S')
        self.entry2=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
        self.entry1.grid(row=0, column=1, columnspan=2)
        self.entry2.grid(row=1, column=1, columnspan=2)

        Label(frame, text="wrong:", font=('Arial 12 bold'), width=10, height=2).grid(row=2)
        Label(frame, text="truth:", font=('Arial 12 bold'), width=10, height=2).grid(row=3)
        self.entry3=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
        # self.entry3.insert(END, '0')
        self.entry3.grid(row=2, column=1, columnspan=2)
        self.entry4=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
        # self.entry4.insert(END, '12')
        self.entry4.grid(row=3, column=1, columnspan=2)

        Button(frame, text='Quit', font=('Arial 12 bold'), width=10, height=2, command=self.window.quit).grid(row=4, column=0,sticky=W, padx=5, pady=5)
        Button(frame, text='Filter(space)', font=('Arial 12 bold'), width=10, height=2, command=cb_bn1).grid(row=4, column=1, sticky=W, padx=5, pady=5)
        Button(frame, text='Change(esc)', font=('Arial 12 bold'), width=10, height=2, command=cb_bn2).grid(row=4, column=2, sticky=W, padx=5, pady=5)

        # self.window.bind("<Up>", up)
        def key_q(event):
            return cb_bn2()
        def key_space(event):
            return cb_bn1()
        self.window.bind("q", key_q)
        self.window.bind("<Escape>", key_q)
        self.window.bind("<space>", key_space)
        self.window.bind("w", up)
        self.window.bind("s", down)
        self.window.bind("<Return>", enter)

    def run(self):
        self.window.mainloop()

class Visualization:
    def __init__(self, window_name, point_cloud):
        import open3d as o3d
        self.vis = o3d.visualization.VisualizerWithVertexSelection()
        self.vis.create_window(window_name, 1309, 1018, 560, 0)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 4
        # opt.show_coordinate_frame = True
        self.param = o3d.io.read_pinhole_camera_parameters('view.json')
        self.vis.add_geometry(point_cloud)
        # change view
        self.ctr = self.vis.get_view_control()
        self.ctr.convert_from_pinhole_camera_parameters(self.param)

    def __del__(self):
        import open3d as o3d
        self.vis.destroy_window()
        self.param = self.ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('view.json', self.param)


class ImgThread(threading.Thread):
    def __init__(self, jpg_path):
        super().__init__()
        import open3d as o3d
        self.jpg_path = jpg_path
        self.img_vis = o3d.visualization.Visualizer()
        self.img_vis.create_window('jpg', 540, 320, 0, 600)

    def __del__(self):
        self.img_vis.destroy_window()

    def run(self):
        import open3d as o3d
        img = o3d.io.read_image(self.jpg_path)
        self.img_vis.add_geometry(img)
        self.img_vis.poll_events()