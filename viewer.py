import PIL.Image
from tkinter import *
from tkinter import filedialog
import PIL.ImageTk
from InstColorization.models import create_model
from InstColorization.options.train_options import TestOptions
from InstColorization.util import util
import torch

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from InstColorization.fusion_dataset import Fusion_Testing_Dataset
from shutil import copyfile

import cv2
import numpy as np
import os
setup_logger()

class App(Frame):
    def chg_image(self):
        if self.im.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(self.im, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(self.im)
        self.la_grey.config(image=self.img, bg="#000000",
            width=self.img.width(), height=self.img.height())

    def open(self):
        tmp = filedialog.askopenfilename()
        if tmp != "":
            self.filename = tmp
            self.output_gray_image_dir = "{0}_image_gray".format("example")
            if os.path.isdir(self.output_gray_image_dir) is False:
                os.makedirs(self.output_gray_image_dir)
            copyfile(self.filename, os.path.join(self.output_gray_image_dir, self.filename.split("/")[-1]))

            self.im = PIL.Image.open(self.filename)
        self.chg_image()


    def get_bb(self):
        output_npz_dir = "{0}_bbox".format("example")
        if os.path.isdir(output_npz_dir) is False:
            os.makedirs(output_npz_dir)


        self.opt.test_img_dir = output_npz_dir

        img = cv2.imread(self.filename)
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
        outputs = self.predictor(l_stack)
        pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
        pred_scores = outputs["instances"].scores.cpu().data.numpy()
        np.savez(os.path.join(output_npz_dir, self.filename.split("/")[-1].split(".")[0]), bbox=pred_bbox, scores=pred_scores)


    def chg_color_image(self):

        img = cv2.imread(self.filename)
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab_image)

        img = cv2.imread(os.path.join(self.output_color_image_mask_dir, self.filename.split("/")[-1].split(".")[0]+".png"))
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a_pred, b_pred = cv2.split(lab_image)

        a_pred = cv2.resize(a_pred, (l_channel.shape[1], l_channel.shape[0]))
        b_pred = cv2.resize(b_pred, (l_channel.shape[1], l_channel.shape[0]))
        gray_color = np.ones_like(a_pred) * 128

        gray_image = cv2.cvtColor(np.stack([l_channel, gray_color, gray_color], 2), cv2.COLOR_LAB2BGR)
        color_image = cv2.cvtColor(np.stack([l_channel, a_pred, b_pred], 2), cv2.COLOR_LAB2BGR)


        self.output_color_image_dir = "{0}_image_color".format("example")
        if os.path.isdir(self.output_color_image_dir) is False:
            os.makedirs(self.output_color_image_dir)

        cv2.imwrite(os.path.join(self.output_color_image_dir, self.filename.split("/")[-1].split(".")[0]+".png"), color_image)

        self.im_color = PIL.Image.open(os.path.join(self.output_color_image_dir, self.filename.split("/")[-1].split(".")[0]+".png"))
        if self.im_color.mode == "1": # bitmap image
            self.img_color = PIL.ImageTk.BitmapImage(self.im_color, foreground="white")
        else:              # photo image
            self.img_color = PIL.ImageTk.PhotoImage(self.im_color)

        self.la_color.config(image=self.img_color, bg="#000000",
            width=self.img.width(), height=self.img.height())


    def colorize(self):
        self.get_bb()

        self.opt.batch_size = 1

        dataset = Fusion_Testing_Dataset(self.opt, self.output_gray_image_dir)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size)

        for data_raw in dataset_loader:
            data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
            if data_raw['empty_box'][0] == 0:
                data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
                box_info = data_raw['box_info'][0]
                box_info_2x = data_raw['box_info_2x'][0]
                box_info_4x = data_raw['box_info_4x'][0]
                box_info_8x = data_raw['box_info_8x'][0]
                cropped_data = util.get_colorization_data(data_raw['cropped_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                self.model.set_input(cropped_data)
                self.model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                self.model.forward()
            else:
                full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                self.model.set_forward_without_box(full_img_data)

            self.output_color_image_mask_dir = "{0}_image_color_mask".format("example")
            if os.path.isdir(self.output_color_image_mask_dir) is False:
                os.makedirs(self.output_color_image_mask_dir)

            self.model.save_current_imgs(os.path.join(self.output_color_image_mask_dir, data_raw['file_id'][0] + '.png'))
        self.chg_color_image()


    def __init__(self, master=None):

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

        self.opt = TestOptions().parse()
        self.model = create_model(self.opt)
        self.model.setup_to_test('coco_finetuned_mask_256_ffs')
        Frame.__init__(self, master)
        self.master.title('Image Viewer')

        self.num_page_tv = StringVar()

        fram1 = Frame(self)

        Button(fram1, text="Open File", command=self.open).pack(side=LEFT)
        Button(fram1, text="Add color", command=self.colorize).pack(side=LEFT)
        Label(fram1, textvariable=self.num_page_tv).pack(side=LEFT)

        fram1.pack(side=TOP, fill=BOTH)

        self.la_grey = Label(self)
        self.la_grey.config(bg="white", width=0, height=0)
        self.la_grey.pack(side=LEFT)

        self.la_color = Label(self)
        self.la_color.config(bg="white", width=0, height=0)
        self.la_color.pack(side=RIGHT)

        self.pack()

if __name__ == "__main__":
    app = App(); app.mainloop()