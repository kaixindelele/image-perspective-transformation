import os
import numpy as np
import cv2
import tkinter.filedialog
import time
import tkinter.messagebox
import re
import matplotlib.pyplot as plt
'''
上传照片，批量处理并保存,包括图片平移、图片旋转、图片缩放、图片翻转、透视变换。
'''
# dir = r'D:/deal_pics/' + time.strftime('%Y-%m-%d')


class ImagePerspective:
    def __init__(self,):            
        self.dir = time.strftime('%Y-%m-%d')
        self.filenames = None
        self.visual_flag = False
        self.root = tkinter.Tk()
        self.root.title('图片批量处理')
        self.button = tkinter.Button(self.root, text="上传图片", command=self.get_files,width=20,height=2)
        self.button.grid(row=0, column=0, padx=180, pady=20)

        self.text = tkinter.StringVar()
        self.text.set("可视化每一张图？"+str(self.visual_flag))
        self.button0 = tkinter.Button(self.root, textvariable=self.text, command=self.visual, width=20,height=2)
        self.button0.grid(row=1, column=0, padx=1, pady=1)

        self.button1 = tkinter.Button(self.root, text="图片平移-存到本地", command=self.pic_translate,width=20,height=2)
        self.button1.grid(row=2, column=0, padx=1, pady=1)
        self.button2 = tkinter.Button(self.root, text="图片旋转", command=self.pic_rotation,width=20,height=2)
        self.button2.grid(row=3, column=0, padx=1, pady=1)
        self.button3 = tkinter.Button(self.root, text="图片缩放", command=self.pic_resize,width=20,height=2)
        self.button3.grid(row=4, column=0, padx=1, pady=1)
        self.button4 = tkinter.Button(self.root, text="图片翻转", command=self.pic_flip,width=20,height=2)
        self.button4.grid(row=5, column=0, padx=1, pady=1)
        self.button4 = tkinter.Button(self.root, text="透视变换-交互", command=self.pic_perspective,width=20,height=2)
        self.button4.grid(row=6, column=0, padx=1, pady=1)
        self.root.geometry('500x400+600+300')
        self.root.mainloop()

    def get_files(self,):        
        self.filenames = tkinter.filedialog.askopenfilenames(title="选择图片", filetypes=[('图片', 'jpg'), ('图片', 'png')])
        CN_Pattern = re.compile(u'[\u4E00-\u9FBF]+')
        JP_Pattern = re.compile(u'[\u3040-\u31fe]+')
        if self.filenames:
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            CN_Match = CN_Pattern.search(str(self.filenames))
            JP_Match = JP_Pattern.search(str(self.filenames))
            if CN_Match:
                self.filenames=None
                tkinter.messagebox.showinfo('提示','文件路径或文件名不能含有中文,请修改!')
                return
            elif JP_Match:
                self.filenames = None
                tkinter.messagebox.showinfo('提示','文件路径或文件名不能含有日文,请修改!')
                return
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def translate(self, image, x, y):
        # 定义平移矩阵
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # 返回转换后的图像
        return shifted

    def visual(self):
        self.visual_flag = bool(1 - self.visual_flag)
        print("self.visual_flag:", self.visual_flag)        
        self.text.set("可视化每一张图？"+str(self.visual_flag))

    def cv2visual(self, image):
        if self.visual_flag:
            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def pic_translate(self, ):
        if not self.filenames:
            tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片平移!')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.filenames:
            for filename in self.filenames:
                if filename:
                    img = cv2.imread(filename)
                    newFile = filename.split('/')[-1]
                    name = newFile.split('.')[0]
                    filetype = newFile.split('.')[-1]
                    # 将原图分别做上、下、左、右平移操作
                    for x in range(4):
                        for y in range(4):
                            translated_img = self.translate(img, -50+x*25, -50+y*25)
                            self.cv2visual(translated_img)
                            img_name = "_translated_img_x" + str(x) + "_y" + str(y) + '.' +filetype
                            cv2.imwrite(self.dir + '/' + name + img_name, translated_img)

            tkinter.messagebox.showinfo('提示', '平移后的图片已经保存到了'+self.dir+'中!')


    # 定义旋转rotate函数
    def rotation(self, image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        # 返回旋转后的图像
        return rotated

    def pic_rotation(self, ):
        if not self.filenames:
            tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片旋转!')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.filenames:
            for filename in self.filenames:
                if filename:
                    img = cv2.imread(filename)
                    newFile = filename.split('/')[-1]
                    name = newFile.split('.')[0]
                    filetype = newFile.split('.')[-1]
                    # 将原图每隔15度旋转一次操作
                    for r in range(24):
                        rotated_img = self.rotation(img, (r+1)*15)
                        self.cv2visual(rotated_img)
                        cv2.imwrite(self.dir + '/' + name + '_Rotated'+str((r+1)*15)+'Degrees.' + filetype, rotated_img)
                    
            tkinter.messagebox.showinfo('提示', '旋转后的图片已经保存到了'+self.dir+'中!')

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # 初始化缩放比例，并获取图像尺寸
        dim = None
        (h, w) = image.shape[:2]
        # 如果宽度和高度均为0，则返回原图
        if width is None and height is None:
            return image
        # 宽度是空
        if width is None:
            # 则根据高度计算缩放比例
            r = height / float(h)
            dim = (int(w * r), height)
        # 如果高度为空
        else:
            # 根据宽度计算缩放比例
            r = width / float(w)
            dim = (width, int(h * r))
        # 缩放图像
        resized = cv2.resize(image, dim, interpolation=inter)
        # 返回缩放后的图像
        return resized

    def pic_resize(self, ):
        # 创建插值方法数组
        methods = [
            # ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
            ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
            # ("cv2.INTER_AREA", cv2.INTER_AREA),
            # ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
            # ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)
        ]

        if not self.filenames:
            tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片缩放!')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.filenames:
            for filename in self.filenames:
                if filename:
                    img = cv2.imread(filename)
                    newFile = filename.split('/')[-1]
                    name = newFile.split('.')[0]
                    filetype = newFile.split('.')[-1]
                    # 将原图做缩放操作
                    scale_list = [0.3, 0.5, 2, 3]
                    for (resizeTpye, method) in methods:
                        for r in scale_list:  
                            ResizedImage = self.resize(img, width=img.shape[1] * 2, inter=method)
                            self.cv2visual(ResizedImage)
                            cv2.imwrite(self.dir + '/' + name + '_Resized'+str(r).replace('.', '_')+'Times.' + filetype, ResizedImage)  # 保存

            tkinter.messagebox.showinfo('提示', '缩放后的'+str(len(scale_list))+'张图片已经保存到了'+self.dir+'中!')


    def pic_flip(self, ):
        if not self.filenames:
            tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片翻转!')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.filenames:
            for filename in self.filenames:
                if filename:
                    img = cv2.imread(filename)
                    newFile = filename.split('/')[-1]
                    name = newFile.split('.')[0]
                    filetype = newFile.split('.')[-1]
                    # 将原图分别做翻转操作
                    Horizontallyflipped = cv2.flip(img, 1)
                    self.cv2visual(Horizontallyflipped)
                    cv2.imwrite(self.dir + '/' + name + '_Horizontallyflipped.' + filetype, Horizontallyflipped)  # 保存
                    Verticallyflipped = cv2.flip(img, 0)
                    self.cv2visual(Verticallyflipped)
                    cv2.imwrite(self.dir + '/' + name + '_Verticallyflipped.' + filetype, Verticallyflipped)  # 保存
                    HorizontallyAndVertically = cv2.flip(img, -1)
                    self.cv2visual(HorizontallyAndVertically)
                    cv2.imwrite(self.dir + '/' + name + '_HorizontallyAndVertically.' + filetype, HorizontallyAndVertically)  # 保存

            tkinter.messagebox.showinfo('提示', '翻转后的图片已经保存到了'+self.dir+'中!')

    def mouse(self, event, x, y, flags, param):
        image = param[0]
        pts1 = param[1]
        pts2 = param[2]
        if event == cv2.EVENT_LBUTTONDOWN:
            pts1.append([x, y])
            xy = "%d,%d" % (x, y)
            cv2.circle(image, (x, y), 4, (0, 255, 255), thickness = -1)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 255, 255), thickness = 2)
            cv2.imshow("image", image)   
        if event == cv2.EVENT_RBUTTONDOWN:
            pts2.append([x, y])
            xy = "%d,%d" % (x, y)
            cv2.circle(image, (x, y), 4, (255, 0, 255), thickness = -1)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 255), thickness = 2)
            cv2.imshow("image", image)    

    def pic_perspective(self,):    
        if self.filenames:
            for filename in self.filenames:
                if filename:
                    print("file:", filename)
                    image = cv2.imread(filename)
        # 原图中卡片的四个角点
        cv2.namedWindow("image")
        
        tips_str = "Left click the original image\nRight click the target\nLeft2right, Top2Bottom\nEnter end"
        y0, dy = 20, 20
        for i, line in enumerate(tips_str.split('\n')):
            y = y0 + i*dy
            cv2.putText(image, line, (2, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)

        cv2.imshow("image", image)
        pts1 = []
        pts2 = []
        cv2.setMouseCallback("image", self.mouse, param=(image, pts1, pts2))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("pts1:", pts1)
        pts1 = np.float32(pts1[:4])
        print("pts2:", pts2)
        pts2 = np.float32(pts2[:4])

        assert len(pts1)==4, "每个只允许四个点"    
    
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        cv2.imwrite('dst.jpg', dst)
        # matplotlib默认以RGB通道显示，所以需要用[:, :, ::-1]翻转一下
        plt.subplot(121), plt.imshow(image[:, :, ::-1]), plt.title('input')
        plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
        plt.show()

        # tkinter.messagebox.showinfo('提示', '透视变换的图片处理完毕!')
        return dst

def main():
    img_proc = ImagePerspective()


if __name__=="__main__":
    main()

