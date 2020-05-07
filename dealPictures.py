
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
dir = time.strftime('%Y-%m-%d')
filenames='right'
def get_files():
    global filenames
    filenames = tkinter.filedialog.askopenfilenames(title="选择图片", filetypes=[('图片', 'jpg'), ('图片', 'png')])
    CN_Pattern = re.compile(u'[\u4E00-\u9FBF]+')
    JP_Pattern = re.compile(u'[\u3040-\u31fe]+')
    if filenames:
        if not os.path.exists(dir):
            os.makedirs(dir)
        CN_Match = CN_Pattern.search(str(filenames))
        JP_Match = JP_Pattern.search(str(filenames))
        if CN_Match:
            filenames=None
            tkinter.messagebox.showinfo('提示','文件路径或文件名不能含有中文,请修改!')
            return
        elif JP_Match:
            filenames = None
            tkinter.messagebox.showinfo('提示','文件路径或文件名不能含有日文,请修改!')
            return

    if not os.path.exists(dir):
        os.makedirs(dir)

def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # 返回转换后的图像
    return shifted

def pic_translate():
    if not filenames:
        tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片平移!')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if filenames:
        for filename in filenames:
            if filename:
                img = cv2.imread(filename)
                newFile = filename.split('/')[-1]
                name = newFile.split('.')[0]
                filetype = newFile.split('.')[-1]
                # 将原图分别做上、下、左、右平移操作
                Downshifted25 = translate(img, 0, 25)
                cv2.imwrite(dir + '/' + name + '_Downshifted25.' + filetype, Downshifted25)  # 保存
                Upshifted25 = translate(img, 0, -25)
                cv2.imwrite(dir + '/' + name + '_Upshifted25.' + filetype, Upshifted25)  # 保存
                Rightshifted25 = translate(img, 25, 0)
                cv2.imwrite(dir + '/' + name + '_Rightshifted25.' + filetype, Rightshifted25)  # 保存
                Leftshifted25 = translate(img, -25, 0)
                cv2.imwrite(dir + '/' + name + '_Leftshifted25.' + filetype, Leftshifted25)  # 保存
                Downshifted50 = translate(img, 0, 50)
                cv2.imwrite(dir + '/' + name + '_Downshifted50.' + filetype, Downshifted50)  # 保存
                Upshifted50 = translate(img, 0, -50)
                cv2.imwrite(dir + '/' + name + '_Upshifted50.' + filetype, Upshifted50)  # 保存
                Rightshifted50 = translate(img, 50, 0)
                cv2.imwrite(dir + '/' + name + '_Rightshifted50.' + filetype, Rightshifted50)  # 保存
                Leftshifted50 = translate(img, -50, 0)
                cv2.imwrite(dir + '/' + name + '_Leftshifted50.' + filetype, Leftshifted50)  # 保存
                Downshifted100 = translate(img, 0, 100)
                cv2.imwrite(dir + '/' + name + '_Downshifted100.' + filetype, Downshifted100)  # 保存
                Upshifted100 = translate(img, 0, -100)
                cv2.imwrite(dir + '/' + name + '_Upshifted100.' + filetype, Upshifted100)  # 保存
                Rightshifted100 = translate(img, 100, 0)
                cv2.imwrite(dir + '/' + name + '_Rightshifted100.' + filetype, Rightshifted100)  # 保存
                Leftshifted100 = translate(img, -100, 0)
                cv2.imwrite(dir + '/' + name + '_Leftshifted100.' + filetype, Leftshifted100)  # 保存
        tkinter.messagebox.showinfo('提示', '平移后的图片已经保存到了'+dir+'中!')


# 定义旋转rotate函数
def rotation(image, angle, center=None, scale=1.0):
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

def pic_rotation():
    if not filenames:
        tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片旋转!')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if filenames:
        for filename in filenames:
            if filename:
                img = cv2.imread(filename)
                newFile = filename.split('/')[-1]
                name = newFile.split('.')[0]
                filetype = newFile.split('.')[-1]
                # 将原图分别做旋转操作
                Rotated15Degrees = rotation(img, 15)
                cv2.imwrite(dir + '/' + name + '_Rotated15Degrees.' + filetype, Rotated15Degrees)  # 保存
                Rotated30Degrees = rotation(img, 30)
                cv2.imwrite(dir + '/' + name + '_Rotated30Degrees.' + filetype, Rotated30Degrees)  # 保存
                Rotated45Degrees = rotation(img, 45)
                cv2.imwrite(dir + '/' + name + '_Rotated45Degrees.' + filetype, Rotated45Degrees)  # 保存
                Rotated60Degrees = rotation(img, 60)
                cv2.imwrite(dir + '/' + name + '_Rotated60Degrees.' + filetype, Rotated60Degrees)  # 保存
                Rotated75Degrees = rotation(img, 75)
                cv2.imwrite(dir + '/' + name + '_Rotated75Degrees.' + filetype, Rotated75Degrees)  # 保存
                Rotated90Degrees = rotation(img, 90)
                cv2.imwrite(dir + '/' + name + '_Rotated90Degrees.' + filetype, Rotated90Degrees)  # 保存
                # Rotated180Degrees = rotation(img, 180)
                # cv2.imwrite(dir + '/' + name + '_Rotated180Degrees.' + filetype, Rotated180Degrees)  # 保存

                Rotated105Degrees = rotation(img, 105)
                cv2.imwrite(dir + '/' + name + '_Rotated105Degrees.' + filetype, Rotated105Degrees)  # 保存

                Rotated120Degrees = rotation(img, 120)
                cv2.imwrite(dir + '/' + name + '_Rotated120Degrees.' + filetype, Rotated120Degrees)  # 保存

                Rotated135Degrees = rotation(img, 135)
                cv2.imwrite(dir + '/' + name + '_Rotated135Degrees.' + filetype, Rotated135Degrees)  # 保存

                Rotated150Degrees = rotation(img, 150)
                cv2.imwrite(dir + '/' + name + '_Rotated150Degrees.' + filetype, Rotated150Degrees)  # 保存

                Rotated165Degrees = rotation(img, 165)
                cv2.imwrite(dir + '/' + name + '_Rotated165Degrees.' + filetype, Rotated165Degrees)  # 保存

                Rotated180Degrees = rotation(img, 180)
                cv2.imwrite(dir + '/' + name + '_Rotated180Degrees.' + filetype, Rotated180Degrees)  # 保存

                Rotated195Degrees = rotation(img, 195)
                cv2.imwrite(dir + '/' + name + '_Rotated195Degrees.' + filetype, Rotated195Degrees)  # 保存

                Rotated210Degrees = rotation(img, 210)
                cv2.imwrite(dir + '/' + name + '_Rotated210Degrees.' + filetype, Rotated210Degrees)  # 保存

                Rotated225Degrees = rotation(img, 225)
                cv2.imwrite(dir + '/' + name + '_Rotated225Degrees.' + filetype, Rotated225Degrees)  # 保存

                Rotated240Degrees = rotation(img, 240)
                cv2.imwrite(dir + '/' + name + '_Rotated240Degrees.' + filetype, Rotated240Degrees)  # 保存

                Rotated255Degrees = rotation(img, 255)
                cv2.imwrite(dir + '/' + name + '_Rotated255Degrees.' + filetype, Rotated255Degrees)  # 保存

                Rotated270Degrees = rotation(img, 270)
                cv2.imwrite(dir + '/' + name + '_Rotated270Degrees.' + filetype, Rotated270Degrees)  # 保存

                Rotated285Degrees = rotation(img, 285)
                cv2.imwrite(dir + '/' + name + '_Rotated285Degrees.' + filetype, Rotated285Degrees)  # 保存

                Rotated300Degrees = rotation(img, 300)
                cv2.imwrite(dir + '/' + name + '_Rotated300Degrees.' + filetype, Rotated300Degrees)  # 保存

                Rotated315Degrees = rotation(img, 315)
                cv2.imwrite(dir + '/' + name + '_Rotated315Degrees.' + filetype, Rotated315Degrees)  # 保存

                Rotated330Degrees = rotation(img, 330)
                cv2.imwrite(dir + '/' + name + '_Rotated330Degrees.' + filetype, Rotated330Degrees)  # 保存

                Rotated345Degrees = rotation(img, 345)
                cv2.imwrite(dir + '/' + name + '_Rotated345Degrees.' + filetype, Rotated345Degrees)  # 保存


                # RotatedNeg15Degrees = rotation(img, -15)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg15Degrees.' + filetype, RotatedNeg15Degrees)  # 保存
                # RotatedNeg30Degrees = rotation(img, -30)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg30Degrees.' + filetype, RotatedNeg30Degrees)  # 保存
                # RotatedNeg45Degrees = rotation(img, -45)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg45Degrees.' + filetype, RotatedNeg45Degrees)  # 保存
                # RotatedNeg60Degrees = rotation(img, -60)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg60Degrees.' + filetype, RotatedNeg60Degrees)  # 保存
                # RotatedNeg75Degrees = rotation(img, -75)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg75Degrees.' + filetype, RotatedNeg75Degrees)  # 保存
                # RotatedNeg90Degrees = rotation(img, -90)
                # cv2.imwrite(dir + '/' + name + '_RotatedNeg90Degrees.' + filetype, RotatedNeg90Degrees)  # 保存

        tkinter.messagebox.showinfo('提示', '旋转后的图片已经保存到了'+dir+'中!')

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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

# 创建插值方法数组
methods = [
    # ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    # ("cv2.INTER_AREA", cv2.INTER_AREA),
    # ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    # ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)
]

def pic_resize():
    if not filenames:
        tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片缩放!')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if filenames:
        for filename in filenames:
            if filename:
                img = cv2.imread(filename)
                newFile = filename.split('/')[-1]
                name = newFile.split('.')[0]
                filetype = newFile.split('.')[-1]
                # 将原图做缩放操作
                for (resizeTpye, method) in methods:
                    Resized2Times = resize(img, width=img.shape[1] * 2, inter=method)
                    cv2.imwrite(dir + '/' + name + '_Resized2Times.' + filetype, Resized2Times)  # 保存
                    Resized3Times = resize(img, width=img.shape[1] * 3, inter=method)
                    cv2.imwrite(dir + '/' + name + '_Resized3Times.' + filetype, Resized3Times)  # 保存
                    ResizedAHalf = resize(img, width=img.shape[1] //2, inter=method)
                    cv2.imwrite(dir + '/' + name + '_ResizedAHalf.' + filetype, ResizedAHalf)  # 保存
                    ResizedOneThird = resize(img, width=img.shape[1] // 3, inter=method)
                    cv2.imwrite(dir + '/' + name + '_ResizedOneThird.' + filetype, ResizedOneThird)  # 保存
        tkinter.messagebox.showinfo('提示', '缩放后的图片已经保存到了'+dir+'中!')


def pic_flip():
    if not filenames:
        tkinter.messagebox.showinfo('提示', '请先选择图片才能进行图片翻转!')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if filenames:
        for filename in filenames:
            if filename:
                img = cv2.imread(filename)
                newFile = filename.split('/')[-1]
                name = newFile.split('.')[0]
                filetype = newFile.split('.')[-1]
                # 将原图分别做翻转操作
                Horizontallyflipped = cv2.flip(img, 1)
                cv2.imwrite(dir + '/' + name + '_Horizontallyflipped.' + filetype, Horizontallyflipped)  # 保存
                Verticallyflipped = cv2.flip(img, 0)
                cv2.imwrite(dir + '/' + name + '_Verticallyflipped.' + filetype, Verticallyflipped)  # 保存
                HorizontallyAndVertically = cv2.flip(img, -1)
                cv2.imwrite(dir + '/' + name + '_HorizontallyAndVertically.' + filetype, HorizontallyAndVertically)  # 保存


        tkinter.messagebox.showinfo('提示', '翻转后的图片已经保存到了'+dir+'中!')

def mouse(event, x, y, flags, param):
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

def pic_perspective():    
    if filenames:
        for filename in filenames:
            if filename:
                print("file:", filename)
                image = cv2.imread(filename)
    # 原图中卡片的四个角点
    cv2.namedWindow("image")
    
    tips_str = "Left to right, top to bottom\nLeft click the original image\nRight click the target"
    y0, dy = 20, 20
    for i, line in enumerate(tips_str.split('\n')):
        y = y0 + i*dy
        cv2.putText(image, line, (2, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)

    cv2.imshow("image", image)
    pts1 = []
    pts2 = []
    cv2.setMouseCallback("image", mouse, param=(image, pts1, pts2))
    
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


root = tkinter.Tk()
root.title('批量处理')
button = tkinter.Button(root, text="上传图片", command=get_files,width=20,height=2)
button.grid(row=0, column=0, padx=180, pady=20)
button1 = tkinter.Button(root, text="图片平移", command=pic_translate,width=20,height=2)
button1.grid(row=1, column=0, padx=1, pady=1)
button2 = tkinter.Button(root, text="图片旋转", command=pic_rotation,width=20,height=2)
button2.grid(row=2, column=0, padx=1, pady=1)
button3 = tkinter.Button(root, text="图片缩放", command=pic_resize,width=20,height=2)
button3.grid(row=3, column=0, padx=1, pady=1)
button4 = tkinter.Button(root, text="图片翻转", command=pic_flip,width=20,height=2)
button4.grid(row=4, column=0, padx=1, pady=1)

button4 = tkinter.Button(root, text="透视变换", command=pic_perspective,width=20,height=2)

button4.grid(row=5, column=0, padx=1, pady=1)
root.geometry('500x400+600+300')
root.mainloop()
