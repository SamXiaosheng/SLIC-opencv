#coding:utf-8

import numpy as np
import sys
import  cv2

class SLIC:
    #step表示每个聚类块步长，nc表示颜色距离权重参数，ns表示空间距离权重参数
    def __init__(self, img, step, nc):
        self.img = img
        self.height, self.width = img.shape[:2]#get the h and w of image
        self._convertToLAB()
        self.step = step
        self.nc = nc
        self.ns = step #用步长来做空间权重
        self.FLT_MAX = 1000000
        self.ITERATIONS = 10

    def _convertToLAB(self):
        try:
            import cv2
            self.labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        except ImportError:#如果opencv执行有错误的话那么就使用自己的颜色空间转换程序
            self.labimg = np.copy(self.img)
            for i in xrange(self.labimg.shape[0]):
                for j in xrange(self.labimg.shape[1]):
                    rgb = self.labimg[i, j]
                    self.labimg[i, j] = self._rgb2lab(tuple(reversed(rgb))) #transform a pixel

    def _rgb2lab ( self, inputColor ) :

       num = 0
       RGB = [0, 0, 0]

       for value in inputColor :
           value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]

       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )

       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

       num = 0
       for value in XYZ :

           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )

           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )

       return Lab

    def generateSuperPixels(self):
        self._initData()
        indnp = np.mgrid[0:self.height,0:self.width].swapaxes(0,2).swapaxes(0,1)#swapaxes交换坐标轴,将产生的网格每个点空间位置
        for i in range(self.ITERATIONS):
            #self.img = cv2.imread("0010.jpg")
            self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])#重新初始化距离矩阵最大值（这个矩阵是每个像素到中心的距离）
            for j in xrange(self.centers.shape[0]):
                #print(self.centers)
                xlow, xhigh = int(self.centers[j][3] - self.step), int(self.centers[j][3] + self.step)#搜索区域是2s*2s
                ylow, yhigh = int(self.centers[j][4] - self.step), int(self.centers[j][4] + self.step)
                #边界溢出处理
                if xlow <= 0:
                    xlow = 0
                if xhigh > self.width:
                    xhigh = self.width
                if ylow <=0:
                    ylow = 0
                if yhigh > self.height:
                    yhigh = self.height
                #计算颜色像素距离
                cropimg = self.labimg[ylow : yhigh , xlow : xhigh] #图像存储是按行列也就是高宽：h*w，这里拷贝出矩形区域
                colordiff = cropimg - self.labimg[int(self.centers[j][4]), int(self.centers[j][3])]#抠出的块与聚类块中心点像素做差，产生颜色距离
                colorDist = np.sqrt(np.sum(np.square(colordiff), axis=2))#计算每个像素的平方值然后做该点位置的所有值相加（也就三个值），最后全体做一个开根号
                #计算空间距离
                yy, xx = np.ogrid[ylow : yhigh, xlow : xhigh]#用于生成所选聚类区域的空间坐标信息，分别产生两个数组列表,用于空间距离计算
                pixdist = ((yy-self.centers[j][4])**2 + (xx-self.centers[j][3])**2)**0.5#计算出区域块的所有点的空间距离
                #print(pixdist)
                dist = ((colorDist/self.nc)**2 + (pixdist/self.ns)**2)**0.5

                distanceCrop = self.distances[ylow : yhigh, xlow : xhigh]#Python好处就是可以基于块操作
                idx = dist < distanceCrop#返回小于的位置信息
                distanceCrop[idx] = dist[idx]#更新值
                self.distances[ylow : yhigh, xlow : xhigh] = distanceCrop#更新最小距离值
                self.clusters[ylow : yhigh, xlow : xhigh][idx] = j#把最小位置都标上当前聚类中心点
            #所有的聚类块都遍历完毕，更新聚类中心
            for k in xrange(len(self.centers)):
                idx = (self.clusters == k)
                colornp = self.labimg[idx]#将第k聚类点全部拷贝
                distnp = indnp[idx]#将第k类点空间位置信息拷贝
                self.centers[k][0:3] = np.sum(colornp, axis=0)#按列相加，对应像素通道求总和！
                sumy, sumx = np.sum(distnp, axis=0)#一般图像等二维都是先写y再x，这里要注意对应关系
                self.centers[k][3:] = sumx, sumy
                self.centers[k] /= np.sum(idx)#求出总的第k类的点数，计算更新
            #slic.createConnectivity()
            #slic.displayContours([255, 255, 255])  # 分割线使用白色线分割
            #cv2.imshow("superpixels", slic.img)
            #cv2.waitKey(10)
            print('Iteration=%d'%(i+1))
        print("done....")
        #cv2.waitKey(0)
        #cv2.imwrite("SLICimg.jpg",self.img)

    def _initData(self):
        self.clusters = -1 * np.ones(self.img.shape[:2])#初始化一张图像大小的矩阵全都是-1（表示聚类），以及一张图像大小矩阵全都是最大值用于图像距离
        self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])

        centers = []#按照分割块步长计算一个5d的中心点信息
        for i in xrange(self.step, self.width - self.step/2, self.step):
            for j in xrange(self.step, self.height - self.step/2, self.step):
                
                nc = self._findLocalMinimum(center=(i, j))#使用局部3*3搜索来初始化各聚类中心位置
                color = self.labimg[nc[1], nc[0]]
                center = [color[0], color[1], color[2], nc[0], nc[1]]#5d list
                centers.append(center)
        self.center_counts = np.zeros(len(centers))#产生中心点个数列表并初始化为0，用于统计整张图片每个像素属于哪个类个数！
        self.centers = np.array(centers)#产生一个二维数组（列表），用一个列表产生一个列表！

    def createConnectivity(self):#采用4邻域来消除孤立点
        label = 0
        adjlabel = 0
        lims = self.width * self.height / self.centers.shape[0]#得到每个小块的大小
        dx4 = [-1, 0, 1, 0]#4方向的最近邻
        dy4 = [0, -1, 0, 1]
        new_clusters = -1 * np.ones(self.img.shape[:2]).astype(np.int64)
        #elements = []
        for i in xrange(self.width):
            for j in xrange(self.height):
                if new_clusters[j, i] == -1:
                    elements = []
                    elements.append((j, i))
                    for dx, dy in zip(dx4, dy4):
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if (x>=0 and x < self.width and
                            y>=0 and y < self.height and
                            new_clusters[y, x] >=0):
                            adjlabel = new_clusters[y, x]
                    count = 1
                    c = 0
                    while c < count:
                        for dx, dy in zip(dx4, dy4):
                            x = elements[c][1] + dx
                            y = elements[c][0] + dy

                            if (x>=0 and x<self.width and y>=0 and y<self.height):
                                if new_clusters[y, x] == -1 and self.clusters[j, i] == self.clusters[y, x]:#如果老表中
                                    elements.append((y, x))
                                    new_clusters[y, x] = label
                                    count+=1
                        c+=1
                    if (count <= lims >> 2):
                        for c in range(count):
                            new_clusters[elements[c]] = adjlabel
                        label-=1
                    label+=1
            #self.clusters = new_clusters

    def displayContours(self, color):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]#8方向的近邻
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

        isTaken = np.zeros(self.img.shape[:2], np.bool)#创建一副轮廓线标志位，标志每一个像素是否作为轮廓线
        contours = []

        for i in xrange(self.width):
            for j in xrange(self.height):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x>=0 and x < self.width and y>=0 and y < self.height:
                        if isTaken[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:#计算8邻域内有多少个不同点
                            nr_p += 1

                if nr_p >= 2:#至少有两个不同就可以确定该点为轮廓线
                    isTaken[j, i] = True#已被作为轮廓
                    contours.append([j, i])

        for i in xrange(len(contours)):
            self.img[contours[i][0], contours[i][1]] = color#为轮廓线画上颜色

    def _findLocalMinimum(self, center):#这个算法的目的是避免所选的聚类中心是边缘和噪声
        min_grad = self.FLT_MAX
        loc_min = center
        #Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
        #method is called upon initialization of the cluster centers.
        for i in xrange(center[0] - 1, center[0] + 2):
            for j in xrange(center[1] - 1, center[1] + 2):
                c1 = self.labimg[j+1, i]
                c2 = self.labimg[j, i+1]
                c3 = self.labimg[j, i]
                # 只用0通道
                if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
                    min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                    loc_min = [i, j]
        return loc_min
#通过后台执行的传入参数
#img = cv2.imread(sys.argv[1])
#nr_superpixels = int(sys.argv[2])
#nc = int(sys.argv[3])
if __name__ == "__main__":
    img = cv2.imread('0010.jpg')
    nr_superpixels = 100
    nc = 10
    step = int((img.shape[0]*img.shape[1]/nr_superpixels)**0.5)#assume the area is regular
    slic = SLIC(img, step, nc)
    #cv2.imshow("lab",slic.labimg)
    #cv2.waitKey(10)
    slic.generateSuperPixels()
    slic.createConnectivity()
    slic.displayContours([255,255,255])
    cv2.imshow("superpixels",slic.img)
    cv2.imwrite("SLICimg.jpg",slic.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#slic.createConnectivity()
#slic.displayContours([255,255,255])#分割线使用白色线分割
#cv2.imshow("superpixels", slic.img)
#cv2.waitKey(0)
#cv2.imwrite("SLICimg.jpg", slic.img)