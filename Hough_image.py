import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb
## code from:
##https://pypi.python.org/pypi/opencv-python


img = cv2.imread('images.png',0)
edges = cv2.Canny(img,100,200)

##pdb.set_trace()
##lines = cv2.HoughLines(edges,1,np.pi/180,200)
##for rho,theta in lines[0]:
##    a = np.cos(theta)
##    b = np.sin(theta)
##    x0 = a*rho
##    y0 = b*rho
##    x1 = int(x0 + 1000*(-b))
##    y1 = int(y0 + 1000*(a))
##    x2 = int(x0 - 1000*(-b))
##    y2 = int(y0 - 1000*(a))
##
##    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
##
##hough = cv2.imwrite('houghlines3.jpg',img)



plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
##plt.subplot(133),plt.imshow(lines,cmap = 'gray')
##plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
plt.show()

##plt.savefig(Thanksgiving_meal, dpi=None, facecolor='w', edgecolor='w',
##            orientation='portrait', papertype=None, format=None,
##            transparent=False, bbox_inches='tight', pad_inches=0.1,
##            frameon=None)
