import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from hsv_segmentation import hsv_segmentation
import pcl
import cv2
import numpy as np
import os


class Msg2Pixels(object):
    def __init__(self):
        os.environ['ROS_MASTER_URI'] = 'http://192.168.1.44:11311'  # connection to raspberry pi
        os.environ['ROS_IP'] = '192.168.1.44'

        rospy.init_node('msg_to_pixels', anonymous=True, disable_signals=True)

        self.depth = False
        self.rgb = False

        self.images = [[], []]

        self.pcl_points = pcl.PointCloud()

        self.sub_depth = rospy.Subscriber("/camera/depth/image_rect_raw", Image,
                                          callback=self.convert_image, callback_args=False, queue_size=1)

        self.sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback=self.convert_image,
                                        callback_args=True, queue_size=1)

        self.sub_pcl = rospy.Subscriber("/camera/depth/points", PointCloud2, callback=self.convert_pcl,
                                        queue_size=1)

    def convert_pcl(self, pointcloud):
        points_list = []
        for data in pc2.read_points(pointcloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2]])

        self.pcl_points.from_list(points_list)

    def convert_image(self, ros_image, arg):  # arg = True(RGB)/False(depth)
        bridge = CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            if img.size != 0:
                if arg:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.images[0] = img
                    self.rgb = True
                if not arg:
                    self.images[1] = img
                    self.depth = True
            if img.size == 0:
                if arg:
                    self.rgb = False
                if not arg:
                    self.depth = False
        except CvBridgeError:
            print(CvBridgeError)

    def save_images(self, path):
        print('Saving images...')
        while True:
            if self.depth & self.rgb:
                while True:
                    if np.count_nonzero(self.images[1] == 0) < 1000000:
                        print(np.count_nonzero(self.images[1] == 0))
                        cv2.imwrite(path + ".png", self.images[0])
                        cv2.imwrite(path + ".tif", self.images[1])
                        hsv_segmentation(path)
                        pcl.save(self.pcl_points, path + '_pcl.pcd', binary=True)
                        break
                break

    def unsubscribe(self):
        self.sub_depth.unregister()
        self.sub_rgb.unregister()
        rospy.signal_shutdown('Images acquired')
