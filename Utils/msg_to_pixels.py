import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from Utils.hsv_segmentation import hsv_segmentation
import pcl
import cv2


class Msg2Pixels(object):
    def __init__(self):
        rospy.init_node('msg_to_pixels', anonymous=True, disable_signals=True)

        self.depth = False
        self.rgb = False
        self.pcl = False

        self.images = [[], []]

        self.pcl_points = pcl.PointCloud()

        self.sub_depth = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw", Image,
                                          callback=self.convert_image_depth, callback_args=False, queue_size=1)

        self.sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback=self.convert_image_rgb,
                                        callback_args=True, queue_size=1)

        self.sub_pcl = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, callback=self.convert_pcl,
                                        queue_size=1)
        self.r = rospy.Rate(10)

    def convert_pcl(self, pointcloud):
        if self.rgb and not self.pcl:
            points_list = []
            for data in pc2.read_points(pointcloud):
                points_list.append([data[0], data[1], data[2]])

            self.pcl_points.from_list(points_list)
            print('pcl True! :)')
            self.pcl = True

    def convert_image_rgb(self, ros_image, arg):  # arg = True(RGB)/False(depth)
        if not self.rgb:
            bridge = CvBridge()
            try:
                img = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.images[0] = img
                print('rgb True! :)')
                self.rgb = True
            except CvBridgeError:
                print(CvBridgeError)

        self.r.sleep()

    def convert_image_depth(self, ros_image, arg):  # arg = True(RGB)/False(depth)
        if self.rgb and not self.depth:
            bridge = CvBridge()
            try:
                img = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
                self.images[1] = img
                print('depth True! :)')
                self.depth = True
            except CvBridgeError:
                print(CvBridgeError)

    def save_images(self, path):
        self.rgb = False
        self.pcl = False
        self.depth = False
        print('Saving images...')
        while True:
            if self.depth and self.rgb and self.pcl:
                while True:
                    cv2.imwrite(path + ".png", self.images[0])
                    cv2.imwrite(path + ".tif", self.images[1])
                    hsv_segmentation(path)
                    pcl.save(self.pcl_points, path + '.pcd', binary=True)
                    break
                break

    def return_images(self):
        self.rgb = False
        self.depth = False
        while not (self.rgb and self.depth):
            continue

        return self.images[0], self.images[1]

    def unsubscribe(self):
        self.sub_depth.unregister()
        self.sub_rgb.unregister()
        rospy.signal_shutdown('Images acquired')
