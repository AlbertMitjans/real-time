import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class Msg2Pixels(object):
    def __init__(self):
        rospy.init_node('msg_to_pixels', anonymous=True, disable_signals=True)

        self.rgb = False
        self.depth = False

        self.images = [[], []]

        self.sub_depth = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw", Image,
                                          callback=self.convert_image_depth, callback_args=False, queue_size=1)

        self.sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback=self.convert_image_rgb,
                                        callback_args=True, queue_size=1)

        self.r = rospy.Rate(10)

    def convert_image_rgb(self, ros_image, arg):  # arg = True(RGB)/False(depth)
        bridge = CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            self.rgb = True
            self.images[0] = img
        except CvBridgeError:
            print(CvBridgeError)

        self.r.sleep()

    def convert_image_depth(self, ros_image, arg):  # arg = True(RGB)/False(depth)
        bridge = CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            self.depth = True
            self.images[1] = img
        except CvBridgeError:
            print(CvBridgeError)

        self.r.sleep()

    def return_images(self):
        while not (self.rgb and self.depth):
            continue

        return self.images[0], self.images[1]

    def unsubscribe(self):
        self.sub_depth.unregister()
        self.sub_rgb.unregister()
        rospy.signal_shutdown('Images acquired')
