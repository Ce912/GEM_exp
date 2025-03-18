import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np


def talker():
    pub = rospy.Publisher('/robot_pos', Float32MultiArray, queue_size=10)
    rospy.init_node('talke', anonymous=True)
    msg = Float32MultiArray()
    msg.data = np.array([0.477, -0.55, 0.3, 45])
    rospy.loginfo(msg)
    pub.publish(msg)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
