#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
import numpy as np


def talker():
    #Publish gripper state
    pub = rospy.Publisher('/gripper_close', Bool, queue_size=10)
    pub = rospy.Publisher('/gripper_open', Bool, queue_size=10)
    rospy.init_node('talk', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    rospy.spin()
    # while not rospy.is_shutdown():
    #     msg = Bool()
    #     msg.data = True
    #     rospy.loginfo(msg)
    #     pub.publish(msg)
    #     rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
