# Third Party
import numpy as np
import rospy
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation
from frankapy import FrankaArm, FrankaConstants
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import SensorDataMessageType
from frankapy.proto import CartesianImpedanceSensorMessage, PosePositionSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from std_msgs.msg import Bool, Float32MultiArray

#Defines a controller to give high level commands to the robot
class LLMTaskController:
    
    def __init__(self, franka_arm):
        # Initializes the configuration, goals, and robot for the task
        self._home_joint_angles = [
            0.0,
            -1.76076077e-01,
            0.0,
            -1.86691416e00,
            0.0,
            1.69344379e00,
            np.pi / 4,
        ]

        self._ros_publisher = None
        self._ros_rate = None
        self._ros_msg_count = 0
        self.eval_freq = 10
        self.motion_duration = 300  # 5 minutes to complete the task
        self.control_gains = [1000, 1000, 1000, 250, 250, 250]
        self.max_step_size = 0.015

        # Variables to hold gripper state and robot position/rotation
        self.gripper_open = False
        self.gripper_close = False

        # Inital Position
        franka_arm.reset_joints()
        T_ee_world = franka_arm.get_pose()
        T_ee_world.translation = [0.3, -0.07, 0.6]
        T_ee_world_target = T_ee_world
        franka_arm.goto_pose(T_ee_world_target, use_impedance=False)

        init_state = franka_arm.get_robot_state()
        franka_arm.open_gripper()
        self.robot_pos = init_state["pose"].translation
        self.init_robot_rot = init_state["pose"].rotation

        # Subscribers
        self._gripper_open_sub = rospy.Subscriber('/gripper_open', Bool, self._gripper_open_callback)
        self._gripper_close_sub = rospy.Subscriber('/gripper_close', Bool, self._gripper_close_callback)
        self._robot_pos_sub = rospy.Subscriber('/robot_pos', Float32MultiArray, self._robot_pos_callback)

    def _gripper_open_callback(self, msg):
       #Callback to update the gripper state
        self.gripper_open = msg.data

    def _gripper_close_callback(self, msg):
        #Callback to update the gripper state
        self.gripper_close = msg.data

    def _robot_pos_callback(self, msg):
        #Callback to update the robot's position and rotation around z-axis.
        self.robot_pos = np.array(msg.data)
        print(f"Target position received is: {self.robot_pos}")

    def get_actions(self, franka_arm):
        #Get current robot pose
        curr_state = franka_arm.get_robot_state()
        curr_pos = curr_state["pose"].translation
        print(f"Current position is: {curr_pos}")
        target_pos = self.robot_pos[:3]

        # Compute the difference between current and target positions
        delta_pos = target_pos - curr_pos
        distance = np.linalg.norm(delta_pos)

        # Adjust the step size dynamically
        if distance > self.max_step_size:
            step_delta_pos = (delta_pos / distance) * self.max_step_size
        else:
            step_delta_pos = delta_pos

        return step_delta_pos

    def go_to_goal(self, franka_arm):
        #Start communication with frankapy and goes to goal
        self._start_target_stream(franka_arm=franka_arm)
        print("\nStarted streaming targets.")
        print("\nGoing to goal pose...")

        # Get actions, send targets, and repeat
        initial_time = rospy.get_time()
        while rospy.get_time() - initial_time < self.motion_duration:
            curr_state = franka_arm.get_robot_state()
            # Get incremental actions here
            actions = self.get_actions(franka_arm)

            self._send_targets(
                actions=actions,
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
            )
            if self.gripper_close:
                franka_arm.close_gripper()

            if self.gripper_open:
                franka_arm.open_gripper()

            self._ros_rate.sleep()

        print("Finished going to goal pose.")
        franka_arm.stop_skill()
        print("\nStopped streaming targets.")

    def _start_target_stream(self, franka_arm):
        #Starts streaming targets to franka-interface via frankapy
        self._ros_rate = rospy.Rate(self.eval_freq)
        self._ros_publisher = rospy.Publisher(
            FrankaConstants.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        # Initiate streaming with dummy command to go to current pose
        # NOTE: Closely adapted from
        # https://github.com/iamlab-cmu/frankapy/blob/master/examples/run_dynamic_pose.py
        franka_arm.goto_pose(
            tool_pose=franka_arm.get_pose(),
            duration=1.0,
            use_impedance=True,
            dynamic=True,
            buffer_time=self.motion_duration * 10.0,
            cartesian_impedances=list(self.control_gains),
            ignore_virtual_walls=True,
        )

    def _send_targets(self, actions, curr_pos, curr_ori_mat):
        #Sends pose targets to franka-interface via frankapy

        targ_pos = curr_pos + actions
        targ_ori_mat = self.init_robot_rot
        ros_msg = self.compose_ros_msg(
            targ_pos=targ_pos,
            targ_ori_quat=np.roll(
                Rotation.from_matrix(targ_ori_mat).as_quat(), shift=1
            ),  # (w, x, y, z)
            prop_gains=self.control_gains,
            msg_count=self._ros_msg_count,
        )

        self._ros_publisher.publish(ros_msg)
        self._ros_msg_count += 1

    def compose_ros_msg(self, targ_pos, targ_ori_quat, prop_gains, msg_count):
        #Composes a ROS message to send to franka-interface for task-space impedance control
        # NOTE: Closely adapted from
        # https://github.com/iamlab-cmu/frankapy/blob/master/examples/run_dynamic_pose.py
        # NOTE: The sensor message classes expect the input quaternions to be represented as
        # (w, x, y, z).

        curr_time = rospy.Time.now().to_time()
        proto_msg_pose = PosePositionSensorMessage(
            id=msg_count, timestamp=curr_time, position=targ_pos, quaternion=targ_ori_quat
        )
        proto_msg_impedance = CartesianImpedanceSensorMessage(
            id=msg_count,
            timestamp=curr_time,
            translational_stiffnesses=prop_gains[:3],
            rotational_stiffnesses=prop_gains[3:6],
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                sensor_proto_msg=proto_msg_pose, sensor_data_type=SensorDataMessageType.POSE_POSITION
            ),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                sensor_proto_msg=proto_msg_impedance,
                sensor_data_type=SensorDataMessageType.CARTESIAN_IMPEDANCE,
            ),
        )

        return ros_msg


if __name__ == "__main__":
    franka_arm = FrankaArm()
    controller = LLMTaskController(franka_arm)
    controller.go_to_goal(franka_arm)
