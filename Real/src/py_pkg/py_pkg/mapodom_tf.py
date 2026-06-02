#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import  QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TransformStamped
import tf2_ros

class SmoothTFBroadcasterNode(Node):
    def __init__(self):
        super().__init__('smooth_tf_broadcaster_node')
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.latest_transform = TransformStamped()
        self.latest_transform.header.frame_id = 'map'
        self.latest_transform.child_frame_id = 'odom'
        
        self.latest_transform.transform.translation.x = 0.0
        self.latest_transform.transform.translation.y = 0.0
        self.latest_transform.transform.translation.z = 0.0
        self.latest_transform.transform.rotation.w = 1.0 
        self.latest_transform.transform.rotation.x = 0.0
        self.latest_transform.transform.rotation.y = 0.0
        self.latest_transform.transform.rotation.z = 0.0
        

        transform_stamp_qos = QoSProfile(depth=1)
        transform_stamp_qos.reliability = ReliabilityPolicy.RELIABLE
        transform_stamp_qos.durability = DurabilityPolicy.VOLATILE
        transform_stamp_qos.history = HistoryPolicy.KEEP_LAST

        self.sub = self.create_subscription(
            TransformStamped,
            '/map_to_odom',
            self.correction_callback,
            transform_stamp_qos
        )
        
        # Pub TF liên tục ở 50Hz
        self.timer = self.create_timer(0.02, self.timer_callback)

    def correction_callback(self, msg: TransformStamped):
        # Chỉ cập nhật tọa độ (x, y, z, quaternion). 
        # BỎ QUA cái msg.header.stamp (thời gian quá khứ của LiDAR)
        self.latest_transform.transform = msg.transform

    def timer_callback(self):
        # LUÔN LUÔN DÙNG NOW() TRONG MỌI TRƯỜNG HỢP
        self.latest_transform.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.latest_transform)


def main(args=None):
    rclpy.init(args=args)
    node = SmoothTFBroadcasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()