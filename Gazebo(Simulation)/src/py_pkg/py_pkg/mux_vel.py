#!/usr/bin/env python3

from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool


class CmdVelPriorityMux(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_priority_mux')

        # ---------- Parameters ----------
        self.declare_parameter('tele_topic', 'tele_vel')
        self.declare_parameter('con_topic', 'con_vel')
        self.declare_parameter('out_topic', 'cmd_vel_out')
        self.declare_parameter('mode_topic', '/mode')

        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('speed_multiplier', 1.0)

        # "Không gửi gì nữa" trong ROS thường phải hiểu bằng timeout
        self.declare_parameter('tele_timeout_sec', 0.10)
        self.declare_parameter('con_timeout_sec', 0.10)

        self.declare_parameter('default_frame_id', 'base_footprint')
        self.declare_parameter('publish_zero_when_idle', True)

        tele_topic = self.get_parameter('tele_topic').get_parameter_value().string_value
        con_topic = self.get_parameter('con_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value
        mode_topic = self.get_parameter('mode_topic').get_parameter_value().string_value

        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.speed_multiplier = self.get_parameter('speed_multiplier').get_parameter_value().double_value
        self.tele_timeout_sec = self.get_parameter('tele_timeout_sec').get_parameter_value().double_value
        self.con_timeout_sec = self.get_parameter('con_timeout_sec').get_parameter_value().double_value
        self.default_frame_id = self.get_parameter('default_frame_id').get_parameter_value().string_value
        self.publish_zero_when_idle = (
            self.get_parameter('publish_zero_when_idle').get_parameter_value().bool_value
        )

        # ---------- QoS ----------
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # ---------- Pub/Sub ----------
        self.cmd_pub = self.create_publisher(TwistStamped, out_topic, qos)
        self.tele_sub = self.create_subscription(
            TwistStamped, tele_topic, self.tele_callback, qos
        )
        self.con_sub = self.create_subscription(
            TwistStamped, con_topic, self.con_callback, qos
        )
        self.mode_sub = self.create_subscription(
            Bool, mode_topic, self.mode_callback, qos
        )

        # ---------- State ----------
        self.latest_tele_msg: Optional[TwistStamped] = None
        self.latest_con_msg: Optional[TwistStamped] = None

        self.latest_tele_rx_ns: Optional[int] = None
        self.latest_con_rx_ns: Optional[int] = None

        self.last_selected_source: Optional[str] = None
        
        # State cho việc nhân tốc độ
        self.speed_boost_active: bool = False

        period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 0.0333
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f'cmd_vel_priority_mux started | tele="{tele_topic}" | con="{con_topic}" | out="{out_topic}" | mode="{mode_topic}"'
        )
        self.get_logger().info(
            f'tele_timeout={self.tele_timeout_sec:.3f}s | con_timeout={self.con_timeout_sec:.3f}s | '
            f'publish_rate={1.0 / period:.1f}Hz | speed_multiplier={self.speed_multiplier}'
        )

    # ---------- Callbacks ----------
    def mode_callback(self, msg: Bool) -> None:
        if self.speed_boost_active != msg.data:
            self.speed_boost_active = msg.data
            self.get_logger().info(f'Speed boost mode changed to: {self.speed_boost_active}')

    def tele_callback(self, msg: TwistStamped) -> None:
        self.latest_tele_msg = msg
        self.latest_tele_rx_ns = self.get_clock().now().nanoseconds

    def con_callback(self, msg: TwistStamped) -> None:
        self.latest_con_msg = msg
        self.latest_con_rx_ns = self.get_clock().now().nanoseconds

    def on_timer(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        source, msg = self.select_source(now_ns)

        if msg is not None:
            out = self.build_output_msg(msg, now)
            self.cmd_pub.publish(out)

            if source != self.last_selected_source:
                self.get_logger().info(f'Source switched to: {source}')
                self.last_selected_source = source
            return

        # Không có nguồn hợp lệ -> pub zero để dừng xe
        if self.publish_zero_when_idle:
            zero = TwistStamped()
            zero.header.stamp = now.to_msg()
            zero.header.frame_id = self.default_frame_id
            self.cmd_pub.publish(zero)

            if self.last_selected_source != 'idle':
                self.get_logger().info('Source switched to: idle (publish zero)')
                self.last_selected_source = 'idle'

    # ---------- Helpers ----------
    def select_source(self, now_ns: int) -> Tuple[Optional[str], Optional[TwistStamped]]:
        tele_ok = self.is_fresh(self.latest_tele_rx_ns, self.tele_timeout_sec, now_ns)
        con_ok = self.is_fresh(self.latest_con_rx_ns, self.con_timeout_sec, now_ns)

        # Ưu tiên tele trước
        if tele_ok and self.latest_tele_msg is not None:
            return 'tele', self.latest_tele_msg

        if con_ok and self.latest_con_msg is not None:
            return 'con', self.latest_con_msg

        return None, None

    @staticmethod
    def is_fresh(rx_ns: Optional[int], timeout_sec: float, now_ns: int) -> bool:
        if rx_ns is None:
            return False
        return (now_ns - rx_ns) <= int(timeout_sec * 1e9)

    def build_output_msg(self, src: TwistStamped, now) -> TwistStamped:
        out = TwistStamped()
        out.header.stamp = now.to_msg()

        # Giữ frame_id từ nguồn nếu có, không thì dùng mặc định
        out.header.frame_id = src.header.frame_id if src.header.frame_id else self.default_frame_id

        # Xác định hệ số nhân tốc độ
        multiplier = self.speed_multiplier if self.speed_boost_active else 1.0

        # Áp dụng hệ số vào từng trục
        out.twist.linear.x = src.twist.linear.x * multiplier
        out.twist.linear.y = src.twist.linear.y * multiplier
        out.twist.linear.z = src.twist.linear.z * multiplier

        out.twist.angular.x = src.twist.angular.x * multiplier
        out.twist.angular.y = src.twist.angular.y * multiplier
        out.twist.angular.z = src.twist.angular.z * multiplier

        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelPriorityMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()