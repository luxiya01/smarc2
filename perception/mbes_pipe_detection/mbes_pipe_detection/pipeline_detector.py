import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PointStamped
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import numpy as np

from mbes_pipe_detection.tf2_sensor_msgs import do_transform_cloud
from mbes_pipe_detection import utils as mbes_utils

class PipelineDetector(Node):
    def __init__(self):
        super().__init__('pipeline_detector')
        self.get_logger().info('Pipeline Detector Node has been started.')
        self._declare_and_initialize_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.circular_pcl_buffer = None
        self.ping_counter = 0
        self.pcl_fields = None
        self.last_centroid = None
        self.point_cloud_subscriber = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            100
        )

        self.cv_bridge = CvBridge()
        self.create_timer(1.0 / self.detection_frequency, self.detection_callback)
        self.gradient_image_pub = self.create_publisher(
            Image,
            'gradient_image',
            10
        )
        self.detection_image_pub = self.create_publisher(
            Image,
            'pipeline_detection_image',
            10
        )
        self.pipeline_point_pub = self.create_publisher(
            PointStamped,
            'pipeline_point',
            10
        )

    def _declare_and_initialize_parameters(self):
        self.declare_parameter('input_topic', '/lolo/sensors/mbes/bathymetry/points')
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.declare_parameter('output_topic', 'pipeline_detection')
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.declare_parameter('frame_id', 'lolo/base_link')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.declare_parameter('utm_zone', '33')
        self.utm_zone = self.get_parameter('utm_zone').get_parameter_value().string_value
        self.declare_parameter('utm_band', 'V')
        self.utm_band = self.get_parameter('utm_band').get_parameter_value().string_value
        self.utm_frame = f'utm_{self.utm_zone}_{self.utm_band}'
        self.get_logger().info(f'Using UTM frame: {self.utm_frame}')

        # Declare parameters for pipeline detection
        self.declare_parameter('num_pings_for_detection', 100)
        self.num_pings_for_detection = self.get_parameter('num_pings_for_detection').get_parameter_value().integer_value
        self.declare_parameter('detection_frequency', 1.)  # Hz
        self.detection_frequency = self.get_parameter('detection_frequency').get_parameter_value().double_value
        self.declare_parameter('normalize_intensity', True)
        self.normalize_intensity = self.get_parameter('normalize_intensity').get_parameter_value().bool_value
        self.declare_parameter('resolution', 0.5)  # meters
        self.declare_parameter('min_translation', 0.1)  # meters
        self.min_translation = self.get_parameter('min_translation').get_parameter_value().double_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.get_logger().info(f'Pipeline detection parameters: num_pings={self.num_pings_for_detection}, '
                               f'detection_frequency={self.detection_frequency}, '
                               f'normalize_intensity={self.normalize_intensity}, resolution={self.resolution}, '
                               f'min_translation={self.min_translation}')

    def _initiate_circular_buffer(self, msg):
        """
        Initializes the circular buffer with the first point cloud message.
        """
        num_bins = msg.width
        self.get_logger().info(f'Number of bins in pcl: {num_bins}')
        self.circular_pcl_buffer = np.zeros((self.num_pings_for_detection, num_bins, 4), dtype=np.float32)
        self.get_logger().info(f'Initialized circular pcl buffer with shape: {self.circular_pcl_buffer.shape}')
        self.fields = msg.fields
        self.get_logger().info(f'Point cloud fields: {self.fields}')

    def point_cloud_callback(self, msg):
        """
        Callback function for the point cloud subscriber.
        This function processes incoming PointCloud2 messages, transforms them to the UTM frame if necessary,
        and appends the points to the circular_pcl_buffer.
        """
        # Transform the point cloud to the desired frame if necessary
        if msg.header.frame_id != self.utm_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.utm_frame,
                                                            # msg.header.frame_id,
                                                            self.frame_id,
                                                            rclpy.time.Time())
                msg = do_transform_cloud(msg, transform)
            except Exception as e:
                self.get_logger().error(f'Error transforming point cloud: {e}')
                return

        if self.circular_pcl_buffer is None:
            self._initiate_circular_buffer(msg=msg)

        self.update_circular_buffer(msg)

    def update_circular_buffer(self, msg):
        """
        Updates the circular buffer with the latest point cloud message.
        Ignores the message if all xyz values are zero.
        """

        pcl = point_cloud2.read_points_numpy(msg, ['x', 'y', 'z', 'intensity'])
        if pcl.size == 0 or np.all(pcl[:, :3] == 0):
            self.get_logger().warn('Received point cloud with all xyz values as zero, ignoring this ping.')
            return

        centroid = np.mean(pcl[:, :3], axis=0)
        if self.last_centroid is not None:
            distance = np.linalg.norm(centroid - self.last_centroid)
            if distance < self.min_translation:
                # self.get_logger().warn(f'Ignoring ping due to small translation: {distance:.2f} < {self.min_translation:.2f}')
                return

        self.last_centroid = centroid
        self.circular_pcl_buffer[self.ping_counter % self.num_pings_for_detection, ...] = pcl.reshape(-1, 4)
        self.ping_counter += 1


    def get_ordered_pings(self, normalize=True):
        """
        Returns an ordered chronological view of the circular point cloud buffer.
        If not enough pings have been received, it returns None.
        """
        if self.ping_counter < self.num_pings_for_detection:
            self.get_logger().warn(f'Not enough pings received yet: {self.ping_counter} < {self.num_pings_for_detection}')
            return None
        start_index = self.ping_counter % self.num_pings_for_detection
        ordered_pings = np.roll(self.circular_pcl_buffer, -start_index, axis=0)

        if normalize:
            mean_intensity = np.mean(ordered_pings[:, :, -1], axis=0)
            ordered_pings[..., -1] /= mean_intensity
        return ordered_pings

    def detection_callback(self):
        """
        This callback is called at the specified detection frequency.
        It retrieves the ordered pings from the circular buffer and performs pipeline detection.
        The results are published as an Image message.
        """
        ordered_pings = self.get_ordered_pings(normalize=self.normalize_intensity)
        intensity_dict = mbes_utils.pcl_buffer_to_intensity(ordered_pings, self.resolution)

        if intensity_dict is None:
            self.get_logger().warn('Not enough pings received to construct intensity image.')
            return None
        gradient_image = intensity_dict['gradient_image']   # used to be intensity_dict['intensity_image']
        mask = intensity_dict['mask']
        normalized_gradient_image = mbes_utils.normalize_image(gradient_image)
        gradient_image_msg = self.cv_bridge.cv2_to_imgmsg(normalized_gradient_image, encoding='mono8')
        self.gradient_image_pub.publish(gradient_image_msg)
    
        gradient_image_uint8 = mbes_utils.img_to_uint8(gradient_image)
        pipeline, mid_x, mid_y, pipeline_image = mbes_utils.pipeline_detect(gradient_image_uint8)
        if pipeline:
            self.get_logger().info(f'Pipeline detected at mid_x:,  mid_y: {mid_x}, {mid_y}')    # img coordinates atm
            self.detection_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(pipeline_image, encoding='mono8'))
            mid_x = int(mid_x)
            mid_y = int(mid_y)
            x = intensity_dict['x'][mid_y, mid_x]
            y = intensity_dict['y'][mid_y, mid_x]
            # publish xy coordinates in utm frame
            point_msg = PointStamped()
            point_msg.header = Header()
            point_msg.header.frame_id = self.utm_frame
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.point.x = x
            point_msg.point.y = y
            point_msg.point.z = 0.0
            self.pipeline_point_pub.publish(point_msg)
            self.get_logger().info(f'Published pipeline point at x: {x}, y: {y} in frame {self.utm_frame}')


        else:
            self.get_logger().info('No pipeline detected in this patch.')


def main(args=None):
    rclpy.init(args=args)
    node = PipelineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()