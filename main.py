import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings

from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')



# ====================== 新增：中文绘制工具函数 ======================
def draw_chinese_text(img, text, pos, font_size=20, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文（解决乱码问题）
    :param img: OpenCV格式的图像
    :param text: 要显示的中文文本
    :param pos: 文本位置 (x, y)
    :param font_size: 字体大小
    :param color: 字体颜色 (B, G, R)
    :return: 绘制后的图像
    """
    # 转换OpenCV图像到PIL格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 设置中文字体（使用系统自带的中文字体，适配Windows）
    try:
        # Windows系统默认中文字体路径
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        # 备用方案：使用PIL默认字体（可能显示方块，但不会乱码）
        font = ImageFont.load_default()
    
    # 绘制中文文本
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


# ====================== 1. 手掌检测与关键点提取模块 ======================
class HandPoseDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        """初始化MediaPipe手掌检测模型"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化手掌检测模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        
        # 手掌骨骼连接关系（MediaPipe定义的21个关键点连接）
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
        # 关键点名称映射（便于姿态评估）
        self.keypoint_names = [
            '腕部', '拇指根', '拇指中', '拇指尖', '食指根', '食指中1', '食指中2', '食指尖',
            '中指根', '中指中1', '中指中2', '中指尖', '无名指根', '无名指中1', '无名指中2', '无名指尖',
            '小拇指根', '小拇指中1', '小拇指中2', '小拇指尖'
        ]

    def detect_hand_keypoints(self, frame):
        """检测单帧图像中的手掌关键点"""
        # 转换颜色空间（MediaPipe需要RGB格式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        keypoints_list = []  # 存储所有检测到的手掌关键点
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取关键点坐标（归一化坐标转像素坐标）
                h, w, _ = frame.shape
                keypoints = []
                for lm in hand_landmarks.landmark:
                    x, y, z = int(lm.x * w), int(lm.y * h), lm.z
                    keypoints.append([x, y, z])
                
                keypoints_list.append(np.array(keypoints))
                
                # 绘制骨骼线
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame, keypoints_list

# ====================== 2. 姿态评估模块 ======================
class HandPoseEvaluator:
    def __init__(self):
        pass
    
    def calculate_angle(self, p1, p2, p3):
        """计算三个点组成的夹角（p2为顶点）"""
        # 转换为向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算夹角（弧度转角度）
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle
    
    def evaluate_pose(self, keypoints):
        """评估手掌姿态（基于关键角度）"""
        pose_evaluation = {
            '手指弯曲度': {},
            '整体姿态': '',
            '评分': 0
        }
        
        # 计算各手指弯曲角度
        finger_joints = [
            (4, 3, 2),    # 拇指
            (8, 7, 6),    # 食指
            (12, 11, 10), # 中指
            (16, 15, 14), # 无名指
            (20, 19, 18)  # 小拇指
        ]
        
        finger_names = ['拇指', '食指', '中指', '无名指', '小拇指']
        total_angle = 0
        
        for i, (p1_idx, p2_idx, p3_idx) in enumerate(finger_joints):
            angle = self.calculate_angle(
                keypoints[p1_idx][:2],
                keypoints[p2_idx][:2],
                keypoints[p3_idx][:2]
            )
            pose_evaluation['手指弯曲度'][finger_names[i]] = angle
            total_angle += angle
        
        # 基于角度判断整体姿态
        avg_angle = total_angle / 5
        if avg_angle > 160:
            pose_evaluation['整体姿态'] = '完全张开'
            pose_evaluation['评分'] = 90 + (avg_angle - 160) / 2
        elif avg_angle > 120:
            pose_evaluation['整体姿态'] = '半张开'
            pose_evaluation['评分'] = 70 + (avg_angle - 120) / 4
        elif avg_angle > 60:
            pose_evaluation['整体姿态'] = '半握拳'
            pose_evaluation['评分'] = 50 + (avg_angle - 60) / 6
        else:
            pose_evaluation['整体姿态'] = '完全握拳'
            pose_evaluation['评分'] = 30 + avg_angle / 2
        
        # 评分限制在0-100
        pose_evaluation['评分'] = np.clip(pose_evaluation['评分'], 0, 100)
        
        return pose_evaluation

# ====================== 3. 多视角可视化模块 ======================
class MultiViewVisualizer:
    def __init__(self):
        self.fig = None
        self.axs = None
    
    def create_multi_view(self, keypoints):
        """生成骨骼线的四个视角视图"""
        # 创建2x2的子图布局
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('手掌骨骼多视角视图', fontsize=16)
        
        # 定义四个视角的参数
        views = [
            ('正视图', (90, 0), self.axs[0, 0]),   # 正视图 (仰角90°, 方位角0°)
            ('左视图', (0, 90), self.axs[0, 1]),    # 左视图 (仰角0°, 方位角90°)
            ('右视图', (0, -90), self.axs[1, 0]),   # 右视图 (仰角0°, 方位角-90°)
            ('俯视图', (0, 0), self.axs[1, 1])      # 俯视图 (仰角0°, 方位角0°)
        ]
        
        # 手掌骨骼连接关系
        connections = [
            (0,1), (1,2), (2,3), (3,4),    # 拇指
            (0,5), (5,6), (6,7), (7,8),    # 食指
            (0,9), (9,10), (10,11), (11,12),# 中指
            (0,13), (13,14), (14,15), (15,16),# 无名指
            (0,17), (17,18), (18,19), (19,20) # 小拇指
        ]
        
        for view_name, (elev, azim), ax in views:
            # 创建3D坐标轴
            ax3d = self.fig.add_subplot(ax.get_subplotspec(), projection='3d')
            
            # 绘制关键点
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            z = keypoints[:, 2] * 100  # 放大z轴便于观察
            
            ax3d.scatter(x, y, z, c='red', s=50, marker='o')
            
            # 绘制骨骼线
            for (i, j) in connections:
                ax3d.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b-', linewidth=2)
            
            # 设置视角
            ax3d.view_init(elev=elev, azim=azim)
            
            # 设置标题和标签
            ax3d.set_title(view_name)
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Z')
            
            # 隐藏原2D坐标轴
            ax.remove()
        
        plt.tight_layout()
        plt.show()

# ====================== 4. 主程序 ======================
def main(video_path=None):
    """主函数：处理视频并实现所有功能"""
    # 初始化各个模块
    detector = HandPoseDetector()
    evaluator = HandPoseEvaluator()
    visualizer = MultiViewVisualizer()
    
    # 打开视频（如果没有指定路径则使用摄像头）
    cap = cv2.VideoCapture(video_path if video_path else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("无法打开视频/摄像头")
        return
    
    # 设置视频显示窗口
    cv2.namedWindow('手掌姿态分析', cv2.WINDOW_NORMAL)

    
    # 用于存储最新的关键点（用于生成多视角视图）
    latest_keypoints = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测手掌关键点并绘制骨骼线
        annotated_frame, keypoints_list = detector.detect_hand_keypoints(frame)
        
        
        # 如果检测到手掌，进行姿态评估
        if keypoints_list:
            latest_keypoints = keypoints_list[0]  # 取第一个检测到的手掌
            pose_eval = evaluator.evaluate_pose(latest_keypoints)
            
            # ========== 关键修改：使用自定义函数绘制中文 ==========
            # 显示整体姿态
            annotated_frame = draw_chinese_text(
                annotated_frame, 
                f'整体姿态: {pose_eval["整体姿态"]}', 
                (10, 30), 
                font_size=28, 
                color=(0, 255, 0)
            )
            # 显示姿态评分
            annotated_frame = draw_chinese_text(
                annotated_frame, 
                f'姿态评分: {pose_eval["评分"]:.1f}', 
                (10, 70), 
                font_size=28, 
                color=(0, 255, 0)
            )
            # 显示手指弯曲度
            y_offset = 110
            for finger, angle in pose_eval['手指弯曲度'].items():
                annotated_frame = draw_chinese_text(
                    annotated_frame, 
                    f'{finger}: {angle:.1f}°', 
                    (10, y_offset), 
                    font_size=24, 
                    color=(255, 0, 255)
                )
                y_offset += 40
        
        # 显示处理后的画面
        cv2.imshow('手掌姿态分析', annotated_frame)
        
        # 按键控制：
        # 'q'退出，'v'生成多视角视图
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v') and latest_keypoints is not None:
            # 生成多视角视图
            visualizer.create_multi_view(latest_keypoints)
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# ====================== 运行程序 ======================
if __name__ == "__main__":
    # 使用摄像头运行（也可以传入视频路径，如：main("test_video.mp4")）
    main()