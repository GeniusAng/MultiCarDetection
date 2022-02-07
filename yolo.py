from kalman import *
import time
import cv2
import numpy as np

# 虚拟线圈设置
line = [(0, 100), (2560, 100)]
# 车辆总数初始化
counter = 0
# 正向车道的车辆数
counter_up = 0
# 逆向车道的车辆数
counter_down = 0

# 创建跟踪器对象
tracker = Sort()
# 初始化存放跟踪结果的字典
memory = {}


# 线与线的碰撞检测：利用叉乘判断两条线是否相交
# 计算叉乘符号
def ccw(A, B, C):
    # 可以看成AB变换到AC是顺时针还是逆时针，逆时针为True
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# 利用yolov3模型进行目标检测
# 加载模型相关信息

# 加载可以检测的目标的类型
labelPath = "./yolo-coco/coco.names"
LABELS = open(labelPath).read().strip().split('\n')

# 生成多种不同的颜色的检测框 用来标注物体
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')

# 加载预训练的模型：权重、配置信息
weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# 获取yolo中每一层的名称
ln = net.getLayerNames()
# 获取输出层的名称：['yolo_82', 'yolo_94', 'yolo_106'
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# # 读取图像
# frame = cv2.imread('./images/car1.jpg')
# # print(frame.shape)  # (720, 1280, 3)
# H, W = frame.shape[:2]

# 读取视频
vs = cv2.VideoCapture('./input/test_1.mp4')
W, H = None, None
# 视频文件写对象
writer = None

# 获取视频的总帧数
prop = cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))
print("[INFO] {} total frames in video".format(total))

# 遍历每一帧图像
while True:
    # 读取帧：grabbed是bool,表示是否成功捕获帧，frame是捕获的帧
    grabed, frame = vs.read()
    # 若未捕获帧，则退出循环
    if not grabed:
        break
    # 若W和H为空，则将第一帧画面的大小赋值给他
    if W is None or H is None:
        H, W = frame.shape[:2]

    # 将图片构建成一个blob，设置图片尺寸，进行前向传播
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # print(blob.shape)  # (1, 3, 416, 416)
    # outimg = np.transpose(blob[0], (1, 2, 0))
    # plt.imshow(outimg)
    # plt.show()

    # 将blob送入网络
    net.setInput(blob)
    start = time.time()
    # 前向传播，进行预测，返回目标框的边界和置信度
    layerOutputs = net.forward(ln)
    # 三种尺度共10647（507+2028+8112）个检测结果
    # print(layerOutputs[0].shape)  # (507, 85) 507=13*13*3
    # print(layerOutputs[1].shape)  # (2028, 85) 2028=26*26*3
    # print(layerOutputs[2].shape)  # (8112, 85) 8112=52*52*3
    end = time.time()

    # 下面对网络输出的bbox进行检查：
    # 判定每一个bbx的置信度是否足够的高，以及执行NMS算法去除冗余的bbox

    # 存放目标的检测框 x,y,h,w
    boxes = []
    # 置信度
    confidences = []
    # 目标类别
    classIDs = []

    # 遍历每一个输出层的输出
    for output in layerOutputs:
        # 遍历检测结果
        for d in output:
            # d : 1*85 0-3位置，4置信度 5-84类别
            scores = d[5:]  # 当前目标属于某一类别的概率
            classID = np.argmax(scores)  # 目标的类别ID
            confidence = scores[classID]  # 得到目标属于该类别的置信度

            if confidence > 0.3:
                # 将检测结果与原图片匹配，yolo的输出的是边界框的中心坐标和宽高，是对应图片的比例
                box = d[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype('int')
                # 左上角坐标
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                # 更新目标框、置信度、类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # 非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 检测框：左上角和右下角坐标
    dets = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == 'car':
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                dets.append([x, y, x + w, y + h, confidences[i]])

    # 设置数据类型，将整型数据转换为浮点数类型，且保留小数点后三位
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # 将检测框数据转换为ndarray,其数据类型为浮点
    dets = np.asarray(dets)
    # print(dets)
    # 显示
    # plt.imshow(frame[:, :, ::-1])
    # plt.show()

    # SORT目标跟踪
    # yolo无检测结果，未检测到目标时不进行目标追踪
    if np.size(dets) == 0:
        continue
    else:
        # 将检测结果传入跟踪器中，返回当前画面中跟踪成功的目标
        # 包含五个信息：目标框的左上角和右下角横纵坐标，目标的id
        tracks = tracker.update(dets)
    # 跟踪框
    boxes = []
    # id的索引
    indexIDs = []
    # 前一帧的跟踪结果
    previous = memory.copy()  # 用于存放上一帧的跟踪结果，用于碰撞检测
    memory = {}  # 是一个字典：存放当前帧目标的跟踪结果，用于碰撞检测
    for track in tracks:
        # 更新目标框坐标信息
        boxes.append([track[0], track[1], track[2], track[3]])
        # 更新id
        indexIDs.append(int(track[4]))
        # 设置memory字典，键为id，值为此id对应的跟踪框
        memory[indexIDs[-1]] = boxes[-1]

    # 碰撞检测
    if len(boxes) > 0:
        i = int(0)
        # 遍历跟踪框
        for box in boxes:
            # 左上角坐标和右下角坐标
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[2]), int(box[3])
            # # 对方框的颜色进行设定
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # 将方框绘制在画面上
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 根据当前帧的检测结果，与上一帧的检测结果结合，通过虚拟线圈完成车流量统计
            if indexIDs[i] in previous:
                #  获取上一帧识别的目标框
                previous_box = previous[indexIDs[i]]
                # 获取上一帧画面追踪框的左上角坐标和宽高
                x3, y3 = int(previous_box[0]), int(previous_box[1])
                x4, y4 = int(previous_box[2]), int(previous_box[3])
                # 获取当前帧检测框的中心点
                p0 = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                # 获取上一帧检测框的中心点
                p1 = (int(x3 + (x4 - x3) / 2), int(y3 + (y4 - y3) / 2))
                # 进行碰撞检测
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1  # 总计数加1
                    # 判断行进的方向
                    if y3 < y1:
                        counter_down += 1  # 逆向行驶+1
                    else:
                        counter_up += 1  # 正向行驶+1
            i += 1
    # 将车辆计数的相关结果绘制在视频上
    # 根据设置的基准线将其绘制在画面上
    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    # 绘制车辆的总计数
    cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    # 绘制车辆正向行驶的计数
    cv2.putText(frame, str(counter_up), (850, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    # 绘制车辆逆向行驶的计数
    cv2.putText(frame, str(counter_down), (350, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    if writer is None:
        # 设置编码格式
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 视频信息设置
        writer = cv2.VideoWriter("./output/output.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
writer.release()
vs.release()
cv2.destroyAllWindows()
