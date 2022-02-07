from numba import jit
import numpy as np
# 用于线性分配，匈牙利匹配的实现
from scipy.optimize import linear_sum_assignment
# 卡尔曼滤波器
from filterpy.kalman import KalmanFilter


@jit
def iou(bb_test, bb_gt):
    """
    计算交并比, 坐标为框的左上角和右下角坐标
    :param bb_test: [x1 y1 x2 y2]
    :param bb_gt: [x1 y1 x2 y2]
    :return: IOU
    """
    # 获得相交四边形坐标
    x1 = np.maximum(bb_test[0], bb_gt[0])
    y1 = np.maximum(bb_test[1], bb_gt[1])
    x2 = np.minimum(bb_test[2], bb_gt[2])
    y2 = np.minimum(bb_test[3], bb_gt[3])

    # 相交面积
    inter_area = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)

    # 相并面积
    test_area = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    gt_area = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    # 计算交并比
    test_gt_iou = inter_area / (test_area + gt_area - inter_area)

    return test_gt_iou


def convert_bbox_to_z(bbox):
    """
    :param bbox: [x1 y1 x2 y2] 分别是左上角坐标和右下角坐标
    :return: [x y s r], 其中x,y是box中心位置的坐标，s是面积，r是宽高比w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape(4, 1)


def convert_x_to_bbox(x, score=None):
    """
    :param x: [x y s r], 其中x,y是box中心位置的坐标，s是面积，r是宽高比w/h
    :param score: 置信度
    :return: [x1 y1 x2 y2] 分别是左上角坐标和右下角坐标
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape(1, 4)
    else:
        return np.array([x1, y1, x2, y2, score]).reshape(1, 5)


class KalmanBoxTracker(object):
    # 记录跟踪框个数
    count = 0

    def __init__(self, bbox):
        """
        初始化边界框和跟踪器
        :param bbox: 边界框
        """

        # 等速模型，7个状态变量和4个观测输入
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # F：状态转移矩阵，7*7，用当前矩阵预测下一时刻状态
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        # H：量测矩阵，4*7
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # P、Q、R是根据经验值设定的
        # P = diag([10, 10, 10, 10, 1e4, 1e4, 1e4])
        # Q = diag([1, 1, 1, 1, 0.01, 0.01, 1e-4])
        # R = diag([1, 1, 10, 10])
        # P：先验估计的协方差
        self.kf.P[4:, 4:] *= 1000  # 设置一个较大的值，给无法观测的初始速度带来很大的不确定性
        self.kf.P *= 10
        # Q：过程激励噪声的协方差
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # R：测量噪声的协方差，即为真实值与测量值差的协方差
        self.kf.R[2:, 2:] *= 10

        # x：观测结果、状态估计
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # 连续预测的次数，每预测一次time_since_update+=1
        # 连续预测时，一旦更新则置time_since_update=0
        # 连续预测时，只要time_since_update>0，则置hit_streak=0，表示连续预测的过程中没有进行update
        self.time_since_update = 0

        # 卡尔曼滤波器的个数，即目标框个数
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 用于保存单个目标连续预测的多个结果，一旦update就会清空history
        # 要保存convert_x_to_bbox的格式
        self.history = []

        # 该目标框进行更新的总次数，每更新一次，hits+=1
        self.hits = 0

        # 连续更新的次数，每update一次hit_streak+=1，
        self.hit_streak = 0

        # 该目标框进行预测的总次数，每预测一次，age+=1
        self.age = 0

    def update(self, bbox):
        """
        使用观测到的目标框更新状态向量
        :param bbox: 目标框
        :return: None
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计
        :return: history[-1] 边界框
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()  # 进行目标框的预测
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        获得当前边界框的预测结果
        :return: 边界框估计值
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshole=0.3):
    """
    将YOLO模型的检测框和卡尔曼滤波的跟踪框进行关联匹配
    :param detections: yolov3的检测框
    :param trackers: 卡尔曼滤波跟踪结果
    :param iou_threshole: IOU阈值
    :return: 跟踪成功的目标矩阵 matches
             新增目标的矩阵：unmatched_detections
             跟踪失败（离开画面）的目标矩阵：unmatched_trackers
    """

    # 跟踪/检测目标为0时，直接返回结果
    if len(trackers) == 0 or len(detections) == 0:
        matches = np.empty((0, 2), dtype=int)
        unmatched_detections = np.arange(len(detections))
        unmatched_trackers = np.empty((0, 5), dtype=int)
        return matches, unmatched_detections, unmatched_trackers

    # 跟踪/检测目标不为0时
    # IOU不支持数组计算，故IOU要逐个进行交并比计算，构造矩阵调用linear_sum_assignment进行匹配
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    # 遍历目标检测的bbox集合，每个检测框的标识为d
    for d, det in enumerate(detections):
        # 遍历跟踪框（卡尔曼滤波器预测）bbox集合，每个跟踪框标识为t
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    # 通过匈牙利算法（linear_sum_assignment）
    # 将跟踪框和检测框以[[d,t]...]的二维矩阵的形式存储在match_indices中
    # linear_sum_assignment找到的是最小的匹配，此处要取负值就可以找到iou最大的值
    result = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*result)))

    # 记录未匹配的检测框及跟踪框
    # 未匹配的检测框放入unmatched_detections中
    # 表示有新的目标进入画面，要新增跟踪器跟踪目标
    unmatched_detections = []
    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    # 未匹配的跟踪框放入unmatched_trackers中
    # 表示目标离开之前的画面，应删除对应的跟踪器
    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    # 将匹配成功的跟踪框放入matches中
    matches = []
    for m in matched_indices:
        # 过滤掉IOU低的匹配，将其放入到unmatched_detections和unmatched_trackers中
        if iou_matrix[m[0], m[1]] < iou_threshole:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            # 满足条件的以[[d,t]...]的形式放入matches中
            matches.append(m.reshape(1, 2))

    # 格式转换，以np.array形式返回
    if len(matches) == 0:
        matches = np.array((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    """
    Sort：多目标跟踪器的管理类，跟踪多个KalmanBoxTracker对象
    """

    def __init__(self, max_age=1, min_hits=3):
        """
        :param max_age: 目标未被检测到的帧数，连续N帧超过的话，从卡尔曼滤波器对象中删除该跟踪框
        :param min_hits: 目标连续命中的最小次数，至少需要连续min_hits命中才能成功追踪到目标
        """
        self.max_age = max_age
        self.min_hits = min_hits
        # 卡尔曼滤波跟踪器链，存储多个KalmanBoxTracker对象
        self.trackers = []
        # 帧计数，表示当前视频经过了多少帧的计数
        self.frame_count = 0

    def update(self, dets):
        """
        实现SORT算法
        输入dets：
            当前帧中yolo所检测出的所有目标的检测框的集合，包含每个目标的score
            以[[x1,y1,x2,y2,score]，[x1,y1,x2,y2,score]，...]形式输入的np.array
            x1、y1 代表检测框的左上角坐标；x2、y2代表检测框的右下角坐标；score代表检测框对应预测类别的概率值。
        输出ret：
            当前帧中跟踪目标成功的跟踪框/预测框的集合，包含目标的跟踪的id(也即该跟踪框(卡尔曼滤波实例对象)是创建出来的第几个)
            [[左上角的x坐标, 左上角的y坐标, 右下角的x坐标, 右下角的y坐标, trk.id] ...]
            trk.id：卡尔曼滤波器的个数/目标框的个数，也即该跟踪框(卡尔曼滤波实例对象)是创建出来的第几个。
        注意：
            即使检测框为空，也必须对每一帧调用此方法，返回一个类似的输出数组，最后一列是目标对像的id。
            返回的目标对象数量可能与检测框的数量不同。
        :param dets: 以[[x1,y1,x2,y2,score]，[x1,y1,x2,y2,score]，...]
        :return: 如上述
        """
        # 每经过一帧，frame_count加1
        self.frame_count += 1

        # 根据当前所有的卡尔曼跟踪器的个数（上一帧跟踪的目标个数）创建二维矩阵
        # 行为卡尔曼滤波器的索引，列为跟踪框的位置和id
        trks = np.zeros((len(self.trackers), 5))  # 跟踪器对当前帧的预测结果
        # 存储要战术的目标框
        to_del = []
        # 存储要返回的追踪目标框
        ret = []
        # 循环遍历卡尔曼跟踪器列表
        for t, trk in enumerate(trks):
            # 上一帧中的所有跟踪框在当前帧中进行预测新的跟踪框
            # 使用卡尔曼跟踪器t产生对应目标的跟踪框，即对目标进行预测
            pos = self.trackers[t].predict()[0]
            # pos中存储：左上角的x坐标和y坐标、右下角的x坐标和y坐标、置信度 的一共5个值
            # 遍历完成后，trk中存储了上一帧中跟踪的目标在这一帧的预测跟踪框
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果pos中包含空值，则将该跟踪框删除
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # numpy.ma.masked_invalid 屏蔽出现的无效值（NaN 或 inf）
        # numpy.ma.compress_rows 将包含掩码值的整行去除
        # 最终跟踪器链trks矩阵只包含上一帧中的跟踪器链中所有跟踪框在当前帧中成功进行预测的新跟踪框
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # 逆向删除异常的跟踪器
        # 逆向是为了防止破坏索引，正向删除的话元素会向前补齐，导致后面的索引异常
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 使用匈牙利算法：将目标检测框和卡尔曼滤波器预测的跟踪框进行匹配，分别获取跟踪成功的目标，新增的目标，离开画面的目标
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # 将追踪成功的目标框更新到对应的卡尔曼滤波器
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                # 找到索引t对应的行，取出第0列的值（检测框索引值）
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # 使用检测框更新状态向量x，即使用检测框更新跟踪框
                trk.update(dets[d, :][0])

        # 为新增的目标创建新的卡尔曼滤波器对象进行目标追踪
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits的跟踪结果
        # 如果未命中时间大于self.max_age则删除跟踪器。
        # hit_streak忽略目标初始的若干帧
        # i为trackers跟踪器链(列表)长度，每遍历一次i-1
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框的估计值
            d = trk.get_state()[0]
            # 追踪成功目标的box与id放入ret列表中
            if trk.time_since_update < 1 and \
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # 跟踪失败或离开画面的目标从卡尔曼跟踪器中删除
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            # ret：[[左上角x, 左上角y, 右下角x, 右下角y, trk.id]
            #       [...]
            #       [...]]
            return np.concatenate(ret)
        return np.empty((0, 5))
