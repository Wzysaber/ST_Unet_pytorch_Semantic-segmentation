# 对相应的参数进行定义
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  #当前值
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  #平均值