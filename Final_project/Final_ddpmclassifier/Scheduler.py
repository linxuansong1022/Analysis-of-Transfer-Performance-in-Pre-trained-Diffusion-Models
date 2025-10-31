from torch.optim.lr_scheduler import _LRScheduler

'''
GradualWarmupScheduler 实现了一个学习率预热策略。在训练开始的 warm_epoch 个周期内，
学习率从优化器设置的初始值线性增加到 multiplier 倍。预热结束后，如果指定了 after_scheduler（例如余弦退火调度器），则后续的学习率将由 after_scheduler 控制，
并且 after_scheduler 的起始学习率会被设置为预热结束时的学习率。
如果没有指定 after_scheduler，学习率将在预热结束后保持在 multiplier 倍的初始学习率不变。
'''
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        # 参数说明:
        # optimizer: 关联的优化器对象 (例如 AdamW)
        # multiplier: 预热结束时，学习率相较于初始学习率要达到的倍数。如果 > 1，则学习率在预热期间逐渐升高；如果 = 1，则预热期间学习率不变（通常不这样用）；如果 < 1，则逐渐降低（也很少见）。
        # warm_epoch: 预热持续的 epoch 数量。
        # after_scheduler: (可选) 在预热结束后要继续使用的另一个学习率调度器 (例如 CosineAnnealingLR)。
        self.multiplier = multiplier# 将传入的 multiplier 存储为实例变量
        self.total_epoch = warm_epoch# 将传入的 warm_epoch 存储为实例变量
        self.after_scheduler = after_scheduler# 将传入的 after_scheduler 存储为实例变量
        self.finished = False# 初始化一个标志变量，用于跟踪是否已经完成预热 
        self.last_epoch = None# 初始化一个变量，用于存储最后一个 epoch 的索引
        self.base_lrs = None# 初始化一个变量，用于存储基础学习率    
        super().__init__(optimizer)# 调用父类 _LRScheduler 的初始化方法，将优化器传递给父类

    def get_lr(self):# 计算当前 epoch 的学习率
        if self.last_epoch > self.total_epoch:# 如果当前 epoch 大于预热持续的 epoch 数量
            if self.after_scheduler:# 如果存在另一个学习率调度器
                if not self.finished:# 如果还没有完成预热
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]# 更新另一个学习率调度器的初始学习率
                    self.finished = True# 设置标志变量为 True，表示预热已完成
                return self.after_scheduler.get_lr()# 返回另一个学习率调度器计算的学习率
            return [base_lr * self.multiplier for base_lr in self.base_lrs]# 如果预热结束，直接返回基础学习率乘以 multiplier
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]# 返回当前 epoch 的学习率


    def step(self, epoch=None, metrics=None):# 更新学习率
        if self.finished and self.after_scheduler:# 如果预热已完成，并且存在另一个学习率调度器
            if epoch is None:# 如果 epoch 为 None
                self.after_scheduler.step(None)# 调用另一个学习率调度器的 step 方法，传递 None
            else:
                self.after_scheduler.step(epoch - self.total_epoch)# 调用另一个学习率调度器的 step 方法，传递当前 epoch 减去预热持续的 epoch 数量
        else:
            return super(GradualWarmupScheduler, self).step(epoch)# 调用父类 _LRScheduler 的 step 方法，传递当前 epoch  