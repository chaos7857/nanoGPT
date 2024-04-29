import matplotlib.pyplot as plt

# 模拟训练过程，每个 epoch 结束后更新并绘制损失值
train_losses = []
val_losses = []

# 开启交互模式
plt.ion()

# 创建一个新的图形对象
fig, ax = plt.subplots(figsize=(8, 4))

# 初始化空的线性图
line_train, = ax.plot([], [], 'b-', label='Training Loss')
line_val, = ax.plot([], [], 'r-', label='Validation Loss')

# 设置图形标题和轴标签
ax.set_title('Dynamic Loss Plot')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

# 显示图形
plt.show()

# 模拟训练过程
for epoch in range(1, 11):  # 假设进行10个 epoch 的训练
    # 模拟损失值（实际训练过程中，您需要从模型训练中获取这些值）
    train_loss = 1 / epoch
    val_loss = 1 / (epoch + 0.5)
    
    # 更新损失列表
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 更新图形
    line_train.set_data(range(1, epoch + 1), train_losses)
    line_val.set_data(range(1, epoch + 1), val_losses)
    
    # 设置轴的范围
    ax.set_xlim(0.5, epoch + 0.5)
    ax.set_ylim(0, max(max(train_losses), max(val_losses)) + 0.1)
    
    # 重新绘制图形
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # 短暂暂停以观察
    plt.pause(0.5)

# 关闭交互模式
plt.ioff()
plt.show()
