import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils_jiazhuangxian import save_imgs
import torch.nn.functional as F

# 在文件开头定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 一个函数定义，定义了完成一轮训练所需要的各种参数
def train_one_epoch(train_loader,
                    model,
                    criterion,  # 损失函数
                    optimizer,
                    scheduler,  # 学习率调整器
                    epoch,
                    step,  # 当前轮次训练的步骤数（迭代次数）
                    logger,
                    config,
                    writer,  # tensorboard 用来记录训练过程中的图像、标量等等
                    scaler=None):  # 动态梯度缩放调整
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []  # 初始化一个空列表，记录损失函数值

    for iter, data in enumerate(train_loader):  # 遍历训练数据加载器
        step += iter  # iter 与batch size有关  size大，iter就会少些，平均下来嘛，肯定少很跟多呀  同样step也会减少
        optimizer.zero_grad()  # 清除优化器中的参数梯度，防止梯度积累
        images, targets = data  # 从输入把image 和 mask 给image target

        # 将image targets转移到GPU上，数据类型为float
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:  # 如果使用自动混合精度，就是在训练过程中，自己调整数据的精度
            with autocast():  #
                out = model(images)  # 这里是一个输入，传入两个输入的话（images，name）  传入输入到模型得到输出

                ########  这里加的sigmoid函数实际实在训练过程中，模型在权重反向传播变换的，所以这里加了sigmoid，我认为是会对权重有影响的
                #out = torch.sigmoid(out)
                #######
                #assert torch.all(out >= 0) and torch.all(out <= 1), "Model output out is not in [0, 1] 确保上一步模型输出在0-1之间"
                loss = criterion(out, targets)  # 只使用模型输出的第二部分进行损失计算  将out与target进行损失计算
            scaler.scale(loss).backward()  #
            scaler.step(optimizer)  # 更新优化器状态
            scaler.update()  # 更新梯度
        else:

            out = model(images)  # 传入两个输入

            # 这里上边的输出out他的值可能小于0或者大于1，下边注释了一个断言out值非0-1时会报错，这里加了个sigmoid，源代码是下边注释部分运行提示值超出0-1，这里注释掉了那部分直接用了sigmoid，确保 out[1] 在 [0, 1] 之间  这里加了个sigmoid
            ########
            #out = torch.sigmoid(out)
            #######

            # assert torch.all(out >= 0) and torch.all(out <= 1), "Model output out is not in [0, 1]"
            loss = criterion(out, targets)  # 只使用模型输出的第二部分进行损失计算

            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)  #

        if iter % config.print_interval == 0:  # 每个iter的输出信息
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'  # 将这些信息存到这个列表中
            print(log_info)  # 输出存入的列表信息
            logger.info(log_info)  # 将列表信息存入日志文件中
        scheduler.step()  # 调用学习率调整器，调整学习率
    return step  # 返回迭代次数

def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config,
                  val_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():  # 这里时验证阶段。没有梯度变化，所以加的这个sigmoid不会对模型训练的权重有影响
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)

            # 这里上边的输出out他的值可能小于0或者大于1，下边注释了一个断言out值非0-1时会报错，这里加了个sigmoid，源代码是下边注释部分运行提示值超出0-1，这里注释掉了那部分直接用了sigmoid，确保 out[1] 在 [0, 1] 之间  这里加了个sigmoid
            ########
            #out = torch.sigmoid(out)
            ######
            # assert torch.all(out >= 0) and torch.all(out <= 1), "Model output out is not in [0, 1]"

            loss = criterion(out, msk)
            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TP + TN) / (TP + FP + TN + FN) if float(TP + TN) / (TP + TN + FP + FN) else 0
        # accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float((TP + FP)) != 0 else 0
        # sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        # specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0

        hd95 = 0

        if val_data_name is not None:
            log_info = f'val_datasets_name: {val_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, accuracy: {accuracy}, dice: {dice}, iou: {iou}, recall: {recall}, precision: {precision}  hd95: {hd95}, \
               '  # specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)

            # 这里上边的输出out他的值可能小于0或者大于1，下边注释了一个断言out值非0-1时会报错，这里加了个sigmoid，源代码是下边注释部分运行提示值超出0-1，这里注释掉了那部分直接用了sigmoid，确保 out[1] 在 [0, 1] 之间  这里加了个sigmoid
            ########
            #out = torch.sigmoid(out)
            #######

            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                      test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TP + TN) / (TP + FP + TN + FN) if float(TP + TN) / (TP + TN + FP + FN) else 0
        # accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float((TP + FP)) != 0 else 0
        # sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        # specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        hd95 = 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, accuracy: {accuracy}, dice: {dice}, iou: {iou}, recall: {recall}, precision: {precision}  hd95: {hd95}, \
               '  # specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)