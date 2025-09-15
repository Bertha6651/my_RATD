import torch
import numpy as np
from TCN.word_cnn.model import TCN  # 导入TCN模型定义
import datautils
from tqdm import tqdm
import argparse
import yaml


def all_retrieval(model, config):
    """
    功能：根据训练数据和已保存的编码向量，检索最相似的历史片段
    """
    L = config["retrieval"]["L"]   # 输入序列长度
    H = config["retrieval"]["H"]   # 预测序列长度
    num = config["retrieval"]["num"]  # 每个片段要检索的参考数量

    # 加载训练集（电力数据）
    train_set = datautils.Dataset_Electricity(
        root_path=config["path"]["dataset_path"],
        data_path="electricity_2012_hour.csv",
        flag='train',
        size=[L, 0, L]
    )

    # 读取之前保存的历史向量（由 all_encode 生成）
    all_repr = torch.load('./data/TCN/ele_hisvec_list.pt')

    references = []
    with torch.no_grad():  # 推理时关闭梯度，节省显存和计算量
        for i in tqdm(range(len(train_set) - L - H + 1)):
            # 取长度为L的片段
            x = train_set.data_x[i:i+L]
            x = x[np.newaxis, :, :]  # 增加 batch 维度
            x = torch.tensor(x,dtype=torch.float32).transpose(1, 2).to(config["retrieval"]["device"])  # 转换成张量，调整维度 (batch, channel, length)

            # 使用模型得到向量表示
            x_vec = model(x)

            # 展平为一维向量
            k, l = x_vec.shape[-2], x_vec.shape[-1]
            x_vec = x_vec.reshape(1, k * l)
            all_repr = all_repr.reshape(-1, k * l)

            # 计算欧式距离并找到最相似的 num 个片段
            distances = torch.norm(x_vec.cpu() - all_repr, dim=1)
            _, idx = torch.topk(-1 * distances, num)  # 越小越相似，所以取 -1*distances
            references.append(idx.int())

        # 保存检索结果
        references = torch.cat(references, dim=0)
        torch.save(references, config["path"]["ref_path"])
    return references


def all_encode(model, config):
    """
    功能：将训练数据集编码成向量，保存下来供后续检索使用
    """
    hisvec_list = []
    L = config["retrieval"]["L"]
    H = config["retrieval"]["H"]

    train_set = datautils.Dataset_Electricity(
        root_path=config["path"]["dataset_path"],
        data_path="electricity_2012_hour.csv",
        flag='train',
        size=[L, 0, L]
    )

    with torch.no_grad():
        for i in tqdm(range(len(train_set) - L - H + 1)):
            x = train_set.data_x[i:i+L]
            x = x[np.newaxis, :, :]
            x = torch.tensor(x,dtype=torch.float32).transpose(1, 2).to(config["retrieval"]["device"])

            # 编码为向量
            x_vec = model(x)
            hisvec_list.append(x_vec.cpu())

    # 拼接并保存编码向量
    hisvec_list = torch.cat(hisvec_list, dim=0)
    torch.save(hisvec_list.float(), config["path"]["vec_path"])


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Retrieval with TCN Encoder")
    parser.add_argument("--config", type=str, default="retrieval_ele.yaml", help="配置文件路径")
    parser.add_argument("--modelfolder", type=str, default="", help="模型文件夹（可选）")
    parser.add_argument("--type", type=str, default="encode", choices=["encode", "retrieval"], help="运行模式：encode 或 retrieval")
    parser.add_argument("--encoder", default="TCN", help="选择编码器类型，目前支持 TCN")

    args = parser.parse_args()
    print(args)

    # 加载配置文件
    path = "./TCN/config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["retrieval"]["encoder"] = args.encoder

    # 1. 构建模型
    model = TCN(
        input_size=config["retrieval"]["length"],
        output_size=config["retrieval"]["length"],
        num_channels=[config["retrieval"]["length"]] * config["retrieval"]["level"] + [config["retrieval"]["length"]],
    ).to(config["retrieval"]["device"])

    # 2. 加载权重（推荐方式：state_dict）
    model = torch.load(config["path"]["encoder_path"], map_location='cpu', weights_only=False)
    model.to(config["retrieval"]["device"])

    # 3. 执行对应的任务
    if args.type == 'encode':
        all_encode(model, config)
    elif args.type == 'retrieval':
        all_retrieval(model, config)
