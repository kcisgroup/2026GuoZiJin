import os
from PIL import Image
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32")
# print(model)
model.cuda().eval()

prompt = "masterpiece, best quality, high res,1girl, Asian beauty, " \
         " soft natural makeup," \
         "delicate facial features,glossy lips, smooth skin texture,"
prompt='a man stand in the room'


# 准备你的图像和文本
#your_image_folder = "/home/gzj/test/BrushNetSimple-main/out/blendbeauty/"  # 替换为你的图像文件夹路径

#tokenflow
your_image_folder = "/home/gzj/test/BrushNetSimple-main/out/20250401_191118/bg/"
your_texts = [prompt]  # 替换为你的文本列表


def compute_global_similarity(features):
    cov_matrix=np.cov(features.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    return np.mean(eigenvalues)

images = []
for filename in os.listdir(your_image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(your_image_folder, filename)
        image = Image.open(path).convert("RGB")
        images.append(preprocess(image))

# 图像和文本预处理
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(your_texts).cuda()

# 计算特征
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

# 计算相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)
print('similarity:',similarity)

# 提取相似度向量并计算方差
similarity_scores = similarity.squeeze(0)  # 降维为[N]
similarity_mean = np.mean(similarity_scores)
similarity_var = np.var(similarity_scores)  # 方差计算
print('similarity_mean:',similarity_mean)
print('similarity_var:',similarity_var)
'''
# 可视化相似度
plt.imshow(similarity, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.xlabel("Images")
plt.ylabel("Texts")
plt.title("Similarity between Texts and Images")
plt.xticks(range(len(images)), range(len(images)), rotation=90)
plt.yticks(range(len(your_texts)), your_texts, rotation='vertical', va='center')  # 设置标签竖向显示并居中

#save_path = os.path.join(your_image_folder, "similarity.png")
save_path = '/home/gzj/test/BrushNetSimple-main/out/similarity.png'

# 保存图像到文件
plt.savefig(save_path, bbox_inches='tight')  # 确保所有内容都在保存的图片里

# 显示图像
plt.show()
'''
