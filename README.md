# 图像拼接实验报告

## 一、实验目的
- 掌握图像特征提取与匹配的基本原理和方法，理解 SIFT（尺度不变特征变换）算法基本原理，学习特征描述符的生成和匹配过程。  
- 熟悉图像拼接的技术流程，掌握特征点检测、匹配、单应性矩阵估计和图像融合的完整流程，理解透视变换在图像拼接中的应用。  
- 培养图像处理编程能力，熟练使用 OpenCV 进行图像处理操作，掌握使用 Matplotlib 进行结果可视化展示。  
- 分析算法性能及优化方向，评估不同参数对拼接效果的影响，识别算法在实际应用中的局限性。  

## 二、实验内容
### 2.1 实验环境搭建
配置 Python 开发环境并安装 OpenCV、NumPy、Matplotlib 等必要库。设置中文字体支持解决 Matplotlib 中文显示问题。准备测试图像数据（a.jpg 和 b.jpg）。

![image](a.jpg)
![image](b.jpg)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline
```
### 2.2 图像预处理
读取图像文件并转换为灰度图像，对图像进行必要的预处理操作。
```
# 读取图像文件
img_a = cv2.imread("a.jpg")
img_b = cv2.imread("b.jpg")

# 转换为灰度图像
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 显示原始图像
img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(img_a_rgb)
axes[0].set_title("图像A")
axes[0].axis('off')
axes[1].imshow(img_b_rgb)
axes[1].set_title("图像B")
axes[1].axis('off')
plt.tight_layout()
plt.show()
```
### 2.3 SIFT 特征提取
初始化 SIFT 检测器，提取图像关键点和描述符，分析特征点的分布和数量特征。
```
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)
kp_b, des_b = sift.detectAndCompute(gray_b, None)

print(f"图像A特征点数量: {len(kp_a)}")
print(f"图像B特征点数量: {len(kp_b)}")
```
### 2.4 特征匹配与优化
使用 FLANN 匹配器进行特征匹配，应用 Lowe's 比率测试筛选优质匹配点，可视化特征匹配结果。
```
# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"优质匹配点数量: {len(good_matches)}")

# 绘制匹配的SIFT关键点
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 显示匹配结果
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB))
plt.title("特征点匹配结果")
plt.axis('off')
plt.show()
```
### 2.5 单应性矩阵估计
使用 RANSAC 算法估计两幅图像间的透视变换关系，计算单应性矩阵 H 描述图像间的坐标映射关系。
```
# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法估计单应矩阵（透视变换矩阵）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(f"单应矩阵:\n{H}")
```
### 2.6 图像拼接与融合
基于单应性矩阵进行透视变换，计算拼接后图像尺寸并进行图像融合，实现无缝的图像拼接效果。
```
# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像变换后的四个角坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸（包含所有像素）
all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() + 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

print(f"拼接图像尺寸: {x_max - x_min} x {y_max - y_min}")

# 创建平移矩阵，确保所有像素都在可视区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 对图像进行透视变换和平移
fus_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,  # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
)

# 将图像A复制到拼接结果的对应位置
fus_img[-y_min:h_a - y_min, -x_min:w_a - x_min] = img_a
```
### 2.7 结果可视化与分析
展示原始图像、特征匹配结果和最终拼接效果，分析拼接质量并评估算法性能。
```
# 显示所有结果
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 原始图像A
axes[0, 0].imshow(img_a_rgb)
axes[0, 0].set_title("图像A (原始)")
axes[0, 0].axis('off')

# 原始图像B
axes[0, 1].imshow(img_b_rgb)
axes[0, 1].set_title("图像B (原始)")
axes[0, 1].axis('off')

# 特征匹配
axes[1, 0].imshow(cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("特征点匹配")
axes[1, 0].axis('off')

# 拼接结果
fus_img_rgb = cv2.cvtColor(fus_img, cv2.COLOR_BGR2RGB)
axes[1, 1].imshow(fus_img_rgb)
axes[1, 1].set_title("拼接结果")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
```
## 三、实验结果与分析
### 3.1 特征提取结果分析
| 图像 | 特征点数量 | 特征分布特点 |
|------|--------------|--------------|
| 图像 A | 115 个 | 特征点数量相对较少 |
| 图像 B | 221 个 | 特征点数量适中，约为图像 A 的 2 倍 |

**分析：**  
图像 A 的特征点数量较少（115 个）可能因图像纹理简单或对比度较低；图像 B 的特征点数量（221 个）相对较多，表明该图像包含更丰富的纹理信息。两幅图像特征点数量差异较大可能影响后续匹配的平衡性。

### 3.2 特征匹配效果评估
| 匹配阶段 | 匹配数量 | 质量评价 |
|----------|------------|------------|
| 初始匹配 | 未记录 | - |
| Lowe's 筛选后 | 20 对 | 匹配数量较少 |

**关键发现：**  
经过 Lowe's 比率测试筛选后仅获得 20 对优质匹配点。匹配点数量较少的可能原因包括两幅图像重叠区域有限、图像间存在较大的视角或光照变化、图像纹理特征不够丰富。尽管匹配点数量不多，但已满足单应性矩阵估计的最低要求（至少 4 对匹配点）。

### 3.3 单应性矩阵分析
获得的单应性矩阵为：  
```
[[ 1.20313065e+00  3.59206907e-02 -8.14270607e+01]
 [-1.61219793e-02  1.18897706e+00 -4.07578920e+01]
 [-2.00421305e-05  2.02702983e-05  1.00000000e+00]]
```

**矩阵特性分析：**  
左上角 2×2 子矩阵接近单位矩阵，表明图像间旋转角度较小。存在明显的平移分量（-81.43, -40.76）说明图像间有显著位移；透视变换分量较小（最后一行前两个元素接近 0）表明视角变化不大。

### 3.4 图像拼接质量评估
**技术指标：**
- 拼接图像尺寸：337 × 351 像素  
- 特征匹配数量：20 对  
- 单应性矩阵估计：成功收敛  

**拼接效果分析：**
- 成功方面：算法成功计算出单应性矩阵并完成图像变换，生成了一定尺寸的拼接图像（337×351），基本实现了图像对齐和融合。  
- 可能存在的问题：匹配点数量有限可能导致拼接精度不高，可能存在对齐误差或拼接缝隙，对于复杂场景的适应性有待验证。  

### 3.5 算法性能分析
**优势表现：**
- 基础功能实现：在匹配点数量有限的情况下仍完成了拼接任务。  
- 鲁棒性：RANSAC 算法在少量匹配点情况下仍能估计合理的单应性矩阵。  
- 实用性：算法流程完整，具备实际应用的基础。  

**局限性：**
- 特征提取不足：图像 A 特征点数量较少影响匹配效果。  
- 匹配效率低：仅获得 20 对匹配点可能影响拼接精度。  
- 场景适应性弱：对于低纹理或高相似度场景表现不佳。  

## 四、实验小结
### 4.1 实验成果总结
本次实验成功实现了基于 SIFT 特征的图像自动拼接系统。主要成果包括：  
- **技术实现方面：** 完整实现了从特征提取到图像融合的全流程，在匹配点数量有限（20 对）的情况下仍完成了拼接任务，获得了 337×351 像素的拼接结果图像。  
- **算法理解方面：** 深入理解了 SIFT 特征提取的原理和特点，掌握了特征匹配和误匹配剔除的策略，理解了单应性矩阵在图像配准中的作用和计算方法。  
- **实践能力方面：** 提升了图像处理编程能力和参数调优技巧，培养了在非理想条件下的问题解决能力。  

### 4.2 经验与启示
通过本次实验获得以下重要经验：  
- 特征质量的重要性：特征点数量和质量直接影响拼接效果。  
- 参数调优的必要性：合适的阈值设置对匹配质量有显著影响。  
- 算法鲁棒性的价值：即使在匹配点较少时，RANSAC 等鲁棒算法仍能提供可用结果。  

### 4.3 改进方向
- **特征提取优化：** 调整 SIFT 参数（contrastThreshold、edgeThreshold）提高特征点数量，或使用 ORB、AKAZE 等特征检测算法。  
- **匹配策略改进：** 优化 Lowe’s 比率测试阈值，尝试双向匹配，加入几何一致性约束剔除误匹配。  
- **应用场景拓展：** 针对特定场景优化参数，开发多尺度特征提取策略，结合边缘或色彩信息辅助匹配。  

### 4.4 实验价值
尽管匹配点数量有限，本实验仍完整实现了图像拼接流程，并深入理解了算法原理与局限性。这种在非理想条件下的实践经验对开发鲁棒的计算机视觉系统具有重要意义。图像拼接技术作为计算机视觉的基础，在虚拟现实、医学影像、遥感测绘等领域具有广泛应用前景。本实验为后续研究和应用开发奠定了基础。
