#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 下午1:12
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_loss.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的HNet损失函数
"""
import numpy as np
import torch

def hnet_loss(gt_pts, transformation_coefficient):
    """

    :param gt_pts: 原始的标签点对 [x, y, 1]
    :param transformation_coeffcient: 映射矩阵参数(6参数矩阵) [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """
    total_loss = torch.zeros(1, requires_grad=True).cuda()

    # print("new set")
    batch_size = transformation_coefficient.size(0)
    for tc in range(batch_size):
        transformation_coeffcient = torch.cat([transformation_coefficient[tc], torch.tensor([1.0], dtype=torch.float32).cuda()], -1).type(torch.double)
        H_indices = torch.tensor([[0], [1], [2], [4], [5], [7], [8]], requires_grad=False).cuda()
        result = torch.zeros(9, dtype=torch.double).cuda()
        result[H_indices[:, 0]] = transformation_coeffcient
        H = torch.reshape(result, shape=[3, 3])
        # print(H)
        # print(gt_pts[tc].shape)
        # label_pts = torch.tensor(gt_pts[tc], dtype=torch.float32).cuda()


        # random_subset = np.random.permutation(len(pts))[:4]
        # pts = np.array(pts)[random_subset, :]
        # print(pts)
        label_pts = gt_pts[tc]
        label_pts = label_pts[label_pts[:,2]==1]
        # print(label_pts)
        # label_pts = label_pts[(label_pts[:,2]==1)]
        label_pts = label_pts.transpose(0, 1)


        # print(label_pts)
        pts_projects = torch.matmul(H, label_pts)

        # print(pts_projects)
        # 求解最小二乘二阶多项式拟合参数矩阵
        Y = (pts_projects[1, :] / pts_projects[2, :])
        X = (pts_projects[0, :] / pts_projects[2, :])
        # print(X)
        # print(Y)
        Y_One = torch.add(Y-Y, torch.tensor(1.0, dtype=torch.float64, requires_grad=False).cuda())
        # print(Y_One)
        Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1).type(torch.double)
        # print(Y_stack)
        # print(Y_stack.transpose(0,1))
        # sol, _ = torch.solve(torch.eye(Y_stack.size(1)).cuda(), torch.matmul(Y_stack.transpose(0,1), Y_stack))
        # print(torch.matmul(Y_stack.transpose(0,1), Y_stack))
        # print(torch.matmul(Y_stack.transpose(0,1), Y_stack).round())
        sol2 = torch.inverse(torch.matmul(Y_stack.transpose(0,1), Y_stack))

        wi = torch.matmul(sol2, Y_stack.transpose(0,1))
        # print("ww", Y_stack.transpose(0,1))
        # print(wi)
        # print( X.unsqueeze(-1))
        w = torch.matmul(wi,
                      X.unsqueeze(-1))
        # 利用二阶多项式参数求解拟合位置并反算到原始投影空间计算损失
        # print("w", w)
        x_preds = torch.matmul(Y_stack, w)

        preds = (torch.stack([torch.squeeze(x_preds, -1) * pts_projects[2, :], Y * pts_projects[2, :], pts_projects[2, :]], dim=1)).transpose(0, 1)
        # print("preds", preds)
        x_transformation_back = torch.matmul(torch.inverse(H), preds)
        #
        # print(torch.inverse(H))
        # print("preds", x_transformation_back[0, :])
        # print("gt", gt_pts[0, :])
        # print(gt_pts[:,0, :])
        # print(x_transformation_back[:,0, :])
        loss = torch.mean(torch.pow(label_pts[0, :] - x_transformation_back[0, :], 2))
        # print(loss)
        # return loss
        total_loss = total_loss + loss

    return total_loss / batch_size


def hnet_transformation(gt_pts, transformation_coeffcient):
    """
    :param gt_pts:
    :param transformation_coeffcient:
    :param name:
    :return:
    """

    transformation_coeffcient = torch.cat(
        [transformation_coeffcient, torch.tensor([1.0], dtype=torch.float32)], -1).type(torch.double)
    multiplier = torch.tensor([1., 1., 4., 1., 4., 0.25, 1.]).type(torch.double)
    transformation_coeffcient = transformation_coeffcient * multiplier
    H_indices = torch.tensor([[0], [1], [2], [4], [5], [7], [8]], requires_grad=False)
    result = torch.zeros(9, dtype=torch.double)
    result[H_indices[:, 0]] = transformation_coeffcient
    H = torch.reshape(result, shape=[3, 3])

    # print(H)
    # print(gt_pts[tc].shape)
    # label_pts = torch.tensor(gt_pts[tc], dtype=torch.float32).cuda()

    # random_subset = np.random.permutation(len(pts))[:4]
    # pts = np.array(pts)[random_subset, :]
    # print(pts)
    label_pts = gt_pts
    label_pts = label_pts[label_pts[:, 2] == 1]
    # print(label_pts)
    # label_pts = label_pts[(label_pts[:,2]==1)]
    label_pts = label_pts.transpose(0, 1)

    # print(label_pts)
    pts_projects = torch.matmul(H, label_pts)


    # 求解最小二乘二阶多项式拟合参数矩阵
    Y = (pts_projects[1, :] / pts_projects[2, :])
    X = (pts_projects[0, :] / pts_projects[2, :])
    # print(X)
    # print(Y)
    Y_One = torch.add(Y - Y, torch.tensor(1.0, dtype=torch.float64, requires_grad=False))
    # print(Y_One)
    Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1).type(torch.double)


    # print(Y_stack)
    sol2 = torch.inverse(torch.matmul(Y_stack.transpose(0, 1), Y_stack))

    # print("mul", torch.matmul(Y_stack.transpose(0, 1), Y_stack))
    wi = torch.matmul(sol2, Y_stack.transpose(0, 1))
    # print("ww", Y_stack.transpose(0,1))
    # print(sol2)
    # print( X.unsqueeze(-1))
    w = torch.matmul(wi,
                     X.unsqueeze(-1))
    # 利用二阶多项式参数求解拟合位置并反算到原始投影空间计算损失
    # print("w", w)
    x_preds = torch.matmul(Y_stack, w)

    preds = (torch.stack([torch.squeeze(x_preds, -1) * pts_projects[2, :], Y * pts_projects[2, :], pts_projects[2, :]], dim=1)).transpose(0, 1)
    # print("preds", preds)
    x_transformation_back = torch.matmul(torch.inverse(H), preds)

    return x_transformation_back, H


def hnet_transformation_np(gt_pts):
    """
    :param gt_pts:
    :param transformation_coeffcient:
    :param name:
    :return:
    """


    H = np.array([[-2.0552388e-01, -2.9437799e+00,  2.9556082e+02],
 [ 0.0000000e+00, -2.8249352e+00,  2.6057458e+02],
 [ 0.0000000e+00, -1.1119454e-02,  1.0000000e+00]])
    # print(H)
    # print(gt_pts[tc].shape)
    # label_pts = torch.tensor(gt_pts[tc], dtype=torch.float32).cuda()

    # random_subset = np.random.permutation(len(pts))[:4]
    # pts = np.array(pts)[random_subset, :]
    # print(pts)
    label_pts = gt_pts
    label_pts = label_pts[label_pts[:, 2] == 1]
    # print(label_pts)
    # label_pts = label_pts[(label_pts[:,2]==1)]
    label_pts = label_pts.T
    print(label_pts)

    # print(label_pts)
    pts_projects = np.matmul(H, label_pts)


    # 求解最小二乘二阶多项式拟合参数矩阵
    Y = (pts_projects[1, :])
    X = (pts_projects[0, :])
    # print(X)
    # print(Y)
    Y_One = np.add(Y - Y, 1)
    # print(Y_One)
    Y_stack = np.stack([Y**3, Y**2, Y, Y_One], axis=1)


    # print(Y_stack)
    sol2 = np.linalg.inv(Y_stack.T @ Y_stack)

    # print("mul", torch.matmul(Y_stack.transpose(0, 1), Y_stack))
    wi = np.matmul(sol2, Y_stack.T)
    # print("ww", Y_stack.transpose(0,1))
    # print(sol2)
    # print( X.unsqueeze(-1))
    w = np.matmul(wi,
                     np.expand_dims(X, axis=-1))
    print(w)
    # 利用二阶多项式参数求解拟合位置并反算到原始投影空间计算损失
    # print("w", w)
    x_preds = np.matmul(Y_stack, w)

    preds = (np.stack([x_preds.ravel(), Y, Y_One], axis=1)).T
    print("preds", preds)
    x_transformation_back = np.matmul(np.linalg.inv(H), preds)

    return x_transformation_back, H





if __name__ == '__main__':
    gt_labels = torch.tensor([[1.0, 1.0, 1.0], [4.0, 2.0, 1.0], [3.0, 3.0, 1.0], [3.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
                            dtype=torch.double)
    transformation_coffecient = torch.tensor([0.58348501, -0.79861236, 2.30343866,
                                              -0.09976104, -1.22268307, 2.43086767],
                                            dtype=torch.float32)
    #
    # import numpy as np
    # c_val = [0.58348501, -0.79861236, 2.30343866,
    #          -0.09976104, -1.22268307, 2.43086767]
    # R = np.zeros([3, 3], np.float32)
    # R[0, 0] = c_val[0]
    # R[0, 1] = c_val[1]
    # R[0, 2] = c_val[2]
    # R[1, 1] = c_val[3]
    # R[1, 2] = c_val[4]
    # R[2, 1] = c_val[5]
    # R[2, 2] = 1
    #
    # print(np.mat(R).I)
    #
    pts, H = hnet_transformation(gt_labels, transformation_coffecient)
    # _loss.backward()
    #
    # _pred = hnet_transformation(gt_labels, transformation_coffecient, 'inference')
    #
    print(pts)
    # print(_pred)

    labels = [[401, 260, 1], [427, 270, 1], [441, 280, 1], [434, 290, 1], [412, 300, 1], [390, 310, 1], [368, 320, 1],
              [347, 330, 1], [325, 340, 1], [303, 350, 1], [277, 360, 1], [247, 370, 1], [216, 380, 1], [185, 390, 1],
              [154, 400, 1], [124, 410, 1], [94, 420, 1], [64, 430, 1], [34, 440, 1], [4, 450, 1], [507, 270, 2],
              [521, 280, 2], [530, 290, 2], [539, 300, 2], [539, 310, 2], [538, 320, 2], [537, 330, 2], [536, 340, 2],
              [534, 350, 2], [530, 360, 2], [521, 370, 2], [512, 380, 2], [504, 390, 2], [495, 400, 2], [486, 410, 2],
              [478, 420, 2], [469, 430, 2], [460, 440, 2], [452, 450, 2], [443, 460, 2], [434, 470, 2], [426, 480, 2],
              [417, 490, 2], [408, 500, 2], [400, 510, 2], [391, 520, 2], [382, 530, 2], [374, 540, 2], [365, 550, 2],
              [355, 560, 2], [346, 570, 2], [337, 580, 2], [328, 590, 2], [318, 600, 2], [309, 610, 2], [300, 620, 2],
              [291, 630, 2], [282, 640, 2], [272, 650, 2], [263, 660, 2], [254, 670, 2], [245, 680, 2], [236, 690, 2],
              [226, 700, 2], [217, 710, 2], [709, 320, 3], [729, 330, 3], [748, 340, 3], [764, 350, 3], [780, 360, 3],
              [795, 370, 3], [811, 380, 3], [827, 390, 3], [842, 400, 3], [855, 410, 3], [868, 420, 3], [881, 430, 3],
              [894, 440, 3], [907, 450, 3], [920, 460, 3], [933, 470, 3], [946, 480, 3], [959, 490, 3], [972, 500, 3],
              [985, 510, 3], [999, 520, 3], [1012, 530, 3], [1025, 540, 3], [1039, 550, 3], [1053, 560, 3],
              [1066, 570, 3], [1080, 580, 3], [1094, 590, 3], [1108, 600, 3], [1122, 610, 3], [1135, 620, 3],
              [1149, 630, 3], [1163, 640, 3], [1177, 650, 3], [1191, 660, 3], [1205, 670, 3], [1218, 680, 3],
              [1232, 690, 3], [1246, 700, 3], [1260, 710, 3], [726, 290, 4], [777, 300, 4], [817, 310, 4],
              [858, 320, 4], [897, 330, 4], [935, 340, 4], [974, 350, 4], [1012, 360, 4], [1050, 370, 4],
              [1087, 380, 4], [1121, 390, 4], [1155, 400, 4], [1189, 410, 4], [1223, 420, 4], [1257, 430, 4]]
    labels = torch.tensor(labels, dtype=torch.double)

    coffecient = torch.tensor([-2.04835137e-01, -3.09995252e+00, 7.99098762e+01, -2.94687413e+00, 7.06836681e+01, -4.67392998e-02],
                             dtype=torch.float32)

    _loss, H = hnet_transformation(labels, coffecient)
    print(_loss)