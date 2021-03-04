import numpy as np
from PIL import Image
import os
import math

def similarity(a, b):
    x = a[:, -1, :]
    y = b[:, 0, :]
    return -np.sum((x - y)**2)

for root, dirs, files in os.walk('shredded-images'):
    break

for dir in dirs:
    if dir[:7] != 'simple_':
        continue
    print(dir)
    for subroot, subdirs, subfiles in os.walk(root + '/' + dir):
        break
    max_hight = 0
    total_width = 0
    for file in subfiles:
        im = Image.open(subroot + '/' + file)
        width, height = im.size
        if height > max_hight:
            max_hight = height
        total_width += width

    shreds = [np.array(Image.open(subroot + '/' + file)) for file in subfiles]
    edges = [[-math.inf]*len(shreds) for i in range(len(shreds))]
    leftmax = [-math.inf]*len(shreds)
    rightmax =[-math.inf]*len(shreds)
    left_neighbor = [None]*len(shreds)
    right_neigbor = [None]*len(shreds)
    for i in range(len(shreds)):
        for j in range(i+1, len(shreds)):
            edges[i][j] = similarity(shreds[i], shreds[j])
            if edges[i][j] > rightmax[i]:
                rightmax[i] = edges[i][j]
                right_neigbor[i] = j
            if edges[i][j] > leftmax[j]:
                leftmax[j] = edges[i][j]
                left_neighbor[j] = i
            edges[j][i] = similarity(shreds[j], shreds[i])
            if edges[j][i] > rightmax[j]:
                rightmax[j] = edges[j][i]
                right_neigbor[j] = i
            if edges[j][i] > leftmax[i]:
                leftmax[i] = edges[j][i]
                left_neighbor[i] = j

    overlap_left = set()
    overlap_right = set()
    left_flag, right_flag = None, None
    for i in range(len(shreds)):
        if left_neighbor[i] not in overlap_left:
            overlap_left.add(left_neighbor[i])
        else:
            left_flag = left_neighbor[i]
        if right_neigbor[i] not in overlap_right:
            overlap_right.add(right_neigbor[i])
        else:
            right_flag = right_neigbor[i]

    left_edge = []
    right_edge = []
    for i in range(len(shreds)):
        if left_neighbor[i] == left_flag:
            left_edge.append(i)
        if right_neigbor[i] == right_flag:
            right_edge.append(i)
    if len(left_edge):
        if leftmax[left_edge[0]] > leftmax[left_edge[1]]:
            leftmax[left_edge[1]] = -math.inf
            left_neighbor[left_edge[1]] = None
        else:
            leftmax[left_edge[0]] = -math.inf
            left_neighbor[left_edge[0]] = None
    if len(right_edge):
        if rightmax[right_edge[0]] > rightmax[right_edge[1]]:
            rightmax[right_edge[1]] = -math.inf
            right_neigbor[right_edge[1]] = None
        else:
            rightmax[right_edge[0]] = -math.inf
            right_neigbor[right_edge[0]] = None

    whole_image = shreds[0]
    leftmost = 0
    rightmost = 0
    left_score = leftmax[0]
    right_score = rightmax[0]
    selected = set()
    selected.add(0)
    while len(selected) != len(shreds):
        if left_score == right_score:
            print(selected)
            print("strange tie", left_score, right_score)
            exit(1)
        elif left_score > right_score:
            new_shred = left_neighbor[leftmost]
            whole_image = np.concatenate((shreds[new_shred], whole_image), axis=1)
            leftmost = new_shred
            left_score = leftmax[new_shred]
        else:
            new_shred = right_neigbor[rightmost]
            whole_image = np.concatenate((whole_image, shreds[new_shred]), axis=1)
            rightmost = new_shred
            right_score = rightmax[new_shred]
        selected.add(new_shred)

    save_image = Image.fromarray(whole_image.astype(np.uint8))
    save_image.save(dir + '_sorted.png')
