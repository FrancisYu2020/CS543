import numpy as np
from PIL import Image
import os
import math

def normalize(x):
    #ZNCC normalization for the given vector x
    x = x.astype(float)
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x

def similarity(a, b):
    #compute the similarity between vector a and b, and also compute the
    #best offset of vector a with respect to vector b to make the image aligned
    x = a[:, -1, :].astype(float)
    y = b[:, 0, :].astype(float)
    sim = -math.inf
    shift = 0
    offsets = [int(a.shape[0]*(i - 20)/100) for i in range(41)]
    for offset in offsets:
        x_start = max(0, -offset)
        x_end = min(x.shape[0], y.shape[0] - offset)
        y_start = max(0, offset)
        y_end = min(x.shape[0] + offset, y.shape[0])
        x_t = x[x_start:x_end, :]
        y_t = y[y_start:y_end, :]
        x_t = normalize(x_t)
        y_t = normalize(y_t)
        curr_sim = np.sum(x_t*y_t)/(x_end - x_start)
        if curr_sim > sim:
            sim = curr_sim
            shift = offset
    return sim, shift

def align(shreds, order):
    #function to shift the shreds and get them aligned
    shifts = []
    for i in range(len(shreds)-1):
        shifts.append(similarity(shreds[order[i]], shreds[order[i+1]])[1])
    absolute_shifts = [0]
    max = 0
    min = -shreds[order[0]].shape[0]
    for i in range(len(shreds)-1):
        absolute_shifts.append(shifts[i] + absolute_shifts[-1])
        if absolute_shifts[-1] > max:
            max = absolute_shifts[-1]
        if absolute_shifts[-1] - shreds[order[i+1]].shape[0] < min:
            min = absolute_shifts[-1] - shreds[order[i+1]].shape[0]
    whole_image = np.pad(shreds[order[0]], ((max, -shreds[order[0]].shape[0] - min), (0, 0), (0, 0)))
    for i in range(1, len(shreds)):
        right_shred = np.pad(shreds[order[i]], ((max - absolute_shifts[i], absolute_shifts[i] - shreds[order[i]].shape[0] - min), (0, 0), (0, 0)))
        whole_image = np.concatenate((whole_image, right_shred), axis=1)
    return whole_image


for root, dirs, files in os.walk('shredded-images'):
    break

for dir in dirs:
    if dir[:5] != 'hard_':
        continue
    if dir == 'hard_text':
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
    padding_shreds = [np.pad(array, ((0, max_hight - array.shape[0]), (0, 0), (0, 0))) for array in shreds]
    edges = [[-math.inf]*len(shreds) for i in range(len(shreds))]
    leftmax = [-math.inf]*len(shreds)
    rightmax =[-math.inf]*len(shreds)
    left_neighbor = [None]*len(shreds)
    right_neigbor = [None]*len(shreds)
    shifts = [None]*len(shreds)

    for i in range(len(shreds)):
        print('working on shred ', i)
        for j in range(i+1, len(shreds)):
            edges[i][j], shift = similarity(shreds[i], shreds[j])
            if edges[i][j] > rightmax[i]:
                rightmax[i] = edges[i][j]
                right_neigbor[i] = j
                shifts[i] = shift
            if edges[i][j] > leftmax[j]:
                leftmax[j] = edges[i][j]
                left_neighbor[j] = i
            edges[j][i], shift = similarity(shreds[j], shreds[i])
            if edges[j][i] > rightmax[j]:
                rightmax[j] = edges[j][i]
                right_neigbor[j] = i
                shifts[j] = shift
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

    true_left_neighbor = [-math.inf]*len(shreds)
    true_leftmax = [-math.inf]*len(shreds)
    true_right_neighbor = [-math.inf]*len(shreds)
    true_righmax = [-math.inf]*len(shreds)

    whole_image = padding_shreds[0]
    leftmost = 0
    rightmost = 0
    left_score = leftmax[0]
    right_score = rightmax[0]
    selected = set()
    unselected = set()
    selected.add(0)
    for i in range(1, len(shreds)):
        unselected.add(i)
    order = [0]
    while len(unselected):
        if left_score == right_score:
            print(selected)
            print("strange tie", left_score, right_score)
            exit(1)
        elif left_score > right_score:
            new_shred = left_neighbor[leftmost]
            order.insert(0, new_shred)
            leftmost = new_shred
            left_score = leftmax[new_shred]
        else:
            new_shred = right_neigbor[rightmost]
            order.append(new_shred)
            rightmost = new_shred
            right_score = rightmax[new_shred]
        selected.add(new_shred)
        if new_shred in unselected:
            unselected.remove(new_shred)

    whole_image = align(shreds, order)

    save_image = Image.fromarray(whole_image.astype(np.uint8))
    save_image.save(dir + '_sorted.png')
