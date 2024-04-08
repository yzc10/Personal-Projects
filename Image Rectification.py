import numpy as np
import cv2
import os
import pandas as pd
import math
import time

def findIntersection (params1, params2):
    x = -1
    y = -1
    det = params1[0] * params2[1] - params2[0] * params1[1]
    if (det < 0.5 and det > -0.5): # i.e. ~ parallel lines 
        return (-1, -1)
    else:
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det
    return (int(x), int(y))

def calcParams(p1,p2):
    if p2[1] - p1[1] ==0:
        a = 0.0
        b = -1.0
    elif p2[0] - p1[0] ==0:
        a = -1.0
        b =0.0
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0
    c = (-a * p1[0]) - b * p1[1]
    return a,b,c,p1,p2 # gradient, coefficient for y, y-intercept

def sort_list_to_tuple(list0):
    temp_list = list0.copy()
    temp_list.sort()
    return tuple(temp_list)

def dist(a,b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def set_clockwise(corners):
    centre_x, centre_y = sum([x for x,_ in corners])/len(corners), sum([y for _,y in corners])/len(corners)
    angles = [math.atan2(y - centre_y, x - centre_x) for x,y in corners]
    counterclockwise_idx = sorted(range(len(corners)), key=lambda i: angles[i])
    counterclockwise_points = [corners[i] for i in counterclockwise_idx]
    counterclockwise_points = np.array(counterclockwise_points)
    return counterclockwise_points

def get_perimeter(corner_combi):
    c_list = set_clockwise(corner_combi)
    distance = dist(c_list[0],c_list[1]) + dist(c_list[1],c_list[2]) + dist(c_list[2],c_list[3]) + dist(c_list[3],c_list[0])
    return distance

def between_corners(p,c1,c2):
    '''
    p,c1,c2: (x,y)
    '''
    x_max = max(c1[0],c2[0])
    x_min = min(c1[0],c2[0])
    y_max = max(c1[1],c2[1])
    y_min = min(c1[1],c2[1])
    if p[0] <= x_max and p[0] >= x_min and p[1] <= y_max and p[1] >= y_min:
        return True
    else: 
        return False
    
def intersecting_corner(p_in,p_out,c1,c2):
    if p_in[0] == c1[0]:
        return c1
    if p_in[0] == c2[0]:
        return c2
    if p_in[0]==p_out[0]:
        return None
    grad_p = (p_in[1] - p_out[1])/(p_in[0] - p_out[0])
    grad_1 = (p_in[1] - c1[1])/(p_in[0] - c1[0])
    grad_2 = (p_in[1] - c2[1])/(p_in[0] - c2[0])
    err_1 = abs(grad_1-grad_p)
    err_2 = abs(grad_2-grad_p)
    if err_1 < err_2:
        return c1
    else:
        return c2

def imrect(im1):
# Perform Image rectification on an 3D array im.
# Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
# Returns: out: numpy.ndarray, rectified image。
    
    # # 1) Detect Lines
    im1 = np.uint8(im1)
    edges = cv2.Canny(im1,250,420) 
    lines = cv2.HoughLinesP(edges,1,np.pi/180,
                            threshold=120,
                            minLineLength=100,
                            maxLineGap=50
                            )

    # # 2a) Generate Intersections
    corners = []
    lines = np.squeeze(lines)
    params = []
    lines_to_corner = []
    for i in range(len(lines)):
        params.append(calcParams([lines[i][0], lines[i][1]], [lines[i][2], lines[i][3]]))
    for i in range(len(params)):
        for j in range(i,len(params)):
            intersect = findIntersection(params[i],params[j])
            if intersect[1]>0 and intersect[0]>0 and intersect[1]<im1.shape[0] and intersect[0]<im1.shape[1]:
                corners.append(intersect)
                lines_to_corner.append([(params[i],params[j]),intersect]) 
    print('Intersections generated.')

    # # 2b) Consolidate lines so that very similar lines (e.g.if disconnected by Hough transform algorithm) are combined together
    consolidated_lines = pd.DataFrame(columns=['main_line','sec_lines','corners'])
    for i in range(len(params)):
        main_list = consolidated_lines['main_line'].tolist()
        if len(main_list) > 0:
            check = [(abs(params[i][0] - main_list[j][0]) < 1) and (abs((params[i][2]-main_list[j][2])/params[i][2]) < 0.03) for j in range(len(main_list))]
        else: check = [False]
        if True not in check:
            consolidated_lines.loc[len(consolidated_lines),'main_line'] = params[i]
            if pd.isna(consolidated_lines.at[len(consolidated_lines)-1,'sec_lines']):
                consolidated_lines.loc[len(consolidated_lines)-1,'sec_lines'] = []
        else:
            main_idx = check.index(True)
            if True not in pd.isna(consolidated_lines.at[main_idx,'sec_lines']):
                consolidated_lines.loc[main_idx,'sec_lines'] = []
            sec_list = consolidated_lines.at[main_idx,'sec_lines']
            sec_list.append(params[i])
            consolidated_lines.loc[main_idx,'sec_lines'] = sec_list
    consolidated_lines['corners'] = [[]]*len(consolidated_lines)
    for corner in lines_to_corner:
        for i in range(len(consolidated_lines)):
            if (corner[0][0] == consolidated_lines.at[i,'main_line']) or (corner[0][0] in consolidated_lines.at[i,'sec_lines']) or (corner[0][1] == consolidated_lines.at[i,'main_line']) or (corner[0][1] in consolidated_lines.at[i,'sec_lines']):
                corner_list = consolidated_lines.at[i,'corners'].copy()
                corner_list.append(corner[1])
                consolidated_lines.at[i,'corners'] = corner_list
    consolidated_lines['corners'] = consolidated_lines['corners'].apply(lambda x:list(set(x)))
    consolidated_lines['corners_num'] = consolidated_lines['corners'].apply(len)
    print('Corner and line combinations generated.')

    # # 3) a. Find quadrilateral combinations, i.e. 4 intersections/corners that form a 'loop' 
    # # -> b. Identify combination with largest sum of distances (as a proxy for area)
    line_combis = []
    corner_combis = []
    def get_quads(other_idx,next_idx,quad,corner_list,n=3):
        if n == 0:
            line_combis.append(quad)
            corner_combis.append(corner_list)
            return
        else:
            for i in other_idx:
                intersect_corners = list(set(consolidated_lines.at[i,'corners']) & set(consolidated_lines.at[next_idx,'corners']))
                if n == 1:
                    intersect_corners_2 = list(set(consolidated_lines.at[i,'corners']) & set(consolidated_lines.at[quad[0],'corners']))
                    test_condition = (len(intersect_corners) > 0 and len(intersect_corners_2) > 0)
                else:
                    test_condition = (len(intersect_corners) > 0)
                if test_condition:
                    quad_1 = quad.copy()
                    quad_1.append(i)
                    other_idx_1 = other_idx.copy()
                    other_idx_1.remove(i)
                    corner_list_1 = corner_list.copy()
                    corner_list_1.append(intersect_corners[0])
                    if n==1:
                        corner_list_1.append(intersect_corners_2[0])
                    get_quads(other_idx_1,i,quad_1,corner_list_1,n-1)
    
    for idx in range(len(consolidated_lines)):
        get_quads([i for i in range(len(consolidated_lines)) if i != idx],idx,[idx],[]) 
    quad_combis = list(zip(line_combis,corner_combis))
    quad_combis = list(set([(sort_list_to_tuple(combi[0]),tuple(combi[1])) for combi in quad_combis.copy()]))
    quad = pd.DataFrame(columns=['lines','corners','distance'])
    for combi in quad_combis:
        quad.loc[len(quad)] = [combi[0],combi[1],get_perimeter(combi[1])]
    
    def line_coverage(line_combi,corner_combi):
        coverage = []
        n = 0
        while n < len(line_combis):
            if set(line_combis[n]) == set(line_combi):
                line_combi1 = line_combis[n]
                break
            n += 1
        for i in range(len(line_combi1)):
            all_lines = [consolidated_lines.at[line_combi1[i],'main_line']]
            all_lines.extend(consolidated_lines.at[line_combi1[i],'sec_lines'])
            total_distance = 0
            if i == 3:
                last_idx = 0
            else:
                last_idx = i+1
            valid_quad = True
            try:
                c1 = [c for c in corner_combi if c in list(set(consolidated_lines.at[line_combi1[i-1],'corners']) & set(consolidated_lines.at[line_combi1[i],'corners']))][0]
                c2 = [c for c in corner_combi if c in list(set(consolidated_lines.at[line_combi1[i],'corners']) & set(consolidated_lines.at[line_combi1[last_idx],'corners']))][0]
            except:
                valid_quad = False
            if valid_quad:
                for sub_line in all_lines:
                        if between_corners(sub_line[3],c1,c2) and between_corners(sub_line[4],c1,c2):
                            total_distance += dist(sub_line[3],sub_line[4])
                        elif between_corners(sub_line[3],c1,c2):
                            target_corner = intersecting_corner(sub_line[3],sub_line[4],c1,c2)
                            if target_corner is not None:
                                total_distance += dist(sub_line[3],intersecting_corner(sub_line[3],sub_line[4],c1,c2))
                        elif between_corners(sub_line[4],c1,c2):
                            target_corner = intersecting_corner(sub_line[4],sub_line[3],c1,c2)
                            if target_corner is not None:
                                total_distance += dist(sub_line[4],intersecting_corner(sub_line[4],sub_line[3],c1,c2))
                actual_distance = max(dist(c1,c2),1e-5)
                ratio = min(total_distance / actual_distance,1)
            else:
                ratio = 0
            coverage.append(ratio)
        return np.average(coverage)
    quad['line_coverage'] = quad.apply(lambda row:line_coverage(row.lines,row.corners),axis=1) 
    if quad['line_coverage'].max() < 0.5:
        quad_filtered = quad
    else:
        quad_filtered = quad.drop(quad[quad['line_coverage']<0.74].index).reset_index(drop=True)
    final_corners = quad_filtered.at[quad_filtered['distance'].idxmax(),'corners']
    print('Corners confirmed.')
    
    # 3c) Re-orient source and target points: Get line pair with longer sum of lengths
    src = set_clockwise(final_corners)
    sum_distances = [dist(src[0],src[1])+dist(src[2],src[3]),dist(src[1],src[2])+dist(src[3],src[0])]
    if im1.shape[0] < im1.shape[1]:
        BR_idx = 2 
        if sum_distances[1] > sum_distances[0]:
            BR_idx = 3
    else:
        BR_idx = 3
        if sum_distances[1] > sum_distances[0]:
            BR_idx = 2
    rearr_idx = [BR_idx]
    for i in range(3):
        if rearr_idx[-1] == 3:
            rearr_idx.append(0)
        else:
            rearr_idx.append(rearr_idx[-1]+1)
    rearr_idx = np.array(rearr_idx)
    src = src[rearr_idx]
    x_pct = 0.8
    y_pct = 0.8
    x0 = int((1-x_pct)*im1.shape[1])
    x1 = int(x_pct*im1.shape[1])
    y0 = int((1-y_pct)*im1.shape[0])
    y1 = int(y_pct*im1.shape[0])
    dst = np.array([[x1,y1],[x0,y1],[x0,y0],[x1,y0]],dtype=np.float32)

    # 4) Homography -> Perspective Transform
    homography_mat, _ = cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    rect_im1 = cv2.warpPerspective(im1,homography_mat,(im1.shape[1],im1.shape[0]))
    return rect_im1    


if __name__ == "__main__":
    path = r'PATH'
    os.chdir(path)
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        start = time.perf_counter()
        image = np.array(cv2.imread(name, -1), dtype=np.float32)
        rectified = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectified,decimals=0),0,255)))
        print(f'Image {name} generated in the following time:',time.perf_counter()-start)