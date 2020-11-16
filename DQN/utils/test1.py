import numpy as np

if __name__ == "__main__":

    # 存储起点,从(0,0)开始
    point_list = [(0, 0)]

    count = 0  # 控制轮数,一共三轮
    while True:
        print(point_list)
        print(len(point_list))
        result_list = []
        count += 1
        if (count == 4):
            break
        for i in range(len(point_list)):
            x0 = point_list[i][0]
            y0 = point_list[i][1]

            # 圆弧半径
            rad = 4
            # 初步选点
            #必须是下半区域的
            amount=0
            for x1 in range(x0+1, x0 + rad + 1):
                for y1 in range(y0+1, y0 + rad + 1):
                    distance = pow(x1 - x0, 2) + pow(y1 - y0, 2)
                    if (distance < pow(rad-1,2) or distance > pow(rad,2)):
                        continue
                    amount+=1
                    if(amount==5):
                        break
                    result_list.append((x1, y1))
        point_list = result_list

