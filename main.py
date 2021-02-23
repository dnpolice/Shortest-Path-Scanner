import cv2
import numpy as np
from utils import breadthFirstSearch


def get_shortest_path(matrix_rows=15, matrix_columns=15, start_row=0, start_col=0, end_row=14, end_col=14, file='grid.JPG'):
    img_height = matrix_rows*50
    img_width = matrix_columns*50
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    cv2.imshow('original', img)

    # Convert to binary image (Black/White)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    def reorder(my_points):
        my_points = my_points.reshape((4, 2))
        my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
        add = my_points.sum(1)
        my_points_new[0] = my_points[np.argmin(add)]
        my_points_new[3] = my_points[np.argmax(add)]
        diff = np.diff(my_points, axis=1)
        print(diff)
        print(my_points)
        my_points_new[1] = my_points[np.argmin(diff)]
        my_points_new[2] = my_points[np.argmax(diff)]
        return my_points_new

    biggest = reorder(biggest)

    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.drawContours(img, biggest, -1, (0, 255, 0), 20)

    points_initial = np.float32(biggest)
    point_end = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])

    transfer_matrix = cv2.getPerspectiveTransform(points_initial, point_end)
    img = cv2.warpPerspective(img, transfer_matrix, (img_width, img_height))

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = []
    mean_value_cells = []
    img_rows = np.vsplit(grey, matrix_rows)
    img_color_rows = np.vsplit(img, matrix_rows)
    for rowNum in range(len(img_rows)):
        row_cols = np.hsplit(img_rows[rowNum], matrix_columns)
        img_color_rows_cols = np.hsplit(img_color_rows[rowNum], matrix_columns)
        for colNum in range(len(row_cols)):
            cells.append(img_color_rows_cols[colNum])
            mean_value_cell = np.mean(row_cols[colNum])
            mean_value = 0 if mean_value_cell >= 127 else 1
            mean_value_cells.append(mean_value)

    matrix_to_solve = []
    split_image = []
    for i in range (matrix_rows-1, -1, -1):
        matrix_row_to_solve = []
        split_image_row = []
        for j in range (0, matrix_columns):
            matrix_row_to_solve.append(mean_value_cells[i*matrix_rows+j])
            split_image_row.append(cells[i*matrix_rows+j])
        matrix_to_solve.append(matrix_row_to_solve)
        split_image.append(split_image_row)

    shortest_path = breadthFirstSearch(matrix_to_solve, [start_row, start_col], [end_row, end_col])
    if shortest_path is not None:
        for idx in range(len(shortest_path)):
            cords = shortest_path[idx]
            if idx == 0:
                split_image[cords[0]][cords[1]][:] = (0, 255, 0)
            elif idx == len(shortest_path)-1:
                split_image[cords[0]][cords[1]][:] = (0, 0, 255)
            else:
                split_image[cords[0]][cords[1]][:] = (255, 255, 0)
    else:
        print('Shortest path is unable to be calculated')

    cv2.startWindowThread()
    cv2.imshow('Shortest Path', img)
    cv2.waitKey(10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_shortest_path(15, 15, 14, 0, 0, 14, 'grid.JPG')
