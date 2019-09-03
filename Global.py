import torch
import cv2
import math
import numpy as np
import Conn_Functions as my_connectivity
import Help_Functions as my_help
from numpy.linalg import det


def initVariables():
    """
    int: Module level variable documented inline.
    The docstring may span multiple lines. The type may optionally be specified
    on the first line, separated by a colon.
    """


    global IMAGE1, IMAGE2, frame0, frame1, HEIGHT, WIDTH, dx, dy, PI, d, D, C_D, N, K, K_C, loc_scale, int_scale, opt_scale, colors, LOC, ALPHA, BETA, ETA, PI_0, H, Y, X, M, M_0, HARD_EM, LOOKUP_TABLE
    global D_T, ALPHA_T, M_T, ETA_T, M_0_T, M_ETA_T, BETA_T, ETA_M_ETA_T, inf_T, log1_T, one_T, small_T, logSmall_T, WIDTH_T, idx, ne_idx, idx_all, idx_pixels, shift, shift_index, N_index, ne4_idx, ne4_idx_split, opt_scale_T
    global PI_0_T, idx_pixels_cuda, LOOKUP_TABLE_CUD0A, c_idx, C_prior, PSI_prior, NI_prior, argmax_range, ones, zeros, TrueFlow, SIGMA_INT_FLOW, SIGMA_INT, A_prior, C_prior
    global X_C_X1, X_C_X2, X_C_Y1, X_C_Y2, SIGMA_XY_X1, SIGMA_XY_X2, SIGMA_XY_Y1, SIGMA_XY_Y2
    global D_12, D_, D_Inv, det_PSI, PI, inside_padded, FrameN, c_idx_9, c_idx_25
    global Epsilon
    global ALPHA_MS, ALPHA_MS2, LOG_ALPHA_MS2
    global psi_prior, ni_prior
    global psi_prior_sons, ni_prior_sons0
    global neig_num, potts_area
    global detInt
    global dtype
    global split_lvl
    global Padding, Padding0
    global Beta_P
    global Distances_padded, Cluster_2_pixel
    global neig_padded
    global K_C_ORIGINAL
    global idx_pixels_1, idx_pixels_2, idx_pixels_3, idx_pixels_4
    global idx_1, idx_2, idx_3, idx_4
    global Plot
    global csv_file
    global Folder
    global RegSize
    global K_C_LOW, K_C_HIGH
    global Sintel, SintelSave
    global K_C_temp
    global Beta_P
    global int_scale
    global frame0
    global DP_prior
    global save_folder
    global repeat
    global split_merges
    global covarince_estimation
    global exp_name
    global csv
    global Print
    covarince_estimation=True
    split_merges=True
    # OR: change limits
    K_C_HIGH = 999
    # OR: change limits
    K_C_LOW =  0

    Sintel = False
    Padding = torch.nn.ConstantPad2d((1, 1, 1, 1), (-1)).cuda()
    Padding0 = torch.nn.ConstantPad2d((1, 1, 1, 1), 0).cuda()

    neig_num = 4
    potts_area = 25

    Beta_P = torch.from_numpy(np.array([2.7], dtype=np.float)).cuda().float()
    C_prior = 1800
    # OR: Increase for more splits

    ALPHA_MS = 2675

    ALPHA_MS2 = 0.0001
    # OR: Decrease for more merges

    LOG_ALPHA_MS2 = -26.2
    # ALPHA_MS=-99999
    # LOG_ALPHA_MS2=999999
    # IMAGE1 = "eval-d1ata/Backyard/frame10.png"
    IMAGE2 = "eval-data/Backyard/frame11.png"
    # MAGE1I = "benchmark_opt/train/60079.jpg"
    # IMAGE1 = "Sintel/training/final/market_2/frame_0005.png"
    # # IMAGE1 = "other-data/Dimetrodon/frame10.png"
    # IMAGE2 = "Dog.jpg"
    #IMAGE1 = "BSR/BSDS500/data/images/val/160068.jpg"
    # # IMAGE1 = "BSR/BSDS500/data/images/test/80085.jpg"
    # IMAGE1 = "iccv09Data/images/8000811.jpg"
    # IMAGE1 = "176.jpg"


    #
    #
    # #IMAGE1 = "Parlay-bottle-white-background.jpg"
    # IMAGE1 = "3096.jpg"

    # IMAGE2 = "car2.jpg"
    frame0 = cv2.imread(IMAGE1)
    # frame0=frame0[0:480,0:320]0
    #frame0 = cv2.resize(frame0,(480,360))
    # frame0 = frame0[0:480, 0:320]
    # frame1 = cv2.resize(frame1, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

    HEIGHT, WIDTH, _ = frame0.shape

    dx = np.array([[-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12]])

    """
        **Global Paramter**:
        Derivation X vector
    """
    dy = np.array([[-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12]]).T

    """
        **Global Paramter**:
        Derivation Y vector
    """

    """
        **Global Paramter**:
       0 Distinct color list
    """

    PI = np.pi
    """
        **Global Paramter**:
        PI math constant
    """

    d = 1
    """
        **Global Paramter**:
        Dimension of projection
    """
    D = 2
    """
        **Global Paramter**:
        Dimension of data
    """

    C_D = 3
    """
        **Global Paramter**:
        Dimension of color space
    """
    N = HEIGHT * WIDTH
    """
        **Global Paramter**:
        Number of total points
    """

    K = 4
    """
        **Glob Paramter**:
        Number of arttficial motions
    """

    K_C = 330
    """
        **Global Paramter**:
        Number of clusters
    """
    loc_scale = 1
    """
        **Global Paramter**:
        Location scaling in k-means
    """
    int_scale = 8.7

    """
        **Global Paramter**:
        Intesity scaling in k-means
    """
    opt_scale = 1
    """
        **Global Paramter**:
        Optical flow scaling in k-means
    """

    LOC = 0
    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    ALPHA = 1
    """
        **Global Paramter**
        Prior normal gamma prameters
    """
    BETA = 2

    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    ETA = np.ones((D, 1)) * 0
    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    PI_0 = 0.00
    """
        **Global Paramter**
        Paramters of the outlier
    """

    HARD_EM = True
    """
        **Global Paramter**
        Paramters of the Hard / Soft EM
    """

    DP_prior=0.01
    """
    int: Module level variable documented inline.
    The docstring may span multiple lines. The type may optionally be specified
    on the first line, separated by a colon.
    """

    Y = np.zeros((d, N))
    H = np.zeros((N, d, D))
    X = np.zeros((D, N))
    M = np.eye(D) * (1.0 / 100) * 0
    M_0 = np.eye(d) * 100000 * 0
    LOOKUP_TABLE = torch.from_numpy(my_connectivity.Create_LookupTable()).cuda().byte()
    LOOKUP_TABLE_CUDA = my_connectivity.Create_LookupTable()

    M_T = torch.from_numpy(M).cuda().float()
    ETA_T = torch.from_numpy(ETA).cuda().float()
    M_0_T = torch.from_numpy(M_0).cuda().float()
    ALPHA_T = torch.tensor(ALPHA).cuda().float()
    D_T = torch.tensor(D).cuda().float()
    BETA_T = torch.tensor(BETA).cuda().float()
    M_ETA_T = M_T @ ETA_T.cuda().float()
    ETA_M_ETA_T = (ETA_T.transpose(0, 1) @ M_T @ (ETA_T)).cuda().float()
    inf_T = torch.tensor(np.inf, dtype=torch.float).cuda()
    log1_T = torch.from_numpy(np.array([np.log(1)], dtype=np.float)).cuda().float()
    one_T = torch.from_numpy(np.array([1], dtype=np.float)).cuda().float()
    logSmall_T = torch.from_numpy(np.array([np.log(0.0000001)], dtype=np.float)).cuda().float()
    small_T = torch.from_numpy(np.array([0.0000001], dtype=np.float)).cuda().float()
    WIDTH_T = torch.tensor(WIDTH).cuda().int()
    opt_scale_T = torch.from_numpy(np.array([opt_scale], dtype=np.float)).cuda().float()
    PI_0_T = torch.from_numpy(np.array([PI_0], dtype=np.float)).cuda().float()

    idx_all = np.arange(0, N)

    i_ne_idx = np.zeros((4, int(N / 4) * 5))
    j_ne_idx = np.zeros((4, int(N / 4) * 5))
    i_idx = np.zeros((4, int(N / 4) * 9))
    j_idx = np.zeros((4, int(N / 4) * 9))
    i_conn = np.zeros((4, int(N / 4) * 9))
    j_conn = np.zeros((4, int(N / 4) * 9))

    i = np.floor_divide(idx_all, WIDTH)
    j = idx_all % WIDTH
    j += 1
    i += 1
    idx_pixels1 = np.array([])
    idx_pixels2 = np.array([])
    idx_pixels3 = np.array([])
    idx_pixels4 = np.array([])
    i1 = i[np.logical_and((i % 2 == 0), (j % 2 == 0))]
    j1 = j[np.logical_and((j % 2 == 0), (i % 2 == 0))]
    idx_pixels1 = np.append(idx_pixels1, np.array([i1 - 1, i1 - 1, i1 - 1, i1, i1, i1, i1 + 1, i1 + 1, i1 + 1]) * (
                WIDTH + 2) + np.array([j1 - 1, j1, j1 + 1, j1 - 1, j1, j1 + 1, j1 - 1, j1, j1 + 1]))
    i2 = i[np.logical_and((i % 2 == 0), (j % 2 == 1))]
    j2 = j[np.logical_and((j % 2 == 1), (i % 2 == 0))]
    idx_pixels2 = np.append(idx_pixels2, np.array([i2 - 1, i2 - 1, i2 - 1, i2, i2, i2, i2 + 1, i2 + 1, i2 + 1]) * (
                WIDTH + 2) + np.array(
        [j2 - 1, j2, j2 + 1, j2 - 1, j2, j2 + 1, j2 - 1, j2, j2 + 1]))
    idx2 = (i2 - 1) * (WIDTH) + (j2 - 1)
    i3 = i[np.logical_and((i % 2 == 1), (j % 2 == 0))]
    j3 = j[np.logical_and((j % 2 == 0), (i % 2 == 1))]
    idx_pixels3 = np.append(idx_pixels3, np.array([i3 - 1, i3 - 1, i3 - 1, i3, i3, i3, i3 + 1, i3 + 1, i3 + 1]) * (
                WIDTH + 2) + np.array(
        [j3 - 1, j3, j3 + 1, j3 - 1, j3, j3 + 1, j3 - 1, j3, j3 + 1]))
    idx3 = (i3 - 1) * (WIDTH) + (j3 - 1)
    i4 = i[np.logical_and((i % 2 == 1), (j % 2 == 1))]
    j4 = j[np.logical_and((j % 2 == 1), (i % 2 == 1))]
    idx_pixels4 = np.append(idx_pixels4, np.array([i4 - 1, i4 - 1, i4 - 1, i4, i4, i4, i4 + 1, i4 + 1, i4 + 1]) * (
                WIDTH + 2) + np.array(
        [j4 - 1, j4, j4 + 1, j4 - 1, j4, j4 + 1, j4 - 1, j4, j4 + 1]))

    idx1 = (i3 - 1) * (WIDTH) + (j3 - 1)
    idx2 = (i2 - 1) * (WIDTH) + (j2 - 1)
    idx3 = (i1 - 1) * (WIDTH) + (j1 - 1)
    idx4 = (i4 - 1) * (WIDTH) + (j4 - 1)

    i = np.floor_divide(idx_all, WIDTH)
    j = idx_all % WIDTH
    i += 1
    j += 1
    inside_padded = i * (WIDTH + 2) + j
    inside_padded = torch.from_numpy(inside_padded).cuda().long()
    # ne_idx=i_ne_idx+j_ne_idx
    idx_pixels = i_idx + j_idx
    # idx_pixels_cuda=idx_pixels.reshape(4,9,-1).transpose(0,2,1)
    # ne_idx=torch.from_numpy(ne_idx).cuda().long()
    # idx=torch.from_numpy(idx).cuda()
    idx_all = torch.from_numpy(idx_all).cuda()
    # idx_pixels=torch.from_numpy(idx_pixels).cuda().long()
    idx_pixels_1 = torch.from_numpy(idx_pixels3).cuda().long()
    idx_pixels_2 = torch.from_numpy(idx_pixels2).cuda().long()
    idx_pixels_3 = torch.from_numpy(idx_pixels1).cuda().long()
    idx_pixels_4 = torch.from_numpy(idx_pixels4).cuda().long()

    idx_1 = torch.from_numpy(idx1).cuda()
    idx_2 = torch.from_numpy(idx2).cuda()
    idx_3 = torch.from_numpy(idx3).cuda()
    idx_4 = torch.from_numpy(idx4).cuda()

    shift = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7])).cuda().byte().unsqueeze(0).unsqueeze(2)
    shift_index = torch.from_numpy(np.array([0, 1, 2, 3, 5, 6, 7, 8])).cuda().long()
    ne4_idx = torch.from_numpy(np.array([1, 3, 4, 5, 7])).cuda().long()
    ne4_idx_split = torch.from_numpy(np.array([-(WIDTH), -1, 0, 1, WIDTH])).cuda().long()
    N_index = torch.arange(0, N).cuda()

    c_idx = np.zeros((N, 5))

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            temp_idx = (i) * (WIDTH) + j

            # up
            if (i == 0):
                c_idx[temp_idx, 0] = temp_idx
            else:
                c_idx[temp_idx, 0] = temp_idx - WIDTH

            # left
            if (j == 0):
                c_idx[temp_idx, 1] = temp_idx
            else:
                c_idx[temp_idx, 1] = temp_idx - 1

            # down
            if (i == (HEIGHT - 1)):
                c_idx[temp_idx, 2] = temp_idx
            else:
                c_idx[temp_idx, 2] = temp_idx + WIDTH

            # right
            if (j == (WIDTH - 1)):
                c_idx[temp_idx, 3] = temp_idx
            else:
                c_idx[temp_idx, 3] = temp_idx + 1

            c_idx[temp_idx, 4] = temp_idx

    c_idx = c_idx[:, 0:neig_num]
    c_idx = torch.from_numpy(c_idx).cuda().long().reshape(-1)

    c_idx_9 = np.zeros((N, 9))

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            temp_idx = (i) * (WIDTH) + j

            if (i == 1 and j == 0):
                b = 3
            # up
            if (i == 0):
                c_idx_9[temp_idx, 0] = temp_idx
            else:
                c_idx_9[temp_idx, 0] = temp_idx - WIDTH

            # left
            if (j == 0):
                c_idx_9[temp_idx, 1] = temp_idx
            else:
                c_idx_9[temp_idx, 1] = temp_idx - 1

            # down
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 2] = temp_idx
            else:
                c_idx_9[temp_idx, 2] = temp_idx + WIDTH

            # right
            if (j == (WIDTH - 1)):
                c_idx_9[temp_idx, 3] = temp_idx
            else:
                c_idx_9[temp_idx, 3] = temp_idx + 1

            # up_left
            if (i == 0):
                c_idx_9[temp_idx, 4] = temp_idx
            else:
                if (j > 0):
                    c_idx_9[temp_idx, 4] = temp_idx - WIDTH - 1
                else:
                    c_idx_9[temp_idx, 4] = temp_idx

            # up_right
            if (i == 0):
                c_idx_9[temp_idx, 5] = temp_idx
            else:
                if (j < WIDTH - 1):
                    c_idx_9[temp_idx, 5] = temp_idx - WIDTH + 1
                else:
                    c_idx_9[temp_idx, 5] = temp_idx

            # down left
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 6] = temp_idx
            else:
                if (j > 0):
                    c_idx_9[temp_idx, 6] = temp_idx + WIDTH - 1
                else:
                    c_idx_9[temp_idx, 6] = temp_idx

            # down right
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 7] = temp_idx
            else:
                if (j < WIDTH - 1):
                    c_idx_9[temp_idx, 7] = temp_idx + WIDTH + 1
                else:
                    c_idx_9[temp_idx, 7] = temp_idx

            c_idx_9[temp_idx, 8] = temp_idx

            for m in range(0, 9):
                if (c_idx_9[temp_idx, m] == -1):
                    b = 5
    c_idx_9 = torch.from_numpy(c_idx_9).cuda().long().reshape(-1)

    matrix = np.arange(0, N).reshape(HEIGHT, WIDTH)
    padded_matrix = np.pad(matrix, 5, pad_with, padder=-1)
    c_idx_25 = np.zeros((N, 25))
    for i in range(5, 5 + HEIGHT):
        for j in range(5, 5 + WIDTH):
            temp_idx = (i - 5) * (WIDTH) + j - 5
            c_idx_25[temp_idx] = padded_matrix[i - 2:i + 3, j - 2:j + 3].reshape(-1)
            c_idx_25[temp_idx] = np.where(c_idx_25[temp_idx] == -1, padded_matrix[i, j], c_idx_25[temp_idx])

    c_idx_25 = torch.from_numpy(c_idx_25).cuda().long().reshape(-1)
    padded_matrix = np.pad(matrix, 7, pad_with, padder=-1)

    if (potts_area == 49):

        c_idx_25 = np.zeros((N, 49))
        for i in range(7, 7 + HEIGHT):
            for j in range(7, 7 + WIDTH):
                temp_idx = (i - 7) * (WIDTH) + j - 7
                c_idx_25[temp_idx] = padded_matrix[i - 3:i + 4, j - 3:j + 4].reshape(-1)
                c_idx_25[temp_idx] = np.where(c_idx_25[temp_idx] == -1, padded_matrix[i, j], c_idx_25[temp_idx])

        c_idx_25 = torch.from_numpy(c_idx_25).cuda().long().reshape(-1)

    A_prior = N / (K_C)
    PSI_prior = A_prior * A_prior * np.eye(2)
    det_PSI = torch.from_numpy(np.array([det(PSI_prior)])).cuda().float()
    NI_prior = C_prior * A_prior
    C_prior = torch.from_numpy(np.array([C_prior], dtype=np.float)).cuda().float()
    NI_prior = torch.from_numpy(np.array([NI_prior - 3], dtype=np.float)).cuda().float()
    PSI_prior = torch.from_numpy(PSI_prior).cuda().float().reshape(-1)
    argmax_range = torch.from_numpy(np.arange(0, N * (K_C + 1), K_C + 1)).cuda()
    ones = torch.ones(N).cuda()
    zeros = torch.zeros(N).cuda()
    X_C_X1 = torch.arange(0, N * 5 * 12, 12).cuda()
    X_C_X2 = torch.arange(5, N * 5 * 12, 12).cuda()
    X_C_Y1 = torch.arange(1, N * 5 * 12, 12).cuda()
    X_C_Y2 = torch.arange(6, N * 5 * 12, 12).cuda()

    SIGMA_XY_X1 = torch.cat((torch.arange(0, N * 5 * 8, 8), torch.arange(1, N * 5 * 8, 8))).cuda()
    SIGMA_XY_Y1 = torch.cat((torch.arange(2, N * 5 * 8, 8), torch.arange(3, N * 5 * 8, 8))).cuda()
    SIGMA_XY_X2 = torch.cat((torch.arange(4, N * 5 * 8, 8), torch.arange(5, N * 5 * 8, 8))).cuda()
    SIGMA_XY_Y2 = torch.cat((torch.arange(6, N * 5 * 8, 8), torch.arange(7, N * 5 * 8, 8))).cuda()

    a = 3
    SIGMA_INT_FLOW = torch.from_numpy(
        np.array([1.0 / int_scale, 1.0 / int_scale, 1.0 / int_scale, 1.0 / opt_scale, 1.0 / opt_scale])).unsqueeze(
        0).unsqueeze(0).float().cuda()
    SIGMA_INT = torch.from_numpy(np.array([1.0 / int_scale, 1.0 / int_scale, 1.0 / int_scale])).unsqueeze(0).unsqueeze(
        0).float().cuda()
    PI = torch.from_numpy(np.array([np.pi])).float().cuda()
    D_12 = False
    if (D_12):
        D_ = 12
        D_Inv = 8
    else:
        D_ = 5
        D_Inv = 4

    Epsilon = torch.zeros(N).cuda().float() + 0.000000001
    split_lvl = torch.zeros(N).cuda(0).float()
    Distances_padded = torch.zeros(2, ((HEIGHT + 2) * (WIDTH + 2))).float().cuda().transpose(0, 1)
    Cluster_2_pixel = torch.zeros(2, ((HEIGHT + 2) * (WIDTH + 2))).float().cuda().transpose(0, 1)
    Cluster_2_pixel[:, 0] = torch.arange(0, (HEIGHT + 2) * (WIDTH + 2)).int().cuda()
    neig_padded = torch.from_numpy(np.array([-1, 1, -(WIDTH + 2), (WIDTH + 2)])).cuda().int()

    detInt = int_scale * int_scale * int_scale
    detInt = torch.from_numpy(np.array([detInt], dtype=np.float)).cuda().float()


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


colors = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
          "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
          "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
          "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
          "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
          "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
          "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
          "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
          "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
          "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
          "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
          "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
          "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
          "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
          "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
          "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
          "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
          "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
          "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
          "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
          "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
          "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
          "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
          "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
          "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
          "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
          "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
          "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
          "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
          "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
          "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
          "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
          "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B", "#1E2324", "#DEC9B2", "#9D4948",
          "#85ABB4", "#342142", "#D09685", "#A4ACAC", "#00FFFF", "#AE9C86", "#742A33", "#0E72C5",
          "#AFD8EC", "#C064B9", "#91028C", "#FEEDBF", "#FFB789", "#9CB8E4", "#AFFFD1", "#2A364C",
          "#4F4A43", "#647095", "#34BBFF", "#807781", "#920003", "#B3A5A7", "#018615", "#F1FFC8",
          "#976F5C", "#FF3BC1", "#FF5F6B", "#077D84", "#F56D93", "#5771DA", "#4E1E2A", "#830055",
          "#02D346", "#BE452D", "#00905E", "#BE0028", "#6E96E3", "#007699", "#FEC96D", "#9C6A7D",
          "#3FA1B8", "#893DE3", "#79B4D6", "#7FD4D9", "#6751BB", "#B28D2D", "#E27A05", "#DD9CB8",
          "#AABC7A", "#980034", "#561A02", "#8F7F00", "#635000", "#CD7DAE", "#8A5E2D", "#FFB3E1",
          "#6B6466", "#C6D300", "#0100E2", "#88EC69", "#8FCCBE", "#21001C", "#511F4D", "#E3F6E3",
          "#FF8EB1", "#6B4F29", "#A37F46", "#6A5950", "#1F2A1A", "#04784D", "#101835", "#E6E0D0",
          "#FF74FE", "#00A45F", "#8F5DF8", "#4B0059", "#412F23", "#D8939E", "#DB9D72", "#604143",
          "#B5BACE", "#989EB7", "#D2C4DB", "#A587AF", "#77D796", "#7F8C94", "#FF9B03", "#555196",
          "#31DDAE", "#74B671", "#802647", "#2A373F", "#014A68", "#696628", "#4C7B6D", "#002C27",
          "#7A4522", "#3B5859", "#E5D381", "#FFF3FF", "#679FA0", "#261300", "#2C5742", "#9131AF",
          "#AF5D88", "#C7706A", "#61AB1F", "#8CF2D4", "#C5D9B8", "#9FFFFB", "#BF45CC", "#493941",
          "#863B60", "#B90076", "#003177", "#C582D2", "#C1B394", "#602B70", "#887868", "#BABFB0",
          "#030012", "#D1ACFE", "#7FDEFE", "#4B5C71", "#A3A097", "#E66D53", "#637B5D", "#92BEA5",
          "#00F8B3", "#BEDDFF", "#3DB5A7", "#DD3248", "#B6E4DE", "#427745", "#598C5A", "#B94C59",
          "#8181D5", "#94888B", "#FED6BD", "#536D31", "#6EFF92", "#E4E8FF", "#20E200", "#FFD0F2",
          "#4C83A1", "#BD7322", "#915C4E", "#8C4787", "#025117", "#A2AA45", "#2D1B21", "#A9DDB0",
          "#FF4F78", "#528500", "#009A2E", "#17FCE4", "#71555A", "#525D82", "#00195A", "#967874",
          "#555558", "#0B212C", "#1E202B", "#EFBFC4", "#6F9755", "#6F7586", "#501D1D", "#372D00",
          "#741D16", "#5EB393", "#B5B400", "#DD4A38", "#363DFF", "#AD6552", "#6635AF", "#836BBA",
          "#98AA7F", "#464836", "#322C3E", "#7CB9BA", "#5B6965", "#707D3D", "#7A001D", "#6E4636",
          "#443A38", "#AE81FF", "#489079", "#897334", "#009087", "#DA713C", "#361618", "#FF6F01",
          "#006679", "#370E77", "#4B3A83", "#C9E2E6", "#C44170", "#FF4526", "#73BE54", "#C4DF72",
          "#ADFF60", "#00447D", "#DCCEC9", "#BD9479", "#656E5B", "#EC5200", "#FF6EC2", "#7A617E",
          "#DDAEA2", "#77837F", "#A53327", "#608EFF", "#B599D7", "#A50149", "#4E0025", "#C9B1A9",
          "#03919A", "#1B2A25", "#E500F1", "#982E0B", "#B67180", "#E05859", "#006039", "#578F9B",
          "#305230", "#CE934C", "#B3C2BE", "#C0BAC0", "#B506D3", "#170C10", "#4C534F", "#224451",
          "#3E4141", "#78726D", "#B6602B", "#200441", "#DDB588", "#497200", "#C5AAB6", "#033C61",
          "#71B2F5", "#A9E088", "#4979B0", "#A2C3DF", "#784149", "#2D2B17", "#3E0E2F", "#57344C",
          "#0091BE", "#E451D1", "#4B4B6A", "#5C011A", "#7C8060", "#FF9491", "#4C325D", "#005C8B",
          "#E5FDA4", "#68D1B6", "#032641", "#140023", "#8683A9", "#CFFF00", "#A72C3E", "#34475A",
          "#B1BB9A", "#B4A04F", "#8D918E", "#A168A6", "#813D3A", "#425218", "#DA8386", "#776133",
          "#563930", "#8498AE", "#90C1D3", "#B5666B", "#9B585E", "#856465", "#AD7C90", "#E2BC00",
          "#E3AAE0", "#B2C2FE", "#FD0039", "#009B75", "#FFF46D", "#E87EAC", "#DFE3E6", "#848590",
          "#AA9297", "#83A193", "#577977", "#3E7158", "#C64289", "#EA0072", "#C4A8CB", "#55C899",
          "#E78FCF", "#004547", "#F6E2E3", "#966716", "#378FDB", "#435E6A", "#DA0004", "#1B000F",
          "#5B9C8F", "#6E2B52", "#011115", "#E3E8C4", "#AE3B85", "#EA1CA9", "#FF9E6B", "#457D8B",
          "#92678B", "#00CDBB", "#9CCC04", "#002E38", "#96C57F", "#CFF6B4", "#492818", "#766E52",
          "#20370E", "#E3D19F", "#2E3C30", "#B2EACE", "#F3BDA4", "#A24E3D", "#976FD9", "#8C9FA8",
          "#7C2B73", "#4E5F37", "#5D5462", "#90956F", "#6AA776", "#DBCBF6", "#DA71FF", "#987C95",
          "#52323C", "#BB3C42", "#584D39", "#4FC15F", "#A2B9C1", "#79DB21", "#1D5958", "#BD744E",
          "#160B00", "#20221A", "#6B8295", "#00E0E4", "#102401", "#1B782A", "#DAA9B5", "#B0415D",
          "#859253", "#97A094", "#06E3C4", "#47688C", "#7C6755", "#075C00", "#7560D5", "#7D9F00",
          "#C36D96", "#4D913E", "#5F4276", "#FCE4C8", "#303052", "#4F381B", "#E5A532", "#706690",
          "#AA9A92", "#237363", "#73013E", "#FF9079", "#A79A74", "#029BDB", "#FF0169", "#C7D2E7",
          "#CA8869", "#80FFCD", "#BB1F69", "#90B0AB", "#7D74A9", "#FCC7DB", "#99375B", "#00AB4D",
          "#ABAED1", "#BE9D91", "#E6E5A7", "#332C22", "#DD587B", "#F5FFF7", "#5D3033", "#6D3800",
          "#FF0020", "#B57BB3", "#D7FFE6", "#C535A9", "#260009", "#6A8781", "#A8ABB4", "#D45262",
          "#794B61", "#4621B2", "#8DA4DB", "#C7C890", "#6FE9AD", "#A243A7", "#B2B081", "#181B00",
          "#286154", "#4CA43B", "#6A9573", "#A8441D", "#5C727B", "#738671", "#D0CFCB", "#897B77",
          "#1F3F22", "#4145A7", "#DA9894", "#A1757A", "#63243C", "#ADAAFF", "#00CDE2", "#DDBC62",
          "#698EB1", "#208462", "#00B7E0", "#614A44", "#9BBB57", "#7A5C54", "#857A50", "#766B7E",
          "#014833", "#FF8347", "#7A8EBA", "#274740", "#946444", "#EBD8E6", "#646241", "#373917",
          "#6AD450", "#81817B", "#D499E3", "#979440", "#011A12", "#526554", "#B5885C", "#A499A5",
          "#03AD89", "#B3008B", "#E3C4B5", "#96531F", "#867175", "#74569E", "#617D9F", "#E70452",
          "#067EAF", "#A697B6", "#B787A8", "#9CFF93", "#311D19", "#3A9459", "#6E746E", "#B0C5AE",
          "#84EDF7", "#ED3488", "#754C78", "#384644", "#C7847B", "#00B6C5", "#7FA670", "#C1AF9E",
          "#2A7FFF", "#72A58C", "#FFC07F", "#9DEBDD", "#D97C8E", "#7E7C93", "#62E674", "#B5639E",
          "#FFA861", "#C2A580", "#8D9C83", "#B70546", "#372B2E", "#0098FF", "#985975", "#20204C",
          "#FF6C60", "#445083", "#8502AA", "#72361F", "#9676A3", "#484449", "#CED6C2", "#3B164A",
          "#CCA763", "#2C7F77", "#02227B", "#A37E6F", "#CDE6DC", "#CDFFFB", "#BE811A", "#F77183",
          "#EDE6E2", "#CDC6B4", "#FFE09E", "#3A7271", "#FF7B59", "#4E4E01", "#4AC684", "#8BC891",
          "#BC8A96", "#CF6353", "#DCDE5C", "#5EAADD", "#F6A0AD", "#E269AA", "#A3DAE4", "#436E83",
          "#002E17", "#ECFBFF", "#A1C2B6", "#50003F", "#71695B", "#67C4BB", "#536EFF", "#5D5A48",
          "#890039", "#969381", "#371521", "#5E4665", "#AA62C3", "#8D6F81", "#2C6135", "#410601",
          "#564620", "#E69034", "#6DA6BD", "#E58E56", "#E3A68B", "#48B176", "#D27D67", "#B5B268",
          "#7F8427", "#FF84E6", "#435740", "#EAE408", "#F4F5FF", "#325800", "#4B6BA5", "#ADCEFF",
          "#9B8ACC", "#885138", "#5875C1", "#7E7311", "#FEA5CA", "#9F8B5B", "#A55B54", "#89006A",
          "#AF756F", "#2A2000", "#7499A1", "#FFB550", "#00011E", "#D1511C", "#688151", "#BC908A",
          "#78C8EB", "#8502FF", "#483D30", "#C42221", "#5EA7FF", "#785715", "#0CEA91", "#FFFAED",
          "#B3AF9D", "#3E3D52", "#5A9BC2", "#9C2F90", "#8D5700", "#ADD79C", "#00768B", "#337D00",
          "#C59700", "#3156DC", "#944575", "#ECFFDC", "#D24CB2", "#97703C", "#4C257F", "#9E0366",
          "#88FFEC", "#B56481", "#396D2B", "#56735F", "#988376", "#9BB195", "#A9795C", "#E4C5D3",
          "#9F4F67", "#1E2B39", "#664327", "#AFCE78", "#322EDF", "#86B487", "#C23000", "#ABE86B",
          "#96656D", "#250E35", "#A60019", "#0080CF", "#CAEFFF", "#323F61", "#A449DC", "#6A9D3B",
          "#FF5AE4", "#636A01", "#D16CDA", "#736060", "#FFBAAD", "#D369B4", "#FFDED6", "#6C6D74",
          "#927D5E", "#845D70", "#5B62C1", "#2F4A36", "#E45F35", "#FF3B53", "#AC84DD", "#762988",
          "#70EC98", "#408543", "#2C3533", "#2E182D", "#323925", "#19181B", "#2F2E2C", "#023C32",
          "#9B9EE2", "#58AFAD", "#5C424D", "#7AC5A6", "#685D75", "#B9BCBD", "#834357", "#1A7B42",
          "#2E57AA", "#E55199", "#316E47", "#CD00C5", "#6A004D", "#7FBBEC", "#F35691", "#D7C54A",
          "#62ACB7", "#CBA1BC", "#A28A9A", "#6C3F3B", "#FFE47D", "#DCBAE3", "#5F816D", "#3A404A",
          "#7DBF32", "#E6ECDC", "#852C19", "#285366", "#B8CB9C", "#0E0D00", "#4B5D56", "#6B543F",
          "#E27172", "#0568EC", "#2EB500", "#D21656", "#EFAFFF", "#682021", "#2D2011", "#DA4CFF",
          "#70968E", "#FF7B7D", "#4A1930", "#E8C282", "#E7DBBC", "#A68486", "#1F263C", "#36574E",
          "#52CE79", "#ADAAA9", "#8A9F45", "#6542D2", "#00FB8C", "#5D697B", "#CCD27F", "#94A5A1",
          "#790229", "#E383E6", "#7EA4C1", "#4E4452", "#4B2C00", "#620B70", "#314C1E", "#874AA6",
          "#E30091", "#66460A", "#EB9A8B", "#EAC3A3", "#98EAB3", "#AB9180", "#B8552F", "#1A2B2F",
          "#94DDC5", "#9D8C76", "#9C8333", "#94A9C9", "#392935", "#8C675E", "#CCE93A", "#917100",
          "#01400B", "#449896", "#1CA370", "#E08DA7", "#8B4A4E", "#667776", "#4692AD", "#67BDA8",
          "#69255C", "#D3BFFF", "#4A5132", "#7E9285", "#77733C", "#E7A0CC", "#51A288", "#2C656A",
          "#4D5C5E", "#C9403A", "#DDD7F3", "#005844", "#B4A200", "#488F69", "#858182", "#D4E9B9",
          "#3D7397", "#CAE8CE", "#D60034", "#AA6746", "#9E5585", "#BA6200"]
