import numpy as np
tmp_np = np.load("brain_lesion_array_32_32_test.npy")
tmp_np = tmp_np[:2458,:,:,:]
list_i = [201, 205, 206, 207, 208, 209, 210, 212, 214, 216, 218, 220, 222, 224, 226, 258, 260, 262, 264, 268, 270, 274, 564, 566, 568, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 596, 598, 600, 603, 604, 605, 701, 702, 703, 704, 705, 716, 717, 718, 719, 720, 721, 722, 726, 728, 730, 732, 734, 791, 953, 954, 959, 961, 963, 965, 967, 1001, 1003, 1005, 1016, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1495, 1496, 1497, 1555, 1557, 1558, 1559, 1582, 1693, 1752, 1839, 1841, 1843, 1845, 2213, 2214, 2394, 2395]
for i in range(0,len(list_i)-1):
    # print(max(list_i))
    tmp_np = np.delete(tmp_np, max(list_i), axis = 0)
    list_i.remove(max(list_i))
    print(tmp_np.shape, max(list_i))
np.save("brain_lesion_array_32_32_test_final.npy",tmp_np)    
    