import numpy as np
health0 = np.load("resize/data_array_c0.npy")[:,:,:,np.newaxis]
health1 = np.load("resize/fake_B_0.npy")
health2 = np.load("resize/fake_B_10.npy")
health3 = np.load("resize/fake_B_30.npy")
health4 = np.load("resize/fake_B_50.npy")
health5 = np.load("resize/fake_B_100.npy")
health6 = np.load("resize/fake_B_200.npy")
health7 = np.load("resize/fake_B_300.npy")
health8 = np.load("resize/fake_B_400.npy")
health9 = np.load("resize/fake_B_400.npy")
health10 = np.load("resize/fake_B_400.npy")
health11 = np.load("resize/fake_B_400.npy")
health12 = np.load("resize/fake_B_400.npy")
health13 = np.load("resize/fake_B_400.npy")
lesion = np.load("resize/gt_array_c0.npy")
edge = np.load("resize/gt_edge_c0.npy")
print(health0.shape, health1.shape, edge.shape)
health_array = np.concatenate((health0, health1), axis=0)
health_array = np.concatenate((health_array, health2), axis=0)
health_array = np.concatenate((health_array, health3), axis=0)
health_array = np.concatenate((health_array, health4), axis=0)
health_array = np.concatenate((health_array, health5), axis=0)
health_array = np.concatenate((health_array, health6), axis=0)
health_array = np.concatenate((health_array, health7), axis=0)
health_array = np.concatenate((health_array, health8), axis=0)
health_array = np.concatenate((health_array, health9), axis=0)
health_array = np.concatenate((health_array, health10), axis=0)
health_array = np.concatenate((health_array, health11), axis=0)
health_array = np.concatenate((health_array, health12), axis=0)
health_array = np.concatenate((health_array, health13), axis=0)
lesion_array = np.concatenate((lesion, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
lesion_array = np.concatenate((lesion_array, lesion), axis=0)
edge_array = np.concatenate((edge, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)
edge_array = np.concatenate((edge_array, edge), axis=0)


# health_array = np.concatenate((health_array, health3[240:300,:,:]), axis=0)
# lesion_array = np.concatenate((lesion1[240:300,:,:], lesion2[240:300,:,:]), axis=0)
# lesion_array = np.concatenate((lesion_array, lesion3[240:300,:,:]), axis=0)
print(health_array.shape, lesion_array.shape, edge_array.shape)
np.save('train_0_1000.npy', health_array)
np.save('test_0_1000.npy', lesion_array)
np.save('edge_0_1000.npy', edge_array)