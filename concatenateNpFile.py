import numpy as np

health = np.load("/Users/chenjingkun/Documents/code/image_tools/health.npy")
lesion = np.load("/Users/chenjingkun/Documents/code/image_tools/lesion.npy")
print(health.shape)
print(lesion.shape)
# health_array = np.concatenate((health1[240:300,:,:], health2[240:300,:,:]), axis=0)
# health_array = np.concatenate((health_array, health3[240:300,:,:]), axis=0)
# lesion_array = np.concatenate((lesion1[240:300,:,:], lesion2[240:300,:,:]), axis=0)
# lesion_array = np.concatenate((lesion_array, lesion3[240:300,:,:]), axis=0)
health_array = health[2001:2594,:,:]
lesion_array = lesion[2001:2594,:,:]
np.save('health_test.npy', health_array)
np.save('lesion_test.npy', lesion_array)