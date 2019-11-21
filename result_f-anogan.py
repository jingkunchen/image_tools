with open("/Users/chenjingkun/Documents/result/f-anogan/mapping_results-enc_ckpt_it50000.csv", "r") as f:
    for line in f.readlines():
        if int(line.strip().split(",")[0]) == 0:
            print(float(line.strip().split(",")[1])+float(line.strip().split(",")[2]))
            #print float(line.strip().split(",")[1])