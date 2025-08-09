import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["MAX_PIXELS"] = '1229312' # 1003520
from gme_inference import GmeQwen2VL
gme = GmeQwen2VL(model_name='/data/coding/llm_model/iic/gme-Qwen2-VL-7B-Instruct',max_image_tokens=1280)
import os
import pandas as pd
import numpy as np
import tqdm
from warnings import filterwarnings
# 过滤掉一些警告
filterwarnings("ignore")
base_dir = '/data/coding/patent_qa/test/'
pdf_file_list = [x for x in os.listdir(base_dir+'/pdf_img/')]
pdf_file_list = sorted(pdf_file_list)[:400]
files_total_cnt = 0
for pdf_file in pdf_file_list:
    file_name = pdf_file.split('.')[0]
    file_list = [x for x in os.listdir(base_dir+'/pdf_img/'+file_name) if 'jpg' in x]
    files_total_cnt +=len(file_list)

from tqdm import trange,tqdm
# 文件有多个，每个都需要存映射关系
img_page_num_list = []
img_name_list = []
img_vectors = np.empty((files_total_cnt, 3584)) # 向量维度是3584维
idx = 0
cnt = 0 
for pdf_file in pdf_file_list:
    file_name = pdf_file.split('.')[0]
    file_list = [x for x in os.listdir(base_dir+'/pdf_img/'+file_name) if 'jpg' in x] # pdf文件存储的jpg
    for k in range(len(file_list)):  
        image_path = base_dir+'/pdf_img/'+file_name + '/' + file_list[k]
        e_text = gme.get_image_embeddings(images=[image_path])
        img_vectors[idx] = e_text[0].to('cpu').numpy()
        page_num = int(file_list[k].split('.')[0])
        img_page_num_list.append(page_num)
        img_name_list.append(file_name)
        idx+=1
    cnt +=1 
    print(cnt)
    
# 映射关系存储到pandas里面比较方便
img_page_num_mapping = pd.DataFrame({'index': range(len(img_page_num_list)), 'page_num': img_page_num_list, 'file_name': img_name_list})
# 将向量和页码映射关系存储到文件
np.save('test_pdf_img_vectors1.npy', img_vectors)
img_page_num_mapping.to_csv('test_pdf_img_page_num_mapping1.csv', index=False) # 存储映射关系