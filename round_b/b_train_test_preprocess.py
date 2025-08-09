#### 训练集预处理
#
# 1. pdf转jpg
import fitz  # PyMuPDF
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
base_dir = '/data/coding/patent_b/train/'
pdf_file_list  = [x for x in os.listdir(base_dir+'/documents/') if 'pdf' in x]

# 已经完成这部分
for file_name in tqdm(pdf_file_list):
    pdf_document = fitz.open(base_dir+'/documents/'+file_name)
    os.makedirs(base_dir+'/pdf_img/'+file_name.split('.')[0],exist_ok=True)
    # 获取第一页
    for i in range(pdf_document.page_count):
        page = pdf_document.load_page(i)  # 注意：页码从0开始
        # 将页面转换为图像
        pix = page.get_pixmap(dpi=600) # 这些文档600的dpi够了
        pix.save(base_dir+'/pdf_img/'+file_name.split('.')[0]+'/'+str(i+1)+'.jpg')


# 2. 图片结果存入向量库
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
base_dir = '/data/coding/patent_b/train/'
pdf_file_list = [x for x in os.listdir(base_dir+'/pdf_img/')]
files_total_cnt = 0
for pdf_file in pdf_file_list:
    file_name = pdf_file.split('.')[0]
    file_list = [x for x in os.listdir(base_dir+'/pdf_img/'+file_name) if 'jpg' in x]
    files_total_cnt +=len(file_list)

# 文件有多个，每个都需要存映射关系
img_page_num_list = []
img_name_list = []
img_vectors = np.empty((files_total_cnt, 3584)) # 向量维度是3584维
idx = 0
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
# 映射关系存储到pandas里面比较方便
img_page_num_mapping = pd.DataFrame({'index': range(len(img_page_num_list)), 'page_num': img_page_num_list, 'file_name': img_name_list})
# 将向量和页码映射关系存储到文件
np.save('train_b_pdf_img_vectors.npy', img_vectors)
img_page_num_mapping.to_csv('train_b_pdf_img_page_num_mapping.csv', index=False) # 存储映射关系

# 4. 读取问题生成问题的向量
df_question = pd.read_json('/data/coding/patent_b/train/train.jsonl',lines=True)

# 问题的vector进行保存
question_vectors = np.empty((len(df_question), 3584))
for i in range(len(df_question)):
    question = df_question.loc[i,'question']
    document_name = df_question.loc[i,'document']
    true_answer = df_question.loc[i,'answer']
    full_question = question
    query_vec = gme.get_text_embeddings(texts=[full_question])
    question_vectors[i] = query_vec[0].to('cpu').numpy()
# 保存问题的向量
np.save('all_train_b_question_vectors.npy', question_vectors)


###########测试集预处理

base_dir = '/data/coding/patent_b/test/'
# with open(base_dir+'test_finished_list_1.txt', 'r') as f:
#     finished_text = f.read()
# finished_files = finished_text.split('\n')

## 1.pdf转jpg
import fitz  # PyMuPDF
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
base_dir = '/data/coding/patent_b/test/'
pdf_file_list  = [x for x in os.listdir(base_dir+'/documents/') if 'pdf' in x]

finished_list = []
for file_name in tqdm(pdf_file_list):
    # if file_name in finished_files:
    #     continue
    pdf_document = fitz.open(base_dir+'/documents/'+file_name)
    os.makedirs(base_dir+'/pdf_img/'+file_name.split('.')[0],exist_ok=True)
    # 获取第一页
    for i in range(pdf_document.page_count):
        page = pdf_document.load_page(i)  # 注意：页码从0开始
        # 将页面转换为图像
        pix = page.get_pixmap(dpi=600) # 这些文档600的dpi够了
        pix.save(base_dir+'/pdf_img/'+file_name.split('.')[0]+'/'+str(i+1)+'.jpg')
    finished_list.append(file_name)
    # 完成的写入txt防止之后重复
    with open(base_dir+'test_finished_list_2.txt', 'a') as f:
        f.write(file_name + '\n')

# 2.图片存入向量库
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["MAX_PIXELS"] = '1229312' # 1003520
# from gme_inference import GmeQwen2VL
# gme = GmeQwen2VL(model_name='/data/coding/llm_model/iic/gme-Qwen2-VL-7B-Instruct',max_image_tokens=1280)


import os
import pandas as pd
import numpy as np
import tqdm
from warnings import filterwarnings
# 过滤掉一些警告
filterwarnings("ignore")
base_dir = '/data/coding/patent_b/test/'
pdf_file_list = [x for x in os.listdir(base_dir+'/pdf_img/')]
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
for i in range(len(pdf_file_list)):
    if i % 20==0:
        print('finished -- ', i,'\n')
    pdf_file = pdf_file_list[i]
    file_name = pdf_file.split('.')[0]
    file_list = [x for x in os.listdir(base_dir+'/pdf_img/'+file_name) if 'jpg' in x] # pdf文件存储的jpg
    for k in trange(len(file_list)):  
        image_path = base_dir+'/pdf_img/'+file_name + '/' + file_list[k]
        e_text = gme.get_image_embeddings(images=[image_path])
        img_vectors[idx] = e_text[0].to('cpu').numpy()
        page_num = int(file_list[k].split('.')[0])
        img_page_num_list.append(page_num)
        img_name_list.append(file_name)
        idx+=1
# 映射关系存储到pandas里面比较方便
img_page_num_mapping = pd.DataFrame({'index': range(len(img_page_num_list)), 'page_num': img_page_num_list, 'file_name': img_name_list})
# 将向量和页码映射关系存储到文件
np.save('test_b_pdf_img_vectors.npy', img_vectors)
img_page_num_mapping.to_csv('test_b_pdf_img_page_num_mapping.csv', index=False) # 存储映射关系

# 3.读取问题，生成问题向量
df_question = pd.read_json('/data/coding/patent_b/test/test.jsonl',lines=True)
# 问题的vector进行保存
question_vectors = np.empty((len(df_question), 3584))
for i in range(len(df_question)):
    question = df_question.loc[i,'question']
    document_name = df_question.loc[i,'document']
    full_question = question
    query_vec = gme.get_text_embeddings(texts=[full_question])
    question_vectors[i] = query_vec[0].to('cpu').numpy()
# 保存问题的向量
np.save('all_test_b_question_vectors.npy', question_vectors)

