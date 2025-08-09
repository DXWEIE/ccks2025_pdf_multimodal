import numpy as np
import pandas as pd
base_dir = '/data/coding/patent_qa/test/'
df_question = pd.read_json("/data/coding/patent_qa/test/questions.jsonl",lines=True)
quesion_vector = np.load('all_test_question_vectors.npy')
test_ocr_page_num_mapping = pd.read_csv('test_ocr_page_num_mapping.csv')
test_ocr_vectors = np.load('test_ocr_vectors.npy')

# index	page_num	file_name
test_pdf_image_vectors_1 = np.load("test_pdf_img_vectors1.npy") # 前400个
test_pdf_image_page_num_mapping_1 = pd.read_csv('test_pdf_img_page_num_mapping1.csv') # 前400个
test_pdf_image_vectors_2 = np.load("test_pdf_img_vectors2.npy") # 后400个
test_pdf_image_page_num_mapping_2 = pd.read_csv('test_pdf_img_page_num_mapping2.csv') # 后400个

test_pdf_image_vectors = np.concatenate([test_pdf_image_vectors_1,test_pdf_image_vectors_2])
test_pdf_image_page_num_mapping_2['index']=test_pdf_image_page_num_mapping_2['index']+test_pdf_image_vectors_1.shape[0]
test_pdf_image_page_num_mapping = pd.concat([test_pdf_image_page_num_mapping_1,test_pdf_image_page_num_mapping_2]).reset_index(drop=True)

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
from swift.plugin import InferStats
#from swift.llm import VllmEngine
from vllm import LLM, SamplingParams
import os
#os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["MAX_PIXELS"] = "1229312"
#os.environ['FPS_MAX_FRAMES'] = "2"
#model_path = "/data/coding/lora_qwen3_8b/v0-20250608-134839/checkpoint-25-merged"
#engine = VllmEngine(model_path,model_type='qwen3',gpu_memory_utilization=0.9,max_model_len=12288)
# 试一下qwen3的llm是不是效果好一些
from vllm import LLM, SamplingParams
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
text_qa_llm = LLM(model="/data/coding/lora_qwen3_32b/v0-20250609-220537/checkpoint-25-merged",gpu_memory_utilization=0.9,max_model_len=12288,tensor_parallel_size=4)

def infer_batch(engine, infer_requests):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config)
    response = resp_list[0].choices[0].message.content
    return response


# 给定了图片的情况下，有图片本身，也需要召回对应的文本
def get_similar_text_embedding(base_dir,document_name,question_idx,top_k):
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = test_ocr_page_num_mapping[test_ocr_page_num_mapping['file_name']==document_name]['index'].values
    candidate_vec = test_ocr_vectors[vec_idx]
    query_vec = quesion_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-top_k:][::-1]
    retrived_idx = vec_idx[top_k_indices] # 最相近的top_k个
    retrived_page_num = test_ocr_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    # text_list = []
    # for i in range(len(retrived_page_num)):
    #     text_file = base_dir + '/pdf_ocr/' + document_name + '/' + str(retrived_page_num[i]) +'.txt'
    #     with open(text_file,'r',encoding='utf-8') as f:
    #         text_list.append(f.read())
    return retrived_page_num # 返回一个list,大小最大为top_k


def get_similar_image_embedding(base_dir,document_name,question_idx,top_k):
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = test_pdf_image_page_num_mapping[test_pdf_image_page_num_mapping['file_name']==document_name]['index'].values
    candidate_vec = test_pdf_image_vectors[vec_idx]
    query_vec = quesion_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-top_k:][::-1]
    retrived_idx = vec_idx[top_k_indices] # 最相近的top_k个
    retrived_page_num = test_pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    # image_list = []
    # for i in range(len(retrived_page_num)):
    #     image_file = base_dir + '/pdf_img/' + document_name + '/' + str(retrived_page_num[i]) +'.jpg'
    #     image_list.append(image_file)
    return retrived_page_num # 返回一个list,大小最大为top_k

def get_text_answer_vl(document_name,question,options,quesion_idx):
    prompt ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    prompt += "【我的问题】【"
    prompt += (question +"】\n")
    prompt += "【选项】【"
    prompt += (' '.join(options) + "】\n")
    prompt += ("专利内容为：\n")
    retrived_list = get_similar_text_embedding(base_dir,document_name,quesion_idx,2)
    prompt += '\n'.join(retrived_list)
    prompt += ("\n\n请你分析专利内容后，回答我的单选题，回答正确选项字母，你的答案为：\n")
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    data = dict()
    data['messages'] = messages
    infer_requests = [InferRequest(**data)]
    response = infer_batch(engine, infer_requests)
    return response

def get_text_answer_qwen3(document_name,question,options,quesion_idx):
    prompt ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    prompt += "【我的问题】【"
    prompt += (question +"】\n")
    prompt += "【选项】【"
    prompt += (' '.join(options) + "】\n")
    prompt += ("专利内容为：\n")
    retrived_page_list = get_similar_text_embedding(base_dir,document_name,quesion_idx,2)
    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for i in range(len(retrived_page_num)):
        text_file = base_dir + '/pdf_ocr/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.txt'
        with open(text_file,'r',encoding='utf-8') as f:
            retrived_list.append(f.read())
    prompt += '\n'.join(retrived_list)
    prompt += ("\n\n请你分析专利内容后，回答我的单选题，回答正确选项字母，你的答案为：\n")
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    sampling_params = SamplingParams(temperature=0)
    outputs = text_qa_llm.generate(prompt,sampling_params)
    return outputs[0].outputs[0].text
    # data = dict()
    # data['messages'] = messages
    # infer_requests = [InferRequest(**data)]
    # response = infer_batch(engine, infer_requests)
    # return response

def get_image_answer(document_name,question,options,quesion_idx):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的单选题。\n"
    question1 += "【我的问题】【"
    question1 += (question +"】\n")
    question1 += "【选项】【"
    question1 += (' '.join(options) + "】\n")
    question1 += ("专利内容为：\n")
    retrived_page_list = get_similar_image_embedding(base_dir,document_name,question_idx,2)
    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for i in range(len(retrived_page_num)):
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.jpg'
        retrived_list.append(image_file)
    question2 = ("\n\n请你分析专利内容后，回答我的单选题，直接回答正确选项字母，你的答案为：\n")
    if len(retrived_list)>1:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question1},
                    {
                        "type": "image",
                        "image": retrived_list[0],
                    },
                    {
                        "type": "image",
                        "image": retrived_list[1],
                    },
                    {"type": "text", "text": question2},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question1},
                    {
                        "type": "image",
                        "image": retrived_list[0],
                    },
                    {"type": "text", "text": question2},
                ],
            }
        ]
    data = dict()
    data['messages'] = messages
    infer_requests = [InferRequest(**data)]
    response = infer_batch(engine, infer_requests)
    return response

def get_mix_answer(document_name,pic_page_num,question,options,question_idx):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += "【我的问题】【"
    question1 += (question +"】\n")
    question1 += "【选项】【"
    question1 += (' '.join(options) + "】\n")
    question1 += ("该问题直接指向的专利页内容为：\n")
    retrived_page_list = get_similar_text_embedding(base_dir,document_name,question_idx,2)
    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for i in range(len(retrived_page_num)):
        text_file = base_dir + '/pdf_ocr/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.txt'
        with open(text_file,'r',encoding='utf-8') as f:
            retrived_list.append(f.read())
    question2 = ("\n\n除了问题直接指向的专利页外，该专利其他相关内容为：\n")
    question3 = ("\n\n请你分析专利内容后，回答我的单选题，直接回答正确选项字母，你的答案为：\n")
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question1},
                    {
                        "type": "image",
                        "image": base_dir + '/pdf_img/' + document_name.split('.')[0]+ '/' + str(pic_page_num) +'.jpg',
                    },
                    {"type": "text", "text": question2},
                    {"type": "text", "text": '\n'.join(retrived_list)},
                    {"type": "text", "text": question3},
                ],
            }
    ]
    data = dict()
    data['messages'] = messages
    infer_requests = [InferRequest(**data)]
    response = infer_batch(engine, infer_requests)
    return response



import re
from tqdm import trange
import json
# 实时把结果写入jsonl文件
for i in trange(len(df_question)):
    question = df_question.loc[i,'question']
    document_name = df_question.loc[i,'document']
    options = df_question.loc[i,'options']
    #true_answer = df_question.loc[i,'answer']
    full_question = question + ' '.join(options)
    answer = ''
    text_answer = ''
    image_answer = ''
    if "第" in question and "页" in question and "图": # 问题含有图片
        continue
    else:
        text_answer = get_text_answer_qwen3(document_name,question,options,i) # 使用文本来回答
        #image_answer = get_image_answer(document_name,question,options,i) # 使用图像来回答
    result_dict= dict()
    result_dict['idx'] = str(i)
    result_dict['document'] = document_name
    result_dict['question'] = question
    result_dict['options'] = options
    #result_dict['pure_pic_answer'] = answer if answer else ''
    result_dict['text_answer'] = text_answer if text_answer else ''
    #result_dict['image_answer'] = image_answer if image_answer else ''
    with open('test_set_by_qwen3_32b_0609.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
