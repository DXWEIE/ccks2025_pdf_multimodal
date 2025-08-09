import numpy as np
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os


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


#os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MAX_PIXELS"] = "1568000" # 1568000  2000 token
#os.environ['FPS_MAX_FRAMES'] = "2"
model_path = "/data/coding/llm_model/Qwen/Qwen2___5-VL-32B-Instruct"
vl_model = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 3},
    gpu_memory_utilization=0.9,
    tensor_parallel_size=4,
    max_model_len=8192,
    max_num_seqs=1
)
processor = AutoProcessor.from_pretrained(model_path)

def origin_vllm(messages,max_tokens=768):
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens
    )

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }
    else:
        llm_inputs = {
            "prompt": prompt
        }

    outputs = vl_model.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text


# 给定了图片的情况下，有图片本身，也需要召回对应的文本
def get_similar_text_embedding(base_dir,document_name,question_idx,top_k,pic_page_num):
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = train_ocr_page_num_mapping[train_ocr_page_num_mapping['file_name']==document_name]['index'].values
    candidate_vec = train_ocr_vectors[vec_idx]
    query_vec = quesion_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-(top_k+1):][::-1]
    retrived_idx = vec_idx[top_k_indices] # 最相近的top_k个
    retrived_page_num = train_pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    retrived_page_num = [int(x) for x in retrived_page_num]
    # 如果pic_page_num>=0，返回的top_k个中不包含pic_page_num
    if pic_page_num >= 0:
        if pic_page_num in retrived_page_num:
            retrived_page_num.remove(pic_page_num)
    # 只返回前top_k个
    retrived_page_num = retrived_page_num[:top_k] # 最多返回top_k个
    # retrive_page_num排序
    retrived_page_num = sorted(retrived_page_num) # 按照page_num排序
    return retrived_page_num # 返回一个list,大小最大为top_k

def get_similar_image_embedding(base_dir,document_name,question_idx,top_k,pic_page_num):
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = train_pdf_image_page_num_mapping[train_pdf_image_page_num_mapping['file_name']==document_name]['index'].values
    candidate_vec = train_pdf_image_vectors[vec_idx]
    query_vec = quesion_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-(top_k+1):][::-1]
    retrived_idx = vec_idx[top_k_indices] # 最相近的top_k个
    retrived_page_num = train_pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
    retrived_page_num = [int(x) for x in retrived_page_num]
    # 如果pic_page_num>=0，返回的top_k个中不包含pic_page_num
    if pic_page_num >= 0:
        if pic_page_num in retrived_page_num:
            retrived_page_num.remove(pic_page_num)
    # 只返回前top_k个
    retrived_page_num = retrived_page_num[:top_k] # 最多返回top_k个
    # retrive_page_num排序
    retrived_page_num = sorted(retrived_page_num) # 按照page_num排序
    return retrived_page_num # 返回一个list,大小最大为top_k


def get_image_answer(document_name,question,options,question_idx):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += ("专利内容为：\n")
    retrived_page_list = get_similar_image_embedding(base_dir,document_name,question_idx,2,-1)
    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for i in range(len(retrived_page_num)):
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.jpg'
        retrived_list.append(image_file)
    question2 = ("\n\n请你在分析专利内容后，回答我的问题：\n")
    question2 += "【我的问题】【"
    question2 += (question +"】请从下面的选项中选择最符合的一项。\n")
    question2 += "【选项】【"
    question2 += (' '.join(options) + "】\n")
    question2 += ('请仔细思考，直接回答正确答案对应的选项字母。在思考结束后，你的答案为：')
    
    # 改成简便的形式，retrived_list的大小是不固定的
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question1},
            ]
        }
    ]
    for i in range(0,len(retrived_list)):
        messages[0]['content'].append({
            "type": "image",
            "image": retrived_list[i],
            "max_pixels": 1568000
        })
    messages[0]['content'].append({
        "type": "text",
        "text": question2
    })
    return origin_vllm(messages)


def get_mix_answer_img(document_name,pic_page_num,question,options,question_idx):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += ("该问题针对于这页专利内容里面的图进行提问：\n")
    retrived_page_list = get_similar_image_embedding(base_dir,document_name,question_idx,2,pic_page_num)
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for i in range(len(retrived_page_num)):
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.jpg'
        retrived_list.append(image_file)
    print(retrived_list)
    question2 = ("\n\n其他的相关专利内容为：\n")
    question3 = ("\n\n请你在分析专利内容后，回答我的问题：\n")
    question3 += "【我的问题】【"
    question3 += (question +"】请从下面的选项中选择最符合的一项。\n")
    question3 += "【选项】【"
    question3 += (' '.join(options) + "】\n")
    if "位置" in question:
        question3 += ('请仔细思考，你主要参考提问针对的专利内容里面的对应的图的信息，如果可以回答，则直接回答(例如看图判断位置关系等)，如果无法回答，才参考其他信息(注意能直接回答的问题参考其他信息可能造成回答错误)。思考后，你的答案为：')
    else:
        question3 += ('请仔细思考，直接回答正确答案对应的选项字母。在思考结束后，你的答案为：')
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question1},
                {
                        "type": "image",
                        "image": base_dir + '/pdf_img/' + document_name.split('.')[0]+ '/' + str(pic_page_num) +'.jpg',
                        "max_pixels": 1568000
                },
                {"type": "text", "text": question2}
            ]
        }
    ]
    for i in range(0,len(retrived_list)):
        messages[0]['content'].append({
            "type": "image",
            "image": retrived_list[i],
            "max_pixels": 1568000
        })
    messages[0]['content'].append({
        "type": "text",
        "text": question3
    })
    return origin_vllm(messages)

def get_final_answer(text):
    question = "你是一个内容提取专家，请从文本中判断，这段描述想表达的准确答案是什么，要求直接回答代表准确答案的字母。文本内容为："
    question += text
    question += "请直接回答文本想要表达的准确答案的字母，不要解释，该字母为："
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question}
            ]
        }
    ]
    return origin_vllm(messages)



import re
from tqdm import trange
import json
# 实时把结果写入jsonl文件
for i in trange(len(df_question)):
    question = df_question.loc[i,'question']
    document_name = df_question.loc[i,'document']
    options = df_question.loc[i,'options']
    true_answer = df_question.loc[i,'answer']
    full_question = question + ' '.join(options)
    answer_img = ''
    answer1 = ''
    answer2 = ''
    if "第" in question and "页" in question and "图": # 问题含有图片
        pic_page_num = re.findall(r"第(\d+)页", question)[0]
        pic_page_num = int(pic_page_num)
        answer1 = get_mix_answer_img(document_name,pic_page_num,question,options,i)
        answer2 = get_final_answer(answer1)
    else:
        #answer = get_image_answer(document_name,question,options,i) # 使用图像来回答
        answer1 = ''
        answer2 = ''
    result_dict= dict()
    result_dict['idx'] = str(i)
    result_dict['document'] = document_name
    result_dict['question'] = question
    result_dict['options'] = options
    result_dict['answer1'] = answer1 if answer1 else ''
    result_dict['answer2'] = answer2 if answer2 else ''
    result_dict['true_answer'] = true_answer if true_answer else ''
    with open('test_qwen32b_vl_rag_3.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
