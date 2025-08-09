import numpy as np
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os

train_base_dir = '/data/coding/patent_b/train/'
df_train_question = pd.read_json("/data/coding/patent_b/train/train.jsonl",lines=True)
train_question_vector = np.load('all_train_b_question_vectors.npy')

base_dir = '/data/coding/patent_b/test/'
df_question = pd.read_json("/data/coding/patent_b/test/test.jsonl",lines=True)
question_vector = np.load('all_test_b_question_vectors.npy')

test_pdf_image_vectors = np.load("test_b_pdf_img_vectors.npy")
test_pdf_image_page_num_mapping = pd.read_csv('test_b_pdf_img_page_num_mapping.csv')

#os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MAX_PIXELS"] = "1568000" # 1568000  2000 token
#os.environ['FPS_MAX_FRAMES'] = "2"
model_path = "/data/coding/lora_qwen25_vl_32b_for_b/v0-20250802-085531/checkpoint-215-merged/"
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


def get_similar_question_embedding(question_idx,top_k=2):
    query_vec = question_vector[question_idx] # 当前的
    cos_sim = np.dot(train_question_vector, query_vec) / (np.linalg.norm(train_question_vector, axis=1) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-(top_k):][::-1]
    # 返回的top_k_indices中不包含question_idx本身
    # if question_idx in top_k_indices:
    #     top_k_indices = top_k_indices[top_k_indices != question_idx]
    retrived_question_idx = top_k_indices[:top_k] # 最相近的top_k个
    return retrived_question_idx # 返回一个list,大小最大为top_k

def get_options_for_similar_answer(retrived_question_idx):
    options_str = '回答风格示例: '
    for idx in retrived_question_idx:
        ans = df_train_question.loc[idx, 'answer']
        options_str += (ans +'\n')
    return options_str+'\n\n'



def get_similar_image_embedding(base_dir,document_name,question_idx,top_k,pic_page_num):
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = test_pdf_image_page_num_mapping[test_pdf_image_page_num_mapping['file_name']==document_name]['index'].values
    candidate_vec = test_pdf_image_vectors[vec_idx]
    query_vec = question_vector[question_idx]
    cos_sim = np.dot(candidate_vec, query_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(query_vec))
    # 获取最相似的top_k个索引
    top_k_indices = np.argsort(cos_sim)[-(top_k+1):][::-1]
    retrived_idx = vec_idx[top_k_indices] # 最相近的top_k个
    retrived_page_num = test_pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()
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


def get_image_answer(document_name,question,question_idx):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += ("专利内容为：\n")
    retrived_page_list = get_similar_image_embedding(base_dir,document_name,question_idx,2,-1)
    # 排序
    retrived_page_num = sorted(retrived_page_list)
    query = ''
    images = []
    retrived_list = []
    for i in range(len(retrived_page_num)):
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.jpg'
        retrived_list.append(image_file)
    question2 = ("\n\n请你在分析专利内容后，回答我的问题：\n")
    question2 += "【我的问题】【"
    question2 += (question +"】\n")
    question2 += ('请仔细思考，在思考结束后，请直接给出你的答案：')
    query += question1
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question1},
            ]
        }
    ]
    for i in range(0,len(retrived_list)):
        query += '<image>'
        images.append(retrived_list[i])
        messages[0]['content'].append({
            "type": "image",
            "image": retrived_list[i],
            "max_pixels": 1568000
        })
    query += question2
    messages[0]['content'].append({
        "type": "text",
        "text": question2
    })
    return origin_vllm(messages,2000)


def get_mix_answer_img(document_name,pic_page_num,question,question_idx,if_need_other=True):
    question1 ="你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
    question1 += ("该问题针对于这页专利内容里面的图进行提问：\n")
    retrived_page_list = get_similar_image_embedding(base_dir,document_name,question_idx,2,pic_page_num)
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    images = []
    query = ''
    for i in range(len(retrived_page_num)):
        image_file = base_dir + '/pdf_img/' + document_name.split('.')[0] + '/' + str(retrived_page_num[i]) +'.jpg'
        retrived_list.append(image_file)
    #print(retrived_list)
    if if_need_other: # 如果不是只用图片回答
        question2 = ("\n\n其他的相关专利内容为：\n")
    question3 = ("\n\n请你在分析专利内容后，回答我的问题：\n")
    question3 += "【我的问题】【"
    question3 += (question +"】\n")
    
    if "位置" in question and if_need_other:
        question3 += ('请仔细思考，在思考结束后，请直接给出你的答案：')
    elif "位置" in question:
        question3 += (
            "请仔细思考，你需要特别注意，图中部件的上下、前后、左右位置判断应以标号线所指代的实际结构为准，而不是仅凭直观看数字。"
        )
        question3 += (
            "在思考结束后，请直接给出你的答案："
        )
    else:
        question3 += ('请仔细思考，在思考结束后，请直接给出你的答案：')

    query += question1
    query += '<image>'
    images.append(base_dir + '/pdf_img/' + document_name.split('.')[0]+ '/' + str(pic_page_num) +'.jpg')
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question1},
                {
                        "type": "image",
                        "image": base_dir + '/pdf_img/' + document_name.split('.')[0]+ '/' + str(pic_page_num) +'.jpg',
                        "max_pixels": 1568000 if if_need_other else 2352000
                },
                
            ]
        }
    ]
    if if_need_other: # 如果不是只用图片回答
        query += question2
        
        messages[0]['content'].append({"type": "text", "text": question2})
        for i in range(0,len(retrived_list)):
            query += '<image>'
            images.append(retrived_list[i])
            messages[0]['content'].append({
                "type": "image",
                "image": retrived_list[i],
                "max_pixels": 1568000
            })
    query += question3
    messages[0]['content'].append({
        "type": "text",
        "text": question3
    })
    return origin_vllm(messages,768)


def classify_question(text):
    question = """你是一个内容分类专家，请判断用户的这个问题能否直接通过看图回答，还是需要参考其他的相关信息来回答。
    判断规则：已知图是结构图，里面只有部件序号，没有部件名称。如果用户的问题是要通过看图判断某些部件的位置关系，这类问题可以直接通过看图回答；如果用户问题涉及到询问部件是什么、部件名称功能和原理等，这些问题需要参考其他的相关信息来回答。
    对于只需要看图回答的问题，请回答字母"Y"；对于需要参考其他的相关信息来回答的问题，请回答字母"N"。
    
    给你提供一些示例
    示例1： 在文件中第5页提供的图片中，编号为4的部件是什么？
    解析：询问部件名称，图里面是没有的
    回答：N

    示例2:基于文件中第6页的图片，部件4位于哪个部件的延伸方向上？
    解析：询问部件位置关系，图里面是可以看出来的
    回答：Y

    示例3:根据文件中第7页的图片，部件41位于部件3的什么位置？
    解析：询问部件位置关系，图里面是可以看出来的
    回答：Y

    你要判断的用户问题是：
    """
    question += text + "\n"
    question += "请直接回答分类结果，不要解释，你的答案为："
    messages = [
        {
            "role": "user",
            "content":[
                {"type": "text", "text": question}
            ]
        }
    ]
    return origin_vllm(messages,2000)



def get_final_answer(text,answer_style):
    question = "你是一个内容提取专家，请从文本中判断，这段描述想表达的准确答案是什么。请仔细思考，在思考结束后，输出简要的答案（通常20个字词以内）。"
    question +="""
    为了便于你回答，我给你提供几个示例：
    示例1
    文本内容为："根据专利内容和图2的描述：\n\n- 编号为15的部件是**滤网**。\n- 编号为12的部件是**连接法兰**。\n\n从图2中可以看出，滤网（15）位于连接法兰（12）的**下方**。\n\n因此，正确答案是在12的下方"
    输出的答案为：在12的下方

    示例2
    文本内容为："根据专利内容，调节可移动折弯模架的位置是通过丝杆机构（部件7）实现的。丝杆机构包括丝杆（71）和丝杆滑块（72），通过调节手轮（74）转动丝杆，从而带动丝杆滑块移动，进而控制导轨滑块（6）沿导轨（5）移动，最终实现可移动折弯模架（1）的位置调节。因此，首先需要操作的部件是丝杆机构（部件7）。"
    输出的答案为：部件7

    示例3
    文本内容为："该专利提供了一种用于滚筒输送机的货物靠边规整处理机构，通过倾斜设置的转辊和联动皮带，实现货物自动靠边规整，减少损伤，提高输送效率。"
    输出的答案为：实现货物自动靠边规整

    示例4:
    文本内容为："定位杆"
    输出的答案为： 定位杆

    示例5:
    文本内容为："部件22位于部件23的左侧"
    输出的答案为： 部件22位于部件23的左侧
    """
    question += ('同时为了便于你回答，我再给你提供一些答案的风格示例：\n')
    question += answer_style + '\n'
    question += "你要判断的文本内容为：\n"
    question += text
    question += "请直接回答文本想要表达的准确答案（风格和前面的示例类似，通常20个字词以内，并且不要改变原始回答的意思），不要解释，你输出的答案为："
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
query_list = []
image_list = []
for i in trange(len(df_question)):
    question = df_question.loc[i,'question']
    document_name = df_question.loc[i,'document']
    question_type = ''
    if_need_other = True
    answer = ''
    style_answer = ''
    answer_style = get_options_for_similar_answer(get_similar_question_embedding(i,2))
    if "第" in question and "页" in question and "图": # 问题含有图片
        pic_page_num = re.findall(r"第(\d+)页", question)[0]
        pic_page_num = int(pic_page_num)
        question_type = classify_question(question) # 判断问题类型
        if 'Y' in question_type or 'y' in question_type: # 直接通过看图回答
            if_need_other = False
        else:
            if_need_other = True
        answer = get_mix_answer_img(document_name,pic_page_num,question,i,if_need_other)
        style_answer = get_final_answer(answer,answer_style)
    else:
        answer = get_image_answer(document_name,question,i)
        style_answer = get_final_answer(answer,answer_style)
    result_dict= dict()
    result_dict['idx'] = str(i)
    result_dict['document'] = document_name
    result_dict['question'] = question
    result_dict['question_type'] = question_type
    result_dict['answer'] = answer
    result_dict['style_answer'] = style_answer
    with open('test_b_style_infer_if_need_ck215.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

