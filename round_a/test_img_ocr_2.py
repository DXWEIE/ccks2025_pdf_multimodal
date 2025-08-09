from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
import os
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["MAX_PIXELS"] = "1229312"
os.environ['FPS_MAX_FRAMES'] = "2"
model_path = "/data/coding/llm_model/Qwen/Qwen2___5-VL-7B-Instruct/"
# 多卡设置tensor_parallel_size为卡数
#engine = VllmEngine(model_path,model_type='qwen2_5_vl',gpu_memory_utilization=0.9,limit_mm_per_prompt={"image": 1},tensor_parallel_size=2)
processor = AutoProcessor.from_pretrained(model_path)

vllm_qwen = LLM(
        model=model_path,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,limit_mm_per_prompt={"image": 1},tensor_parallel_size=2
)

from qwen_vl_utils import process_vision_info
def vllm_qwen_ocr_cn(img_path):
    prompt_ocr="你是一个OCR专家，请提取以下图片中的所有文字内容，并原样返回。"
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_ocr},
                    {
                        "type": "image",
                        "image": img_path,
                    },
                ],
            }
    ]

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)

    sampling_params = SamplingParams(temperature=0.0,
                                max_tokens=10240,
                                stop_token_ids=stop_token_ids)

    outputs = vllm_qwen.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
        },
            sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
    return generated_text

base_dir = '/data/coding/patent_qa/test/'
pdf_file_list = [x for x in os.listdir(base_dir+'/pdf_img/')]
# 前400个，排序
pdf_file_list = sorted(pdf_file_list)[400:]

for pdf_file in pdf_file_list:
    file_name = pdf_file.split('.')[0]
    file_list = [x for x in os.listdir(base_dir+'/pdf_img/'+file_name) if 'jpg' in x] # pdf文件存储的jpg
    os.makedirs(base_dir+'/pdf_ocr/'+file_name,exist_ok=True)
    for k in tqdm(range(len(file_list))):
        img_path = base_dir+'/pdf_img/'+file_name + '/' + file_list[k]
        page_num = int(file_list[k].split('.')[0])
        # ocr
        response = vllm_qwen_ocr_cn(img_path)
        # 保存ocr结果为txt
        with open(base_dir+f'/pdf_ocr/{file_name}/{page_num}.txt', 'w', encoding='utf-8') as text_file:
            text_file.write(response)
    print(file_name,'  ok')
    with open(base_dir+'test_ocr_finished_list_2.txt', 'a') as f:
        f.write(file_name + '\n')