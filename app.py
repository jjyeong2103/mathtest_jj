from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

st.title("자동 채점 모델 기반 자동 피드백")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")


st.markdown("---")
# 문항1-1

st.subheader("문항1-1")
st.markdown("$$ a^{2} \\times a^{5} = $$")

response = st.text_input('답안 :', key='answer_input_1_1')

# 모델의 이름 정하기
model_name_1_1 = "1-1_rnn_sp_60"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 60  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d = 1  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

model_1_1 = RNNModel(output_d, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model_1_1.load_state_dict(torch.load("./save/"+model_name_1_1+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_1 = AutoTokenizer.from_pretrained("./save/"+model_name_1_1)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_1(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_1(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기', key='button1_1_1_1'):
    
    # output차원에 맞추어 피드백 넣기
 st.write(response)
 if label[0] == 1:
     st.success('거듭제곱의 곱셈을 이해하고 있구나!', icon="✅")
 else:
     st.info('거듭제곱의 곱셈을 복습하세요!', icon="⚠️")

if st.button('힌트 보기', key='button1_1_1_2'):
    st.write('밑이 a로 같아요!')


