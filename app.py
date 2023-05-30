from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

st.title("자동 채점 모델 기반 자동 피드백")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")

#문항1-7

st.subheader("문항1-7")
st.markdown("$$ (2^4)^x \\times (2^2)^x=2^3 \\times 2^{3x} $$일 때, 자연수 $$x$$의 값을 구하시오.")

response = st.text_input('답안 :', "과정과 함께 작성하세요!")

#모델의 이름 정하기
model_name = "1-7_att_sp_140" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 140 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d = 3 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model = ATTModel(output_d, c) #ATTModel 쓰는경우

model.load_state_dict(torch.load("./save/"+model_name+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer = AutoTokenizer.from_pretrained("./save/"+model_name) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기'):
    #output차원에 맞추어 피드백 넣기
    st.write(response)
    
    if label:
        if label[1] == 1:
            st.success('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 이해하고 있구나!', icon="✅")
        else:
            st.info('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 복습하세요!', icon="ℹ️")
    else:
        st.write('힌트 버튼을 누른 후 고민하세요!')
else:
    if '과정과 함께 작성하세요!' in label:
        st.write('힌트 버튼을 누른 후 고민하세요!')
        
if st.button('힌트 보기'):
    st.write('밑이 2로 같으니, 지수를 정리하세요!')
 
st.markdown("---")
#문항1-8

st.subheader("문항1-8")
st.markdown("$$ (2^4)^x \\times (2^2)^x=2^3 \\times 2^{3x} $$일 때, 자연수 $$x$$의 값을 구하시오.")

response = st.text_input('답안 :', "과정과 함께 작성하세요!")

#모델의 이름 정하기
model_name = "1-7_att_sp_140" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 140 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d = 3 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model = ATTModel(output_d, c) #ATTModel 쓰는경우

model.load_state_dict(torch.load("./save/"+model_name+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer = AutoTokenizer.from_pretrained("./save/"+model_name) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기'):
    """
    output차원에 맞추어 피드백 넣기
    """
    st.write(response)
    if label[1] == 1:
        st.success('(다항식) 곱하기 (단항식)을 잘하는구나!', icon="✅")
    else :
        st.info('(다항식) 곱하기 (단항식)을 잘 생각해보자!', icon="ℹ️")

if st.button('힌트 보기'):
    st.write('지수를 정리하세요!')