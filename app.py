from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

st.title("🤖자동 채점 모델 기반 자동 피드백")

st.markdown("---")
st.write("**팀원** : ✨수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진✨")
st.markdown("---")

import streamlit as st
import time
with st.spinner(text='오늘 수업 즐겁게 들었나요? 이제 여러분들이 얼마나 공부를 열심히 했는지 알아보도록 해요!'):
    time.sleep(2)
    st.success('자, 시작해볼까요?')


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
output_d_1_1 = 1  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

model_1_1 = RNNModel(output_d_1_1, c)  # RNNModel 쓰는경우
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
label_1_1 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_1_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if label_1_1 == 1:
       st.success('거듭제곱의 곱셈을 이해하고 있구나!', icon="✅")
   else:
       st.info('거듭제곱의 곱셈을 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button1_1_1_2'):
    st.write('밑이 a로 같아요!')



st.markdown("---")
# 문항1-2

st.subheader("문항1-2")
st.markdown("$$ (x^{4})^{3} \\times (x^{2})^{5} = $$")

response = st.text_input('답안 :', key='answer_input_1_2')

# 모델의 이름 정하기
model_name_1_2 = "1-2_att_sp_60"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 60  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d_1_2 = 2  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_2 = ATTModel(output_d_1_2, c) #ATTModel 쓰는경우

model_1_2.load_state_dict(torch.load("./save/"+model_name_1_2+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_2 = AutoTokenizer.from_pretrained("./save/"+model_name_1_2)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_2(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_2(pad_ten)
label_1_2 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_2_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if len(label_1_2) >= 2:
       if label_1_2[0] == 1 and label_1_2[1] == 1 :
           st.success('거듭제곱의 거듭제곱, 거듭제곱의 곱셈을 이해하고 있구나!', icon="✅")
       #elif label_1_2[0] == 1 and label_1_2[1] == 0:
            #st.success('거듭제곱의 거듭제곱을 이해하고 있구나! 거듭제곱의 곱셈을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
       #elif label_1_2[0] == 0 and label_1_2[1] == 1:
            #st.success('거듭제곱의 곱셈을 이해하고 있구나! 거듭제곱의 거듭제곱을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")     
       else:
            st.info('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 복습하세요!', icon="⚠️")


if st.button('❓힌트 보기', key='button1_1_2_2'):
    st.write('괄호를 먼저 정리하세요!')



st.markdown("---")
# 문항1-3

st.subheader("문항1-3")
st.markdown("$$ b^{3} \\div b^{6} = $$")

response = st.text_input('답안 :', key='answer_input_1_3')

# 모델의 이름 정하기
model_name_1_3 = "1-3_att_sp_62"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 62  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d_1_3 = 1  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_3 = ATTModel(output_d_1_3, c) #ATTModel 쓰는경우

model_1_3.load_state_dict(torch.load("./save/"+model_name_1_3+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_3 = AutoTokenizer.from_pretrained("./save/"+model_name_1_3)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_3(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_3(pad_ten)
label_1_3 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_3_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if label_1_3 == 1:
       st.success('거듭제곱의 나눗셈3을 이해하고 있구나!', icon="✅")   
   else:
            st.info('거듭제곱의 나눗셈을 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button1_1_3_2'):
    st.write('밑이 b로 같아요!')


st.markdown("---")
# 문항1-4

st.subheader("문항1-4")
st.markdown("$$ a^{12} \\div a^{3} \\div a^{9} = $$")

response = st.text_input('답안 :', key='answer_input_1_4')

# 모델의 이름 정하기
model_name_1_4 = "1-4_att_sp_74"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 74  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d_1_4 = 2  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_4 = ATTModel(output_d_1_4, c) #ATTModel 쓰는경우

model_1_4.load_state_dict(torch.load("./save/"+model_name_1_4+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_4 = AutoTokenizer.from_pretrained("./save/"+model_name_1_4)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_4(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_4(pad_ten)
label_1_4 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_4_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if len(label_1_4) >= 2:
       if label_1_4[0] == 1 and label_1_4[1] == 1 :
           st.success('거듭제곱의 나눗셈1, 거듭제곱의 나눗셈2를 이해하고 있구나!', icon="✅")
       #elif label_1_2[0] == 1 and label_1_2[1] == 0:
            #st.success('거듭제곱의 나눗셈1을 이해하고 있구나! 거듭제곱의 나눗셈2를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
       #elif label_1_2[0] == 0 and label_1_2[1] == 1:
            #st.success('거듭제곱의 나눗셈2를 이해하고 있구나! 거듭제곱의 나눗셈1을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")     
       else:
            st.info('거듭제곱의 나눗셈1, 거듭제곱의 나눗셈2를 복습하세요!', icon="⚠️")


if st.button('❓힌트 보기', key='button1_1_4_2'):
    st.write('지수의 대소를 체크하세요!')


st.markdown("---")
# 문항1-5

st.subheader("문항1-5")
st.markdown("$$ (2a^{4})^{3} = $$")

response = st.text_input('답안 :', key='answer_input_1_5')

# 모델의 이름 정하기
model_name_1_5 = "1-5_att_sp_74"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 74  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d_1_5 = 2  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_5 = ATTModel(output_d_1_5, c) #ATTModel 쓰는경우

model_1_5.load_state_dict(torch.load("./save/"+model_name_1_5+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_5 = AutoTokenizer.from_pretrained("./save/"+model_name_1_5)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_5(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_5(pad_ten)
label_1_5 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_5_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if len(label_1_5) >= 2:
       if label_1_5[0] == 1 and label_1_5[1] == 1 :
           st.success('곱의 거듭제곱, 거듭제곱의 거듭제곱을 이해하고 있구나!', icon="✅")
       #elif label_1_5[0] == 1 and label_1_5[1] == 0:
            #st.success('곱의 거듭제곱을 이해하고 있구나! 거듭제곱의 거듭제곱을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
       #elif label_1_5[0] == 0 and label_1_5[1] == 1:
            #st.success('거듭제곱의 거듭제곱을 이해하고 있구나! 곱의 거듭제곱을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")     
       else:
            st.info('곱의 거듭제곱, 거듭제곱의 거듭제곱을 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button1_1_5_2'):
    st.write('괄호 안의 식을 세제곱하세요!')


st.markdown("---")
# 문항1-6

st.subheader("문항1-6")
st.markdown("$$ \\left( {b} \\over {3}\\right)^{4} = $$")

response = st.text_input('답안 :', key='answer_input_1_6')

# 모델의 이름 정하기
model_name_1_6 = "1-6_att_sp_74"  # 모델 이름 넣어주기 확장자는 넣지말기!
# 모델에 맞는 hyperparameter 설정
vs = 74  # vocab size
emb = 16  # default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32  # default 값 지정 안했으면 건드리지 않아도 됨
nh = 4  # default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu"  # default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
# output_d 설정
output_d_1_6 = 1  # 자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d_1_1, c)  # RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_6 = ATTModel(output_d_1_6, c) #ATTModel 쓰는경우

model_1_6.load_state_dict(torch.load("./save/"+model_name_1_6+".pt"))

# 자신에게 맞는 모델로 부르기
tokenizer_1_6 = AutoTokenizer.from_pretrained("./save/"+model_name_1_6)  # sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_6(response)["input_ids"]  # sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len:
    pad = (max_len - l) * [0] + enc
else:
    pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1, max_len)
y = model_1_6(pad_ten)
label_1_6 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_1_6_1'):
    
    # output차원에 맞추어 피드백 넣기
   st.write(response)
   if label_1_6 == 1:
       st.success('분수의 거듭제곱을 이해하고 있구나!', icon="✅")
   else:
       st.info('분수의 거듭제곱을 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button1_1_6_2'):
    st.write('분모와 분자 모두 네제곱하세요!')



st.markdown("---")
#문항1-7

st.subheader("문항1-7")
st.markdown("$$ (2^4)^x \\times (2^2)^x=2^3 \\times 2^{3x} $$일 때, 자연수 $$x$$의 값을 구하시오.")

response = st.text_input('답안 :', key='answer_input_1_7')

#모델의 이름 정하기
model_name_1_7 = "1-7_att_sp_140" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 140 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d_1_7 = 3 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_1_7 = ATTModel(output_d_1_7, c) #ATTModel 쓰는경우

model_1_7.load_state_dict(torch.load("./save/"+model_name_1_7+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer_1_7 = AutoTokenizer.from_pretrained("./save/"+model_name_1_7) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_7(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model_1_7(pad_ten)
label_1_7 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button1_7_1'):
    
    #output차원에 맞추어 피드백 넣기
    
    st.write(response)
    if len(label_1_7) >= 3:
        if label_1_7[0] == 1 and label_1_7[1] == 1 and label_1_7[2] == 1:
            st.success('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 이해하고 있구나!', icon="✅")   
        elif label_1_7[0] == 1 and label_1_7[1] == 1 and label_1_7[2] == 0:
            st.success('거듭제곱의 거듭제곱, 거듭제곱의 곱셈을 이해하고 있구나! 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        elif label_1_7[0] == 1 and label_1_7[1] == 0 and label_1_7[2] == 0:
            st.success('거듭제곱의 거듭제곱를 이해하고 있구나! 거듭제곱의 곱셈, 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        elif label_1_7[0] == 0 and label_1_7[1] == 1 and label_1_7[2] == 0:
            st.success('거듭제곱의 곱셈을 이해하고 있구나! 거듭제곱의 거듭제곱, 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        elif label_1_7[2] == 0 and label_1_7[2] == 0 and label_1_7[2] == 1:
            st.success('일차방정식 풀이를 이해하고 있구나! 거듭제곱의 거듭제곱, 거듭제곱의 곱셈을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        else:
            st.info('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 복습하세요!', icon="⚠️")

if st.button('❓힌트1️⃣', key='button1_7_2'):
    st.write('밑이 2로 같으니, 지수를 정리하세요!')

if st.button('❓힌트2️⃣', key='button1_7_3'):
    st.write('거듭제곱의 거듭제곱을 적용해서 식을 정리하세요!')

if st.button('❓힌트3️⃣', key='button1_7_4'):
    st.write('거듭제곱의 곱셈을 적용해서 식을 정리하세요!')

if st.button('💯모범답안', key='button1_7_5'):
    image_path = "save/1-7 모범답안.png-.png"
    st.image(image_path, caption='1-7모범답안')


st.markdown("---")
#문항1-8

st.subheader("문항1-8")
st.markdown("저장 매체의 용량을 나타내는 단위로 B, KB, MB 등이 있고, 1KB=$2^{10}$B, 1MB=$2^{10}$KB이다. 찬혁이가 컴퓨터로 용량이 36MB인 자료를 내려받으려고 한다. 이 컴퓨터에서 1초당 내려받는 자료의 용량이 $9 \\times 2^{20}$KB일 때, 찬혁이가 자료를 모두 내려받는 데 몇 초가 걸리는지 구하시오.")

response = st.text_input('답안 :', key='answer_input_1_8')

#모델의 이름 정하기
model_name_1_8 = "1-8_lstm_sp_140" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 140 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d_1_8 = 5 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

# model = RNNModel(output_d, c) #RNNModel 쓰는경우
model_1_8 = LSTMModel(output_d_1_8, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model_1_8.load_state_dict(torch.load("./save/"+model_name_1_8+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer_1_8 = AutoTokenizer.from_pretrained("./save/"+model_name_1_8) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name_1_8+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_8(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model_1_8(pad_ten)
label_1_8 = y.squeeze().detach().cpu().numpy().round()

if st.button('👀피드백 받기', key='button_1_8_1'):
    #output차원에 맞추어 피드백 넣기
    st.write(response)
    if len(label_1_8) >= 5:
        if label_1_8[0] == 1 and label_1_8[1] == 1 and label_1_8[2] == 1 and label_1_8[3] == 0 and label_1_8[4] == 0:
            st.success('거듭제곱의 곱셈, 거듭제곱의 나눗셈, 단위 변환을 이해하고 있구나!', icon="✅")
        elif label_1_8[0] == 0 and label_1_8[1] == 0 and label_1_8[2] == 0 and label_1_8[3] == 1 and label_1_8[4] == 1:
            st.success('거듭제곱의 나눗셈, 수의 나눗셈을 이해하고 있구나!', icon="✅")
        #elif label[0] == 1 and label[1] == 0 and label[2] == 1 and label[3] == 0 and label[4] == 0:
        #    st.success('거듭제곱의 곱셈, 단위 변환을 이해하고 있구나! 거듭제곱의 나눗셈을 올바르게 적용해서 풀어보세요!', icon="ℹ️")
        #elif label[0] == 0 and label[1] == 0 and label[2] == 1 and label[3] == 0 and label[4] == 0:
        #    st.success('단위 변환을 이해하고 있구나! 거듭제곱의 곱셈, 거듭제곱의 나눗셈, 수의 나눗셈을 올바르게 적용해서 풀어보세요!', icon="ℹ️")       
        #elif label[0] == 0 and label[1] == 1 and label[2] == 0 and label[3] == 0 and label[4] == 0:
        #    st.success('거듭제곱의 거듭제곱을 이해하고 있구나! 단위 변환, 수의 나눗셈을 올바르게 적용해서 풀어보세요!', icon="ℹ️")
        #elif label[0] == 0 and label[1] == 0 and label[2] == 0 and label[3] == 1 and label[4] == 0:
        #    st.success('거듭제곱의 나눗셈을 이해하고 있구나! 거듭제곱의 곱셈, 단위 변환을 올바르게 적용해서 풀어보세요!', icon="ℹ️")
        #elif label[0] == 1 and label[1] == 0 and label[2] == 0 and label[3] == 0 and label[4] == 0:
        #    st.success('거듭제곱의 곱셈을 이해하고 있구나! 거듭제곱의 나눗셈, 단위 변환을 올바르게 적용해서 풀어보세요!', icon="ℹ️")
       
        else:
            st.info('거듭제곱의 곱셈, 거듭제곱의 나눗셈, 단위 변환, 수의 나눗셈을 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button_1_8_2'):
    st.write('단위 변환을 해보세요!')

#st.markdown("---")
#file_path = ""  
#st.download_button('🌻복습 문항 다운받기🌻', file_path)
#st.markdown("---")



st.markdown("---")
####조별 과제 부분 체크####
#st.set_page_config(layout="wide")
#st.title("대표 문항 설계")
#st.divider()

import pandas as pd
###1-7
st.header("1-7")
st.write("📖식을 정리하기 위해 거듭제곱의 거듭제곱과 거듭제곱의 곱셈을 적절히 사용할 수 있는가?")
st.write("📖지수의 미지수를 구하기 위해 일차방정식의 풀이를 적절히 사용할 수 있는가?")
st.write("👉**문제** : $ (2^4)^x \\times (2^3)^x = 2^3 \\times 2^{3x} $")
st.write("👉**지식요소** : 거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식의 풀이")

st.write("👉**모범답안1**")

image_url = 'save/사진자료/1-7 모범답안1.png'
st.image(image_url)

st.write("👉**모범답안2**")

image_url = 'save/사진자료/1-7 모범답안2.png'
st.image(image_url)


st.write("👉**모범답안3**")

image_url = 'save/사진자료/1-7 모범답안3.png'
st.image(image_url)


st.write("👉**지식맵**")
image_url = 'save/사진자료/1-7 지식맵.png'
st.image(image_url)

st.write("👉**오개념**")
st.markdown('<span style="color: blue;">1. 등호오류</span>', unsafe_allow_html=True)
st.write("⚠️1-1. 거듭제곱과 지수의 계산을 혼동하는 것으로 보임")
st.write("예 : $(2^{4})^{x} \\times ( 2^{2})^{x} = 2^{3} \\times 2^{ 3x } = x = 1$")
st.write("예 : $2^{ 4 \\times x } \\times 2^{ 2 \\times x } = 2^{3} \\times 2^{ 3x } = 2^{ 4x + 2x } = 2^{ 3x + 3 } = 6x = 3x + 3$ ")

st.write("⚠️1-2. 등호를 계산 진행과정 사이에 사용함")
st.write("예 : $(2^{4})^{x} \\times (2^{2})^{x} = 2^{3} \\times 2^{ 3x } = 2^{ 4x } \\times 2^{ 2x } = 2^{3} \\times 2^{ 3x } = 2^{ 3x } = 2^{3}$")

st.write("⚠️1-3. 논리적 오류가 없는 부분도 있지만 등호를 계산 진행 과정 사이에 사용함")
st.write("예 : $(2^{4})^{x} \\times (2^{2})^{x} = 2^{3} \\times (2^{3})^{x} = 2^{ 4x } \\times 2^{ 2x } = 2^{3} \\times 2^{ 3x } = 2^{ 6x } \\div 2^{ 3x } = 2^3 = 2^{ 3x } = 2^{3} $")

st.markdown('<span style="color: blue;">2. 식 오류</span>', unsafe_allow_html=True)
st.write("⚠️2-1. 곱셈 기호를 덧셈 기호로 혼동하여 작성함")
st.write("예 : $2^{ 4x } \\times 2^{ 2x } = 2^{ 6x }$, $2^{ 6x } = 2^3 + 2^{ 3x }$")
st.write("⚠️2-2. 중간에 옮겨적는 과정에서 수나 문자를 잘못 적음")
st.write("예 : $( 2^{4})^{x} \\times ( 2^{2})^{x} = 2^{3} \\times 2^{ 3x }$, $2^{ 4x } \\times 2^{ 2x } = 2^3 \\times 2^{x}$")
st.write("예 : $( 2^{4})^{x} \\times ( 2^{2})^{x} = 2^{ 4x } \\times 2^{ 2x } = 2^{ 6x } = 2^{ 3x } \\times 2^{ 3x }$")
st.write("예 : $x^{ 4x } \\times 2^{ 2x } = 2^{ 3 + 3x }$")
st.write("⚠️2-3. 지수에 있는 미지수를 빼고 계산함")
st.write("예 : $2^{4} \\times 2^{2} = 2^{3} \\times 2^{3} = 16 \\times 4 = 8 \\times 8$, $x = 1$")

st.markdown('<span style="color: blue;">3. 이항오개념</span>', unsafe_allow_html=True)
st.write("⚠️3-1. 등식의 성질을 이용하여 양변에 2^{3x}를 나눈 것인데 이항이라는 용어로 표현함")
st.write("예 : ${ 2^{6}}^{x} = 2^{3} \\times 2^{ 3x }$ 이항 $2^{ 3x } = 2^{3}$")

st.markdown('<span style="color: blue;">4. 대입으로 해결</span>', unsafe_allow_html=True)
st.write("⚠️4-1. 오류는 아니지만 이 문제에서 평가하고자 하는 요소가 아닌 대입으로 해결함")
st.write("예 : $x$에 1 대입  $2^{ 4 \\times 1 } \\times 2^{ 2 \\times 1}$, $2^{6}=2^{3}\\times2^{3\\times 1 }$, $2^{4} \\times 2^{2} = 2^{6}$, $2^{6} = 2^{6}$, 1")
st.write("예 : $( 2^{4})^{1} \\times ( 2^{2})^{1} = 2^{3} \\times 2^{3}$, $x = 1$")
st.write("예 : $x = 1$, $2^{3} \\times 2^{3} = 2^{4} \\times 2^{2}$")

###1-8
st.markdown("---")

st.header("1-8")
st.write("👉**문제** : 저장 매체의 용량을 나타내는 단위로 B, KB, MB 등이 있고, 1KB=$2^{10}$B, 1MB=$2^{10}$KB이다. 찬혁이가 컴퓨터로 용량이 36MB인 자료를 내려받으려고 한다. 이 컴퓨터에서 1초당 내려받는 자료의 용량이 $9 \\times{2^{20}}$B일 때, 찬혁이가 자료를 모두 내려받는 데 몇 초가 걸리는지 구하시오.")
st.write("👉**지식요소** : 풀이 방법에 따라 지식 요소 종류 및 순서가 다소 다름")

with st.container():
    st.write("풀이방법1️⃣ 거듭제곱의 곱셈, 거듭제곱의 나눗셈2, 단위의 이해")
    st.write("풀이방법2️⃣ 거듭제곱의 나눗셈1, 수의 나눗셈, 단위의 이해")
    st.write("풀이방법3️⃣ 단위의 이해, 거듭제곱의 나눗셈, 거듭제곱의 나눗셈2")


st.write("👉**모범답안1**")

image_url = 'save/사진자료/1-8 모범답안1.png'
st.image(image_url)

st.write("👉**모범답안2**")

image_url = 'save/사진자료/1-8 모범답안2.png'
st.image(image_url)

st.write("👉**모범답안3**")

image_url = 'save/사진자료/1-8 모범답안3.png'
st.image(image_url)

st.write("✔️문제 풀이 방향이 크게 3가지 종류로 나누어질 수 있음")
st.write("✔️실제로 학생들은 모범답안 1, 2의 방향으로의 풀이가 있고, 모범답안 3의 풀이는 없었음")
st.write("✔️풀이 방법에 따라 평가할 인지 요소의 종류 및 순서가 달라지는데, 이를 하나의 모델에 적용하는 것이 쉽지 않았던 것 같음")

st.write("👉**지식맵**")
st.write("1-8 지식맵 파일 넣기")

st.write("👉**오개념**")
st.markdown('<span style="color: blue;">1. 등호오류: 서로 다른 식들을 등호로 계속 연결하는 오류</span>', unsafe_allow_html=True)
st.write("예: $2^{10} \\times 2^{10} = {{ 2^{20} \\times 36 } \\over { 2^{20} \\times 9 }}$ = 4초")

st.markdown('<span style="color: blue;">2. 식오류: 논리적으로 맞지 않는 식을 전개하였음. 특정한 부분의 오개념이라기보단 전체적인 식 전개에 오류가 있다고 판단됨</span>', unsafe_allow_html=True)
st.write("예 : $1KB = 2^{10}B , 1MB = 2^{10}KB$ 용량 $36MB$ $1$초당 $9 \\times 2^{20}B$이기에 $(2^{10}KB)^{36} = 2^{360}(K^{36})(B^{36})$, $2^{360}(K^{36})(B^{36}) \\div 9 \\times 2^{20}B = 9 \\times 2^{18}(K^{36})(B^{17})$초")

st.markdown('<span style="color: blue;">3. 단위 혼동: B, KB, MB를 통일하지 않고 혼용해서 사용하는 오류</span>', unsafe_allow_html=True)
st.write("예 : $36MB = 36 \\times 2^{10}KB, 9 \\times 2^{20}KB \\times 4 = 36MB$, $4$초")

st.markdown('<span style="color: blue;">4. 나눗셈 괄호 오류: 나눗셈을 할 때, 분자,분모에 해당하는 식 전체에 괄호를 하지 않는 오류</span>', unsafe_allow_html=True)
st.write("예 : $36MB = 36 \\times 2^{10}KB$,  $36 \\times 2^{10}KB = 36 \\times 2^{10} \\times 2^{10}$B $36 \\times 2^{10} \\times 2^{10}B = 2^{22} 3^{2}B$ , $2^{22} 3^{2}B \\div 9 \\times 2^{20}B = 2^{2}$, $2^{2} = 4$")

###2-6
st.markdown("---")

st.header("2-6")
st.write("📖목표가 되는 식을 구하기 위해 등식의 성질을 적절히 사용할 수 있는가?")
st.write("📖단항식의 곱셈과 나눗셈을 할 수 있는가?")

st.write("👉**문제** : $( - 12x^{3}y^{2} ) \\div \\square \\times 18x^{3}y^{3} = 8x^{2}y^{3}$일 때 $\\square$ 안에 알맞은 식을 구하시오.")
st.write("👉**지식요소** : 등식의 성질, 단항식의 곱셈, 단항식의 나눗셈, 거듭제곱의 곱셈, 거듭제곱의 나눗셈")

st.write("👉**모범답안1**")

image_url = 'save/사진자료/2-6 모범답안1.png'
st.image(image_url)


st.write("👉**모범답안2**")

image_url = 'save/사진자료/2-6 모범답안2.png'
st.image(image_url)


st.write("👉**모범답안3**")

image_url = 'save/사진자료/2-6 모범답안3.png'
st.image(image_url)

st.write("👉**모범답안4**")

image_url = 'save/사진자료/2-6 모범답안4.png'
st.image(image_url)

st.write("👉**지식맵**")
image_url = 'save/사진자료/2-6 지식맵.png'
st.image(image_url)

st.write("👉**오개념**")
st.markdown('<span style="color: blue;">1. 역수를 구하는 유형: 단항식의 곱셈과 나눗셈을 할 수 있지만 역수를 구함</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-6 오류유형1.png'
st.image(image_url)

st.markdown('<span style="color: blue;">2. 부호 오류: 부호를 잘못 구한 경우</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-6 오류유형2.png'
st.image(image_url)

st.markdown('<span style="color: blue;">3. 식을 잘못 본 경우: 18x^{3}y^{3}을 8x^{3}y^{3}로 잘못보고 계산함</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-6 오류유형3.png'
st.image(image_url)

st.markdown('<span style="color: blue;">4. 식의 계산을 할 수 있지만 등식의 성질을 이해하지 못한 경우: 식의 계산에는 오류가 없지만 등식의 성질을 이해하지 못하여 4/9x의 역수를 곱하지 않음</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-6 오류유형4.png'
st.image(image_url)

###2-7
st.markdown("---")

st.header("2-7")
st.write("👉**문제** :  높이가 $(2x)^{2}$인 삼각형의 넓이가 $48x^{3}y^{2}$일 때, 이 삼각형의 밑변의 길이를 구하시오.")
st.write("👉**지식요소** : 곱의 거듭제곱, 거듭제곱의 나눗셈, 다항식의 나눗셈, 삼각형의 넓이")
st.write("👉**피드백 요소** : 미지수의 의미를 명시함, 12xy2(삼각형의 넓이공식에서 실수)")

st.write("👉**모범답안1**")
st.write("밑변을 미지수로 놓고 삼각형의 넓이에 대한 식을 세워 계산")
image_url = 'save/사진자료/2-7 모범답안1.png'
st.image(image_url)


st.write("👉**모범답안2**")
st.write("밑변에 대한 식을 세운 후 계산함")
image_url = 'save/사진자료/2-7 모범답안2.png'
st.image(image_url)

st.write("👉**지식맵**")
image_url = 'save/사진자료/2-7 지식맵.png'
st.image(image_url)

st.write("👉**오개념 및 오류**")
st.markdown('<span style="color: blue;">1. 삼각형의 넓이 공식 오류: 삼각형의 넓이 공식에서 2를 나누어야하는데 이를 생략하거나 2를 곱하는 등의 오류를 보임</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-7 오개념 및 오류1.png'
st.image(image_url)

st.markdown('<span style="color: blue;">2. 계산 실수: 단순 계산 실수를 한 것으로 보임</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-7 오개념 및 오류2.png'
st.image(image_url)


st.markdown('<span style="color: blue;">3. 문제 파악 오류: 문제 파악을 제대로 하지 못함</span>', unsafe_allow_html=True)
image_url = 'save/사진자료/2-7 오개념 및 오류3.png'
st.image(image_url)



###3-3
st.markdown("---")

st.header("3-3")
st.write("👉**문제** : $A \\div 3y/2 = 4x^{2}y + 2xy + 6$일 때. 다항식 A를 구하시오.")
st.write("👉**지식요소** : 등식의 성질, 다항식과 단항식의 곱셈, 단항식의 곱셈, 다항식과 단항식의 나눗셈, 단항식의 나눗셈")

st.write("👉**모범답안1**")
st.write("양변에 3y/2를 곱하여 곱셈으로 풀이")
image_url = 'save/사진자료/3-3 모범답안1.png'
st.image(image_url)

st.write("👉**모범답안2**")
st.write("나눗셈을 고치지 않고 다항식과 단항식의 나눗셈으로 풀이")
image_url = 'save/사진자료/3-3 모범답안2.png'
st.image(image_url)

st.write("👉**모범답안3**")
st.write("나눗셈을 역수의 곱셈으로 바꾸어 풀이")
image_url = 'save/사진자료/3-3 모범답안3.png'
st.image(image_url)


st.write("👉**지식맵**")
st.write("3-3 지식맵 넣기")

st.write("👉**오개념**")
st.markdown('<span style="color: blue;">1. 등호 사용의 오류</span>', unsafe_allow_html=True)
st.write("예 : $A \\div 3y/2 = 4x^{2}y + 2xy + 6= (4x^{2}y + 2xy + 6) \\times 3y/2$")
st.write("예 : $A \\div 3y/2 = 4x^2y + 2xy + 6 =  6x^{2}y^{2} + 3xy^{2} + 9y$")