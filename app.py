from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

st.title("자동 채점 모델 기반 자동 피드백")

st.markdown("---")
st.write("**팀원** : ✨수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진✨")
st.markdown("---")

import streamlit as st
import time
with st.spinner(text='오늘 수업 즐겁게 들었나요? 이제 여러분들이 얼마나 공부를 열심히 했는지 알아보도록 해요!'):
    time.sleep(10)
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
        #elif label_1_7[0] == 1 and label_1_7[1] == 1 and label_1_7[2] == 0:
            #st.success('거듭제곱의 거듭제곱, 거듭제곱의 곱셈을 이해하고 있구나! 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        #elif label_1_7[0] == 1 and label_1_7[1] == 0 and label_1_7[2] == 0:
            #st.success('거듭제곱의 거듭제곱를 이해하고 있구나! 거듭제곱의 곱셈, 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        #elif label_1_7[0] == 0 and label_1_7[1] == 1 and label_1_7[2] == 0:
            #st.success('거듭제곱의 곱셈을 이해하고 있구나! 거듭제곱의 거듭제곱, 일차방정식 풀이를 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        #elif label_1_7[2] == 0 and label_1_7[2] == 0 and label_1_7[2] == 1:
            #st.success('일차방정식 풀이를 이해하고 있구나! 거듭제곱의 거듭제곱, 거듭제곱의 곱셈을 올바르게 적용해서 풀이를 완성해보자!', icon="ℹ️")
        else:
            st.info('거듭제곱의 거듭제곱, 거듭제곱의 곱셈, 일차방정식 풀이를 복습하세요!', icon="⚠️")

if st.button('❓힌트 보기', key='button1_7_2'):
    st.write('밑이 2로 같으니, 지수를 정리하세요!')


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
