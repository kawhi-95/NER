from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus, build_infer_corpus

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

# 读取数据
print("读取数据...")
_, _, word2id, tag2id = build_corpus("train")

sentences = []
sentence = input("请输入要判别的语句：")
sentence = list(sentence)
sentences.append(sentence)

print("加载并评估hmm模型...")
hmm_model = load_model(HMM_MODEL_PATH)
hmm_pre_tag = hmm_model.test(sentences, word2id, tag2id)
# for i in range(len(hmm_pre_tag[0])):
#     print(sentences[0][i],hmm_pre_tag[0][i])

print("加载并评估crf模型...")
crf_model = load_model(CRF_MODEL_PATH)
crf_pre_tag = crf_model.test(sentences)
# for i in range(len(crf_pre_tag[0])):
#     print(sentences[0][i],crf_pre_tag[0][i])

print("加载并评估bilstm模型...")
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
bilstm_model = load_model(BiLSTM_MODEL_PATH)
bilstm_pre_tag = bilstm_model.test(sentences, sentences, word2id, tag2id)
for i in range(len(bilstm_pre_tag)-1):
    print(sentences[i])
    print(bilstm_pre_tag[i])


print("加载并评估bilstm+crf模型...")
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
bilstmCRF_model = load_model(BiLSTMCRF_MODEL_PATH)
bilstmCRF_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(sentences, sentences, test=True)
bilstmCRF_pre_tag = bilstmCRF_model.test(sentences, sentences, word2id, tag2id)
# for i in range(len(bilstmCRF_pre_tag)-1):
#     print(sentences[i])
#     print(bilstmCRF_pre_tag[i])