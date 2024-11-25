import time
import akshare as ak
# from snownlp import SnowNLP

#爬取实时股票的新闻从离现在最近时刻开始
stock_code = '000010'
date = time.strftime("%Y%m%d", time.localtime())
stock_news_em_df = ak.stock_news_em(symbol=stock_code)
count1=0;
count2=0;

for i in range(28):
    import torch
    text=stock_news_em_df['新闻内容'][i]#爬取28条对应股票的新闻内容
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./my_awesome_model/checkpoint-7392")
    inputs = tokenizer(text, return_tensors="pt")#使用AutoTokenizer加载之前训练好的分词器，用于将文本数据转换成模型可以处理的张量（tensor）形式。
    
    #用之前加载好的情感分类模型对处理后的文本张量进行预测
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("./my_awesome_model/checkpoint-7392")
    with torch.no_grad():
        logits = model(**inputs).logits #使用这个模型对处理后的文本张量进行预测，得到了模型的预测输出logits

    predicted_class_id = logits.argmax().item()#预测输出中概率最大的类别标签，将其保存在predicted_class_id中，即输出为0或者1。
    print(model.config.id2label[predicted_class_id],end=' ')#使用模型参数中的id2label，即0对应negative，1对应positive
    print(stock_news_em_df['新闻内容'][i])#将模型预测的情感分类结果和对应的新闻内容打印输出。
    if model.config.id2label[predicted_class_id]=="POSITIVE":
        count1=count1+1#计算positive的个数
    else:
        count2=count2+1#计算negative个数
print("POSITIVE："+str(count1))
print("NEGATIVE："+str(count2))#输出positive、negative的个数


