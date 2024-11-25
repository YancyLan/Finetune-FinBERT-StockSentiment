import pandas as pd
import numpy as np
#导入Transformer库里的BertTokenizer训练器，并加载一个预训练的分词器，路径为"./finbert"。这个分词器用于对输入文本数据进行预处理
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("./finbert")
#加载训练集和测试集文件
data = pd.read_csv("part2.csv")
data1 = pd.read_csv("part1.csv")

#使用dataset加载csv的数据保存在数据集变量imdb和ds02中
from datasets import load_dataset
imdb = load_dataset('csv', data_files=r'part1.csv')
ds02 = load_dataset('csv', data_files=r'part2.csv')

# datafile中包含两个键值对,用于划分训练集以及测试集；键"train"对应着训练集数据的文件路径，键"test"对应着测试集数据的文件路径
data_files = {"train": "./part1.csv", "test": "part2.csv"}
#创建imdb数据集，其中包含来自"./part1.csv"和"part2.csv"两个CSV文件的数据，并且根据刚才的设置，分别保存为训练集和测试集
imdb = load_dataset("csv", data_files=data_files)

#定义了preprocess_function函数，使用之前加载的分词器对文本数据进行分词预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=510)
#使用map函数将预处理函数应用到整个imdb数据集上，并将分词后的数据保存到tokenized_imdb中。
tokenized_imdb = imdb.map(preprocess_function, batched=True)

#使用Transformers库中的DataCollatorWithPadding类创建一个数据整合器（Data Collator）。将输入数据整合成适合训练的批次，并进行填充（padding）操作，确保输入的序列长度一致。
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)#参数tokenizer=tokenizer，将之前加载的BertTokenizer分词器传递给数据整合器，以便在整合数据时使用相同的分词器进行分词和填充
#导入了evaluate模块加载"accuracy"的评估指标
import evaluate
accuracy = evaluate.load("accuracy")

#定义一个评估函数，compute_metrics函数接受一个元组eval_pred作为输入，输出为模型的性能评估
def compute_metrics(eval_pred):
    predictions, labels = eval_pred 
    predictions = np.argmax(predictions, axis=1)#使用numpy的argmax函数，沿着轴1（即第二个维度，表示每个样本的不同类别预测概率）找到每个样本的最大预测概率对应的类别。这样，predictions就变成了一维数组，包含了每个样本的预测类别。
    return accuracy.compute(predictions=predictions, references=labels)

#id2label是一个字典，将整数标签映射到对应的文本标签。label2id是将文本标签映射回整数标签的字典。这两个字典的作用是将模型预测的整数标签映射回文本标签，或者将真实的文本标签映射成整数标签
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

#使用Hugging Face Transformers库的BertForSequenceClassification类来配置BERT模型用于序列分类任务。from_pretrained方法加载预训练的BERT模型，并使用提供的参数配置模型。
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
model = BertForSequenceClassification.from_pretrained(
    "./finbert", num_labels=2, id2label=id2label, label2id=label2id
)#num_labels=2：这个参数指定了模型的标签类别数。在这个例子中，我们的序列分类任务有两个类别，即"NEGATIVE"和"POSITIVE"，所以这里num_labels设置为2。

training_args = TrainingArguments(
    output_dir="my_awesome_model",#参数指定了模型训练过程中保存模型和结果的输出目录
    learning_rate=2e-5,#学习率（Learning Rate）的设置，控制模型在每次优化步骤中更新权重的幅度
    per_device_train_batch_size=5,#指定了每个（GPU或CPU）上的训练批次大小
    per_device_eval_batch_size=5,#指定了每个设备上的评估（验证或测试）批次大小
    num_train_epochs=2,#训练的轮数
    weight_decay=0.01,#设置了权重衰减（Weight Decay）的大小。权重衰减是一种正则化方法，用于控制模型的复杂度，以避免过拟合。
    evaluation_strategy="epoch",#参数指定了评估策略。在每个轮数（epoch）结束时，会对验证集进行一次评估。
    save_strategy="epoch",#指定了保存策略。在每个轮数结束时，会保存一个模型快照。
    load_best_model_at_end=True,#表示训练结束时加载最佳模型的权重。如果设置为True，训练结束后将加载在验证集上性能最好的模型权重。
)

trainer = Trainer(
    model=model,#参数指定了要训练的BERT模型，即之前定义的model对象
    args=training_args,
    train_dataset=tokenized_imdb["train"],#参数指定了训练数据集，即之前进行过分词处理的tokenized_imdb["train"]
    eval_dataset=tokenized_imdb["test"],#参数指定了验证（测试）数据集，即之前进行过分词处理的tokenized_imdb["test"]
    tokenizer=tokenizer,#传递了之前定义的tokenizer对象
    data_collator=data_collator,#传递了之前定义的data_collator对象，它将用于整合和填充训练数据。
    compute_metrics=compute_metrics,#传递了之前定义的compute_metrics函数，它用于计算模型的性能指标。
)
trainer.train()
eval_results = trainer.evaluate(eval_dataset)