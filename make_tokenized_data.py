from transformers import BertTokenizer
from tqdm import tqdm
from dataset import ProcessedIdDataset
import argparse
import torch
import os

def tokenized_and_process(data, mode='train'):
  _tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

  text_list=[]
  text_id_list=[]

  aspect_question_list=[]
  aspect_question_id_list=[]
  opinion_answer_list=[]

  opinion_question_list=[]
  opinion_question_id_list=[]
  aspect_answer_list=[]

  sentiment_list=[]
  header_fmt='Tokenize data {:>5s}'
  for sample in tqdm(data,desc=f'{header_fmt.format(mode.upper())}'):

    ##Temp data
    ###Text
    temp_text=sample.text_tokens
    text_list.append(temp_text)
    ##Aspect question
    aspect_question=sample.aspect_queries
    aspect_question_list.append(aspect_question)
    ##Opinion question
    opinion_question=sample.opinion_queries
    opinion_question_list.append(opinion_question)

    ###Convert tokens to ids

    ##The text
    text_ids=_tokenizer.convert_tokens_to_ids(
        [word.lower() for word in temp_text]
    )
    text_id_list.append(text_ids)

    ##Aspect question
    aspect_queries_ids=_tokenizer.convert_tokens_to_ids(
        [word.lower() for word in aspect_question]
    )
    aspect_question_id_list.append(aspect_queries_ids)

    ##Opinion question
    opinion_queries_ids=_tokenizer.convert_tokens_to_ids(
        [word.lower() for word in opinion_question]
    )
    opinion_question_id_list.append(opinion_queries_ids)

    assert len(text_ids)==len(sample.aspect_answers)==len(sample.opinion_answers)

    #Apsect answer
    aspect_answer_list.append(sample.aspect_answers)

    #Opinion answer
    opinion_answer_list.append(sample.opinion_answers)

    #Sentiment
    sentiment_list.append(sample.sentiments)

  result={
      'texts':text_list,
      'texts_ids':text_id_list,
      'aspect_questions':aspect_question_list,
      'aspect_questions_ids':aspect_question_id_list,
      'opinion_answers':opinion_answer_list,
      'opinion_questions':opinion_question_list,
      'opinion_questions_ids':opinion_question_id_list,
      'aspect_answers':aspect_answer_list,
      'sentiments':sentiment_list
  }

  final_data=ProcessedIdDataset(result)
  return final_data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Processing data')
    ##Define path where save unprocessed data and where to save processed data
    parser.add_argument('--data_path', type=str, default="./data/14lapV2/preprocess")
    parser.add_argument('--output_path', type=str, default="./data/14lapV2/preprocess")
    
    args=parser.parse_args()

    train_data_path = f"{args.data_path}/train_PREPROCESSED.pt"
    dev_data_path = f"{args.data_path}/dev_PREPROCESSED.pt"
    test_data_path = f"{args.data_path}/test_PREPROCESSED.pt"

    train_data=torch.load(train_data_path)
    dev_data=torch.load(dev_data_path)
    test_data=torch.load(test_data_path)

    '''##Making tokenize data before preprocess to id
    train_tokenized,train_max_len=tokenize_data(train_data,version=args_version,mode='train')
    dev_tokenized,dev_max_len=tokenize_data(dev_data,version=args_version,mode='dev')
    test_tokenized,test_max_len=tokenize_data(test_data,version=args_version,mode='test')'''

    '''print(f"train_max_len : {train_max_len}")
    print(f"dev_max_len : {dev_max_len}")
    print(f"test_max_len : {test_max_len}")'''

    ##Processing tokenied data to ids
    train_preprocess = tokenized_and_process(train_data,mode='train')
    dev_preprocess = tokenized_and_process(dev_data, mode='dev')
    test_preprocess = tokenized_and_process(test_data, mode='test')

    ##Saving preprocessing full data
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path=f'{args.output_path}/data.pt'
    print(f"Saved data : `{output_path}`.")
    torch.save({
        'train':train_preprocess,
        'dev':dev_preprocess,
        'test':test_preprocess
    },output_path)