import torch.nn as nn
from transformers import BertTokenizer,BertModel
from utils import calculate_A_O_loss
from dataset_support import generating_next_query
import torch
import torch.nn.functional as F

##This cell contain class to create model
class BertFFNN(nn.Module):
  '''
    Module question answering, bao gồm:
      + Mọt lớp pre-trained model (trong notebook này là bert-large hay bert-based )
      + Một lớp Linear với hai mode:
        - Một cái dành cho trả lời câu hỏi để tìm aspect
        - Cái còn lại dành cho trả lời câu hỏi để tìm opinion
  '''
  def __init__(self,args):
    hidden_size=args.hidden_size
    super(BertFFNN,self).__init__()
    self._bert = BertModel.from_pretrained(args.model_type)
    self._tokenizer = BertTokenizer.from_pretrained(args.model_type)
    print(f"Loaded `{args.model_type}` model !")
    
    self.asp_ffnn=nn.Linear(hidden_size,3)
    self.opi_ffnn=nn.Linear(hidden_size,3)
  def forward(self,input_ids=[], attention_mask=[], token_type_ids=[],answer='aspect'):
    hidden_states = self._bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[0]

    ##Logits calculation
    if answer=='aspect':
      logits=self.asp_ffnn(hidden_states)
    elif answer=='opinion':
      logits=self.opi_ffnn(hidden_states)

    return hidden_states,logits

class RoleFlippedModule(nn.Module):
  def __init__(self,args):
    '''
      Module để thay đổi giữa query aspect và query opinion theo args.T vòng, vòng số 0 luôn là initial
    '''
    super(RoleFlippedModule,self).__init__()
    self._model=BertFFNN(args)  ##Thành phần cho chiều A2O
    self._model2=BertFFNN(args) ##Thành phần cho chiều O2A

    if args.ifgpu==True:
      self._model.cuda()
      self._model2.cuda()
    self.args=args
  
  def forward(self,batch_dict,model_mode='train'):
    '''
      Các biến cur_answer sẽ lật qua lật lại để role flipped
        + Todo: Tương lai thiết kế có thể ngắt 1 trong hai phần A2O hoặc O2A
    '''
    lossA=0
    lossO=0
    #A2O
    ##Initial
    A2O_aspect_hidden_states,aspect_logits=self._model(batch_dict['initial_input_ids'],batch_dict['initial_attention_mask'],batch_dict['initial_token_type_ids'],answer='aspect')
    ##Nếu model đang trong quá trình train mới tính loss
    if model_mode=='train':
      lossA+=self.args.lambda_aspect*calculate_A_O_loss(batch_dict['initial_aspect_answers'],aspect_logits,ifgpu=self.args.ifgpu)
    input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,batch_dict['initial_input_ids'],self.args,query_type='aspect',model_mode=model_mode)
    ##Multihop turn
    cur_answer='opinion'
    for i in range(self.args.T):
      if cur_answer=='opinion':
        queries_for_opinion=input_ids
        A2O_opinion_hidden_states,opinion_logits=self._model(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        if model_mode=='train': ##Tương tự chỉ tính loss cho model khi ở mode train (*)
          lossO+=self.args.lambda_opinion*calculate_A_O_loss(answers,opinion_logits,ifgpu=self.args.ifgpu)
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,queries_for_opinion,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='aspect'
      elif cur_answer=='aspect':
        queries_for_aspect=input_ids
        A2O_aspect_hidden_states,aspect_logits=self._model(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        if model_mode=='train':##(*)
          lossA+=self.args.lambda_aspect*calculate_A_O_loss(answers,aspect_logits,ifgpu=self.args.ifgpu)
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,queries_for_aspect,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='opinion'
    A2O_aspect_hidden_states,A2O_aspects_list,A2O_opinion_hidden_states,A2O_opinions_list=self.processOutput(A2O_aspect_hidden_states,A2O_opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict,model_mode=model_mode)
    
    aspect_logits=[]
    opinion_logits=[]
    aspect_answers=0
    opinion_answers=0
    #O2A
    ##Initial
    O2A_opinion_hidden_states,opinion_logits=self._model2(batch_dict['initial_input_ids'],batch_dict['initial_attention_mask'],batch_dict['initial_token_type_ids'],answer='opinion')
    if model_mode=='train':##(*)
      lossO+=self.args.lambda_opinion*calculate_A_O_loss(batch_dict['initial_opinion_answers'],opinion_logits,ifgpu=self.args.ifgpu)
    input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,batch_dict['initial_input_ids'],self.args,query_type='opinion',model_mode=model_mode)
    ##Multihop turn
    cur_answer='aspect'
    for i in range(self.args.T):
      if cur_answer=='aspect':
        queries_for_aspect=input_ids
        O2A_aspect_hidden_states,aspect_logits=self._model2(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        if model_mode=='train': ##(*)
          lossA+=self.args.lambda_aspect*calculate_A_O_loss(answers,aspect_logits,ifgpu=self.args.ifgpu)
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,queries_for_aspect,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='opinion'
      elif cur_answer=='opinion':
        queries_for_opinion=input_ids
        O2A_opinion_hidden_states,opinion_logits=self._model2(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        if model_mode=='train':##(*)
          lossO+=self.args.lambda_opinion*calculate_A_O_loss(answers,opinion_logits,ifgpu=self.args.ifgpu)
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,queries_for_opinion,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='aspect'
    O2A_aspect_hidden_states,O2A_aspects_list,O2A_opinion_hidden_states,O2A_opinions_list=self.processOutput(O2A_aspect_hidden_states,O2A_opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict,model_mode=model_mode)
    
    result={
        'A2O_aspect_hidden_states':A2O_aspect_hidden_states,
        'A2O_opinion_hidden_states':A2O_opinion_hidden_states,
        'A2O_aspects_list':A2O_aspects_list,
        'A2O_opinions_list':A2O_opinions_list,
        'O2A_aspect_hidden_states':O2A_aspect_hidden_states,
        'O2A_opinion_hidden_states':O2A_opinion_hidden_states,
        'O2A_aspects_list':O2A_aspects_list,
        'O2A_opinions_list':O2A_opinions_list,
        'lossA':lossA,
        'lossO':lossO,
    }
    ##If in training process we should add ground truth sentiment labels to result for calculate sentiment loss next step
    if model_mode=='train':
      result['sentiment_labels_list']=batch_dict['sentiments']

    return result

  def processOutput(self,aspect_hidden_states,opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict=None,model_mode='train'):
    '''
      Hàm hỗ trợ Module xử lý dữ liệu đầu ra từ logits thành list các aspect và opinion dự đoán được qua
       input.
      Hàm còn hỗ trợ cắt lấy các hidden states chính xác của câu input (như đã biết câu query được padding nên
      hàm này sẽ chỉ lấy đúng hidden states theo vị trí đúng của các token trong câu input trong context của
      thành phần question) 
    '''
    ##Define list to save data output
    aspect_list=[]
    opinion_list=[]
    aspect_hidden_states_list=[]
    opinion_hidden_states_list=[]
    for i in range(len(aspect_logits)):
      ##Aspect
      passenge_index = (queries_for_aspect[i]==102).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long).unsqueeze(1)
      aspects=[]
      logits=aspect_logits[i]
      aspect_prob=F.softmax(logits,dim=-1)
      prob_val,prob_label=torch.max(aspect_prob,dim=-1)
      passenge_labels=prob_label[passenge_index].squeeze(1)
      passenge_prob_vals=prob_val[passenge_index].squeeze(1)
      _,index_list=torch.sort(passenge_prob_vals,descending=True)
      ##Tương tự xử lý khi không xuất hiện nhãn 1 trong dự đoán
      if 1 not in passenge_labels:
        if model_mode=='train':
          ##In training process if doesn't find any B label we use grouth truth to learning
          ##Teacher forcing
          passenge_labels=torch.tensor(batch_dict['aspect_answers'][i])
          one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]
        else:
          _aspect_prob=aspect_prob.transpose(0,1)[1]
          passenge_aspect_prob=_aspect_prob[passenge_index].squeeze(1)
          _,one_index=torch.sort(passenge_aspect_prob,descending=True)
          index_list=one_index
        if 0 not in passenge_labels:
          two_index=torch.tensor([])
        else:
          two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]
      else:
        one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]
        two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]
      count=0
      for j in range(len(index_list)):
        idx=index_list[j].item()
        if idx in two_index:
          continue
        if idx in one_index:
          aspect=[idx]
          count+=1
          idx+=1
          while idx<len(passenge_index) and idx in two_index:
            aspect.append(idx)
            idx+=1
          aspects.append(aspect)
          if count>=self.args.p:
            break
        else:
          continue
      aspect_list.append(aspects)
      ##Opinion
      opinions=[]
      passenge_index = (queries_for_opinion[i]==102).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long).unsqueeze(1)
      logits=opinion_logits[i]
      opinion_prob=F.softmax(logits,dim=-1)
      prob_val,prob_label=torch.max(opinion_prob,dim=-1)
      passenge_labels=prob_label[passenge_index].squeeze(1)
      passenge_prob_vals=prob_val[passenge_index].squeeze(1)
      _,index_list=torch.sort(passenge_prob_vals,descending=True)
      ##Xử lý khi nhãn 1 không có trong dự đoán, tiến hành như trên kia
      if 1 not in passenge_labels:
        if model_mode=='train':
          ##In training process if doesn't find any B label we use grouth truth to learning
          ##Teacher forcing
          passenge_labels=torch.tensor(batch_dict['opinion_answers'][i])
          one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]
        else:
          _opinion_prob=opinion_prob.transpose(0,1)[1]
          passenge_opinion_prob=_opinion_prob[passenge_index].squeeze(1)
          _,one_index=torch.sort(passenge_opinion_prob,descending=True)
          index_list=one_index
        if 0 not in passenge_labels:
          two_index=torch.tensor([])
        else:
          two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]
      else:
        one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]
        two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]
      count=0
      for j in range(len(index_list)):
        idx=index_list[j].item()
        if idx in two_index:
          continue
        if idx in one_index:
          opinion=[idx]
          count+=1
          idx+=1
          while idx<len(passenge_index) and idx in two_index:
            opinion.append(idx)
            idx+=1
          opinions.append(opinion)
          if count>self.args.q:
            break
        else:
          continue
      opinion_list.append(opinions)
      ##Aspect Hidden states
      passenge_index = (queries_for_aspect[i]==102).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long)
      aspect_hidden_states_list.append(aspect_hidden_states[i,passenge_index,:])
      ##Opinion Hidden states
      passenge_index = (queries_for_opinion[i]==102).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long)
      opinion_hidden_states_list.append(opinion_hidden_states[i,passenge_index,:])
    return aspect_hidden_states_list,aspect_list,opinion_hidden_states_list,opinion_list

##This cell contain function for Matching Module
class MatchingModule(nn.Module):
  '''
    Module áp dụng attention mechanism để tìm ra những opinion terms cao điểm nhất với mỗi aspect term dự
     đoán được sau RoleFlipped.
  '''
  def __init__(self,args):
    super(MatchingModule,self).__init__()
    self.sent_ffnn_A2O=nn.Linear(args.hidden_size*2,3) ##Thành phần dán nhãn sentiment cho chiều A2O
    self.sent_ffnn_O2A=nn.Linear(args.hidden_size*2,3) ##Thành phần dán nhãn sentiment cho chiều O2A

    if args.ifgpu==True:
      self.sent_ffnn_A2O.cuda()
      self.sent_ffnn_A2O.cuda()
      
    self.args=args

  def forward(self,result_dict,batch_dict,model_mode='train'):
    predicts_list=[]
    _aspects_list=[]
    _opinions_list=[]
    lossS=0
    for i in range(len(result_dict['A2O_aspects_list'])):
      aspects_list=[]
      opinions_list=[]
      ##A2O
      A2O_aspect_terms=result_dict['A2O_aspects_list'][i]
      A2O_opinion_terms=result_dict['A2O_opinions_list'][i]
      A2O_aspect_hidden_states=result_dict['A2O_aspect_hidden_states'][i]
      A2O_opinion_hidden_states=result_dict['A2O_opinion_hidden_states'][i]
      final_hidden_states=self.matching(A2O_aspect_hidden_states,A2O_opinion_hidden_states,A2O_aspect_terms,A2O_opinion_terms,calc_type='all',direct='A2O')
      A2O_logits=[]
      for row in final_hidden_states:
        if torch.sum(row).item()==0:
          A2O_logits.append([0]*3)
        else:
          A2O_logits.append(self.sent_ffnn_A2O(row).tolist())
      
      ##O2A
      O2A_aspect_terms=result_dict['O2A_aspects_list'][i]
      O2A_opinion_terms=result_dict['O2A_opinions_list'][i]
      O2A_aspect_hidden_states=result_dict['O2A_aspect_hidden_states'][i]
      O2A_opinion_hidden_states=result_dict['O2A_opinion_hidden_states'][i]
      final_hidden_states=self.matching(O2A_aspect_hidden_states,O2A_opinion_hidden_states,O2A_aspect_terms,O2A_opinion_terms,calc_type='all',direct='O2A')
      O2A_logits=[]
      for row in final_hidden_states:
        if torch.sum(row).item()==0:
          O2A_logits.append([0]*3)
        else:
          O2A_logits.append(self.sent_ffnn_O2A(row).tolist())
      
      ##Final Decision (nhãn sentiment của một token sẽ là trung bình cộng của hai chiều)
      A2O_logits=torch.tensor(A2O_logits)
      O2A_logits=torch.tensor(O2A_logits)
      final_logits=0.5*(A2O_logits+O2A_logits)
      
      ##Nếu ở chế độ train tính loss cho việc dán nhãn phân loại sentiment
      ##Trích xuất ra nhãn và logits tương ứng với những dòng logits toàn 0 để tránh việc tính toán loss bị sai
      if model_mode=='train':
        temp_final_logits=[]
        temp_sentiments=[]
        ##Trích xuất ra đúng những nhãn và logits của những tokens có logits khác không
        for inde in range(len(final_logits)):
          token_logits=final_logits[inde]
          if torch.sum(token_logits).item()==0:
            continue
          temp_final_logits.append(token_logits.tolist())
          temp_sentiments.append(result_dict['sentiment_labels_list'][i][inde])

        if self.args==True:
          temp_final_logits=torch.tensor(temp_final_logits).cuda()
          temp_sentiments=torch.tensor(temp_sentiments).cuda()
          weight = torch.tensor([1,2,4]).float().cuda()
        else:
          temp_final_logits=torch.tensor(temp_final_logits)
          temp_sentiments=torch.tensor(temp_sentiments)
          weight = torch.tensor([1,2,4]).float()

        '''if self.args.ifgpu==True:
          weight = torch.tensor([4,1,2]).float().cuda()
          #sentiment_targets=torch.tensor(result_dict['sentiment_labels_list'][i]).cuda()
        else:
          weight = torch.tensor([4,1,2]).float()
          #sentiment_targets=torch.tensor(result_dict['sentiment_labels_list'][i])'''
        lossS+=F.cross_entropy(temp_final_logits,temp_sentiments,weight=weight,ignore_index=-1)
      
      ##Getting label of aspect tokens:
      labels=[-1]*len(final_logits)
      for inde in range(len(final_logits)):
        token=final_logits[inde]
        if torch.sum(token).item()==0:
          continue
        max_index=torch.argmax(token).item()
        labels[inde]=max_index

      predicts_list.append(labels)
      aspects_list=self.filterOutput(result_dict['A2O_aspects_list'][i],result_dict['O2A_aspects_list'][i])
      opinions_list=self.filterOutput(result_dict['A2O_opinions_list'][i],result_dict['O2A_opinions_list'][i])
      _aspects_list.append(aspects_list)
      _opinions_list.append(opinions_list)
    return _aspects_list,_opinions_list,predicts_list,result_dict['lossA'],result_dict['lossO'],lossS
  
  def matching(self,aspect_hidden_states,opinion_hidden_states,aspect_terms,opinion_terms,calc_type='all',direct=None):
    '''
      Hàm hỗ trợ Module kết nối hidden_state của aspect và opinion term tương ứng dựa trên điểm attention
      Mỗi apect term sẽ được kết nối với opinion terms có điểm attention cao nhất
    '''
    if direct=='A2O':
      if self.args.T%2!=0:
        hidden_states=opinion_hidden_states
      else:
        hidden_states=aspect_hidden_states
    elif direct=='O2A':
      if self.args.T%2!=0:
        hidden_states=aspect_hidden_states
      else:
        hidden_states=opinion_hidden_states
    attention_matrix,asp_index=self.calculate_attention(hidden_states,aspect_terms,opinion_terms,calc_type='all')
    max_vals,max_inds=torch.max(attention_matrix,dim=-1)
    final_hidden_states=torch.zeros(hidden_states.size(0),self.args.hidden_size*2)
    for idx in asp_index:
      final_hidden_states[idx]=torch.cat((aspect_hidden_states[idx],opinion_hidden_states[max_inds[idx].item()]),dim=-1)

    if self.args.ifgpu==True:
      return final_hidden_states.cuda()
    else:
      return final_hidden_states

  def calculate_attention(self,hidden_states,aspect_terms,opinion_terms,calc_type='all'):
    '''
      Hàm hỗ trợ module tính điểm attention the công thức do bài báo cung cấp.
      calc_type là phương pháp sẽ tính:
        + Với 'all' tất cả các aspect token dù là nhãn 1 hay 2 đều được xem là một aspect, opinion riêng biệt để tìm
        tương đồng với nhau.
        + Todo: sum hoặc averaged: sẽ tính tổng hoặc trung bình cộng hidden_states của các token nằm trong
        cùng một nhãn aspect hay opinion.
    '''
    n=hidden_states.size(1)
    score=torch.zeros(n,n)
    A=torch.zeros(n,n)
    ##Check if calc_type is all,we treat label 1 or 2 as the same
    if calc_type=='all':
      asp_index=[idx for aspects in aspect_terms for idx in aspects]
      opi_index=[idx for opinions in opinion_terms for idx in opinions]
      ##Calculate score first
      for idx in asp_index:
        for idy in opi_index:
          if idx!=idy:
            score[idx,idy]=torch.matmul(hidden_states[idx].t(),hidden_states[idy])/100
      ##Calculate attention score
      for idx in asp_index:
        for idy in opi_index:
          if idx!=idy:
            A[idx,idy]=torch.exp(score[idx,idy])/torch.exp(torch.sum(score[idx]))
    return A,asp_index

  def filterOutput(self,first_output,second_output):
    '''
    Hàm để xử lý dữ liệu đầu ra cuối cùng:
      + Sau RoeFlipped Module ta hiện đang có hai danh sách aspect và hai danh sách opinion.
      + Hàm này hỗ trợ xóa bớt trùng ở cả hai hoặc ghép các aspect và opinion có sự trùng lắp về vị trí
       lại với nhau. Ví dụ: [3->5] với [4->7] sẽ được ghép thành [3->7]
    '''
    filtered_output=set()
    ##Adding first output to the filters
    for output in first_output:
      filtered_output.add((output[0],output[-1]))
    
    ##Adding second output to the filters
    for outputs in second_output:
      filtered_output.add((output[0],output[-1]))
    
    filtered_output=sorted(list(filtered_output),key=lambda x:(x[0],x[1]))
    for i in range(len(filtered_output)):
      filtered_output[i]=list(filtered_output[i])
    ##Filtered
    remove_ind=[]
    fixing_ind=[]
    idx=0
    idy=1
    while idx<len(filtered_output)-1 and idy<len(filtered_output):
      if filtered_output[idx][0]==filtered_output[idy][0]:
        remove_ind.append(idx)
        idx+=1
        idy+=1
        continue
      elif filtered_output[idy][0]>filtered_output[idx][0]:
        if filtered_output[idy][0]<filtered_output[idx][1]:
          if filtered_output[idy][1]==filtered_output[idx][1]:
            remove_ind.append(idy)
            idy+=1
            continue
          else:
            filtered_output[idx][1]=filtered_output[idx][1]
            remove_ind.append(idy)
            idy+=1
            continue
        elif filtered_output[idy][0]==filtered_output[idx][1]:
          filtered_output[idx][1]=filtered_output[idx][1]
          remove_ind.append(idy)
          idy+=1
          continue
        else:
          idx+=1
          idy+=1
          continue

    ##Remove index:
    remove_ind=sorted(remove_ind)
    for ind in remove_ind[::-1]:
      filtered_output.pop(ind)
    return filtered_output

#Grab everything in to one model:
class RFMRC(nn.Module):
  ##Gom mọi thứ vào một model duy nhất
  def __init__(self,args):
    super(RFMRC,self).__init__()
    self.args=args

    ##RoleFlipped Module
    self._RF_Module=RoleFlippedModule(args)

    ##Matching Module
    self._Matching_Module=MatchingModule(args)

  def forward(self,batch_dict,model_mode='train'):
    result_dict=self._RF_Module(batch_dict,model_mode)
    return self._Matching_Module(result_dict,batch_dict,model_mode)