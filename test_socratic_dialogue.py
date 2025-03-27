from socratic_dialogue import SocraticDialogueEnv
from Algorithms import GPT2
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

policy = GPT2(get_device = 0, from_checkpoint='/home/vasters/titan-rl/OfflineArcher/lightning_logs/lightning_logs/version_8/checkpoints/epoch=12-step=4010.ckpt')

env = SocraticDialogueEnv()

auto = True

articles = {}
df = pd.read_csv('SocraticDialogues_final_rewards_v3_shorts.csv')
for article in df['article'].unique():
    if article == 'none': continue
    perarticle = df[df['article'] == article]
    excerpt = perarticle[perarticle['id'] == perarticle['id'].unique()[0]].reset_index().loc[:2, 'line'].tolist()
    excerpt[0] += '\n'
    excerpt[2] += '\n'
    interaction = excerpt[0]+' '.join([excerpt[1], excerpt[2]])
    articles[article] = {'input': interaction, 'dialogue': perarticle[perarticle['id'] == perarticle['id'].unique()[0]].reset_index().loc[3:, 'line']}

print("*** In batch_simulate_on_environment ***")

article_no = input('Select an article from the list:\n\n'+"\n".join([f'{i}. '+article for i, article in enumerate(list(articles.keys()))])+'\n\nArticle No. (default: 0): ')
if article_no == '': article_no = '0'
while article_no.isalpha() or (int(article_no) < 0 or int(article_no) >= len(list(articles.keys()))):
    article_no = input('Select a number between 0 and '+str(len(list(articles.keys()))-1)+': ')
    if article_no == '': article_no = '0'

article_prompt = articles[list(articles.keys())[int(article_no)]]
obs = env.reset(article_prompt=article_prompt['input'])
current_idx = 1
checkdict = {}
done = False

while not done:
    with torch.no_grad():
        action = policy.forward(obs)

    if action[0] not in checkdict.keys():
        if action[0] in article_prompt['dialogue']: checkdict[action[0]] = 'OK'
        else:
            if current_idx >= len(article_prompt['dialogue']):
                checkdict[action[0]] = article_prompt['dialogue'].iloc[-1]
            else:
                checkdict[action[0]] = article_prompt['dialogue'].iloc[current_idx-1]
    else:
        print(f'\n*** Question "{action[0]}" has already been asked! Terminating...')
        break

    if auto:
        feedback = env.step(action, article_prompt['dialogue'], current_idx)
    else:
        feedback = env.step(action)
    current_idx += 2

    next_obs, r, done = feedback
    # print({"observation": obs,
    #         "action": action,
    #         "reward": r,
    #         "next_observation": next_obs,
    #         "done": done,
    #         })
    obs = next_obs
    done = done

del policy

model = AutoModel.from_pretrained('meta-llama/Llama-3.1-8B')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

print('\n*** RESULTS ***')
avg_score = 0.0
for question, status in checkdict.items():
    print(f'\nChatbot question: {question}')
    print(f'Status: {status}')

    if status != 'OK':
        tq = tokenizer(question, return_tensors='pt')
        tstatus = tokenizer(status, return_tensors='pt')
        with torch.no_grad():
            qembed = model(**tq).last_hidden_state.mean(dim=1)
            statusembed = model(**tstatus).last_hidden_state.mean(dim=1)
        normalized_qembed = normalize(qembed.numpy(), norm='l2')
        normalized_statusembed = normalize(statusembed.numpy(), norm='l2')
        similarity = cosine_similarity(normalized_qembed, normalized_statusembed)
        avg_score += similarity.squeeze()/float(len(checkdict))
        print('Similarity: {:.2f}'.format(similarity.squeeze()))
    else:
        avg_score += 1.0/float(len(checkdict))
    
    print('='*50)

print('\n*** Average score: {:.2f} ***'.format(avg_score))