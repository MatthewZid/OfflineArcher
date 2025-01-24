from socratic_dialogue import SocraticDialogueEnv
from Algorithms import GPT2
import torch

policy = GPT2(get_device = 0, from_checkpoint='/home/vasters/titan-rl/OfflineArcher/lightning_logs/lightning_logs/version_2/checkpoints/epoch=19-step=640.ckpt')

env = SocraticDialogueEnv()

print("*** In batch_simulate_on_environment ***")
articles = {
    'Transform Your Body with the Latest Breakthrough in Weight Loss – SkinnySecretsExposed.com': 'Hello, I found the article “Transform Your Body with the Latest Breakthrough in Weight Loss – SkinnySecretsExposed.com” and I would like to discuss it further. Of course. I am here to discuss anything you want. First of all, what made you choose to read the article in the first place? I wanted to get informed about new ways to lose weight.\n',
    'Bologna, 23-year-old girl in alcoholic coma. Pakistani immigrant rapes her in hospital': "Hello, I came across the article “Bologna, 23-year-old girl in alcoholic coma. Pakistani immigrant rapes her in hospital” and I would like to discuss it further. Of course. I am here to discuss anything you want. Could you please inform me about the main topic of the article? Yes. A 50-year-old pakistani immigrant tried to rape a 23-year-old Italian girl in alcoholic comma in the Sant'Orsola Hospital in Bologna.\n",
    'The Cholesterol Conspiracy: Are We Being Deceived by Big Pharma and the Food Industry?': 'I read the article The Cholesterol Conspiracy: Are We Being Deceived by Big Pharma and the Food Industry?"" and would like to discuss it. This is fine. What would you like to discuss about this article? I need to check the credibility of the article.\n',
    'Rapes, brawls, ""cultural"" crimes: the criminal immigration that scares Italy': 'I read the article "" Rapes, brawls, ""cultural"" crimes: the criminal immigration that scares Italy"" and would like to discuss it. Of course. What would you like to discuss about this article? I want to check the facts cited in the article about migrants and crime in Italy.\n',
    'Medvedev: Neo-Nazism must be totally destroyed!': "Hello! You've selected the article: “Medvedev: Neo-Nazism must be totally destroyed! Yes.\n",
    'They knew : why didn’t the unvaccinated do more to warn us': 'Hi there! Have you read the article “They knew : why didn’t the unvaccinated do more to warn us” Yes, I have.\n',
    'Politicians appearing in entertainment television are also campaigning': 'Hi there! Have you read the article “Politicians appearing in entertainment television are also campaigning” Yes, I have.\n',
    'The government is spying on you through LED-lighting!!': "Hello! You've selected the article: “The government is spying on you through LED-lighting!!” Yes.\n"
}

article_no = input('Select an article from the list:\n\n'+"\n".join([f'{i}. '+article for i, article in enumerate(list(articles.keys()))])+'\n\nArticle No. (default: 0): ')
if article_no == '': article_no = '0'
while article_no.isalpha() or (int(article_no) < 0 or int(article_no) >= len(list(articles.keys()))):
    article_no = input('Select a number between 0 and '+str(len(list(articles.keys()))-1)+': ')
    if article_no == '': article_no = '0'

article_prompt = articles[list(articles.keys())[int(article_no)]]
obs = env.reset(article_prompt=article_prompt)
done = False

while not done:
    with torch.no_grad():
        action = policy.forward(obs)
    feedback = env.step(action)

    next_obs, r, done = feedback
    print({"observation": obs,
            "action": action,
            "reward": r,
            "next_observation": next_obs,
            "done": done,
            })
    obs = next_obs
    done = done