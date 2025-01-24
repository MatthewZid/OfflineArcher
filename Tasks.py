from lightning import LightningDataModule
import torch.utils.data as data
from Dataset import TrajectoryDataset, EmptyDataset
from SimulateOnEnv import batch_simulate_on_environment
import numpy as np
from copy import deepcopy
import sys
import pandas as pd
from transformers import AutoTokenizer

class Task(LightningDataModule):
    def __init__(self, batch_size: int, n_traj_eval: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size                 = batch_size
        self.eval_batch_size            = self.batch_size
        self.n_traj_eval                = n_traj_eval

        # Set Defaults
        self.shuffle                    = True
        self.drop_last                  = True # skips last batch to make sure gradient accumulation works as intended

    def setup(self, stage: str):
        raise NotImplementedError
    
    def train_dataloader(self):
        return data.DataLoader(dataset = self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last = self.drop_last)

    def val_dataloader(self):
        return data.DataLoader(dataset = EmptyDataset(length = self.n_traj_eval), batch_size=self.eval_batch_size)

    def get_eval_log(self, **kwargs):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
    
class TwentyQuestions(Task):
    def __init__(self, word_list = None, **kwargs):
        super().__init__(**kwargs)

        self.word_list                  = word_list
        self.max_horizon                = 20
        
        from twenty_questions import BatchedTwentyQuestionsEnv
        self.env    = BatchedTwentyQuestionsEnv(max_conversation_length = self.max_horizon, bsize = self.eval_batch_size, word_list=self.word_list)

    def setup(self, stage: str):
        self.env.model.to(self.trainer.model.device) # Ensure environment is on GPU
        self.dataset = self.read_data()
        self.dataset.check_consistency()
        print("\n *** Dataset Trimming Now Disabled. Please Called the Subroutine for triming")

    def read_data(self):
        import json
        from Dataset import TrajectoryDataset

        # f = open('datasets/20q_train.json')
        f = open('datasets/twenty_questions.json')
        data    = json.load(f)
        # data[0] (game):
        # {'lines': ['Is the object alive? Yes.', 'Is the object a mammal? No.', 'Is the object a plant? Yes.', 'Is the object a tree? No.',
        # 'Is the object a flower? No.', 'Is the object a vegetable? Yes.', 'Is the object a root vegetable? No.', 'Is the object an herb? No.',
        # 'Is the object a fruit? Yes.', 'Is the object a citrus fruit? No.', 'Is the object a berry? No.', 'Is the object a stone fruit? No.',
        # 'Is the object a tropical fruit? No.', 'Is the object a melon? No.', 'Is the object a kiwi? No.', 'Is the object a banana? No.',
        # 'Is the object a melon? No.', 'Is the object a cherry? No.', 'Is the object a fig? No.', 'Is the object a pomegranate? No.'],
        # 'correct': False, 'word': ['Tomato']}
        dataset              = TrajectoryDataset()

        for game in data:
            assert(len(game['lines']) <= 20)
            history = "Questions:\n" # assertion is checked with history = ''
            for interaction in game['lines']:
                yesAnswer = interaction[-5:] == ' Yes.'
                noAnswer  = interaction[-4:] == ' No.' 
                assert(yesAnswer or noAnswer)
                observation  = history
                
                done = True if interaction == game['lines'][-1] else False # if the interaction is the last interaction we are done
                reward = 0 if done and game['correct'] else -1
                
                if yesAnswer:
                    action = interaction[:-5]
                if noAnswer:
                    action = interaction[:-4]

                history += interaction + '\n'
                dataset.append_observation_action_reward(observation, action, reward)
            dataset.append_terminal_observation(history, 
                                                trajectory_info = {"correct": game["correct"], 
                                                                "word": game["word"]})

        dataset.check_consistency()
        return dataset 

class Socratic(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.word_list                  = word_list
        # self.max_horizon                = 20
        
        # from socratic_dialogue import BatchedSocraticDialogueEnv
        # self.env    = BatchedSocraticDialogueEnv(max_conversation_length = self.max_horizon, bsize = self.eval_batch_size, word_list=self.word_list)
        # self.env    = BatchedSocraticDialogueEnv(bsize = self.eval_batch_size)

    def setup(self, stage: str):
        # self.env.model.to(self.trainer.model.device) # Ensure environment is on GPU
        self.dataset = self.read_data()
        self.dataset.check_consistency()
        print("\n *** Dataset Trimming Now Disabled. Please Called the Subroutine for triming")

    def read_data(self):
        import json
        from Dataset import TrajectoryDataset

        # f = open('datasets/20q_train.json')
        # f = open('datasets/twenty_questions.json')
        df = pd.read_csv('/home/vasters/titan-rl/SocraticDialogues_final_rewards_v3_shorts.csv', sep=',', skipinitialspace=True)
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'].fillna('Hello!')
        dataset              = TrajectoryDataset()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

        for i in df['id'].unique().tolist():
            dialogue = df[df['id'] == i].reset_index(drop=True)
            tokenized = tokenizer(dialogue['line'].to_list(), add_special_tokens=False)['input_ids']
            tokenized = sum(tokenized, [])
            if len(tokenized) > 1024: continue

            # history = [Interaction(chatbot='', user=dialogue.loc[0, 'line'])]
            history = str(dialogue.loc[0, 'line']) + ' '
            for j in range(1, len(dialogue), 2):
                observation  = history

                action = dialogue.loc[j, 'line']
                
                # done = True if interaction == game['lines'][-1] else False # if the interaction is the last interaction we are done
                done = True if j == (len(dialogue)-1) else False # if the interaction is the last interaction we are done
                # reward = 0 if done and game['correct'] else -1

                if done:
                    # interaction = Interaction(chatbot=dialogue.loc[j, 'line'], user='')
                    interaction = dialogue.loc[j, 'line']
                    reward = dialogue.loc[j-1, 'reward']    # temporary final reward: get the last user reward
                else:
                    # interaction = Interaction(chatbot=dialogue.loc[j, 'line'], user=dialogue.loc[j+1, 'line'])
                    interaction = dialogue.loc[j, 'line'] + ' ' + dialogue.loc[j+1, 'line']
                    reward = dialogue.loc[j+1, 'reward']
                
                
                history += interaction + '\n'
                # history.append(interaction)
                dataset.append_observation_action_reward(observation, action, reward)

            # dataset.append_terminal_observation(history, 
            #                                     trajectory_info = {"correct": game["correct"], 
            #                                                     "word": game["word"]})
            dataset.append_terminal_observation(history)

        dataset.check_consistency()
        return dataset