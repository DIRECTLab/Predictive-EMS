import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()

        # self.fc1 = nn.Linear(state_space, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 1024)
        # self.fc4 = nn.Linear(1024, 512)
        # self.fc5 = nn.Linear(512, 256)
        # self.fc6 = nn.Linear(256, action_space)
        
        # self.fc1 = nn.Linear(state_space, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, action_space
        #  self.fc1 = nn.Linear(state_space, 256)
        # self.fc1 = nn.Linear(state_space, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 256)
        # # self.fc4 = nn.Linear(1024, 512)
        # self.fc5 = nn.Linear(256, 128)
        # self.fc6 = nn.Linear(128, action_space)
    

        # self.fc1 = nn.Linear(state_space, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 1024)
        # self.fc4 = nn.Linear(1024, 2048)
        # self.fc5 = nn.Linear(2048, 1024)
        # self.fc6 = nn.Linear(1024, 512)
        # self.fc7 = nn.Linear(512, 256)
        # self.fc8 = nn.Linear(256, 128)
        # self.fc9 = nn.Linear(128, action_space)
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(1024, 2048)
        # self.fc5 = nn.Linear(2048, 1024)
        # self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, action_space)

        # Below is the good_with_car
        # self.fc1 = nn.Linear(state_space, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 1024)
        # self.fc4 = nn.Linear(1024, 512)
        # self.fc5 = nn.Linear(512, 256)
        # self.fc6 = nn.Linear(256, action_space)

    def forward(self, state):
        # out = F.relu(self.fc1(state))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))

        # out = self.fc4(out)
        # out = F.relu(self.fc1(state))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))

        # out = self.fc6(out)

        # out = F.relu(self.fc1(state))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))

        # out = self.fc6(out)

        # out = F.relu(self.fc1(state))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))
        # out = F.relu(self.fc6(out))
        # out = F.relu(self.fc7(out))
        # out = F.relu(self.fc8(out))
        # out = self.fc9(out)

        # return out

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))
        # out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        out = F.relu(self.fc8(out))

        out = self.fc9(out)

        return out

        # Below is the good_with_car

        # out = F.relu(self.fc1(state))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = F.relu(self.fc5(out))

        # out = self.fc6(out)

        # return out

class Agent:
    def __init__(self, model_path, device, state_space, action_space=195):
        self.device = device

        self.action_space = action_space # Give actions in 10s of kW. (So 1 is 10kw)
        self.state_space = state_space

        self.policy_net = DQN(self.state_space, self.action_space)
        self.load_model(model_path)


    def predict(self, state, device):
        state = torch.tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            # action = np.argmax(self.policy_net(state).detach().cpu())
            return self.policy_net(state).max(0)[1].view(1, 1).detach().cpu().numpy()


    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.to(self.device)