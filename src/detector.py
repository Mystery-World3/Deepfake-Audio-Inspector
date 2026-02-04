import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

class DeepfakeDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioModel().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[+] Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[-] Gagal load model: {e}")

    def predict(self, features):
        input_tensor = torch.tensor(features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()
            
        return prediction, confidence