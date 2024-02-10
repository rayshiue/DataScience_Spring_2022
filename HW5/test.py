import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('rayshiue/DS_HW5_t5small')
model = AutoModelForSeq2SeqLM.from_pretrained('rayshiue/DS_HW5_t5small').to(device)
raw_dataset = load_dataset("json", data_files="./hw5_dataset/test.json")

max_input_length = 1024
max_target_length = 60
num_beams = 5

class MyDataset(Dataset):
    def __init__(self, test_raw_dataset, tokenizer):
        self.input_ids = []
        self.attention_mask = []
        print('Loading Test Data ...')
        p = tqdm(total = len(test_raw_dataset))
        with torch.no_grad():
            for d in test_raw_dataset:
                p.update(1)
                input_text = d['body']
                if not input_text:
                    self.input_ids.append(torch.zeros(max_input_length).long())
                    self.attention_mask.append(torch.zeros(max_input_length).long())
                else:
                    input_text = "summarize: " + input_text
                    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_input_length, return_tensors="pt")
                    self.input_ids.append(inputs.input_ids[0])
                    self.attention_mask.append(inputs.attention_mask[0])
        p.close()
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
    
batch_size = 12
test_dataset = MyDataset(raw_dataset['train'], tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

intab = "\xe9\xf3\xe1\xfc\xe8\xe4\xc9\xe7\xf6\xea"
outtab = "eoaueaEcoe"
trantab = str.maketrans(intab, outtab)
print('Making Predictions ...')
with open("./0810892.json", "w") as outfile:
    with torch.no_grad():
        p = tqdm(total=len(test_dataloader))
        for input_ids, attention_masks in test_dataloader:
            p.update(1)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            beam_outputs = model.generate(
                input_ids = input_ids,
                attention_mask = attention_masks,
                max_length = max_target_length,
                num_beams = num_beams,
                early_stopping = False,
            )
            for output in beam_outputs:
                result = tokenizer.decode(output, skip_special_tokens=True).translate(trantab)
                outfile.write("{\"title\":\"" + result + "\"}\n")
        p.close()