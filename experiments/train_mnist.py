import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from sim.layers import AnalogLinear
from sim.physics import PHYSICS_CONFIG
import pandas as pd
import time
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, physics_type, seed=123):
        super().__init__()
        torch.manual_seed(seed)
        self.flat = nn.Flatten()
        self.l1 = AnalogLinear(28*28, 128, physics_type, seed=seed)
        self.act = nn.ReLU()
        self.l2 = AnalogLinear(128, 10, physics_type, seed=seed+1)
    def forward(self, x):
        x = self.flat(x)
        x = self.act(self.l1(x))
        x = self.l2(x)
        return x
    def energy_mJ(self):
        return (self.l1.energy_fJ + self.l2.energy_fJ) * 1e-12  # 1 fJ = 1e-12 mJ

def eval_acc(model, loader, device):
    model.eval()
    for l in [model.l1, model.l2]:
        l.accumulate_energy = False
    corr = 0
    with torch.no_grad():
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            p = model(d).argmax(1)
            corr += (p==t).sum().item()
    for l in [model.l1, model.l2]:
        l.accumulate_energy = True
    return 100.0 * corr / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physics', choices=list(PHYSICS_CONFIG.keys()), default='RRAM')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default='runs')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    tr = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True)
    te = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False)

    model = Net(args.physics, seed=args.seed).to(dev)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    cel = nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    step = 0
    print(f'=== {PHYSICS_CONFIG[args.physics]["name"]} ===')
    for ep in range(args.epochs):
        model.train()
        for d, t in tr:
            d, t = d.to(dev), t.to(dev)
            opt.zero_grad()
            y = model(d)
            loss = cel(y, t)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                acc = eval_acc(model, te, dev)
                energy = model.energy_mJ()
                rows.append(dict(step=step, epoch=ep, acc=acc, energy_mJ=energy, physics=args.physics))
                print(f'step {step:5d} | acc {acc:5.1f}% | energy {energy:8.2f} mJ')

    import pandas as pd, matplotlib.pyplot as plt, time
    df = pd.DataFrame(rows)
    ts = time.strftime('%Y%m%d-%H%M%S')
    os.makedirs('figures', exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(args.out, f'mnist_{args.physics}_{ts}.csv')
    df.to_csv(csv, index=False)
    fig = os.path.join('figures', f'energy_vs_acc_{args.physics}_{ts}.png')
    plt.figure(figsize=(5,4)); plt.plot(df.energy_mJ, df.acc, marker='o'); plt.xlabel('Energy (mJ)'); plt.ylabel('Accuracy (%)'); plt.grid(True, ls='--', alpha=0.6); plt.tight_layout(); plt.savefig(fig, dpi=180)
    print('Saved:', csv, fig)

if __name__ == '__main__':
    main()