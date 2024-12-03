from model import FModel, GModel, SimulatorDiscriminator, DecoderDiscriminator, Decoder
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain


def label_flip(num_class, flip_rate):
    def wrapper(ys):
        if flip_rate > 0.0:
            ys_one_hot = F.one_hot(ys.view(-1), num_class).float()
            uniform_dist = torch.ones_like(ys_one_hot) / num_class
            noisy_dist = (1 - flip_rate) * ys_one_hot + flip_rate * uniform_dist
            ys_noisy = torch.multinomial(noisy_dist, 1).squeeze()
            return ys_noisy
        return ys
    return wrapper


def dist_corr(X, Y):
    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)
    def pairwise_dist(Z):
        Z_norm = (Z**2).sum(1).view(-1, 1)
        dist = Z_norm - 2 * torch.mm(Z, Z.t()) + Z_norm.t()
        return torch.sqrt(torch.clamp(dist, min=1e-10))
    n = float(X.size(0))
    a = pairwise_dist(X)
    b = pairwise_dist(Y)
    a_cent = a - a.mean(dim=1).view(-1, 1) - a.mean(dim=0).view(1, -1) + a.mean()
    b_cent = b - b.mean(dim=1).view(-1, 1) - b.mean(dim=0).view(1, -1) + b.mean()
    dCovXY = torch.sqrt(1e-10 + torch.sum(a_cent * b_cent) / (n ** 2))
    dVarXX = torch.sqrt(1e-10 + torch.sum(a_cent * a_cent) / (n ** 2))
    dVarYY = torch.sqrt(1e-10 + torch.sum(b_cent * b_cent) / (n ** 2))
    dCorXY = dCovXY / torch.sqrt(1e-10 + dVarXX * dVarYY)
    return dCorXY

INTERMIDIATE_SHAPE = lambda level: (16, 32, 32) if level == 3 else (32, 16, 16) if level < 7 else (64, 8, 8)
    
class SDARAttacker:
    def __init__(self, client_loader, server_loader, num_classes, device) -> None:
        self.client_loader = client_loader
        self.server_loader = server_loader
        self.num_classes = num_classes
        self.device = torch.device(device)

    def preprocess(self, level, num_iters, p_config, conditional=True, use_e_dis=True, use_d_dis=True, verbose_freq=100):
        # hyperparameters preparation
        self.level = level
        self.num_iters = num_iters
        self.conditional = conditional
        self.use_e_dis = use_e_dis
        self.use_d_dis = use_d_dis
        self.verbose_freq = verbose_freq
        self.input_shape = (3, 32, 32)
        self.p_config = p_config
        self.alpha = p_config["alpha"]
        self.flip_rate = p_config["flip_rate"]
        # model and optimizer preparation 
        self.intermidiate_shape = INTERMIDIATE_SHAPE(level)
        self.f = FModel(self.level, self.input_shape, width=p_config['width']).to(self.device)
        self.g = GModel(self.level, self.intermidiate_shape, self.num_classes, dropout=p_config["dropout"], width=p_config['width']).to(self.device)
        self.fg_optimizer = torch.optim.Adam( chain(self.f.parameters(), self.g.parameters()), lr=p_config["fg_lr"], eps=1e-07)

        self.e = FModel(self.level, self.input_shape, width=p_config['width']).to(self.device)
        self.e_optimizer = torch.optim.Adam(self.e.parameters() , lr=p_config["e_lr"], eps=1e-07)

        self.d = Decoder(self.level, self.intermidiate_shape, conditional=conditional, num_classes=self.num_classes, width=p_config['width']).to(self.device)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=p_config["d_lr"], eps=1e-07)

        if use_e_dis:
            self.e_dis = SimulatorDiscriminator(self.level, self.intermidiate_shape, self.conditional, self.num_classes, width=p_config['width']).to(self.device)
            self.e_dis_optimizer = torch.optim.Adam(self.e_dis.parameters(), lr=p_config["e_dis_lr"], eps=1e-07)
        else:
            self.e_dis = None
        if use_d_dis:
            self.d_dis = DecoderDiscriminator(self.input_shape, self.conditional, self.num_classes).to(self.device)
            self.d_dis_optimizer = torch.optim.Adam(self.d_dis.parameters(), lr=p_config["d_dis_lr"], eps=1e-07)
        else:
            self.d_dis = None

        # dataset preparation
        self.flip_label = label_flip(self.num_classes, self.flip_rate)
        self.data_iterator = zip(self.client_loader, self.server_loader)
        # prepare log
        self.log = {}
        self.log["fg_loss"] = np.zeros(self.num_iters)
        self.log["fg_acc"] = np.zeros(self.num_iters)
        self.log["eg_loss"] = np.zeros(self.num_iters)
        self.log["eg_acc"] = np.zeros(self.num_iters)
        self.log["e_gen_loss"] = np.zeros(self.num_iters)
        self.log["e_loss"] = np.zeros(self.num_iters)
        self.log["e_dis_real_loss"] = np.zeros(self.num_iters)
        self.log["e_dis_real_acc"] = np.zeros(self.num_iters)
        self.log["e_dis_fake_loss"] = np.zeros(self.num_iters)
        self.log["e_dis_fake_acc"] = np.zeros(self.num_iters)
        self.log["e_dis_loss"] = np.zeros(self.num_iters)
        self.log["d_mse_loss"] = np.zeros(self.num_iters)
        self.log["d_gen_loss"] = np.zeros(self.num_iters)
        self.log["d_loss"] = np.zeros(self.num_iters)
        self.log["d_dis_real_loss"] = np.zeros(self.num_iters)
        self.log["d_dis_real_acc"] = np.zeros(self.num_iters)
        self.log["d_dis_fake_loss"] = np.zeros(self.num_iters)
        self.log["d_dis_fake_acc"] = np.zeros(self.num_iters)
        self.log["d_dis_loss"] = np.zeros(self.num_iters)
        self.log["attack_loss"] = np.zeros(self.num_iters)

    def fg_train_step(self, x, y):
        self.fg_optimizer.zero_grad()
        self.f.train()
        self.g.train()
        z = self.f(x)
        y_pred = self.g(z)
        if self.alpha > 0.0:
            ce_loss = F.cross_entropy(y_pred, y.view(-1))
            dist_corr_loss = dist_corr(x, z)
            fg_loss = ce_loss * (1 - self.alpha) + self.alpha * dist_corr_loss
        else:
            fg_loss = F.cross_entropy(y_pred, y.view(-1))
        
        torch.autograd.backward(fg_loss, inputs=list(self.f.parameters())+list(self.g.parameters()))
        self.fg_optimizer.step()

        return fg_loss.item(), z
    
    def e_train_step(self, xs, ys, y, z):
        self.e_optimizer.zero_grad()
        if self.e_dis is not None:
            self.e_dis_optimizer.zero_grad()

        self.e.train()
        zs = self.e(xs)

        self.g.eval()
        y_pred_simulator = self.g(zs)

        eg_loss = F.cross_entropy(y_pred_simulator, ys.view(-1))

        e_dis_real_loss = 0.0
        e_dis_fake_loss = 0.0
        e_gen_loss = 0.0
        e_dis_loss = 0.0

        if self.e_dis is not None:
            self.e_dis.train()
            e_dis_fake_output = self.e_dis(zs, ys) if self.conditional else self.e_dis(zs)
            e_dis_real_output = self.e_dis(z, y) if self.conditional else self.e_dis(z)
            e_dis_real_loss = F.binary_cross_entropy_with_logits(e_dis_real_output, torch.ones_like(e_dis_real_output))
            e_dis_fake_loss = F.binary_cross_entropy_with_logits(e_dis_fake_output, torch.zeros_like(e_dis_fake_output))            
            e_dis_loss = e_dis_real_loss + e_dis_fake_loss
            e_gen_loss = F.binary_cross_entropy_with_logits(e_dis_fake_output, torch.ones_like(e_dis_fake_output))

            if self.alpha > 0.0:
                e_loss = eg_loss * (1 - self.alpha) + e_gen_loss * self.p_config["lambda1"] + self.alpha * self.dist_corr(xs, zs)
            else:
                e_loss = eg_loss + e_gen_loss * self.p_config["lambda1"]
        else:
            e_loss = eg_loss
        torch.autograd.backward(e_loss, inputs=list(self.e.parameters()), create_graph=True)
        self.e_optimizer.step()
        if self.e_dis is not None:
            torch.autograd.backward(e_dis_loss, inputs=list(self.e_dis.parameters()), create_graph=True)
            self.e_dis_optimizer.step()
        return eg_loss.item(), e_gen_loss.item(), e_loss.item(), float(e_dis_real_loss), float(e_dis_fake_loss), float(e_dis_loss)

    def d_train_step(self, xs, ys, y, z):
        with torch.no_grad():
            self.e.train()
            zs = self.e(xs)

        self.d.train()
        decoded_xs  = self.d(zs, ys.view(-1)) if self.conditional else self.d(zs)
        d_mse_loss = torch.mean(torch.square(xs - decoded_xs))
        d_dis_loss = 0.0
        d_gen_loss = 0.0
        if self.d_dis is not None:
            decoded_x = self.d(z, y.view(-1)) if self.conditional else self.d(z)
            self.d_dis.train()
            d_dis_fake_output = self.d_dis(decoded_x, y.view(-1)) if self.conditional else self.d_dis(decoded_x)
            d_dis_real_output = self.d_dis(xs, ys.view(-1)) if self.conditional else self.d_dis(xs)
            
            d_dis_fake_loss = F.binary_cross_entropy_with_logits(d_dis_fake_output, torch.zeros_like(d_dis_fake_output))
            d_dis_real_loss = F.binary_cross_entropy_with_logits(d_dis_real_output, torch.ones_like(d_dis_real_output))
                        
            # self.d_dis_real_acc.update(torch.mean((d_dis_real_output.sigmoid() > 0.5).float()))
            # self.d_dis_fake_acc.update(torch.mean((d_dis_fake_output.sigmoid() < 0.5).float()))
            d_dis_loss = d_dis_real_loss + d_dis_fake_loss
            d_gen_loss = F.binary_cross_entropy_with_logits(d_dis_fake_output, torch.ones_like(d_dis_fake_output))
            
            d_loss = d_mse_loss + d_gen_loss * self.p_config["lambda2"]
        else:
            d_loss = d_mse_loss
        self.d_optimizer.zero_grad()
        if self.d_dis is not None:
            self.d_dis_optimizer.zero_grad()

        torch.autograd.backward(d_loss, inputs=list(self.d.parameters()), create_graph=True)
        self.d_optimizer.step()

        if self.d_dis is not None:
            torch.autograd.backward(d_dis_loss, inputs=list(self.d_dis.parameters()), create_graph=True)
            # d_dis_loss.backward(retain_graph=True)
            self.d_dis_optimizer.step()
        return d_mse_loss.item(), d_gen_loss.item(), d_loss.item(), float(d_dis_real_loss), float(d_dis_fake_loss), float(d_dis_loss)

    def train_pipeline(self):
        for i, ((x, y), (xs, ys)) in enumerate(self.data_iterator):
            x, y = x.to(self.device), y.to(self.device)
            xs, ys = xs.to(self.device), ys.to(self.device)
            ys = self.flip_label(ys)
            fg_loss, z = self.fg_train_step(x, y)
            eg_loss, e_gen_loss, e_loss, e_dis_real_loss, e_dis_fake_loss, e_dis_loss = self.e_train_step(xs, ys, y, z)
            d_mse_loss, d_gen_loss, d_loss, d_dis_real_loss, d_dis_fake_loss, d_dis_loss = self.d_train_step(xs, ys, y, z)
            with torch.no_grad():
                self.d.eval()
                x_reconstructed = self.d(z, y.view(-1)) if self.conditional else self.d(z)
                attack_loss = torch.mean(torch.square(x - x_reconstructed)).detach().cpu().numpy()

            self.log["fg_loss"][i] = np.mean(fg_loss)
            # self.log["fg_acc"][i] = self.fg_acc.result().numpy()
            self.log["eg_loss"][i] = np.mean(eg_loss)
            # self.log["eg_acc"][i] = self.eg_acc.result().numpy()
            self.log["e_gen_loss"][i] = np.mean(e_gen_loss)
            self.log["e_loss"][i] = np.mean(e_loss)
            self.log["e_dis_real_loss"][i] = np.mean(e_dis_real_loss)
            # self.log["e_dis_real_acc"][i] = self.e_dis_real_acc.result().numpy()
            self.log["e_dis_fake_loss"][i] = np.mean(e_dis_fake_loss)
            # self.log["e_dis_fake_acc"][i] = self.e_dis_fake_acc.result().numpy()
            self.log["e_dis_loss"][i] = np.mean(e_dis_loss)
            self.log["d_mse_loss"][i] = np.mean(d_mse_loss)
            self.log["d_gen_loss"][i] = np.mean(d_gen_loss)
            self.log["d_loss"][i] = np.mean(d_loss)
            self.log["d_dis_real_loss"][i] = np.mean(d_dis_real_loss)
            # self.log["d_dis_real_acc"][i] = self.d_dis_real_acc.result().numpy()
            self.log["d_dis_fake_loss"][i] = np.mean(d_dis_fake_loss)
            # self.log["d_dis_fake_acc"][i] = self.d_dis_fake_acc.result().numpy()
            self.log["d_dis_loss"][i] = np.mean(d_dis_loss)
            self.log["attack_loss"][i] = np.mean(attack_loss)

            if self.verbose_freq is not None and (i+1) % self.verbose_freq == 0:
                print(f"[{i}]: fg_loss: {np.mean(self.log['fg_loss'][i+1-self.verbose_freq:i+1]):.4f}, e_total_loss: {np.mean(self.log['e_loss'][i+1-self.verbose_freq:i+1]):.4f}, e_dis_loss: {np.mean(self.log['e_dis_loss'][i+1-self.verbose_freq:i+1]):.4f}, d_total_loss: {np.mean(self.log['d_loss'][i+1-self.verbose_freq:i+1]):.4f}, d_dis_loss: {np.mean(self.log['d_dis_loss'][i+1-self.verbose_freq:i+1]):.4f}, attack_loss: {np.mean(self.log['attack_loss'][i+1-self.verbose_freq:i+1]):.4f}")
            if i == self.num_iters - 1:
                break
        return self.log

    def attack(self, x, y):
        self.d.eval()
        self.f.train()
        x, y = x.to(self.device), y.to(self.device)
        x_recon = self.d(self.f(x), y) if self.conditional else self.d(self.f(x))
        mse = torch.mean(torch.square(x - x_recon)).detach().cpu().numpy()
        x_recon = x_recon.detach().cpu()
        return x_recon,  mse
    
    def evaluate(self, client_ds):
        total_mse = 0.0
        count = 0
        for i, (x,y) in enumerate(client_ds):
            x, y = x.to(self.device), y.to(self.device)
            self.f.train()
            z = self.f(x)
            self.d.eval()
            x_recon = self.d(z, y) if self.conditional else self.d(z)
            total_mse += torch.mean(torch.square(x - x_recon)).detach().cpu().numpy()
            count += len(x)
        print(f"Average MSE over all client's images: {total_mse / count}.")
        return total_mse / count
    
