from model import FModel, GModel, SimulatorDiscriminator, DecoderDiscriminator, Decoder
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain

def do_flip(labels, flip_rate):
    if flip_rate > 0:
        flip = torch.rand(labels.size(0)) < flip_rate
        labels[flip] = 1 - labels[flip]
    return labels

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
        self.fg_optimizer = torch.optim.Adam( chain(self.f.parameters(), self.g.parameters()), lr=p_config["fg_lr"])

        self.e = FModel(self.level, self.input_shape, width=p_config['width']).to(self.device)
        self.e_optimizer = torch.optim.Adam(self.e.parameters() , lr=p_config["e_lr"])

        self.d = Decoder(self.level, self.intermidiate_shape, conditional=conditional, num_classes=self.num_classes, width=p_config['width']).to(self.device)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=p_config["d_lr"])

        if use_e_dis:
            self.e_dis = SimulatorDiscriminator(self.level, self.intermidiate_shape, self.conditional, self.num_classes, width=p_config['width']).to(self.device)
            self.e_dis_optimizer = torch.optim.Adam(self.e_dis.parameters(), lr=p_config["e_dis_lr"])
        else:
            self.e_dis = None
        if use_d_dis:
            self.d_dis = DecoderDiscriminator(self.intermidiate_shape, self.conditional, self.num_classes).to(self.device)
            self.d_dis_optimizer = torch.optim.Adam(self.d_dis.parameters(), lr=p_config["d_dis_lr"])
        else:
            self.d_dis = None

        # dataset preparation
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
        z = self.f(x)
        y_pred = self.g(z)
        
        if self.u_shape:
            y_pred = self.h(y_pred)
        if self.alpha > 0.0:
            ce_loss = F.cross_entropy(y_pred, y)
            dist_corr_loss = dist_corr(x, z)
            fg_loss = ce_loss * (1 - self.alpha) + self.alpha * dist_corr_loss
        else:
            fg_loss = F.cross_entropy(y_pred, y)
            
        # self.fg_acc.update(torch.mean((y_pred.argmax(1) == y).float()))
        
        fg_loss.backward()
        self.fg_optimizer.step()
        return fg_loss.item(), z
    
    def e_train_step(self, xs, ys, y, z):
        self.e_optimizer.zero_grad()
        if self.e_dis is not None:
            self.e_dis_optimizer.zero_grad()

        zs = self.e(xs)
        y_pred_simulator = self.g(zs)
        
        eg_loss = F.cross_entropy(y_pred_simulator, ys)

        e_dis_real_loss = 0.0
        e_dis_fake_loss = 0.0
        e_gen_loss = 0.0
        e_dis_loss = 0.0
        
        if self.e_dis is not None:
            if self.conditional:
                e_dis_fake_output = self.e_dis(torch.cat([zs, ys.unsqueeze(1)], dim=1))
                e_dis_real_output = self.e_dis(torch.cat([z, y.unsqueeze(1)], dim=1))
            else:
                e_dis_fake_output = self.e_dis(zs)
                e_dis_real_output = self.e_dis(z)

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

        e_loss.backward()

        self.e_optimizer.step()
        if self.e_dis is not None:
            self.e_dis_optimizer.step()
        
        return eg_loss.item(), e_gen_loss.item(), e_loss.item(), float(e_dis_real_loss), float(e_dis_fake_loss), float(e_dis_loss)

    def d_train_step(self, xs, ys, y, z):
        self.d_optimizer.zero_grad()
        if self.d_dis is not None:
            self.d_dis_optimizer.zero_grad()
        # Generate reconstructed samples
        zs = self.e(xs)
        decoded_xs = self.d(torch.cat([zs, ys.unsqueeze(1)], dim=1) if self.conditional else zs)
        
        # MSE loss
        d_mse_loss = torch.mean(torch.square(xs - decoded_xs))
        
        # Discriminator loss
        d_dis_loss = 0.0
        d_gen_loss = 0.0
        
        if self.d_dis is not None:
            # Generate reference decoded sample
            decoded_x = self.d(torch.cat([z, y.unsqueeze(1)], dim=1) if self.conditional else z)
            
            # Conditional or unconditional discriminator
            if self.conditional:
                d_dis_fake_output = self.d_dis(torch.cat([decoded_x, y.unsqueeze(1)], dim=1))
                d_dis_real_output = self.d_dis(torch.cat([xs, ys.unsqueeze(1)], dim=1))
            else:
                d_dis_fake_output = self.d_dis(decoded_x)
                d_dis_real_output = self.d_dis(xs)
            
            # Real/fake discriminator losses
            d_dis_real_loss = F.binary_cross_entropy_with_logits(d_dis_real_output, torch.ones_like(d_dis_real_output))
            d_dis_fake_loss = F.binary_cross_entropy_with_logits(d_dis_fake_output, torch.zeros_like(d_dis_fake_output))
            
            self.d_dis_real_acc.update(torch.mean((d_dis_real_output.sigmoid() > 0.5).float()))
            self.d_dis_fake_acc.update(torch.mean((d_dis_fake_output.sigmoid() < 0.5).float()))
            
            d_dis_loss = d_dis_real_loss + d_dis_fake_loss
            d_gen_loss = F.binary_cross_entropy_with_logits(d_dis_fake_output, torch.ones_like(d_dis_fake_output))
            
            d_loss = d_mse_loss + d_gen_loss * self.config.get("lambda2", 1.0)
        else:
            d_loss = d_mse_loss
        
        # Backward pass
        d_loss.backward()
        
        # Update parameters
        self.d_optimizer.step()
        if self.d_dis is not None:
            self.d_dis_optimizer.step()
        
        return d_mse_loss.item(), d_gen_loss.item(), d_loss.item(), float(d_dis_real_loss), float(d_dis_fake_loss), float(d_dis_loss)

    def train_pipeline(self):
        for i, ((x, y), (xs, ys)) in enumerate(self.data_iterator):
            x, y = x.to(self.device), y.to(self.device)
            xs, ys = xs.to(self.device), ys.to(self.device)
            ys = do_flip(ys, self.flip_rate)

            fg_loss, z = self.fg_train_step(x, y)
            eg_loss, e_gen_loss, e_loss, e_dis_real_loss, e_dis_fake_loss, e_dis_loss = self.e_train_step(xs, ys, y, z)
            d_mse_loss, d_gen_loss, d_loss, d_dis_real_loss, d_dis_fake_loss, d_dis_loss = self.d_train_step(xs, ys, y, z)
            with torch.no_grad():
                x_reconstructed = (self.d(torch.cat([z, y.unsqueeze(1)], dim=1) if self.conditional else z))
                attack_loss = torch.mean(torch.square(x - x_reconstructed), dim=[1,2,3])

            self.log["fg_loss"][i] = np.mean(fg_loss)
            self.log["fg_acc"][i] = self.fg_acc.result().numpy()
            self.log["eg_loss"][i] = np.mean(eg_loss)
            self.log["eg_acc"][i] = self.eg_acc.result().numpy()
            self.log["e_gen_loss"][i] = np.mean(e_gen_loss)
            self.log["e_loss"][i] = np.mean(e_loss)
            self.log["e_dis_real_loss"][i] = np.mean(e_dis_real_loss)
            self.log["e_dis_real_acc"][i] = self.e_dis_real_acc.result().numpy()
            self.log["e_dis_fake_loss"][i] = np.mean(e_dis_fake_loss)
            self.log["e_dis_fake_acc"][i] = self.e_dis_fake_acc.result().numpy()
            self.log["e_dis_loss"][i] = np.mean(e_dis_loss)
            self.log["d_mse_loss"][i] = np.mean(d_mse_loss)
            self.log["d_gen_loss"][i] = np.mean(d_gen_loss)
            self.log["d_loss"][i] = np.mean(d_loss)
            self.log["d_dis_real_loss"][i] = np.mean(d_dis_real_loss)
            self.log["d_dis_real_acc"][i] = self.d_dis_real_acc.result().numpy()
            self.log["d_dis_fake_loss"][i] = np.mean(d_dis_fake_loss)
            self.log["d_dis_fake_acc"][i] = self.d_dis_fake_acc.result().numpy()
            self.log["d_dis_loss"][i] = np.mean(d_dis_loss)
            self.log["attack_loss"][i] = np.mean(attack_loss)

            if self.verbose_freq is not None and (i+1) % self.verbose_freq == 0:
                print(f"[{i}]: fg_loss: {np.mean(self.log['fg_loss'][i+1-self.verbose_freq:i+1]):.4f}, e_total_loss: {np.mean(self.log['e_loss'][i+1-self.verbose_freq:i+1]):.4f}, e_dis_loss: {np.mean(self.log['e_dis_loss'][i+1-self.verbose_freq:i+1]):.4f}, d_total_loss: {np.mean(self.log['d_loss'][i+1-self.verbose_freq:i+1]):.4f}, d_dis_loss: {np.mean(self.log['d_dis_loss'][i+1-self.verbose_freq:i+1]):.4f}, attack_loss: {np.mean(self.log['attack_loss'][i+1-self.verbose_freq:i+1]):.4f}")
            return self.log

    def attack(self, x, y):
        x_recon = self.d([self.f(x, training=True), y], training=False) if self.conditional else self.d(self.f(x, training=True), training=False)
        return x_recon,  torch.mean(torch.square(x - x_recon), dim=[1,2,3])
